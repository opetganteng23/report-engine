"""Tahap 1: Data Acquisition.

Crawling post X per seed word dengan twscrape.
Putaran 1: seed word terpilih, tanpa rentang tanggal.
Putaran 2: semua seed word, per bulan kalender (since/until).
Hasil di-append ke data/raw/raw_posts.jsonl; progres per query dicatat
supaya crawling bisa dilanjutkan bila terputus.

Pemakaian: python src/01_acquisition.py --config config.yaml
"""

import argparse
import asyncio
import csv
import json
import logging
import random
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from twscrape import API


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_seed_words(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    words = [r["seed_word"].strip() for r in sorted(rows, key=lambda r: int(r["no"]))]
    if len(words) != 32 or len(set(words)) != 32:
        sys.exit(f"seed_words.csv harus berisi 32 seed word unik, ditemukan {len(words)}")
    return words


def month_range(start, end):
    """Daftar (label 'YYYY-MM', since 'YYYY-MM-01', until awal bulan berikutnya)."""
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    out = []
    while (y, m) <= (ey, em):
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        out.append((f"{y:04d}-{m:02d}", date(y, m, 1).isoformat(), date(ny, nm, 1).isoformat()))
        y, m = ny, nm
    return out


def build_queries(cfg, t1, seed_words):
    """Setiap item: key unik (= nilai kolom query), putaran, seed word, teks query ke X, limit."""
    suffix = t1["query_suffix"]
    p1, p2 = t1["putaran1"], t1["putaran2"]

    # Teks query putaran 1 boleh berupa alias seed word (lihat config putaran1.alias)
    alias = p1.get("alias") or {}
    seed_of = {s: alias.get(s, s) for s in p1["seed_words"]}
    missing = [s for s, sw in seed_of.items() if sw not in seed_words]
    if missing:
        sys.exit(f"Seed word putaran 1 tidak ada di seed_words.csv: {missing}")
    if len(set(seed_of.values())) != len(seed_of):
        sys.exit("Putaran 1 berisi seed word ganda (setelah alias)")
    if len(p1["seed_words"]) != 16:
        sys.exit(f"Putaran 1 harus 16 seed word, ditemukan {len(p1['seed_words'])}")
    for name, p in (("putaran1", p1), ("putaran2", p2)):
        if p["maks_post_per_query"] is None:
            sys.exit(f"tahap1_acquisition.{name}.maks_post_per_query belum diisi di config")
    if cfg.get("bulan_mulai") is None or cfg.get("bulan_akhir") is None:
        sys.exit("bulan_mulai / bulan_akhir belum diisi di config")

    queries = []
    for s in p1["seed_words"]:
        q = f"{s} {suffix}"
        queries.append({"key": q, "putaran": 1, "seed_word": seed_of[s], "q": q,
                        "limit": p1["maks_post_per_query"]})
    for label, since, until in month_range(cfg["bulan_mulai"], cfg["bulan_akhir"]):
        for s in seed_words:
            q = f"{s} {suffix} since:{since} until:{until}"
            queries.append({"key": f"{q} [{label}]", "putaran": 2, "seed_word": s, "q": q,
                            "limit": p2["maks_post_per_query"]})
    return queries


def load_env(path):
    """Baca KEY=VALUE dari berkas .env (baris kosong dan '#' diabaikan)."""
    if not path.exists():
        sys.exit(f"{path} tidak ada; kredensial akun X harus diisi di sana")
    env = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


async def setup_api(t1, engine_dir):
    env = load_env(engine_dir / t1["env_file"])
    if not env.get("TWITTER_USERNAME"):
        sys.exit("TWITTER_USERNAME belum diisi di .env")
    api = API(str(engine_dir / t1["twscrape_db"]))
    await api.pool.add_account(env["TWITTER_USERNAME"], env.get("TWITTER_PASSWORD", ""),
                               env.get("TWITTER_EMAIL", ""), env.get("TWITTER_EMAIL_PASSWORD", ""),
                               cookies=env.get("TWITTER_COOKIES") or None)
    await api.pool.login_all()
    return api


def tweet_row(tw, key):
    return {
        "id": tw.id_str,
        "date": tw.date.isoformat(),
        "user": tw.user.username,
        "text": tw.rawContent,
        "reply_to": tw.inReplyToTweetIdStr,
        "query": key,
    }


def load_progress(path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"selesai": {}}


def save_progress(path, progress):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


async def search_with_retry(api, q, jeda, log):
    """Jalankan satu query; bila error, ulangi dengan jeda. Setelah jeda habis, berhenti
    (bukan melewati query) agar bisa dilanjutkan dengan menjalankan ulang skrip."""
    for percobaan in range(len(jeda) + 1):
        try:
            if (await api.pool.stats()).get("active", 0) == 0:
                raise RuntimeError("tidak ada akun X yang aktif (cek login/cookie)")
            # limit twscrape dibulatkan per halaman (20 post), jadi dipotong tepat di q["limit"]
            rows = []
            async for tw in api.search(q["q"], limit=q["limit"]):
                rows.append(tweet_row(tw, q["key"]))
                if len(rows) >= q["limit"]:
                    break
            return rows, percobaan
        except Exception as e:
            if percobaan == len(jeda):
                log.error("Query gagal setelah %d percobaan: %s (%s)", percobaan + 1, q["key"], e)
                sys.exit("Crawling berhenti; jalankan ulang skrip untuk melanjutkan dari query ini.")
            log.warning("Query error (%s), ulang %d detik lagi: %s", e, jeda[percobaan], q["key"])
            await asyncio.sleep(jeda[percobaan])


async def crawl(api, queries, out_path, progress, progress_path, log, jeda):
    todo = [q for q in queries if q["key"] not in progress["selesai"]]
    log.info("Query total %d, sudah selesai %d, sisa %d",
             len(queries), len(queries) - len(todo), len(todo))
    for i, q in enumerate(todo, 1):
        # Kumpulkan satu query penuh dulu; baru ditulis setelah selesai,
        # sehingga query yang terputus diulang dari awal tanpa duplikat parsial.
        rows, n_ulang = await search_with_retry(api, q, jeda, log)
        with open(out_path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        progress["selesai"][q["key"]] = {
            "putaran": q["putaran"], "seed_word": q["seed_word"], "jumlah": len(rows), "ulang": n_ulang,
            "waktu_crawl": datetime.now(timezone.utc).isoformat(),
        }
        save_progress(progress_path, progress)
        log.info("[%d/%d] %s -> %d post", i, len(todo), q["key"], len(rows))


def summarize(queries, progress):
    done = progress["selesai"]
    per_putaran, per_seed = Counter(), {}
    for q in queries:
        d = done.get(q["key"])
        if d is None:
            continue
        per_putaran[f"putaran{q['putaran']}"] += d["jumlah"]
        s = per_seed.setdefault(q["seed_word"], {"putaran1": 0, "putaran2": 0, "total": 0})
        s[f"putaran{q['putaran']}"] += d["jumlah"]
        s["total"] += d["jumlah"]
    waktu = [d["waktu_crawl"] for d in done.values()]
    return {
        "total_post_mentah": sum(per_putaran.values()),
        "per_putaran": {"putaran1": per_putaran["putaran1"], "putaran2": per_putaran["putaran2"]},
        "per_seed_word": per_seed,
        "query_selesai": len([q for q in queries if q["key"] in done]),
        "query_total": len(queries),
        "query_gagal": len([q for q in queries if q["key"] not in done]),
        "query_perlu_diulang": sum(1 for d in done.values() if d.get("ulang")),
        "tanggal_crawl_terakhir": max(waktu) if waktu else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    engine_dir = cfg_path.parent
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    t1 = cfg["tahap1_acquisition"]
    set_seed(cfg["seed"])

    (engine_dir / cfg["paths"]["logs"]).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(engine_dir / cfg["paths"]["logs"] / "tahap1.log", encoding="utf-8")])
    logging.getLogger("httpx").setLevel(logging.WARNING)  # tanpa log URL per request
    log = logging.getLogger("tahap1")

    seed_words = load_seed_words(engine_dir / t1["seed_words"])
    queries = build_queries(cfg, t1, seed_words)

    out_path = engine_dir / t1["output"]
    progress_path = engine_dir / t1["progress"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not progress_path.exists():
        sys.exit(f"{out_path} sudah ada tanpa berkas progres; tidak akan ditimpa. Periksa manual.")
    progress = load_progress(progress_path)

    async def run():
        api = await setup_api(t1, engine_dir)
        await crawl(api, queries, out_path, progress, progress_path, log, t1["jeda_ulang_detik"])

    asyncio.run(run())

    ringkasan = summarize(queries, progress)
    eval_dir = engine_dir / cfg["paths"]["eval"]
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "ringkasan_tahap1.json").write_text(
        json.dumps(ringkasan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(ringkasan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
