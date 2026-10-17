"""Tahap 2: Data Preprocessing.

Urutan: (1) text cleaning -> teks_bersih, buang post kosong;
(2) filtering akun instansi resmi dan media; (3) deduplikasi teks_bersih.

Pemakaian: python src/02_preprocessing.py --config config.yaml
"""

import argparse
import csv
import html
import json
import random
import re
import sys
from pathlib import Path

import emoji
import numpy as np
import yaml

RE_RT = re.compile(r"^\s*RT\b\s*(@\w+:?)?")
RE_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
RE_MENTION = re.compile(r"@\w+")
RE_HTML_ENTITY = re.compile(r"&(#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);")
RE_HASHTAG = re.compile(r"#(\w+)")
RE_CAMEL = re.compile(r"[a-z][A-Z]")
RE_SPACE = re.compile(r"\s+")

KOLOM = ["id", "date", "user", "text", "teks_bersih", "reply_to", "query"]
RE_LABEL_BULAN = re.compile(r"\[\d{4}-\d{2}\]$")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def clean_text(text):
    """Kembalikan (teks_bersih, daftar kata hashtag yang dipertahankan)."""
    t = RE_RT.sub(" ", text)
    t = RE_URL.sub(" ", t)
    t = RE_MENTION.sub(" ", t)
    t = RE_HTML_ENTITY.sub(" ", t)
    t = emoji.replace_emoji(t, replace=" ")
    hashtags = RE_HASHTAG.findall(t)
    t = RE_HASHTAG.sub(r"\1", t)
    t = RE_SPACE.sub(" ", t).strip()
    return t, hashtags


def load_akun(path, kolom, jumlah):
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if kolom not in (reader.fieldnames or []):
            sys.exit(f"Kolom '{kolom}' tidak ada di {path}; kolom tersedia: {reader.fieldnames}")
        akun = {r[kolom].strip().lstrip("@").lower() for r in reader if r[kolom].strip()}
    if len(akun) != jumlah:
        sys.exit(f"{path} berisi {len(akun)} akun unik, seharusnya {jumlah}")
    return akun


def putaran(query):
    return 2 if RE_LABEL_BULAN.search(query) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    engine_dir = cfg_path.parent
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    t2 = cfg["tahap2_preprocessing"]
    set_seed(cfg["seed"])
    if t2["kolom_akun"] is None:
        sys.exit("tahap2_preprocessing.kolom_akun belum diisi di config")

    akun_resmi = load_akun(engine_dir / t2["akun_resmi_media"], t2["kolom_akun"],
                           t2["jumlah_akun_resmi_media"])

    with open(engine_dir / t2["input"], encoding="utf-8") as f:
        raw = [json.loads(line) for line in f if line.strip()]
    n_mentah = len(raw)

    # 1. Text cleaning
    tahap1 = []
    for r in raw:
        teks, tags = clean_text(r["text"])
        if teks:
            tahap1.append((dict(r, teks_bersih=teks), tags))
    n_kosong = n_mentah - len(tahap1)

    # 2. Filtering akun instansi resmi dan media
    tahap2, akun_muncul = [], set()
    for r, tags in tahap1:
        u = r["user"].lower()
        if u in akun_resmi:
            akun_muncul.add(u)
        else:
            tahap2.append((r, tags))
    n_akun = len(tahap1) - len(tahap2)

    # 3. Deduplikasi (lowercase + normalisasi spasi), simpan kemunculan pertama
    bersih, seen, n_tag, n_tag_gabungan = [], set(), 0, 0
    for r, tags in tahap2:
        key = RE_SPACE.sub(" ", r["teks_bersih"].lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        bersih.append({k: r[k] for k in KOLOM})
        n_tag += len(tags)
        n_tag_gabungan += sum(1 for t in tags if RE_CAMEL.search(t))
    n_duplikat = len(tahap2) - len(bersih)
    n_bersih = len(bersih)

    if n_mentah - n_kosong - n_akun - n_duplikat != n_bersih:
        sys.exit(f"Cek jumlah gagal: {n_mentah} - {n_kosong} - {n_akun} - {n_duplikat} != {n_bersih}")

    out_path = engine_dir / t2["output"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in bersih:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ringkasan = {
        "post_mentah": n_mentah,
        "post_kosong_dibuang": n_kosong,
        "post_akun_resmi_media_dibuang": n_akun,
        "akun_resmi_media_muncul": len(akun_muncul),
        "duplikat_dibuang": n_duplikat,
        "post_bersih": n_bersih,
        "post_bersih_per_putaran": {
            "putaran1": sum(1 for r in bersih if putaran(r["query"]) == 1),
            "putaran2": sum(1 for r in bersih if putaran(r["query"]) == 2),
        },
        "hashtag_dipertahankan": n_tag,
        "hashtag_gabungan_camelcase": n_tag_gabungan,
    }
    eval_dir = engine_dir / cfg["paths"]["eval"]
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "ringkasan_tahap2.json").write_text(
        json.dumps(ringkasan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(ringkasan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
