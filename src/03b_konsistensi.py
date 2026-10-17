"""Cek konsistensi anotasi antar-run.

Menganotasi ulang N post (1-50, diambil acak dengan benih global) dari
labeled_posts.jsonl dengan model, prompt, dan decoding yang sama, lalu
menghitung persen label yang sama dengan hasil tahap 3.

Pemakaian: python src/03b_konsistensi.py --config config.yaml --n 50
"""

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("annotation", Path(__file__).with_name("03_annotation.py"))
ann = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ann)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()
    if not 1 <= args.n <= 50:
        sys.exit("--n harus 1 sampai 50")

    cfg, engine_dir = ann.load_config(args.config)
    a = cfg["annot"]
    ann.set_seed(cfg["seed"])
    template = ann.load_template(engine_dir / a["prompt"])

    with open(engine_dir / a["output"], encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    sampel = random.Random(cfg["seed"]).sample(rows, min(args.n, len(rows)))

    backend = ann.make_backend(cfg)
    detail = []
    for r in sampel:
        hasil, _, _ = ann.annotate(backend, template, r["teks_bersih"], cfg["labels"])
        ulang = hasil or {"Kategori": None, "Sentimen": None, "Keluhan": None}
        lama = {k: r[k] for k in ann.KUNCI.values()}
        detail.append({"id": r["id"], "run1": lama, "run2": ulang,
                       "sama": {k: lama[k] == ulang[k] for k in lama}})

    n = len(detail)
    ringkasan = {
        "jumlah_post": n,
        "persen_sama": {k: round(100 * sum(d["sama"][k] for d in detail) / n, 2)
                        for k in ann.KUNCI.values()},
        "persen_ketiga_label_sama": round(100 * sum(all(d["sama"].values()) for d in detail) / n, 2),
        "detail": detail,
    }
    out = engine_dir / a["konsistensi_output"]
    out.write_text(json.dumps(ringkasan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in ringkasan.items() if k != "detail"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
