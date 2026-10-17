"""Uji prompt anotasi pada sejumlah kecil post (mis. 10 post uji dari OPET).

Setiap post dijalankan terpisah dengan prompt, model, dan decoding yang sama seperti tahap 3.
Dicatat: teks, keluaran mentah, status validasi JSON, alasan gagal, dan tiga label.
Bila berkas masukan memuat label manusia (kolom kategori_manual, sentimen_manual,
keluhan_manual), dihitung kesesuaian dengan label itu. Tanpa label manusia, hasilnya hanya
pemeriksaan format dan konsistensi, BUKAN akurasi anotasi.

Masukan: .txt (satu post per baris) atau .csv/.xlsx dengan kolom "teks".
Pemakaian:
  python src/03c_uji_prompt.py --config config.yaml --input referensi/uji_10_post.csv --label v2
  python src/03c_uji_prompt.py --banding outputs/eval/uji_prompt_v1.csv outputs/eval/uji_prompt_v2.csv
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location("annotation", Path(__file__).with_name("03_annotation.py"))
ann = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ann)

LABEL = ["Kategori", "Sentimen", "Keluhan"]


def baca_post(path):
    path = Path(path)
    if path.suffix.lower() == ".txt":
        teks = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        return pd.DataFrame({"teks": teks})
    df = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path)
    if "teks" not in df.columns:
        sys.exit("berkas masukan harus punya kolom 'teks'")
    return df


def uji(args):
    cfg, engine_dir = ann.load_config(args.config)
    a = cfg["annot"]
    ann.set_seed(cfg["seed"])
    template = ann.load_template(engine_dir / a["prompt"])
    retry = ann.load_retry(engine_dir / a.get("prompt_retry", "referensi/prompt_retry_format.txt"))
    df = baca_post(engine_dir / args.input if not Path(args.input).is_absolute() else args.input)
    backend = ann.make_backend(cfg)

    rows = []
    for i, r in df.iterrows():
        d = ann.annotate_detail(backend, template, str(r["teks"]), cfg["labels"], retry)
        h = d["hasil"] or {k: None for k in LABEL}
        row = {"no": i + 1, "teks": r["teks"], "keluaran_mentah_1": d["raw1"],
               "alasan_1": d["alasan1"], "keluaran_mentah_2": d["raw2"], "alasan_2": d["alasan2"],
               "status": d["status"], **h}
        for k in LABEL:
            kol = f"{k.lower()}_manual"
            if kol in df.columns and pd.notna(r[kol]):
                row[kol] = r[kol]
                row[f"{k}_sesuai"] = h[k] == r[kol]
        rows.append(row)
        print(f"[{i + 1}/{len(df)}] {d['status']}: {d['raw1']!r}")

    out = pd.DataFrame(rows)
    eval_dir = engine_dir / cfg["paths"]["eval"]
    eval_dir.mkdir(parents=True, exist_ok=True)
    csv = eval_dir / f"uji_prompt_{args.label}.csv"
    out.to_csv(csv, index=False, encoding="utf-8-sig")

    n = len(out)
    print(f"\nPost diuji: {n}")
    print(f"Sah pada percobaan 1: {(out.status == 'sah_percobaan_1').sum()}")
    print(f"Sah setelah retry: {(out.status == 'sah_setelah_retry').sum()}")
    print(f"Gagal: {(out.status == 'gagal').sum()}")
    sesuai = [c for c in out.columns if c.endswith("_sesuai")]
    if sesuai:
        for c in sesuai:
            print(f"Kesesuaian {c[:-7]} dengan label manusia: {out[c].sum()}/{out[c].notna().sum()}")
    else:
        print("Tidak ada label manusia: hasil ini pemeriksaan format dan konsistensi, bukan akurasi.")
    print(f"Tersimpan: {csv}")


def banding(a, b):
    x, y = pd.read_csv(a), pd.read_csv(b)
    m = x.merge(y, on=["no", "teks"], suffixes=("_sebelum", "_sesudah"))
    kol = ["no", "teks"] + [f"{k}_{s}" for k in ["status"] + LABEL for s in ("sebelum", "sesudah")]
    m["berubah"] = [any(r[f"{k}_sebelum"] != r[f"{k}_sesudah"] for k in LABEL) for _, r in m.iterrows()]
    print(m[kol + ["berubah"]].to_string(index=False))
    print(f"\nPost dengan label berubah: {m.berubah.sum()}/{len(m)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--input")
    ap.add_argument("--label", default="uji")
    ap.add_argument("--banding", nargs=2, metavar=("SEBELUM", "SESUDAH"))
    args = ap.parse_args()
    if args.banding:
        banding(*args.banding)
    elif args.config and args.input:
        uji(args)
    else:
        sys.exit("pakai --config dan --input, atau --banding SEBELUM SESUDAH")


if __name__ == "__main__":
    main()
