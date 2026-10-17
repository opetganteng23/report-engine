"""Tahap 4: Data Splitting.

Data uji = dua sampel acak dari data bersih (n1 dengan seed1, n2 dengan seed2,
sampel 2 tidak berisi post sampel 1). Data latih = post metode "LLM" dikurangi
post uji (pencocokan teks_bersih persis). Lembar pelabelan manual dibuat buta
(tanpa label LLM), urutan diacak, satu lembar per anotator.

Pemakaian: python src/04_split.py --config config.yaml
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import yaml
from openpyxl import Workbook
from openpyxl.worksheet.datavalidation import DataValidation

KOLOM_MANUAL = ["No", "ID Sampel", "Teks", "Kategori", "Sentimen", "Keluhan", "Catatan"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def manual_paths(base, n_anotator):
    """Satu anotator: test_manual.xlsx; lebih: test_manual_anotator1.xlsx, dst."""
    if n_anotator == 1:
        return [base]
    return [base.with_name(f"{base.stem}_anotator{i}{base.suffix}") for i in range(1, n_anotator + 1)]


def write_manual(path, test_rows, order, labels):
    wb = Workbook()
    ws = wb.active
    ws.title = "Pelabelan"
    ws.append(KOLOM_MANUAL)
    for no, idx in enumerate(order, 1):
        r = test_rows[idx]
        ws.append([no, r["id_sampel"], r["teks_bersih"], None, None, None, None])
    last = len(order) + 1
    for col, key in (("D", "kategori"), ("E", "sentimen"), ("F", "keluhan")):
        dv = DataValidation(type="list", formula1='"' + ",".join(labels[key]) + '"', allow_blank=True)
        ws.add_data_validation(dv)
        dv.add(f"{col}2:{col}{last}")
    ws.column_dimensions["C"].width = 100
    for col in "DEF":
        ws.column_dimensions[col].width = 24
    ws.column_dimensions["G"].width = 30
    ws.freeze_panes = "A2"
    wb.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    engine_dir = cfg_path.parent
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    sp = cfg["split"]
    set_seed(cfg["seed"])
    for k in ("n1", "n2", "seed1", "seed2", "n_anotator"):
        if sp.get(k) is None:
            sys.exit(f"config split.{k} belum diisi")

    base_manual = engine_dir / sp["test_manual"]
    paths_manual = manual_paths(base_manual, sp["n_anotator"])
    ada = [p for p in paths_manual if p.exists()]
    if ada:
        sys.exit(f"Lembar manual sudah ada, tidak akan ditimpa: {[str(p) for p in ada]}")

    with open(engine_dir / sp["input"], encoding="utf-8") as f:
        data = [json.loads(l) for l in f if l.strip()]
    N = len(data)
    e = sp["slovin_e"]
    slovin = N / (1 + N * e ** 2)
    print(f"Slovin: N = {N}, e = {e}, n = {slovin:.2f} (dibulatkan ke atas {math.ceil(slovin)}); "
          f"config n1 = {sp['n1']}")

    # Sampel 1, lalu sampel 2 dari sisa data
    idx_all = list(range(N))
    s1 = random.Random(sp["seed1"]).sample(idx_all, sp["n1"])
    sisa = [i for i in idx_all if i not in set(s1)]
    s2 = random.Random(sp["seed2"]).sample(sisa, sp["n2"])
    assert not set(s1) & set(s2)

    test = []
    for sampel, idxs in ((1, s1), (2, s2)):
        for i in idxs:
            test.append(dict(data[i], sampel=sampel))
    for k, r in enumerate(test, 1):
        r["id_sampel"] = f"S{k:04d}"

    # Data latih: metode LLM, tanpa post uji (teks_bersih persis sama)
    teks_uji = {r["teks_bersih"] for r in test}
    llm = [r for r in data if r["metode"] == "LLM"]
    train = [r for r in llm if r["teks_bersih"] not in teks_uji]
    dibuang = len(llm) - len(train)
    assert not {r["teks_bersih"] for r in train} & teks_uji

    out_dir = engine_dir / sp["train"]
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    for path, rows in ((engine_dir / sp["train"], train), (engine_dir / sp["test"], test)):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Lembar manual buta, urutan diacak (benih global), sama untuk semua anotator
    order = list(range(len(test)))
    random.Random(cfg["seed"]).shuffle(order)
    for p in paths_manual:
        write_manual(p, test, order, cfg["labels"])

    n_test = len(test)
    pct = lambda kol, lab: round(100 * sum(r[kol] == lab for r in train) / len(train), 2) if train else None
    moe = lambda n: round(sp["z"] * math.sqrt(0.25 / n) * 100, 2)
    ringkasan = {
        "N_data_bersih": N,
        "slovin_n": round(slovin, 2),
        "n1_config": sp["n1"],
        "n2_config": sp["n2"],
        "ukuran_data_uji": n_test,
        "ukuran_data_latih": len(train),
        "post_metode_LLM": len(llm),
        "post_uji_dibuang_dari_latih": dibuang,
        "persen_latih_Lahan_Parkir": pct("Kategori", "Lahan Parkir"),
        "persen_latih_Positif": pct("Sentimen", "Positif"),
        "persen_latih_Keluhan_Ya": pct("Keluhan", "Ya"),
        "margin_of_error_persen": {"data_uji": moe(n_test), "sampel1": moe(sp["n1"])},
        "lembar_manual": [str(p.relative_to(engine_dir)) for p in paths_manual],
    }
    eval_dir = engine_dir / cfg["paths"]["eval"]
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "ringkasan_tahap4.json").write_text(
        json.dumps(ringkasan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(ringkasan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
