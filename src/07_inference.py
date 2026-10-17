"""Tahap 7: Model Inference.

Menjalankan keempat metode (keyword, NB, SVM, IndoBERT) pada data uji yang sama
memakai model tersimpan (tanpa pelatihan ulang), membandingkan hasilnya dengan
prediksi tahap 5 dan 6, lalu menggabungkan semuanya ke semua_metode.xlsx.
Berkas pred_*.jsonl dari tahap 5 dan 6 tidak ditimpa.

Pemakaian: python src/07_inference.py --config config.yaml
"""

import argparse
import importlib.util
import json
from pathlib import Path

import joblib
import pandas as pd
import torch
import yaml

SRC = Path(__file__).parent
TUGAS = ["Kategori", "Sentimen", "Keluhan"]
METODE = {"keyword": "Keyword", "nb": "NB", "svm": "SVM", "indobert": "IndoBERT"}


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, SRC / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


kw = load_module("keyword", "05_keyword.py")
ib = load_module("indobert", "06_indobert.py")


def pred_keyword(cfg, engine_dir, texts):
    neg = kw.load_kata_negatif(engine_dir / cfg["paths"]["referensi"] / "kode_sentiment.py")
    return [kw.predict(t, neg) for t in texts]


def pred_tfidf(cfg, engine_dir, texts, metode):
    preds = [{} for _ in texts]
    for tugas in TUGAS:
        pipe = joblib.load(engine_dir / cfg["paths"]["models"] / metode / tugas.lower() / "model.joblib")
        for p, lab in zip(preds, pipe.predict(texts)):
            p[tugas] = str(lab)
    return preds


def pred_indobert(cfg, engine_dir, texts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    preds = [{} for _ in texts]
    for tugas in TUGAS:
        model_dir = engine_dir / cfg["paths"]["models"] / "indobert" / tugas.lower()
        tok = ib.AutoTokenizer.from_pretrained(model_dir)
        model = ib.AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
        loader = ib.make_loader(tok, texts, None, cfg["indobert"], False, cfg["seed"])
        _, idx = ib.run_eval(model, loader, device, amp)
        for p, i in zip(preds, idx):
            p[tugas] = model.config.id2label[i]
    return preds


def bandingkan(path, test, preds):
    """Jumlah prediksi yang berbeda per tugas dibanding berkas tahap 5/6."""
    if not path.exists():
        return {"status": "berkas pembanding tidak ada"}
    with open(path, encoding="utf-8") as f:
        lama = {r["id"]: r for r in (json.loads(l) for l in f if l.strip())}
    if set(lama) != {r["id"] for r in test}:
        return {"status": "ID post berbeda dengan data uji"}
    beda = {t: [r["id"] for r, p in zip(test, preds) if lama[r["id"]][t] != p[t]] for t in TUGAS}
    return {"status": "sama" if not any(beda.values()) else "BERBEDA",
            "jumlah_beda": {t: len(v) for t, v in beda.items()},
            "contoh_id_beda": {t: v[:10] for t, v in beda.items() if v}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    engine_dir = cfg_path.parent
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    ib.set_seed(cfg["seed"])

    test = ib.load_jsonl(engine_dir / cfg["split"]["test"])
    texts = [r["teks_bersih"] for r in test]
    pred_dir = engine_dir / cfg["paths"]["predictions"]

    hasil = {
        "keyword": pred_keyword(cfg, engine_dir, texts),
        "nb": pred_tfidf(cfg, engine_dir, texts, "nb"),
        "svm": pred_tfidf(cfg, engine_dir, texts, "svm"),
        "indobert": pred_indobert(cfg, engine_dir, texts),
    }
    cek = {m: bandingkan(pred_dir / f"pred_{m}.jsonl", test, p) for m, p in hasil.items()}

    rows = []
    for i, r in enumerate(test):
        row = {"ID": r["id"], "Teks": r["teks_bersih"]}
        row.update({f"{t} LLM": r.get(t) for t in TUGAS})
        for m, nama in METODE.items():
            row.update({f"{t} {nama}": hasil[m][i][t] for t in TUGAS})
        rows.append(row)
    pd.DataFrame(rows).to_excel(pred_dir / "semua_metode.xlsx", index=False)

    ringkasan = {
        "jumlah_post_uji": len(test),
        "post_uji_tanpa_label_LLM": sum(r.get("metode") != "LLM" for r in test),
        "cek_konsistensi_dengan_tahap5_6": cek,
        "semua_sama": all(c.get("status") == "sama" for c in cek.values()),
    }
    (engine_dir / cfg["paths"]["eval"] / "ringkasan_tahap7.json").write_text(
        json.dumps(ringkasan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(ringkasan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
