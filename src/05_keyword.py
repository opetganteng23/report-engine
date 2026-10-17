"""Tahap 5a: Keyword baseline (tanpa representasi, tanpa pelatihan).

Aturan disalin dari referensi/kode_classifier.py dan referensi/kode_sentiment.py.
Daftar KATA_NEGATIF dibaca langsung (ast) dari referensi/kode_sentiment.py.

Pemakaian: python src/05_keyword.py --config config.yaml
"""

import argparse
import ast
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import yaml

# Urutan dict = urutan pemecah seri.
KATA_KATEGORI = {
    "Angkutan Umum": [
        "bus", "angkot", "transjakarta", "halte", "terminal", "penumpang", "sopir", "supir", "rute",
        "tarif", "ojek", "angkutan", "trayek", "kondektur", "armada", "busway", "tj", "jaklingko",
        "jak lingko", "mikrotrans", "metrotrans", "mikrolet", "pramudi",
    ],
    "Lahan Parkir": [
        "parkir", "jukir", "retribusi parkir", "gedung parkir", "parkir liar", "tarif parkir",
        "lahan parkir", "karcis parkir", "parkiran", "juru parkir", "tempat parkir", "derek",
    ],
    "Manajemen Lalu Lintas": [
        "macet", "kemacetan", "rambu", "lampu merah", "trotoar", "dishub", "ganjil genap",
        "lalu lintas", "lalin", "kecelakaan", "u-turn", "putar balik", "pengaturan jalan",
        "lampu jalan", "penerangan jalan", "pju", "jpo", "zebra cross", "pejalan kaki",
        "marka jalan", "traffic light", "rekayasa lalu lintas",
    ],
}
JUMLAH_KATA = {"Angkutan Umum": 23, "Lahan Parkir": 12, "Manajemen Lalu Lintas": 22}
KATEGORI_DEFAULT = "Manajemen Lalu Lintas"

KATA_POSITIF = [
    "bagus", "nyaman", "lancar", "bersih", "tepat waktu", "ramah", "cepat", "baik", "aman", "rapi",
    "puas", "membantu", "tertib", "terjangkau", "layak", "memuaskan", "terima kasih", "terang",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_kata_negatif(path):
    """Ambil literal KATA_NEGATIF dari kode lama tanpa menjalankan kodenya."""
    if not path.exists():
        sys.exit(f"{path} belum ada; KATA_NEGATIF harus disalin persis dari berkas itu")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == "KATA_NEGATIF" for t in targets):
                return list(ast.literal_eval(node.value))
    sys.exit(f"KATA_NEGATIF tidak ditemukan di {path}")


def count_matches(text, groups):
    """Hitung kecocokan kata utuh per grup. Frasa terpanjang dicocokkan lebih dulu
    (lintas semua grup) lalu dihapus dari teks agar tidak terhitung dua kali."""
    t = text.lower()
    frasa = sorted(((p, g) for g, ps in groups.items() for p in ps), key=lambda x: -len(x[0]))
    hitung = {g: 0 for g in groups}
    for p, g in frasa:
        pola = re.compile(r"\b" + re.escape(p) + r"\b")
        n = len(pola.findall(t))
        if n:
            hitung[g] += n
            t = pola.sub(" ", t)
    return hitung


def predict(text, kata_negatif):
    h = count_matches(text, KATA_KATEGORI)
    terbanyak = max(h.values())
    kategori = KATEGORI_DEFAULT if terbanyak == 0 else next(k for k in KATA_KATEGORI if h[k] == terbanyak)

    s = count_matches(text, {"Positif": KATA_POSITIF, "Negatif": kata_negatif})
    if s["Positif"] == s["Negatif"]:
        sentimen = "Netral"
    else:
        sentimen = "Positif" if s["Positif"] > s["Negatif"] else "Negatif"
    return {"Kategori": kategori, "Sentimen": sentimen, "Keluhan": "Ya" if sentimen == "Negatif" else "Tidak"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    engine_dir = cfg_path.parent
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    set_seed(cfg["seed"])

    for k, n in JUMLAH_KATA.items():
        assert len(KATA_KATEGORI[k]) == n, k
    assert len(KATA_POSITIF) == 18
    kata_negatif = load_kata_negatif(engine_dir / cfg["paths"]["referensi"] / "kode_sentiment.py")

    with open(engine_dir / cfg["split"]["test"], encoding="utf-8") as f:
        test = [json.loads(l) for l in f if l.strip()]

    pred_dir = engine_dir / cfg["paths"]["predictions"]
    pred_dir.mkdir(parents=True, exist_ok=True)
    preds = []
    with open(pred_dir / "pred_keyword.jsonl", "w", encoding="utf-8") as f:
        for r in test:
            p = {"id": r["id"], "teks_bersih": r["teks_bersih"], **predict(r["teks_bersih"], kata_negatif)}
            preds.append(p)
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    ringkasan = {
        "jumlah_post_uji": len(preds),
        "jumlah_kata": {**JUMLAH_KATA, "Positif": len(KATA_POSITIF), "Negatif": len(kata_negatif)},
        "sebaran_prediksi": {
            kol: {lab: sum(p[kol] == lab for p in preds) for lab in cfg["labels"][kol.lower()]}
            for kol in ("Kategori", "Sentimen", "Keluhan")},
    }
    path = engine_dir / cfg["paths"]["eval"] / "ringkasan_tahap5_keyword.json"
    path.write_text(json.dumps(ringkasan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(ringkasan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
