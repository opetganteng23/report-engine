"""Bagian bersama metode TF-IDF (Naive Bayes & SVM): data latih, pelatihan, validasi, bukti, prediksi.

Data latih  : label LLM dari analisis massal (Downloads/analisis_llm/progres_llm_15000.jsonl),
              DIKURANGI 385 sampel uji (supaya tidak ada kebocoran ke data evaluasi),
              hanya baris yang benar-benar dilabeli LLM (bukan fallback/kosong).
Data uji    : 385 sampel di Downloads/evaluasi_akurasi/hasil_engine_sampel.xlsx (dilabeli manual).
Model       : satu pipeline TF-IDF + klasifier untuk KATEGORI, satu untuk SENTIMEN.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Callable

import joblib
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

logger = logging.getLogger("tfidf")

# EVAL_VERSI=v2 -> data latih = dataset final 10.000 (label LLM v3.1 + pasca-proses), uji = 385 sampel v2,
# dan model juga belajar "Keluhan". Tanpa variabel itu -> konfigurasi evaluasi versi 1.
VERSI = os.environ.get("EVAL_VERSI", "v1")
if VERSI == "v3":  # evaluasi final 1.000 sampel (data latih tanpa 1.000 sampel uji)
    FILE_PROGRES = Path.home() / "Downloads" / "Hasil_Penelitian" / "_file_kerja" / "analisis_llm" / "hasil_v3_10000_final.jsonl"
    FILE_SAMPEL_UJI = Path.home() / "Downloads" / "Hasil_Penelitian" / "_file_kerja" / "evaluasi_akurasi" / "hasil_engine_sampel_v3.xlsx"
    TUGAS = ("Kategori", "Sentimen", "Keluhan")
elif VERSI == "v2":
    FILE_PROGRES = Path.home() / "Downloads" / "Hasil_Penelitian" / "_file_kerja" / "analisis_llm" / "hasil_v3_10000_final.jsonl"
    FILE_SAMPEL_UJI = Path.home() / "Downloads" / "Hasil_Penelitian" / "_file_kerja" / "evaluasi_akurasi" / "hasil_engine_sampel_v2.xlsx"
    TUGAS = ("Kategori", "Sentimen", "Keluhan")
else:
    FILE_PROGRES = Path.home() / "Downloads" / "Hasil_Penelitian" / "_file_kerja" / "analisis_llm" / "progres_llm_15000.jsonl"
    FILE_SAMPEL_UJI = Path.home() / "Downloads" / "Hasil_Penelitian" / "_file_kerja" / "evaluasi_akurasi" / "hasil_engine_sampel.xlsx"
    TUGAS = ("Kategori", "Sentimen")
BENIH = 2026
TEBAL = Font(bold=True)

# Parameter TF-IDF (sama untuk kedua metode supaya perbandingannya adil)
PARAM_TFIDF = {
    "lowercase": True,
    "ngram_range": (1, 2),        # unigram + bigram
    "min_df": 2,                  # abaikan istilah yang hanya muncul di 1 tweet
    "sublinear_tf": True,         # 1 + log(tf)
    "token_pattern": r"(?u)\b[^\W\d_]{2,}\b",  # kata huruf saja, minimal 2 huruf
}


def teks_sampel_uji() -> list[tuple[str, str]]:
    """(ID Sampel, Teks) dari 385 sampel uji."""
    sheet = load_workbook(FILE_SAMPEL_UJI, read_only=True)["Label Engine"]
    return [(b[0], b[1] or "") for b in list(sheet.iter_rows(values_only=True))[1:]]


def muat_data_latih() -> tuple[list[str], dict[str, list[str]], int]:
    """Teks + label per tugas untuk pelatihan; kembalikan juga jumlah baris uji yang dikeluarkan."""
    uji = {t for _, t in teks_sampel_uji()}
    teks, label = [], {t: [] for t in TUGAS}
    dikeluarkan = 0
    with FILE_PROGRES.open(encoding="utf-8") as f:
        for d in map(json.loads, f):
            if d["Metode"] != "LLM" or not d["teks_bersih"].strip():
                continue
            if d["teks_bersih"] in uji:
                dikeluarkan += 1
                continue
            teks.append(d["teks_bersih"])
            for t in TUGAS:
                label[t].append(d[t])
    return teks, label, dikeluarkan


def fitur_teratas(model: Pipeline, n: int = 15) -> dict[str, list[tuple[str, float]]]:
    """Istilah dengan bobot tertinggi per kelas (bukti cara kerja model)."""
    vocab = model.named_steps["tfidf"].get_feature_names_out()
    klf = model.named_steps["klasifier"]
    bobot = klf.feature_log_prob_ if hasattr(klf, "feature_log_prob_") else klf.coef_
    hasil = {}
    for kelas, baris in zip(klf.classes_, bobot):
        urut = baris.argsort()[::-1][:n]
        hasil[kelas] = [(vocab[i], float(baris[i])) for i in urut]
    return hasil


def jalankan(nama: str, singkat: str, buat_klasifier: Callable, folder: Path) -> None:
    """Latih + validasi silang + simpan model + tulis bukti + prediksi 385 sampel uji."""
    teks, label, dikeluarkan = muat_data_latih()
    logger.info("%s: data latih %d tweet (sampel uji dikeluarkan: %d)", nama, len(teks), dikeluarkan)
    (folder / "model").mkdir(exist_ok=True)

    buku = Workbook()
    ring = buku.active
    ring.title = "Ringkasan"
    ring.append([f"Bukti cara kerja: {nama}"])
    ring.append(["Dibuat", datetime.now().strftime("%d-%m-%Y %H:%M")])
    ring.append(["Data latih", f"{len(teks)} tweet berlabel LLM (385 sampel uji dikeluarkan: {dikeluarkan} cocok)"])
    ring.append(["Parameter TF-IDF", json.dumps({k: str(v) for k, v in PARAM_TFIDF.items()})])
    ring.append(["Klasifier", repr(buat_klasifier())])
    ring.append([])
    ring.append(["Validasi silang 5-fold terhadap label LLM (kesesuaian dengan LLM, BUKAN akurasi terhadap manusia)"])
    ring.append(["Tugas", "Kesesuaian (%)", "Macro-F1 (%)", "Jumlah per kelas (data latih)"])
    for sel in ring[1] + ring[8]:
        sel.font = TEBAL

    sampel = teks_sampel_uji()
    prediksi: dict[str, list[str]] = {}
    for tugas in TUGAS:
        y = label[tugas]
        model = Pipeline([("tfidf", TfidfVectorizer(**PARAM_TFIDF)), ("klasifier", buat_klasifier())])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=BENIH)
        tebak = cross_val_predict(model, teks, y, cv=cv)
        acc, mf1 = accuracy_score(y, tebak), f1_score(y, tebak, average="macro")
        sebaran = {k: y.count(k) for k in sorted(set(y))}
        ring.append([tugas, round(acc * 100, 1), round(mf1 * 100, 1), json.dumps(sebaran, ensure_ascii=False)])
        logger.info("  %s: kesesuaian CV %.1f%%, macro-F1 %.1f%%", tugas, acc * 100, mf1 * 100)

        model.fit(teks, y)
        joblib.dump(model, folder / "model" / (f"{tugas.lower()}_{VERSI}.joblib" if VERSI != "v1" else f"{tugas.lower()}.joblib"))
        prediksi[tugas] = list(model.predict([t for _, t in sampel]))

        sheet = buku.create_sheet(f"Fitur {tugas}")
        sheet.append(["Kelas", "Peringkat", "Istilah", "Bobot"])
        for sel in sheet[1]:
            sel.font = TEBAL
        for kelas, daftar in fitur_teratas(model).items():
            for r, (istilah, b) in enumerate(daftar, 1):
                sheet.append([kelas, r, istilah, round(b, 4)])
        for h, w in zip("ABCD", [24, 10, 28, 10]):
            sheet.column_dimensions[h].width = w
    ring.column_dimensions["A"].width = 30
    ring.column_dimensions["B"].width = 16
    ring.column_dimensions["C"].width = 14
    ring.column_dimensions["D"].width = 90

    bukti = FILE_SAMPEL_UJI.parent / (f"bukti_{singkat}_{VERSI}.xlsx" if VERSI != "v1" else f"bukti_{singkat}.xlsx")
    buku.save(bukti)
    tulis_prediksi_uji(singkat, sampel, prediksi)
    logger.info("  Model: %s | Bukti: %s", folder / "model", bukti)


def tulis_prediksi_uji(singkat: str, sampel: list[tuple[str, str]], prediksi: dict[str, list[str]]) -> None:
    """Tambahkan/ganti kolom 'Kategori <singkat>' & 'Sentimen <singkat>' di sheet Label Engine."""
    buku = load_workbook(FILE_SAMPEL_UJI)
    sheet = buku["Label Engine"]
    judul = [c.value for c in sheet[1]]
    for tugas in TUGAS:
        nama_kolom = f"{tugas} {singkat}"
        kolom = judul.index(nama_kolom) + 1 if nama_kolom in judul else len(judul) + 1
        if nama_kolom not in judul:
            judul.append(nama_kolom)
        sheet.cell(row=1, column=kolom, value=nama_kolom).font = TEBAL
        for r, nilai in enumerate(prediksi[tugas], 2):
            sheet.cell(row=r, column=kolom, value=nilai)
        sheet.column_dimensions[sheet.cell(row=1, column=kolom).column_letter].width = 22
    buku.save(FILE_SAMPEL_UJI)
