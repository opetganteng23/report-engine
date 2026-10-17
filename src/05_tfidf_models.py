"""Tahap 5b: TF-IDF + Naive Bayes dan TF-IDF + SVM.

Satu Pipeline (tfidf -> klasifier) per metode per tugas. Vectorizer di-fit hanya
pada teks latih (E.1); data uji hanya di-transform (E.2) lewat pipeline.predict.
CV 5-fold terhadap label LLM dicatat sebagai informasi tambahan, bukan evaluasi.

Pemakaian: python src/05_tfidf_models.py --config config.yaml
"""

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

TUGAS = ["Kategori", "Sentimen", "Keluhan"]
TOP_N = 15


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def make_pipeline(metode, seed):
    tfidf = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                            token_pattern=r"(?u)\b[^\W\d_]{2,}\b")
    if metode == "nb":
        klasifier = MultinomialNB(alpha=1.0)
    else:
        klasifier = LinearSVC(C=1.0, class_weight="balanced", random_state=seed)
    return Pipeline([("tfidf", tfidf), ("klasifier", klasifier)])


def top_terms(pipe, metode, tugas):
    """15 istilah berbobot tertinggi per kelas.
    NB: log P(istilah | kelas). SVM: koefisien kelas; untuk tugas biner LinearSVC
    hanya punya satu vektor (kelas classes_[1]), kelas classes_[0] memakai -koefisien."""
    vocab = pipe.named_steps["tfidf"].get_feature_names_out()
    clf = pipe.named_steps["klasifier"]
    classes = list(clf.classes_)
    if metode == "nb":
        bobot = {c: clf.feature_log_prob_[i] for i, c in enumerate(classes)}
    elif clf.coef_.shape[0] == 1:
        bobot = {classes[1]: clf.coef_[0], classes[0]: -clf.coef_[0]}
    else:
        bobot = {c: clf.coef_[i] for i, c in enumerate(classes)}
    rows = []
    for c in classes:
        for rank, j in enumerate(np.argsort(-bobot[c])[:TOP_N], 1):
            rows.append({"Tugas": tugas, "Kelas": c, "Peringkat": rank,
                         "Istilah": vocab[j], "Bobot": float(bobot[c][j])})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    engine_dir = cfg_path.parent
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    seed = cfg["seed"]
    set_seed(seed)

    def load(p):
        with open(engine_dir / p, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    train, test = load(cfg["split"]["train"]), load(cfg["split"]["test"])
    X_train = [r["teks_bersih"] for r in train]
    X_test = [r["teks_bersih"] for r in test]
    assert not set(X_train) & set(X_test), "teks uji bocor ke data latih"

    eval_dir = engine_dir / cfg["paths"]["eval"]
    pred_dir = engine_dir / cfg["paths"]["predictions"]
    pred_dir.mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    ringkasan = {"ukuran_data_latih": len(train), "ukuran_data_uji": len(test), "metode": {}}
    for metode in ("nb", "svm"):
        preds = [{"id": r["id"], "teks_bersih": r["teks_bersih"]} for r in test]
        fitur, info = [], {}
        for tugas in TUGAS:
            y = [r[tugas] for r in train]
            pipe = make_pipeline(metode, seed)

            # Informasi tambahan: CV 5-fold terhadap label LLM (bukan hasil evaluasi)
            skor = cross_validate(make_pipeline(metode, seed), X_train, y, cv=cv,
                                  scoring=["accuracy", "f1_macro"])

            pipe.fit(X_train, y)                       # E.1: fit tfidf + klasifier pada data latih
            for p, lab in zip(preds, pipe.predict(X_test)):  # E.2: transform data uji, tanpa refit
                p[tugas] = str(lab)

            model_dir = engine_dir / cfg["paths"]["models"] / metode / tugas.lower()
            model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipe, model_dir / "model.joblib")
            fitur += top_terms(pipe, metode, tugas)
            info[tugas] = {
                "jumlah_fitur_tfidf": len(pipe.named_steps["tfidf"].vocabulary_),
                "cv5_label_LLM": {
                    "accuracy_mean": round(float(skor["test_accuracy"].mean()), 4),
                    "accuracy_std": round(float(skor["test_accuracy"].std()), 4),
                    "f1_macro_mean": round(float(skor["test_f1_macro"].mean()), 4),
                    "f1_macro_std": round(float(skor["test_f1_macro"].std()), 4),
                },
                "sebaran_prediksi_uji": {lab: sum(p[tugas] == lab for p in preds)
                                         for lab in cfg["labels"][tugas.lower()]},
            }

        with open(pred_dir / f"pred_{metode}.jsonl", "w", encoding="utf-8") as f:
            for p in preds:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        pd.DataFrame(fitur).to_excel(eval_dir / f"fitur_{metode}.xlsx", index=False)
        ringkasan["metode"][metode] = info

    kw = eval_dir / "ringkasan_tahap5_keyword.json"
    if kw.exists():
        ringkasan["keyword"] = json.loads(kw.read_text(encoding="utf-8"))
    (eval_dir / "ringkasan_tahap5.json").write_text(
        json.dumps(ringkasan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(ringkasan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
