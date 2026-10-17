"""Tahap 8: Model Evaluation.

Membandingkan prediksi keempat metode dan label LLM (acuan) dengan label manual
(test_gold.jsonl): accuracy, precision/recall/F1 per kelas, macro-F1, Cohen's kappa,
Wilson 95% CI untuk accuracy, confusion matrix, angka SAFE, tabel dan grafik paper.

Pemakaian: python src/08_evaluation.py --config config.yaml
"""

import argparse
import json
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

TUGAS = ["Kategori", "Sentimen", "Keluhan"]
TUGAS_EN = {"Kategori": "Category", "Sentimen": "Sentiment", "Keluhan": "Complaint"}
# kolom di semua_metode.xlsx -> nama di tabel/grafik
METODE = {"Keyword": "Keyword", "NB": "Naive Bayes", "SVM": "SVM", "IndoBERT": "IndoBERT"}
REF = "LLM labels (ref.)"
EXPLAINABLE = {"Keyword": "Full (rules)", "Naive Bayes": "High (probabilities)",
               "SVM": "High (weights)", "IndoBERT": "Low (post-hoc)"}
TIDAK_RELEVAN = "Tidak Relevan"
LEBAR_KOLOM_IEEE = 3.45


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def wilson(k, n, z):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return c - h, c + h


def evaluasi(y_true, y_pred, labels, z):
    pr, rc, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    k = sum(a == b for a, b in zip(y_true, y_pred))
    lo, hi = wilson(k, len(y_true), z)
    return {
        "n": len(y_true), "accuracy": acc,
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred, labels=labels),
        "wilson_lo": lo, "wilson_hi": hi,
        "per_kelas": {l: {"precision": pr[i], "recall": rc[i], "f1": f1[i], "support": int(sup[i])}
                      for i, l in enumerate(labels)},
        "cm": confusion_matrix(y_true, y_pred, labels=labels),
    }


def gaya_ieee():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "legend.fontsize": 6,
                         "xtick.labelsize": 6, "ytick.labelsize": 6})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    engine_dir = cfg_path.parent
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    set_seed(cfg["seed"])
    z = cfg["split"]["z"]
    en = cfg["label_en"]
    labels = {t: cfg["labels"][t.lower()] for t in TUGAS}
    eval_dir = engine_dir / cfg["paths"]["eval"]

    with open(engine_dir / cfg["split"]["test_gold"], encoding="utf-8") as f:
        gold = pd.DataFrame([json.loads(l) for l in f if l.strip()])
    gold["id"] = gold["id"].astype(str)
    pred = pd.read_excel(engine_dir / cfg["paths"]["predictions"] / "semua_metode.xlsx", dtype={"ID": str})
    df = gold.merge(pred, left_on="id", right_on="ID", how="inner", validate="one_to_one")
    assert len(df) == len(gold) == len(pred), "ID test_gold dan semua_metode tidak cocok"

    # Hasil per sistem (4 metode + label LLM) per tugas
    hasil = {}
    for kol, nama in METODE.items():
        hasil[nama] = {t: evaluasi(df[t].tolist(), df[f"{t} {kol}"].tolist(), labels[t], z) for t in TUGAS}
    # Label LLM: hanya post uji yang berhasil dianotasi LLM (metode_LLM = "LLM")
    llm = df[df["metode_LLM"] == "LLM"]
    hasil[REF] = {t: evaluasi(llm[t].tolist(), llm[f"{t}_LLM"].tolist(), labels[t], z) for t in TUGAS}
    sistem = list(METODE.values()) + [REF]

    # Tabel III: Acc dan MF1
    t3 = pd.DataFrame([{"Method": s, **{f"{TUGAS_EN[t]} {m}": round(100 * hasil[s][t][k], 2)
                                         for t in TUGAS for m, k in (("Acc.", "accuracy"), ("MF1", "macro_f1"))}}
                       for s in sistem])
    t3.to_csv(eval_dir / "tabel_III_akurasi_macroF1.csv", index=False)

    # Tabel IV: kappa dan Wilson CI
    t4 = pd.DataFrame([{"Method": s, **{k: v for t in TUGAS for k, v in (
        (f"{TUGAS_EN[t]} kappa", round(hasil[s][t]["kappa"], 3)),
        (f"{TUGAS_EN[t]} Acc. 95% CI", f"[{100 * hasil[s][t]['wilson_lo']:.2f}, {100 * hasil[s][t]['wilson_hi']:.2f}]"))}}
                       for s in sistem])
    t4.to_csv(eval_dir / "tabel_IV_kappa_wilson.csv", index=False)

    # Tabel V: SAFE (empat metode)
    relevan = df[df["Kategori"] != TIDAK_RELEVAN]
    tr = df[df["Kategori"] == TIDAK_RELEVAN]
    safe, safe_rows = {}, []
    for kol, nama in METODE.items():
        p = f"Kategori {kol}"
        misrouting = 100 * (relevan[p] != relevan["Kategori"]).mean() if len(relevan) else float("nan")
        false_dispatch = 100 * (tr[p] != TIDAK_RELEVAN).mean() if len(tr) else float("nan")
        f1_semua = [(t, l, hasil[nama][t]["per_kelas"][l]["f1"]) for t in TUGAS for l in labels[t]]
        fair = min(f1_semua, key=lambda x: x[2])
        safe[nama] = {"misrouting": misrouting, "false_dispatch": false_dispatch,
                      "accurate": float(np.mean([hasil[nama][t]["macro_f1"] for t in TUGAS])),
                      "fair": fair[2], "fair_kelas": f"{TUGAS_EN[fair[0]]}: {en.get(fair[1], fair[1])}"}
        safe_rows.append({"Method": nama, "Misrouting (%)": round(misrouting, 2),
                          "False dispatch (%)": round(false_dispatch, 2),
                          "Accurate (mean MF1, %)": round(100 * safe[nama]["accurate"], 2),
                          "Fair (min class F1, %)": round(100 * fair[2], 2),
                          "Fair class": safe[nama]["fair_kelas"], "Explainable": EXPLAINABLE[nama]})
    pd.DataFrame(safe_rows).to_csv(eval_dir / "tabel_V_safe.csv", index=False)

    # F1 per kelas dan confusion matrix
    pd.DataFrame([{"Method": s, "Task": TUGAS_EN[t], "Class": en.get(l, l), "Label": l,
                   **{k: (round(v, 4) if k != "support" else v) for k, v in hasil[s][t]["per_kelas"][l].items()}}
                  for s in sistem for t in TUGAS for l in labels[t]]
                 ).to_csv(eval_dir / "f1_per_kelas.csv", index=False)
    pd.DataFrame([{"Method": s, "Task": TUGAS_EN[t], "Manual label": labels[t][i], "Predicted": labels[t][j],
                   "Count": int(hasil[s][t]["cm"][i, j])}
                  for s in sistem for t in TUGAS for i in range(len(labels[t])) for j in range(len(labels[t]))]
                 ).to_csv(eval_dir / "confusion_matrix.csv", index=False)

    # Persentase macro-F1 label LLM yang dipertahankan
    dipertahankan = {nama: {t: 100 * hasil[nama][t]["macro_f1"] / hasil[REF][t]["macro_f1"]
                            if hasil[REF][t]["macro_f1"] else float("nan") for t in TUGAS}
                     for nama in METODE.values()}

    # Grafik
    gaya_ieee()
    metode = list(METODE.values())
    lebar = 0.8 / len(metode)

    fig, ax = plt.subplots(figsize=(LEBAR_KOLOM_IEEE, 2.0))
    x = np.arange(len(TUGAS))
    for i, m in enumerate(metode):
        v = [100 * hasil[m][t]["macro_f1"] for t in TUGAS]
        bars = ax.bar(x + (i - (len(metode) - 1) / 2) * lebar, v, lebar, label=m)
        ax.bar_label(bars, fmt="%.1f", fontsize=4.5, padding=1)
    ax.set_xticks(x, [TUGAS_EN[t] for t in TUGAS])
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_ylim(0, 105)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)
    fig.tight_layout()
    fig.savefig(eval_dir / "fig2_macro_f1.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(LEBAR_KOLOM_IEEE, 4.6))
    for ax, t, huruf in zip(axes, TUGAS, "abc"):
        x = np.arange(len(labels[t]))
        for i, m in enumerate(metode):
            v = [100 * hasil[m][t]["per_kelas"][l]["f1"] for l in labels[t]]
            ax.bar(x + (i - (len(metode) - 1) / 2) * lebar, v, lebar, label=m)
        ax.set_xticks(x, [en.get(l, l) for l in labels[t]])
        ax.set_ylabel("F1 (%)")
        ax.set_ylim(0, 100)
        ax.set_title(f"({huruf}) {TUGAS_EN[t]}")
    axes[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.45), frameon=False)
    fig.tight_layout()
    fig.savefig(eval_dir / "fig3_f1_per_kelas.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(LEBAR_KOLOM_IEEE, 3.2))
    lab_k = labels["Keluhan"]
    for ax, m in zip(axes.flat, metode):
        cm = hasil[m]["Keluhan"]["cm"].astype(float)
        pct = 100 * cm / np.where(cm.sum(1, keepdims=True) == 0, 1, cm.sum(1, keepdims=True))
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        for i in range(len(lab_k)):
            for j in range(len(lab_k)):
                ax.text(j, i, f"{pct[i, j]:.1f}%", ha="center", va="center", fontsize=6,
                        color="white" if pct[i, j] > 60 else "black")
        ax.set_xticks(range(len(lab_k)), [en.get(l, l) for l in lab_k])
        ax.set_yticks(range(len(lab_k)), [en.get(l, l) for l in lab_k])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Manual label")
        ax.set_title(m)
    fig.tight_layout()
    fig.savefig(eval_dir / "fig4_cm_keluhan.png", dpi=300)
    plt.close(fig)

    # Angka tambahan untuk naskah
    netral_ke_negatif = {kol_nama: int(((df["Sentimen"] == "Netral") & (df[f"Sentimen {kol}"] == "Negatif")).sum())
                         for kol, kol_nama in METODE.items()}
    keluhan_ya = {m: {"recall": hasil[m]["Keluhan"]["per_kelas"]["Ya"]["recall"],
                      "precision": hasil[m]["Keluhan"]["per_kelas"]["Ya"]["precision"]} for m in metode}

    ringkasan = {
        "jumlah_post_uji": len(df),
        "jumlah_post_uji_untuk_label_LLM": len(llm),
        "sebaran_label_manual": {t: {l: int((df[t] == l).sum()) for l in labels[t]} for t in TUGAS},
        "hasil": {s: {t: {k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in hasil[s][t].items() if k not in ("per_kelas", "cm")} for t in TUGAS}
                  for s in sistem},
        "safe": {m: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()} for m, s in safe.items()},
        "persen_macro_f1_LLM_dipertahankan": {m: {t: round(v, 2) for t, v in d.items()} for m, d in dipertahankan.items()},
        "netral_diprediksi_negatif": netral_ke_negatif,
        "keluhan_ya_recall_presisi": {m: {k: round(v, 4) for k, v in d.items()} for m, d in keluhan_ya.items()},
    }
    (eval_dir / "ringkasan_tahap8.json").write_text(json.dumps(ringkasan, ensure_ascii=False, indent=2),
                                                    encoding="utf-8")
    tulis_angka_naskah(eval_dir, engine_dir, cfg, ringkasan, hasil, safe, dipertahankan, labels, en)
    print(json.dumps(ringkasan, ensure_ascii=False, indent=2))


def tulis_angka_naskah(eval_dir, engine_dir, cfg, rk, hasil, safe, dipertahankan, labels, en):
    """Kumpulkan angka dari ringkasan semua tahap ke satu berkas markdown."""
    def baca(n):
        p = eval_dir / f"ringkasan_tahap{n}.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    pct = lambda v: f"{100 * v:.2f}%"
    t1, t2, t3, t4, t6 = baca(1), baca(2), baca(3), baca(4), baca(6)
    metode = list(METODE.values())
    L = ["# Angka untuk naskah", "",
         "Dihasilkan otomatis oleh src/08_evaluation.py dari outputs/eval/ringkasan_tahap*.json.", ""]

    L += ["## Abstract", ""]
    for t in TUGAS:
        terbaik = max(metode, key=lambda m: hasil[m][t]["macro_f1"])
        L.append(f"- Macro-F1 terbaik {TUGAS_EN[t]}: {terbaik} {pct(hasil[terbaik][t]['macro_f1'])}")
    L += [f"- Jumlah post uji: {rk['jumlah_post_uji']}", ""]

    L += ["## Proposed Method", "",
          f"- Post mentah: {t1.get('total_post_mentah')} (putaran 1: {t1.get('per_putaran', {}).get('putaran1')}, "
          f"putaran 2: {t1.get('per_putaran', {}).get('putaran2')})",
          f"- Post kosong dibuang: {t2.get('post_kosong_dibuang')}; post akun resmi/media dibuang: "
          f"{t2.get('post_akun_resmi_media_dibuang')} ({t2.get('akun_resmi_media_muncul')} akun muncul); "
          f"duplikat dibuang: {t2.get('duplikat_dibuang')}; post bersih: {t2.get('post_bersih')}",
          f"- Hashtag dipertahankan: {t2.get('hashtag_dipertahankan')}, gabungan: {t2.get('hashtag_gabungan_camelcase')}",
          f"- Anotasi LLM: {t3.get('post_dianotasi')} post, gagal {t3.get('gagal')}, perangkat {t3.get('perangkat')}, "
          f"lama {t3.get('lama_proses_detik')} detik",
          f"- Sebaran label LLM: {json.dumps(t3.get('sebaran_label'), ensure_ascii=False)}",
          f"- Slovin n = {t4.get('slovin_n')}; data uji {t4.get('ukuran_data_uji')}; data latih {t4.get('ukuran_data_latih')}; "
          f"margin of error {json.dumps(t4.get('margin_of_error_persen'))}",
          f"- Data latih: Lahan Parkir {t4.get('persen_latih_Lahan_Parkir')}%, Positif {t4.get('persen_latih_Positif')}%, "
          f"Keluhan Ya {t4.get('persen_latih_Keluhan_Ya')}%"]
    for t, d in (t6.get("latih", {}).get("tugas") or {}).items():
        L.append(f"- IndoBERT {TUGAS_EN[t]}: latih {d['ukuran_latih']}, validasi {d['ukuran_validasi']}, "
                 f"epoch terbaik {d['epoch_terbaik']}, lama {d['lama_pelatihan_detik']} detik")
    if t6.get("latih"):
        L.append(f"- GPU: {t6['latih'].get('gpu')}")
    kappa_manual = baca("4b").get("cohen_kappa")
    if kappa_manual:
        L.append(f"- Kappa antar-anotator: {json.dumps(kappa_manual)}")
    L.append(f"- Sebaran label manual: {json.dumps(rk['sebaran_label_manual'], ensure_ascii=False)}")
    L.append("")

    L += ["## Results", "", "| Method | " + " | ".join(f"{TUGAS_EN[t]} Acc. | {TUGAS_EN[t]} MF1 | {TUGAS_EN[t]} kappa | "
                                                     f"{TUGAS_EN[t]} 95% CI" for t in TUGAS) + " |",
          "|---|" + "---|" * (4 * len(TUGAS))]
    for s in metode + [REF]:
        L.append(f"| {s} | " + " | ".join(
            f"{pct(hasil[s][t]['accuracy'])} | {pct(hasil[s][t]['macro_f1'])} | {hasil[s][t]['kappa']:.3f} | "
            f"[{pct(hasil[s][t]['wilson_lo'])}, {pct(hasil[s][t]['wilson_hi'])}]" for t in TUGAS) + " |")
    L += [f"", f"- Label LLM dievaluasi pada {rk['jumlah_post_uji_untuk_label_LLM']} post uji yang berhasil dianotasi.", "",
          "| Method | Misrouting | False dispatch | Accurate | Fair (kelas) | Explainable |", "|---|---|---|---|---|---|"]
    for m in metode:
        s = safe[m]
        L.append(f"| {m} | {s['misrouting']:.2f}% | {s['false_dispatch']:.2f}% | {pct(s['accurate'])} | "
                 f"{pct(s['fair'])} ({s['fair_kelas']}) | {EXPLAINABLE[m]} |")
    L.append("")

    L += ["## Discussion", ""]
    for m in metode:
        L.append(f"- {m}: macro-F1 label LLM dipertahankan " + ", ".join(
            f"{TUGAS_EN[t]} {dipertahankan[m][t]:.2f}%" for t in TUGAS)
                 + f"; post Netral diprediksi Negatif: {rk['netral_diprediksi_negatif'][m]}; "
                   f"keluhan recall {pct(rk['keluhan_ya_recall_presisi'][m]['recall'])}, "
                   f"presisi {pct(rk['keluhan_ya_recall_presisi'][m]['precision'])}")
    L.append("")

    L += ["## Limitations", "",
          f"- Margin of error data uji: {json.dumps(t4.get('margin_of_error_persen'))}",
          f"- Post gagal anotasi LLM: {t3.get('gagal')}",
          f"- Kelas dengan F1 terendah per metode: " + "; ".join(f"{m} = {safe[m]['fair_kelas']} ({pct(safe[m]['fair'])})"
                                                                for m in metode), ""]

    L += ["## Conclusion", ""]
    for t in TUGAS:
        urut = sorted(metode, key=lambda m: -hasil[m][t]["macro_f1"])
        L.append(f"- {TUGAS_EN[t]} (macro-F1): " + " > ".join(f"{m} {pct(hasil[m][t]['macro_f1'])}" for m in urut))
    L.append("")
    (eval_dir / "angka_naskah.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
