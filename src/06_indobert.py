"""Tahap 6: Fine-tuning IndoBERT (indobenchmark/indobert-base-p1).

Satu model per tugas (Kategori, Sentimen, Keluhan): AutoModelForSequenceClassification
(lapisan linear di atas [CLS]). Validasi 10% data latih berstrata. Checkpoint dari
epoch dengan macro-F1 validasi tertinggi disimpan ke models/indobert/<tugas>/.

Pemakaian:
  python src/06_indobert.py latih --config config.yaml
  python src/06_indobert.py prediksi --config config.yaml
"""

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

TUGAS = ["Kategori", "Sentimen", "Keluhan"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def encode(tok, texts, max_length):
    enc = tok(texts, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def make_loader(tok, texts, y, ib, shuffle, seed):
    ids, mask = encode(tok, texts, ib["max_length"])
    tensors = [ids, mask] + ([torch.tensor(y)] if y is not None else [])
    g = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(*tensors), batch_size=ib["batch_size"], shuffle=shuffle, generator=g)


def run_eval(model, loader, device, amp):
    """Kembalikan (loss rata-rata atau None, prediksi indeks)."""
    model.eval()
    preds, total, n = [], 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = [b.to(device) for b in batch]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                out = model(input_ids=batch[0], attention_mask=batch[1],
                            labels=batch[2] if len(batch) == 3 else None)
            if len(batch) == 3:
                total += out.loss.item() * batch[0].size(0)
                n += batch[0].size(0)
            preds.extend(out.logits.argmax(-1).tolist())
    return (total / n if n else None), preds


def gpu_name(device):
    return torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU (tanpa CUDA)"


def latih(cfg, engine_dir):
    ib, seed = cfg["indobert"], cfg["seed"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    train = load_jsonl(engine_dir / cfg["split"]["train"])
    tok = AutoTokenizer.from_pretrained(ib["model_id"])

    riwayat, ringkasan = [], {"gpu": gpu_name(device), "fp16": amp, "tugas": {}}
    for tugas in TUGAS:
        set_seed(seed)
        labels = cfg["labels"][tugas.lower()]
        lab2id = {l: i for i, l in enumerate(labels)}
        texts = [r["teks_bersih"] for r in train]
        y = [lab2id[r[tugas]] for r in train]
        X_tr, X_val, y_tr, y_val = train_test_split(texts, y, test_size=ib["val_size"],
                                                    stratify=y, random_state=seed)
        # Subset tetap untuk akurasi latih per epoch
        idx_eval = random.Random(seed).sample(range(len(X_tr)), min(ib["n_akurasi_latih"], len(X_tr)))

        dl_tr = make_loader(tok, X_tr, y_tr, ib, True, seed)
        dl_val = make_loader(tok, X_val, y_val, ib, False, seed)
        dl_tr_eval = make_loader(tok, [X_tr[i] for i in idx_eval], [y_tr[i] for i in idx_eval], ib, False, seed)

        model = AutoModelForSequenceClassification.from_pretrained(
            ib["model_id"], num_labels=len(labels),
            id2label=dict(enumerate(labels)), label2id=lab2id).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=ib["learning_rate"], weight_decay=ib["weight_decay"])
        total_steps = len(dl_tr) * ib["epochs"]
        sched = get_linear_schedule_with_warmup(opt, int(ib["warmup_ratio"] * total_steps), total_steps)
        scaler = torch.amp.GradScaler(enabled=amp)

        out_dir = engine_dir / cfg["paths"]["models"] / "indobert" / tugas.lower()
        out_dir.mkdir(parents=True, exist_ok=True)
        best_f1, best_epoch, t_tugas = -1.0, None, time.perf_counter()
        for epoch in range(1, ib["epochs"] + 1):
            t0 = time.perf_counter()
            model.train()
            total, n = 0.0, 0
            for ids, mask, yb in dl_tr:
                ids, mask, yb = ids.to(device), mask.to(device), yb.to(device)
                opt.zero_grad()
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    loss = model(input_ids=ids, attention_mask=mask, labels=yb).loss
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), ib["max_grad_norm"])
                scaler.step(opt)
                scaler.update()
                sched.step()
                total += loss.item() * ids.size(0)
                n += ids.size(0)

            _, p_tr = run_eval(model, dl_tr_eval, device, amp)
            val_loss, p_val = run_eval(model, dl_val, device, amp)
            f1 = f1_score(y_val, p_val, average="macro")
            durasi = time.perf_counter() - t0
            riwayat.append({
                "Tugas": tugas, "Epoch": epoch, "Loss latih": total / n, "Loss validasi": val_loss,
                "Akurasi latih": accuracy_score([y_tr[i] for i in idx_eval], p_tr),
                "Akurasi validasi": accuracy_score(y_val, p_val), "Macro-F1 validasi": f1,
                "Lama epoch (detik)": round(durasi, 1),
            })
            print(f"{tugas} epoch {epoch}: " + ", ".join(
                f"{k}={v:.4f}" for k, v in riwayat[-1].items() if isinstance(v, float)))
            if f1 > best_f1:
                best_f1, best_epoch = f1, epoch
                model.save_pretrained(out_dir)
                tok.save_pretrained(out_dir)

        ringkasan["tugas"][tugas] = {
            "ukuran_latih": len(X_tr), "ukuran_validasi": len(X_val),
            "epoch_terbaik": best_epoch, "macro_f1_validasi_terbaik": round(best_f1, 4),
            "lama_pelatihan_detik": round(time.perf_counter() - t_tugas, 1),
        }
        del model, opt
        if amp:
            torch.cuda.empty_cache()

    eval_dir = engine_dir / cfg["paths"]["eval"]
    df = pd.DataFrame(riwayat)
    df.to_excel(eval_dir / "riwayat_indobert.xlsx", index=False)
    plot_riwayat(df, ringkasan, cfg, eval_dir / "epoch_indobert.png")
    return ringkasan


def plot_riwayat(df, ringkasan, cfg, path):
    """Baris = tugas; kolom kiri = loss, kolom kanan = akurasi dan macro-F1.
    Garis putus vertikal di epoch terbaik."""
    fig, axes = plt.subplots(len(TUGAS), 2, figsize=(10, 3.2 * len(TUGAS)), squeeze=False)
    nama_en = {"Kategori": "Category", "Sentimen": "Sentiment", "Keluhan": "Complaint"}
    for row, tugas in enumerate(TUGAS):
        d = df[df["Tugas"] == tugas]
        best = ringkasan["tugas"][tugas]["epoch_terbaik"]
        ax = axes[row][0]
        ax.plot(d["Epoch"], d["Loss latih"], marker="o", label="Train loss")
        ax.plot(d["Epoch"], d["Loss validasi"], marker="o", label="Validation loss")
        ax.set_ylabel("Loss")
        ax2 = axes[row][1]
        ax2.plot(d["Epoch"], d["Akurasi latih"], marker="o", label="Train accuracy")
        ax2.plot(d["Epoch"], d["Akurasi validasi"], marker="o", label="Validation accuracy")
        ax2.plot(d["Epoch"], d["Macro-F1 validasi"], marker="o", label="Validation macro-F1")
        ax2.set_ylabel("Score")
        for a in (ax, ax2):
            a.axvline(best, linestyle="--", color="gray", linewidth=1)
            a.set_xticks(list(d["Epoch"]))
            a.set_xlabel("Epoch")
            a.set_title(nama_en[tugas])
            a.legend(fontsize=8)
            a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def prediksi(cfg, engine_dir):
    ib = cfg["indobert"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    test = load_jsonl(engine_dir / cfg["split"]["test"])
    texts = [r["teks_bersih"] for r in test]
    preds = [{"id": r["id"], "teks_bersih": r["teks_bersih"]} for r in test]
    for tugas in TUGAS:
        model_dir = engine_dir / cfg["paths"]["models"] / "indobert" / tugas.lower()
        tok = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
        _, p = run_eval(model, make_loader(tok, texts, None, ib, False, cfg["seed"]), device, amp)
        for row, i in zip(preds, p):
            row[tugas] = model.config.id2label[i]
    out = engine_dir / cfg["paths"]["predictions"] / "pred_indobert.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in preds:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"jumlah_post_uji": len(preds), "gpu": gpu_name(device),
            "sebaran_prediksi": {t: {l: sum(r[t] == l for r in preds) for l in cfg["labels"][t.lower()]}
                                 for t in TUGAS}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("perintah", choices=["latih", "prediksi"])
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    engine_dir = cfg_path.parent
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    set_seed(cfg["seed"])

    eval_dir = engine_dir / cfg["paths"]["eval"]
    eval_dir.mkdir(parents=True, exist_ok=True)
    path = eval_dir / "ringkasan_tahap6.json"
    ringkasan = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    ringkasan[args.perintah] = latih(cfg, engine_dir) if args.perintah == "latih" else prediksi(cfg, engine_dir)
    path.write_text(json.dumps(ringkasan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(ringkasan[args.perintah], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
