"""Tahap 3: Data Annotation dengan LLM lokal (Qwen2.5 1.5B, tanpa fine-tuning).

Satu post per prompt. Prompt = bagian 1 (instruksi) + bagian 2 (definisi kelas)
dari referensi/prompt_anotasi.txt, ditambah bagian 3 (teks post).
Decoding greedy (temperature 0). Keluaran harus satu baris JSON
{"kategori": ..., "sentimen": ..., "keluhan": ...} dengan label yang valid;
jika tidak lolos validator (src/validasi_anotasi.py) diulang sekali dengan prompt yang sama
ditambah instruksi retry format (referensi/prompt_retry_format.txt), lalu ditandai "gagal".
Keluaran mentah, alasan gagal, dan status kedua percobaan dicatat untuk audit.

Pemakaian: python src/03_annotation.py --config config.yaml
"""

import argparse
import json
import platform
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validasi_anotasi import validasi  # noqa: E402

KUNCI = {"kategori": "Kategori", "sentimen": "Sentimen", "keluhan": "Keluhan"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_config(path):
    cfg_path = Path(path).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    a = cfg["annot"]
    for k in ("model_id", "backend", "max_new_tokens"):
        if a.get(k) is None:
            sys.exit(f"config annot.{k} belum diisi")
    if a["backend"] not in ("transformers", "ollama"):
        sys.exit("annot.backend harus 'transformers' atau 'ollama'")
    return cfg, cfg_path.parent


def load_template(path):
    """Baris yang diawali '###' adalah penanda bagian, tidak ikut dikirim ke model."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    tpl = "\n".join(l for l in lines if not l.startswith("###")).strip()
    if tpl.count("{teks}") != 1:
        sys.exit("prompt_anotasi.txt harus memuat tepat satu placeholder {teks}")
    return tpl


def build_prompt(template, teks):
    return template.replace("{teks}", teks)


def load_retry(path):
    """Instruksi retry format, ditempel di akhir prompt pada percobaan kedua. Kosong bila berkas tidak ada."""
    path = Path(path)
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(l for l in lines if not l.startswith("###")).strip()


class TransformersBackend:
    def __init__(self, model_id, max_new_tokens):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device).eval()
        self.max_new_tokens = max_new_tokens

    def perangkat(self):
        if self.device == "cuda":
            return f"cuda ({self.torch.cuda.get_device_name(0)})"
        return f"cpu ({platform.processor()})"

    def generate(self, prompt):
        if self.tok.chat_template:
            text = self.tok.apply_chat_template([{"role": "user", "content": prompt}],
                                                tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        enc = self.tok(text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, do_sample=False, max_new_tokens=self.max_new_tokens,
                                      pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


class OllamaBackend:
    def __init__(self, model_id, max_new_tokens, url, seed):
        import requests
        self.requests = requests
        self.model_id, self.url, self.seed = model_id, url.rstrip("/"), seed
        self.max_new_tokens = max_new_tokens
        if not any(h in self.url for h in ("localhost", "127.0.0.1")):
            sys.exit("annot.ollama_url harus server lokal")

    def perangkat(self):
        ps = self.requests.get(f"{self.url}/api/ps", timeout=30).json().get("models", [])
        vram = sum(m.get("size_vram", 0) for m in ps)
        return f"ollama lokal ({'GPU' if vram else 'CPU'})"

    def generate(self, prompt):
        r = self.requests.post(f"{self.url}/api/chat", timeout=600, json={
            "model": self.model_id, "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0, "seed": self.seed, "num_predict": self.max_new_tokens},
        })
        r.raise_for_status()
        return r.json()["message"]["content"]


def make_backend(cfg):
    a = cfg["annot"]
    if a["backend"] == "transformers":
        return TransformersBackend(a["model_id"], a["max_new_tokens"])
    return OllamaBackend(a["model_id"], a["max_new_tokens"], a["ollama_url"], cfg["seed"])


def parse_output(raw, labels):
    """Label dict (kunci Kategori/Sentimen/Keluhan) bila keluaran lolos validator, selain itu None."""
    obj, _ = validasi(raw, labels)
    return None if obj is None else {KUNCI[k]: obj[k] for k in KUNCI}


def annotate_detail(backend, template, teks, labels, retry=""):
    """Maksimal dua percobaan. Percobaan kedua memakai prompt yang sama ditambah instruksi retry
    format (decoding greedy, jadi prompt yang persis sama akan memberi keluaran yang sama).
    Kembalikan dict: hasil, percobaan, status, raw1, alasan1, raw2, alasan2."""
    prompt = build_prompt(template, teks)
    d = {"hasil": None, "percobaan": 1, "status": "gagal",
         "raw1": "", "alasan1": None, "raw2": None, "alasan2": None}
    d["raw1"] = backend.generate(prompt)
    obj, d["alasan1"] = validasi(d["raw1"], labels)
    if obj is None:
        d["percobaan"] = 2
        prompt2 = f"{prompt}\n\n{retry}" if retry else prompt
        d["raw2"] = backend.generate(prompt2)
        obj, d["alasan2"] = validasi(d["raw2"], labels)
    if obj is not None:
        d["hasil"] = {KUNCI[k]: obj[k] for k in KUNCI}
        d["status"] = "sah_percobaan_1" if d["percobaan"] == 1 else "sah_setelah_retry"
    return d


def annotate(backend, template, teks, labels, retry=""):
    """Kembalikan (label dict atau None, jumlah percobaan, keluaran mentah terakhir)."""
    d = annotate_detail(backend, template, teks, labels, retry)
    return d["hasil"], d["percobaan"], d["raw2"] if d["percobaan"] == 2 else d["raw1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg, engine_dir = load_config(args.config)
    a = cfg["annot"]
    set_seed(cfg["seed"])
    labels = cfg["labels"]
    template = load_template(engine_dir / a["prompt"])
    retry = load_retry(engine_dir / a.get("prompt_retry", "referensi/prompt_retry_format.txt"))

    with open(engine_dir / a["input"], encoding="utf-8") as f:
        posts = [json.loads(l) for l in f if l.strip()]

    out_path = engine_dir / a["output"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_suffix(".partial.jsonl")
    done = {}
    if partial.exists():
        with open(partial, encoding="utf-8") as f:
            for l in f:
                r = json.loads(l)
                done[r["id"]] = r
    print(f"Post total {len(posts)}, sudah dianotasi {len(done)}")

    backend = make_backend(cfg)
    perangkat = backend.perangkat()
    with open(partial, "a", encoding="utf-8") as f:
        for i, p in enumerate(posts, 1):
            if p["id"] in done:
                continue
            t0 = time.perf_counter()
            d = annotate_detail(backend, template, p["teks_bersih"], labels, retry)
            r = dict(p)
            if d["hasil"] is None:
                r.update(Kategori=None, Sentimen=None, Keluhan=None, metode="gagal")
            else:
                r.update(d["hasil"], metode="LLM")
            r["_percobaan"] = d["percobaan"]
            r["_raw"] = d["raw2"] if d["percobaan"] == 2 else d["raw1"]
            r["_audit"] = {k: d[k] for k in ("status", "raw1", "alasan1", "raw2", "alasan2")}
            r["_detik"] = time.perf_counter() - t0
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            done[p["id"]] = r
            if i % 100 == 0:
                print(f"[{i}/{len(posts)}]")

    rows = [done[p["id"]] for p in posts]
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")},
                               ensure_ascii=False) + "\n")
    # Keluaran mentah model disimpan terpisah untuk audit.
    with open(engine_dir / cfg["paths"]["logs"] / "tahap3_keluaran_mentah.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"id": r["id"], "metode": r["metode"], "percobaan": r["_percobaan"],
                                "keluaran": r["_raw"], **r.get("_audit", {})}, ensure_ascii=False) + "\n")

    ok = [r for r in rows if r["metode"] == "LLM"]
    ringkasan = {
        "model_id": a["model_id"],
        "backend": a["backend"],
        "post_dianotasi": len(rows),
        "berhasil": len(ok),
        "berhasil_percobaan_pertama": sum(1 for r in ok if r["_percobaan"] == 1),
        "berhasil_setelah_ulang": sum(1 for r in ok if r["_percobaan"] == 2),
        "gagal": len(rows) - len(ok),
        "alasan_gagal_percobaan_1": dict(Counter(r["_audit"]["alasan1"] for r in rows
                                                 if r.get("_audit", {}).get("alasan1"))),
        "sebaran_label": {kol: {lab: sum(1 for r in ok if r[kol] == lab) for lab in labels[k]}
                          for k, kol in KUNCI.items()},
        "perangkat": perangkat,
        "lama_proses_detik": round(sum(r["_detik"] for r in rows), 1),
    }
    eval_dir = engine_dir / cfg["paths"]["eval"]
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "ringkasan_tahap3.json").write_text(
        json.dumps(ringkasan, ensure_ascii=False, indent=2), encoding="utf-8")
    partial.unlink()
    print(json.dumps(ringkasan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
