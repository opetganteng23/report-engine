"""Validator keluaran anotasi Qwen (dipakai src/03_annotation.py dan src/03c_uji_prompt.py).

Keluaran dinyatakan sah hanya jika seluruh teks (setelah spasi di tepi dibuang) adalah satu
objek JSON dengan tepat kunci kategori, sentimen, keluhan, setiap nilai berupa string, dan
setiap nilai persis ada di daftar label config.yaml. Label tidak pernah diubah atau ditebak.
"""

import json

KUNCI = ("kategori", "sentimen", "keluhan")


def validasi(raw, labels):
    """Kembalikan (obj, alasan). obj berisi tiga label bila sah; alasan None bila sah."""
    teks = (raw or "").strip()
    if not teks:
        return None, "kosong"
    try:
        obj, akhir = json.JSONDecoder().raw_decode(teks)
    except json.JSONDecodeError:
        return None, "bukan_json"
    if teks[akhir:].strip():
        return None, "teks_tambahan"
    if not isinstance(obj, dict):
        return None, "bukan_objek"
    if set(obj) != set(KUNCI):
        return None, "kunci_salah"
    for k in KUNCI:
        if not isinstance(obj[k], str):
            return None, f"nilai_bukan_string:{k}"
        if obj[k] not in labels[k]:
            return None, f"label_tidak_sah:{k}"
    return {k: obj[k] for k in KUNCI}, None
