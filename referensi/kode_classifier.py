"""Logika klasifikasi kategori keluhan transportasi berbasis skor kata kunci."""

import re

KATEGORI_KEYWORDS: dict[str, list[str]] = {
    "Angkutan Umum": [
        "bus", "angkot", "transjakarta", "halte", "terminal", "penumpang",
        "sopir", "supir", "rute", "tarif", "ojek", "angkutan", "trayek",
        "kondektur", "armada", "busway", "tj", "jaklingko", "jak lingko",
        "mikrotrans", "metrotrans", "mikrolet", "pramudi",
    ],
    "Lahan Parkir": [
        "parkir", "jukir", "retribusi parkir", "gedung parkir",
        "parkir liar", "tarif parkir", "lahan parkir", "karcis parkir",
        "parkiran", "juru parkir", "tempat parkir", "derek",
    ],
    "Manajemen Lalu Lintas": [
        "macet", "kemacetan", "rambu", "lampu merah", "trotoar", "dishub",
        "ganjil genap", "lalu lintas", "lalin", "kecelakaan", "u-turn",
        "putar balik", "pengaturan jalan", "lampu jalan", "penerangan jalan",
        "pju", "jpo", "zebra cross", "pejalan kaki", "marka jalan",
        "traffic light", "rekayasa lalu lintas",
    ],
}

KATEGORI_DEFAULT = "Manajemen Lalu Lintas"


def kata_cocok(teks: str, kamus: dict[str, list[str]]) -> dict[str, list[str]]:
    """Mengembalikan kata kunci yang cocok per label (kata utuh, satu entri per kemunculan).

    Frasa terpanjang dicocokkan lebih dulu lalu ditandai, sehingga misalnya
    "tarif parkir" tidak ikut terhitung lagi sebagai "tarif".
    """
    teks_sisa = teks.lower()
    cocok: dict[str, list[str]] = {label: [] for label in kamus}
    pasangan = sorted(
        ((kata, label) for label, daftar in kamus.items() for kata in set(daftar)),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    for kata, label in pasangan:
        pola = re.compile(rf"(?<!\w){re.escape(kata)}(?!\w)")
        teks_sisa, jumlah = pola.subn(" ", teks_sisa)
        cocok[label].extend([kata] * jumlah)
    return cocok


def hitung_kemunculan(teks: str, kamus: dict[str, list[str]]) -> dict[str, int]:
    """Menghitung kemunculan kata kunci per label (lihat kata_cocok)."""
    return {label: len(daftar) for label, daftar in kata_cocok(teks, kamus).items()}


def hitung_skor_kategori(teks: str) -> dict[str, int]:
    """Menghitung skor kata kunci tiap kategori pada satu teks keluhan."""
    return hitung_kemunculan(teks, KATEGORI_KEYWORDS)


def klasifikasi_keluhan(teks: str) -> str:
    """Menentukan kategori keluhan berdasarkan skor kata kunci tertinggi.

    Jika seluruh skor nol, kembalikan kategori default. Jika seri,
    kategori yang lebih dulu di KATEGORI_KEYWORDS yang dipilih.
    """
    if not teks or not teks.strip():
        return KATEGORI_DEFAULT

    skor = hitung_skor_kategori(teks)
    if max(skor.values()) == 0:
        return KATEGORI_DEFAULT
    return max(skor, key=skor.get)
