"""Logika analisis sentimen keluhan berbasis leksikon kata positif/negatif."""

from metode_2_kata_kunci.classifier import hitung_kemunculan

KATA_NEGATIF: list[str] = [
    # Kondisi layanan
    "macet", "rusak", "buruk", "lambat", "kotor", "bau", "penuh sesak",
    "berdesakan", "ngetem", "telat", "terlambat", "tidak layak", "bocor",
    "berbahaya", "kecelakaan", "mahal", "curang", "liar", "semrawut",
    "kacau", "bising", "sesak", "tidak nyaman", "parah", "jorok",
    "berisik", "ugal-ugalan", "ngebut", "tidak tertib", "tidak aman",
    "tidak ramah", "lama", "gelap", "pungli", "mogok",
    # Emosi
    "kesal", "kecewa", "marah", "menyebalkan",
    # Kata kasar
    "kontol", "bajingan", "bangsat", "anjing", "babi", "brengsek",
    "kampret", "gembel", "tolol", "idiot", "bodoh", "goblok", "sialan",
]

KATA_POSITIF: list[str] = [
    "bagus", "nyaman", "lancar", "bersih", "tepat waktu", "ramah",
    "cepat", "baik", "aman", "rapi", "puas", "membantu", "tertib",
    "terjangkau", "layak", "memuaskan", "terima kasih", "terang",
]


def hitung_skor_sentimen(teks: str) -> tuple[int, int]:
    """Menghitung jumlah kata positif dan negatif (kata utuh) dalam teks."""
    skor = hitung_kemunculan(teks, {"positif": KATA_POSITIF, "negatif": KATA_NEGATIF})
    return skor["positif"], skor["negatif"]


def analisis_sentimen(teks: str) -> str:
    """Menentukan label sentimen (Positif, Negatif, Netral) dari satu teks keluhan."""
    if not teks or not teks.strip():
        return "Netral"

    skor_positif, skor_negatif = hitung_skor_sentimen(teks)

    if skor_positif == skor_negatif:
        return "Netral"
    return "Positif" if skor_positif > skor_negatif else "Negatif"
