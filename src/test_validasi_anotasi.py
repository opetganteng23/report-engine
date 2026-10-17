"""Uji validator dengan keluaran buatan (bukan keluaran model). Jalankan: python src/test_validasi_anotasi.py"""

from validasi_anotasi import validasi

LABELS = {"kategori": ["Angkutan Umum", "Lahan Parkir", "Manajemen Lalu Lintas", "Tidak Relevan"],
          "sentimen": ["Positif", "Negatif", "Netral"], "keluhan": ["Ya", "Tidak"]}

KASUS = [
    ('{"kategori": "Angkutan Umum", "sentimen": "Netral", "keluhan": "Tidak"}', None),
    ('  {"kategori": "Lahan Parkir", "sentimen": "Negatif", "keluhan": "Ya"}\n', None),
    ("", "kosong"),
    ("Kategori: Angkutan Umum", "bukan_json"),
    ('```json\n{"kategori": "Angkutan Umum", "sentimen": "Netral", "keluhan": "Tidak"}\n```', "bukan_json"),
    ('{"kategori": "Angkutan Umum", "sentimen": "Netral", "keluhan": "Tidak"} Penjelasan: info.', "teks_tambahan"),
    ('{"kategori": "Angkutan Umum", "sentimen": "Netral", "keluhan": "Tidak"}\n{"kategori": "Angkutan Umum", "sentimen": "Netral", "keluhan": "Tidak"}', "teks_tambahan"),
    ('["Angkutan Umum", "Netral", "Tidak"]', "bukan_objek"),
    ('{"kategori": "Angkutan Umum", "sentimen": "Netral"}', "kunci_salah"),
    ('{"kategori": "Angkutan Umum", "sentimen": "Netral", "keluhan": "Tidak", "alasan": "x"}', "kunci_salah"),
    ('{"kategori": "Angkutan Umum", "sentimen": "Netral", "keluhan": null}', "nilai_bukan_string:keluhan"),
    ('{"kategori": "Angkutan Umum", "sentimen": "Netral", "keluhan": false}', "nilai_bukan_string:keluhan"),
    ('{"kategori": "angkutan umum", "sentimen": "Netral", "keluhan": "Tidak"}', "label_tidak_sah:kategori"),
    ('{"kategori": "Parkir", "sentimen": "Netral", "keluhan": "Tidak"}', "label_tidak_sah:kategori"),
    ('{"kategori": "Angkutan Umum", "sentimen": "Netral ", "keluhan": "Tidak"}', "label_tidak_sah:sentimen"),
    ("{'kategori': 'Angkutan Umum', 'sentimen': 'Netral', 'keluhan': 'Tidak'}", "bukan_json"),
]

gagal = 0
for raw, harapan in KASUS:
    obj, alasan = validasi(raw, LABELS)
    if alasan != harapan or (harapan is None) != (obj is not None):
        gagal += 1
        print(f"GAGAL: {raw!r} -> {alasan}, harapan {harapan}")
print(f"{len(KASUS) - gagal}/{len(KASUS)} kasus lolos")
raise SystemExit(1 if gagal else 0)
