# Laporan prompt anotasi Qwen2.5-1.5B-Instruct (versi 2, 17 Oktober 2026)

Disusun oleh Claude mengikuti spesifikasi OPET ("TASK: Rancang dan Validasi Prompt Anotasi").
Model belum dijalankan sama sekali dalam pekerjaan ini. Laporan ini tidak memuat hasil inferensi, akurasi, atau angka kesesuaian.

## 1. Masalah pada prompt dan implementasi lama

1. **Retry tidak berguna.** `03_annotation.py` lama mengulang dengan prompt yang persis sama. Karena decoding greedy (temperature 0), percobaan kedua hampir pasti memberi keluaran yang sama, sehingga retry tidak memperbaiki apa pun.
2. **Kegagalan tidak bisa dianalisis.** Hanya keluaran mentah terakhir yang disimpan, tanpa keluaran pertama dan tanpa alasan gagal.
3. **Prompt v1 terlalu tipis untuk spesifikasi.** Belum ada batas pembeda antarkategori, aturan keputusan bernomor, contoh pertanyaan informatif, negasi, multi-topik, Negatif tanpa keluhan, atau Positif dengan keluhan. Prompt v1 juga hanya punya dua contoh.
4. **Prompt v1 menyamakan keluhan dengan sentimen Negatif** ("Keluhan, termasuk sindiran = Negatif"). Ini bertentangan dengan aturan 7 di spesifikasi.

## 2. Prompt final

Prompt final ada di `referensi/prompt_anotasi.txt`. Instruksi retry ada di berkas terpisah, `referensi/prompt_retry_format.txt`, agar prompt utama tetap ringkas.

Strukturnya mengikuti enam bagian spesifikasi: instruksi tugas, definisi operasional, 11 aturan keputusan, 11 contoh, format keluaran, dan placeholder `{teks}`. Panjangnya sekitar 780 kata.

## 3. Tabel pengujian sepuluh post

**Belum ada.** Sepuluh post uji tidak ditemukan di proyek, engine, maupun percakapan, jadi saya tidak mengarangnya.

Model juga tidak bisa dijalankan di lingkungan saya karena akses ke huggingface.co diblokir. Pengujian harus dijalankan di komputer OPET:

```
python src/03c_uji_prompt.py --config config.yaml --input referensi/uji_10_post.csv --label v2
```

Format berkas `uji_10_post.csv`:
- kolom wajib: `teks`;
- kolom opsional: `kategori_manual`, `sentimen_manual`, `keluhan_manual`.

Kesesuaian hanya dihitung bila label manual diisi. Tanpa label manual, hasilnya adalah pemeriksaan format, bukan akurasi.

## 4. Perubahan dan alasannya

| Perubahan | Alasan |
|---|---|
| Definisi kategori diberi pola cakupan, aspek, kapan dipakai, dan batas pembeda | Diminta spesifikasi (Bagian 2). Isinya tetap Table II dan suplemen S4 naskah v45, tidak ada aspek yang ditambah atau dibuang. |
| Sentimen dan keluhan dipisah (aturan 6 dan 7) | Spesifikasi aturan 6 dan 7. |
| Aturan 11: post Tidak Relevan selalu Keluhan Tidak, sentimen tetap dinilai dari isi | Spesifikasi aturan 12 membolehkan pengecualian dengan alasan metodologis. Alasannya, keluhan di studi ini adalah keluhan transportasi yang diteruskan ke instansi. **OPET memilih "Selalu Tidak" pada kartu keputusan.** |
| Aturan 12 spesifikasi digabung ke aturan 11, sehingga ada 11 aturan, bukan 12 | Aturan 5 (Tidak Relevan) sudah tercakup di definisi. Aturan 5 pada prompt (pengumuman tetap masuk topiknya) berasal dari v3.1. |
| 11 contoh ilustrasi buatan | Setiap kategori punya minimal 2 contoh. Ketiga sentimen dan kedua label keluhan terwakili. Contoh mencakup pasangan kontrasif (info dan keluhan tentang halte dan jalan; pertanyaan informatif dan pertanyaan keluhan), negasi ("nggak mahal"), multi-topik (bus terhalang parkir menjadi Lahan Parkir), Negatif tanpa keluhan, dan Positif dengan keluhan. |
| Format keluaran dan daftar label ditaruh tepat sebelum post | Model kecil lebih patuh pada instruksi yang paling dekat dengan teks masukan. |
| Retry memakai instruksi tambahan | Supaya percobaan kedua benar-benar berbeda dari yang pertama. |
| Validator baru (`src/validasi_anotasi.py`) | Memakai parser JSON standar. Hasilnya ditolak bila: <ul><li>keluaran kosong;</li><li>bukan JSON;</li><li>ada teks tambahan atau code fence;</li><li>bukan objek;</li><li>kuncinya salah atau ada kunci tambahan;</li><li>ada nilai null atau nilai bukan string;</li><li>labelnya di luar daftar.</li></ul>Label tidak pernah diperbaiki diam-diam. |
| `03_annotation.py` mencatat `raw1`, `alasan1`, `raw2`, `alasan2`, dan `status` per post | Kegagalan bisa dianalisis. Ringkasan tahap 3 menambah hitungan alasan gagal. Arsitektur, backend, dan format penyimpanan lain tidak berubah. Fungsi `annotate()` tetap ada untuk `03b_konsistensi.py`. |

### Konflik yang ditemukan dan cara penyelesaiannya

1. **Aturan 12 spesifikasi berbeda dengan prompt v3.1 dan naskah.** Diselesaikan lewat keputusan OPET: Keluhan selalu Tidak untuk post Tidak Relevan.
2. **Definisi Public Transport contoh OPET berbeda dengan Table II naskah.** Contoh OPET memasukkan MRT, LRT, KRL, taksi, integrasi antarmoda, dan aksesibilitas. Table II naskah dan prompt v3.1 hanya memuat bus kota dan BRT, JakLingko atau Mikrotrans, angkot, ojek, dan halte. Prompt mengikuti naskah, karena spesifikasi melarang mengubah definisi tanpa alasan.
   - **OPET perlu memutuskan:** apakah MRT, LRT, KRL, dan taksi masuk Angkutan Umum. Kalau ya, Table II dan suplemen S4 juga harus diubah.
   - Dengan definisi sekarang, post tentang MRT atau KRL kemungkinan diberi Tidak Relevan.
3. **Contoh "Positif dengan keluhan"** (jalan ditambal setelah tiga bulan dilaporkan) adalah kasus campuran yang bisa diperdebatkan. Contoh ini dipertahankan karena spesifikasi meminta contoh yang menunjukkan bahwa sentimen Positif tidak otomatis berarti Keluhan Tidak. Kalau pengujian menunjukkan contoh ini membingungkan model, contoh inilah yang pertama diganti.

## 5. Validasi sebelum dan sesudah revisi

Validasi yang sudah benar-benar dijalankan:
- Validator diuji dengan 16 keluaran buatan (`src/test_validasi_anotasi.py`), dan 16/16 lolos sesuai harapan.
- Kabel retry diuji dengan backend palsu tanpa model: percobaan kedua memakai prompt plus instruksi retry, lalu status dan alasan tercatat.
- Prompt memuat tepat satu `{teks}`, ke-11 contoh JSON lolos validator, dan tidak ada em dash.

Yang belum dijalankan:
- inferensi Qwen;
- uji sepuluh post;
- perbandingan sebelum dan sesudah revisi pada keluaran model.

Setelah sepuluh post tersedia, jalankan `03c_uji_prompt.py` dengan prompt v1 (`--label v1`, memakai `arsip/prompt_anotasi_v1_claude.txt`) dan v2 (`--label v2`). Bandingkan keduanya dengan:

```
python src/03c_uji_prompt.py --banding outputs/eval/uji_prompt_v1.csv outputs/eval/uji_prompt_v2.csv
```

## 6. Berkas yang dibuat atau diubah (di engine/)

- `referensi/prompt_anotasi.txt` (diubah, v2)
- `referensi/prompt_retry_format.txt` (baru)
- `src/validasi_anotasi.py` (baru)
- `src/test_validasi_anotasi.py` (baru)
- `src/03c_uji_prompt.py` (baru)
- `src/03_annotation.py` (diubah: validator, retry dengan instruksi, log audit)
- `referensi/arsip/` berisi versi lama:
  - `03_annotation_sebelum_validator.py`;
  - `prompt_anotasi_v1_claude.txt`;
  - `prompt_anotasi_draf_lama.txt`.
