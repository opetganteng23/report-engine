# Daftar angka untuk uji ulang (Paper 2 v43)

Naskah: Draft_Paper2_ICDSG2026_v43.docx. Suplemen: Supplementary_Paper2_search_terms_v43.xlsx.
Semua "--" di bawah perlu diisi setelah data tanpa 1.455 post ekspansi diuji ulang. Lokasi memakai judul bab dan subbab di naskah.
Folder sumber = Hasil_Penelitian (zip).

## Angka bantu yang sudah terbukti dari data lama (cek ulang setelah run baru)
- Data bersih lama 19.016 = 1.410 (putaran 1) + 16.151 (putaran 2 bulanan) + 1.455 (ekspansi). Tanpa ekspansi: 17.561. Sumber: 01_Dataset/data_bersih_19016_tweet.xlsx, kolom `query`.
- Post ekspansi di sampel LLM 10.000: 776, sisa 9.224. Sumber: 01_Dataset/dataset_final_10000_tweet.xlsx, kolom `seed_query`.
- Post ekspansi di 1.000 data uji: 52, sisa 948. Sumber: 07_Evaluasi_Akurasi/label_manual_1000_sampel.xlsx dicocokkan ke data bersih (lihat Penjelasan_alur_metode.md bagian 5b).
- Angka mentah 26.771, 218 kosong, 7.358 akun resmi/media, dan 179 duplikat ikut berubah karena memuat post ekspansi; berkas mentah tidak ada di zip, jadi harus dihitung ulang dari skrip pembersih di komputer Anda (pembersih.py, seleksi.py).

## 1. Abstract
| No | Isi | Sumber |
|---|---|---|
| A1 | Jumlah data latih ("trained on -- posts") | ukuran data latih baru |
| A2 | Jumlah post uji berlabel manual | ukuran data uji baru |
| A3 | Nama metode dengan macro-F1 tertinggi dan tiga nilainya (kategori, sentimen, keluhan) | Tabel III |
| A4 | Persen kualitas label LLM yang dipertahankan (kategori, sentimen, keluhan) = macro-F1 metode terbaik / macro-F1 label LLM | Tabel III |
| A5 | Kesimpulan SAFE (satu klausa) | Tabel V |

## 2. Proposed Method
| No | Lokasi | Isi | Sumber |
|---|---|---|---|
| M1 | Fig. 1, kotak 1 | jumlah post mentah | skrip crawl/pembersih |
| M2 | Fig. 1, kotak 2 | jumlah post bersih | data bersih baru |
| M3 | Fig. 1, kotak 3 | jumlah post berlabel LLM | sampel LLM baru |
| M4 | Fig. 1, kotak 4 | Train dan Test | data latih dan uji baru |
| M5 | A. Data Acquisition, kalimat terakhir | jumlah post mentah ("yielded -- posts") | sama dengan M1 |
| M6 | B. Data Preprocessing | post kosong, post akun resmi/media, duplikat | skrip pembersih |
| M7 | B. Data Preprocessing, kalimat terakhir | total post bersih, putaran 1, putaran 2 | data bersih, kolom `query` |
| M8 | C. Data Annotation | ukuran sampel LLM dan total post bersih | dataset_final (versi baru) |
| M9 | D. Data Splitting, paragraf 1 | ukuran data uji, dua ukuran sampel (dulu 385 dan 615), n dan margin of error (z x akar(0,25/n) x 100) | label_manual_1000_sampel.xlsx (versi baru) |
| M10 | D. Data Splitting, paragraf 2 | jumlah label manual per kelas (4 kategori, 3 sentimen, 2 keluhan), jumlah post uji yang juga ada di sampel LLM, ukuran data latih, persen Parking, positif, keluhan di data latih | label manual baru; kode_tfidf_umum.py (pembuangan tumpang tindih) |
| M11 | F. Classification Model Training, paragraf IndoBERT | jumlah post validasi (10%) dan post latih IndoBERT | 06_Metode_5_IndoBERT/kode_indobert.py |
| M12 | G. Model Inference | jumlah post uji | ukuran data uji |
| M13 | H. Model Evaluation | jumlah post relevan dan post Not Relevant di data uji (penyebut misrouting dan false dispatch) | label manual baru |

## 3. Results
| No | Lokasi | Isi | Sumber |
|---|---|---|---|
| R1 | A. Overall Performance | metode terbaik per tugas dan macro-F1-nya, serta macro-F1 label LLM | Tabel III |
| R2 | Fig. 2 (gambar placeholder) dan judulnya (n = --) | grafik macro-F1 empat metode | 07_Evaluasi_Akurasi (skrip grafik) |
| R3 | Tabel III (n = --) | akurasi dan macro-F1 per metode per tugas, plus baris LLM labels (ref.); tebalkan nilai terbaik | label_semua_metode + label_manual |
| R4 | Paragraf Tabel IV | kappa label LLM (tiga tugas) | sama |
| R5 | Tabel IV | kappa dan interval Wilson 95% akurasi per metode | sama |
| R6 | B. Per-Class Results | jumlah post uji Parking dan positif, serta F1 Parking dan F1 positif untuk empat metode | sama |
| R7 | Fig. 3 (placeholder) dan judulnya | F1 per kelas | sama |
| R8 | C. Error Analysis, paragraf 1 | jumlah post Not Relevant, false dispatch terendah dan tertinggi beserta nama metodenya | sama |
| R9 | C. Error Analysis, sentimen | jumlah post netral yang diprediksi negatif per metode | confusion matrix sentimen |
| R10 | C. Error Analysis, keluhan | recall dan presisi keluhan per metode | confusion matrix keluhan |
| R11 | Fig. 4 (placeholder) | confusion matrix keluhan | sama |

## 4. Discussion, Limitations, Conclusion
| No | Lokasi | Isi |
|---|---|---|
| D1 | Discussion paragraf 1 | metode terbaik per tugas; persen macro-F1 LLM yang dipertahankan metode lokal terbaik (tiga tugas) |
| D2 | Discussion paragraf 2 | metode terkuat pada safe, accurate, fair; desain SAFE yang disarankan |
| D3 | Tabel V | misrouting, false dispatch, rata-rata macro-F1, F1 kelas terendah per metode (kolom Explainable tetap) |
| D4 | Limitations | jumlah hashtag gabungan dan total hashtag; jumlah post uji Parking |
| D5 | Conclusion | jumlah post uji; metode paling akurat dan persen yang dipertahankan; kesimpulan SAFE |

Catatan: setelah angka ada, kalimat netral di Results dan Discussion bisa dikembalikan menjadi kalimat temuan (mis. metode mana yang unggul dan seberapa jauh).

## 5. Suplemen (Supplementary_Paper2_search_terms_v43.xlsx)
| Sheet | Isi yang diisi |
|---|---|
| S1 Seed words round 1 | kolom post di data akhir |
| S2 Seed words | post putaran 2, putaran 1, total |
| S3 Method details | sampel uji (ukuran, N, margin of error), epoch terbaik IndoBERT, perilaku latih dan biaya |

Sheet ekspansi lama (S3 sampai S5) sudah dihapus. Kalimat kalibrasi prompt (71% ke 90%) dibiarkan karena berasal dari 100 post kalibrasi terpisah, bukan dari data ekspansi; mohon dicek bila kalibrasi juga diulang.

## 6. Tambahan v45: pelabelan dengan Qwen2.5 1.5B
Naskah: Draft_Paper2_ICDSG2026_v45.docx (bagian C. Data Annotation, D. Data Splitting, Limitations). Suplemen: Supplementary_Paper2_search_terms_v45.xlsx.
| No | Lokasi | Isi | Sumber |
|---|---|---|---|
| Q1 | C. Data Annotation | jumlah post yang dianotasi (semua post hasil preprocessing) | data bersih baru |
| Q2 | C. Data Annotation, kalimat terakhir | jumlah post uji dan jumlah anotator manusia | lembar label manual baru |
| Q3 | Limitations | jumlah anotator test set dan nilai kesepakatan antaranotator (mis. kappa) | lembar label manual |
| Q4 | Suplemen S3 | waktu anotasi Qwen dan perangkatnya | log run Qwen |
| Q5 | Tabel II dan suplemen S4 | pastikan definisi kategori sama dengan prompt Qwen yang dipakai | kode/prompt pelabelan Qwen |
| Q6 | Gambar prompt (Gambar 2 di teks Anda) | belum dimasukkan; kirim template prompt Qwen bila ingin ditampilkan | kode pelabelan Qwen |
Catatan: label sentimen dan keluhan tetap ditulis sebagai keluaran anotasi. Bila prompt Qwen hanya menghasilkan kategori, bagian ini perlu diubah.
