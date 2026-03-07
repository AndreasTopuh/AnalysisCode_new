# Analisis: Mengapa Model Mendapat Score ~99%?

## Eksperimen Overview

| Item | Detail |
|------|--------|
| **Dataset** | PhiUSIIL Phishing URL Dataset |
| **Jumlah Sampel** | 235,795 |
| **Jumlah Fitur** | 54 fitur numerik (dari 59 kolom total) |
| **Label** | 0 = Phishing (100,945 / 42.8%), 1 = Legitimate (134,850 / 57.2%) |
| **Validasi** | 5-Fold Stratified Cross Validation (file fold terpisah) |
| **Model** | Random Forest, XGBoost (GradientBoosting), SVM |
| **Feature Selection** | Boruta, RFE, Correlation, ContrastFS (masing-masing Top 10) + All Features |

---

## Hasil Eksperimen (F1 Score)

| Feature Set | Random Forest | XGBoost | SVM |
|-------------|:---:|:---:|:---:|
| **Boruta** | 0.9971 | 0.9976 | 0.9944 |
| **RFE** | 0.9957 | 0.9980 | 0.9934 |
| **Correlation** | 0.9756 | 0.9770 | 0.9740 |
| **ContrastFS** | 0.9754 | 0.9769 | 0.9740 |
| **All Features** | 0.9993 | 0.9993 | 0.9981 |

---

## Pertanyaan Utama: Apakah Score 99% Itu Wajar atau Ada Kesalahan?

### **Kesimpulan: Score 99% WAJAR. Tidak ada kesalahan dalam pipeline.**

Berikut analisis mendalam dari berbagai aspek:

---

## 1. Dataset PhiUSIIL Memang Sangat Mudah Dipisahkan (Easily Separable)

Dataset PhiUSIIL bukan hanya menggunakan fitur URL saja, tetapi juga **fitur konten halaman web** (content-based features). Artinya, setiap URL sudah dikunjungi dan informasi halaman diekstrak. Hal ini membuat dua kelas (phishing vs legitimate) **sangat berbeda secara fundamental**.

### Perbandingan Karakteristik Dua Kelas

| Fitur | Phishing (label=0) | Legitimate (label=1) | Rasio |
|-------|:---:|:---:|:---:|
| **Mean LineOfCode** | 66 | 1,947 | 29.5x |
| **Mean NoOfJS** | 0.9 | 17.7 | 19.7x |
| **Mean NoOfExternalRef** | 1.1 | 85.3 | 77.5x |
| **HasSocialNet = 1** | 511 (0.5%) | 107,146 (79.5%) | 158x |
| **HasCopyrightInfo = 1** | 5,788 (5.7%) | 108,991 (80.8%) | 14x |
| **HasDescription = 1** | 4,458 (4.4%) | 99,335 (73.7%) | 17x |

**Insight**: Halaman phishing cenderung sangat sederhana (sedikit kode, sedikit JavaScript, hampir tidak ada referensi eksternal), sedangkan halaman legitimate cenderung kompleks dengan banyak resource. Perbedaan ini **sangat ekstrem** — bukan hanya sedikit berbeda, tapi berbeda puluhan hingga ratusan kali lipat.

### Bukti: Classifier Trivial Sudah Sangat Akurat

Bahkan **tanpa machine learning**, aturan sederhana sudah bisa mendapat akurasi tinggi:

| Aturan Sederhana | Akurasi |
|-----------------|:---:|
| `HasSocialNet == label` | **88.03%** |
| `HasCopyrightInfo == label` | **86.58%** |
| `LineOfCode >= 100` → label=1 | **90.04%** |
| `LineOfCode >= 200` → label=1 | **95.48%** |
| Majority vote (HasSocialNet + HasCopyrightInfo + HasDescription) | **91.32%** |

Jika aturan sesederhana `LineOfCode >= 200` saja sudah dapat 95.5%, maka model ensemble seperti Random Forest dengan 10 fitur mendapat 99%+ adalah hal yang **sangat masuk akal**.

---

## 2. Tidak Ada Data Leakage

### Analisis Duplikasi Data

| Pemeriksaan | Hasil | Status |
|------------|-------|:---:|
| URL duplikat | 425 dari 235,795 (0.18%) | ✅ Aman |
| Feature vector duplikat | 1,721 dari 235,795 (0.73%) | ✅ Aman |
| Duplikat dengan label konflik | 0 | ✅ Aman |
| Domain yang mendominasi | Top 100 domain = 3.3% data | ✅ Aman |

### Struktur Fold

- Fold files sudah di-split terpisah (5 file CSV)
- Kolom `URL`, `Domain`, `FILENAME`, `Title`, `TLD` sudah **di-drop** dari fold files
- Tidak ada cara bagi model untuk "mengintip" informasi identitas URL
- Setiap fold memiliki ~47,159 sampel dengan distribusi label yang seimbang (stratified)

**Kesimpulan**: Tidak ada data leakage antara training dan testing set.

---

## 3. Korelasi Fitur dengan Label

### Top 20 Fitur Paling Berkorelasi dengan Label

| Rank | Fitur | Korelasi (abs) | Tipe |
|:---:|-------|:---:|:---:|
| 1 | HasSocialNet | 0.7843 | Binary |
| 2 | HasCopyrightInfo | 0.7434 | Binary |
| 3 | HasDescription | 0.6902 | Binary |
| 4 | DomainTitleMatchScore | 0.5849 | Continuous |
| 5 | HasSubmitButton | 0.5786 | Binary |
| 6 | IsResponsive | 0.5486 | Binary |
| 7 | URLTitleMatchScore | 0.5394 | Continuous |
| 8 | SpacialCharRatioInURL | 0.5335 | Continuous |
| 9 | HasHiddenFields | 0.5077 | Binary |
| 10 | HasFavicon | 0.4937 | Binary |
| 11 | URLCharProb | 0.4697 | Continuous |
| 12 | CharContinuationRate | 0.4677 | Continuous |
| 13 | HasTitle | 0.4597 | Binary |
| 14 | DegitRatioInURL | 0.4320 | Continuous |
| 15 | Robots | 0.3926 | Binary |
| 16 | NoOfJS | 0.3735 | Continuous |
| 17 | LetterRatioInURL | 0.3678 | Continuous |
| 18 | Pay | 0.3597 | Binary |
| 19 | NoOfOtherSpecialCharsInURL | 0.3589 | Continuous |
| 20 | NoOfSelfRef | 0.3162 | Continuous |

Banyak fitur memiliki korelasi **sangat tinggi** (>0.5) dengan label. Ketika model menggabungkan informasi dari 10+ fitur berkorelasi tinggi, akurasi 99% adalah hasil yang diharapkan.

---

## 4. Mengapa Correlation/ContrastFS Mendapat Score Lebih Rendah (~97%)?

### Analisis Tipe Fitur per Metode

| Metode | Binary Features | Continuous Features | F1 Score (rata-rata) |
|--------|:---:|:---:|:---:|
| **Boruta** | 2/10 | 8/10 | ~0.9964 |
| **RFE** | 0/10 | 10/10 | ~0.9957 |
| **Correlation** | 6/10 | 4/10 | ~0.9755 |
| **ContrastFS** | 6/10 | 4/10 | ~0.9754 |

**Penyebab utama**:
- Correlation dan ContrastFS memilih banyak **fitur binary** (HasCopyrightInfo, HasDescription, HasHiddenFields, HasFavicon, HasSubmitButton, IsResponsive)
- Fitur binary hanya bisa membagi data menjadi 2 kelompok per fitur → kapasitas diskriminatif terbatas
- Boruta dan RFE memilih lebih banyak **fitur continuous** (NoOfExternalRef, NoOfJS, NoOfImage, dll.) yang memiliki range nilai luas → lebih informatif untuk tree-based models

### Tambahan: Correlation/ContrastFS memilih fitur yang hampir identik
- 9 dari 10 fitur **sama persis** antara Correlation dan ContrastFS
- Hasilnya pun hampir identik (perbedaan < 0.001)

---

## 5. Perbandingan Hasil OLD vs NEW

Pada versi NEW, beberapa fitur diganti karena alasan tertentu (misalnya `LineOfCode`, `HasSocialNet`, `URLSimilarityIndex`, `IsHTTPS` dihapus dan diganti fitur lain).

### Fitur yang Diganti

| Metode | Fitur Dihapus | Fitur Pengganti |
|--------|--------------|-----------------|
| Boruta | `LineOfCode` | `DomainTitleMatchScore` |
| Boruta | `HasSocialNet` | `NoOfOtherSpecialCharsInURL` |
| RFE | `LineOfCode` | `NoOfLettersInURL` |
| Correlation | `HasSocialNet` | `URLCharProb` |
| ContrastFS | `HasSocialNet` | `URLCharProb` |

### Dampak Perubahan (F1 Score: NEW - OLD)

| Feature Set | Random Forest | XGBoost | SVM |
|-------------|:---:|:---:|:---:|
| **Boruta** | +0.0007 | +0.0000 | +0.0031 |
| **RFE** | -0.0022 | -0.0009 | -0.0035 |
| **Correlation** | -0.0060 | -0.0060 | -0.0066 |
| **ContrastFS** | -0.0063 | -0.0060 | -0.0066 |
| **All Features** | 0.0000 | 0.0000 | 0.0000 |

**Analisis**:
- **All Features** tidak berubah sama sekali → pipeline konsisten
- **Boruta** sedikit membaik karena pengganti yang dipilih cukup baik
- **RFE** turun sedikit (0.1–0.3%) karena `LineOfCode` (korelasi tinggi) diganti `NoOfLettersInURL`
- **Correlation/ContrastFS** turun paling banyak (~0.6%) karena `HasSocialNet` (korelasi **0.78** — fitur paling diskriminatif di seluruh dataset) diganti `URLCharProb` (korelasi 0.47)

---

## 6. Catatan Penting (Bukan Kesalahan, tapi Keterbatasan)

### a. Content-Based Features = Butuh Akses ke Halaman
Fitur seperti `NoOfJS`, `HasCopyrightInfo`, `LineOfCode`, dll. mengharuskan **mengunjungi URL terlebih dahulu** untuk mengekstrak informasi halaman. Dalam deployment real-time, ini:
- Memperlambat proses prediksi
- Berisiko mengekspos sistem ke konten berbahaya
- Membutuhkan rendering halaman web

### b. XGBoost Menggunakan GradientBoosting
Notebook menggunakan `GradientBoostingClassifier` dari sklearn sebagai pengganti XGBoost karena masalah OpenMP runtime. Hasilnya tetap valid tetapi akan sedikit berbeda dari library XGBoost asli (biasanya XGBoost asli sedikit lebih cepat dan akurat).

### c. Score 99% Konsisten dengan Literatur
Paper asli PhiUSIIL dan reproduksi lainnya juga melaporkan akurasi 99%+. Ini bukan anomali — ini memang karakteristik dataset.

---

## 7. Ringkasan Akhir

| Aspek | Status | Penjelasan |
|-------|:---:|-----------|
| **Score 99% wajar?** | ✅ Ya | Dataset memang sangat mudah dipisahkan |
| **Data leakage?** | ✅ Tidak ada | Duplikasi minimal, fold terpisah dengan baik |
| **Pipeline benar?** | ✅ Ya | Scaling per fold, clone model per fold, stratified split |
| **Fitur redundan?** | ⚠️ Perlu perhatian | Correlation & ContrastFS memilih fitur yang hampir sama |
| **Penggantian fitur OK?** | ⚠️ Sedikit menurun | Pengganti tidak se-diskriminatif aslinya, terutama untuk Correlation/ContrastFS |
| **Deployment realistis?** | ⚠️ Terbatas | Content-based features butuh akses halaman |

### Kesimpulan Final

**Score ~99% pada dataset PhiUSIIL adalah hasil yang wajar dan tidak mengindikasikan adanya kesalahan.** Tingginya score disebabkan oleh perbedaan yang sangat ekstrem antara halaman phishing (sederhana) dan halaman legitimate (kompleks) pada fitur-fitur content-based. Bahkan aturan satu fitur sederhana saja sudah bisa mencapai 88–95%, sehingga model ML dengan 10+ fitur mendapat 99% adalah konsekuensi logisnya.

---

*Analisis ini dibuat berdasarkan hasil eksperimen pada notebook `3_Feature_Selection_Model_Training_5FoldCV copy.ipynb` menggunakan data dari folder `FULL_PIPELINE/`.*
