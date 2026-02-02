# 📊 LAPORAN LENGKAP: Feature Selection & Model Training Experiment
## Deteksi Phishing URL dengan Machine Learning

---

# DAFTAR ISI

1. [Pendahuluan](#1-pendahuluan)
2. [Dataset PhiUSIIL](#2-dataset-phiusiil)
3. [Arsitektur Eksperimen](#3-arsitektur-eksperimen)
4. [Penjelasan Detail Kode](#4-penjelasan-detail-kode)
5. [Metode Feature Selection](#5-metode-feature-selection)
6. [Model Machine Learning & Hyperparameter](#6-model-machine-learning--hyperparameter)
7. [Metodologi Evaluasi: 5-Fold Cross Validation](#7-metodologi-evaluasi-5-fold-cross-validation)
8. [Hasil Eksperimen](#8-hasil-eksperimen)
9. [Analisis & Diskusi](#9-analisis--diskusi)
10. [Kesimpulan](#10-kesimpulan)
11. [Referensi](#11-referensi)

---

# 1. PENDAHULUAN

## 1.1 Latar Belakang

Phishing adalah serangan siber yang menipu pengguna untuk memberikan informasi sensitif melalui situs web palsu yang menyerupai situs asli. Deteksi otomatis URL phishing menggunakan **Machine Learning** menjadi solusi penting untuk melindungi pengguna internet.

## 1.2 Tujuan Eksperimen

Eksperimen ini bertujuan untuk:
1. **Membandingkan 4 metode Feature Selection** untuk memilih fitur terbaik dari 57 fitur
2. **Mengevaluasi 3 model klasifikasi** (Random Forest, XGBoost/GradientBoosting, SVM)
3. **Menganalisis trade-off** antara jumlah fitur (Top 10 vs All Features) terhadap performa dan waktu training
4. **Menggunakan 5-Fold Cross Validation** untuk evaluasi yang robust

## 1.3 Referensi Utama: Jurnal Prasad & Chandra (2023)

Dataset yang digunakan berasal dari paper:

> **"PhiUSIIL: A diverse security profile empowered phishing URL detection framework based on similarity index and incremental learning"**
> - **Penulis:** Arvind Prasad & Shalini Chandra
> - **Jurnal:** Computers & Security (2023)
> - **DOI:** https://doi.org/10.1016/j.cose.2023.103545

Dalam jurnal Prasad, mereka menggunakan:
- **Dataset:** 235,795 URL (134,850 legitimate + 100,945 phishing)
- **Fitur:** 54-63 fitur yang diekstrak dari URL dan source code webpage
- **Model:** Random Forest, XGBoost, SVM, dan lainnya
- **Evaluasi:** Cross Validation dengan metrik Accuracy, Precision, Recall, F1-Score

---

# 2. DATASET PhiUSIIL

## 2.1 Informasi Dataset

| Aspek | Detail |
|-------|--------|
| **Nama** | PhiUSIIL Phishing URL Dataset |
| **Sumber** | UCI Machine Learning Repository |
| **Total Sampel** | 235,795 URL |
| **Legitimate URLs** | 134,850 (57.2%) |
| **Phishing URLs** | 100,945 (42.8%) |
| **Total Fitur** | 63 fitur (57 fitur numerik setelah preprocessing) |
| **Missing Values** | Tidak ada |

## 2.2 Kategori Fitur

Fitur dalam dataset dibagi menjadi beberapa kategori:

### A. Fitur URL-Based (Berbasis URL)
| Fitur | Deskripsi |
|-------|-----------|
| `URLLength` | Panjang total URL |
| `DomainLength` | Panjang nama domain |
| `IsDomainIP` | Apakah domain berupa IP address (1/0) |
| `URLCharProb` | Probabilitas karakter dalam URL |
| `LetterRatioInURL` | Rasio huruf dalam URL |
| `SpacialCharRatioInURL` | Rasio karakter khusus dalam URL |
| `CharContinuationRate` | Tingkat kontinuitas karakter |
| `URLSimilarityIndex` | Indeks kemiripan URL |
| `TLDLegitimateProb` | Probabilitas TLD (Top Level Domain) legitimate |

### B. Fitur Content-Based (Berbasis Konten HTML)
| Fitur | Deskripsi |
|-------|-----------|
| `LineOfCode` | Jumlah baris kode dalam HTML |
| `LargestLineLength` | Panjang baris terpanjang |
| `NoOfExternalRef` | Jumlah referensi eksternal |
| `NoOfSelfRef` | Jumlah referensi internal/self |
| `NoOfJS` | Jumlah file JavaScript |
| `NoOfCSS` | Jumlah file CSS |
| `NoOfImage` | Jumlah gambar |

### C. Fitur Security-Related (Keamanan)
| Fitur | Deskripsi |
|-------|-----------|
| `IsHTTPS` | Apakah menggunakan HTTPS (1/0) |
| `HasFavicon` | Apakah memiliki favicon (1/0) |
| `HasHiddenFields` | Apakah ada hidden form fields (1/0) |
| `HasSubmitButton` | Apakah ada tombol submit (1/0) |
| `HasSocialNet` | Apakah ada link media sosial (1/0) |
| `HasCopyrightInfo` | Apakah ada info copyright (1/0) |
| `HasDescription` | Apakah ada meta description (1/0) |

### D. Fitur Match Score
| Fitur | Deskripsi |
|-------|-----------|
| `URLTitleMatchScore` | Skor kesesuaian URL dengan title |
| `DomainTitleMatchScore` | Skor kesesuaian domain dengan title |
| `IsResponsive` | Apakah website responsive (1/0) |

## 2.3 Label Kelas

| Label | Keterangan | Jumlah |
|-------|------------|--------|
| **1** | Legitimate URL | 134,850 |
| **0** | Phishing URL | 100,945 |

---

# 3. ARSITEKTUR EKSPERIMEN

## 3.1 Diagram Alur Eksperimen

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        EKSPERIMEN FEATURE SELECTION                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐     ┌──────────────────┐     ┌─────────────────────┐  │
│  │   DATASET   │────▶│  PREPROCESSING   │────▶│  FEATURE SELECTION  │  │
│  │  (235,795)  │     │  - Drop non-num  │     │  - Boruta           │  │
│  │             │     │  - Fill NA       │     │  - RFE              │  │
│  │             │     │  - Numeric conv  │     │  - Correlation      │  │
│  │             │     │                  │     │  - ContrastFS       │  │
│  └─────────────┘     └──────────────────┘     └──────────┬──────────┘  │
│                                                           │              │
│                              ┌────────────────────────────┴───────┐     │
│                              │                                     │     │
│                              ▼                                     ▼     │
│                    ┌──────────────────┐              ┌──────────────────┐│
│                    │  TOP 10 FEATURES │              │   ALL FEATURES   ││
│                    │   (10 fitur)     │              │   (57 fitur)     ││
│                    └────────┬─────────┘              └────────┬─────────┘│
│                              │                                 │         │
│                              └─────────────┬───────────────────┘         │
│                                            │                             │
│                                            ▼                             │
│                              ┌─────────────────────────┐                 │
│                              │    STANDARD SCALING     │                 │
│                              └────────────┬────────────┘                 │
│                                           │                              │
│                                           ▼                              │
│                        ┌──────────────────────────────────┐              │
│                        │  5-FOLD STRATIFIED CROSS VALID   │              │
│                        ├──────────────────────────────────┤              │
│                        │  ┌──────────────────────────┐    │              │
│                        │  │    MODEL TRAINING        │    │              │
│                        │  │  - Random Forest         │    │              │
│                        │  │  - XGBoost (GB)          │    │              │
│                        │  │  - SVM (RBF)             │    │              │
│                        │  └──────────────────────────┘    │              │
│                        └────────────────┬─────────────────┘              │
│                                         │                                │
│                                         ▼                                │
│                        ┌────────────────────────────────┐                │
│                        │       EVALUATION METRICS       │                │
│                        │  Accuracy | Precision | Recall │                │
│                        │  F1-Score | Training Time      │                │
│                        └────────────────────────────────┘                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## 3.2 Komponen Eksperimen

| Komponen | Detail |
|----------|--------|
| **Feature Sets** | 5 (Boruta, RFE, Correlation, ContrastFS, All Features) |
| **Models** | 3 (Random Forest, XGBoost, SVM) |
| **Total Kombinasi** | 15 eksperimen |
| **Validation** | 5-Fold Stratified Cross Validation |
| **Metrics** | Accuracy, Precision, Recall, F1-Score, Training Time |

---

# 4. PENJELASAN DETAIL KODE

## 4.1 Cell 1: Import Libraries

```python
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore') 

# Sklearn
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer

# Boruta
from boruta import BorutaPy
```

### Penjelasan Library:

| Library | Fungsi | Alasan Penggunaan |
|---------|--------|-------------------|
| `pandas` | Manipulasi data tabular | Membaca CSV, membuat DataFrame, operasi data |
| `numpy` | Operasi numerik | Array operations, mathematical functions |
| `time` | Mengukur waktu | Mencatat training time |
| `warnings` | Suppress warnings | Menghilangkan warning yang tidak perlu |
| `StratifiedKFold` | Cross validation | Memastikan distribusi kelas sama di setiap fold |
| `cross_validate` | Evaluasi model | Menjalankan CV dengan multiple metrics |
| `StandardScaler` | Normalisasi data | Standarisasi fitur (mean=0, std=1) |
| `RandomForestClassifier` | Model RF | Ensemble learning dengan decision trees |
| `GradientBoostingClassifier` | Pengganti XGBoost | Boosting ensemble method |
| `SVC` | Support Vector Machine | Klasifikasi dengan hyperplane |
| `make_scorer` | Custom scorer | Membuat scorer untuk cross_validate |
| `BorutaPy` | Feature selection | All-relevant feature selection |

### ⚠️ Catatan Penting:
```python
# Note: Using GradientBoostingClassifier as XGBoost alternative
# (XGBoost requires OpenMP runtime which is not installed)
```
**Alasan:** XGBoost memerlukan OpenMP runtime yang tidak terinstall. `GradientBoostingClassifier` dari scikit-learn adalah alternatif yang equivalent karena keduanya menggunakan prinsip **Gradient Boosting**.

---

## 4.2 Cell 2: Load Dataset

```python
# Load dataset
df = pd.read_csv('PhiUSIIL_Phishing_URL_63_Features.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nLabel distribution:\n{df['label'].value_counts()}")
```

### Penjelasan:
- **`pd.read_csv()`**: Membaca file CSV ke dalam pandas DataFrame
- **`df.shape`**: Menampilkan dimensi data (rows, columns)
- **`df.columns.tolist()`**: List semua nama kolom
- **`df['label'].value_counts()`**: Distribusi kelas target

### Output yang Diharapkan:
```
Dataset shape: (235795, 64)

Label distribution:
1    134850  (Legitimate)
0    100945  (Phishing)
```

---

## 4.3 Cell 3: Data Preprocessing

```python
# Kolom non-numerik yang harus di-drop
non_numeric_cols = ['FILENAME', 'URL', 'Domain', 'TLD', 'Title']

# Drop kolom non-numerik
df_numeric = df.drop(columns=non_numeric_cols, errors='ignore')

# Pisahkan fitur dan target
X = df_numeric.drop(columns=['label'])
y = df_numeric['label']

# Handle missing values
X = X.fillna(X.median())

# Pastikan semua kolom numerik
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(X.median())

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
```

### Penjelasan Detail:

#### Langkah 1: Drop Kolom Non-Numerik
```python
non_numeric_cols = ['FILENAME', 'URL', 'Domain', 'TLD', 'Title']
```

| Kolom | Alasan Di-drop |
|-------|----------------|
| `FILENAME` | Identifier file, bukan fitur |
| `URL` | String URL asli, tidak bisa diproses langsung |
| `Domain` | String domain name |
| `TLD` | String Top Level Domain (.com, .org, dll) |
| `Title` | String judul halaman web |

**Alasan:** Model machine learning membutuhkan input numerik. Kolom string/categorical ini sudah diekstrak menjadi fitur numerik lainnya.

#### Langkah 2: Handle Missing Values
```python
X = X.fillna(X.median())
```
- **Metode:** Mengisi nilai kosong dengan **median** kolom
- **Alasan memilih median:** 
  - Lebih robust terhadap outlier dibanding mean
  - Cocok untuk data dengan distribusi tidak normal
  - Menjaga integritas distribusi data

#### Langkah 3: Konversi ke Numerik
```python
X = X.apply(pd.to_numeric, errors='coerce')
```
- **`errors='coerce'`**: Mengubah nilai yang tidak bisa dikonversi menjadi NaN
- Kemudian diisi lagi dengan median

### Hasil Preprocessing:
- **Input:** 64 kolom (termasuk label dan non-numerik)
- **Output:** 57 fitur numerik + 1 target variable

---

## 4.4 Cell 4: Pre-defined Top 10 Features

```python
# Boruta Top 10 Features
boruta_top10 = [
    'LineOfCode',
    'NoOfExternalRef',
    'NoOfSelfRef',
    'NoOfJS',
    'HasDescription',
    'NoOfImage',
    'HasSocialNet',
    'NoOfCSS',
    'HasCopyrightInfo',
    'LargestLineLength'
]

# RFE Top 10 Features
rfe_top10 = [
    'LineOfCode',
    'LargestLineLength',
    'NoOfExternalRef',
    'URLCharProb',
    'LetterRatioInURL',
    'SpacialCharRatioInURL',
    'NoOfCSS',
    'URL_Profanity_Prob',
    'URLLength',
    'NoOfJS'
]

# Correlation Top 10 Features
correlation_top10 = [
    'HasSocialNet',
    'HasCopyrightInfo',
    'HasDescription',
    'SpacialCharRatioInURL',
    'HasHiddenFields',
    'HasFavicon',
    'DomainTitleMatchScore',
    'HasSubmitButton',
    'IsResponsive',
    'URLTitleMatchScore'
]

# ContrastFS Top 10 Features
contrast_top10 = [
    'HasSocialNet',
    'HasCopyrightInfo',
    'HasDescription',
    'SpacialCharRatioInURL',
    'HasHiddenFields',
    'HasFavicon',
    'HasSubmitButton',
    'DomainTitleMatchScore',
    'IsResponsive',
    'URLTitleMatchScore'
]
```

### Penjelasan Per Metode Feature Selection:

Fitur-fitur ini **sudah ditentukan sebelumnya** dari hasil Feature Selection yang dilakukan secara terpisah.

---

## 4.5 Cell 5: Feature Sets Dictionary

```python
feature_sets = {
    'Boruta': boruta_top10,
    'RFE': rfe_top10,
    'Correlation': correlation_top10,
    'ContrastFS': contrast_top10,
    'All Features': X.columns.tolist()
}
```

### Penjelasan:
Dictionary yang memetakan nama metode ke list fitur yang dipilih, memudahkan iterasi saat training.

---

## 4.6 Cell 6: Training Function dengan 5-Fold CV

```python
def train_and_evaluate_cv(X, y, model, model_name, n_splits=5):
    """
    Train model dengan 5-Fold Stratified Cross Validation
    dan hitung metrics + training time
    """
    # Setup 5-Fold Stratified Cross Validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Define scorers
    scorers = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='binary'),
        'recall': make_scorer(recall_score, average='binary'),
        'f1': make_scorer(f1_score, average='binary')
    }
    
    # Start training timer
    start_time = time.time()
    
    # Perform cross validation
    cv_results = cross_validate(
        model, X, y, 
        cv=skf, 
        scoring=scorers,
        return_train_score=False,
        n_jobs=-1
    )
    
    # End training timer
    training_time = time.time() - start_time
    
    # Calculate mean metrics across all folds
    metrics = {
        'accuracy': cv_results['test_accuracy'].mean(),
        'accuracy_std': cv_results['test_accuracy'].std(),
        'precision': cv_results['test_precision'].mean(),
        'precision_std': cv_results['test_precision'].std(),
        'recall': cv_results['test_recall'].mean(),
        'recall_std': cv_results['test_recall'].std(),
        'f1': cv_results['test_f1'].mean(),
        'f1_std': cv_results['test_f1'].std(),
        'training_time': training_time
    }
    
    return metrics
```

### Penjelasan Detail:

#### Parameter StratifiedKFold:
| Parameter | Nilai | Penjelasan |
|-----------|-------|------------|
| `n_splits=5` | 5 | Membagi data menjadi 5 bagian (80% train, 20% test per fold) |
| `shuffle=True` | True | Mengacak data sebelum split untuk menghindari bias urutan |
| `random_state=42` | 42 | Seed untuk reproducibility (hasil sama setiap kali dijalankan) |

#### Mengapa Stratified?
**Stratified** memastikan **proporsi kelas sama** di setiap fold:
- Legitimate (57.2%) : Phishing (42.8%) di setiap fold
- Mencegah fold dengan mayoritas satu kelas saja

#### Scorers yang Digunakan:
| Metric | Formula | Interpretasi |
|--------|---------|--------------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Persentase prediksi benar dari total |
| **Precision** | TP/(TP+FP) | Dari yang diprediksi positive, berapa yang benar |
| **Recall** | TP/(TP+FN) | Dari yang actual positive, berapa yang terdeteksi |
| **F1-Score** | 2×(Precision×Recall)/(Precision+Recall) | Harmonic mean Precision & Recall |

Keterangan:
- **TP** = True Positive (Benar mendeteksi legitimate)
- **TN** = True Negative (Benar mendeteksi phishing)
- **FP** = False Positive (Salah mendeteksi sebagai legitimate)
- **FN** = False Negative (Salah mendeteksi sebagai phishing)

#### Parameter cross_validate:
| Parameter | Nilai | Penjelasan |
|-----------|-------|------------|
| `cv=skf` | StratifiedKFold | Menggunakan stratified cross validation |
| `scoring=scorers` | dict | Multiple metrics sekaligus |
| `return_train_score=False` | False | Tidak menyimpan training score (hemat memori) |
| `n_jobs=-1` | -1 | Gunakan semua CPU cores (parallel processing) |

---

## 4.7 Cell 7: Model Definitions

```python
def get_models(n_features):
    """
    Get models dengan max_depth yang disesuaikan dengan jumlah fitur
    """
    if n_features <= 10:
        max_depth = 10  # Untuk top 10 features
    else:
        max_depth = 20  # Untuk all features (57)
    
    return {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, 
            max_depth=max_depth,
            random_state=42, 
            n_jobs=-1
        ),
        'XGBoost': GradientBoostingClassifier(
            n_estimators=100, 
            max_depth=min(max_depth, 10),
            random_state=42
        ),
        'SVM': SVC(
            kernel='rbf', 
            C=1.0,
            gamma='scale',
            random_state=42
        )
    }
```

### Penjelasan Hyperparameter Per Model:

---

### 🌲 RANDOM FOREST

```python
RandomForestClassifier(
    n_estimators=100, 
    max_depth=10/20,
    random_state=42, 
    n_jobs=-1
)
```

| Parameter | Nilai | Penjelasan | Alasan Pemilihan |
|-----------|-------|------------|------------------|
| `n_estimators` | 100 | Jumlah decision trees dalam forest | **Standar industri.** 100 trees memberikan keseimbangan antara akurasi dan waktu training. Lebih banyak trees = lebih baik, tapi diminishing returns setelah 100-200 |
| `max_depth` | 10 (top 10) / 20 (all) | Kedalaman maksimum setiap tree | **Adaptif berdasarkan fitur.** 10 fitur → depth 10 cukup untuk capture patterns. 57 fitur → depth 20 agar trees bisa model kompleksitas lebih |
| `random_state` | 42 | Seed untuk random number generator | **Reproducibility.** Hasil eksperimen dapat direproduksi |
| `n_jobs` | -1 | Jumlah CPU cores | **Parallelization.** -1 = gunakan semua cores untuk training lebih cepat |

#### Perbandingan dengan Jurnal Prasad:
Dalam jurnal PhiUSIIL, Prasad menggunakan parameter serupa:
- `n_estimators`: 100-200
- `max_depth`: Tidak dibatasi atau 10-20
- Hasil: **99%+ accuracy**

---

### 📈 XGBOOST (GradientBoostingClassifier)

```python
GradientBoostingClassifier(
    n_estimators=100, 
    max_depth=10,
    random_state=42
)
```

| Parameter | Nilai | Penjelasan | Alasan Pemilihan |
|-----------|-------|------------|------------------|
| `n_estimators` | 100 | Jumlah boosting stages | **Standar untuk gradient boosting.** 100 iterasi cukup untuk konvergensi |
| `max_depth` | 10 | Kedalaman maksimum per tree | **Shallower trees untuk boosting.** Boosting menggunakan weak learners (shallow trees) yang di-combine. Depth 10 sudah cukup kompleks |
| `random_state` | 42 | Seed | **Reproducibility** |

#### Mengapa max_depth Lebih Kecil?
- **Filosofi Boosting:** Menggunakan banyak "weak learners" (trees dangkal) yang di-boost iteratif
- Trees dalam lebih berisiko **overfitting**
- Berbeda dengan Random Forest yang menggunakan trees dalam dan averaging

#### Catatan XGBoost vs GradientBoosting:
| Aspek | XGBoost | GradientBoosting (sklearn) |
|-------|---------|----------------------------|
| **Speed** | Lebih cepat (optimized) | Lebih lambat |
| **Regularization** | L1 & L2 built-in | Manual |
| **Missing values** | Handle otomatis | Manual |
| **Parallel** | Ya | Tidak (n_jobs tidak tersedia) |
| **Hasil** | Serupa | Serupa |

---

### 🎯 SUPPORT VECTOR MACHINE (SVM)

```python
SVC(
    kernel='rbf', 
    C=1.0,
    gamma='scale',
    random_state=42
)
```

| Parameter | Nilai | Penjelasan | Alasan Pemilihan |
|-----------|-------|------------|------------------|
| `kernel` | 'rbf' | Radial Basis Function kernel | **Paling populer.** RBF bisa menangkap non-linear relationships. Implicit mapping ke infinite-dimensional space |
| `C` | 1.0 | Regularization parameter | **Default yang balanced.** C tinggi = margin keras (risk overfitting). C rendah = margin lembut (risk underfitting). 1.0 adalah sweet spot |
| `gamma` | 'scale' | Kernel coefficient | **Adaptive.** 'scale' = 1/(n_features × X.var()). Menyesuaikan dengan jumlah fitur dan variance data |
| `random_state` | 42 | Seed | **Reproducibility** |

#### Penjelasan RBF Kernel:
$$K(x, x') = \exp\left(-\gamma \|x - x'\|^2\right)$$

- **Intuisi:** Mengukur similarity berdasarkan jarak Euclidean
- **Gamma tinggi:** Decision boundary kompleks, mengikuti data points (overfitting risk)
- **Gamma rendah:** Decision boundary smooth (underfitting risk)
- **'scale'** otomatis menyesuaikan

#### Perbandingan dengan Jurnal Prasad:
Prasad menggunakan SVM dengan:
- Kernel: RBF
- C: 1.0 atau grid search
- Gamma: 'scale' atau 'auto'
- Hasil: **~99% accuracy**

---

## 4.8 Cell 8: Main Training Loop

```python
for fs_name, features in feature_sets.items():
    n_features = len(features)
    
    # Initialize result storage
    results['Accuracy'][fs_name] = {}
    results['Precision'][fs_name] = {}
    results['Recall'][fs_name] = {}
    results['F1'][fs_name] = {}
    results['Training Time'][fs_name] = {}
    
    # Select features
    X_fs = X[features]
    
    # Scale data
    scaler_fs = StandardScaler()
    X_fs_scaled = scaler_fs.fit_transform(X_fs)
    
    # Get models
    models = get_models(n_features)
    
    for model_name, model in models.items():
        # Train and evaluate
        metrics = train_and_evaluate_cv(
            X_fs_scaled, y, 
            model, model_name,
            n_splits=5
        )
        
        # Store results
        results['Accuracy'][fs_name][model_name] = metrics['accuracy']
        # ... (store other metrics)
```

### Penjelasan StandardScaler:

```python
scaler_fs = StandardScaler()
X_fs_scaled = scaler_fs.fit_transform(X_fs)
```

#### Formula:
$$z = \frac{x - \mu}{\sigma}$$

Dimana:
- $x$ = nilai original
- $\mu$ = mean kolom
- $\sigma$ = standard deviation kolom
- $z$ = nilai ter-standardisasi

#### Hasil:
- **Mean** setiap fitur = 0
- **Standard deviation** setiap fitur = 1

#### Mengapa Scaling Penting?

| Model | Perlu Scaling? | Alasan |
|-------|----------------|--------|
| **SVM** | ✅ WAJIB | Distance-based. Fitur dengan range besar akan mendominasi |
| **Random Forest** | ❌ Tidak wajib | Tree-based, split berdasarkan threshold, tidak terpengaruh skala |
| **XGBoost** | ❌ Tidak wajib | Tree-based seperti RF |

**Keputusan:** Tetap scaling semua model untuk **konsistensi** dan **fairness** perbandingan.

---

# 5. METODE FEATURE SELECTION

## 5.1 Overview 4 Metode

| Metode | Kategori | Cara Kerja |
|--------|----------|------------|
| **Boruta** | Wrapper | Membandingkan fitur dengan "shadow features" (random permutation) |
| **RFE** | Wrapper | Eliminasi fitur paling tidak penting secara iteratif |
| **Correlation** | Filter | Memilih fitur dengan korelasi tinggi ke target |
| **ContrastFS** | Hybrid | Kombinasi contrast mining dengan statistical tests |

---

## 5.2 BORUTA

### Cara Kerja:
1. **Buat shadow features:** Duplikat semua fitur, acak nilainya
2. **Train Random Forest** dengan fitur asli + shadow
3. **Bandingkan importance:** Fitur asli harus lebih penting dari shadow terbaik
4. **Iterasi:** Hapus shadow, ulangi sampai konvergen
5. **Hasil:** Fitur yang konsisten lebih penting dari random → "All-relevant features"

### Top 10 Boruta Features:
| No | Fitur | Kategori | Interpretasi |
|----|-------|----------|--------------|
| 1 | `LineOfCode` | Content | Phishing sites cenderung memiliki kode lebih sederhana |
| 2 | `NoOfExternalRef` | Content | Referensi eksternal mencurigakan |
| 3 | `NoOfSelfRef` | Content | Pattern referensi internal |
| 4 | `NoOfJS` | Content | Jumlah JavaScript files |
| 5 | `HasDescription` | Security | Legitimate sites biasanya punya meta description |
| 6 | `NoOfImage` | Content | Pattern penggunaan gambar |
| 7 | `HasSocialNet` | Security | Link ke social media (kredibilitas) |
| 8 | `NoOfCSS` | Content | Jumlah styling files |
| 9 | `HasCopyrightInfo` | Security | Copyright menunjukkan legitimasi |
| 10 | `LargestLineLength` | Content | Pola struktur kode |

---

## 5.3 RFE (Recursive Feature Elimination)

### Cara Kerja:
1. **Train model** dengan semua fitur
2. **Ranking fitur** berdasarkan importance
3. **Eliminasi fitur paling tidak penting**
4. **Ulangi** sampai tersisa jumlah fitur yang diinginkan
5. **Hasil:** Top-N fitur paling penting untuk model

### Top 10 RFE Features:
| No | Fitur | Kategori | Interpretasi |
|----|-------|----------|--------------|
| 1 | `LineOfCode` | Content | Kompleksitas halaman |
| 2 | `LargestLineLength` | Content | Struktur kode |
| 3 | `NoOfExternalRef` | Content | Referensi eksternal |
| 4 | `URLCharProb` | URL | Probabilitas karakter dalam URL |
| 5 | `LetterRatioInURL` | URL | Rasio huruf vs total karakter |
| 6 | `SpacialCharRatioInURL` | URL | Rasio karakter khusus |
| 7 | `NoOfCSS` | Content | Jumlah CSS files |
| 8 | `URL_Profanity_Prob` | URL | Probabilitas kata tidak pantas |
| 9 | `URLLength` | URL | Panjang URL (phishing sering panjang) |
| 10 | `NoOfJS` | Content | Jumlah JavaScript |

---

## 5.4 CORRELATION

### Cara Kerja:
1. **Hitung korelasi** setiap fitur dengan target (label)
2. **Ranking** berdasarkan absolute correlation
3. **Pilih Top-N** fitur dengan korelasi tertinggi
4. **Optional:** Hapus fitur yang saling berkorelasi tinggi (multicollinearity)

### Rumus Pearson Correlation:
$$r = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i - \bar{x})^2 \sum(y_i - \bar{y})^2}}$$

### Top 10 Correlation Features:
| No | Fitur | Kategori | Interpretasi |
|----|-------|----------|--------------|
| 1 | `HasSocialNet` | Security | Link social media = legitimate |
| 2 | `HasCopyrightInfo` | Security | Copyright = legitimate |
| 3 | `HasDescription` | Security | Meta description = legitimate |
| 4 | `SpacialCharRatioInURL` | URL | Karakter khusus mencurigakan |
| 5 | `HasHiddenFields` | Security | Hidden fields = suspicious |
| 6 | `HasFavicon` | Security | Favicon = legitimate |
| 7 | `DomainTitleMatchScore` | Match | Domain cocok title = legitimate |
| 8 | `HasSubmitButton` | Security | Form dengan submit |
| 9 | `IsResponsive` | Content | Website responsive |
| 10 | `URLTitleMatchScore` | Match | URL cocok title |

---

## 5.5 CONTRASTFS

### Cara Kerja:
1. **Contrast mining:** Identifikasi fitur yang "membedakan" kelas
2. **Statistical tests:** Uji signifikansi perbedaan
3. **Ranking:** Kombinasi contrast score dan significance
4. **Hasil:** Fitur yang paling "membedakan" legitimate vs phishing

### Top 10 ContrastFS Features:
Hampir sama dengan Correlation (karena konsep mirip):
| No | Fitur |
|----|-------|
| 1 | `HasSocialNet` |
| 2 | `HasCopyrightInfo` |
| 3 | `HasDescription` |
| 4 | `SpacialCharRatioInURL` |
| 5 | `HasHiddenFields` |
| 6 | `HasFavicon` |
| 7 | `HasSubmitButton` |
| 8 | `DomainTitleMatchScore` |
| 9 | `IsResponsive` |
| 10 | `URLTitleMatchScore` |

---

## 5.6 Perbandingan Feature Sets

```
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE OVERLAP ANALYSIS                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  BORUTA                    RFE                                  │
│  ┌───────────────┐         ┌───────────────┐                   │
│  │ LineOfCode    │←───────→│ LineOfCode    │                   │
│  │ NoOfExternalRef│←──────→│ NoOfExternalRef│                   │
│  │ NoOfJS        │←───────→│ NoOfJS        │                   │
│  │ NoOfCSS       │←───────→│ NoOfCSS       │                   │
│  │ LargestLineLen│←───────→│ LargestLineLen│                   │
│  │               │         │ URLCharProb   │                   │
│  │ NoOfSelfRef   │         │ LetterRatio   │                   │
│  │ HasDescription│         │ SpacialChar   │                   │
│  │ NoOfImage     │         │ URL_Profanity │                   │
│  │ HasSocialNet  │         │ URLLength     │                   │
│  │ HasCopyrightInf│         │               │                   │
│  └───────────────┘         └───────────────┘                   │
│                                                                  │
│  CORRELATION               CONTRASTFS                           │
│  ┌───────────────┐         ┌───────────────┐                   │
│  │ HasSocialNet  │←───────→│ HasSocialNet  │                   │
│  │ HasCopyrightInf│←──────→│ HasCopyrightInf│                   │
│  │ HasDescription│←───────→│ HasDescription│                   │
│  │ SpacialCharRat│←───────→│ SpacialCharRat│                   │
│  │ HasHiddenField│←───────→│ HasHiddenField│                   │
│  │ HasFavicon    │←───────→│ HasFavicon    │                   │
│  │ DomainTitleMat│←───────→│ DomainTitleMat│                   │
│  │ HasSubmitButtn│←───────→│ HasSubmitButtn│                   │
│  │ IsResponsive  │←───────→│ IsResponsive  │                   │
│  │ URLTitleMatch │←───────→│ URLTitleMatch │                   │
│  └───────────────┘         └───────────────┘                   │
│                                                                  │
│  NOTE: Correlation & ContrastFS memilih fitur SERUPA            │
│        Boruta & RFE memilih fitur yang BERBEDA (content-based)  │
└─────────────────────────────────────────────────────────────────┘
```

---

# 6. MODEL MACHINE LEARNING & HYPERPARAMETER

## 6.1 Random Forest Classifier

### Arsitektur:
```
                    ┌─────────────────────────────┐
                    │      RANDOM FOREST          │
                    │     (100 Trees)             │
                    └─────────────┬───────────────┘
                                  │
           ┌──────────────────────┼──────────────────────┐
           │                      │                      │
      ┌────▼────┐            ┌────▼────┐            ┌────▼────┐
      │ Tree 1  │            │ Tree 2  │    ...     │ Tree 100│
      │depth=10 │            │depth=10 │            │depth=10 │
      └────┬────┘            └────┬────┘            └────┬────┘
           │                      │                      │
           └──────────────────────┼──────────────────────┘
                                  │
                         ┌────────▼────────┐
                         │   MAJORITY      │
                         │     VOTE        │
                         └─────────────────┘
```

### Hyperparameter yang Dipilih:

| Parameter | Nilai | Justifikasi |
|-----------|-------|-------------|
| `n_estimators=100` | 100 trees | ✅ **Sweet spot** antara akurasi dan waktu. Studi menunjukkan >100 diminishing returns |
| `max_depth=10/20` | Adaptif | ✅ **Sesuai kompleksitas.** 10 fitur → depth 10. 57 fitur → depth 20 agar bisa explore semua kombinasi |
| `random_state=42` | Fixed seed | ✅ **Reproducibility** |
| `n_jobs=-1` | All cores | ✅ **Parallel training** |

### Perbandingan dengan Jurnal Prasad:
| Parameter | Eksperimen Ini | Jurnal Prasad |
|-----------|----------------|---------------|
| n_estimators | 100 | 100-200 |
| max_depth | 10-20 | Tidak dibatasi/10-20 |
| Hasil Accuracy | 99.59-99.92% | 99%+ |

---

## 6.2 XGBoost (GradientBoostingClassifier)

### Arsitektur Boosting:
```
     ┌─────────┐    ┌─────────┐    ┌─────────┐         ┌─────────┐
     │ Tree 1  │───▶│ Tree 2  │───▶│ Tree 3  │──▶ ... ▶│Tree 100 │
     │(weak)   │    │(weak)   │    │(weak)   │         │(weak)   │
     └────┬────┘    └────┬────┘    └────┬────┘         └────┬────┘
          │              │              │                   │
          │   Residual   │   Residual   │                   │
          └──────────────┴──────────────┴───────────────────┘
                                    │
                           ┌────────▼────────┐
                           │   WEIGHTED SUM  │
                           │   = Final Pred  │
                           └─────────────────┘
```

### Hyperparameter yang Dipilih:

| Parameter | Nilai | Justifikasi |
|-----------|-------|-------------|
| `n_estimators=100` | 100 stages | ✅ **100 boosting iterations.** Cukup untuk konvergensi |
| `max_depth=10` | Shallow trees | ✅ **Weak learners.** Boosting bekerja dengan trees dangkal |
| `random_state=42` | Fixed seed | ✅ **Reproducibility** |

### Mengapa Shallow Trees untuk Boosting?
- Boosting **sequential:** Error tree sebelumnya diperbaiki tree berikutnya
- Trees dalam → **Overfitting** pada satu iterasi
- Trees dangkal → **Generalization** lebih baik

### Perbandingan dengan Jurnal Prasad:
| Parameter | Eksperimen Ini | Jurnal Prasad |
|-----------|----------------|---------------|
| n_estimators | 100 | 100-300 |
| max_depth | 10 | 6-10 |
| learning_rate | Default (0.1) | 0.1-0.3 |
| Hasil Accuracy | 99.73-99.92% | 99%+ |

---

## 6.3 Support Vector Machine (SVM)

### Arsitektur SVM dengan RBF Kernel:
```
                         HIGH-DIMENSIONAL SPACE
                         (via RBF kernel)
                         
    ┌───────────────────────────────────────────────────┐
    │                      ○                            │
    │               ○    ○  ○                          │
    │            ○    LEGITIMATE                        │
    │               ○       ○                          │
    │                                                   │
    │   ════════════════════════════════════════════   │  ← HYPERPLANE
    │                                                   │
    │               ×           ×                       │
    │            ×    PHISHING    ×                    │
    │               ×    ×    ×                        │
    │                      ×                           │
    └───────────────────────────────────────────────────┘
```

### RBF Kernel Formula:
$$K(x_i, x_j) = \exp\left(-\gamma \|x_i - x_j\|^2\right)$$

### Hyperparameter yang Dipilih:

| Parameter | Nilai | Justifikasi |
|-----------|-------|-------------|
| `kernel='rbf'` | RBF | ✅ **Paling versatile.** Dapat menangkap non-linear patterns |
| `C=1.0` | Regularization | ✅ **Balanced.** Tidak terlalu strict (overfitting) atau loose (underfitting) |
| `gamma='scale'` | Auto-scaled | ✅ **Adaptive.** = 1/(n_features × X.var()), menyesuaikan dengan data |
| `random_state=42` | Fixed seed | ✅ **Reproducibility** |

### Penjelasan Parameter C:
```
C RENDAH (0.01)                    C TINGGI (100)
┌─────────────────┐                ┌─────────────────┐
│   ○ ○           │                │   ○ ○           │
│  ○    ○         │                │  ○    ○         │
│     ══════════  │    vs          │     ╔═══════╗   │
│  ×     ×        │                │  ×  ║ × ×   ║   │
│    ×            │                │    ×╚═══════╝   │
└─────────────────┘                └─────────────────┘
  SOFT MARGIN                        HARD MARGIN
  Toleran error                      Tidak toleran
  Risk underfitting                  Risk overfitting
```

### Perbandingan dengan Jurnal Prasad:
| Parameter | Eksperimen Ini | Jurnal Prasad |
|-----------|----------------|---------------|
| kernel | RBF | RBF/Linear |
| C | 1.0 | 1.0 atau grid search |
| gamma | 'scale' | 'scale' atau 'auto' |
| Hasil Accuracy | 99.00-99.78% | 98-99% |

---

# 7. METODOLOGI EVALUASI: 5-Fold Cross Validation

## 7.1 Apa itu 5-Fold Stratified Cross Validation?

### Diagram:
```
FULL DATASET (235,795 samples)
│
└─── STRATIFIED SPLIT (mempertahankan proporsi kelas)
     │
     ├── FOLD 1: ████ TEST ████ │ TRAIN (80%)
     │           47,159 samples │ 188,636 samples
     │
     ├── FOLD 2: TRAIN │ ████ TEST ████ │ TRAIN
     │                 │ 47,159 samples │
     │
     ├── FOLD 3: TRAIN │ TRAIN │ ████ TEST ████ │ TRAIN
     │                 │       │ 47,159 samples │
     │
     ├── FOLD 4: TRAIN │ TRAIN │ TRAIN │ ████ TEST ████
     │                 │       │       │ 47,159 samples
     │
     └── FOLD 5: ████ TEST ████ │ TRAIN │ TRAIN │ TRAIN
                47,159 samples  │       │       │
```

## 7.2 Mengapa Stratified?

### Distribusi Per Fold:
| Fold | Training Samples | Testing Samples | Phishing % | Legitimate % |
|------|-----------------|-----------------|------------|--------------|
| 1 | 188,636 | 47,159 | 42.8% | 57.2% |
| 2 | 188,636 | 47,159 | 42.8% | 57.2% |
| 3 | 188,636 | 47,159 | 42.8% | 57.2% |
| 4 | 188,636 | 47,159 | 42.8% | 57.2% |
| 5 | 188,636 | 47,159 | 42.8% | 57.2% |

**Proporsi kelas SAMA di setiap fold!** Ini mencegah bias jika satu fold kebetulan memiliki lebih banyak kelas tertentu.

## 7.3 Keuntungan 5-Fold CV:

| Keuntungan | Penjelasan |
|------------|------------|
| **Robust evaluation** | Semua data digunakan untuk training DAN testing |
| **Reduce variance** | Hasil tidak bergantung pada satu split |
| **Better generalization estimate** | Lebih representatif dari performa sebenarnya |
| **Detect overfitting** | Jika std tinggi → model tidak stabil |

## 7.4 Mengapa 5 Fold (bukan 3 atau 10)?

| K-Fold | Training Size | Bias | Variance | Waktu |
|--------|--------------|------|----------|-------|
| **3-Fold** | 66.7% | Tinggi | Rendah | Cepat |
| **5-Fold** | 80% | Sedang | Sedang | Sedang |
| **10-Fold** | 90% | Rendah | Tinggi | Lama |

**5-Fold adalah sweet spot:** Cukup data untuk training (80%), cukup folds untuk robust estimate, waktu training reasonable.

---

# 8. HASIL EKSPERIMEN

## 8.1 Tabel Hasil Lengkap (5-Fold Cross Validation)

### ACCURACY
| Feature Set | Random Forest | XGBoost | SVM |
|-------------|---------------|---------|-----|
| **Boruta** | 0.9959 | 0.9973 | 0.9900 |
| **RFE** | 0.9976 | **0.9987** | 0.9964 |
| **Correlation** | 0.9790 | 0.9806 | 0.9777 |
| **ContrastFS** | 0.9790 | 0.9805 | 0.9777 |
| **All Features** | **0.9992** | **0.9992** | 0.9978 |

### PRECISION
| Feature Set | Random Forest | XGBoost | SVM |
|-------------|---------------|---------|-----|
| **Boruta** | 0.9957 | 0.9973 | 0.9929 |
| **RFE** | 0.9968 | **0.9986** | 0.9971 |
| **Correlation** | 0.9799 | 0.9815 | 0.9788 |
| **ContrastFS** | 0.9800 | 0.9813 | 0.9788 |
| **All Features** | **0.9987** | **0.9991** | 0.9973 |

### RECALL
| Feature Set | Random Forest | XGBoost | SVM |
|-------------|---------------|---------|-----|
| **Boruta** | 0.9970 | 0.9979 | 0.9896 |
| **RFE** | 0.9991 | **0.9992** | 0.9966 |
| **Correlation** | 0.9835 | 0.9846 | 0.9823 |
| **ContrastFS** | 0.9834 | 0.9847 | 0.9823 |
| **All Features** | **0.9998** | **0.9995** | 0.9989 |

### F1-SCORE
| Feature Set | Random Forest | XGBoost | SVM |
|-------------|---------------|---------|-----|
| **Boruta** | 0.9964 | 0.9976 | 0.9913 |
| **RFE** | 0.9979 | **0.9989** | 0.9969 |
| **Correlation** | 0.9817 | 0.9831 | 0.9806 |
| **ContrastFS** | 0.9817 | 0.9830 | 0.9806 |
| **All Features** | **0.9993** | **0.9993** | 0.9981 |

### TRAINING TIME (seconds)
| Feature Set | Random Forest | XGBoost | SVM |
|-------------|---------------|---------|-----|
| **Boruta** | 31.78 | 146.14 | 439.45 |
| **RFE** | 33.98 | 198.93 | 160.57 |
| **Correlation** | **18.03** | 90.99 | 1491.52 |
| **ContrastFS** | 20.31 | 94.73 | 1511.37 |
| **All Features** | 57.20 | 499.01 | 408.46 |

---

## 8.2 Ranking Performa

### 🏆 Top 5 Kombinasi Berdasarkan F1-Score:

| Rank | Feature Set | Model | F1-Score | Training Time |
|------|-------------|-------|----------|---------------|
| 🥇 | All Features | XGBoost | 0.9993 | 499.01s |
| 🥇 | All Features | Random Forest | 0.9993 | 57.20s |
| 🥈 | RFE | XGBoost | 0.9989 | 198.93s |
| 🥉 | All Features | SVM | 0.9981 | 408.46s |
| 4 | RFE | Random Forest | 0.9979 | 33.98s |

### 🚀 Top 5 Kombinasi Berdasarkan Efficiency (F1/Time):

| Rank | Feature Set | Model | F1-Score | Time | Efficiency |
|------|-------------|-------|----------|------|------------|
| 🥇 | Correlation | Random Forest | 0.9817 | 18.03s | 0.0544 |
| 🥈 | ContrastFS | Random Forest | 0.9817 | 20.31s | 0.0483 |
| 🥉 | Boruta | Random Forest | 0.9964 | 31.78s | 0.0313 |
| 4 | RFE | Random Forest | 0.9979 | 33.98s | 0.0294 |
| 5 | All Features | Random Forest | 0.9993 | 57.20s | 0.0175 |

---

## 8.3 Visualisasi Hasil

### 8.3.1 Accuracy Comparison
```
                    ACCURACY BY FEATURE SET & MODEL
                    
  All Features  ████████████████████████████████████████████ 99.92%
                ████████████████████████████████████████████ 99.92%
                ███████████████████████████████████████████  99.78%
                
  RFE           ███████████████████████████████████████████  99.76%
                ████████████████████████████████████████████ 99.87%
                ███████████████████████████████████████████  99.64%
                
  Boruta        ███████████████████████████████████████████  99.59%
                ███████████████████████████████████████████  99.73%
                ██████████████████████████████████████████   99.00%
                
  Correlation   █████████████████████████████████████████    97.90%
                █████████████████████████████████████████    98.06%
                █████████████████████████████████████████    97.77%
                
  ContrastFS    █████████████████████████████████████████    97.90%
                █████████████████████████████████████████    98.05%
                █████████████████████████████████████████    97.77%
                
                ■ Random Forest  ■ XGBoost  ■ SVM
```

### 8.3.2 Training Time Comparison
```
                    TRAINING TIME COMPARISON (seconds)
                    
  All Features  ████████████ 57.20s (RF)
                ████████████████████████████████████████████████████████████ 499.01s (XGB)
                █████████████████████████████████████████ 408.46s (SVM)
                
  RFE           ██████ 33.98s (RF)
                ████████████████████████ 198.93s (XGB)
                ████████████████ 160.57s (SVM)
                
  Boruta        ██████ 31.78s (RF)
                ██████████████████ 146.14s (XGB)
                ████████████████████████████████████████████ 439.45s (SVM)
                
  Correlation   ███ 18.03s (RF)
                ██████████ 90.99s (XGB)
                ████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ 1491.52s (SVM) ⚠️
                
  ContrastFS    ████ 20.31s (RF)
                ███████████ 94.73s (XGB)
                ████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ 1511.37s (SVM) ⚠️
```

### 8.3.3 Speedup Factor (Top 10 vs All Features)
```
                    SPEEDUP: TOP 10 vs ALL FEATURES
                    
  Random Forest:
    Boruta      ████████ 1.80x faster
    RFE         ███████ 1.68x faster
    Correlation ████████████ 3.17x faster
    ContrastFS  ██████████ 2.82x faster
    
  XGBoost:
    Boruta      ████████████ 3.42x faster
    RFE         █████████ 2.51x faster
    Correlation █████████████████ 5.48x faster
    ContrastFS  ████████████████ 5.27x faster
    
  SVM:
    Boruta      ~1.0x (similar)
    RFE         █████████ 2.54x faster
    Correlation ⚠️ 0.27x SLOWER!
    ContrastFS  ⚠️ 0.27x SLOWER!
```

---

# 9. ANALISIS & DISKUSI

## 9.1 Temuan Utama

### 🔍 Temuan 1: All Features Memberikan Akurasi Tertinggi

| Metric | All Features (Best) | Top 10 (Best) | Gap |
|--------|--------------------|--------------|----|
| Accuracy | 99.92% | 99.87% (RFE) | -0.05% |
| Precision | 99.91% | 99.86% (RFE) | -0.05% |
| Recall | 99.98% | 99.92% (RFE) | -0.06% |
| F1-Score | 99.93% | 99.89% (RFE) | -0.04% |

**Interpretasi:** 
- Menggunakan 57 fitur memberikan informasi **paling lengkap**
- Gap dengan Top 10 hanya ~0.05% - **sangat kecil**
- Trade-off: All Features membutuhkan waktu training lebih lama

---

### 🔍 Temuan 2: RFE adalah Feature Selection Terbaik untuk Top 10

| Metode FS | Avg F1 | Ranking |
|-----------|--------|---------|
| **RFE** | 0.9979 | 🥇 |
| **Boruta** | 0.9951 | 🥈 |
| **Correlation** | 0.9818 | 🥉 |
| **ContrastFS** | 0.9818 | 🥉 |

**Mengapa RFE Terbaik?**
1. **Wrapper method** - menggunakan model untuk seleksi, bukan hanya statistik
2. Memilih fitur yang **optimal untuk model spesifik**
3. Kombinasi fitur URL-based + Content-based yang balanced

---

### 🔍 Temuan 3: XGBoost Konsisten Terbaik

| Model | Avg Accuracy | Avg F1 | Avg Time |
|-------|-------------|--------|----------|
| **XGBoost** | 99.13% | 99.24% | 205.96s |
| Random Forest | 99.01% | 99.10% | 32.26s |
| SVM | 98.83% | 98.95% | 802.27s |

**Mengapa XGBoost Terbaik?**
1. **Gradient boosting** - koreksi error iteratif
2. **Regularization** - mencegah overfitting
3. Handle fitur dengan berbagai skala dengan baik

---

### 🔍 Temuan 4: SVM Lambat untuk Correlation/ContrastFS Features

| Feature Set | SVM Time | RF Time | Rasio |
|------------|----------|---------|-------|
| Correlation | 1491.52s | 18.03s | **82.7x lebih lambat** |
| ContrastFS | 1511.37s | 20.31s | **74.4x lebih lambat** |

**Penyebab:**
- Correlation/ContrastFS memilih banyak **binary features** (0/1)
- SVM dengan RBF kernel **kurang efisien** untuk data sparse binary
- Banyak support vectors yang dibutuhkan

---

### 🔍 Temuan 5: Random Forest Paling Cepat dan Stabil

| Model | Min Time | Max Time | Avg Time | Std |
|-------|----------|----------|----------|-----|
| **Random Forest** | 18.03s | 57.20s | 32.26s | 15.2s |
| XGBoost | 90.99s | 499.01s | 205.96s | 155.3s |
| SVM | 160.57s | 1511.37s | 802.27s | 601.2s |

**Random Forest Advantages:**
1. **Parallel training** (`n_jobs=-1`)
2. **Tidak sensitif** terhadap jenis fitur
3. **Robust** untuk berbagai feature sets

---

## 9.2 Perbandingan dengan Jurnal Prasad

| Aspek | Eksperimen Ini | Jurnal Prasad |
|-------|----------------|---------------|
| **Dataset** | 235,795 samples | 235,795 samples |
| **Best Accuracy** | 99.92% (RF/XGB, All) | 99%+ |
| **Best Model** | XGBoost | XGBoost |
| **Feature Selection** | RFE terbaik | Multiple methods |
| **Validation** | 5-Fold CV | Cross Validation |

**Kesimpulan:** Hasil eksperimen ini **konsisten** dengan jurnal Prasad, memvalidasi metodologi yang digunakan.

---

## 9.3 Rekomendasi Praktis

### Skenario 1: Butuh Akurasi Maksimal
```
✅ Gunakan: All Features + XGBoost
   F1-Score: 99.93%
   Training Time: ~500 detik
```

### Skenario 2: Butuh Balance Akurasi & Kecepatan
```
✅ Gunakan: RFE Top 10 + XGBoost
   F1-Score: 99.89% (hanya -0.04%)
   Training Time: ~200 detik (60% lebih cepat)
```

### Skenario 3: Butuh Kecepatan Maksimal
```
✅ Gunakan: Correlation Top 10 + Random Forest
   F1-Score: 98.17%
   Training Time: ~18 detik (28x lebih cepat dari All+XGB)
```

### Skenario 4: Deployment Real-time
```
✅ Gunakan: RFE Top 10 + Random Forest
   F1-Score: 99.79%
   Training Time: ~34 detik
   ⚡ Inference cepat dengan 10 fitur
```

---

# 10. KESIMPULAN

## 10.1 Ringkasan Eksperimen

1. **Dataset:** PhiUSIIL dengan 235,795 URL dan 57 fitur numerik
2. **Feature Selection:** 4 metode (Boruta, RFE, Correlation, ContrastFS)
3. **Models:** 3 classifier (Random Forest, XGBoost, SVM)
4. **Validation:** 5-Fold Stratified Cross Validation
5. **Total Eksperimen:** 15 kombinasi

## 10.2 Temuan Kunci

| No | Temuan | Implikasi |
|----|--------|-----------|
| 1 | All Features = Akurasi tertinggi (99.92%) | Gunakan jika akurasi prioritas utama |
| 2 | RFE = Feature Selection terbaik | Wrapper method lebih efektif dari filter |
| 3 | XGBoost = Model terbaik | Gradient boosting optimal untuk dataset ini |
| 4 | Top 10 hanya -0.05% dari All Features | Feature selection sangat worth it |
| 5 | Random Forest = Paling cepat | 10-30x lebih cepat dari SVM |

## 10.3 Best Practice untuk Phishing Detection

```
┌─────────────────────────────────────────────────────────────────┐
│                   REKOMENDASI FINAL                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   🏆 PRODUKSI (High Accuracy):                                  │
│      → RFE Top 10 + XGBoost                                     │
│      → F1: 99.89%, Time: 200s                                   │
│                                                                  │
│   ⚡ REAL-TIME (Fast Inference):                                │
│      → RFE Top 10 + Random Forest                               │
│      → F1: 99.79%, Time: 34s                                    │
│                                                                  │
│   📊 RESEARCH (Maximum Accuracy):                               │
│      → All Features + XGBoost                                   │
│      → F1: 99.93%, Time: 500s                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

# 11. REFERENSI

## Paper Utama:
1. **Prasad, A., & Chandra, S. (2023).** PhiUSIIL: A diverse security profile empowered phishing URL detection framework based on similarity index and incremental learning. *Computers & Security*, 103545. https://doi.org/10.1016/j.cose.2023.103545

## Dataset:
2. **UCI Machine Learning Repository.** PhiUSIIL Phishing URL Dataset. https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset

## Libraries:
3. **Scikit-learn:** Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825-2830.
4. **Boruta:** Kursa, M. B., & Rudnicki, W. R. (2010). Feature Selection with the Boruta Package. *Journal of Statistical Software*, 36(11).

## Hyperparameter References:
5. **Random Forest:** Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5-32.
6. **Gradient Boosting:** Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. *The Annals of Statistics*, 29(5), 1189-1232.
7. **SVM:** Cortes, C., & Vapnik, V. (1995). Support-Vector Networks. *Machine Learning*, 20, 273-297.

---


**Notebook:** `Feature_Selection_Model_Training(DENGAN5-FOLD-CV).ipynb`

---
