# 📊 LAPORAN LENGKAP: Feature Selection dan Model Training untuk Deteksi URL Phishing

---

## 📋 DAFTAR ISI
1. [Ringkasan Eksekutif](#1-ringkasan-eksekutif)
2. [Pendahuluan & Latar Belakang](#2-pendahuluan--latar-belakang)
3. [Dataset yang Digunakan](#3-dataset-yang-digunakan)
4. [Metodologi Penelitian](#4-metodologi-penelitian)
5. [Penjelasan Kode Secara Detail](#5-penjelasan-kode-secara-detail)
6. [Metode Feature Selection](#6-metode-feature-selection)
7. [Model Machine Learning](#7-model-machine-learning)
8. [Hyperparameter dan Alasannya](#8-hyperparameter-dan-alasannya)
9. [Perbandingan dengan Jurnal Prasad](#9-perbandingan-dengan-jurnal-prasad)
10. [Hasil Eksperimen](#10-hasil-eksperimen)
11. [Analisis dan Interpretasi Hasil](#11-analisis-dan-interpretasi-hasil)
12. [Kesimpulan dan Rekomendasi](#12-kesimpulan-dan-rekomendasi)

---

## 1. RINGKASAN EKSEKUTIF

### 🎯 Tujuan Penelitian
Penelitian ini bertujuan untuk **mendeteksi URL phishing** menggunakan machine learning dengan mengevaluasi:
- **4 metode Feature Selection**: Boruta, RFE, Correlation, dan ContrastFS
- **3 model klasifikasi**: Random Forest, XGBoost (GradientBoosting), dan SVM
- **Perbandingan**: Top 10 Features vs All Features (57 fitur)

### 📈 Hasil Utama
| Metrik | Best Performer | Nilai |
|--------|----------------|-------|
| **Accuracy Tertinggi** | All Features + Random Forest | 99.92% |
| **Top 10 Features Terbaik** | RFE + XGBoost | 99.87% |
| **Training Tercepat** | Correlation + Random Forest | 18.03 detik |
| **Efisiensi Terbaik** | RFE (akurasi tinggi + waktu cepat) | 99.87% dalam 33.98 detik |

---

## 2. PENDAHULUAN & LATAR BELAKANG

### 2.1 Apa itu URL Phishing?
**Phishing** adalah serangan cyber di mana penyerang membuat website palsu yang meniru website asli (bank, e-commerce, sosial media) untuk mencuri informasi sensitif pengguna seperti:
- Username dan password
- Nomor kartu kredit
- Data pribadi lainnya

### 2.2 Mengapa Perlu Deteksi Otomatis?
- **Volume tinggi**: Ribuan URL phishing baru muncul setiap hari
- **Evolusi cepat**: Penyerang terus mengembangkan teknik baru
- **Manusia tidak cukup**: Manual review tidak skalabel
- **Machine Learning** memberikan solusi otomatis, cepat, dan akurat

### 2.3 Peran Feature Selection
Dengan **63 fitur original**, memilih fitur yang paling relevan sangat penting karena:
1. **Mengurangi overfitting** - model lebih general
2. **Mempercepat training** - lebih sedikit fitur = lebih cepat
3. **Meningkatkan interpretabilitas** - memahami fitur mana yang penting
4. **Mengurangi noise** - menghilangkan fitur yang tidak relevan

---

## 3. DATASET YANG DIGUNAKAN

### 3.1 Informasi Dataset
| Atribut | Nilai |
|---------|-------|
| **Nama Dataset** | PhiUSIIL Phishing URL Dataset |
| **File** | `PhiUSIIL_Phishing_URL_63_Features.csv` |
| **Jumlah Sampel** | 235,795 URL |
| **Jumlah Fitur Original** | 63 fitur |
| **Jumlah Fitur Numerik** | 57 fitur (setelah drop kolom non-numerik) |
| **Target Variable** | `label` (0 = Legitimate, 1 = Phishing) |

### 3.2 Kolom yang Di-drop (Non-Numerik)
Kolom berikut tidak digunakan dalam training karena bukan fitur numerik:
```
- FILENAME: Nama file
- URL: String URL lengkap
- Domain: Nama domain
- TLD: Top Level Domain (.com, .org, dll)
- Title: Judul halaman web
```

### 3.3 Distribusi Kelas
Dataset ini relatif **seimbang (balanced)**, yang penting untuk evaluasi yang fair:
- **Phishing (1)**: ~50% dari dataset
- **Legitimate (0)**: ~50% dari dataset

### 3.4 Kategori Fitur dalam Dataset
Fitur-fitur dalam dataset dapat dikategorikan sebagai berikut:

#### A. URL-based Features (Fitur berbasis URL)
| Fitur | Deskripsi |
|-------|-----------|
| `URLLength` | Panjang URL (phishing URL cenderung lebih panjang) |
| `URLCharProb` | Probabilitas karakter dalam URL |
| `LetterRatioInURL` | Rasio huruf dalam URL |
| `SpacialCharRatioInURL` | Rasio karakter spesial dalam URL |
| `URL_Profanity_Prob` | Probabilitas konten tidak pantas dalam URL |

#### B. Content-based Features (Fitur berbasis konten halaman)
| Fitur | Deskripsi |
|-------|-----------|
| `LineOfCode` | Jumlah baris kode HTML |
| `LargestLineLength` | Panjang baris terpanjang dalam kode |
| `NoOfJS` | Jumlah file JavaScript yang dimuat |
| `NoOfCSS` | Jumlah file CSS yang dimuat |
| `NoOfImage` | Jumlah gambar pada halaman |
| `NoOfExternalRef` | Jumlah referensi eksternal |
| `NoOfSelfRef` | Jumlah referensi ke diri sendiri |

#### C. Metadata Features (Fitur metadata halaman)
| Fitur | Deskripsi |
|-------|-----------|
| `HasDescription` | Apakah memiliki meta description (0/1) |
| `HasSocialNet` | Apakah ada link ke sosial media (0/1) |
| `HasCopyrightInfo` | Apakah ada informasi copyright (0/1) |
| `HasFavicon` | Apakah ada favicon (0/1) |
| `HasSubmitButton` | Apakah ada tombol submit (0/1) |
| `HasHiddenFields` | Apakah ada field tersembunyi (0/1) |
| `IsResponsive` | Apakah halaman responsive (0/1) |

#### D. Similarity Features (Fitur kesamaan)
| Fitur | Deskripsi |
|-------|-----------|
| `DomainTitleMatchScore` | Skor kecocokan domain dengan title |
| `URLTitleMatchScore` | Skor kecocokan URL dengan title |

---

## 4. METODOLOGI PENELITIAN

### 4.1 Alur Penelitian (Pipeline)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE PENELITIAN                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌───────────────────┐    ┌────────────────────┐    │
│  │  1. LOAD     │───▶│  2. PREPROCESSING │───▶│  3. FEATURE        │    │
│  │  DATASET     │    │  - Drop non-numerik│   │  SELECTION         │    │
│  │  (235,795)   │    │  - Handle missing  │   │  - Boruta          │    │
│  └──────────────┘    │  - Standardization │   │  - RFE             │    │
│                      └───────────────────┘    │  - Correlation     │    │
│                                               │  - ContrastFS      │    │
│                                               └────────────────────┘    │
│                                                         │               │
│                                                         ▼               │
│  ┌──────────────┐    ┌───────────────────┐    ┌────────────────────┐   │
│  │  6. HASIL    │◀───│  5. EVALUASI      │◀───│  4. MODEL          │   │
│  │  & ANALISIS  │    │  5-Fold CV        │    │  TRAINING          │   │
│  │              │    │  - Accuracy       │    │  - Random Forest   │   │
│  └──────────────┘    │  - Precision      │    │  - XGBoost         │   │
│                      │  - Recall         │    │  - SVM             │   │
│                      │  - F1 Score       │    └────────────────────┘   │
│                      │  - Training Time  │                              │
│                      └───────────────────┘                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Validasi: 5-Fold Stratified Cross Validation

#### Apa itu 5-Fold Stratified Cross Validation?

```
Dataset Total (235,795 sampel)
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STRATIFIED K-FOLD (K=5)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Fold 1: [████████████████████] TRAIN (80%) │ [████] TEST (20%) │
│          188,636 sampel                     │ 47,159 sampel     │
│                                                                  │
│  Fold 2: [████] TEST │ [████████████████████] TRAIN (80%)       │
│          47,159      │ 188,636 sampel                           │
│                                                                  │
│  Fold 3: [████████] TRAIN │ [████] TEST │ [████████] TRAIN      │
│                                                                  │
│  Fold 4: [████████████] TRAIN │ [████] TEST │ [████] TRAIN      │
│                                                                  │
│  Fold 5: [████████████████████] TRAIN (80%) │ [████] TEST       │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│  "STRATIFIED" = Proporsi kelas (Phishing/Legitimate) SAMA       │
│                 di setiap fold untuk evaluasi yang FAIR         │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
    Final Score = Mean(Fold1, Fold2, Fold3, Fold4, Fold5) ± Std
```

#### Mengapa Menggunakan 5-Fold CV?
1. **Mengurangi bias**: Setiap data digunakan untuk testing tepat 1x
2. **Evaluasi robust**: Hasil lebih dapat diandalkan daripada single split
3. **Stratified**: Menjaga proporsi kelas di setiap fold
4. **Standard industri**: Banyak digunakan dalam penelitian ML

---

## 5. PENJELASAN KODE SECARA DETAIL

### 5.1 Cell 1 - Import Libraries

```python
import pandas as pd          # Manipulasi data (DataFrame)
import numpy as np           # Operasi numerik
import time                  # Mengukur waktu training
import warnings
warnings.filterwarnings('ignore')  # Sembunyikan warning

# Sklearn - Library machine learning
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer

# Boruta - Feature Selection
from boruta import BorutaPy
```

**Penjelasan setiap import:**

| Library | Fungsi |
|---------|--------|
| `pandas` | Membaca CSV, manipulasi DataFrame |
| `numpy` | Operasi array dan matematika |
| `time` | Mengukur durasi training |
| `StratifiedKFold` | Membagi data dengan proporsi kelas sama |
| `cross_validate` | Menjalankan cross validation |
| `StandardScaler` | Normalisasi fitur (mean=0, std=1) |
| `RandomForestClassifier` | Model ensemble berbasis decision tree |
| `GradientBoostingClassifier` | Model boosting (alternatif XGBoost) |
| `SVC` | Support Vector Machine untuk klasifikasi |

---

### 5.2 Cell 2 - Load Dataset

```python
# Load dataset
df = pd.read_csv('PhiUSIIL_Phishing_URL_63_Features.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nLabel distribution:\n{df['label'].value_counts()}")
```

**Penjelasan:**
- `pd.read_csv()`: Membaca file CSV ke dalam DataFrame
- `df.shape`: Menampilkan (jumlah_baris, jumlah_kolom)
- `df.columns.tolist()`: Daftar semua nama kolom
- `df['label'].value_counts()`: Menghitung distribusi kelas target

---

### 5.3 Cell 3 - Data Preprocessing

```python
# Kolom non-numerik yang harus di-drop
non_numeric_cols = ['FILENAME', 'URL', 'Domain', 'TLD', 'Title']

# Drop kolom non-numerik
df_numeric = df.drop(columns=non_numeric_cols, errors='ignore')

# Pisahkan fitur dan target
X = df_numeric.drop(columns=['label'])  # Semua kolom kecuali label
y = df_numeric['label']                  # Hanya kolom label

# Handle missing values dengan median
X = X.fillna(X.median())

# Pastikan semua kolom numerik
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(X.median())
```

**Penjelasan Step-by-Step:**

| Step | Kode | Alasan |
|------|------|--------|
| 1 | Drop non-numerik | Model ML hanya bisa proses angka |
| 2 | Pisah X dan y | X = fitur input, y = target output |
| 3 | Fill missing dengan median | Median robust terhadap outlier |
| 4 | Konversi ke numerik | Memastikan tipe data konsisten |

**Mengapa Median, bukan Mean?**
- **Median** tidak terpengaruh oleh outlier
- Contoh: [1, 2, 3, 4, 100] → Mean=22, Median=3
- Median lebih representatif untuk data dengan outlier

---

### 5.4 Cell 4 - Pre-defined Top 10 Features

```python
# Boruta Top 10 Features
boruta_top10 = [
    'LineOfCode',        # Jumlah baris kode HTML
    'NoOfExternalRef',   # Jumlah referensi eksternal
    'NoOfSelfRef',       # Jumlah referensi internal
    'NoOfJS',            # Jumlah file JavaScript
    'HasDescription',    # Ada meta description?
    'NoOfImage',         # Jumlah gambar
    'HasSocialNet',      # Ada link sosial media?
    'NoOfCSS',           # Jumlah file CSS
    'HasCopyrightInfo',  # Ada info copyright?
    'LargestLineLength'  # Panjang baris terpanjang
]

# RFE Top 10 Features
rfe_top10 = [
    'LineOfCode',             # Jumlah baris kode HTML
    'LargestLineLength',      # Panjang baris terpanjang
    'NoOfExternalRef',        # Jumlah referensi eksternal
    'URLCharProb',            # Probabilitas karakter URL
    'LetterRatioInURL',       # Rasio huruf dalam URL
    'SpacialCharRatioInURL',  # Rasio karakter spesial
    'NoOfCSS',                # Jumlah file CSS
    'URL_Profanity_Prob',     # Probabilitas konten tidak pantas
    'URLLength',              # Panjang URL
    'NoOfJS'                  # Jumlah file JavaScript
]

# Correlation Top 10 Features
correlation_top10 = [
    'HasSocialNet',           # Ada link sosial media?
    'HasCopyrightInfo',       # Ada info copyright?
    'HasDescription',         # Ada meta description?
    'SpacialCharRatioInURL',  # Rasio karakter spesial
    'HasHiddenFields',        # Ada field tersembunyi?
    'HasFavicon',             # Ada favicon?
    'DomainTitleMatchScore',  # Kecocokan domain-title
    'HasSubmitButton',        # Ada tombol submit?
    'IsResponsive',           # Halaman responsive?
    'URLTitleMatchScore'      # Kecocokan URL-title
]

# ContrastFS Top 10 Features (sama dengan Correlation)
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

**Insight Menarik:**
- **Boruta & RFE** fokus pada fitur **teknis** (LineOfCode, NoOfJS, dll)
- **Correlation & ContrastFS** fokus pada fitur **metadata** (HasSocialNet, HasCopyrightInfo, dll)
- Ini menunjukkan bahwa ada **dua pendekatan** untuk mendeteksi phishing

---

### 5.5 Cell 5 - Training Function dengan 5-Fold CV

```python
def train_and_evaluate_cv(X, y, model, model_name, n_splits=5):
    """
    Train model dengan 5-Fold Stratified Cross Validation
    """
    # Setup 5-Fold Stratified Cross Validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Print distribusi kelas di setiap fold
    print("DISTRIBUSI KELAS DI SETIAP FOLD")
    for fold_num, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        print(f"FOLD {fold_num}")
        print(f"  TRAINING: {len(y_train)} samples")
        print(f"  TESTING: {len(y_test)} samples")

    # Define scorers untuk multiple metrics
    scorers = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='binary'),
        'recall': make_scorer(recall_score, average='binary'),
        'f1': make_scorer(f1_score, average='binary')
    }
    
    # Start timer
    start_time = time.time()
    
    # Perform cross validation
    cv_results = cross_validate(
        model, X, y, 
        cv=skf, 
        scoring=scorers,
        return_train_score=False,
        n_jobs=-1  # Gunakan semua CPU cores
    )
    
    # End timer
    training_time = time.time() - start_time
    
    # Calculate mean dan std untuk setiap metric
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

**Penjelasan Parameter:**

| Parameter | Nilai | Penjelasan |
|-----------|-------|------------|
| `n_splits=5` | 5 | Jumlah fold untuk cross validation |
| `shuffle=True` | True | Acak data sebelum split |
| `random_state=42` | 42 | Seed untuk reproducibility |
| `n_jobs=-1` | -1 | Gunakan semua CPU cores (paralel) |
| `average='binary'` | binary | Untuk klasifikasi binary (2 kelas) |

---

### 5.6 Cell 6 - Model Definitions

```python
def get_models(n_features):
    """
    Get models dengan max_depth yang disesuaikan jumlah fitur:
    - Untuk 10 features: max_depth = 10
    - Untuk 57 features (all): max_depth = 20
    """
    # Tentukan max_depth berdasarkan jumlah fitur
    if n_features <= 10:
        max_depth = 10  # Untuk top 10 features
    else:
        max_depth = 20  # Untuk all features (57)
    
    return {
        # Random Forest
        'Random Forest': RandomForestClassifier(
            n_estimators=100,      # Jumlah decision tree
            max_depth=max_depth,   # Kedalaman maksimum tree
            random_state=42,       # Untuk reproducibility
            n_jobs=-1              # Paralel processing
        ),
        
        # XGBoost (GradientBoosting sebagai alternatif)
        'XGBoost': GradientBoostingClassifier(
            n_estimators=100,              # Jumlah boosting stages
            max_depth=min(max_depth, 10),  # Max depth (lebih shallow)
            random_state=42
        ),
        
        # SVM dengan RBF kernel
        'SVM': SVC(
            kernel='rbf',      # Radial Basis Function kernel
            C=1.0,             # Regularization parameter
            gamma='scale',     # Kernel coefficient
            random_state=42
        )
    }
```

---

### 5.7 Cell 7 - Main Training Loop

```python
for fs_name, features in feature_sets.items():
    n_features = len(features)
    print(f"Feature Set: {fs_name} ({n_features} features)")
    
    # Inisialisasi hasil untuk setiap feature set
    results['Accuracy'][fs_name] = {}
    results['Precision'][fs_name] = {}
    results['Recall'][fs_name] = {}
    results['F1'][fs_name] = {}
    results['Training Time'][fs_name] = {}
    
    # Select features
    X_fs = X[features]
    
    # PENTING: StandardScaler untuk normalisasi
    scaler_fs = StandardScaler()
    X_fs_scaled = scaler_fs.fit_transform(X_fs)
    
    # Get models dengan max_depth yang sesuai
    models = get_models(n_features)
    
    # Training setiap model
    for model_name, model in models.items():
        print(f"  Training {model_name}...")
        
        # Train dan evaluate dengan 5-Fold CV
        metrics = train_and_evaluate_cv(
            X_fs_scaled, y, 
            model, model_name,
            n_splits=5
        )
        
        # Simpan hasil
        results['Accuracy'][fs_name][model_name] = metrics['accuracy']
        results['Precision'][fs_name][model_name] = metrics['precision']
        results['Recall'][fs_name][model_name] = metrics['recall']
        results['F1'][fs_name][model_name] = metrics['f1']
        results['Training Time'][fs_name][model_name] = metrics['training_time']
```

**Mengapa StandardScaler?**

```
SEBELUM SCALING:          SETELAH SCALING:
┌─────────────────┐       ┌─────────────────┐
│ URLLength: 0-2000│       │ URLLength: -2 to +2│
│ NoOfJS: 0-10    │  ───▶  │ NoOfJS: -2 to +2   │
│ LineOfCode: 0-50000│    │ LineOfCode: -2 to +2│
└─────────────────┘       └─────────────────┘
      SKALA BEDA                SKALA SAMA

Rumus: X_scaled = (X - mean) / std
- Mean = 0
- Std = 1
```

**Mengapa penting untuk SVM?**
- SVM sangat sensitif terhadap skala fitur
- Fitur dengan nilai besar akan mendominasi
- Scaling membuat semua fitur sama pentingnya

---

## 6. METODE FEATURE SELECTION

### 6.1 Boruta

```
┌────────────────────────────────────────────────────────────────┐
│                        CARA KERJA BORUTA                        │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. BUAT SHADOW FEATURES (kopian acak dari fitur asli)         │
│     ┌─────────────────────────────────────────────────────┐    │
│     │ Original: [F1, F2, F3, F4, F5]                      │    │
│     │ Shadow:   [S1, S2, S3, S4, S5] ← nilai diacak       │    │
│     └─────────────────────────────────────────────────────┘    │
│                                                                 │
│  2. TRAIN RANDOM FOREST dengan semua fitur (original + shadow) │
│                                                                 │
│  3. HITUNG IMPORTANCE setiap fitur                             │
│     ┌─────────────────────────────────────────────────────┐    │
│     │ F1: 0.15  |  F2: 0.08  |  F3: 0.22  |  F4: 0.05     │    │
│     │ S1: 0.02  |  S2: 0.03  |  S3: 0.01  |  S4: 0.02     │    │
│     └─────────────────────────────────────────────────────┘    │
│                                                                 │
│  4. BANDINGKAN: Fitur dengan importance > max(shadow) = PENTING │
│     - Max shadow importance = 0.03                              │
│     - F1(0.15) > 0.03 ✅ | F2(0.08) > 0.03 ✅                  │
│     - F3(0.22) > 0.03 ✅ | F4(0.05) > 0.03 ✅                  │
│                                                                 │
│  5. ULANGI beberapa iterasi untuk hasil stabil                 │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

**Kelebihan Boruta:**
- ✅ Wrapper method - mempertimbangkan interaksi antar fitur
- ✅ Statistik rigorous - menggunakan statistical test
- ✅ Menghindari false positives - shadow features sebagai control

**Fitur yang Dipilih Boruta:**
1. LineOfCode, 2. NoOfExternalRef, 3. NoOfSelfRef, 4. NoOfJS
5. HasDescription, 6. NoOfImage, 7. HasSocialNet, 8. NoOfCSS
9. HasCopyrightInfo, 10. LargestLineLength

---

### 6.2 RFE (Recursive Feature Elimination)

```
┌────────────────────────────────────────────────────────────────┐
│                        CARA KERJA RFE                           │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  MULAI: 57 fitur                                               │
│                                                                 │
│  Iterasi 1: Train model → Hapus fitur paling tidak penting     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ [F1, F2, F3, ..., F57] → Hapus F42 (paling tidak penting)│   │
│  │ Sisa: 56 fitur                                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Iterasi 2: Train model → Hapus fitur paling tidak penting     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ [F1, F2, F3, ..., F56] → Hapus F31 (paling tidak penting)│   │
│  │ Sisa: 55 fitur                                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ... (ulangi sampai tersisa 10 fitur)                          │
│                                                                 │
│  Iterasi 47: Tersisa 10 fitur terbaik!                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ [LineOfCode, LargestLineLength, NoOfExternalRef, ...]    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

**Kelebihan RFE:**
- ✅ Systematic - eliminasi bertahap dari yang terburuk
- ✅ Mempertimbangkan model spesifik
- ✅ Ranking fitur yang jelas

**Fitur yang Dipilih RFE:**
1. LineOfCode, 2. LargestLineLength, 3. NoOfExternalRef, 4. URLCharProb
5. LetterRatioInURL, 6. SpacialCharRatioInURL, 7. NoOfCSS
8. URL_Profanity_Prob, 9. URLLength, 10. NoOfJS

---

### 6.3 Correlation-based Selection

```
┌────────────────────────────────────────────────────────────────┐
│                   CARA KERJA CORRELATION                        │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. HITUNG KORELASI setiap fitur dengan target (label)         │
│                                                                 │
│     Correlation Matrix dengan Target:                          │
│     ┌──────────────────────────────────────────────────────┐   │
│     │ HasSocialNet      ←──────→ label  │ r = 0.45 ★★★    │   │
│     │ HasCopyrightInfo  ←──────→ label  │ r = 0.42 ★★★    │   │
│     │ URLLength         ←──────→ label  │ r = 0.15 ★      │   │
│     │ NoOfJS            ←──────→ label  │ r = 0.08        │   │
│     └──────────────────────────────────────────────────────┘   │
│                                                                 │
│  2. RANKING berdasarkan |korelasi| (nilai absolut)             │
│                                                                 │
│  3. PILIH TOP 10 dengan korelasi tertinggi                     │
│                                                                 │
│  CATATAN: Tidak mempertimbangkan interaksi antar fitur!        │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

**Kelebihan:**
- ✅ Cepat dan sederhana
- ✅ Mudah diinterpretasi

**Kekurangan:**
- ❌ Tidak mempertimbangkan interaksi antar fitur
- ❌ Bisa memilih fitur redundan

---

### 6.4 ContrastFS

```
┌────────────────────────────────────────────────────────────────┐
│                     CARA KERJA CONTRASTFS                       │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. BAGI data menjadi dua kelompok berdasarkan target          │
│     ┌─────────────────────────────────────────────────────┐    │
│     │ Group 1: Phishing (label = 1)                       │    │
│     │ Group 2: Legitimate (label = 0)                     │    │
│     └─────────────────────────────────────────────────────┘    │
│                                                                 │
│  2. HITUNG distribusi fitur di setiap grup                     │
│                                                                 │
│  3. UKUR KONTRAS: Fitur yang paling BERBEDA antar grup         │
│     ┌─────────────────────────────────────────────────────┐    │
│     │ HasSocialNet:                                        │    │
│     │   - Phishing: 10% memiliki link sosmed               │    │
│     │   - Legitimate: 85% memiliki link sosmed             │    │
│     │   → KONTRAS TINGGI! ★★★ (Selected)                   │    │
│     │                                                       │    │
│     │ URLLength:                                            │    │
│     │   - Phishing: avg 50 chars                           │    │
│     │   - Legitimate: avg 45 chars                         │    │
│     │   → KONTRAS RENDAH (Not selected)                    │    │
│     └─────────────────────────────────────────────────────┘    │
│                                                                 │
│  4. PILIH fitur dengan kontras tertinggi                       │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

**Kelebihan:**
- ✅ Fokus pada fitur yang benar-benar membedakan kelas
- ✅ Intuitif dan interpretable

---

### 6.5 Perbandingan Hasil Feature Selection

| Rank | Boruta | RFE | Correlation | ContrastFS |
|------|--------|-----|-------------|------------|
| 1 | LineOfCode | LineOfCode | HasSocialNet | HasSocialNet |
| 2 | NoOfExternalRef | LargestLineLength | HasCopyrightInfo | HasCopyrightInfo |
| 3 | NoOfSelfRef | NoOfExternalRef | HasDescription | HasDescription |
| 4 | NoOfJS | URLCharProb | SpacialCharRatioInURL | SpacialCharRatioInURL |
| 5 | HasDescription | LetterRatioInURL | HasHiddenFields | HasHiddenFields |
| 6 | NoOfImage | SpacialCharRatioInURL | HasFavicon | HasFavicon |
| 7 | HasSocialNet | NoOfCSS | DomainTitleMatchScore | HasSubmitButton |
| 8 | NoOfCSS | URL_Profanity_Prob | HasSubmitButton | DomainTitleMatchScore |
| 9 | HasCopyrightInfo | URLLength | IsResponsive | IsResponsive |
| 10 | LargestLineLength | NoOfJS | URLTitleMatchScore | URLTitleMatchScore |

**Insight:**
- **Boruta & RFE**: Memilih fitur teknis (content-based)
- **Correlation & ContrastFS**: Memilih fitur metadata (hampir identik!)

---

## 7. MODEL MACHINE LEARNING

### 7.1 Random Forest

```
┌────────────────────────────────────────────────────────────────┐
│                      RANDOM FOREST                              │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  KONSEP: "Wisdom of the Crowd" - banyak kepala lebih baik!     │
│                                                                 │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐       ┌─────────┐       │
│  │ Tree 1  │  │ Tree 2  │  │ Tree 3  │  ...  │Tree 100 │       │
│  │   🌳    │  │   🌳    │  │   🌳    │       │   🌳    │       │
│  │ Pred: 1 │  │ Pred: 0 │  │ Pred: 1 │       │ Pred: 1 │       │
│  └────┬────┘  └────┬────┘  └────┬────┘       └────┬────┘       │
│       │            │            │                  │            │
│       └────────────┴────────────┴──────────────────┘            │
│                           │                                     │
│                           ▼                                     │
│                   ┌───────────────┐                             │
│                   │  VOTING       │                             │
│                   │  75 × "1"     │                             │
│                   │  25 × "0"     │                             │
│                   │  → Pred = 1 ★ │                             │
│                   └───────────────┘                             │
│                                                                 │
│  KEUNGGULAN:                                                    │
│  ✅ Robust terhadap overfitting                                │
│  ✅ Handle data besar dengan baik                              │
│  ✅ Tidak perlu scaling (tree-based)                           │
│  ✅ Feature importance built-in                                │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

### 7.2 XGBoost (Gradient Boosting)

```
┌────────────────────────────────────────────────────────────────┐
│                  XGBOOST / GRADIENT BOOSTING                    │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  KONSEP: "Learn from mistakes" - perbaiki error secara iteratif│
│                                                                 │
│  Iterasi 1:                                                     │
│  ┌─────────┐                                                    │
│  │ Tree 1  │ → Prediksi → Error = [0.3, -0.2, 0.5, ...]        │
│  │   🌳    │                                                    │
│  └─────────┘                                                    │
│       │                                                         │
│       ▼                                                         │
│  Iterasi 2:                                                     │
│  ┌─────────┐                                                    │
│  │ Tree 2  │ → Fokus pada ERROR dari Tree 1!                   │
│  │   🌳    │ → Perbaiki kesalahan                              │
│  └─────────┘                                                    │
│       │                                                         │
│       ▼                                                         │
│  ... (100 iterasi)                                              │
│       │                                                         │
│       ▼                                                         │
│  Final Prediction = Tree1 + Tree2 + ... + Tree100              │
│                                                                 │
│  KEUNGGULAN:                                                    │
│  ✅ Akurasi sangat tinggi                                      │
│  ✅ Handle missing values                                      │
│  ✅ Regularization built-in                                    │
│  ❌ Lebih lambat dari Random Forest                            │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

**Catatan:** 
Dalam notebook ini, kami menggunakan `GradientBoostingClassifier` dari scikit-learn sebagai alternatif XGBoost karena XGBoost memerlukan OpenMP runtime yang tidak terinstall.

---

### 7.3 SVM (Support Vector Machine)

```
┌────────────────────────────────────────────────────────────────┐
│                  SUPPORT VECTOR MACHINE (SVM)                   │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  KONSEP: Cari "hyperplane" yang memisahkan kelas dengan margin │
│          terbesar                                               │
│                                                                 │
│        Feature 2                                                │
│           ▲                                                     │
│           │     ○ ○ ○                                          │
│           │   ○   ○   ○                                        │
│           │ ○       ○                                          │
│           │         ╱ ← Hyperplane (garis pemisah)             │
│           │       ╱                                            │
│           │     ╱   ● ●                                        │
│           │   ╱   ●   ● ●                                      │
│           │ ╱   ●       ●                                      │
│           └──────────────────────▶ Feature 1                   │
│                                                                 │
│  RBF KERNEL: Untuk data yang tidak linear separable            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Original Space:         RBF Kernel Space:                │   │
│  │    ○ ● ○ ●                    ○ ○                        │   │
│  │    ● ○ ● ○    ────▶           ○ ○   (bisa dipisahkan!)   │   │
│  │    ○ ● ○ ●                  ● ● ● ●                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  KEUNGGULAN:                                                    │
│  ✅ Efektif di high-dimensional space                          │
│  ✅ Memory efficient (hanya support vectors)                   │
│  ❌ LAMBAT untuk dataset besar (O(n²) sampai O(n³))            │
│  ❌ WAJIB scaling                                              │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## 8. HYPERPARAMETER DAN ALASANNYA

### 8.1 Random Forest Hyperparameters

| Parameter | Nilai | Alasan |
|-----------|-------|--------|
| `n_estimators=100` | 100 trees | **Standard value** yang memberikan balance antara akurasi dan kecepatan. Lebih dari 100 jarang memberikan improvement signifikan. |
| `max_depth=10` (top 10) | 10 levels | **Cukup untuk 10 fitur**. Rule of thumb: max_depth ≈ jumlah fitur. Mencegah overfitting. |
| `max_depth=20` (all features) | 20 levels | **Lebih dalam untuk 57 fitur** agar model bisa capture pattern yang lebih kompleks. |
| `random_state=42` | 42 | **Reproducibility** - hasil sama setiap kali dijalankan. 42 adalah "magic number" dari Hitchhiker's Guide to Galaxy. |
| `n_jobs=-1` | All cores | **Parallel processing** - memanfaatkan semua CPU cores untuk training lebih cepat. |

**Mengapa max_depth disesuaikan dengan jumlah fitur?**
```
┌─────────────────────────────────────────────────────────────┐
│                    MAX_DEPTH EXPLANATION                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  10 FITUR → max_depth = 10                                  │
│  ┌────────────────────────────────────┐                     │
│  │ Depth 1: Split on Feature A        │                     │
│  │ Depth 2: Split on Feature B        │                     │
│  │ ...                                 │                     │
│  │ Depth 10: Leaf node (prediction)   │                     │
│  └────────────────────────────────────┘                     │
│  → Setiap fitur berpeluang di-split sekali                  │
│                                                              │
│  57 FITUR → max_depth = 20                                  │
│  ┌────────────────────────────────────┐                     │
│  │ Lebih dalam untuk menangkap        │                     │
│  │ interaksi kompleks antar 57 fitur  │                     │
│  └────────────────────────────────────┘                     │
│                                                              │
│  TRADE-OFF:                                                  │
│  - Terlalu dangkal → Underfitting (tidak capture pattern)   │
│  - Terlalu dalam → Overfitting (menghafal noise)            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

### 8.2 XGBoost (GradientBoosting) Hyperparameters

| Parameter | Nilai | Alasan |
|-----------|-------|--------|
| `n_estimators=100` | 100 stages | **Standard boosting iterations**. Lebih banyak = lebih akurat tapi lebih lambat. |
| `max_depth=10` | 10 | **Lebih shallow dari RF** karena boosting sudah menambah kompleksitas secara iteratif. |
| `random_state=42` | 42 | Reproducibility |

**Mengapa XGBoost lebih shallow?**
```
Random Forest:      XGBoost:
Trees PARALEL       Trees SEKUENSIAL
┌───┐ ┌───┐ ┌───┐   ┌───┐
│ T1│ │ T2│ │ T3│   │ T1│ → Predict → Error
└───┘ └───┘ └───┘        │
       │                  ▼
       ▼             ┌───┐
    VOTING           │ T2│ → Fokus pada Error T1
                     └───┘
                          │
                          ▼
                     ┌───┐
                     │ T3│ → Fokus pada Error T2
                     └───┘

→ Kompleksitas sudah ditambah melalui boosting
→ Tidak perlu tree yang sangat dalam
→ max_depth = 10 sudah cukup
```

---

### 8.3 SVM Hyperparameters

| Parameter | Nilai | Alasan |
|-----------|-------|--------|
| `kernel='rbf'` | RBF | **Radial Basis Function** - kernel paling versatile, bisa handle non-linear patterns. |
| `C=1.0` | 1.0 | **Default regularization**. Balance antara margin besar dan misclassification rendah. |
| `gamma='scale'` | scale | **Automatic scaling** berdasarkan jumlah fitur: `1 / (n_features * X.var())` |

**Penjelasan Parameter C:**
```
┌─────────────────────────────────────────────────────────────┐
│                    PARAMETER C EXPLAINED                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  C KECIL (0.01):                  C BESAR (100):            │
│  ┌──────────────────┐             ┌──────────────────┐      │
│  │ ○       ╱ ●      │             │ ○     ╱   ●      │      │
│  │    ○  ╱    ●     │             │    ○╱      ●     │      │
│  │  ●  ╱  ○    ●    │             │     ╱ ○     ●    │      │
│  │   ╱              │             │   ╱              │      │
│  └──────────────────┘             └──────────────────┘      │
│  → Margin LEBAR                   → Margin SEMPIT           │
│  → Toleransi error tinggi         → Toleransi error rendah  │
│  → Simple model (underfitting?)   → Complex model (overfit?)│
│                                                              │
│  C = 1.0 adalah SWEET SPOT untuk kebanyakan kasus          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Penjelasan Parameter Gamma:**
```
┌─────────────────────────────────────────────────────────────┐
│                  PARAMETER GAMMA EXPLAINED                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  gamma = 'scale' → γ = 1 / (n_features × variance)          │
│                                                              │
│  GAMMA KECIL:                     GAMMA BESAR:              │
│  ┌──────────────────┐             ┌──────────────────┐      │
│  │ Jangkauan LUAS   │             │ Jangkauan SEMPIT │      │
│  │ Pengaruh ke      │             │ Pengaruh hanya   │      │
│  │ banyak titik     │             │ titik terdekat   │      │
│  │       ~~~        │             │        .         │      │
│  │     ~~~~~~~      │             │       ...        │      │
│  │   ~~~~~~~~~~~    │             │        .         │      │
│  └──────────────────┘             └──────────────────┘      │
│  → Smooth boundary               → Wiggly boundary          │
│  → Underfitting risk             → Overfitting risk         │
│                                                              │
│  'scale' = automatic adjustment berdasarkan data            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. PERBANDINGAN DENGAN JURNAL PRASAD

### 9.1 Informasi Jurnal
Berdasarkan file **Prasad.pdf** yang dilampirkan, jurnal ini membahas tentang **feature selection dan machine learning untuk deteksi phishing**.

### 9.2 Perbandingan Parameter

| Aspek | Jurnal Prasad | Eksperimen Anda |
|-------|---------------|-----------------|
| **Dataset** | PhiUSIIL | PhiUSIIL (sama) |
| **Jumlah Fitur** | 63 → Top features | 63 → Top 10 + All 57 |
| **Feature Selection** | Multiple methods | Boruta, RFE, Correlation, ContrastFS |
| **Model** | RF, XGBoost, SVM, dll | RF, XGBoost (GB), SVM |
| **Validation** | Cross Validation | 5-Fold Stratified CV |

### 9.3 Kesamaan Pendekatan
1. **Dataset yang sama** - PhiUSIIL Phishing URL Dataset
2. **Kombinasi Feature Selection + ML** - menguji berbagai metode
3. **Multiple classifier** - membandingkan beberapa model
4. **Cross Validation** - evaluasi yang robust

### 9.4 Perbedaan dan Improvement

| Aspek | Kelebihan Eksperimen Anda |
|-------|---------------------------|
| **Transparency** | Semua hyperparameter dijelaskan dengan alasan |
| **Reproducibility** | `random_state=42` untuk hasil konsisten |
| **Stratified CV** | Menjaga proporsi kelas di setiap fold |
| **Adaptive max_depth** | Disesuaikan dengan jumlah fitur |
| **Comprehensive metrics** | Accuracy, Precision, Recall, F1, Training Time |

---

## 10. HASIL EKSPERIMEN

### 10.1 Tabel Hasil Lengkap (5-Fold Cross Validation)

#### A. ACCURACY

| Feature Set | Random Forest | XGBoost | SVM |
|-------------|---------------|---------|-----|
| **Boruta** | 0.9959 | 0.9973 | 0.9900 |
| **RFE** | 0.9976 | **0.9987** | 0.9964 |
| **Correlation** | 0.9790 | 0.9806 | 0.9777 |
| **ContrastFS** | 0.9790 | 0.9805 | 0.9777 |
| **All Features** | **0.9992** | 0.9992 | 0.9978 |

#### B. PRECISION

| Feature Set | Random Forest | XGBoost | SVM |
|-------------|---------------|---------|-----|
| **Boruta** | 0.9957 | 0.9973 | 0.9929 |
| **RFE** | 0.9968 | **0.9986** | 0.9971 |
| **Correlation** | 0.9799 | 0.9815 | 0.9788 |
| **ContrastFS** | 0.9800 | 0.9813 | 0.9788 |
| **All Features** | **0.9987** | 0.9991 | 0.9973 |

#### C. RECALL

| Feature Set | Random Forest | XGBoost | SVM |
|-------------|---------------|---------|-----|
| **Boruta** | 0.9970 | 0.9979 | 0.9896 |
| **RFE** | 0.9991 | **0.9992** | 0.9966 |
| **Correlation** | 0.9835 | 0.9846 | 0.9823 |
| **ContrastFS** | 0.9834 | 0.9847 | 0.9823 |
| **All Features** | **0.9998** | 0.9995 | 0.9989 |

#### D. F1 SCORE

| Feature Set | Random Forest | XGBoost | SVM |
|-------------|---------------|---------|-----|
| **Boruta** | 0.9964 | 0.9976 | 0.9913 |
| **RFE** | 0.9979 | **0.9989** | 0.9969 |
| **Correlation** | 0.9817 | 0.9831 | 0.9806 |
| **ContrastFS** | 0.9817 | 0.9830 | 0.9806 |
| **All Features** | **0.9993** | 0.9993 | 0.9981 |

#### E. TRAINING TIME (dalam detik)

| Feature Set | Random Forest | XGBoost | SVM |
|-------------|---------------|---------|-----|
| **Boruta** | 31.78 | 146.14 | 439.45 |
| **RFE** | 33.98 | 198.93 | 160.57 |
| **Correlation** | **18.03** | 90.99 | 1491.52 |
| **ContrastFS** | 20.31 | 94.73 | 1511.37 |
| **All Features** | 57.20 | 499.01 | 408.46 |

---

### 10.2 Visualisasi Ringkasan

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         AKURASI TERTINGGI                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  🥇 All Features + RF/XGB  : 99.92%  ████████████████████████████████░   │
│  🥈 RFE + XGBoost          : 99.87%  ███████████████████████████████░░   │
│  🥉 RFE + Random Forest    : 99.76%  ██████████████████████████████░░░   │
│                                                                          │
│  TOP 10 FEATURE TERBAIK: RFE + XGBoost (99.87%)                         │
│  - Hanya 0.05% lebih rendah dari All Features!                          │
│  - Dengan fitur 5x lebih sedikit (10 vs 57)                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         TRAINING TIME                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  TERCEPAT:                                                               │
│  🥇 Correlation + RF  : 18.03s  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   │
│  🥈 ContrastFS + RF   : 20.31s  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   │
│  🥉 Boruta + RF       : 31.78s  ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   │
│                                                                          │
│  TERLAMBAT:                                                              │
│  🐢 ContrastFS + SVM  : 1511.37s (25+ menit!)                           │
│  🐢 Correlation + SVM : 1491.52s (25+ menit!)                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                    SPEEDUP: TOP 10 vs ALL FEATURES                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Random Forest:                                                          │
│    All Features: 57.20s  vs  RFE: 33.98s  → 1.7x lebih cepat           │
│                                                                          │
│  XGBoost:                                                                │
│    All Features: 499.01s vs  RFE: 198.93s → 2.5x lebih cepat           │
│                                                                          │
│  SVM:                                                                    │
│    All Features: 408.46s vs  RFE: 160.57s → 2.5x lebih cepat           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 11. ANALISIS DAN INTERPRETASI HASIL

### 11.1 Temuan Utama

#### 🔑 Finding 1: RFE adalah Feature Selection Terbaik
```
RFE + XGBoost menghasilkan akurasi 99.87% dengan hanya 10 fitur!
- Hampir sama dengan All Features (99.92%)
- Perbedaan hanya 0.05%
- Tapi training 2.5x lebih cepat
```

**Mengapa RFE unggul?**
- RFE menggunakan **backward elimination** yang sistematis
- Mempertimbangkan **model spesifik** saat memilih fitur
- Fitur yang dipilih memiliki **predictive power** tertinggi

#### 🔑 Finding 2: Boruta vs RFE - Pendekatan Berbeda, Hasil Mirip
```
Boruta: 99.73% (XGBoost) - fokus pada fitur KONTEN (LineOfCode, NoOfJS, dll)
RFE:    99.87% (XGBoost) - fokus pada fitur TEKNIS + URL
```

Keduanya valid, tergantung use case:
- **Boruta**: Jika ingin focus pada HTML/JavaScript analysis
- **RFE**: Jika ingin kombinasi URL + konten

#### 🔑 Finding 3: Correlation & ContrastFS - Akurasi Lebih Rendah
```
Correlation/ContrastFS: ~98% - masih bagus tapi lebih rendah 2%
```

**Mengapa?**
- Keduanya memilih **fitur metadata** (HasSocialNet, HasCopyrightInfo)
- Fitur ini **lebih high-level** dan kurang granular
- Tidak menangkap pattern teknis dari kode HTML

#### 🔑 Finding 4: XGBoost Konsisten Terbaik
```
Untuk SEMUA feature set, XGBoost selalu rank 1 atau 2
- Boruta: XGBoost terbaik (99.73%)
- RFE: XGBoost terbaik (99.87%)
- Correlation: XGBoost terbaik (98.06%)
- ContrastFS: XGBoost terbaik (98.05%)
```

**Mengapa XGBoost konsisten?**
- Boosting sangat efektif untuk tabular data
- Regularization mencegah overfitting
- Handle imbalanced features dengan baik

#### 🔑 Finding 5: SVM Sangat Lambat untuk Feature Tertentu
```
SVM dengan Correlation features: 1491 detik (25 menit!)
SVM dengan ContrastFS features:  1511 detik (25 menit!)
```

**Mengapa sangat lambat?**
- Correlation/ContrastFS memilih **binary features** (0/1)
- SVM dengan RBF kernel struggle dengan data binary
- Kompleksitas O(n²) to O(n³) untuk 235,795 sampel

---

### 11.2 Interpretasi Fitur Terpilih

#### Mengapa Fitur Ini Penting untuk Deteksi Phishing?

**Boruta & RFE (Fitur Teknis):**

| Fitur | Interpretasi |
|-------|--------------|
| `LineOfCode` | Phishing sites biasanya sederhana (sedikit kode) |
| `NoOfJS` | Legitimate sites lebih banyak menggunakan JavaScript |
| `NoOfExternalRef` | Phishing sites sering referensi ke banyak domain eksternal |
| `NoOfCSS` | Legitimate sites lebih kompleks dalam styling |
| `LargestLineLength` | Phishing sites sering punya minified/obfuscated code |

**Correlation & ContrastFS (Fitur Metadata):**

| Fitur | Interpretasi |
|-------|--------------|
| `HasSocialNet` | Legitimate sites hampir selalu punya link sosmed |
| `HasCopyrightInfo` | Phishing sites jarang mencantumkan copyright |
| `HasDescription` | SEO legitimate sites memiliki meta description |
| `HasFavicon` | Phishing sites sering tidak punya favicon |
| `HasSubmitButton` | Phishing sites SELALU punya form submit |

---

### 11.3 Trade-off Analysis

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     ACCURACY vs EFFICIENCY                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Accuracy                                                                │
│     ▲                                                                    │
│ 100%│         ★ All Features                                            │
│     │       ★ RFE    ★ Boruta                                           │
│ 99% │                                                                    │
│     │                                                                    │
│ 98% │    ★ Correlation/ContrastFS                                       │
│     │                                                                    │
│     └────────────────────────────────────────────────────────────────▶  │
│           Cepat                          Lambat          Training Time   │
│                                                                          │
│  SWEET SPOT: RFE + XGBoost                                              │
│  - Akurasi: 99.87% (hampir maksimal)                                    │
│  - Training: 198.93s (2.5x lebih cepat dari All Features)               │
│  - Fitur: hanya 10 (mudah diinterpretasi)                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 12. KESIMPULAN DAN REKOMENDASI

### 12.1 Kesimpulan Utama

1. **Feature Selection BERHASIL** mengurangi fitur dari 57 menjadi 10 dengan minimal loss akurasi (0.05%)

2. **RFE adalah metode Feature Selection terbaik** untuk dataset ini dengan akurasi 99.87%

3. **XGBoost adalah model terbaik** secara konsisten di semua feature set

4. **Trade-off optimal**: RFE + XGBoost memberikan balance terbaik antara akurasi dan efisiensi

5. **SVM tidak cocok** untuk fitur binary (Correlation/ContrastFS) - sangat lambat

### 12.2 Rekomendasi untuk Deployment

| Skenario | Rekomendasi | Alasan |
|----------|-------------|--------|
| **Production (Real-time)** | RFE + Random Forest | Cepat (33.98s) dengan akurasi 99.76% |
| **Batch Processing** | RFE + XGBoost | Akurasi tertinggi (99.87%) |
| **Research/Analysis** | All Features + RF/XGB | Untuk baseline comparison |
| **Edge Device** | RFE + RF (10 fitur) | Resource terbatas, hanya butuh 10 fitur |

### 12.3 Limitasi Penelitian

1. **Dataset tunggal** - Perlu validasi di dataset lain
2. **XGBoost alternatif** - Menggunakan GradientBoosting, bukan native XGBoost
3. **Hyperparameter default** - Belum dilakukan hyperparameter tuning
4. **Tidak ada ensemble** - Belum mencoba kombinasi model

### 12.4 Saran Penelitian Lanjutan

1. **Hyperparameter Tuning** menggunakan GridSearchCV atau RandomSearchCV
2. **Ensemble Methods** - Stacking atau Voting Classifier
3. **Deep Learning** - Neural Network untuk comparison
4. **Real-time Testing** - Deploy dan test dengan URL real
5. **Adversarial Testing** - Uji ketahanan terhadap phishing yang sophisticated

---

## 📎 LAMPIRAN

### A. File Output
- `TrainingTime_FIXversion/ResultFIXversion.csv` - Hasil utama
- `TrainingTime_FIXversion/complete_summary.csv` - Ringkasan lengkap
- `TrainingTime_FIXversion/selected_features_summary.csv` - Daftar fitur terpilih
- `TrainingTime_FIXversion/metrics_heatmap.png` - Visualisasi heatmap
- `TrainingTime_FIXversion/training_time_comparison.png` - Perbandingan waktu
- `TrainingTime_FIXversion/radar_chart.png` - Radar chart metrik

### B. Kode Lengkap
Lihat notebook: `Feature_Selection_Model_Training(DENGAN5-FOLD-CV).ipynb`

### C. Referensi
- PhiUSIIL Dataset
- Prasad et al. (Jurnal Feature Selection untuk Phishing Detection)
- Scikit-learn Documentation
- Boruta Documentation

---