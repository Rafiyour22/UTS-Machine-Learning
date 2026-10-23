# UTS Machine Learning

## Identitas Mahasiswa

- **Nama:** Rafi Ul Aziz
- **NIM:** 241011400557
- **Kelas:** Reg B
- **Random state:** `57`

## Tujuan

Tugas ini menerapkan Linear Regression dan Logistic Regression secara manual menggunakan NumPy, mengevaluasi performa model, mengeksplorasi pengaruh learning rate dan threshold, serta membandingkan hasil dengan scikit-learn pada bagian bonus.

## Dataset

- **`Ads_Data.csv`** — 40 baris dengan kolom `BiayaIklan` dan `Penjualan`. Dataset digunakan untuk memodelkan hubungan biaya iklan dengan penjualan.
- **`Email_Data.csv`** — 60 baris dengan kolom `JumlahLink`, `KataPromo`, dan `Spam`. Label `Spam` bernilai 1 untuk spam dan 0 untuk bukan spam; dataset berisi 25 email spam dan 35 email bukan spam.

Kedua dataset merupakan file asli yang disediakan bersama materi UTS.

## Bagian A — Linear Regression

Model utama dibuat dengan Normal Equation menggunakan NumPy. Pembagian data menggunakan `test_size=0.2` dan `random_state=57` (32 data training dan 8 data testing). Model yang diperoleh:

\[
\hat{y} = 28.6394 + 8.9659X
\]

Koefisien positif menunjukkan bahwa kenaikan biaya iklan berkaitan dengan kenaikan prediksi penjualan pada rentang data yang diamati.

### Evaluasi

| Metrik | Training | Testing |
|---|---:|---:|
| MAE | 4.8684 | 3.3922 |
| MSE | 41.6536 | 14.7946 |
| RMSE | 6.4540 | 3.8464 |
| R² | 0.9378 | 0.9668 |

Nilai R² testing menunjukkan model menjelaskan sebagian besar variasi penjualan pada subset testing ini. RMSE testing sekitar 3.8464 dalam satuan target dataset. Nilai tersebut bukan jaminan bahwa model akan sama akuratnya pada periode atau data baru yang berbeda.

### Gradient Descent

Eksperimen menggunakan 5.000 iterasi dan learning rate `0.001`, `0.02`, serta `0.1`, dengan fitur pada skala aslinya seperti rumus Gradient Descent pada materi.

- `0.001`: konvergen lebih lambat; setelah 5.000 iterasi koefisien belum sedekat learning rate yang lebih besar dengan Normal Equation.
- `0.02`: konvergen dan hasil akhirnya mendekati Normal Equation.
- `0.1`: divergen pada eksperimen ini; pembaruan parameter terlalu besar untuk skala fitur. Implementasi mendeteksi kondisi tersebut dan grafik menampilkan loss valid yang tersedia, tanpa menyatakan hasilnya konvergen.

Prediksi biaya iklan 12 juta rupiah merupakan ekstrapolasi karena berada di luar rentang biaya iklan dataset, sehingga perlu ditafsirkan dengan hati-hati.

## Bagian B — Logistic Regression

Logistic Regression manual dilatih menggunakan Gradient Descent dengan learning rate `0.1` dan 10.000 iterasi. Fitur distandardisasi berdasarkan data training untuk membantu kestabilan optimisasi; koefisien dikonversi kembali ke skala fitur asli agar persamaan probabilitas tetap dapat dibaca dalam `JumlahLink` dan `KataPromo`.

Persamaan yang dihasilkan:

\[
P(Spam)=\sigma(-12.1528 + 1.0709\,JumlahLink + 1.0824\,KataPromo)
\]

Log loss turun dari **0.680134** menjadi **0.251342** selama training.

### Evaluasi testing (threshold 0,5)

| Metrik | Hasil |
|---|---:|
| Accuracy | 0.8000 (80%) |
| Precision | 0.6667 (66.67%) |
| Recall | 0.8000 (80%) |
| F1-score | 0.7273 (72.73%) |

Confusion matrix memakai urutan `[[TN, FP], [FN, TP]]`:

```text
[[8, 2],
 [1, 4]]
```

Pada data testing ini, terdapat 8 true negative, 2 false positive, 1 false negative, dan 4 true positive. Threshold 0,7 dipilih untuk prediksi email baru karena pada data testing menghasilkan precision lebih tinggi dibanding threshold 0,5, sementara recall tetap 0,8. Pemilihan threshold selalu merupakan kompromi antara email spam yang lolos dan email normal yang keliru ditandai spam.

## Bonus — Perbandingan scikit-learn

- **Linear Regression:** koefisien dan metrik testing dari implementasi manual sama dengan `sklearn.linear_model.LinearRegression` hingga pembulatan yang ditampilkan; R² testing 0.96684 dan RMSE 3.84637.
- **Logistic Regression:** pada threshold 0,5, implementasi manual dan scikit-learn menghasilkan confusion matrix yang sama pada split ini, yaitu `[[8, 2], [1, 4]]`, serta accuracy 0.80, precision 0.6667, recall 0.80, dan F1-score 0.7273. Koefisiennya berbeda karena scikit-learn memakai regularisasi L2 secara default (`C=1.0`), sedangkan implementasi manual tidak memakai regularisasi. Prosedur optimisasi juga berbeda.

## Kesimpulan

Pada split dengan random state 57, Linear Regression menunjukkan kecocokan yang baik pada data testing, sedangkan Logistic Regression mencapai accuracy 80%. Hasil ini menggambarkan performa pada dataset kecil yang tersedia, bukan jaminan performa untuk seluruh data di masa depan. Grafik, tabel, prediksi, dan output eksekusi tersimpan di `tugas_UTS.ipynb`.

## Struktur Repository

```text
uts-machine-learning-241011400557/
├── README.md
├── tugas_UTS.ipynb
├── Ads_Data.csv
└── Email_Data.csv
```

## Dependensi

- Python 3
- NumPy
- Pandas
- Matplotlib
- scikit-learn (khusus bagian bonus)
- Jupyter Notebook / IPython dan `ipykernel` untuk menjalankan notebook

Buka `tugas_UTS.ipynb` di Jupyter Notebook atau VS Code, lalu jalankan **Run All** dari kernel Python yang memiliki dependensi tersebut.
