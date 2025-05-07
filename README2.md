# Sentiment Analysis with SVM

## 📚 Deskripsi

Proyek ini adalah tugas belajar mandiri untuk mempelajari Sentiment Analysis menggunakan algoritma **Support Vector Machine (SVM)**. Kode program diambil dari [repo GitHub ini](https://github.com/jatinwarade/Sentiment-analysis-using-SVM) sesuai instruksi tugas, dengan beberapa perbaikan dan penyesuaian agar dapat dijalankan dengan baik.

## 🔧 Tools & Library

- Python
- Pandas
- Scikit-learn
- TextBlob
- NLTK
- Matplotlib & Seaborn

## 📊 Dataset

Dataset yang digunakan bernama `Training.txt`, berisi dua kolom:

| liked | text                         |
|-------|------------------------------|
| 1     | I love this movie            |
| 0     | I hate this movie            |
| 1     | What a great experience      |
| 0     | Worst product I ever bought  |
| 1     | Totally recommend this book  |
| 0     | Disappointed by the service  |

Penjelasan kolom:
- `liked`: Label sentimen (1 = positif, 0 = negatif)
- `text`: Ulasan dalam bentuk teks

## 🛠️ Langkah-Langkah

1. **Load Dataset**  
   Membaca data dari file `Training.txt` dengan format tab-delimited.

2. **Preprocessing**  
   - Tokenisasi menggunakan TextBlob
   - Lemmatisasi untuk mendapatkan bentuk dasar kata

3. **Feature Extraction**  
   - Bag of Words (CountVectorizer)
   - Transformasi TF-IDF

4. **Modeling**  
   - Membuat pipeline yang terdiri dari CountVectorizer, TfidfTransformer, dan SVM.
   - Melakukan GridSearch untuk mencari parameter terbaik dengan cross-validation 5-fold.

5. **Evaluasi**  
   - Metrik evaluasi: akurasi, classification report (precision, recall, f1-score)
   - Uji coba prediksi dengan kalimat baru.

## 🖥️ Cara Menjalankan

1. Clone repo ini.
2. Pastikan semua dependensi sudah terinstall:
pip install -r requirements.txt
3. Jalankan script `sentiment_svm.py` atau `sentiment_svm.ipynb` (jika dalam bentuk notebook).
4. Pastikan file `Training.txt` sudah ada di direktori yang sama.

## 🔎 Hasil

Model berhasil dijalankan dengan baik. Berikut contoh hasil prediksi:

| Kalimat                              | Prediksi Sentimen |
|--------------------------------------|-------------------|
| the vinci code is awesome            | 1 (Positif)       |
| the vinci code is bad                | 0 (Negatif)       |

GridSearch juga berhasil menemukan parameter terbaik untuk SVM berdasarkan akurasi validasi.

# ✅ Catatan

- Dataset yang lebih besar akan membantu meningkatkan performa model.
- Jika mengalami error terkait library, pastikan environment Python sudah sesuai (Python 3.x) dan library seperti `nltk` sudah di-download corpus-nya (`stopwords`, `punkt`).
