
# 🧪 Klasifikasi Jeruk vs Anggur Menggunakan Naive Bayes

Dataset yang digunakan merupakan data karakteristik buah (jeruk dan anggur) dari Kaggle:  
[https://www.kaggle.com/datasets/joshmcadams/oranges-vs-grapefruit](https://www.kaggle.com/datasets/joshmcadams/oranges-vs-grapefruit)

---

## 📁 1. Deskripsi Dataset

Dataset terdiri dari 10.000 sampel buah dengan atribut:
- `name`: nama buah (orange atau grapefruit)
- `diameter`: diameter buah (float)
- `weight`: berat buah (float)
- `red`, `green`, `blue`: nilai warna RGB buah (integer)

---

## 🛠️ 2. Tahapan Pembuatan Model

### a. Import Library
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

### b. Load Dataset dan Eksplorasi Awal
```python
df = pd.read_csv('citrus.csv')
print(df.info())
print(df.describe())
print(df['name'].value_counts())
```

### c. Pra-Pemrosesan Data
- Encode label:
```python
df['label'] = df['name'].map({'orange': 0, 'grapefruit': 1})
X = df.drop(columns=['name', 'label'])
y = df['label']
```

---

## 🤖 3. Training Model Naive Bayes
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)
```

---

## 🧪 4. Evaluasi Model
```python
y_pred = model.predict(X_test)
print("Akurasi:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Orange', 'Grapefruit'], yticklabels=['Orange', 'Grapefruit'])
plt.title("Confusion Matrix")
plt.show()
```

### 📊 Hasil Evaluasi:
- **Akurasi**: 92%
- **F1-Score (rata-rata)**: 92%
- Confusion matrix menunjukkan model cukup seimbang dalam mengklasifikasikan kedua jenis buah.

---

## 🧾 5. Pengujian Model (Prediksi Manual)
```python
sample_data = [[10.0, 180.0, 150, 75, 10]]  # Ubah sesuai karakteristik buah
prediction = model.predict(sample_data)
print("Prediksi:", "Orange 🍊" if prediction[0] == 0 else "Grapefruit 🍇")
```

---

## ✅ Kesimpulan
Model **Naive Bayes** bekerja dengan sangat baik dalam mengklasifikasikan buah antara jeruk dan anggur hanya berdasarkan fitur warna, berat, dan ukuran. Akurasi 92% menunjukkan bahwa model ini dapat digunakan sebagai baseline untuk klasifikasi buah berbasis citra sensorik.
