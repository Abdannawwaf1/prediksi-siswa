import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load data
df = pd.read_csv("data_latih_dummy.csv")

# 🔍 Fitur baru
df['RataRata'] = df[['Matematika', 'Bahasa', 'IPA']].mean(axis=1)
df['TotalNilai'] = df[['Matematika', 'Bahasa', 'IPA']].sum(axis=1)
df['NilaiKategori'] = pd.cut(df['RataRata'], bins=[0, 60, 75, 90, 100], labels=['Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi'])

# Label encoding untuk fitur kategori
df['NilaiKategori'] = LabelEncoder().fit_transform(df['NilaiKategori'])

# Fitur yang digunakan
fitur = ['Matematika', 'Bahasa', 'IPA', 'Kehadiran', 'Tugas', 'RataRata', 'TotalNilai', 'NilaiKategori']
X = df[fitur]
y = df['Lulus']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔍 GridSearchCV untuk cari parameter terbaik
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 4],
}

# Cek jumlah minimum per kelas
min_per_class = y_train.value_counts().min()

# Set jumlah cv split minimal 2, maksimal 5 atau min_per_class
cv_split = min(5, min_per_class)
if cv_split < 2:
    raise ValueError("Jumlah data terlalu sedikit di salah satu kelas. Tambahkan lebih banyak data.")

# GridSearchCV untuk model terbaik
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=cv_split,
    scoring='accuracy',
    n_jobs=-1
)

grid.fit(X_train, y_train)

# Model terbaik
model = grid.best_estimator_

# Evaluasi akurasi
y_pred = model.predict(X_test)
print("Akurasi:", accuracy_score(y_test, y_pred))
print("Laporan Klasifikasi:")
print(classification_report(y_test, y_pred))

# Validasi silang
cv_scores = cross_val_score(model, X, y, cv=5)
print("Rata-rata akurasi cross-validation:", cv_scores.mean())

# Simpan model
joblib.dump(model, "model_prediksi.pkl")
print("✅ Model terbaik disimpan sebagai model_prediksi.pkl")
