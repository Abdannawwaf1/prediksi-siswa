import pandas as pd
import numpy as np

# Menghasilkan data dummy
np.random.seed(42)

# Membuat data dummy untuk siswa
data = {
    'Siswa': [f"Siswa {chr(65+i)}" for i in range(10)],
    'Matematika': np.random.randint(40, 100, 10),
    'Bahasa': np.random.randint(40, 100, 10),
    'IPA': np.random.randint(40, 100, 10),
    'Kehadiran': np.random.randint(1, 11, 10),
    'Tugas': np.random.randint(1, 11, 10),
}

df = pd.DataFrame(data)

# Menambahkan fitur RataRata dan TotalNilai
df['RataRata'] = df[['Matematika', 'Bahasa', 'IPA']].mean(axis=1)
df['TotalNilai'] = df[['Matematika', 'Bahasa', 'IPA']].sum(axis=1)

# Menambahkan kategori nilai
df['NilaiKategori'] = pd.cut(df['RataRata'], bins=[0, 60, 75, 90, 100], labels=['Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi'])

# Menambahkan kolom status kelulusan (sebagai data target)
df['Lulus'] = np.where(df['RataRata'] >= 75, 1, 0)

# Menyimpan data dummy ke file CSV
df.to_csv("data_latih_dummy.csv", index=False)
print("Data dummy berhasil disimpan di 'data_latih_dummy.csv'.")
