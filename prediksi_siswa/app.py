import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load model
model = joblib.load("model_prediksi.pkl")

# Proses fitur
def process_input(df):
    df['RataRata'] = df[['Matematika', 'Bahasa', 'IPA']].mean(axis=1)
    df['TotalNilai'] = df[['Matematika', 'Bahasa', 'IPA']].sum(axis=1)
    df['NilaiKategori'] = pd.cut(
        df['RataRata'],
        bins=[0, 60, 75, 90, 100],
        labels=['Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi']
    )
    df['NilaiKategori'] = LabelEncoder().fit_transform(df['NilaiKategori'])

    fitur = ['Matematika', 'Bahasa', 'IPA', 'Kehadiran', 'Tugas',
             'RataRata', 'TotalNilai', 'NilaiKategori']
    X_uji = df[fitur]
    return X_uji, df

# UI
st.title("🎓 Prediksi Kelulusan Siswa")
st.markdown("Upload file CSV berisi data siswa, lalu sistem akan memprediksi siapa yang lulus dan tidak.")

uploaded_file = st.file_uploader("📤 Upload file CSV data uji", type=["csv"])

if uploaded_file is not None:
    df_uji = pd.read_csv(uploaded_file)

    st.subheader("📋 Data yang Diupload")
    st.dataframe(df_uji, use_container_width=True)

    X_uji, df_proses = process_input(df_uji)
    prediksi = model.predict(X_uji)
    df_proses['Prediksi'] = prediksi
    df_proses['Prediksi'] = df_proses['Prediksi'].map({0: 'Tidak Lulus', 1: 'Lulus'})

    hasil_file = "hasil_prediksi.csv"
    df_proses.to_csv(hasil_file, index=False)

    # Tampilkan hasil prediksi di web
    st.subheader("📄 Tabel Hasil Prediksi (Dari hasil_prediksi.csv)")
    df_hasil = pd.read_csv(hasil_file)
    st.dataframe(df_hasil, use_container_width=True)

    # Tombol download
    st.download_button(
        label="⬇️ Download Hasil Prediksi CSV",
        data=open(hasil_file, "rb").read(),
        file_name="hasil_prediksi.csv",
        mime="text/csv"
    )
