import pandas as pd
from sklearn.linear_model import LinearRegression

print("Tugas Regresi Produksi Beras Di 100 Kabupaten/Kota")
print("="*50)
print("Nama : Muh. Panji Raditya")
print("NIM : F5212520049")
print("Kelas : Sistem Informasi B")
print("="*50)

# 1. Memuat Dataset
file_path = r"Data Luas Panen dan Produksi Padi di Indonesia.csv"
# Pastikan menggunakan sep=";" sesuai dengan format file Anda
df = pd.read_csv(file_path, sep=";")

# Membersihkan data dari baris yang kosong (jika ada)
df = df.dropna(subset=['Kabupaten/Kota', 'Luas Lahan/Hektar (X)', 'Produksi Padi/Ton (X)'])

# 2. Membuat Data Target Aktual (Konversi Padi ke Beras dengan standar 64,02%)
df['Beras Aktual/Ton (Y)'] = df['Produksi Padi/Ton (X)'] * 0.6402

# 3. Menentukan Variabel Independen (X) dan Dependen (y)
X = df[['Luas Lahan/Hektar (X)']]
y = df['Beras Aktual/Ton (Y)']

# 4. Membuat dan Melatih Model Regresi Linear
model = LinearRegression()
model.fit(X, y)

# 5. Melakukan Prediksi untuk SETIAP KOTA di dataset
df['Prediksi Beras/Ton'] = model.predict(X)

# Merapikan format angka di belakang koma untuk mempermudah pembacaan
df['Beras Aktual/Ton (Y)'] = df['Beras Aktual/Ton (Y)'].round(2)
df['Prediksi Beras/Ton'] = df['Prediksi Beras/Ton'].round(2)

# Memilih kolom yang ingin ditampilkan
hasil_per_kota = df[['Kabupaten/Kota', 'Luas Lahan/Hektar (X)', 'Beras Aktual/Ton (Y)', 'Prediksi Beras/Ton']]

# 6. Menampilkan Hasil
print("=== Cuplikan Hasil Prediksi 100 Kota ===")
print(hasil_per_kota.head(100).to_string(index=False))

print("\n=== Contoh Pengecekan Hasil Prediksi Kota Tertentu (Contoh: Palu) ===")
# Fitur pencarian untuk mempermudah Anda melihat hasil di wilayah spesifik
kota_spesifik = hasil_per_kota[hasil_per_kota['Kabupaten/Kota'].str.contains('Palu', case=False, na=False)]
print(kota_spesifik.to_string(index=False))

# 7. Menyimpan Hasil Lengkap ke File CSV Baru (Opsional namun disarankan)
# Hapus tanda pagar (#) pada dua baris di bawah ini jika ingin menyimpan hasil ke file excel/csv
# file_output = "Hasil_Prediksi_Beras_Per_Kota.csv"
# hasil_per_kota.to_csv(file_output, index=False, sep=";")
# print(f"\n[INFO] Data prediksi seluruh kota telah berhasil disimpan ke dalam file: {file_output}")