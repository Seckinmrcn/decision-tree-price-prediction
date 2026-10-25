import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

df = pd.read_csv("car_price.csv")

print(" 1. VERİ SETİ BOYUTU ")
print(f"Satır Sayısı: {df.shape[0]}, Sütun Sayısı: {df.shape[1]}\n")

print(" 2. İLK 5 SATIR ")
print(df.head(), "\n")

print(" 3. SÜTUN TİPLERİ VE EKSİK VERİLER")
bilgi_df = pd.DataFrame({
    "Veri Tipi": df.dtypes,
    "Eksik Değer Sayısı": df.isnull().sum(),
    "Eksik Oranı (%)": (df.isnull().sum() / len(df) * 100).round(2),
    "Benzersiz Değer Sayısı (Unique)": df.nunique()
})
print(bilgi_df, "\n")

print("İSTATİSTİKLER")
print(df.describe().round(2))
