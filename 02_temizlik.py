import pandas as pd

df = pd.read_csv("car_price.csv").drop(columns=["ID"]).drop_duplicates()
print(f"Başlangıç Satır Sayısı: {len(df)}")

df = df.drop(columns=["Doors"])

df["Turbo"] = df["Engine volume"].astype(str).apply(lambda x: "Yes" if "Turbo" in x else "No")
df["Engine volume"] = df["Engine volume"].astype(str).str.replace("Turbo", "").str.strip().astype(float)
df["Mileage"] = df["Mileage"].astype(str).str.replace("km", "").str.strip().astype(int)
df["Levy"] = df["Levy"].astype(str).replace("-", "0").astype(int)

df = df[(df["Engine volume"] >= 0.8) & (df["Engine volume"] <= 7.0)]
df = df[(df["Mileage"] <= 400000) & (df["Prod. year"] >= 1995)]
df = df[(df["Price"] >= 1500) & (df["Price"] <= 100000)]

populer_modeller = df["Model"].value_counts()[lambda x: x >= 20].index
df["Model"] = df["Model"].apply(lambda x: x if x in populer_modeller else "Other")

ozellik_sutunlari = [c for c in df.columns if c != "Price"]
df = df.groupby(ozellik_sutunlari, as_index=False)["Price"].median()

grup_medyan = df.groupby(["Manufacturer", "Category", "Prod. year"])["Price"].transform("median")
df = df[(df["Price"] >= grup_medyan * 0.50) & (df["Price"] <= grup_medyan * 2.0)]

print(f"Temizlik Sonrası Net Satır Sayısı: {len(df)}")
print(f"Toplam Sütun Sayısı: {df.shape[1]}")

df.to_csv("araba_temiz.csv", index=False)
print("\nYeni 'araba_temiz.csv' başarıyla kaydedildi!")