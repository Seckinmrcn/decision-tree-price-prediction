import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

best_model = joblib.load("car_price_dt_model.pkl")

preprocessor = best_model.named_steps["preprocessor"]
tree_model = best_model.named_steps["regressor"]

def turkce_nitelik_adi(name):
    clean = name.replace("num__", "").replace("cat__", "").replace("target__", "")
    translations = [
        ("Prod. year", "Üretim Yılı"),
        ("Engine volume", "Motor Hacmi (L)"),
        ("Mileage", "Kilometre"),
        ("Airbags", "Hava Yastığı Sayısı"),
        ("Levy", "Yıllık Vergi ($)"),
        ("Cylinders", "Silindir Sayısı"),
        ("Arac_Yasi", "Araç Yaşı"),
        ("Yillik_Km", "Yıllık Ortalama KM"),
        ("Category_", "Kasa Tipi: "),
        ("Gear box type_", "Vites: "),
        ("Drive wheels_", "Çekiş: "),
        ("Fuel type_", "Yakıt: "),
        ("Leather interior_", "Deri Koltuk: "),
        ("Wheel_", "Direksiyon: "),
        ("Turbo_", "Turbo: "),
        ("Manufacturer_", "Marka: "),
        ("Manufacturer", "Marka"),
        ("Model", "Araç Modeli"),
    ]
    for eng, tr in translations:
        if eng in clean:
            clean = clean.replace(eng, tr)
    return clean

raw_feature_names = preprocessor.get_feature_names_out()
clean_feature_names = [turkce_nitelik_adi(name) for name in raw_feature_names]

importances = tree_model.feature_importances_
fi_df = pd.DataFrame({
    "Nitelik": clean_feature_names,
    "Önem_Oranı": importances * 100
}).sort_values(by="Önem_Oranı", ascending=False)

print(" FİYATI EN ÇOK ETKİLEYEN İLK 10 NİTELİK (%) ")
print(fi_df.head(10).to_string(index=False, float_format="%.2f"))

fi_df.head(15).to_csv("feature_importance.csv", index=False)

plt.figure(figsize=(10, 6), dpi=300)
top_10 = fi_df.head(10).sort_values(by="Önem_Oranı", ascending=True)
plt.barh(top_10["Nitelik"], top_10["Önem_Oranı"], color="#2b5c8f")
plt.xlabel("Önem Düzeyi (%)")
plt.title("Karar Ağacında Araç Fiyatını Belirleyen En Önemli 10 Nitelik")
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

plt.figure(figsize=(24, 12), dpi=300)
plot_tree(
    tree_model,
    feature_names=clean_feature_names,
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=10,
    precision=1
)
plt.title("Karar Ağacı İlk 3 Seviye Dallanma Kural Yapısı (Root Node ve İlk Bölmeler)", fontsize=16)
plt.tight_layout()
plt.savefig("tree_structure.png")
plt.close()

print("\n'feature_importance.png' ve 'tree_structure.png' yüksek çözünürlüklü olarak kaydedildi!")