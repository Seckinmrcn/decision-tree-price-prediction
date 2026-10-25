import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Araç Fiyat Tahmin Sistemi", layout="wide")

@st.cache_data
def veriyi_yukle():
    df = pd.read_csv("araba_temiz.csv")
    return df

df = veriyi_yukle()

X_full = df.drop(columns=["Price"])
y_full = df["Price"]

sayisal_sutunlar = ["Levy", "Prod. year", "Engine volume", "Mileage", "Cylinders", "Airbags"]
kategorik_sutunlar = [
    "Manufacturer", "Model", "Category", "Leather interior",
    "Fuel type", "Gear box type", "Drive wheels", "Wheel", "Color", "Turbo"
]

@st.cache_resource
def model_egit(test_size, max_depth, min_samples_split, min_samples_leaf, criterion="friedman_mse"):
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=test_size, random_state=42
    )

    preprocessor = ColumnTransformer(transformers=[
        ("num", "passthrough", sayisal_sutunlar),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), kategorik_sutunlar)
    ])

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    tree = pipeline.named_steps["regressor"]
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

    return {
        "pipeline": pipeline,
        "train_r2": r2_score(y_train, y_train_pred),
        "test_r2": r2_score(y_test, y_test_pred),
        "mae": mean_absolute_error(y_test, y_test_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "n_leaves": tree.get_n_leaves(),
        "depth": tree.get_depth(),
        "y_test": y_test.values,
        "y_test_pred": y_test_pred,
        "importances": tree.feature_importances_,
        "feature_names": feature_names,
    }

def turkce_nitelik_adi(name):
    clean = name.replace("num__", "").replace("cat__", "")
    translations = [
        ("Prod. year", "Üretim Yılı"),
        ("Engine volume", "Motor Hacmi (L)"),
        ("Mileage", "Kilometre"),
        ("Airbags", "Hava Yastığı Sayısı"),
        ("Levy", "Yıllık Vergi ($)"),
        ("Cylinders", "Silindir Sayısı"),
        ("Category_", "Kasa Tipi: "),
        ("Gear box type_", "Vites: "),
        ("Drive wheels_", "Çekiş: "),
        ("Fuel type_", "Yakıt: "),
        ("Leather interior_", "Deri Koltuk: "),
        ("Wheel_", "Direksiyon: "),
        ("Color_", "Renk: "),
        ("Turbo_", "Turbo: "),
        ("Manufacturer_", "Marka: "),
        ("Model_", "Model: "),
        ("Manufacturer", "Marka"),
        ("Model", "Araç Modeli"),
    ]
    for eng, tr in translations:
        if eng in clean:
            clean = clean.replace(eng, tr)
    return clean

st.title("Karar Ağacı Regresyonu ile Araç Fiyat Tahmini")
st.caption("Makine Öğrenmesi")

st.header("Performans Metrikleri ve Optimum Karşılaştırma Analizi")

metrics_container = st.container()

st.divider()

col_left, col_right = st.columns(2)

with col_left:
    table_container = st.container()

    st.divider()
    st.subheader("Deney ve Hiperparametre Kontrol Paneli")

    oran_secim = st.selectbox(
        "Eğitim / Test Oranı",
        ["%60 Eğitim - %40 Test",
         "%70 Eğitim - %30 Test",
         "%80 Eğitim - %20 Test",
         "%90 Eğitim - %10 Test"],
        index=2,
    )
    oran_map = {
        "%60 Eğitim - %40 Test": 0.40,
        "%70 Eğitim - %30 Test": 0.30,
        "%80 Eğitim - %20 Test": 0.20,
        "%90 Eğitim - %10 Test": 0.10,
    }
    test_size = oran_map[oran_secim]

    sinirsiz = st.checkbox("Sınırsız Derinlik (Budanmamış Ağaç)")
    if sinirsiz:
        user_depth = None
        depth_label = "Sınırsız"
        st.info("Sınırsız mod seçili → Budama kısıtları devre dışı (split=2, leaf=1)")
        user_split = 2
        user_leaf = 1
    else:
        user_depth = st.slider("Maksimum Ağaç Derinliği", 1, 41, 16)
        depth_label = str(user_depth)
        user_split = st.slider("Düğüm Bölünmesi İçin Min. Örnek Sayısı", 2, 50, 12)
        user_leaf = st.slider("Yaprak Başına Düşen Min. Örnek Sayısı", 1, 20, 2)

aktif = model_egit(test_size, user_depth, user_split, user_leaf, criterion="friedman_mse")
optimum = model_egit(test_size, 16, 12, 2, criterion="friedman_mse")
overfit = model_egit(test_size, None, 2, 1, criterion="friedman_mse")

with metrics_container:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Eğitim R² (%)", f"{aktif['train_r2'] * 100:.2f}")
    m2.metric("Test R² (%)", f"{aktif['test_r2'] * 100:.2f}")
    m3.metric("Test MAE ($)", f"{aktif['mae']:,.0f}")
    m4.metric("Test RMSE ($)", f"{aktif['rmse']:,.0f}")
    m5.metric("Toplam Yaprak (Leaves)", f"{aktif['n_leaves']:,}")

with table_container:
    st.subheader("Model Karşılaştırma Tablosu")
    karsilastirma = pd.DataFrame({
        "Metrik": [
            "Ağaç Derinliği",
            "Yaprak Sayısı",
            "Eğitim R² (%)",
            "Test R² (%)",
            "Test MAE ($)",
            "Test RMSE ($)",
        ],
        f"Seçilen Model (Derinlik: {depth_label})": [
            aktif["depth"],
            aktif["n_leaves"],
            round(aktif["train_r2"] * 100, 2),
            round(aktif["test_r2"] * 100, 2),
            round(aktif["mae"], 2),
            round(aktif["rmse"], 2),
        ],
        "Optimum Model (Derinlik: 16)": [
            optimum["depth"],
            optimum["n_leaves"],
            round(optimum["train_r2"] * 100, 2),
            round(optimum["test_r2"] * 100, 2),
            round(optimum["mae"], 2),
            round(optimum["rmse"], 2),
        ],
        "Budanmamış Ağaç (Sınırsız)": [
            overfit["depth"],
            overfit["n_leaves"],
            round(overfit["train_r2"] * 100, 2),
            round(overfit["test_r2"] * 100, 2),
            round(overfit["mae"], 2),
            round(overfit["rmse"], 2),
        ],
    })
    st.dataframe(karsilastirma, width="stretch", hide_index=True)

with col_right:
    st.subheader("Derinlik vs. R² Başarı Eğrisi")

    depth_values = [2, 4, 6, 8, 10, 12, 14, 16, 18, 22, 26, 30, 35, 41]
    train_scores = []
    test_scores = []
    for d in depth_values:
        res = model_egit(test_size, d, 12, 2, criterion="friedman_mse")
        train_scores.append(res["train_r2"] * 100)
        test_scores.append(res["test_r2"] * 100)

    fig_curve, ax_curve = plt.subplots(figsize=(6, 4), dpi=130)
    ax_curve.plot(depth_values, train_scores, marker="o", markersize=4,
                  label="Eğitim R²", color="#2563eb")
    ax_curve.plot(depth_values, test_scores, marker="s", markersize=4,
                  label="Test R²", color="#dc2626")

    ax_curve.axvline(x=16, color="#16a34a", linestyle="--", linewidth=1.2,
                     label="Optimum (Derinlik: 16)")

    if user_depth is not None and user_depth != 16:
        ax_curve.axvline(x=user_depth, color="#f59e0b", linestyle="--", linewidth=1.2,
                         label=f"Seçilen (Derinlik: {user_depth})")

    ax_curve.set_xlabel("max_depth")
    ax_curve.set_ylabel("R² Skoru (%)")
    ax_curve.set_title("Ağaç Derinliği vs. R² Değişimi")
    ax_curve.legend(fontsize=8)
    ax_curve.grid(True, linestyle="--", alpha=0.4)
    st.pyplot(fig_curve)

st.divider()

st.header("Araç Fiyat Tahmini")

inp1, inp2, inp3 = st.columns(3)

with inp1:
    manufacturer = st.selectbox("Marka (Manufacturer)",
                                sorted(df["Manufacturer"].unique().tolist()))
    marka_modelleri = sorted(df[df["Manufacturer"] == manufacturer]["Model"].unique().tolist())
    model_secim = st.selectbox("Model", marka_modelleri)
    category = st.selectbox("Kasa Tipi (Category)",
                            sorted(df["Category"].unique().tolist()))
    fuel_type = st.selectbox("Yakıt Tipi (Fuel type)",
                             sorted(df["Fuel type"].unique().tolist()))
    gear_box = st.selectbox("Vites Tipi (Gear box type)",
                            sorted(df["Gear box type"].unique().tolist()))

with inp2:
    drive_wheels = st.selectbox("Çekiş Tipi (Drive wheels)",
                                sorted(df["Drive wheels"].unique().tolist()))
    wheel = st.selectbox("Direksiyon (Wheel)",
                         sorted(df["Wheel"].unique().tolist()))
    color = st.selectbox("Renk (Color)",
                         sorted(df["Color"].unique().tolist()))
    leather = st.selectbox("Deri Koltuk (Leather interior)", ["Yes", "No"])
    turbo = st.selectbox("Turbo Motor", ["No", "Yes"])

with inp3:
    airbags = st.slider("Hava Yastığı Sayısı (Airbags)",
                         int(df["Airbags"].min()),
                         int(df["Airbags"].max()),
                         int(df["Airbags"].median()))
    prod_year = st.slider("Üretim Yılı (Prod. year)",
                           int(df["Prod. year"].min()),
                           int(df["Prod. year"].max()),
                           int(df["Prod. year"].median()))
    engine_vol = st.number_input("Motor Hacmi (L)",
                                  min_value=float(df["Engine volume"].min()),
                                  max_value=float(df["Engine volume"].max()),
                                  value=float(df["Engine volume"].median()),
                                  step=0.1)
    mileage = st.number_input("Kilometre (Mileage)",
                               min_value=int(df["Mileage"].min()),
                               max_value=int(df["Mileage"].max()),
                               value=int(df["Mileage"].median()),
                               step=5000)
    cylinders = st.selectbox("Silindir Sayısı (Cylinders)",
                              sorted(df["Cylinders"].unique().tolist()))
    levy = st.number_input("Yıllık Vergi ($) (Levy)",
                            min_value=0,
                            max_value=int(df["Levy"].max()),
                            value=int(df["Levy"].median()),
                            step=50)

if st.button("Fiyat Tahmin Et", type="primary", use_container_width=True):
    input_df = pd.DataFrame([{
        "Levy": levy,
        "Manufacturer": manufacturer,
        "Model": model_secim,
        "Prod. year": prod_year,
        "Category": category,
        "Leather interior": leather,
        "Fuel type": fuel_type,
        "Engine volume": engine_vol,
        "Mileage": mileage,
        "Cylinders": float(cylinders),
        "Gear box type": gear_box,
        "Drive wheels": drive_wheels,
        "Wheel": wheel,
        "Color": color,
        "Airbags": airbags,
        "Turbo": turbo,
    }])

    tahmin_aktif = aktif["pipeline"].predict(input_df)[0]
    tahmin_optimum = optimum["pipeline"].predict(input_df)[0]
    mae_opt = optimum["mae"]

    res1, res2 = st.columns(2)
    with res1:
        st.success(
            f"**Seçilen Model (Derinlik: {depth_label}) Tahmini:** "
            f"${tahmin_aktif:,.2f}"
        )
    with res2:
        alt = max(2500, tahmin_optimum - mae_opt)
        ust = tahmin_optimum + mae_opt
        st.info(
            f"**Optimum Model (Derinlik: 16) Tahmini:** "
            f"${tahmin_optimum:,.2f}  \n"
            f"MAE Aralığı: ${alt:,.0f} – ${ust:,.0f}"
        )

st.divider()

st.header("Model Açıklanabilirliği ve Ağaç Görselleri")

viz1, viz2 = st.columns(2)

with viz1:
    st.subheader("En Önemli 10 Nitelik")
    imp = aktif["importances"]
    names = [turkce_nitelik_adi(name) for name in aktif["feature_names"]]
    imp_df = pd.DataFrame({"Nitelik": names, "Önem": imp})
    imp_df = imp_df.sort_values("Önem", ascending=False).head(10)

    fig_imp, ax_imp = plt.subplots(figsize=(6, 4.5), dpi=130)
    bars = ax_imp.barh(
        imp_df["Nitelik"].values[::-1],
        imp_df["Önem"].values[::-1],
        color="#2563eb",
        edgecolor="#1e40af",
    )
    ax_imp.set_xlabel("Önem Skoru")
    ax_imp.set_title("Nitelik Önem Düzeyleri (Aktif Model)")
    ax_imp.grid(True, axis="x", linestyle="--", alpha=0.3)
    st.pyplot(fig_imp)

with viz2:
    st.subheader("Gerçek vs. Tahmin Edilen Fiyat (Test Seti)")
    fig_scatter, ax_scatter = plt.subplots(figsize=(6, 4.5), dpi=130)
    ax_scatter.scatter(aktif["y_test"], aktif["y_test_pred"],
                       alpha=0.25, s=12, color="#2563eb", label="Test Örnekleri")
    max_val = max(aktif["y_test"].max(), aktif["y_test_pred"].max())
    ax_scatter.plot([0, max_val], [0, max_val], color="red", linestyle="--",
                    linewidth=2, label="İdeal Doğru (y = x)")
    ax_scatter.set_xlabel("Gerçek Fiyat ($)")
    ax_scatter.set_ylabel("Tahmin ($)")
    ax_scatter.set_title("Gerçek vs. Tahmin Dağılımı")
    ax_scatter.legend(fontsize=8)
    ax_scatter.grid(True, linestyle="--", alpha=0.4)
    st.pyplot(fig_scatter)

st.divider()
st.subheader("Karar Ağacının İlk 3 Seviye Dallanma Yapısı")
st.image("tree_structure.png", width="stretch")