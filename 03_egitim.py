import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_csv("araba_temiz.csv")

X = df.drop(columns=["Price"])
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

target_sutunlar = ["Manufacturer", "Model"]
kategorik_sutunlar = ["Category", "Leather interior", "Fuel type", "Gear box type", "Drive wheels", "Wheel", "Turbo"]
sayisal_sutunlar = ["Levy", "Prod. year", "Engine volume", "Mileage", "Cylinders", "Airbags"]

preprocessor = ColumnTransformer(transformers=[
    ("num", "passthrough", sayisal_sutunlar),
    ("target", TargetEncoder(smooth="auto", cv=5, random_state=42, target_type="continuous"), target_sutunlar),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), kategorik_sutunlar)
])

unpruned_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", DecisionTreeRegressor(criterion="friedman_mse", random_state=42))
])
unpruned_pipeline.fit(X_train, y_train)

y_train_pred_unpruned = unpruned_pipeline.predict(X_train)
y_test_pred_unpruned = unpruned_pipeline.predict(X_test)
unpruned_tree = unpruned_pipeline.named_steps["regressor"]

#Budanmıs
best_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", DecisionTreeRegressor(
        max_depth=16,
        min_samples_split=15,
        min_samples_leaf=5,
        criterion="friedman_mse",
        random_state=42
    ))
])
best_model.fit(X_train, y_train)
pruned_tree = best_model.named_steps["regressor"]

y_train_pred_pruned = best_model.predict(X_train)
y_test_pred_pruned = best_model.predict(X_test)


karsilastirma_df = pd.DataFrame({
    "Metrik": [
        "Ağaç Derinliği (Depth)",
        "Toplam Yaprak Sayısı (Leaves)",
        "Eğitim R² Skoru (%)",
        "Test R² Skoru (%)",
        "Test MAE (Ortalama Hata $)",
        "Test RMSE (Kök Kare Hata $)"
    ],
    "Budanmamış Ağaç (Overfit)": [
        unpruned_tree.get_depth(),
        unpruned_tree.get_n_leaves(),
        round(r2_score(y_train, y_train_pred_unpruned) * 100, 2),
        round(r2_score(y_test, y_test_pred_unpruned) * 100, 2),
        round(mean_absolute_error(y_test, y_test_pred_unpruned), 2),
        round(np.sqrt(mean_squared_error(y_test, y_test_pred_unpruned)), 2)
    ],
    "Budanmış Ağaç (Optimize)": [
        pruned_tree.get_depth(),
        pruned_tree.get_n_leaves(),
        round(r2_score(y_train, y_train_pred_pruned) * 100, 2),
        round(r2_score(y_test, y_test_pred_pruned) * 100, 2),
        round(mean_absolute_error(y_test, y_test_pred_pruned), 2),
        round(np.sqrt(mean_squared_error(y_test, y_test_pred_pruned)), 2)
    ]
})

print("\n BUDAMA ÖNCESİ VE SONRASI KARŞILAŞTIRMA TABLOSU ")
print(karsilastirma_df.to_string(index=False))

#kayıt
joblib.dump(best_model, "car_price_dt_model.pkl")
print("\n'car_price_dt_model.pkl' başarıyla güncellendi!")