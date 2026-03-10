# ======================================================
# INDIA HOUSING PRICE PREDICTION - COMPLETE TRAINING
# ======================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor


# ------------------------------------------------------
# 1️⃣ Load Dataset
# ------------------------------------------------------

print("📂 Loading Dataset...")

df = pd.read_csv("india_housing_prices.csv")

print("✅ Data Loaded Successfully")
print("Shape:", df.shape)
print("\nFirst 5 Rows:\n", df.head())


# ------------------------------------------------------
# 2️⃣ Basic Cleaning
# ------------------------------------------------------

if "ID" in df.columns:
    df.drop(columns=["ID"], inplace=True)

df.drop_duplicates(inplace=True)

print("✅ Data Cleaning Completed")


# ------------------------------------------------------
# 3️⃣ EDA (Save Plots Professionally)
# ------------------------------------------------------

os.makedirs("plots", exist_ok=True)

# Distribution Plot
plt.figure(figsize=(10,4))
sns.histplot(df["Price_in_Lakhs"], kde=True)
plt.title("Distribution of House Prices")
plt.savefig("plots/price_distribution.png")
plt.close()

# Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(
    df.select_dtypes(include=np.number).corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.savefig("plots/correlation_heatmap.png")
plt.close()

print("✅ Plots Saved in 'plots' folder")


# ------------------------------------------------------
# 4️⃣ Feature & Target Separation
# ------------------------------------------------------

X = df.drop("Price_in_Lakhs", axis=1)
y = df["Price_in_Lakhs"]

num_cols = X.select_dtypes(include=["int64","float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

print("\nNumerical Columns:", num_cols.tolist())
print("Categorical Columns:", cat_cols.tolist())


# ------------------------------------------------------
# 5️⃣ Preprocessing Pipeline
# ------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
    ]
)


# ------------------------------------------------------
# 6️⃣ Model Pipeline (XGBoost)
# ------------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        ))
    ]
)


# ------------------------------------------------------
# 7️⃣ Train-Test Split
# ------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("✅ Data Split Completed")


# ------------------------------------------------------
# 8️⃣ Train Model
# ------------------------------------------------------

print("🚀 Training Model...")

pipeline.fit(X_train, y_train)

print("✅ Model Training Completed")


# ------------------------------------------------------
# 9️⃣ Model Evaluation
# ------------------------------------------------------

y_pred = pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("\n📊 Model Performance")
print("----------------------------")
print("R2 Score :", round(r2, 4))
print("RMSE     :", round(rmse, 4))
print("MAE      :", round(mae, 4))


# ------------------------------------------------------
# 🔟 Save Complete Pipeline
# ------------------------------------------------------

joblib.dump(pipeline, "india_house_pipeline.pkl")

print("\n🎉 Pipeline Saved Successfully as 'india_house_pipeline.pkl'")
