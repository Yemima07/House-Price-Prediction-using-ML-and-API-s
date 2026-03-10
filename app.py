# ==========================================
# INDIA HOUSING PRICE PREDICTION API
# ==========================================

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# ------------------------------------------
# 1️⃣ Create FastAPI App
# ------------------------------------------

app = FastAPI(
    title="India Housing Price Prediction API",
    version="1.0"
)

# ------------------------------------------
# 2️⃣ Load Trained Pipeline Model
# ------------------------------------------

MODEL_PATH = "india_house_pipeline.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file '{MODEL_PATH}' not found. "
        f"Make sure it exists in the same folder as app.py"
    )

model = joblib.load(MODEL_PATH)

print("✅ Model loaded successfully")


# ------------------------------------------
# 3️⃣ Request Schema (Input Format)
# ------------------------------------------

class HouseData(BaseModel):
    City: str
    BHK: int
    Property_Type: str
    Furnished_Status: str
    Size_in_SqFt: float
    Age_of_Property: int


# ------------------------------------------
# 4️⃣ Health Check Endpoint
# ------------------------------------------

@app.get("/")
def home():
    return {"message": "India Housing API is running successfully 🚀"}


@app.get("/health")
def health():
    return {"status": "healthy"}


# ------------------------------------------
# 5️⃣ Prediction Endpoint
# ------------------------------------------

@app.post("/predict")
def predict(data: HouseData):

    # Convert input to DataFrame
    input_df = pd.DataFrame([data.model_dump()])

    # Predict current price
    current_price = model.predict(input_df)[0]

    # Future price after 5 years (6% growth)
    annual_growth = 0.06
    years = 5

    future_price = current_price * ((1 + annual_growth) ** years)

    return {
        "current_price_lakhs": round(current_price, 2),
        "future_price_5_years_lakhs": round(future_price, 2),
        "growth_rate_used": "6% annually",
        "model_used": "XGBoost Pipeline"
    }
