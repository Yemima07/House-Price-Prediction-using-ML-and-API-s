import streamlit as st
import pandas as pd
import requests

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(layout="wide")
st.title("🏠 Residential Property Price Forecasting System (Production Mode)")

# ======================================
# LOAD DATA ONLY (NO MODEL HERE)
# ======================================
df = pd.read_csv("india_housing_prices.csv")

if "ID" in df.columns:
    df = df.drop(columns=["ID"])

# ======================================
# API URL
# ======================================
API_URL = "http://127.0.0.1:8000/predict"

# =====================================================
# 🔎 SIDEBAR FILTER SECTION
# =====================================================
st.sidebar.header("🔎 Property Filters")

filter_df = df.copy()
filter_values = {}

for col in df.columns:
    if col == "Price_in_Lakhs":
        continue

    if df[col].dtype == "object":
        options = sorted(df[col].dropna().unique().tolist())
        selected = st.sidebar.selectbox(
            f"Select {col}",
            ["All"] + options
        )
        filter_values[col] = selected
    else:
        min_val = float(df[col].min())
        max_val = float(df[col].max())

        selected = st.sidebar.slider(
            f"{col} Range",
            min_val,
            max_val,
            (min_val, max_val)
        )
        filter_values[col] = selected

# APPLY FILTERS
for col, val in filter_values.items():
    if isinstance(val, tuple):
        filter_df = filter_df[
            (filter_df[col] >= val[0]) &
            (filter_df[col] <= val[1])
        ]
    elif val != "All":
        filter_df = filter_df[filter_df[col] == val]

# =====================================================
# DISPLAY FILTERED RESULTS
# =====================================================
st.header("📋 Filtered Property Results")

if filter_df.empty:
    st.warning("❌ No properties found with selected filters.")
else:
    st.success(f"✅ {len(filter_df)} Properties Found")
    st.dataframe(filter_df)

    selected_index = st.selectbox(
        "Select Property Index From Filtered Results",
        filter_df.index
    )

    if st.button("Predict Selected Property (Filter Based)"):

        selected_property = df.loc[[selected_index]]
        input_df = selected_property.drop("Price_in_Lakhs", axis=1)

        payload = input_df.iloc[0].to_dict()

        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()

            st.subheader("🏠 Selected Property Details")
            st.dataframe(selected_property)

            st.success(f"💰 Current Price: {result['current_price_lakhs']} Lakhs")
            st.success(f"📈 Price After 5 Years: {result['future_price_5_years_lakhs']} Lakhs")
        else:
            st.error("API Error ❌")

# =====================================================
# 🎯 INDEX BASED SECTION
# =====================================================
st.header("🎯 Direct Prediction Using Index (Independent)")

index_input = st.number_input(
    "Enter Property Index",
    min_value=int(df.index.min()),
    max_value=int(df.index.max()),
    step=1
)

if st.button("Predict Using Index Only"):

    selected_property = df.loc[[index_input]]
    st.subheader("🏠 Property Details (Index Based)")
    st.dataframe(selected_property)

    input_df = selected_property.drop("Price_in_Lakhs", axis=1)
    payload = input_df.iloc[0].to_dict()

    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        result = response.json()

        st.success(f"💰 Current Price: {result['current_price_lakhs']} Lakhs")
        st.success(f"📈 Price After 5 Years: {result['future_price_5_years_lakhs']} Lakhs")
    else:
        st.error("API Error ❌")
