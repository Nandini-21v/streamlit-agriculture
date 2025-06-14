import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Page setup
st.set_page_config(page_title="Crop Recommender", layout="centered")
st.title("🌾 Fine-Tuned Agriculture Crop Recommender")

st.markdown("""
This app helps farmers **fine-tune their agriculture** by recommending the best crop based on:
- Soil nutrients (Nitrogen, Phosphorus, Potassium)
- Soil pH
- Climate (Temperature, Humidity, Rainfall)
""")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Crop_recommendation.csv")
    return df

df = load_data()

# Train model
X = df.drop("label", axis=1)
y = df["label"]
model = RandomForestClassifier()
model.fit(X, y)

# Input form
st.header("📝 Enter Field Conditions")

col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", 0, 140, step=1)
    P = st.number_input("Phosphorus (P)", 0, 140, step=1)
    K = st.number_input("Potassium (K)", 0, 200, step=1)
    pH = st.number_input("Soil pH Level", 3.5, 9.5, step=0.1)

with col2:
    temp = st.number_input("Temperature (°C)", 0.0, 50.0, step=0.1)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, step=0.1)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 300.0, step=0.1)

# Predict
if st.button("🌱 Recommend Best Crop"):
    input_data = [[N, P, K, temp, humidity, pH, rainfall]]
    prediction = model.predict(input_data)[0]
    st.success(f"✅ Based on your inputs, the recommended crop is: **{prediction.upper()}**")

    st.info("Tip: Grow crops suited to soil and climate to ensure maximum yield and sustainability.")
