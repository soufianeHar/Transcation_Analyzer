import streamlit as st
import pandas as pd
import joblib

# Load trained model
@st.cache_data
def load_model():
    return joblib.load("outputs/fraud_model.joblib")

model = load_model()

st.title("💳 Fraud Risk Checker")
st.write("Enter transaction details below:")

# Form inputs
amt = st.number_input("Transaction Amount", value=80.0)
hour = st.slider("Transaction Hour", 0, 23, value=14)
category = st.selectbox("Category", ["shopping_pos", "gas_transport", "grocery_pos", "travel", "shopping_net", "entertainment"])
job = st.selectbox("Customer Job", ["teacher", "engineer", "nurse", "manager", "developer", "driver", "other"])
city_pop = st.number_input("City Population", value=50000)

# Submit button
if st.button("Check Fraud Risk"):
    input_df = pd.DataFrame([{
        "amt": amt,
        "hour": hour,
        "category": category,
        "job": job,
        "city_pop": city_pop
    }])
    
    # Predict
    prob = model.predict_proba(input_df)[0][1]
    
    # Risk level logic
    if prob < 0.2:
        risk = "✅ Not Fraud"
        color = "green"
    elif prob < 0.7:
        risk = "⚠️ Suspicious"
        color = "orange"
    else:
        risk = "🔴 Fraud Likely"
        color = "red"
    
    st.markdown(f"### Probability of Fraud: **{prob:.2f}**")
    st.markdown(f"### Risk Level: <span style='color:{color}; font-size:22px'>{risk}</span>", unsafe_allow_html=True)
