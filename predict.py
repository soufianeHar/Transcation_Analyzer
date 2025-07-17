import joblib
import pandas as pd

# Load trained model
model = joblib.load("outputs/fraud_model.joblib")

# Sample input
sample = pd.DataFrame([{
    "amt": 80.0,
    "hour": 14,
    "category": "shopping_pos",
    "job": "teacher",
    "city_pop": 50000
}])

# Get fraud probability
prob = model.predict_proba(sample)[0][1] 
# Define custom risk levels
if prob < 0.2:
    risk = "✅ Not Fraud"
elif prob < 0.7:
    risk = "⚠️ Suspicious"
else:
    risk = "🔴 Fraud Likely"

print(f"🔎 Probability of Fraud: {prob:.2f}")
print(f"🧠 Risk Level: {risk}")
