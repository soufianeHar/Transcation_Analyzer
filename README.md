# 🔍 Transaction Analyzer – Fraud Detection ML App

An end-to-end machine learning pipeline that detects fraud in financial transactions, with a clean Streamlit UI for real-time prediction.

## 📦 Features
- ML pipeline using XGBoost
- Handles class imbalance and explains risk levels
- Streamlit interface with risk scoring:
  - ✅ Not Fraud
  - ⚠️ Suspicious
  - 🔴 Fraud Likely

## 📊 Technologies Used
- Python · pandas · scikit-learn · XGBoost · Streamlit

## 🚀 How to Run
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py