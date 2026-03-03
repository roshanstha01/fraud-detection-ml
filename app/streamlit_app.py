import streamlit as st
import pandas as pd
from src.predict import predict_one

st.title("Fraud Detection Demo (Random Forest)")

# Load dataset for demo (so you can pick real transactions)
df = pd.read_csv("data/creditcard.csv")

st.write("Pick a real transaction row from the dataset to test predictions.")

row_id = st.number_input("Row number", min_value=0, max_value=len(df)-1, value=0, step=1)
row = df.iloc[int(row_id)]

# Build features dict
cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
features = {c: float(row[c]) for c in cols}

st.subheader("Selected Transaction (preview)")
st.write({k: features[k] for k in ["Time", "Amount", "V1", "V2", "V3"]})
st.write(f"True Class (from dataset): {'FRAUD (1)' if int(row['Class'])==1 else 'NORMAL (0)'}")

if st.button("Predict"):
    out = predict_one(features)
    st.subheader("Result")
    st.write(f"Fraud probability: **{out['fraud_probability']:.4f}**")
    st.write(f"Threshold: **{out['threshold']:.2f}**")
    st.write("Prediction: " + ("**FRAUD**" if out["prediction"] == 1 else "**NOT FRAUD**"))