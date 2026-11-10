"""Streamlit dashboard for the telecom churn model."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("artifacts/telecom_churn_model.joblib")

st.set_page_config(page_title="Telecom Churn Predictor", page_icon="📊", layout="wide")
st.title("Telecom Customer Churn Predictor")
st.caption("A demonstration dashboard powered by the trained Random Forest pipeline.")

if not MODEL_PATH.exists():
    st.error("Trained model not found. First run: python train_model.py")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.subheader("Enter customer details")
st.info("Enter service usage and account details. Phone number is not used by the model.")

with st.form("customer_form"):
    left, middle, right = st.columns(3)
    with left:
        state = st.selectbox("State", options=[
            "AK","AL","AR","AZ","CA","CO","CT","DC","DE","FL","GA","HI","IA","ID","IL","IN","KS","KY","LA","MA","MD","ME","MI","MN","MO","MS","MT","NC","ND","NE","NH","NJ","NM","NV","NY","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VA","VT","WA","WI","WV","WY"
        ], index=16)
        account_length = st.number_input("Account length (months)", min_value=1, max_value=300, value=100)
        area_code = st.selectbox("Area code", [408, 415, 510], index=1)
        international_plan = st.selectbox("International plan", ["no", "yes"])
        voice_mail_plan = st.selectbox("Voice mail plan", ["no", "yes"])
        number_vmail_messages = st.number_input("Voice mail messages", min_value=0, max_value=100, value=0)
    with middle:
        total_day_minutes = st.number_input("Day minutes", min_value=0.0, max_value=1000.0, value=180.0)
        total_day_calls = st.number_input("Day calls", min_value=0, max_value=500, value=100)
        total_day_charge = st.number_input("Day charge", min_value=0.0, max_value=200.0, value=30.6)
        total_eve_minutes = st.number_input("Evening minutes", min_value=0.0, max_value=1000.0, value=200.0)
        total_eve_calls = st.number_input("Evening calls", min_value=0, max_value=500, value=100)
        total_eve_charge = st.number_input("Evening charge", min_value=0.0, max_value=200.0, value=17.0)
        total_night_minutes = st.number_input("Night minutes", min_value=0.0, max_value=1000.0, value=200.0)
    with right:
        total_night_calls = st.number_input("Night calls", min_value=0, max_value=500, value=100)
        total_night_charge = st.number_input("Night charge", min_value=0.0, max_value=200.0, value=9.0)
        total_intl_minutes = st.number_input("International minutes", min_value=0.0, max_value=100.0, value=10.0)
        total_intl_calls = st.number_input("International calls", min_value=0, max_value=100, value=4)
        total_intl_charge = st.number_input("International charge", min_value=0.0, max_value=50.0, value=2.7)
        customer_service_calls = st.number_input("Customer service calls", min_value=0, max_value=50, value=1)

    submitted = st.form_submit_button("Predict churn risk", type="primary")

if submitted:
    customer = pd.DataFrame([{
        "state": state,
        "account length": account_length,
        "area code": area_code,
        "international plan": international_plan,
        "voice mail plan": voice_mail_plan,
        "number vmail messages": number_vmail_messages,
        "total day minutes": total_day_minutes,
        "total day calls": total_day_calls,
        "total day charge": total_day_charge,
        "total eve minutes": total_eve_minutes,
        "total eve calls": total_eve_calls,
        "total eve charge": total_eve_charge,
        "total night minutes": total_night_minutes,
        "total night calls": total_night_calls,
        "total night charge": total_night_charge,
        "total intl minutes": total_intl_minutes,
        "total intl calls": total_intl_calls,
        "total intl charge": total_intl_charge,
        "customer service calls": customer_service_calls,
    }])

    probability = float(model.predict_proba(customer)[0][1])
    prediction = int(model.predict(customer)[0])
    st.divider()
    st.subheader("Prediction")
    metric_col, result_col = st.columns(2)
    metric_col.metric("Estimated churn probability", f"{probability:.1%}")
    if prediction:
        result_col.error("Model classification: Higher churn risk")
    else:
        result_col.success("Model classification: Lower churn risk")
    st.progress(probability)
    st.caption("This is a machine-learning estimate for demonstration, not a certainty or a substitute for customer-level business review.")
