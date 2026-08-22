import streamlit as st
import joblib
import numpy as np

# Set page config
st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

# Title
st.title("🏦 Loan Approval Prediction System")
st.write("Enter the applicant's details below to check if their loan is likely to be approved.")

# Load model artifacts
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("label_encoders.pkl")
    models_loaded = True
except FileNotFoundError:
    models_loaded = False
    st.error(
        "Model files not found! Please run `train.py` first to generate model.pkl, scaler.pkl, and label_encoders.pkl.")

# Create input form if models are loaded
if models_loaded:
    with st.form("loan_form"):
        st.subheader("Applicant Details")

        col1, col2 = st.columns(2)

        with col1:
            no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=20, value=0)
            education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed?", ["Yes", "No"])
            cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=750)

        with col2:
            income_annum = st.number_input("Annual Income (₹)", min_value=0, value=500000, step=50000)
            loan_amount = st.number_input("Requested Loan Amount (₹)", min_value=0, value=1500000, step=100000)
            loan_term = st.number_input("Loan Term (Years)", min_value=1, max_value=30, value=15)

        st.subheader("Asset Details (₹)")
        col3, col4 = st.columns(2)

        with col3:
            residential_assets = st.number_input("Residential Assets Value", min_value=0, value=0, step=100000)
            commercial_assets = st.number_input("Commercial Assets Value", min_value=0, value=0, step=100000)

        with col4:
            luxury_assets = st.number_input("Luxury Assets Value", min_value=0, value=0, step=50000)
            bank_assets = st.number_input("Bank Assets Value", min_value=0, value=0, step=50000)

        submit_button = st.form_submit_button("Predict Loan Status")

    # Handle Prediction
    if submit_button:
        try:
            # 1. Encode Categorical Data
            edu_encoded = encoders['education'].transform([education])[0]
            self_emp_encoded = encoders['self_employed'].transform([self_employed])[0]

            # 2. Feature Engineering (Must match training logic)
            total_assets = residential_assets + commercial_assets + luxury_assets + bank_assets
            loan_to_income = loan_amount / (income_annum + 1)

            # 3. Create Feature Array
            features = np.array([[
                no_of_dependents, edu_encoded, self_emp_encoded, income_annum,
                loan_amount, loan_term, cibil_score, residential_assets,
                commercial_assets, luxury_assets, bank_assets,
                total_assets, loan_to_income
            ]])

            # 4. Scale and Predict
            scaled_features = scaler.transform(features)
            prediction = model.predict(scaled_features)[0]
            probability = model.predict_proba(scaled_features)[0]

            status = "Approved" if prediction == 0 else "Rejected"
            prob_approved = probability[0]
            prob_rejected = probability[1]

            # 5. Display Results
            st.divider()
            if status == "Approved":
                st.success(f"✅ Prediction: **{status}**")
                st.write(f"Model Confidence: **{prob_approved:.1%}**")
            else:
                st.error(f"❌ Prediction: **{status}**")
                st.write(f"Model Confidence: **{prob_rejected:.1%}**")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")