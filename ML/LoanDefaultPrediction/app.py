import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model and preprocessing objects
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Streamlit App
st.title("📋 Loan Default Prediction App")

# Input fields
st.header("Enter Applicant Details")
income = st.number_input("🔹 Enter Monthly Income (in ₹):", min_value=1)  # Monthly income in Rupees
loan_amount = st.number_input("🔹 Enter Loan Amount (in ₹):", min_value=1)  # Loan amount in Rupees
credit_score = st.number_input("🔹 Enter Credit Score:", min_value=300, max_value=850)

# Determine credit history based on credit score
credit_history = "Good" if credit_score >= 700 else "Bad"
st.write(f"🔹 Credit History: {credit_history}")

# Define the correct order of columns
feature_names = [
    'status', 'duration', 'credit_history', 'purpose', 'credit_amount',
    'savings_account', 'employment_since', 'installment_rate',
    'personal_status', 'other_debtors', 'residence_since', 'property',
    'age', 'other_installment_plans', 'housing', 'existing_credits', 'job',
    'num_dependents', 'telephone', 'foreign_worker'
]

# Create an empty DataFrame with the correct columns
input_data = pd.DataFrame(columns=feature_names)

# Calculate installment rate
duration = 24  # Duration in months (default value)
monthly_installment = loan_amount / duration  # Monthly repayment amount
installment_rate = monthly_installment / income if income > 0 else 0

# Fill default values
input_data.loc[0] = [
    'A12',  # status (less risky than A11)
    24,     # duration (longer loan term reduces monthly burden)
    'A30' if credit_score >= 700 else 'A31',  # credit_history
    'A40',  # purpose (car loans are more common but less favorable than business loans)
    loan_amount,  # credit_amount
    'A61',  # savings_account (no savings)
    'A71',  # employment_since (<1 year of employment)
    installment_rate,  # installment_rate (calculated above)
    'A93',  # personal_status (male single)
    'A101',  # other_debtors (no co-applicant)
    5,      # residence_since
    'A121',  # property (no real estate ownership)
    30,     # age
    'A141',  # other_installment_plans (no other plans)
    'A151',  # housing (renting)
    1,      # existing_credits (one existing credit)
    'A172',  # job (skilled employee)
    0,      # num_dependents
    'A191',  # telephone (no registered phone number)
    'A201'  # foreign_worker
]

# Encode categorical variables
categorical_cols = [
    'status', 'credit_history', 'purpose', 'savings_account', 'employment_since',
    'personal_status', 'other_debtors', 'property', 'other_installment_plans', 'housing', 'job',
    'telephone', 'foreign_worker'
]
for col in categorical_cols:
    input_data[col] = label_encoders[col].transform(input_data[col])

# Scale numerical features
numerical_cols = ['duration', 'credit_amount', 'installment_rate', 'residence_since', 'age', 'existing_credits', 'num_dependents']
input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

# Predict
if st.button("📊 Predict"):
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    st.write(f"Probabilities: Approved={probabilities[1]:.2f}, Rejected={probabilities[0]:.2f}")
    if prediction == 0:
        st.error("Result: Rejected ❌")
    else:
        st.success("Result: Approved ✅")