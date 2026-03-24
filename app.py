import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path().resolve().parent

model = joblib.load(BASE_DIR / "/Users/stephenbracco/Projects/Diabetes/calibrated_xgb.pkl")
template = joblib.load(BASE_DIR / "/Users/stephenbracco/Projects/Diabetes/patient_template.pkl")

st.title("30-Day Readmission Risk Calculator")

age = st.selectbox("Age", ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90-99"])
admission_type = st.selectbox("Admission Type", ["Emergency", "Urgent", "Elective"])
time_in_hospital = st.slider("Time in Hospital (Days)", 0, 14, 7)
num_medications = st.slider("Number of Medications", 0, 40, 20)
number_diagnoses = st.slider("Number of Diagnoses", 0, 16, 8)
number_inpatient = st.slider("Prior Inpatient Visits", 0, 10, 5)
number_emergency = st.slider("Prior Emergency Visits", 0, 10, 5)
num_lab_procedures = st.slider("Lab Tests", 0, 150, 75)
num_procedures = st.slider("Procedures (Not Including Lab Tests)", 0, 6, 3)
number_outpatient = st.slider("Outpatient Visits", 0, 10, 5)

insulin = st.selectbox("Insulin", ["No", "Steady", "Up", "Down"])
a1c = st.selectbox("A1C (Blood Sugar) Result", ["None", "Normal", ">7", ">8"])
change = st.selectbox("Medication Change", ["No Change", "Change"])
diabetes_med = st.selectbox("Diabetes Medication", ["No", "Yes"])

diag_1 = st.text_input("Primary Diagnosis Code", "428")
diag_2 = st.text_input("Secondary Diagnosis Code", "250")
diag_3 = st.text_input("Tertiary Diagnosis Code", "276")

if st.button("Predict Readmission Risk"):
    patient = template.copy()

    patient["age"] = age
    patient["Admission_Type"] = admission_type
    patient["time_in_hospital"] = time_in_hospital
    patient["num_medications"] = num_medications
    patient["number_diagnoses"] = number_diagnoses
    patient["number_inpatient"] = number_inpatient
    patient["number_emergency"] = number_emergency
    patient["number_outpatient"] = number_outpatient
    patient["num_lab_procedures"] = num_lab_procedures
    patient["num_procedures"] = num_procedures

    patient["insulin"] = insulin
    patient["A1Cresult"] = a1c
    patient["change"] = change
    patient["diabetesMed"] = diabetes_med

    patient["diag_1"] = diag_1
    patient["diag_2"] = diag_2
    patient["diag_3"] = diag_3

    risk = model.predict_proba(patient)[:, 1][0]

    st.subheader("Predicted 30-Day Readmission Risk")
    st.metric("Risk", f"{risk:.1%}")

    if risk < 0.10:
        st.success("Low risk")
    elif risk < 0.20:
        st.warning("Moderate risk")
    else:
        st.error("High risk")