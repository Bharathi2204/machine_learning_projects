import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="CKD Single Prediction", layout="centered")
st.title("🧠 Chronic Kidney Disease Prediction - Single Patient")

# ---------------- Load Models ----------------
@st.cache_resource
def load_models():
    dct = joblib.load("decision_tree_model.pkl")
    rf = joblib.load("random_forest_model.pkl")
    knn = joblib.load("knn_model.pkl")
    return {"Decision Tree": dct, "Random Forest": rf, "KNN Classifier": knn}

models = load_models()

# ---------------- User Input ----------------
st.subheader("Enter Patient Details:")

# Numeric Inputs (default values reflect a typical CKD patient)
Age = st.number_input("Age", min_value=1, max_value=120, value=60)
Blood_Pressure = st.number_input("Blood Pressure", min_value=50, max_value=200, value=90)
Specific_Gravity = st.number_input("Specific Gravity", min_value=1.0, max_value=1.03, value=1.02, step=0.01)
Albumin = st.number_input("Albumin", min_value=0, max_value=5, value=3)
Sugar = st.number_input("Sugar", min_value=0, max_value=5, value=2)
Blood_Glucose_Random = st.number_input("Blood Glucose Random", min_value=50, max_value=500, value=120)
Blood_Urea = st.number_input("Blood Urea", min_value=10, max_value=200, value=80)
Serum_Creatinine = st.number_input("Serum Creatinine", min_value=0.1, max_value=20.0, value=5.0, step=0.1)
Sodium = st.number_input("Sodium", min_value=100, max_value=200, value=135)
Potassium = st.number_input("Potassium", min_value=2.0, max_value=10.0, value=5.0, step=0.1)
Hemoglobin = st.number_input("Hemoglobin", min_value=3.0, max_value=20.0, value=8.0, step=0.1)
Packed_Cell_Volume = st.number_input("Packed Cell Volume", min_value=10, max_value=60, value=28)
White_Blood_Cell = st.number_input("White Blood Cell Count", min_value=2000, max_value=20000, value=10000)
Red_Blood_Cell = st.number_input("Red Blood Cell Count", min_value=2.0, max_value=10.0, value=3.5, step=0.1)

# Categorical Inputs (exact mapping)
def binary_input(label, options):
    return st.selectbox(label, options)

Red_Blood_Cells = 1 if binary_input("Red Blood Cells", ["normal", "abnormal"]) == "normal" else 0
Pus_Cell = 1 if binary_input("Pus Cell", ["normal", "abnormal"]) == "normal" else 0
Pus_Cell_Clumps = 1 if binary_input("Pus Cell Clumps", ["present", "notpresent"]) == "present" else 0
Bacteria = 1 if binary_input("Bacteria", ["present", "notpresent"]) == "present" else 0
Hypertension = 1 if binary_input("Hypertension", ["yes", "no"]) == "yes" else 0
Diabetes_Mellitus = 1 if binary_input("Diabetes Mellitus", ["yes", "no"]) == "yes" else 0
Coronary_Artery_Disease = 1 if binary_input("Coronary Artery Disease", ["yes", "no"]) == "yes" else 0
Appetite = 1 if binary_input("Appetite", ["good", "poor"]) == "good" else 0
Pedal_Edema = 1 if binary_input("Pedal Edema", ["yes", "no"]) == "yes" else 0
Anemia = 1 if binary_input("Anemia", ["yes", "no"]) == "yes" else 0

# ---------------- Model Selection ----------------
model_name = st.selectbox("Select Model", list(models.keys()))
model = models[model_name]

# ---------------- Prediction ----------------
if st.button("🚀 Predict CKD Classification"):
    input_df = pd.DataFrame({
        "Age":[Age],
        "Blood_Pressure":[Blood_Pressure],
        "Specific_Gravity":[Specific_Gravity],
        "Albumin":[Albumin],
        "Sugar":[Sugar],
        "Red_Blood_Cells":[Red_Blood_Cells],
        "Pus_Cell":[Pus_Cell],
        "Pus_Cell_Clumps":[Pus_Cell_Clumps],
        "Bacteria":[Bacteria],
        "Blood_Glucose_Random":[Blood_Glucose_Random],
        "Blood_Urea":[Blood_Urea],
        "Serum_Creatinine":[Serum_Creatinine],
        "Sodium":[Sodium],
        "Potassium":[Potassium],
        "Hemoglobin":[Hemoglobin],
        "Packed_Cell_Volume":[Packed_Cell_Volume],
        "White_Blood_Cell":[White_Blood_Cell],
        "Red_Blood_Cell":[Red_Blood_Cell],
        "Hypertension":[Hypertension],
        "Diabetes_Mellitus":[Diabetes_Mellitus],
        "Coronary_Artery_Disease":[Coronary_Artery_Disease],
        "Appetite":[Appetite],
        "Pedal_Edema":[Pedal_Edema],
        "Anemia":[Anemia]
    })

    prediction = model.predict(input_df)[0]

    st.subheader("✅ Prediction Result")
    if prediction == 1:
        st.success("The patient is predicted to have CKD.")
    else:
        st.info("The patient is predicted to be Healthy (No CKD).")
