import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="CKD Prediction App", layout="wide")
st.title("🧠 Chronic Kidney Disease Prediction App")

# ---------------- Load Models ----------------
@st.cache_resource
def load_models():
    dct = joblib.load("decision_tree_model.pkl")
    rf = joblib.load("random_forest_model.pkl")
    knn = joblib.load("knn_model.pkl")
    return {"Decision Tree": dct, "Random Forest": rf, "KNN Classifier": knn}

models = load_models()

# ---------------- File Upload ----------------
uploaded_file = st.file_uploader("📂 Upload your preprocessed CSV (processed_train.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Uploaded Data Preview")
    st.dataframe(df.head())

    # ---------------- Model Selection ----------------
    model_name = st.selectbox("Select Model", list(models.keys()))
    model = models[model_name]

    if st.button("🚀 Predict CKD Classification"):
        # Prepare features (all columns except CKD_Classification if exists)
        feature_columns = [col for col in df.columns if col != "CKD_Classification"]
        X = df[feature_columns]

        # Make prediction
        y_pred = model.predict(X)
        df["Predicted_CKD_Classification"] = y_pred

        # Display predictions
        st.subheader("✅ Prediction Results")
        st.dataframe(df.head(10))

        # Show metrics if target exists
        if "CKD_Classification" in df.columns:
            y_true = df["CKD_Classification"]
            acc = accuracy_score(y_true, y_pred)
            pre = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)

            st.subheader("📊 Model Evaluation Metrics")
            st.write(f"**Accuracy:** {acc:.3f}")
            st.write(f"**Precision:** {pre:.3f}")
            st.write(f"**Recall:** {rec:.3f}")
            st.write(f"**F1 Score:** {f1:.3f}")

            st.subheader("📈 Confusion Matrix")
            cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
            st.dataframe(cm_df)

        # Download predictions
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Predictions as CSV",
            data=csv,
            file_name=f"{model_name}_predictions.csv",
            mime="text/csv"
        )
