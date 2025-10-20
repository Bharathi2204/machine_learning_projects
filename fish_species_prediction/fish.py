# iris_app.py
import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---------- Load the dataset ----------
fish =pd.read_csv(r"C:\Users\RAGU\Downloads\Fish.csv")
# ---------- App Title ----------
st.title(" 🐟 Fish Variety Prediction App")
st.write("This simple Streamlit app predicts Fish variety using a Random Forest model.")

# ---------- Show dataset ----------
if st.checkbox("Show dataset"):
    st.dataframe(fish)

# ---------- Data Visualization ----------
st.subheader("Data Visualization")

if st.checkbox("Show pairplot (takes a few seconds)"):
    st.write("### Pairplot of fish Features")
    st.pyplot(sns.pairplot(fish, hue="species").figure)

if st.checkbox("Show correlation heatmap"):
    st.write("### Correlation Heatmap")
    st.pyplot(sns.heatmap(fish.iloc[:, :-1].corr(), annot=True, cmap="coolwarm").figure)

# ---------- Train Model ----------
X = fish.drop('Species',axis=1)
y =fish['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.success(f"Model trained! ✅ Accuracy: {acc*100:.2f}%")

# ---------- Prediction Section ----------
# ---------- User Input Section ----------
st.subheader("Enter Fish Measurements")

Weight = st.number_input("Weight (g)", min_value=float(fish['Weight'].min()), max_value=float(fish['Weight'].max()), value=float(fish['Weight'].mean()))
Length1 = st.number_input("Length1 (Vertical Length in cm)", min_value=float(fish['Length1'].min()), max_value=float(fish['Length1'].max()), value=float(fish['Length1'].mean()))
Length2 = st.number_input("Length2 (Diagonal Length in cm)", min_value=float(fish['Length2'].min()), max_value=float(fish['Length2'].max()), value=float(fish['Length2'].mean()))
Length3 = st.number_input("Length3 (Cross Length in cm)", min_value=float(fish['Length3'].min()), max_value=float(fish['Length3'].max()), value=float(fish['Length3'].mean()))
Height = st.number_input("Height (cm)", min_value=float(fish['Height'].min()), max_value=float(fish['Height'].max()), value=float(fish['Height'].mean()))
Width = st.number_input("Width (cm)", min_value=float(fish['Width'].min()), max_value=float(fish['Width'].max()), value=float(fish['Width'].mean()))


# Predict button
if st.button("Predict"):
    input_data = pd.DataFrame([[Weight, Length1, Length2, Length3, Height, Width]], 
                              columns=['Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width'])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Species: 🐠 **{prediction}**")



