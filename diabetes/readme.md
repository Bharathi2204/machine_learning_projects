**Project Overview**

This project aims to predict whether a person is diabetic based on medical attributes using a supervised machine learning model. The dataset contains diagnostic measurements, and a Support Vector Machine (SVM) classifier is trained to perform binary classification.

The trained model is saved and deployed using a Streamlit web application for real-time predictions.

**Dataset Description**

Dataset name: Diabetes Dataset

Total records: 768

Features: 8

Target variable: Outcome

0 – Non-diabetic

1 – Diabetic

**Features**

Pregnancies

Glucose

BloodPressure

SkinThickness

Insulin

BMI

DiabetesPedigreeFunction

Age

No missing values were found in the dataset.

**Problem Type**

Supervised Learning

Binary Classification

**Tools and Libraries Used**

Python

NumPy

Pandas

Scikit-learn

Streamlit

Pickle

**Data Analysis and Preprocessing**

Loaded and explored dataset using Pandas

Checked data types, shape, and statistical summary

Verified absence of missing values

Analyzed class distribution and feature correlations

**Model Building**

Algorithm used: Support Vector Machine (SVM)

Kernel: Linear

**Data split:**

Training set: 80%

Testing set: 20%

Stratified split used to maintain class balance

**Model Evaluation**

The model was evaluated using the following metrics:

Accuracy

Precision

Recall

Confusion Matrix

Performance

Training Accuracy: ~77.7%

Test Accuracy: ~76.6%

Precision: ~57.4%

Recall: ~70.4%

The model demonstrates reasonable performance in identifying diabetic patients.

**Model Saving**

The trained model and scaler were saved using Pickle:

diabetes_model.pkl

scaler.pkl

These files are used for deployment in the Streamlit application.

**Streamlit Web Application**

A Streamlit-based web app was created to allow users to input medical details and get instant diabetes predictions.

App Features

User-friendly input fields for all medical parameters

Uses the saved scaler and trained model

Displays prediction result as diabetic or non-diabetic

**Conclusion**

This project demonstrates an end-to-end machine learning workflow including data analysis, model training, evaluation, saving the model, and deploying it as a web application. It provides a practical example of applying machine learning in the healthcare domain.
