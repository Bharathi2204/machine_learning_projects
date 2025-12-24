**Project Overview**

This project focuses on predicting whether a loan application will be approved or rejected based on applicant details such as income, education, credit history, and property area. A machine learning classification model is trained using historical loan data and deployed as a Streamlit web application for real-time predictions.

**Dataset**

Total records: 614

Features include:

Gender

Marital Status

Dependents

Education

Self Employed

Applicant Income

Coapplicant Income

Loan Amount

Loan Amount Term

Credit History

Property Area

Target variable:

Loan_Status (1 = Approved, 0 = Rejected)

**Data Preprocessing**

Removed missing values from the dataset

Converted categorical features into numerical format using label replacement

Handled special values such as "3+" dependents

Dropped unnecessary columns like Loan_ID

Performed exploratory data analysis using count plots

**Model Building**

Algorithm used: Support Vector Machine (SVM) with linear kernel

Data split into training and testing sets using stratified sampling

Model trained on processed numerical data

**Model Evaluation**

Training accuracy: ~80%

Test accuracy: ~83%

Precision and recall used to evaluate classification performance

Confusion matrix used to analyze prediction results

**Model Deployment**

Trained model saved using Pickle

Streamlit web application created to allow users to input applicant details and predict loan approval status in real time

**Technologies Used**

Python

Pandas, NumPy

Seaborn

Scikit-learn

Support Vector Machine (SVM)

Streamlit

Pickle

**Conclusion**

This project demonstrates the practical application of machine learning for loan approval prediction using real-world financial data. By performing effective data preprocessing, feature encoding, and model training with Support Vector Machine (SVM), the system is able to achieve reliable prediction performance. The integration of the trained model into a Streamlit web application allows users to easily input applicant details and receive instant loan approval decisions. This project highlights the complete machine learning pipeline from data analysis to model deployment.
