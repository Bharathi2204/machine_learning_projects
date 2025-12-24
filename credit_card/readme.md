**Project Overview**

This project focuses on detecting fraudulent credit card transactions using machine learning classification techniques. Due to the highly imbalanced nature of fraud detection data, appropriate preprocessing and sampling techniques were applied to build effective and reliable models.

The objective is to classify transactions as:

0 – Legitimate transaction

1 – Fraudulent transaction

**Dataset Description**

Total transactions: 284,807

Fraudulent transactions: 492

Legitimate transactions: 284,315

Target variable: Class

Missing values: None

The dataset contains anonymized numerical features (V1 to V28) obtained using PCA, along with Time, Amount, and the target column Class.

Problem Type

Supervised learning

Binary classification

Highly imbalanced dataset

**Tools and Libraries Used**

Python

NumPy

Pandas

Matplotlib

Seaborn

Scikit-learn

**Data Preprocessing**

Verified data integrity and absence of missing values

Visualized class imbalance and transaction patterns

Scaled Amount and Time features using StandardScaler

Handled class imbalance using undersampling, creating a balanced dataset with equal fraud and legitimate transactions

**Exploratory Data Analysis**

Analyzed transaction distribution across time and amount

Compared statistical characteristics of fraudulent and legitimate transactions

Used scatter plots and bar charts to visualize patterns

**Model Development**

The following machine learning models were trained and evaluated:

Logistic Regression (with class balancing)

Decision Tree Classifier

Random Forest Classifier

K-Nearest Neighbors (KNN)

The dataset was split into training and testing sets using stratified sampling.

**Model Evaluation**

Models were evaluated using:

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

**Key Observation**

Logistic Regression achieved the best balance between precision and recall on test data

Tree-based models showed perfect training accuracy but signs of overfitting

Undersampling improved fraud detection performance significantly

**Conclusion**

This project demonstrates effective fraud detection using machine learning on a highly imbalanced dataset. Proper preprocessing, sampling strategies, and model evaluation helped achieve reliable classification performance, especially in identifying fraudulent transactions.
