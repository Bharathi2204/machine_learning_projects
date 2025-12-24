**Bank Note Authentication Using Machine Learning**
**Project Overview**

This project focuses on identifying whether a banknote is genuine or fake using machine learning techniques. The classification is performed using statistical features extracted from banknote images. The model predicts the authenticity of a banknote based on these features.

This is a binary classification problem:

0 – Genuine banknote

1 – Fake banknote

**Dataset Description**

Total records: 1372

Total features: 5

Missing values: None

**Features**

variance: Measures the spread of pixel intensity values in the image

skewness: Indicates asymmetry in the pixel value distribution

curtosis: Represents the peakedness or flatness of the distribution

entropy: Measures randomness or disorder in the image

class: Target variable indicating authenticity

**Problem Type**

Supervised learning

Binary classification

**Tools and Libraries Used**

Python

NumPy

Pandas

Matplotlib

Seaborn

Scikit-learn

**Exploratory Data Analysis**

Verified dataset dimensions and data types

Confirmed absence of null values

Analyzed class distribution

Visualized feature relationships using scatter plots and histograms

**Model Development**

Algorithm used: Logistic Regression

Class balancing applied

Data split into training and testing sets (75% train, 25% test)

**Model Evaluation**

Performance on test data:

Accuracy: 99.41%

Precision: 98.74%

Recall: 100%

**Conclusion**

The Logistic Regression model demonstrates excellent performance in detecting fake banknotes. The high accuracy, precision, and recall indicate that the model is reliable for banknote authentication tasks.
