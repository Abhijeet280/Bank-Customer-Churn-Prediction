# 🏦 Bank Customer Churn Prediction

This project predicts whether a **bank customer will churn** (leave the bank) using **Deep Learning models**.
It involves **data preprocessing, feature scaling, model building, and evaluation** to improve customer retention strategies.

---

## 🚀 Project Overview

The notebook performs the following steps:

1. **Importing Libraries** – Loading Python libraries such as pandas, numpy, scikit-learn, matplotlib, seaborn, and keras.
2. **Data Preprocessing** – Handling categorical data using `LabelEncoder`.
3. **Feature Scaling** – Normalizing numerical values using `MinMaxScaler` to improve neural network performance.
4. **Train-Test Split** – Dividing the dataset into training and testing sets.
5. **Model Building** – Using **Keras Sequential API** to create a deep learning model with:

   * Input layer
   * Multiple hidden layers (Dense layers with activation functions)
   * Output layer for binary classification (Churn / No Churn)
6. **Model Evaluation** – Evaluating model performance using:

   * **Accuracy Score**
   * **Confusion Matrix**
   * **Classification Report** (Precision, Recall, F1-score)

---

## 🛠️ Technologies Used

* Python
* Pandas & NumPy
* Matplotlib & Seaborn
* Scikit-learn (Preprocessing, Metrics)
* TensorFlow / Keras (Deep Learning Model)
* Jupyter Notebook

---

## 📂 Dataset

The dataset contains bank customer details with the following features:

* **CustomerId** – Unique ID for the customer
* **Surname** – Last name of the customer
* **CreditScore** – Credit score of the customer
* **Geography** – Country of residence
* **Gender** – Male/Female
* **Age** – Customer’s age
* **Tenure** – Number of years with the bank
* **Balance** – Account balance
* **NumOfProducts** – Number of bank products used
* **HasCrCard** – Whether the customer has a credit card (1 = Yes, 0 = No)
* **IsActiveMember** – Whether the customer is active (1 = Yes, 0 = No)
* **EstimatedSalary** – Estimated salary of the customer
* **Exited (Target Variable)** – Whether the customer churned (1 = Churn, 0 = Stay)

---

## 📈 Expected Insights

* **Age, balance, and number of products** are strong indicators of customer churn.
* **Active customers with higher credit scores** are less likely to leave.
* Deep learning models can capture complex patterns better than traditional machine learning methods.

---

