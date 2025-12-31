

# 📊 Credit Card Fraud Detection – Machine Learning Project

## 📌 Project Overview

This project focuses on detecting fraudulent credit card transactions using **machine learning classification techniques**.
The objective is to accurately classify transactions as **fraudulent (1)** or **non-fraudulent (0)** despite severe **class imbalance**.

Multiple models were implemented, evaluated, and compared, followed by **model deployment** using a saved pipeline for real-time predictions.

---

## 📂 Dataset Description

* **Dataset Name:** Credit Card Transactions Dataset
* **Source:** Public Kaggle Dataset
* **File:** `creditcard.csv`
* **Total Records:** ~284,807 transactions
* **Target Variable:**

  * `Class = 0` → Non-Fraud
  * `Class = 1` → Fraud
* **Features:**

  * `V1` to `V28`: PCA-transformed features
  * `Amount`: Transaction amount
  * `Time`: Time elapsed between transactions
  * `Class`: Target variable

⚠️ The dataset is **highly imbalanced**, with fraud cases accounting for less than **0.2%** of all transactions.

---

## 🔧 Data Preprocessing

The following preprocessing steps were performed:

### 1️⃣ Missing Values

* Checked for null values using `.isnull().sum()`
* Missing values were handled using **forward fill (ffill)** where applicable

### 2️⃣ Duplicate Removal

* Duplicate records were identified and removed to maintain data integrity

### 3️⃣ Feature Scaling

* **StandardScaler** was applied to numerical features to normalize data
* Scaling is critical for distance-based models like **KNN** and gradient-based models like **Logistic Regression**

### 4️⃣ Class Imbalance Handling

* **Undersampling technique** was used:

  * Randomly sampled legitimate transactions to match fraud transaction count
* This ensured balanced training data for model learning

---

## 🧠 Feature Engineering

Feature engineering focused on improving predictive performance:

* **Transaction Amount Analysis**

  * Statistical comparison between fraud and non-fraud amounts
* **Correlation Analysis**

  * Heatmaps used to identify features most correlated with fraud (`V10`, `V12`, `V14`, `Amount`)
* **Feature Selection**

  * Reduced feature set used for deployment:

    * `Amount`, `V10`, `V12`, `V14`

These engineered features help capture **usage patterns and abnormal behavior** similar to churn prediction metrics.

---

## 🤖 Model Selection & Methodology

Multiple classification models were trained and evaluated:

### 🔹 Logistic Regression

* Simple and interpretable baseline model
* Works well with scaled numerical features
* Used for final deployment due to stability and performance

### 🔹 Decision Tree Classifier

* Captures non-linear relationships
* Visualized using `plot_tree()`
* Risk of overfitting on imbalanced data

### 🔹 K-Nearest Neighbors (KNN)

* Distance-based classifier
* Optimal `k` chosen using **error-rate vs k** plot
* Computationally expensive on large datasets

### 🔹 Model Comparison

Models were compared using standardized metrics to identify the best performer.

---

## 📏 Evaluation Metrics

The following evaluation metrics were used:

* **Accuracy** – Overall correctness of the model
* **Precision** – How many predicted frauds were actually fraud
* **Recall (Sensitivity)** – Ability to detect actual fraud cases
* **F1-Score** – Balance between Precision and Recall
* **Confusion Matrix** – Detailed error analysis

📌 **F1-Score and Recall** were prioritized due to class imbalance.

---

## 📈 Model Performance Comparison

A comparison table was generated:

| Model               | Accuracy | Precision | Recall | F1-Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression | ✔        | ✔         | ✔      | ✔        |
| KNN                 | ✔        | ✔         | ✔      | ✔        |
| Decision Tree       | ✔        | ✔         | ✔      | ✔        |

The **Logistic Regression model** showed the best trade-off between interpretability, recall, and F1-score.

---

## 🚀 Model Deployment

A production-ready pipeline was created using:

* **StandardScaler**
* **Logistic Regression**
* **Scikit-learn Pipeline**

### 🔹 Model Saving

```bash
fraud_detection_model.pkl
scaler.pkl
```

Saved using **joblib** for reuse.

---

## 🧪 Real-Time Prediction

* Interactive prediction interface built using **ipywidgets**
* User inputs:

  * `Amount`
  * `V10`
  * `V12`
  * `V14`
* Model outputs:

  * `0` → Not Fraud
  * `1` → Fraud

---

## 🛠 Technologies Used

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* Joblib
* ipywidgets

---

## 📌 Conclusion

This project demonstrates a complete **end-to-end machine learning workflow**:

* Data preprocessing
* Feature engineering
* Model training & evaluation
* Handling class imbalance
* Model comparison
* Deployment & real-time inference

It mirrors **churn prediction methodologies**, focusing on behavior analysis, imbalance handling, and recall-driven evaluation.

---

