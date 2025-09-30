
# 🩺 Multi-Disease Prediction System



---

## 📌 Project Overview

Healthcare is moving towards **preventive diagnosis** powered by AI. This project builds an **end-to-end machine learning solution** that predicts the risk of:

* **Kidney Disease**
* **Liver Disease**
* **Parkinson’s Disease**

The system provides patients and healthcare professionals with a **personalized health report** based on clinical test results and lifestyle factors.

The project integrates **Python (EDA & ML), Scikit-learn, XGBoost, Imbalanced-learn (SMOTE), and Streamlit (interactive UI)** to deliver actionable health predictions.

---

## 🛠️ What I Did in This Project

### 1. Data Preparation (Python + Pandas)

* Loaded **Kidney (400×26), Liver (583×11), and Parkinson (195×24)** datasets.
* Handled missing values using median/mode imputation.
* Encoded categorical features (LabelEncoder).
* Scaled numerical features with StandardScaler.
* Balanced datasets with **SMOTE**.

### 2. Exploratory Data Analysis (EDA)

* Checked dataset distributions and correlations.
* Identified key disease indicators:

  * Kidney → Creatinine, Urea, Hemoglobin
  * Liver → Bilirubin, Enzymes, Proteins
  * Parkinson → PPE, Fhi, Spread measures
* Reduced redundant features to avoid overfitting.

### 3. Machine Learning Models (Scikit-learn & XGBoost)

Implemented and compared multiple models:

* Logistic Regression
* Random Forest
* Decision Tree
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)
* XGBoost

**Best Models:**

* Kidney → Logistic Regression / Random Forest (Accuracy ~97.5%)
* Liver → Random Forest (Accuracy ~77%)
* Parkinson → Random Forest & XGBoost (Accuracy ~92%)

**Metrics Evaluated:** Accuracy, Precision, Recall, F1-Score, ROC-AUC.

### 4. Streamlit Web App

* **Home Page** – Overview of project & datasets.
* **Kidney, Liver, Parkinson Tabs** – Disease-specific input forms.
* **Personal Info Section** – Age, Gender, Height, Weight, Lifestyle factors (smoking, alcohol, activity).
* **Health Report Generator** – Predicts disease probability with ✅ Normal / ❌ Detected output.
* Integrated **BMI calculation** and lifestyle recommendations.

---

## 🎯 Motive of the Project

* Enable **early disease detection** with machine learning.
* Provide **accessible, personalized health reports**.
* Support doctors and patients with **data-driven insights**.

---

## 🌍 Real-Life Use Cases

* **Patients** → Quick, accessible self-screening tool before visiting a doctor.
* **Doctors** → Use as a **decision support system** alongside medical tests.
* **Hospitals/Clinics** → Integrate into digital health portals for patient monitoring.
* **Researchers** → Study risk factors and trends in kidney, liver, and Parkinson’s disease.

---

## ✅ Conclusion

This project demonstrates how **EDA + Machine Learning + Streamlit** can power healthcare solutions.
By combining predictive models with an interactive interface, the system provides:

* **Accurate, real-time disease predictions**
* **Personalized reports with lifestyle advice**
* **A scalable foundation** for adding more diseases and datasets

It is a step towards **AI-driven preventive healthcare**.

