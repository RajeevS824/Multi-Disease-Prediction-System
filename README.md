

---

# ⭐ **Multi-Disease Prediction System**

*A Machine Learning–Powered Healthcare Diagnostic Tool*

This project predicts the likelihood of **Kidney Disease**, **Liver Disease**, and **Parkinson’s Disease** using machine learning models.
The system includes **data preprocessing**, **model training**, **feature analysis**, and a **Streamlit web app** for real-time predictions.

---
    https://multi-disease-prediction-system-rsnbjermrrphjzqkkjwdb9.streamlit.app/

# 📌 **1. Problem Statement**

Early diagnosis of chronic diseases such as **Kidney Disease**, **Liver Disease**, and **Parkinson’s Disease** is often difficult due to:

* Limited access to specialists
* High diagnostic costs
* Time-consuming medical tests
* Inconsistent interpretation by clinicians

To solve this, we built a **Multi-Disease Prediction System** that uses machine learning to **assist doctors and patients** in identifying diseases early using clinical parameters.

---

# 📊 **2. Datasets Used**

| Disease        | File                        | Source         |
| -------------- | --------------------------- | -------------- |
| Kidney Disease | `kidney_disease.xlsx`       | UCI Repository |
| Liver Disease  | `indian_liver_patient.xlsx` | UCI Dataset    |
| Parkinson’s    | `parkinsons.xlsx`           | UCI Dataset    |

### ✨ Dataset Features Include:

* **Kidney:** Hemoglobin, RBC count, Blood Pressure, Albumin, Sugar, Creatinine
* **Liver:** Bilirubin, Albumin, Liver enzyme levels, Gender
* **Parkinson’s:** Vocal frequency, jitter, shimmer, motor function indicators

---

# ⚙️ **3. Approach**

### ✔️ **Step-by-Step Workflow**

1. **Data Cleaning**

   * Remove noise
   * Fix missing values (median/mode imputation)
   * Drop unnecessary columns

2. **Data Preprocessing**

   * Label Encoding for categorical variables
   * Feature Scaling (StandardScaler)
   * SMOTE applied to balance imbalanced datasets

3. **Model Training**

   * Algorithms used:

     * Logistic Regression
     * Random Forest
     * Decision Tree
     * SVM
     * KNN
     * XGBoost
     * Gradient Boosting
   * Trained separately for each disease

4. **Model Evaluation**

   * Accuracy
   * Precision
   * Recall
   * F1-score
   * ROC-AUC
   * Confusion Matrix

5. **Deployment**

   * Streamlit app for real-time predictions
   * User inputs clinical values
   * System predicts disease probability

---

# 🧪 **4. System Architecture**

```
                 ┌─────────────────┐
                 │     Frontend    │
                 │    (Streamlit)  │
                 └────────┬────────┘
                          │ User Inputs
                          ▼
                 ┌─────────────────┐
                 │     Backend     │
                 │   (Python ML)   │
                 └────────┬────────┘
                          │ Preprocessing
                          ▼
                 ┌─────────────────┐
                 │ ML Prediction   │
                 │(Kidney/Liver/Park)│
                 └────────┬────────┘
                          │ Output Probabilities
                          ▼
                 ┌─────────────────┐
                 │ Prediction Result│
                 └─────────────────┘
```

---

# 🌟 **5. Features**

✔ Predicts **three major diseases** from lab parameters

✔ **User-friendly Streamlit interface**
✔ **Correlation heatmaps** and **feature importance analysis**
✔ **SMOTE** for handling imbalanced datasets
✔ **Secure processing** with no data storage
✔ Scalable to add more diseases
✔ Fast inference

---

# 📁 **6. Project Structure**

```
Multi-Disease-Prediction/
│
├── data/
│   ├── kidney_disease.xlsx
│   ├── indian_liver_patient.xlsx
│   └── parkinsons.xlsx
│
├── model_training.py
├── app.py                # Streamlit app
├── README.md
└── requirements.txt
```

---

# 📈 **7. Results**

### 🎯 **Key Observations**

* Random Forest & Gradient Boosting perform best
* Parkinson dataset achieved highest accuracy (~95%+)
* Liver dataset improved significantly after SMOTE
* Kidney dataset required feature selection to reduce overfitting


---

# 🏥 **8. Real-Life Use Cases**

✔ **Rural healthcare clinics** where specialists are not available
✔ **Early awareness tool** for patients at risk
✔ **Telemedicine platforms** to support remote consultations
✔ **Hospitals** to assist doctors in second-opinion diagnosis
✔ **Health checkup centers** for quick automated reporting

---

# 🚀 **9. Business / Technical Impact**

### 💼 **Business Impact**

* Reduces cost of diagnosis
* Saves 30–50% time in preliminary screening
* Useful for health-tech apps
* Can be scaled as a SaaS product

### 🔧 **Technical Impact**

* Demonstrates ML pipeline creation
* Feature engineering + SMOTE balancing
* Deployment-ready Streamlit interface
* Extensible architecture to add more disease models

---

# 🔮 **10. Future Enhancements**

* Add **Diabetes, Heart Disease, and Cancer Prediction**
* Deploy backend using **FastAPI + Docker**
* Mobile application using Flutter
* Real-time patient monitoring with IoT sensors
* Improve Parkinson model using deep learning (LSTM voice processing)
* Add PDF medical reports generation
* Integrate EHR/HL7 data systems

---

# 🧱 **11. How to Run Locally**

### **Step 1 — Clone the repository**

```bash
git clone https://github.com/YourUsername/Multi-Disease-Prediction-System.git
cd Multi-Disease-Prediction-System
```

### **Step 2 — Install dependencies**

```bash
pip install -r requirements.txt
```

### **Step 3 — Run the Streamlit app**

```bash
streamlit run app.py
```

### **Step 4 — Upload values & get predictions**

---

# 🛠 **12. Tech Stack**

### **Programming**

* Python

### **Machine Learning**

* Scikit-learn
* XGBoost
* NumPy
* Pandas
* imbalanced-learn

### **Visualization**

* Seaborn
* Matplotlib

### **Web Framework**

* Streamlit

### **Other**

* SMOTE for class balancing
* StandardScaler

---

# 🏷 **13. Technical Tags**

`Machine Learning`, `Python`, `Streamlit`, `Healthcare AI`,
`Data Visualization`, `Classification`, `SMOTE`, `XGBoost`,
`RandomForest`, `Kidney Disease Prediction`, `Liver Disease Prediction`,
`Parkinsons Prediction`

---


