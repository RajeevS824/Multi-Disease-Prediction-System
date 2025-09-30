# ==============================
# IMPORTS
# ==============================
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# ==============================
# DATA PREPARATION FUNCTIONS
# ==============================
def preprocess_liver(path="indian_liver_patient.xlsx"):
    df = pd.read_excel(path)
    df["Albumin_and_Globulin_Ratio"].fillna(df["Albumin_and_Globulin_Ratio"].median(), inplace=True)

    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    df['Dataset'] = df['Dataset'].map({1: 1, 2: 0})

    X = df.drop(columns=['Dataset'])
    y = df['Dataset']
    return X, y


def preprocess_kidney(path="kidney_disease.xlsx"):
    df = pd.read_excel(path)
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    num_cols = ['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc']
    cat_cols = ['rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane']

    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

    imputer_num = SimpleImputer(strategy='median')
    df[num_cols] = imputer_num.fit_transform(df[num_cols])
    medians_num = dict(zip(num_cols, imputer_num.statistics_))

    imputer_cat = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    df['classification'] = df['classification'].map({'ckd': 1, 'notckd': 0})

    X = df.drop(columns=['classification'])
    y = df['classification']
    return X, y, medians_num


def preprocess_parkinson(path="parkinsons.xlsx"):
    df = pd.read_excel(path)
    df = df.drop(columns=['name'])
    X = df.drop(columns=['status'])
    y = df['status']
    return X, y


def prepare_data(X, y, test_size=0.2):
    smote = SMOTE(random_state=42)
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler


# ==============================
# MODEL TRAINING FUNCTIONS
# ==============================
@st.cache_resource
def train_models():
    # Kidney (LogisticRegression)
    Xk, yk, medians_num = preprocess_kidney()
    Xk_train, Xk_test, yk_train, yk_test, sk = prepare_data(Xk, yk)
    kidney_model = LogisticRegression(max_iter=2000, class_weight="balanced")
    kidney_model.fit(Xk_train, yk_train)

    # Liver (RandomForest)
    Xl, yl = preprocess_liver()
    Xl_train, Xl_test, yl_train, yl_test, sl = prepare_data(Xl, yl)
    liver_model = RandomForestClassifier(random_state=42, max_depth=8, class_weight="balanced")
    liver_model.fit(Xl_train, yl_train)

    # Parkinson (RandomForest)
    Xp, yp = preprocess_parkinson()
    Xp_train, Xp_test, yp_train, yp_test, sp = prepare_data(Xp, yp)
    parkinson_model = RandomForestClassifier(random_state=42, max_depth=8, class_weight="balanced")
    parkinson_model.fit(Xp_train, yp_train)

    return (kidney_model, sk, Xk.columns, medians_num,
            liver_model, sl, Xl.columns,
            parkinson_model, sp, Xp.columns)


# ==============================
# PREDICTION HELPER
# ==============================
def make_prediction(model, scaler, input_data, feature_names, medians=None):
    df = pd.DataFrame([input_data])
    for col in feature_names:
        if col not in df.columns:
            if medians and col in medians:
                df[col] = medians[col]
            else:
                df[col] = 0
    df = df[feature_names]
    df_scaled = scaler.transform(df)
    pred = model.predict(df_scaled)[0]
    prob = model.predict_proba(df_scaled)[0][1]
    return pred, prob


# ==============================
# LOAD MODELS
# ==============================
(kidney_model, kidney_scaler, kidney_features, kidney_medians,
 liver_model, liver_scaler, liver_features,
 parkinson_model, parkinson_scaler, parkinson_features) = train_models()


# ==============================
# STREAMLIT UI
# ==============================
st.title("🩺 Multiple Disease Prediction System")

# ---- Personal Information ----
st.subheader("👤 Personal Information")
person_name = st.text_input("Full Name")
person_age = st.number_input("Age", 1, 100, 30)
person_gender = st.radio("Gender", ["Male", "Female"])
report_date = datetime.today().strftime("%d-%m-%Y")

col_a, col_b = st.columns(2)
with col_a:
    person_height = st.number_input("Height (cm)", 100, 220, 170)
    smoking_status = st.radio("Smoking Status", ["No", "Yes"])
with col_b:
    person_weight = st.number_input("Weight (kg)", 30, 200, 70)
    alcohol_status = st.radio("Alcohol Consumption", ["No", "Yes"])

physical_activity = st.selectbox("Physical Activity", ["Low", "Moderate", "High"])

# Calculate BMI
bmi = round(person_weight / ((person_height / 100) ** 2), 2)

st.write("---")

# -------- Kidney Inputs --------
st.subheader("Kidney Function Tests ")
col1, col2 = st.columns(2)
with col1:
    kd_sc = st.number_input("Serum Creatinine (mg/dl)", 0.1, 15.0, 1.2, step=0.1)
    kd_hemo = st.slider("Hemoglobin (g/dl)", 3.0, 18.0, 14.0)
    kd_bu = st.number_input("Blood Urea (mg/dl)", 1, 300, 40)
with col2:
    kd_bgr = st.number_input("Blood Glucose Random (mg/dl)", 50, 500, 120)
    kd_sod = st.number_input("Sodium (mEq/L)", 120, 160, 140)
    kd_htn = st.selectbox("History of Hypertension", ["No", "Yes"])

# -------- Liver Inputs --------
st.subheader("Liver Function Tests ")
col3, col4 = st.columns(2)
with col3:
    lv_tb = st.number_input("Total Bilirubin (mg/dl)", 0.0, 10.0, 1.0, step=0.1)
    lv_db = st.number_input("Direct Bilirubin (mg/dl)", 0.0, 5.0, 0.3, step=0.1)
    lv_ap = st.number_input("Alkaline Phosphotase (IU/L)", 50, 3000, 200)
with col4:
    lv_aa = st.number_input("Alamine Aminotransferase (IU/L)", 5, 2000, 30)
    lv_aspartate = st.number_input("Aspartate Aminotransferase (IU/L)", 5, 2000, 30)
    lv_ag = st.number_input("Albumin and Globulin Ratio", 0.0, 3.0, 1.0, step=0.1)

# -------- Parkinson Inputs (Top 6 correlated, custom threshold) --------
st.subheader("Neurological Voice Tests ")
col5, col6 = st.columns(2)
with col5:
    pk_ppe = st.number_input("PPE", 0.0, 1.0, 0.3, step=0.01)
    pk_mdvp_fo = st.number_input("MDVP:Fo(Hz)", 80.0, 300.0, 150.0, step=0.1)
    pk_mdvp_fhi = st.number_input("MDVP:Fhi(Hz)", 100.0, 600.0, 250.0, step=1.0)
with col6:
    pk_mdvp_shimmer = st.number_input("MDVP:Shimmer", 0.0, 1.0, 0.1, step=0.01)
    pk_spread1 = st.number_input("Infection Index (Spread1)", -7.0, 1.0, -3.0, step=0.1)
    pk_spread2 = st.number_input("Spread Factor (Spread2)", -7.0, 1.0, -3.5, step=0.1)


# ==============================
# PREDICTION
# ==============================
if st.button("🔍 Generate Health Report"):
    # Prepare input dicts
    kidney_input = {
        'sc': kd_sc,
        'hemo': kd_hemo,
        'bu': kd_bu,
        'bgr': kd_bgr,
        'sod': kd_sod,
        'htn': 1 if kd_htn == "Yes" else 0
    }

    liver_input = {
        'Age': person_age, 'Gender': 1 if person_gender == "Male" else 0,
        'Total_Bilirubin': lv_tb,
        'Direct_Bilirubin': lv_db,
        'Alkaline_Phosphotase': lv_ap,
        'Alamine_Aminotransferase': lv_aa,
        'Aspartate_Aminotransferase': lv_aspartate,
        'Albumin_and_Globulin_Ratio': lv_ag
    }

    parkinson_input = {
        'PPE': pk_ppe,
        'MDVP:Fo(Hz)': pk_mdvp_fo,
        'MDVP:Fhi(Hz)': pk_mdvp_fhi,
        'MDVP:Shimmer': pk_mdvp_shimmer,
        'spread1': pk_spread1,
        'spread2': pk_spread2
    }

    # Predictions
    kd_pred, kd_prob = make_prediction(kidney_model, kidney_scaler, kidney_input, kidney_features, kidney_medians)
    lv_pred, lv_prob = make_prediction(liver_model, liver_scaler, liver_input, liver_features)
    pk_pred_raw, pk_prob = make_prediction(parkinson_model, parkinson_scaler, parkinson_input, parkinson_features)

    # Apply custom threshold (0.60) for Parkinson’s
    pk_pred = 1 if pk_prob >= 0.60 else 0

    # ==============================
    # REPORT
    # ==============================
    st.subheader("📝 Health Report")
    st.write(f"**Date of Report:** {report_date}")
    st.write(f"**Patient Name:** {person_name}")
    st.write(f"**Age:** {person_age} | **Gender:** {person_gender}")
    st.write(f"**Height:** {person_height} cm | **Weight:** {person_weight} kg | **BMI:** {bmi}")
    st.write(f"**Lifestyle:** Smoking - {smoking_status}, Alcohol - {alcohol_status}, Activity - {physical_activity}")

    st.write("---")
    st.write(f"**Kidney Disease Probability:** {kd_prob:.2%} → {'Detected ❌' if kd_pred==1 else 'Normal ✅'}")
    st.write(f"**Liver Disease Probability:** {lv_prob:.2%} → {'Detected ❌' if lv_pred==1 else 'Normal ✅'}")
    st.write(f"**Parkinson’s Probability:** {pk_prob:.2%} → {'Detected ❌' if pk_pred==1 else 'Normal ✅'}")

    st.write("---")
    if kd_pred==0 and lv_pred==0 and pk_pred==0:
        st.success("✅ You are healthy! Maintain a balanced diet, regular exercise, and hydration.")
    else:
        st.warning("⚠️ Health Advice:")
        if kd_pred==1:
            st.write("- **Kidney Health**: Reduce salt, avoid processed foods, stay hydrated, consult nephrologist.")
        if lv_pred==1:
            st.write("- **Liver Health**: Avoid alcohol, eat leafy greens, turmeric, antioxidants. Consult hepatologist.")
        if pk_pred==1:
            st.write("- **Parkinson’s**: Regular medication, physiotherapy, balanced nutrition, neurologist guidance.")

# python -m streamlit run mutli_disease_streamlit.py
