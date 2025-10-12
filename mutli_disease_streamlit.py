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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from datetime import datetime

# ==============================
# DATA PREPROCESSING FUNCTIONS
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
    kidney_medians = dict(zip(num_cols, imputer_num.statistics_))

    imputer_cat = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    df['classification'] = df['classification'].map({'ckd': 1, 'notckd': 0})
    X = df.drop(columns=['classification'])
    y = df['classification']
    return X, y, kidney_medians

def preprocess_parkinson(path="parkinsons.xlsx"):
    df = pd.read_excel(path)
    df = df.drop(columns=['name'])
    X = df.drop(columns=['status'])
    y = df['status']
    return X, y

# ==============================
# DATA PREPARATION
# ==============================
def prepare_data(X, y, test_size=0.2):
    smote = SMOTE(random_state=42)
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    X_train, y_train = smote.fit_resample(X_train, y_train)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler

# ==============================
# MODEL TRAINING
# ==============================
@st.cache_resource
def train_models():
    # Kidney
    Xk, yk, kidney_medians = preprocess_kidney()
    Xk_train, Xk_test, yk_train, yk_test, sk = prepare_data(Xk, yk)
    kidney_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=42,
        class_weight="balanced"
    )
    kidney_model.fit(Xk_train, yk_train)

    # Liver
    Xl, yl = preprocess_liver()
    Xl_train, Xl_test, yl_train, yl_test, sl = prepare_data(Xl, yl)
    liver_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        random_state=42
    )
    liver_model.fit(Xl_train, yl_train)

    # Parkinson
    Xp, yp = preprocess_parkinson()
    Xp_train, Xp_test, yp_train, yp_test, sp = prepare_data(Xp, yp)
    parkinson_model = RandomForestClassifier(
        random_state=42,
        max_depth=8,
        class_weight="balanced",
        min_samples_leaf=1,
        min_samples_split=2,
        n_estimators=50
    )
    parkinson_model.fit(Xp_train, yp_train)
    
    return (kidney_model, sk, Xk.columns, kidney_medians,
            liver_model, sl, Xl.columns,
            parkinson_model, sp, Xp.columns)

# ==============================
# PREDICTION HELPER
# ==============================
def make_prediction(model, scaler, input_data, feature_names, medians=None):
    df = pd.DataFrame([input_data])
    for col in feature_names:
        if col not in df.columns:
            df[col] = medians[col] if medians and col in medians else 0
    df = df[feature_names]
    df_scaled = scaler.transform(df)
    pred = model.predict(df_scaled)[0]
    prob = model.predict_proba(df_scaled)[0][1] if hasattr(model, "predict_proba") else 0
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
report_date = datetime.today().strftime("%d-%m-%Y")

# ------------------------------
# PERSONAL INFORMATION
# ------------------------------
st.subheader("👤 Personal Information")
person_name = st.text_input("Full Name", key="person_name")
person_age = st.number_input("Age", 1, 100, 30, key="person_age")
person_gender = st.radio("Gender", ["Male", "Female"], key="person_gender")

col1, col2 = st.columns(2)
with col1:
    person_height = st.number_input("Height (cm)", 100, 220, 170, key="person_height")
    smoking_status = st.selectbox("Smoking Status", ["No", "Yes"], key="smoking_status")
with col2:
    person_weight = st.number_input("Weight (kg)", 30, 200, 70, key="person_weight")
    alcohol_status = st.selectbox("Alcohol Consumption", ["No", "Yes"], key="alcohol_status")

physical_activity = st.selectbox("Physical Activity", ["Low", "Moderate", "High"], key="physical_activity")
bmi = round(person_weight / ((person_height / 100) ** 2), 2)

# ------------------------------
# GENERATE INPUTS FUNCTION
# ------------------------------
def generate_inputs(df_columns, disease_name, medians=None):
    input_data = {}
    for col in df_columns:
        key = f"{disease_name}_{col}"
        if col.lower() in ['gender']:
            val = st.radio(col, ["Male", "Female"], key=key)
            input_data[col] = 1 if val == "Male" else 0
        elif col.lower() in ['htn','dm','cad','appet','pe','ane','rbc','pc','pcc','ba']:
            val = st.selectbox(col, ["No", "Yes"], key=key)
            input_data[col] = 1 if val == "Yes" else 0
        else:
            default_val = medians[col] if medians and col in medians else 0.0
            val = st.number_input(col, value=default_val, key=key)
            input_data[col] = val
    return input_data

# ------------------------------
# TABS FOR EACH DISEASE
# ------------------------------
tab1, tab2, tab3 = st.tabs(["🧠 Parkinson’s", "🩸 Liver", "🫀 Kidney"])

# Parkinson’s Tab
with tab1:
    st.subheader("🧠 Parkinson’s Health Parameters")
    parkinson_input = generate_inputs(parkinson_features, "parkinson")

    if st.button("🔍 Generate Parkinson’s Report", key="parkinson_report"):
        st.subheader("🧠 Parkinson’s Disease Prediction Report")
        st.write(f"**Patient:** {person_name} | **Age:** {person_age} | **Gender:** {person_gender}")
        st.write(f"**BMI:** {bmi} | **Lifestyle:** Smoking-{smoking_status}, Alcohol-{alcohol_status}")
        st.markdown("---")

        pk_pred, _ = make_prediction(parkinson_model, parkinson_scaler, parkinson_input, parkinson_features)
        result_text = "Detected ❌" if pk_pred == 1 else "Normal ✅"
        st.header(f"🧠 Parkinson’s Disease Prediction: {result_text}")

        if pk_pred == 1:
            st.warning("⚠️ **Diagnosis: Parkinson’s Disease Confirmed**")
            st.info("🩺 Advice: Regular medication, physiotherapy, and neurologist consultation are essential.")
        else:
            st.success("🎉 **Diagnosis: No Parkinson’s Detected**")
            st.info("🏃‍♂️ Maintain regular physical activity and brain exercises.")

# Liver Tab
with tab2:
    st.subheader("🩸 Liver Health Parameters")
    liver_input = generate_inputs(liver_features, "liver")

    if st.button("🔍 Generate Liver Report", key="liver_report"):
        st.subheader("🩸 Liver Disease Prediction Report")
        st.write(f"**Patient:** {person_name} | **Age:** {person_age} | **Gender:** {person_gender}")
        st.write(f"**BMI:** {bmi} | **Lifestyle:** Smoking-{smoking_status}, Alcohol-{alcohol_status}")
        st.markdown("---")

        lv_pred, _ = make_prediction(liver_model, liver_scaler, liver_input, liver_features)
        result_text = "Detected ❌" if lv_pred == 1 else "Normal ✅"
        st.header(f"🩸 Liver Disease Prediction: {result_text}")

        if lv_pred == 1:
            st.warning("⚠️ **Diagnosis: Liver Disease Confirmed**")
            st.info("🩺 Advice: Avoid alcohol, eat leafy greens, take antioxidants, consult a hepatologist.")
        else:
            st.success("🎉 **Diagnosis: Liver Function Normal**")
            st.info("🥦 Continue a healthy diet and avoid alcohol for liver health.")

# Kidney Tab
with tab3:
    st.subheader("🫀 Kidney Health Parameters")
    kidney_input = generate_inputs(kidney_features, "kidney", kidney_medians)

    if st.button("🔍 Generate Kidney Report", key="kidney_report"):
        st.subheader("🫀 Kidney Disease Prediction Report")
        st.write(f"**Patient:** {person_name} | **Age:** {person_age} | **Gender:** {person_gender}")
        st.write(f"**BMI:** {bmi} | **Lifestyle:** Smoking-{smoking_status}, Alcohol-{alcohol_status}")
        st.markdown("---")

        kd_pred, _ = make_prediction(kidney_model, kidney_scaler, kidney_input, kidney_features, kidney_medians)
        result_text = "Detected ❌" if kd_pred == 1 else "Normal ✅"
        st.header(f"🫀 Kidney Disease Prediction: {result_text}")

        if kd_pred == 1:
            st.warning("⚠️ **Diagnosis: Kidney Disease Confirmed**")
            st.info("🩺 Advice: Reduce salt, avoid processed foods, stay hydrated, consult a nephrologist.")
        else:
            st.success("🎉 **Diagnosis: Kidney Function Normal**")
            st.info("💧 Maintain hydration and healthy diet for kidney wellness.")

# python -m streamlit run mutli_disease_streamlit.py
