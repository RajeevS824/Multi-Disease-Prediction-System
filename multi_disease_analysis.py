"""
Multi-Disease Prediction System
--------------------------------
Datasets: 
  - Kidney Disease
  - Liver Disease
  - Parkinson's Disease

Steps:
1. Load datasets
2. Clean & preprocess (handle missing values, encode categorical vars, scale)
3. Handle imbalance with SMOTE
4. Train/Test split
5. Train ML models (LR, RF, DT, SVM, KNN, XGB)
6. Evaluate with Accuracy, AUC, Classification Report
7. Show feature correlations / importance (to detect overfitting features)
"""

# ==============================
# 1. IMPORT LIBRARIES
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from imblearn.over_sampling import SMOTE

# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


# ==============================
# 2. LOAD DATASETS
# ==============================
kidney_dis = pd.read_excel("kidney_disease.xlsx")
liver_dis = pd.read_excel("indian_liver_patient.xlsx")
parkinson_dis = pd.read_excel("parkinsons.xlsx")

print("Kidney dataset shape:", kidney_dis.shape)
print("Liver dataset shape:", liver_dis.shape)
print("Parkinson dataset shape:", parkinson_dis.shape)

print(liver_dis.info())
print(kidney_dis.info())
print(parkinson_dis.info())


# ==============================
# 3. DATA CLEANING
# ==============================

# --- LIVER DATASET ---
print(liver_dis.isnull().sum())
# Fix: inplace=True already returns None, so remove assignment
liver_dis["Albumin_and_Globulin_Ratio"].fillna(
    liver_dis["Albumin_and_Globulin_Ratio"].median(), inplace=True
)

le = LabelEncoder()
liver_dis['Gender'] = le.fit_transform(liver_dis['Gender'])  # Male=1, Female=0
liver_dis['Dataset'] = liver_dis['Dataset'].map({1: 1, 2: 0})  # Disease=1, No disease=0


# --- PARKINSON DATASET ---
print(parkinson_dis.isnull().sum())
parkinson_dis = parkinson_dis.drop(columns=['name'])  # drop useless ID column


# --- KIDNEY DATASET ---
print(kidney_dis.isnull().sum())
if 'id' in kidney_dis.columns:
    kidney_dis = kidney_dis.drop(columns=['id'])  # drop patient id if exists

# Convert numeric columns properly
num_cols = ['age','bp','sg','al','su','bgr','bu','sc','sod','pot',
            'hemo','pcv','wc','rc']
kidney_dis[num_cols] = kidney_dis[num_cols].apply(pd.to_numeric, errors='coerce')

# Impute missing numeric values (median)
imputer_num = SimpleImputer(strategy='median')
kidney_dis[num_cols] = imputer_num.fit_transform(kidney_dis[num_cols])

# Handle categorical columns
cat_cols = ['rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane']
imputer_cat = SimpleImputer(strategy='most_frequent')
kidney_dis[cat_cols] = imputer_cat.fit_transform(kidney_dis[cat_cols])

# Encode categorical vars
for col in cat_cols:
    kidney_dis[col] = le.fit_transform(kidney_dis[col])

# Encode target column
kidney_dis['classification'] = kidney_dis['classification'].map({'ckd': 1, 'notckd': 0})


# ==============================
# 4. FEATURE IMPORTANCE / CORRELATION
# ==============================
def plot_feature_heatmap(X, y, title):
    """
    Plot correlation heatmap of features with target
    """
    df = pd.DataFrame(X, columns=X.columns if isinstance(X, pd.DataFrame) else range(X.shape[1]))
    df["Target"] = y.values if isinstance(y, pd.Series) else y

    corr = df.corr()
    plt.figure(figsize=(10,6))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
    plt.title(f"Feature Correlation Heatmap - {title}")
    plt.show()


# ==============================
# 5. FEATURE / TARGET SPLIT
# ==============================
# LIVER
X_liver = liver_dis.drop(columns=['Dataset'])
y_liver = liver_dis['Dataset']

# KIDNEY (dropping redundant features to reduce overfitting)
X_kidney = kidney_dis.drop(columns=['classification','sg','rc','al']) 
y_kidney = kidney_dis['classification']

# PARKINSON
X_parkinson = parkinson_dis.drop(columns=['status'])
y_parkinson = parkinson_dis['status']


# Plot for all datasets (moved here so X,y exist)
plot_feature_heatmap(X_liver, y_liver, "Liver")
plot_feature_heatmap(X_kidney, y_kidney, "Kidney")
plot_feature_heatmap(X_parkinson, y_parkinson, "Parkinson")


# ==============================
# FEATURE IMPORTANCE (Correlation-based)
# ==============================
def plot_corr_importance(df, target, title):
    """
    Plot horizontal bar chart of feature importance based on correlation with target
    """
    corr = df.corr()[target].drop(target)        # correlation with target
    corr_abs = corr.abs().sort_values(ascending=True)  # sort for clean plot

    plt.figure(figsize=(10,6))
    corr_abs.plot(kind='barh', color='red')
    plt.title(f"Feature Importance (Correlation with {target}) - {title}")
    plt.xlabel("Absolute Correlation")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()


# Run for all datasets
plot_corr_importance(liver_dis, 'Dataset', "Liver")
plot_corr_importance(kidney_dis, 'classification', "Kidney")
plot_corr_importance(parkinson_dis, 'status', "Parkinson")


# ==============================
# 6. BALANCING & SCALING
# ==============================
scaler = StandardScaler()
smote = SMOTE(random_state=42)

def prepare_data(X, y, dataset_name, test_size=0.2):
    """
    Split Train/Test, balance with SMOTE, scale features
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    print(f"\n🔹 Before SMOTE ({dataset_name}):")
    print(pd.Series(y_train).value_counts())

    # Apply SMOTE
    X_train, y_train = smote.fit_resample(X_train, y_train)

    print(f"\n After SMOTE ({dataset_name}):")
    print(pd.Series(y_train).value_counts())

    # Scale
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


# Prepare datasets
X_train_liver, X_test_liver, y_train_liver, y_test_liver = prepare_data(X_liver, y_liver, "Liver")
X_train_kidney, X_test_kidney, y_train_kidney, y_test_kidney = prepare_data(X_kidney, y_kidney, "Kidney")
X_train_parkinson, X_test_parkinson, y_train_parkinson, y_test_parkinson = prepare_data(X_parkinson, y_parkinson, "Parkinson")


# ==============================
# 7. MODEL TRAINING & EVALUATION
# ==============================
def train_and_evaluate(X, y, dataset_name):
    """
    Train and evaluate multiple ML models on given dataset
    """
    print(f"\n==============================")
    print(f" Results for {dataset_name} Dataset")
    print(f"==============================")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(random_state=42, max_depth=8, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=6, class_weight="balanced"),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced"),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Probabilities for AUC
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.decision_function(X_test)

        auc = roc_auc_score(y_test, y_proba)

        print(f"\n{name}:")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("AUC Score:", round(auc, 3))
        print(classification_report(y_test, y_pred, zero_division=0))


# ==============================
# 8. RUN ON ALL DATASETS
# ==============================
train_and_evaluate(X_liver, y_liver, "Liver")
train_and_evaluate(X_kidney, y_kidney, "Kidney")
train_and_evaluate(X_parkinson, y_parkinson, "Parkinson")

