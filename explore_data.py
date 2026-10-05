# ============================================================
# EMPLOYEE ATTRITION PREDICTION USING MACHINE LEARNING
# Steps 1–9
# ============================================================


# ============================================================
# STEP 1 — IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix
)

from xgboost import XGBClassifier


# ============================================================
# STEP 2 — LOAD DATASET
# ============================================================

data = pd.read_csv(
    "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)


# ============================================================
# STEP 3 — DATA CLEANING
# ============================================================

# Remove columns with no useful information
columns_to_drop = [
    "EmployeeCount",
    "Over18",
    "StandardHours",
    "EmployeeNumber"
]

data = data.drop(columns=columns_to_drop)


# ============================================================
# STEP 4 — DEFINE FEATURES AND TARGET
# ============================================================

X = data.drop("Attrition", axis=1)
y = data["Attrition"]


# ============================================================
# STEP 5 — IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numeric_features = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "Education",
    "EnvironmentSatisfaction",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager"
]

categorical_features = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime"
]


# ============================================================
# STEP 6 — TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)


# ============================================================
# STEP 7 — PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# ============================================================
# STEP 8 — MODEL TRAINING, COMPARISON AND FINAL MODEL
# ============================================================

# ------------------------------------------------------------
# Cross-validation setup
# ------------------------------------------------------------

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=3,
    random_state=42
)


# ------------------------------------------------------------
# PR-AUC scorer
# ------------------------------------------------------------

def pr_auc_score(estimator, X, y):

    probabilities = estimator.predict_proba(X)

    yes_index = list(estimator.classes_).index("Yes")

    yes_probabilities = probabilities[:, yes_index]

    return average_precision_score(
        (y == "Yes").astype(int),
        yes_probabilities
    )


# ------------------------------------------------------------
# Logistic Regression
# ------------------------------------------------------------

logistic_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                C=1
            )
        )
    ]
)


# ------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------

random_forest_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                random_state=42
            )
        )
    ]
)


# ------------------------------------------------------------
# XGBoost
# ------------------------------------------------------------

y_xgb = y.map({
    "No": 0,
    "Yes": 1
})

X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
    X,
    y_xgb,
    test_size=0.20,
    stratify=y_xgb,
    random_state=42
)

xgboost_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            XGBClassifier(
                random_state=42,
                eval_metric="logloss"
            )
        )
    ]
)


# ------------------------------------------------------------
# Model comparison
# ------------------------------------------------------------

logistic_scores = []

for train_index, validation_index in cv.split(X_train, y_train):

    X_cv_train = X_train.iloc[train_index]
    X_cv_validation = X_train.iloc[validation_index]

    y_cv_train = y_train.iloc[train_index]
    y_cv_validation = y_train.iloc[validation_index]

    logistic_pipeline.fit(
        X_cv_train,
        y_cv_train
    )

    score = pr_auc_score(
        logistic_pipeline,
        X_cv_validation,
        y_cv_validation
    )

    logistic_scores.append(score)


random_forest_scores = []

for train_index, validation_index in cv.split(X_train, y_train):

    X_cv_train = X_train.iloc[train_index]
    X_cv_validation = X_train.iloc[validation_index]

    y_cv_train = y_train.iloc[train_index]
    y_cv_validation = y_train.iloc[validation_index]

    random_forest_pipeline.fit(
        X_cv_train,
        y_cv_train
    )

    score = pr_auc_score(
        random_forest_pipeline,
        X_cv_validation,
        y_cv_validation
    )

    random_forest_scores.append(score)


xgboost_scores = []

for train_index, validation_index in cv.split(
    X_train_xgb,
    y_train_xgb
):

    X_cv_train = X_train_xgb.iloc[train_index]
    X_cv_validation = X_train_xgb.iloc[validation_index]

    y_cv_train = y_train_xgb.iloc[train_index]
    y_cv_validation = y_train_xgb.iloc[validation_index]

    xgboost_pipeline.fit(
        X_cv_train,
        y_cv_train
    )

    probabilities = xgboost_pipeline.predict_proba(
        X_cv_validation
    )

    score = average_precision_score(
        y_cv_validation,
        probabilities[:, 1]
    )

    xgboost_scores.append(score)


# ------------------------------------------------------------
# Hyperparameter tuning for Logistic Regression
# ------------------------------------------------------------

param_grid = {
    "model__C": [
        0.01,
        0.1,
        1,
        10,
        100
    ]
}

grid_search = GridSearchCV(
    estimator=logistic_pipeline,
    param_grid=param_grid,
    scoring=pr_auc_score,
    cv=cv,
    n_jobs=-1,
    refit=True
)

grid_search.fit(
    X_train,
    y_train
)


# ------------------------------------------------------------
# Final Logistic Regression model
# ------------------------------------------------------------

final_logistic_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            LogisticRegression(
                C=0.1
            )
        )
    ]
)

final_logistic_pipeline.fit(
    X_train,
    y_train
)


# ------------------------------------------------------------
# Test probabilities and predictions
# ------------------------------------------------------------

test_probabilities = (
    final_logistic_pipeline.predict_proba(X_test)
)

yes_index = list(
    final_logistic_pipeline.classes_
).index("Yes")

test_yes_probabilities = (
    test_probabilities[:, yes_index]
)

test_predictions = (
    final_logistic_pipeline.predict(X_test)
)


# ------------------------------------------------------------
# OOF threshold selection
# ------------------------------------------------------------

cv_threshold = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

oof_probabilities = cross_val_predict(
    final_logistic_pipeline,
    X_train,
    y_train,
    cv=cv_threshold,
    method="predict_proba",
    n_jobs=-1
)

yes_index = list(
    final_logistic_pipeline.classes_
).index("Yes")

oof_yes_probabilities = (
    oof_probabilities[:, yes_index]
)


# ------------------------------------------------------------
# Select threshold using OOF F2-score
# ------------------------------------------------------------

thresholds = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50
]

best_f2 = -1
selected_threshold = 0.20

for threshold in thresholds:

    oof_predictions = np.where(
        oof_yes_probabilities >= threshold,
        "Yes",
        "No"
    )

    current_f2 = fbeta_score(
        (y_train == "Yes").astype(int),
        (oof_predictions == "Yes").astype(int),
        beta=2
    )

    if current_f2 > best_f2:

        best_f2 = current_f2
        selected_threshold = threshold


# ------------------------------------------------------------
# Final test predictions using selected threshold
# ------------------------------------------------------------

final_test_predictions = np.where(
    test_yes_probabilities >= selected_threshold,
    "Yes",
    "No"
)


# ============================================================
# STEP 9 — OVERTIME + JOB SATISFACTION VS ATTRITION
# ============================================================

# Calculate attrition rate for each combination
interaction_data = pd.crosstab(
    [
        data["OverTime"],
        data["JobSatisfaction"]
    ],
    data["Attrition"],
    normalize="index"
) * 100


# Keep only Attrition = Yes
attrition_rates = interaction_data["Yes"]


# Plot
plt.figure(figsize=(10, 6))

attrition_rates.plot(
    kind="bar"
)

plt.xlabel(
    "Overtime + Job Satisfaction"
)

plt.ylabel(
    "Attrition Rate (%)"
)

plt.title(
    "Attrition Rate by Overtime and Job Satisfaction"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.show()

# ============================================================
# STEP 10 — SAVE FINAL MODEL
# ============================================================

import joblib

# Save the complete trained pipeline
joblib.dump(
    final_logistic_pipeline,
    "employee_attrition_model.pkl"
)

# Save the selected decision threshold
joblib.dump(
    selected_threshold,
    "attrition_threshold.pkl"
)

print("Model saved successfully!")
print("Selected threshold:", selected_threshold)

# ============================================================
# STEP 11 — TEST MODEL WITH ONE EMPLOYEE
# ============================================================

import joblib

# Load the saved model
model = joblib.load(
    "employee_attrition_model.pkl"
)

# Load the selected threshold
threshold = joblib.load(
    "attrition_threshold.pkl"
)


# Create one sample employee
sample_employee = pd.DataFrame([{
    "Age": 30,
    "DailyRate": 1000,
    "DistanceFromHome": 5,
    "Education": 3,
    "EnvironmentSatisfaction": 3,
    "HourlyRate": 70,
    "JobInvolvement": 3,
    "JobLevel": 2,
    "JobSatisfaction": 2,
    "MonthlyIncome": 4000,
    "MonthlyRate": 15000,
    "NumCompaniesWorked": 2,
    "PercentSalaryHike": 12,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 3,
    "StockOptionLevel": 0,
    "TotalWorkingYears": 8,
    "TrainingTimesLastYear": 3,
    "WorkLifeBalance": 2,
    "YearsAtCompany": 3,
    "YearsInCurrentRole": 2,
    "YearsSinceLastPromotion": 1,
    "YearsWithCurrManager": 2,

    "BusinessTravel": "Travel_Rarely",
    "Department": "Sales",
    "EducationField": "Life Sciences",
    "Gender": "Female",
    "JobRole": "Sales Executive",
    "MaritalStatus": "Single",
    "OverTime": "Yes"
}])


# Get probability of Attrition = Yes
probability = model.predict_proba(
    sample_employee
)

yes_index = list(
    model.classes_
).index("Yes")

attrition_probability = probability[
    0,
    yes_index
]


# Apply our selected threshold
prediction = (
    "Yes"
    if attrition_probability >= threshold
    else "No"
)


# Display result
print("================================")
print("EMPLOYEE ATTRITION PREDICTION")
print("================================")

print(
    f"Attrition Probability: "
    f"{attrition_probability * 100:.2f}%"
)

print(
    f"Prediction: {prediction}"
)

print(
    f"Decision Threshold: {threshold}"
)
