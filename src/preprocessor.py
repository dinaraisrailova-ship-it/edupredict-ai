import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE


PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")


FEATURE_GROUPS = {
    "demographic": [
        "Marital status", "Gender", "Age at enrollment",
        "International", "Displaced", "Educational special needs",
    ],
    "socioeconomic": [
        "Debtor", "Tuition fees up to date", "Scholarship holder",
        "Mother's occupation", "Father's occupation",
        "Mother's qualification", "Father's qualification",
    ],
    "academic_background": [
        "Application mode", "Application order", "Course",
        "Daytime/evening attendance\t", "Previous qualification",
        "Previous qualification (grade)", "Admission grade",
        "Nationality",
    ],
    "semester_1": [
        "Curricular units 1st sem (credited)",
        "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (evaluations)",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 1st sem (without evaluations)",
    ],
    "semester_2": [
        "Curricular units 2nd sem (credited)",
        "Curricular units 2nd sem (enrolled)",
        "Curricular units 2nd sem (evaluations)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (grade)",
        "Curricular units 2nd sem (without evaluations)",
    ],
    "macroeconomic": [
        "Unemployment rate", "Inflation rate", "GDP",
    ],
}


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    sem1_approved = "Curricular units 1st sem (approved)"
    sem2_approved = "Curricular units 2nd sem (approved)"
    sem1_enrolled = "Curricular units 1st sem (enrolled)"
    sem2_enrolled = "Curricular units 2nd sem (enrolled)"
    sem1_grade = "Curricular units 1st sem (grade)"
    sem2_grade = "Curricular units 2nd sem (grade)"

    df["total_approved"] = df[sem1_approved] + df[sem2_approved]
    df["total_enrolled"] = df[sem1_enrolled] + df[sem2_enrolled]
    df["approval_rate"] = np.where(
        df["total_enrolled"] > 0,
        df["total_approved"] / df["total_enrolled"],
        0,
    )
    df["avg_grade"] = (df[sem1_grade] + df[sem2_grade]) / 2
    df["grade_improvement"] = df[sem2_grade] - df[sem1_grade]
    df["sem1_approval_rate"] = np.where(
        df[sem1_enrolled] > 0, df[sem1_approved] / df[sem1_enrolled], 0
    )
    df["sem2_approval_rate"] = np.where(
        df[sem2_enrolled] > 0, df[sem2_approved] / df[sem2_enrolled], 0
    )
    return df


def preprocess(df: pd.DataFrame, test_size: float = 0.2, apply_smote: bool = True):
    """Full preprocessing pipeline. Returns X_train, X_test, y_train, y_test, feature_names."""
    df = df.copy()

    # Rename attendance column if it has a tab
    df.columns = [c.strip() for c in df.columns]

    df = _engineer_features(df)

    le = LabelEncoder()
    df["Target_encoded"] = le.fit_transform(df["Target"])
    class_names = list(le.classes_)

    target_col = "Target_encoded"
    drop_cols = ["Target", "Target_encoded"]

    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[target_col]

    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    if apply_smote:
        smote = SMOTE(random_state=42)
        X_train_sc, y_train = smote.fit_resample(X_train_sc, y_train)
        print(f"[INFO] After SMOTE — train set size: {X_train_sc.shape[0]}")

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print(f"[INFO] Train: {X_train_sc.shape}, Test: {X_test_sc.shape}")
    print(f"[INFO] Classes: {class_names}\n")

    return X_train_sc, X_test_sc, y_train, y_test, feature_names, class_names, scaler
