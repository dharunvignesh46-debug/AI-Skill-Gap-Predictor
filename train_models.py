"""
Trains the two ML models used by the API:
  1. RandomForestClassifier -> Skill_Gap_Level (Low / Medium / High)
  2. RandomForestRegressor  -> Job_Readiness_Score (0-100)

Also computes the average skill profile per Target_Role, which the API
uses to benchmark a student against their chosen target role.

Run this once before starting the API (app.py runs it automatically the
first time if no saved models are found):

    python train_models.py
"""

import json
import os
import warnings

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "ai_skill_gap_predictor_dataset-1.csv")
MODEL_DIR = os.path.join(HERE, "models")
RANDOM_STATE = 42

SKILL_COLS = [
    "Python", "Machine_Learning", "SQL", "Deep_Learning", "NLP", "DSA",
    "Git", "Cloud", "Statistics", "Communication",
]
NUMERIC_EXTRA_COLS = ["Projects", "Certifications", "Years_Experience", "Internships"]
CATEGORICAL_COLS = ["Target_Role"]
FEATURE_COLS = SKILL_COLS + NUMERIC_EXTRA_COLS + CATEGORICAL_COLS


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    # --- encode categorical ---
    role_encoder = LabelEncoder()
    X = df[FEATURE_COLS].copy()
    X["Target_Role"] = role_encoder.fit_transform(X["Target_Role"])

    # --- classifier: Skill_Gap_Level ---
    level_encoder = LabelEncoder()
    y_clf = level_encoder.fit_transform(df["Skill_Gap_Level"])
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X, y_clf, test_size=0.2, random_state=RANDOM_STATE, stratify=y_clf
    )
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=10, random_state=RANDOM_STATE, class_weight="balanced"
    )
    clf.fit(Xc_train, yc_train)
    clf_acc = accuracy_score(yc_test, clf.predict(Xc_test))

    # --- regressor: Job_Readiness_Score ---
    y_reg = df["Job_Readiness_Score"]
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=RANDOM_STATE
    )
    reg = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=RANDOM_STATE)
    reg.fit(Xr_train, yr_train)
    reg_pred = reg.predict(Xr_test)
    reg_mae = mean_absolute_error(yr_test, reg_pred)
    reg_r2 = r2_score(yr_test, reg_pred)

    # --- per-role average skill profile (for radar-chart benchmarking) ---
    role_profiles = (
        df.groupby("Target_Role")[SKILL_COLS].mean().round(1).to_dict(orient="index")
    )

    # --- overall dataset averages (fallback benchmark) ---
    overall_profile = df[SKILL_COLS].mean().round(1).to_dict()

    # --- save everything ---
    joblib.dump(clf, os.path.join(MODEL_DIR, "skill_gap_level_classifier.pkl"))
    joblib.dump(reg, os.path.join(MODEL_DIR, "job_readiness_regressor.pkl"))
    joblib.dump(role_encoder, os.path.join(MODEL_DIR, "role_encoder.pkl"))
    joblib.dump(level_encoder, os.path.join(MODEL_DIR, "level_encoder.pkl"))

    with open(os.path.join(MODEL_DIR, "role_profiles.json"), "w") as f:
        json.dump(
            {
                "roles": sorted(df["Target_Role"].unique().tolist()),
                "skills": SKILL_COLS,
                "role_profiles": role_profiles,
                "overall_profile": overall_profile,
            },
            f,
            indent=2,
        )

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(
            {
                "classifier_accuracy": round(float(clf_acc), 4),
                "regressor_mae": round(float(reg_mae), 2),
                "regressor_r2": round(float(reg_r2), 4),
            },
            f,
            indent=2,
        )

    print(f"Classifier accuracy: {clf_acc:.3f}")
    print(f"Regressor MAE: {reg_mae:.2f} | R2: {reg_r2:.3f}")
    print(f"Models saved to: {MODEL_DIR}")


if __name__ == "__main__":
    train()
