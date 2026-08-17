"""
AI Skill Gap Predictor - Backend API
====================================

Run:
    python app.py

API:
    http://localhost:8000

Docs:
    http://localhost:8000/docs
"""

import json
import os

import joblib
import pandas as pd
import uvicorn

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import train_models as tm


# ============================================================
# PATHS
# ============================================================

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="AI Skill Gap Predictor API",
    version="1.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODEL FILES
# ============================================================

REQUIRED_FILES = [
    "skill_gap_level_classifier.pkl",
    "job_readiness_regressor.pkl",
    "role_encoder.pkl",
    "level_encoder.pkl",
    "role_profiles.json",
    "metrics.json",
]


# ============================================================
# TRAIN MODELS IF THEY DO NOT EXIST
# ============================================================

if not all(
    os.path.exists(os.path.join(MODEL_DIR, file))
    for file in REQUIRED_FILES
):
    print("No saved models found.")
    print("Training models now...")
    tm.train()
    print("Training completed.")


# ============================================================
# LOAD MODELS
# ============================================================

clf = joblib.load(
    os.path.join(
        MODEL_DIR,
        "skill_gap_level_classifier.pkl"
    )
)

reg = joblib.load(
    os.path.join(
        MODEL_DIR,
        "job_readiness_regressor.pkl"
    )
)

role_encoder = joblib.load(
    os.path.join(
        MODEL_DIR,
        "role_encoder.pkl"
    )
)

level_encoder = joblib.load(
    os.path.join(
        MODEL_DIR,
        "level_encoder.pkl"
    )
)


# ============================================================
# LOAD ROLE DATA
# ============================================================

with open(
    os.path.join(MODEL_DIR, "role_profiles.json"),
    "r"
) as file:
    ROLE_DATA = json.load(file)


# ============================================================
# LOAD MODEL METRICS
# ============================================================

with open(
    os.path.join(MODEL_DIR, "metrics.json"),
    "r"
) as file:
    METRICS = json.load(file)


# ============================================================
# FEATURE INFORMATION
# ============================================================

SKILL_COLS = tm.SKILL_COLS
NUMERIC_EXTRA_COLS = tm.NUMERIC_EXTRA_COLS
FEATURE_COLS = tm.FEATURE_COLS


# ============================================================
# REQUEST MODEL
# ============================================================

class StudentProfile(BaseModel):

    Python: int = Field(
        ..., ge=0, le=100
    )

    Machine_Learning: int = Field(
        ..., ge=0, le=100
    )

    SQL: int = Field(
        ..., ge=0, le=100
    )

    Deep_Learning: int = Field(
        ..., ge=0, le=100
    )

    NLP: int = Field(
        ..., ge=0, le=100
    )

    DSA: int = Field(
        ..., ge=0, le=100
    )

    Git: int = Field(
        ..., ge=0, le=100
    )

    Cloud: int = Field(
        ..., ge=0, le=100
    )

    Statistics: int = Field(
        ..., ge=0, le=100
    )

    Communication: int = Field(
        ..., ge=0, le=100
    )

    Projects: int = Field(
        ..., ge=0, le=20
    )

    Certifications: int = Field(
        ..., ge=0, le=20
    )

    Years_Experience: int = Field(
        ..., ge=0, le=30
    )

    Internships: int = Field(
        ..., ge=0, le=10
    )

    Target_Role: str


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health():

    return {
        "status": "ok",
        "metrics": METRICS
    }


# ============================================================
# GET ROLES
# ============================================================

@app.get("/api/roles")
def get_roles():

    return {
        "roles": ROLE_DATA["roles"]
    }


# ============================================================
# GET SKILLS
# ============================================================

@app.get("/api/skills")
def get_skills():

    return {
        "skills": ROLE_DATA["skills"]
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/api/predict")
def predict(student: StudentProfile):

    # --------------------------------------------------------
    # Check target role
    # --------------------------------------------------------

    if student.Target_Role not in ROLE_DATA["roles"]:

        raise HTTPException(
            status_code=400,
            detail=f"Unknown Target_Role: {student.Target_Role}"
        )


    # --------------------------------------------------------
    # Get all user values
    # --------------------------------------------------------

    user_values = [
        getattr(student, skill)
        for skill in SKILL_COLS
    ]

    extra_values = [
        student.Projects,
        student.Certifications,
        student.Years_Experience,
        student.Internships
    ]


    # ========================================================
    # SPECIAL CASE:
    # EVERYTHING IS ZERO
    # ========================================================

    all_zero = (
        all(value == 0 for value in user_values)
        and
        all(value == 0 for value in extra_values)
    )


    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    student_dict = student.model_dump()

    row = pd.DataFrame(
        [student_dict]
    )[FEATURE_COLS]


    # --------------------------------------------------------
    # Encode Target Role
    # --------------------------------------------------------

    row_encoded = row.copy()

    row_encoded["Target_Role"] = (
        role_encoder.transform(
            row_encoded["Target_Role"]
        )
    )


    # ========================================================
    # SKILL GAP LEVEL
    # ========================================================

    if all_zero:

        gap_level = "high"

        level_confidence = {
            "High": 1.0,
            "Medium": 0.0,
            "Low": 0.0
        }

    else:

        predicted_level = clf.predict(
            row_encoded
        )[0]

        gap_level = (
            level_encoder
            .inverse_transform(
                [predicted_level]
            )[0]
        )

        probabilities = clf.predict_proba(
            row_encoded
        )[0]

        level_confidence = {
            level_encoder.classes_[i]:
                round(float(probabilities[i]), 3)
            for i in range(len(probabilities))
        }


    # ========================================================
    # JOB READINESS
    # ========================================================

    if all_zero:

        # If the user has absolutely no skills,
        # projects, certifications, experience or internships,
        # readiness should logically be zero.

        readiness_score = 0.0

    else:

        readiness_score = float(
            reg.predict(row_encoded)[0]
        )

        # Keep score between 0 and 100

        readiness_score = max(
            0,
            min(
                100,
                readiness_score
            )
        )


    # ========================================================
    # ROLE BENCHMARK
    # ========================================================

    role_profile = ROLE_DATA["role_profiles"].get(
        student.Target_Role,
        ROLE_DATA["overall_profile"]
    )


    # ========================================================
    # SKILL BREAKDOWN
    # ========================================================

    skill_breakdown = []


    for skill in SKILL_COLS:

        user_value = getattr(
            student,
            skill
        )

        benchmark_value = role_profile[skill]

        gap = round(
            benchmark_value - user_value,
            1
        )

        skill_breakdown.append({

            "skill": skill,

            "user_score": user_value,

            "role_benchmark": benchmark_value,

            "gap": gap
        })


    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    recommendations = sorted(
        [
            skill
            for skill in skill_breakdown
            if skill["gap"] > 0
        ],
        key=lambda x: x["gap"],
        reverse=True
    )[:3]


    # ========================================================
    # STRONGEST SKILL
    # ========================================================

    if all_zero:

        strongest_skill = "None"

    else:

        strongest_skill = max(
            skill_breakdown,
            key=lambda x: x["user_score"]
        )["skill"]


    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    return {

        "skill_gap_level": gap_level,

        "skill_gap_level_confidence":
            level_confidence,

        "job_readiness_score":
            round(
                readiness_score,
                1
            ),

        "target_role":
            student.Target_Role,

        "skill_breakdown":
            skill_breakdown,

        "recommendations": [

            {
                "skill":
                    recommendation["skill"]
                    .replace("_", " "),

                "gap":
                    recommendation["gap"],

                "message":
                    (
                        f"You're "
                        f"{recommendation['gap']} pts "
                        f"behind the average "
                        f"{student.Target_Role} "
                        f"on "
                        f"{recommendation['skill'].replace('_', ' ')}."
                    )
            }

            for recommendation in recommendations
        ],

        "strongest_skill":
            strongest_skill.replace("_", " ")
            if strongest_skill != "None"
            else "None"
    }


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
