# AI-Skill-Gap-Predictor
Machine Learning based AI Skill Gap Predictor using Python and FastAPI
# Skill Scan — AI Skill Gap Predictor (Frontend + Backend)

A full-stack app on top of the ML models: a **FastAPI backend** serves live
predictions, a **vanilla HTML/CSS/JS frontend** ("Skill Scan" diagnostic
console) lets you set skill levels and get an instant readiness scan.

```
skillgap_app/
├── backend/
│   ├── app.py                 # FastAPI app (the API)
│   ├── train_models.py        # trains + saves the models (run once)
│   ├── requirements.txt
│   ├── ai_skill_gap_predictor_dataset-1.csv
│   └── models/                # saved .pkl models + profiles (generated)
└── frontend/
    ├── index.html
    ├── styles.css
    └── app.js
```

## 1. Run the backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python train_models.py          # trains models, saves to backend/models/
python app.py                   # starts API at http://localhost:8000
```

Check it's alive: open http://localhost:8000/docs (interactive Swagger UI).

> `app.py` also auto-trains the models on first run if `models/` is empty,
> so `python train_models.py` is optional — but running it explicitly first
> is faster to debug if something goes wrong.

## 2. Run the frontend

The frontend is plain HTML/CSS/JS — no build step. In a **second terminal**:

```bash
cd frontend
python -m http.server 5500
```

Then open **http://localhost:5500** in your browser.

(You can also just double-click `index.html`, but serving it avoids
occasional browser CORS quirks with `file://` URLs.)

## 3. Use it

1. Pick a **target role** from the dropdown.
2. Drag the skill meters to your levels (0–100), set projects/certs/
   experience/internships.
3. Click **Run scan**.
4. You'll get:
   - A **readiness gauge** (0–100, from the regression model)
   - A **skill gap level** badge (Low / Medium / High, from the classifier,
     with confidence breakdown)
   - A **radar chart** comparing your profile against the average profile
     of students who targeted that same role
   - **Priority gaps** — the 3 skills where you're furthest behind the
     role benchmark

## API reference

| Method | Path            | Description                                   |
|--------|-----------------|------------------------------------------------|
| GET    | `/api/health`   | API status + model metrics                     |
| GET    | `/api/roles`    | List of target roles                            |
| GET    | `/api/skills`   | List of skill feature names                     |
| POST   | `/api/predict`  | Body: skill scores + extras + `Target_Role` → readiness score, gap level, radar/benchmark data, recommendations |

Example `POST /api/predict` body:

```json
{
  "Python": 70, "Machine_Learning": 55, "SQL": 60, "Deep_Learning": 40,
  "NLP": 30, "DSA": 50, "Git": 65, "Cloud": 45, "Statistics": 58,
  "Communication": 72, "Projects": 3, "Certifications": 2,
  "Years_Experience": 1, "Internships": 1, "Target_Role": "Data Scientist"
}
```

## Notes

- CORS is wide open (`allow_origins=["*"]`) since this is a local dev setup.
  Lock it down before deploying anywhere public.
- Models are RandomForest (classifier + regressor), trained on
  `ai_skill_gap_predictor_dataset-1.csv`. Retrain any time with
  `python train_models.py`.
- The `Skill_Gap_Level = High` class has only 3 examples in the whole
  dataset, so the classifier essentially never predicts it — worth
  rebalancing the data if that class matters to you.
