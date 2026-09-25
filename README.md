# Credit-Risk-ML-System

**Credit Ledger** — an explainable credit-risk scoring service. A calibrated
XGBoost model estimates the probability that a consumer-loan applicant will
default, a tuned probability cut-off turns that score into a High/Low Risk
verdict, and the whole thing is served through a FastAPI JSON API that also
hosts a static underwriting UI from the same process.

## What this project does

This is an end-to-end, production-shaped machine-learning system for
**consumer credit-risk assessment (probability of default)**. Given a loan
application it answers one question:

> *How likely is this applicant to default, and should we flag them as high risk?*

**Input** — a single loan application with 11 fields:

| Group | Fields |
| ----- | ------ |
| Applicant | `person_age`, `person_income`, `person_home_ownership`, `person_emp_length` |
| Loan request | `loan_intent`, `loan_grade`, `loan_amnt`, `loan_int_rate`, `loan_percent_income` |
| Credit bureau file | `cb_person_default_on_file`, `cb_person_cred_hist_length` |

**Output** — a JSON scoring response:

```json
{
  "default_probability": 0.0262,
  "default_prediction": 0,
  "threshold": 0.6265,
  "Result": "Low Risk"
}
```

`default_probability` is the calibrated probability of default (class 1),
`default_prediction` is that probability compared against `threshold`, and
`Result` is the human-readable verdict.

### What happens under the hood

1. **Validation** — the FastAPI layer validates and types every field with a
   Pydantic `LoanApplication` model before it touches the ML code.
2. **Preprocessing** — the saved scikit-learn `Pipeline` imputes missing numeric
   values (median) and missing categories (`"Missing"`), then one-hot encodes the
   four categorical columns. No scaling is needed because the final model is
   tree-based.
3. **Scoring** — the calibrated XGBoost classifier returns `predict_proba`.
4. **Decision** — the probability is compared against the tuned decision
   threshold in `best_threshold.pkl` to produce High/Low Risk.
5. **Explanation** — the notebook uses SHAP to explain both the global feature
   ranking and individual (local) applicant decisions.

The web service never re-trains anything. Training, evaluation, calibration,
threshold tuning and SHAP analysis all live in `Credit_Risk.ipynb`; the API just
loads the two saved artifacts (`credit_risk_model.pkl`, `best_threshold.pkl`) at
startup and scores requests.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Open <http://127.0.0.1:8000>.

## Endpoints

| Endpoint  | Method | Purpose                                            |
| --------- | ------ | -------------------------------------------------- |
| `/`       | GET    | Static underwriting form (`static/`)               |
| `/health` | GET    | Readiness check — also used as Render's health check |
| `/predict`| POST   | Score one loan application                          |
| `/docs`   | GET    | Swagger UI                                          |

## Technology stack

| Layer | Technology | Why it is used |
| ----- | ---------- | -------------- |
| Language | Python 3.14 | One language across training, serving and tooling |
| Data wrangling | pandas, NumPy | Loading, cleaning, duplicate/outlier handling, split logic |
| ML framework | scikit-learn | `Pipeline`, `ColumnTransformer`, imputation, encoding, CV, calibration, metrics |
| Gradient boosting | XGBoost | Main classifier — strong tabular performance, imbalance handled via `scale_pos_weight` |
| Explainability | SHAP | Global and per-applicant ("local") explanations of the model |
| Visualisation | Matplotlib, Seaborn | EDA plots, confusion matrices, precision-recall and calibration curves |
| Experimentation | Jupyter Notebook | One reproducible home for the whole modelling workflow |
| Model persistence | joblib | Serialising the fitted pipeline and the decision threshold |
| Web API | FastAPI + Pydantic | Typed JSON API, request validation, auto-generated Swagger docs |
| ASGI server | Uvicorn | Serves the FastAPI app locally and on Render |
| Frontend | Static HTML + CSS + vanilla JavaScript | The "Credit Ledger" underwriting form — no build step |
| Deployment | Render (`render.yaml` Blueprint) | Free-tier hosting with a `/health` readiness check |
| Version control | Git / GitHub | Source and artifact tracking |

## Models and machine-learning approach

The project trains more than one model so the final choice is justified rather
than assumed, then calibrates and thresholds the winner.

### 1. Baseline — Logistic Regression
An interpretable linear model wrapped in the same preprocessing pipeline, with
`class_weight="balanced"` and `max_iter=1000`. It is the benchmark every
stronger model has to beat. It is scale-sensitive, so the numeric columns are
standardised (`StandardScaler`) for this variant only.

### 2. Main model — XGBoost (`XGBClassifier`)
Gradient-boosted decision trees, the usual strong choice for tabular credit
data. The XGBoost preprocessing variant skips scaling because trees are
scale-invariant. Class imbalance is handled with
`scale_pos_weight = negatives / positives` (≈3.6:1 on this data) instead of
resampling.

### 3. Tuned model — `RandomizedSearchCV` over XGBoost
`RandomizedSearchCV` runs **150 iterations** with 5-fold **stratified**
cross-validation and optimises **average precision** (`average_precision`, the
area under the precision-recall curve — the right metric for imbalanced default
prediction). The search space:

`n_estimators` (150–600), `max_depth` (3–9), `learning_rate` (0.01–0.5),
`subsample` (0.6–1.0), `colsample_bytree` (0.6–1.0),
`min_child_weight` (1–10), `gamma` (0–5).

### 4. Final artifact — Calibrated XGBoost
Boosted-tree probabilities are not automatically trustworthy as true
probabilities. The tuned model is wrapped in
`CalibratedClassifierCV(method="sigmoid", cv=5)` (Platt scaling), so a 30% score
means roughly a 30% chance of default. A calibration curve compares the
uncalibrated and calibrated models against the perfect diagonal. **This
calibrated model is what `credit_risk_model.pkl` contains and what the API
serves.**

### Evaluation
A single `evaluate_model` helper reports accuracy, precision, recall, F1, a
confusion matrix and a precision-recall curve for every model. Models are also
compared with 5-fold stratified cross-validation (`ROC-AUC`, accuracy,
precision, recall, F1). False positives (safe applicants marked risky) and false
negatives (risky applicants marked safe) are inspected separately.

### Choosing the cut-off
A model outputs a probability, not a yes/no. The notebook sweeps thresholds with
`precision_recall_curve` on the **calibrated** probabilities and selects the one
that maximises F1. That value is stored in `best_threshold.pkl` and applied by
the API; `recompute_threshold.py` reproduces this step outside the notebook.

### Explainability (SHAP)
`shap.TreeExplainer` explains the trained XGBoost classifier over the
one-hot-encoded test set:
- **Global** — `shap.summary_plot` shows which features drive risk overall and in
  which direction.
- **Local** — `shap.plots.waterfall` breaks a single applicant's score down
  feature by feature.

## Dataset

`credit_risk_dataset.csv` — 32,581 loan applications, 12 columns, target
`loan_status` (1 = default, 0 = repaid).

- Default rate ≈ **21.8%** (7,108 defaults / 25,473 repayments) → imbalanced.
- Missing values: `person_emp_length` (895) and `loan_int_rate` (3,116), imputed
  inside the pipeline.
- 165 duplicate rows, removed during cleaning.

Cleaning (`Credit_Risk.ipynb` §3, mirrored by `recompute_threshold.py`) drops
duplicates, keeps ages 18–100, enforces `person_emp_length <= person_age` and
`<= 60`, and requires `loan_amnt > 0`. The cleaned data is split 80/20 with
`train_test_split(..., stratify=y, random_state=42)`.

## Project structure

| Path | Purpose |
| ---- | ------- |
| `main.py` | FastAPI app — loads artifacts, exposes `/`, `/health`, `/predict`, `/docs` |
| `static/` | "Credit Ledger" UI (`index.html`, `style.css`, `script.js`) |
| `Credit_Risk.ipynb` | EDA → cleaning → training → tuning → calibration → threshold → SHAP |
| `credit_risk_model.pkl` | Final calibrated XGBoost pipeline (served by the API) |
| `best_threshold.pkl` | F1-maximising decision threshold |
| `credit_risk_dataset.csv` | Raw training data |
| `recompute_threshold.py` | Rebuilds the threshold against the saved model |
| `requirements.txt` | Pinned Python dependencies |
| `render.yaml`, `.python-version` | Render deployment config and Python version pin |

## Deploy on Render

| Setting           | Value                                             |
| ----------------- | ------------------------------------------------- |
| Runtime           | `Python 3`                                        |
| Build Command     | `pip install -r requirements.txt`                 |
| Start Command     | `uvicorn main:app --host 0.0.0.0 --port $PORT`    |
| Health Check Path | `/health`                                         |
| Python version    | `3.14.3`                                          |

`render.yaml` already contains all of the above for a Blueprint deploy.

**Python version:** set the `PYTHON_VERSION` environment variable to `3.14.3`
(fully qualified) and/or keep the committed `.python-version` file. Render looks
at `PYTHON_VERSION` first, then `.python-version`, then its own default.

> Render does **not** read `runtime.txt` (that is a Heroku convention), so the
> Python version is pinned with `PYTHON_VERSION` / `.python-version` instead.

The pinned dependencies are all available as Linux wheels for Python 3.14, so
the Render build installs from wheels without compiling anything.

## Decision threshold

`best_threshold.pkl` holds the probability cut-off used to turn the model's
score into a High/Low Risk verdict. Rebuild it against the current model with:

```bash
python recompute_threshold.py --write
```

This mirrors notebook cells 21–23 so the notebook, the saved model, and the
served API all use the same threshold.

## Model artifacts

| File                    | Contents                                              |
| ----------------------- | ----------------------------------------------------- |
| `credit_risk_model.pkl` | Calibrated XGBoost pipeline (preprocessing + model)    |
| `best_threshold.pkl`    | Decision threshold applied to `predict_proba`          |
| `Credit_Risk.ipynb`     | Training / evaluation / explanation notebook           |
