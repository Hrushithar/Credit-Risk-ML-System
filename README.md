# Credit-Risk-ML-System

Explainable credit-risk scoring API (FastAPI + XGBoost) that serves a static
underwriting UI from the same service.

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
