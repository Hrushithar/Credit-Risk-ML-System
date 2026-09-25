"""Recompute the decision threshold saved in best_threshold.pkl.

Why this exists
---------------
The threshold currently shipped in best_threshold.pkl (0.9996131) was produced by
two mistakes in the notebook:

1. Cell 54 used the wrong F1 formula -- ``2 * (precisions - recalls) / (precisions + recalls)``
   instead of ``2 * (precisions * recalls) / (precisions + recalls)``.
2. The threshold was tuned on the *uncalibrated* ``xg_model`` probabilities, but the
   saved artifact (credit_risk_model.pkl) is the *calibrated* model, whose
   probabilities live on a different scale.

Result: every applicant is predicted as low risk (0 of 6305 test rows were ever
flagged), so the API can never return "High Risk".

This script mirrors notebook cells 20-23 (data cleaning + split), scores the
calibrated model on the same hold-out set, and writes a corrected F1-maximising
threshold back to best_threshold.pkl.

Usage:
    python recompute_threshold.py            # report only, does not write
    python recompute_threshold.py --write    # overwrite best_threshold.pkl
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "credit_risk_model.pkl"
THRESHOLD_PATH = BASE_DIR / "best_threshold.pkl"
DATA_PATH = BASE_DIR / "credit_risk_dataset.csv"


def load_test_split():
    """Rebuild the exact train/test split used in the notebook."""
    df = pd.read_csv(DATA_PATH)
    df = df.copy()
    df.drop_duplicates(inplace=True)
    df = df[df['person_age'] >= 18]
    df = df[df['person_age'] <= 100]
    df = df[df['person_emp_length'] <= df['person_age']]
    df = df[df['person_emp_length'] <= 60]
    df = df[df['loan_amnt'] > 0]

    X = df.drop('loan_status', axis=1)
    y = df['loan_status']

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="overwrite best_threshold.pkl with the recomputed value")
    args = parser.parse_args()

    _, X_test, _, y_test = load_test_split()

    model = joblib.load(MODEL_PATH)
    current = float(joblib.load(THRESHOLD_PATH))

    probability = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, probability)

    # Correct F1 for every threshold (the notebook's formula was wrong).
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_threshold = float(thresholds[np.argmax(f1_scores[:-1])])

    for label, threshold in (("current", current), ("recomputed", best_threshold)):
        prediction = (probability >= threshold).astype(int)
        print(f"{label:<11} threshold={threshold:.6f}  "
              f"flagged={prediction.sum():>5}/{len(prediction)}  "
              f"f1={f1_score(y_test, prediction):.4f}")

    if args.write:
        joblib.dump(best_threshold, THRESHOLD_PATH)
        print(f"\nWrote {THRESHOLD_PATH.name} = {best_threshold:.6f}")
    else:
        print("\nNothing written. Re-run with --write to update best_threshold.pkl.")


if __name__ == "__main__":
    main()
