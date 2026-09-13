"""
Day 6 — Train tabular models: Experiment 1 (Health-only) & Experiment 2 (Lifestyle-only)
Run with: python3 src/models/train_tabular.py
"""

import pandas as pd
import numpy as np
import json
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score, precision_score,
    accuracy_score, confusion_matrix
)

DATA_DIR = "data/processed"
MODEL_DIR = "models_saved"
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET = "Diabetes_binary"

# ---- Load data ----
train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
val_df = pd.read_csv(f"{DATA_DIR}/val.csv")

with open(f"{DATA_DIR}/feature_sets.json") as f:
    feature_sets = json.load(f)

y_train = train_df[TARGET]
y_val = val_df[TARGET]

# ---- Model factory (imbalance handled via class weights, not resampling) ----
def get_models():
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()  # for XGBoost scale_pos_weight
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, scale_pos_weight=pos_weight,
            eval_metric="logloss", random_state=42, n_jobs=-1
        ),
    }

def evaluate(model, X_val, y_val, threshold=0.5):
    proba = model.predict_proba(X_val)[:, 1]
    preds = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_val, proba),
        "f1": f1_score(y_val, preds),
        "recall": recall_score(y_val, preds),
        "precision": precision_score(y_val, preds),
        "accuracy": accuracy_score(y_val, preds),
    }

# ---- Run experiments ----
EXPERIMENTS = ["health_only", "lifestyle_only"]  # combined comes Day 7
results = []

for exp_name in EXPERIMENTS:
    features = feature_sets[exp_name]
    X_train = train_df[features]
    X_val = val_df[features]

    print("=" * 60)
    print(f"EXPERIMENT: {exp_name}  ({len(features)} features: {features})")
    print("=" * 60)

    models = get_models()
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_val, y_val)
        metrics["experiment"] = exp_name
        metrics["model"] = model_name
        results.append(metrics)

        print(f"\n  {model_name}:")
        for k, v in metrics.items():
            if k not in ["experiment", "model"]:
                print(f"    {k:10s}: {v:.4f}")

        # Save every model (small dataset, cheap to keep all for comparison)
        joblib.dump(model, f"{MODEL_DIR}/{exp_name}_{model_name}.pkl")

# ---- Comparison table ----
results_df = pd.DataFrame(results)[
    ["experiment", "model", "roc_auc", "f1", "recall", "precision", "accuracy"]
]
results_df = results_df.sort_values(["experiment", "roc_auc"], ascending=[True, False])

print("\n" + "=" * 60)
print("COMPARISON TABLE (sorted by ROC-AUC within each experiment)")
print("=" * 60)
print(results_df.to_string(index=False))

results_df.to_csv("reports/experiment1_2_results.csv", index=False)
print("\nSaved comparison table to reports/experiment1_2_results.csv")

print("\nNOTE: Accuracy is reported for completeness only — given the 84.7/15.3")
print("class imbalance, ROC-AUC, recall, and F1 are the metrics that actually")
print("tell us whether the model is useful, per our evaluation plan.")
