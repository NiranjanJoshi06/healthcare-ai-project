"""
Day 7 — Train tabular models: Experiment 3 (Combined) + full 3-way comparison
Run with: python3 src/models/train_tabular_exp3.py
"""

import pandas as pd
import numpy as np
import json
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score, precision_score, accuracy_score
)

DATA_DIR = "data/processed"
MODEL_DIR = "models_saved"
FIG_DIR = "reports/figures"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

TARGET = "Diabetes_binary"

# ---- Load data ----
train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
val_df = pd.read_csv(f"{DATA_DIR}/val.csv")

with open(f"{DATA_DIR}/feature_sets.json") as f:
    feature_sets = json.load(f)

y_train = train_df[TARGET]
y_val = val_df[TARGET]

def get_models():
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
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

# ---- Experiment 3: Combined ----
exp_name = "combined"
features = feature_sets[exp_name]
X_train = train_df[features]
X_val = val_df[features]

print("=" * 60)
print(f"EXPERIMENT: {exp_name}  ({len(features)} features)")
print("=" * 60)

results = []
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

    joblib.dump(model, f"{MODEL_DIR}/{exp_name}_{model_name}.pkl")

combined_results_df = pd.DataFrame(results)[
    ["experiment", "model", "roc_auc", "f1", "recall", "precision", "accuracy"]
]
combined_results_df.to_csv("reports/experiment3_results.csv", index=False)

# ---- Merge with Experiment 1 & 2 results for full comparison ----
prev_results = pd.read_csv("reports/experiment1_2_results.csv")
all_results = pd.concat([prev_results, combined_results_df], ignore_index=True)
all_results = all_results.sort_values(["experiment", "roc_auc"], ascending=[True, False])

print("\n" + "=" * 60)
print("FULL 3-EXPERIMENT COMPARISON (Health-only vs Lifestyle-only vs Combined)")
print("=" * 60)
print(all_results.to_string(index=False))

all_results.to_csv("reports/all_experiments_results.csv", index=False)
print("\nSaved full comparison to reports/all_experiments_results.csv")

# ---- Best model per experiment, for the bar chart ----
best_per_exp = all_results.loc[all_results.groupby("experiment")["roc_auc"].idxmax()]

plt.figure(figsize=(7, 5))
plt.bar(best_per_exp["experiment"], best_per_exp["roc_auc"], color=["#4C72B0", "#DD8452", "#55A868"])
plt.ylabel("ROC-AUC (best model per experiment)")
plt.title("Does Combining Health + Lifestyle Improve Prediction?")
plt.ylim(0, 1)
for i, v in enumerate(best_per_exp["roc_auc"]):
    plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
plt.savefig(f"{FIG_DIR}/experiment_comparison_roc_auc.png", bbox_inches="tight")
plt.close()

print("\nSaved comparison chart: reports/figures/experiment_comparison_roc_auc.png")

# ---- Answer the core research question ----
health_best = all_results[all_results["experiment"] == "health_only"]["roc_auc"].max()
combined_best = all_results[all_results["experiment"] == "combined"]["roc_auc"].max()
diff = combined_best - health_best

print("\n" + "=" * 60)
print("RESEARCH QUESTION: Does adding lifestyle+demographic data to Health-only")
print("genuinely improve prediction?")
print("=" * 60)
print(f"Health-only best ROC-AUC:  {health_best:.4f}")
print(f"Combined best ROC-AUC:     {combined_best:.4f}")
print(f"Improvement:               {diff:+.4f}")
if diff > 0.01:
    print("=> Meaningful improvement: combining modalities is justified.")
elif diff > 0:
    print("=> Marginal improvement: combining helps slightly, report honestly.")
else:
    print("=> No improvement: health features alone are sufficient here.")
print("This is a genuine finding either way — we report it as-is, not as we'd prefer it.")
