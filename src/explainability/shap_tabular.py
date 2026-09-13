"""
Day 10 — SHAP explainability on the best tabular model
(Logistic Regression, Combined features, ROC-AUC 0.8126)
Run with: python3 src/explainability/shap_tabular.py
"""

import pandas as pd
import numpy as np
import json
import joblib
import shap
import matplotlib.pyplot as plt
import os

DATA_DIR = "data/processed"
MODEL_DIR = "models_saved"
FIG_DIR = "reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)

TARGET = "Diabetes_binary"
BEST_MODEL_PATH = f"{MODEL_DIR}/combined_LogisticRegression.pkl"

# ---- Load model, data, feature domains ----
model = joblib.load(BEST_MODEL_PATH)
val_df = pd.read_csv(f"{DATA_DIR}/val.csv")

with open(f"{DATA_DIR}/feature_sets.json") as f:
    feature_sets = json.load(f)
features = feature_sets["combined"]
X_val = val_df[features]

DOMAINS = {
    "Health": ["HighBP", "HighChol", "CholCheck", "BMI", "Stroke", "HeartDiseaseorAttack", "GenHlth"],
    "Lifestyle": ["Smoker", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump"],
    "Demographic": ["Sex", "Age", "Education", "Income"],
    "HealthcareAccess": ["AnyHealthcare", "NoDocbcCost"],
    "Other": ["MentHlth", "PhysHlth", "DiffWalk"],
}

# ---- SHAP on a sample (SHAP on full 34k rows is slow for LR with many features) ----
SAMPLE_SIZE = 2000
X_sample = X_val.sample(n=SAMPLE_SIZE, random_state=42)

print(f"Computing SHAP values on a sample of {SAMPLE_SIZE} validation rows...")
explainer = shap.LinearExplainer(model, X_val)  # background = full val set
shap_values = explainer.shap_values(X_sample)

# ---- 1. Global feature importance (summary plot) ----
plt.figure()
shap.summary_plot(shap_values, X_sample, show=False)
plt.title("SHAP Summary: Feature Impact on Diabetes Risk Prediction")
plt.savefig(f"{FIG_DIR}/shap_summary.png", bbox_inches="tight", dpi=150)
plt.close()
print(f"Saved: {FIG_DIR}/shap_summary.png")

# ---- 2. Mean |SHAP value| per feature, printed and ranked ----
mean_abs_shap = pd.DataFrame({
    "feature": features,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)

print("\n" + "=" * 60)
print("GLOBAL FEATURE IMPORTANCE (mean |SHAP value|)")
print("=" * 60)
print(mean_abs_shap.to_string(index=False))
mean_abs_shap.to_csv("reports/shap_feature_importance.csv", index=False)

# ---- 3. Domain-level attribution (sum feature importance within each domain) ----
domain_importance = {}
for domain, cols in DOMAINS.items():
    cols_present = [c for c in cols if c in features]
    domain_importance[domain] = mean_abs_shap[
        mean_abs_shap["feature"].isin(cols_present)
    ]["mean_abs_shap"].sum()

domain_df = pd.DataFrame(
    list(domain_importance.items()), columns=["domain", "total_importance"]
).sort_values("total_importance", ascending=False)

total = domain_df["total_importance"].sum()
domain_df["pct_contribution"] = (domain_df["total_importance"] / total * 100).round(1)

print("\n" + "=" * 60)
print("DOMAIN-LEVEL ATTRIBUTION (which information type matters most)")
print("=" * 60)
print(domain_df.to_string(index=False))
domain_df.to_csv("reports/shap_domain_attribution.csv", index=False)

# ---- Domain attribution bar chart ----
plt.figure(figsize=(7, 5))
plt.bar(domain_df["domain"], domain_df["pct_contribution"], color="teal")
plt.ylabel("% Contribution to Prediction")
plt.title("Domain-Level Attribution (SHAP)")
plt.xticks(rotation=20)
for i, v in enumerate(domain_df["pct_contribution"]):
    plt.text(i, v + 1, f"{v}%", ha="center")
plt.savefig(f"{FIG_DIR}/shap_domain_attribution.png", bbox_inches="tight")
plt.close()
print(f"\nSaved: {FIG_DIR}/shap_domain_attribution.png")

# ---- 4. Single-patient explanation example (for the dashboard later) ----
example_idx = 0
example_row = X_sample.iloc[[example_idx]]
example_shap = shap_values[example_idx]

print("\n" + "=" * 60)
print("EXAMPLE: Single-Patient Explanation (for dashboard prototype)")
print("=" * 60)
example_df = pd.DataFrame({
    "feature": features,
    "value": example_row.values[0],
    "shap_contribution": example_shap
}).sort_values("shap_contribution", key=abs, ascending=False)
print(example_df.head(10).to_string(index=False))

print("\nSHAP EXPLAINABILITY COMPLETE.")
print("IMPORTANT: SHAP explains what the MODEL relied on, not biological")
print("causation. We report this as model behavior, not medical fact.")
