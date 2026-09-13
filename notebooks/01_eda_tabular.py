"""
Day 2 — EDA on the tabular diabetes dataset (BRFSS 2015)
Run with: python3 notebooks/01_eda_tabular.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---- Setup ----
DATA_PATH = "data/raw/diabetes_tabular.csv"
FIG_DIR = "reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)
sns.set(style="whitegrid")

# ---- Load ----
df = pd.read_csv(DATA_PATH)
print("=" * 60)
print("SHAPE:", df.shape)
print("=" * 60)

# ---- Basic info ----
print("\nCOLUMN TYPES:")
print(df.dtypes)

print("\nMISSING VALUES PER COLUMN:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values found.")

print("\nDUPLICATE ROWS:", df.duplicated().sum())

# ---- Target class balance ----
print("\nTARGET CLASS BALANCE (Diabetes_binary):")
print(df["Diabetes_binary"].value_counts())
print(df["Diabetes_binary"].value_counts(normalize=True).round(3))

plt.figure(figsize=(5, 4))
sns.countplot(x="Diabetes_binary", data=df)
plt.title("Class Balance: Diabetes_binary")
plt.savefig(f"{FIG_DIR}/class_balance.png", bbox_inches="tight")
plt.close()

# ---- Descriptive stats ----
print("\nDESCRIPTIVE STATS:")
print(df.describe().T)

# ---- Key distributions ----
for col in ["BMI", "Age", "GenHlth", "MentHlth", "PhysHlth"]:
    plt.figure(figsize=(5, 4))
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f"Distribution: {col}")
    plt.savefig(f"{FIG_DIR}/dist_{col}.png", bbox_inches="tight")
    plt.close()

# ---- Correlation with target ----
print("\nCORRELATION WITH TARGET (Diabetes_binary), sorted:")
corr_with_target = df.corr(numeric_only=True)["Diabetes_binary"].sort_values(ascending=False)
print(corr_with_target)

# ---- Full correlation heatmap ----
plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", center=0, annot=False)
plt.title("Correlation Heatmap - All Features")
plt.savefig(f"{FIG_DIR}/correlation_heatmap.png", bbox_inches="tight")
plt.close()

# ---- Domain grouping preview (for later SHAP domain attribution) ----
domains = {
    "Health": ["HighBP", "HighChol", "CholCheck", "BMI", "Stroke", "HeartDiseaseorAttack", "GenHlth"],
    "Lifestyle": ["Smoker", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump"],
    "Demographic": ["Sex", "Age", "Education", "Income"],
    "HealthcareAccess": ["AnyHealthcare", "NoDocbcCost"],
}
print("\nFEATURE DOMAIN MAP (for later domain-level SHAP attribution):")
for domain, cols in domains.items():
    print(f"  {domain}: {cols}")

print("\n" + "=" * 60)
print("EDA COMPLETE. Figures saved to:", FIG_DIR)
print("=" * 60)