from pathlib import Path
import json

import joblib
import pandas as pd


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
MODELS_DIR = PROJECT_ROOT / "models"


print("=" * 80)
print("TELCO CUSTOMER CHURN CAPSTONE - FINAL PROJECT VALIDATION")
print("=" * 80)


# =========================================================
# REQUIRED FILES
# =========================================================

required_files = [
    # Raw / processed data
    DATA_DIR / "Telco-Customer-Churn.csv",
    DATA_DIR / "telco_churn_cleaned.csv",
    DATA_DIR / "telco_churn_segmented.csv",
    DATA_DIR / "telco_churn_segmented_labeled.csv",
    # Important analysis outputs
    TABLES_DIR / "data_validation_report.txt",
    TABLES_DIR / "data_cleaning_summary.txt",
    TABLES_DIR / "eda_summary.txt",
    TABLES_DIR / "model_comparison.csv",
    TABLES_DIR / "model_summary.json",
    TABLES_DIR / "threshold_error_summary.txt",
    TABLES_DIR / "clustering_summary.txt",
    TABLES_DIR / "segment_business_summary.txt",
    TABLES_DIR / "final_customer_risk_register.csv",
    TABLES_DIR / "final_risk_scoring_summary.txt",
    # Models
    MODELS_DIR / "best_churn_model.joblib",
    MODELS_DIR / "final_churn_model_full.joblib",
]


print("\nChecking required project files...")

missing_files = []

for file_path in required_files:
    status = "OK" if file_path.exists() else "MISSING"

    print(f"{status:8} {file_path.relative_to(PROJECT_ROOT)}")

    if not file_path.exists():
        missing_files.append(file_path)


# =========================================================
# FIGURE VALIDATION
# =========================================================

figure_files = sorted(FIGURES_DIR.glob("*.png"))

print("\nFigure count:")
print(len(figure_files))

if len(figure_files) >= 35:
    print("Figure validation: PASS")
else:
    print("Figure validation: WARNING")


print("\nFigure list:")

for figure in figure_files:
    print("-", figure.name)


# =========================================================
# CLEAN DATA VALIDATION
# =========================================================

cleaned_df = pd.read_csv(DATA_DIR / "telco_churn_cleaned.csv")

print("\n" + "=" * 80)
print("CLEANED DATA VALIDATION")
print("=" * 80)

print("\nShape:", cleaned_df.shape)

print("Missing values:", cleaned_df.isna().sum().sum())

print("Duplicate rows:", cleaned_df.duplicated().sum())

print("Duplicate customer IDs:", cleaned_df["customerID"].duplicated().sum())

print("Churn rate:", f"{cleaned_df['ChurnValue'].mean() * 100:.2f}%")


# =========================================================
# SEGMENT DATA VALIDATION
# =========================================================

segmented_df = pd.read_csv(DATA_DIR / "telco_churn_segmented_labeled.csv")

print("\n" + "=" * 80)
print("SEGMENTATION VALIDATION")
print("=" * 80)

print("\nShape:", segmented_df.shape)

print("\nCluster counts:")

print(segmented_df["Cluster"].value_counts().sort_index())

print("\nSegment counts:")

print(segmented_df["Segment"].value_counts())

print("\nSegment churn rates:")

print(
    (
        segmented_df.groupby("Segment", observed=True)["ChurnValue"]
        .mean()
        .mul(100)
        .round(2)
    )
)


# =========================================================
# RISK REGISTER VALIDATION
# =========================================================

risk_df = pd.read_csv(TABLES_DIR / "final_customer_risk_register.csv")

print("\n" + "=" * 80)
print("FINAL RISK REGISTER VALIDATION")
print("=" * 80)

print("\nShape:", risk_df.shape)

print("Unique customers:", risk_df["customerID"].nunique())

print("Missing risk probabilities:", risk_df["ChurnRiskProbability"].isna().sum())

print("Minimum risk probability:", round(risk_df["ChurnRiskProbability"].min(), 4))

print("Maximum risk probability:", round(risk_df["ChurnRiskProbability"].max(), 4))


print("\nRisk bands:")

print(risk_df["RiskBand"].value_counts())


print("\nRetention priorities:")

print(risk_df["RetentionPriority"].value_counts())


# =========================================================
# MODEL RESULTS VALIDATION
# =========================================================

model_results = pd.read_csv(TABLES_DIR / "model_comparison.csv")

print("\n" + "=" * 80)
print("MODEL RESULTS")
print("=" * 80)

print("\n", model_results.round(4))


with open(TABLES_DIR / "model_summary.json", "r", encoding="utf-8") as file:
    model_summary = json.load(file)


print("\nSelected model:")
print(model_summary["selected_model"])

print("\nTest ROC-AUC:", round(model_summary["test_roc_auc"], 4))


# =========================================================
# LOAD MODEL TEST
# =========================================================

print("\nTesting saved model files...")

try:
    best_model = joblib.load(MODELS_DIR / "best_churn_model.joblib")

    print("best_churn_model.joblib: OK")

except Exception as exc:
    print("best_churn_model.joblib: FAILED")

    print(exc)


try:
    final_model = joblib.load(MODELS_DIR / "final_churn_model_full.joblib")

    print("final_churn_model_full.joblib: OK")

except Exception as exc:
    print("final_churn_model_full.joblib: FAILED")

    print(exc)


# =========================================================
# KEY BUSINESS RESULTS
# =========================================================

print("\n" + "=" * 80)
print("KEY BUSINESS RESULTS")
print("=" * 80)


overall_churn = cleaned_df["ChurnValue"].mean() * 100


segment_summary = segmented_df.groupby("Segment", observed=True).agg(
    Customers=("customerID", "count"), ChurnRate=("ChurnValue", "mean")
)

segment_summary["ChurnRate"] *= 100


highest_risk_segment = segment_summary.sort_values("ChurnRate", ascending=False).iloc[0]


highest_risk_name = segment_summary["ChurnRate"].idxmax()


priority_summary = risk_df.groupby("RetentionPriority", observed=True).agg(
    Customers=("customerID", "count"), ChurnRate=("ChurnValue", "mean")
)

priority_summary["ChurnRate"] *= 100


print(f"\nOverall churn rate: {overall_churn:.2f}%")

print(f"Highest-risk segment: {highest_risk_name}")

print(f"Highest-risk segment churn: {highest_risk_segment['ChurnRate']:.2f}%")


if "Critical" in priority_summary.index:
    print(
        f"Critical-priority churn: {priority_summary.loc['Critical', 'ChurnRate']:.2f}%"
    )


if "Low" in priority_summary.index:
    print(f"Low-priority churn: {priority_summary.loc['Low', 'ChurnRate']:.2f}%")


# =========================================================
# FINAL PROJECT STATUS
# =========================================================

print("\n" + "=" * 80)

if (
    len(missing_files) == 0
    and cleaned_df.isna().sum().sum() == 0
    and cleaned_df.duplicated().sum() == 0
    and len(figure_files) >= 35
    and risk_df["customerID"].nunique() == 7043
):
    print("FINAL PROJECT VALIDATION: PASSED")

else:
    print("FINAL PROJECT VALIDATION: CHECK WARNINGS ABOVE")

print("=" * 80)


# =========================================================
# SAVE VALIDATION SUMMARY
# =========================================================

validation_path = TABLES_DIR / "final_project_validation.txt"

with open(validation_path, "w", encoding="utf-8") as file:
    file.write("TELCO CUSTOMER CHURN CAPSTONE - FINAL VALIDATION\n")

    file.write("=" * 70 + "\n\n")

    file.write(f"Cleaned dataset shape: {cleaned_df.shape}\n")

    file.write(f"Missing values: {cleaned_df.isna().sum().sum()}\n")

    file.write(f"Duplicate rows: {cleaned_df.duplicated().sum()}\n")

    file.write(f"Figures generated: {len(figure_files)}\n")

    file.write(f"Risk-register customers: {risk_df['customerID'].nunique()}\n")

    file.write(f"Overall churn: {overall_churn:.2f}%\n")

    file.write(f"Selected model: {model_summary['selected_model']}\n")

    file.write(f"Test ROC-AUC: {model_summary['test_roc_auc']:.4f}\n")

    file.write(f"Highest-risk segment: {highest_risk_name}\n")

    file.write(
        f"Highest-risk segment churn: {highest_risk_segment['ChurnRate']:.2f}%\n"
    )

    if len(missing_files) == 0:
        file.write("Required files: PASS\n")

    else:
        file.write("Required files: FAIL\n")


print("\nValidation report saved to:")

print(validation_path)
