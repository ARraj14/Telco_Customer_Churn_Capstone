from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.base import clone
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
)


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "telco_churn_segmented_labeled.csv"

MODEL_PATH = PROJECT_ROOT / "models" / "best_churn_model.joblib"

FINAL_MODEL_PATH = PROJECT_ROOT / "models" / "final_churn_model_full.joblib"

TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

TABLES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# LOAD DATA AND MODEL
# =========================================================

df = pd.read_csv(DATA_PATH)

base_model = joblib.load(MODEL_PATH)

print("=" * 80)
print("TELCO CUSTOMER CHURN - FINAL RISK SCORING SYSTEM")
print("=" * 80)

print("\nDataset shape:")
print(df.shape)


# =========================================================
# MODEL FEATURES
# =========================================================

numeric_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "NumServices",
    "HasInternetService",
    "ContractValue",
    "AutomaticPayment",
]

categorical_features = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaperlessBilling",
    "PaymentMethod",
]

feature_columns = numeric_features + categorical_features

X = df[feature_columns].copy()

y = df["ChurnValue"].copy()


# =========================================================
# OUT-OF-FOLD RISK PROBABILITIES
# =========================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

print("\nGenerating out-of-fold churn probabilities...")

oof_probabilities = cross_val_predict(
    clone(base_model),
    X,
    y,
    cv=cv,
    method="predict_proba",
    n_jobs=-1,
)[:, 1]


df["ChurnRiskProbability"] = oof_probabilities


oof_auc = roc_auc_score(
    y,
    oof_probabilities,
)

print("\nOOF ROC-AUC:")
print(f"{oof_auc:.4f}")


# =========================================================
# RETENTION THRESHOLD
# =========================================================

RETENTION_THRESHOLD = 0.31
DEFAULT_THRESHOLD = 0.50

df["RetentionPrediction"] = (df["ChurnRiskProbability"] >= RETENTION_THRESHOLD).astype(
    int
)


# =========================================================
# RISK BANDS
# =========================================================


def risk_band(probability):

    if probability >= 0.50:
        return "High Risk"

    if probability >= 0.31:
        return "Retention Candidate"

    return "Below Retention Threshold"


df["RiskBand"] = df["ChurnRiskProbability"].apply(risk_band)


# =========================================================
# RETENTION PRIORITY
# =========================================================


def retention_priority(row):

    probability = row["ChurnRiskProbability"]

    segment = row["Segment"]

    if segment == "High-Risk Short-Tenure Customers" and probability >= 0.50:
        return "Critical"

    if probability >= 0.50 or (
        segment == "High-Risk Short-Tenure Customers" and probability >= 0.31
    ):
        return "High"

    if probability >= 0.31 or segment == "High-Risk Short-Tenure Customers":
        return "Medium"

    return "Low"


df["RetentionPriority"] = df.apply(
    retention_priority,
    axis=1,
)


# =========================================================
# RETENTION ACTION
# =========================================================


def recommended_action(row):

    priority = row["RetentionPriority"]

    if priority == "Critical":
        return (
            "Immediate retention outreach; investigate service "
            "issues; offer contract incentive and support package."
        )

    if priority == "High":
        return (
            "Proactive retention contact; personalized offer; "
            "promote longer contract and automatic payment."
        )

    if priority == "Medium":
        return (
            "Monitor customer experience and provide targeted "
            "onboarding, support or loyalty communication."
        )

    return (
        "Routine service; maintain satisfaction and consider "
        "appropriate cross-sell opportunities."
    )


df["RecommendedAction"] = df.apply(
    recommended_action,
    axis=1,
)


# =========================================================
# RISK BAND PERFORMANCE
# =========================================================

risk_band_summary = (
    df.groupby(
        "RiskBand",
        observed=True,
    )
    .agg(
        Customers=(
            "customerID",
            "count",
        ),
        ActualChurners=(
            "ChurnValue",
            "sum",
        ),
        ActualChurnRate=(
            "ChurnValue",
            "mean",
        ),
        AveragePredictedRisk=(
            "ChurnRiskProbability",
            "mean",
        ),
    )
    .reset_index()
)


risk_band_summary["Percentage"] = risk_band_summary["Customers"] / len(df) * 100


risk_band_summary["ActualChurnRate"] *= 100

risk_band_summary["AveragePredictedRisk"] *= 100


risk_band_summary = risk_band_summary.round(2)


print("\nRisk-band summary:")
print(risk_band_summary)

risk_band_summary.to_csv(
    TABLES_DIR / "final_risk_band_summary.csv",
    index=False,
)


# =========================================================
# RETENTION PRIORITY SUMMARY
# =========================================================

priority_order = [
    "Critical",
    "High",
    "Medium",
    "Low",
]

priority_summary = (
    df.groupby(
        "RetentionPriority",
        observed=True,
    )
    .agg(
        Customers=(
            "customerID",
            "count",
        ),
        ActualChurners=(
            "ChurnValue",
            "sum",
        ),
        ActualChurnRate=(
            "ChurnValue",
            "mean",
        ),
        AveragePredictedRisk=(
            "ChurnRiskProbability",
            "mean",
        ),
    )
    .reset_index()
)


priority_summary["RetentionPriority"] = pd.Categorical(
    priority_summary["RetentionPriority"],
    categories=priority_order,
    ordered=True,
)

priority_summary = priority_summary.sort_values("RetentionPriority")


priority_summary["Percentage"] = priority_summary["Customers"] / len(df) * 100


priority_summary["ActualChurnRate"] *= 100

priority_summary["AveragePredictedRisk"] *= 100


priority_summary = priority_summary.round(2)


print("\nRetention priority summary:")
print(priority_summary)

priority_summary.to_csv(
    TABLES_DIR / "retention_priority_summary.csv",
    index=False,
)


# =========================================================
# SEGMENT × RISK MATRIX
# =========================================================

segment_risk_matrix = pd.crosstab(
    df["Segment"],
    df["RiskBand"],
)

print("\nSegment x risk-band matrix:")
print(segment_risk_matrix)

segment_risk_matrix.to_csv(TABLES_DIR / "segment_risk_matrix.csv")


# =========================================================
# SEGMENT RISK PROFILE
# =========================================================

segment_risk_profile = (
    df.groupby(
        "Segment",
        observed=True,
    )
    .agg(
        Customers=(
            "customerID",
            "count",
        ),
        ActualChurnRate=(
            "ChurnValue",
            "mean",
        ),
        AveragePredictedRisk=(
            "ChurnRiskProbability",
            "mean",
        ),
        RetentionCandidates=(
            "RetentionPrediction",
            "sum",
        ),
    )
    .reset_index()
)


segment_risk_profile["ActualChurnRate"] *= 100

segment_risk_profile["AveragePredictedRisk"] *= 100

segment_risk_profile["RetentionCandidateRate"] = (
    segment_risk_profile["RetentionCandidates"]
    / segment_risk_profile["Customers"]
    * 100
)


segment_risk_profile = segment_risk_profile.round(2)


print("\nSegment risk profile:")
print(segment_risk_profile)

segment_risk_profile.to_csv(
    TABLES_DIR / "final_segment_risk_profile.csv",
    index=False,
)


# =========================================================
# OOF THRESHOLD PERFORMANCE
# =========================================================

oof_predictions = (oof_probabilities >= RETENTION_THRESHOLD).astype(int)


precision = precision_score(
    y,
    oof_predictions,
    zero_division=0,
)

recall = recall_score(
    y,
    oof_predictions,
    zero_division=0,
)

f1 = f1_score(
    y,
    oof_predictions,
    zero_division=0,
)

f2 = fbeta_score(
    y,
    oof_predictions,
    beta=2,
    zero_division=0,
)


print("\nOOF retention-threshold metrics:")
print(f"Precision: {precision:.4f}")

print(f"Recall:    {recall:.4f}")

print(f"F1:        {f1:.4f}")

print(f"F2:        {f2:.4f}")


# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y,
    oof_predictions,
)

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual No Churn",
        "Actual Churn",
    ],
    columns=[
        "Predicted No Churn",
        "Predicted Churn",
    ],
)

print("\nOOF confusion matrix:")
print(cm_df)

cm_df.to_csv(TABLES_DIR / "final_oof_confusion_matrix.csv")


# =========================================================
# TOP-RISK CUSTOMERS
# =========================================================

top_risk_columns = [
    "customerID",
    "Segment",
    "RetentionPriority",
    "RiskBand",
    "ChurnRiskProbability",
    "tenure",
    "Contract",
    "InternetService",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "AutomaticPayment",
    "Churn",
    "RecommendedAction",
]


top_risk_customers = df[top_risk_columns].sort_values(
    "ChurnRiskProbability",
    ascending=False,
)


top_risk_customers.head(100).to_csv(
    TABLES_DIR / "top_100_risk_customers.csv",
    index=False,
)


# =========================================================
# SAVE COMPLETE RISK REGISTER
# =========================================================

risk_register_columns = [
    "customerID",
    "Cluster",
    "Segment",
    "ChurnRiskProbability",
    "RiskBand",
    "RetentionPriority",
    "RetentionPrediction",
    "tenure",
    "Contract",
    "InternetService",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "NumServices",
    "AutomaticPayment",
    "Churn",
    "ChurnValue",
    "RecommendedAction",
]


risk_register = df[risk_register_columns].sort_values(
    [
        "RetentionPriority",
        "ChurnRiskProbability",
    ],
    ascending=[
        True,
        False,
    ],
)


risk_register.to_csv(
    TABLES_DIR / "final_customer_risk_register.csv",
    index=False,
)


# =========================================================
# FIGURE 32 - RISK BAND DISTRIBUTION
# =========================================================

risk_order = [
    "High Risk",
    "Retention Candidate",
    "Below Retention Threshold",
]

plt.figure(figsize=(10, 6))

ax = sns.barplot(
    data=risk_band_summary,
    x="RiskBand",
    y="Customers",
    order=risk_order,
)

plt.title("Customer Distribution by Churn Risk Band")

plt.xlabel("Risk Band")

plt.ylabel("Customers")

plt.xticks(
    rotation=15,
)

for container in ax.containers:
    ax.bar_label(container)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "32_customer_risk_band_distribution.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# =========================================================
# FIGURE 33 - ACTUAL CHURN BY RISK BAND
# =========================================================

plt.figure(figsize=(10, 6))

ax = sns.barplot(
    data=risk_band_summary,
    x="RiskBand",
    y="ActualChurnRate",
    order=risk_order,
)

plt.title("Observed Churn Rate by Predicted Risk Band")

plt.xlabel("Predicted Risk Band")

plt.ylabel("Actual Churn Rate (%)")

plt.xticks(
    rotation=15,
)

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.1f",
    )

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "33_actual_churn_by_risk_band.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# =========================================================
# FIGURE 34 - RETENTION PRIORITY
# =========================================================

plt.figure(figsize=(9, 6))

ax = sns.barplot(
    data=priority_summary,
    x="RetentionPriority",
    y="Customers",
    order=priority_order,
)

plt.title("Customer Retention Priority Distribution")

plt.xlabel("Retention Priority")

plt.ylabel("Customers")

for container in ax.containers:
    ax.bar_label(container)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "34_retention_priority_distribution.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# =========================================================
# FIGURE 35 - SEGMENT x RISK HEATMAP
# =========================================================

segment_risk_percent = (
    pd.crosstab(
        df["Segment"],
        df["RiskBand"],
        normalize="index",
    )
    * 100
)

segment_risk_percent = segment_risk_percent.reindex(columns=risk_order)

plt.figure(figsize=(12, 6))

sns.heatmap(
    segment_risk_percent,
    annot=True,
    fmt=".1f",
    cmap="YlOrRd",
)

plt.title("Risk-Band Distribution Within Customer Segments (%)")

plt.xlabel("Predicted Risk Band")

plt.ylabel("Customer Segment")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "35_segment_risk_heatmap.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# =========================================================
# REFIT FINAL DEPLOYMENT MODEL
# =========================================================

print("\nRefitting final deployment model using all available customers...")

final_model = clone(base_model)

final_model.fit(
    X,
    y,
)

joblib.dump(
    final_model,
    FINAL_MODEL_PATH,
)


# =========================================================
# FINAL SUMMARY
# =========================================================

summary_path = TABLES_DIR / "final_risk_scoring_summary.txt"

with open(
    summary_path,
    "w",
    encoding="utf-8",
) as file:
    file.write("TELCO CUSTOMER CHURN - FINAL RISK SCORING SUMMARY\n")

    file.write("=" * 70 + "\n\n")

    file.write(f"Customers scored: {len(df)}\n")

    file.write(f"OOF ROC-AUC: {oof_auc:.4f}\n")

    file.write(f"Retention threshold: {RETENTION_THRESHOLD:.2f}\n\n")

    file.write("OOF THRESHOLD PERFORMANCE\n")

    file.write(f"Precision: {precision:.4f}\n")

    file.write(f"Recall: {recall:.4f}\n")

    file.write(f"F1: {f1:.4f}\n")

    file.write(f"F2: {f2:.4f}\n\n")

    file.write("RISK BAND SUMMARY\n")

    file.write(risk_band_summary.to_string(index=False))

    file.write("\n\nRETENTION PRIORITY SUMMARY\n")

    file.write(priority_summary.to_string(index=False))

    file.write("\n\nSEGMENT RISK PROFILE\n")

    file.write(segment_risk_profile.to_string(index=False))


# =========================================================
# COMPLETE
# =========================================================

print("\n" + "=" * 80)
print("FINAL RISK SCORING COMPLETED")
print("=" * 80)

print("\nRisk register:")
print(TABLES_DIR / "final_customer_risk_register.csv")

print("\nFinal deployment model:")
print(FINAL_MODEL_PATH)

print("\nTotal figures available:")
print(len(list(FIGURES_DIR.glob("*.png"))))

print("\nSummary:")
print(summary_path)
