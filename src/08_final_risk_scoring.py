from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.base import clone
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
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

RETENTION_MODEL_PATH = PROJECT_ROOT / "models" / "retention_churn_model.joblib"

FINAL_MODEL_PATH = PROJECT_ROOT / "models" / "final_churn_model_full.joblib"

THRESHOLD_SUMMARY_PATH = (
    PROJECT_ROOT / "outputs" / "tables" / "calibration_threshold_summary.json"
)

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
# LOAD DATA / MODEL / VALIDATED THRESHOLD
# =========================================================

df = pd.read_csv(DATA_PATH)

retention_model = joblib.load(RETENTION_MODEL_PATH)


with open(
    THRESHOLD_SUMMARY_PATH,
    "r",
    encoding="utf-8",
) as file:
    threshold_summary = json.load(file)


RETENTION_THRESHOLD = float(threshold_summary["selected_threshold"])

DEFAULT_THRESHOLD = 0.50


print("=" * 80)
print("TELCO CUSTOMER CHURN - FINAL CALIBRATED RISK SYSTEM")
print("=" * 80)

print("\nDataset shape:")
print(df.shape)

print("\nValidated retention threshold:")
print(f"{RETENTION_THRESHOLD:.2f}")

print("\nProbability model:")
print(threshold_summary["selected_probability_model"])


# =========================================================
# DATA VALIDATION
# =========================================================

required_columns = [
    "customerID",
    "ChurnValue",
    "Segment",
    "Cluster",
    "ModelSplit",
]


missing_required_columns = [
    column for column in required_columns if column not in df.columns
]


if missing_required_columns:
    raise ValueError(
        "Required columns missing: "
        f"{missing_required_columns}. "
        "Run Steps 06 and 07 first."
    )


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
# IDENTIFY ORIGINAL TRAIN / HOLDOUT CUSTOMERS
# =========================================================

training_mask = df["ModelSplit"] == "Training"

holdout_mask = df["ModelSplit"] == "Holdout"


X_train = X.loc[training_mask].copy()

y_train = y.loc[training_mask].copy()


X_holdout = X.loc[holdout_mask].copy()

y_holdout = y.loc[holdout_mask].copy()


print("\nTraining customers:")
print(len(X_train))

print("\nIndependent holdout customers:")
print(len(X_holdout))


# =========================================================
# INDEPENDENT HOLDOUT PROBABILITIES
# =========================================================
#
# retention_churn_model.joblib was fitted using ONLY the
# original training customers in Step 05.
#
# Therefore these probabilities are genuinely independent
# holdout predictions.
# =========================================================

holdout_probabilities = retention_model.predict_proba(X_holdout)[:, 1]


holdout_auc = roc_auc_score(
    y_holdout,
    holdout_probabilities,
)

holdout_brier = brier_score_loss(
    y_holdout,
    holdout_probabilities,
)

holdout_logloss = log_loss(
    y_holdout,
    holdout_probabilities,
)


print("\nIndependent holdout probability performance:")

print(f"ROC-AUC: {holdout_auc:.4f}")

print(f"Brier Score: {holdout_brier:.4f}")

print(f"Log Loss: {holdout_logloss:.4f}")

print(f"Mean predicted risk: {holdout_probabilities.mean() * 100:.2f}%")

print(f"Actual holdout churn: {y_holdout.mean() * 100:.2f}%")


# =========================================================
# RISK BAND FUNCTION
# =========================================================
#
# High Risk:
#     >= 0.50
#
# Retention Candidate:
#     >= validated retention threshold and < 0.50
#
# Below Threshold:
#     < validated retention threshold
# =========================================================


def risk_band(
    probability,
):

    if probability >= DEFAULT_THRESHOLD:
        return "High Risk"

    if probability >= RETENTION_THRESHOLD:
        return "Retention Candidate"

    return "Below Retention Threshold"


# =========================================================
# RETENTION PRIORITY FUNCTION
# =========================================================
#
# This policy uses ONLY:
#
# - frozen training-defined segment
# - calibrated model probability
# - training-selected thresholds
#
# No holdout outcome is used to create the priority.
# =========================================================


def retention_priority(
    probability,
    segment,
):

    high_risk_segment = segment == "High-Risk Short-Tenure Customers"

    if high_risk_segment and probability >= DEFAULT_THRESHOLD:
        return "Critical"

    if probability >= DEFAULT_THRESHOLD:
        return "High"

    if high_risk_segment and probability >= RETENTION_THRESHOLD:
        return "High"

    if probability >= RETENTION_THRESHOLD:
        return "Medium"

    if high_risk_segment:
        return "Medium"

    return "Low"


# =========================================================
# RETENTION ACTION FUNCTION
# =========================================================


def recommended_action(
    priority,
):

    if priority == "Critical":
        return (
            "Immediate retention outreach; investigate "
            "service issues; provide onboarding/support "
            "intervention and a personalized contract or "
            "service incentive."
        )

    if priority == "High":
        return (
            "Proactive retention contact; personalized "
            "offer; promote suitable longer-term contract "
            "and automatic payment."
        )

    if priority == "Medium":
        return (
            "Monitor customer experience and provide "
            "targeted onboarding, support or loyalty "
            "communication."
        )

    return (
        "Routine service; maintain satisfaction and "
        "consider appropriate cross-sell opportunities."
    )


# =========================================================
# INDEPENDENT HOLDOUT POLICY EVALUATION
# =========================================================

holdout_results = df.loc[
    holdout_mask,
    [
        "customerID",
        "Cluster",
        "Segment",
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
    ],
].copy()


holdout_results["ChurnProbability"] = holdout_probabilities


holdout_results["RiskBand"] = holdout_results["ChurnProbability"].apply(risk_band)


holdout_results["RetentionPrediction"] = (
    holdout_results["ChurnProbability"] >= RETENTION_THRESHOLD
).astype(int)


holdout_results["RetentionPriority"] = [
    retention_priority(
        probability,
        segment,
    )
    for (
        probability,
        segment,
    ) in zip(
        holdout_results["ChurnProbability"],
        holdout_results["Segment"],
    )
]


holdout_results["RecommendedAction"] = holdout_results["RetentionPriority"].apply(
    recommended_action
)


# =========================================================
# HOLDOUT THRESHOLD PERFORMANCE
# =========================================================

holdout_predictions = holdout_results["RetentionPrediction"].to_numpy()


holdout_precision = precision_score(
    y_holdout,
    holdout_predictions,
    zero_division=0,
)

holdout_recall = recall_score(
    y_holdout,
    holdout_predictions,
    zero_division=0,
)

holdout_f1 = f1_score(
    y_holdout,
    holdout_predictions,
    zero_division=0,
)

holdout_f2 = fbeta_score(
    y_holdout,
    holdout_predictions,
    beta=2,
    zero_division=0,
)


print("\nIndependent holdout retention-threshold performance:")

print(f"Precision: {holdout_precision:.4f}")

print(f"Recall: {holdout_recall:.4f}")

print(f"F1: {holdout_f1:.4f}")

print(f"F2: {holdout_f2:.4f}")


# =========================================================
# HOLDOUT CONFUSION MATRIX
# =========================================================

holdout_cm = confusion_matrix(
    y_holdout,
    holdout_predictions,
)


holdout_cm_df = pd.DataFrame(
    holdout_cm,
    index=[
        "Actual No Churn",
        "Actual Churn",
    ],
    columns=[
        "Predicted No Churn",
        "Predicted Churn",
    ],
)


print("\nIndependent holdout confusion matrix:")

print(holdout_cm_df)


holdout_cm_df.to_csv(TABLES_DIR / "final_holdout_confusion_matrix.csv")


# =========================================================
# HOLDOUT RISK-BAND VALIDATION
# =========================================================

holdout_risk_band_summary = (
    holdout_results.groupby(
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
            "ChurnProbability",
            "mean",
        ),
    )
    .reset_index()
)


holdout_risk_band_summary["Percentage"] = (
    holdout_risk_band_summary["Customers"] / len(holdout_results) * 100
)


holdout_risk_band_summary["ActualChurnRate"] *= 100


holdout_risk_band_summary["AveragePredictedRisk"] *= 100


holdout_risk_band_summary = holdout_risk_band_summary.round(2)


print("\nINDEPENDENT HOLDOUT RISK-BAND VALIDATION:")

print(holdout_risk_band_summary)


holdout_risk_band_summary.to_csv(
    TABLES_DIR / "holdout_risk_band_validation.csv",
    index=False,
)


# =========================================================
# HOLDOUT RETENTION-PRIORITY VALIDATION
# =========================================================

priority_order = [
    "Critical",
    "High",
    "Medium",
    "Low",
]


holdout_priority_summary = (
    holdout_results.groupby(
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
            "ChurnProbability",
            "mean",
        ),
    )
    .reset_index()
)


holdout_priority_summary["RetentionPriority"] = pd.Categorical(
    holdout_priority_summary["RetentionPriority"],
    categories=priority_order,
    ordered=True,
)


holdout_priority_summary = holdout_priority_summary.sort_values("RetentionPriority")


holdout_priority_summary["Percentage"] = (
    holdout_priority_summary["Customers"] / len(holdout_results) * 100
)


holdout_priority_summary["ActualChurnRate"] *= 100


holdout_priority_summary["AveragePredictedRisk"] *= 100


holdout_priority_summary = holdout_priority_summary.round(2)


print("\nINDEPENDENT HOLDOUT PRIORITY VALIDATION:")

print(holdout_priority_summary)


holdout_priority_summary.to_csv(
    TABLES_DIR / "holdout_retention_priority_validation.csv",
    index=False,
)


# =========================================================
# SAVE HOLDOUT CUSTOMER RESULTS
# =========================================================

holdout_results.to_csv(
    TABLES_DIR / "holdout_customer_priority_results.csv",
    index=False,
)


# =========================================================
# HOLDOUT SEGMENT × RISK VALIDATION
# =========================================================

holdout_segment_risk_matrix = pd.crosstab(
    holdout_results["Segment"],
    holdout_results["RiskBand"],
)


holdout_segment_risk_matrix.to_csv(TABLES_DIR / "holdout_segment_risk_matrix.csv")


print("\nHoldout segment x risk-band matrix:")

print(holdout_segment_risk_matrix)


# =========================================================
# FULL DATASET CROSS-FITTED PROBABILITIES
# =========================================================
#
# These probabilities are used to produce a customer-level
# analytical risk register for ALL 7,043 customers.
#
# They are cross-fitted / out-of-fold predictions.
#
# IMPORTANT:
# They are NOT used as the independent final validation.
# The independent evaluation is the holdout analysis above.
# =========================================================

full_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


print("\nGenerating full-dataset cross-fitted calibrated probabilities...")


cross_fitted_probabilities = cross_val_predict(
    clone(retention_model),
    X,
    y,
    cv=full_cv,
    method="predict_proba",
    n_jobs=-1,
)[:, 1]


df["ChurnProbability"] = cross_fitted_probabilities


full_oof_auc = roc_auc_score(
    y,
    cross_fitted_probabilities,
)

full_oof_brier = brier_score_loss(
    y,
    cross_fitted_probabilities,
)

full_oof_logloss = log_loss(
    y,
    cross_fitted_probabilities,
)


print("\nFull cross-fitted analytical performance:")

print(f"ROC-AUC: {full_oof_auc:.4f}")

print(f"Brier Score: {full_oof_brier:.4f}")

print(f"Log Loss: {full_oof_logloss:.4f}")


# =========================================================
# ASSIGN FULL-DATA RISK BANDS
# =========================================================

df["RiskBand"] = df["ChurnProbability"].apply(risk_band)


df["RetentionPrediction"] = (df["ChurnProbability"] >= RETENTION_THRESHOLD).astype(int)


df["RetentionPriority"] = [
    retention_priority(
        probability,
        segment,
    )
    for (
        probability,
        segment,
    ) in zip(
        df["ChurnProbability"],
        df["Segment"],
    )
]


df["RecommendedAction"] = df["RetentionPriority"].apply(recommended_action)


# =========================================================
# FULL ANALYTICAL RISK-BAND SUMMARY
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
            "ChurnProbability",
            "mean",
        ),
    )
    .reset_index()
)


risk_band_summary["Percentage"] = risk_band_summary["Customers"] / len(df) * 100


risk_band_summary["ActualChurnRate"] *= 100


risk_band_summary["AveragePredictedRisk"] *= 100


risk_band_summary = risk_band_summary.round(2)


print("\nFull cross-fitted risk-band summary:")

print(risk_band_summary)


risk_band_summary.to_csv(
    TABLES_DIR / "final_risk_band_summary.csv",
    index=False,
)


# =========================================================
# FULL ANALYTICAL PRIORITY SUMMARY
# =========================================================

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
            "ChurnProbability",
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


print("\nFull cross-fitted priority summary:")

print(priority_summary)


priority_summary.to_csv(
    TABLES_DIR / "retention_priority_summary.csv",
    index=False,
)


# =========================================================
# FULL SEGMENT RISK PROFILE
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
            "ChurnProbability",
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


print("\nFull segment risk profile:")

print(segment_risk_profile)


segment_risk_profile.to_csv(
    TABLES_DIR / "final_segment_risk_profile.csv",
    index=False,
)


# =========================================================
# FULL SEGMENT × RISK MATRIX
# =========================================================

segment_risk_matrix = pd.crosstab(
    df["Segment"],
    df["RiskBand"],
)


segment_risk_matrix.to_csv(TABLES_DIR / "segment_risk_matrix.csv")


# =========================================================
# FULL CROSS-FITTED CONFUSION MATRIX
# =========================================================

full_oof_predictions = (cross_fitted_probabilities >= RETENTION_THRESHOLD).astype(int)


full_cm = confusion_matrix(
    y,
    full_oof_predictions,
)


full_cm_df = pd.DataFrame(
    full_cm,
    index=[
        "Actual No Churn",
        "Actual Churn",
    ],
    columns=[
        "Predicted No Churn",
        "Predicted Churn",
    ],
)


full_cm_df.to_csv(TABLES_DIR / "final_oof_confusion_matrix.csv")


# =========================================================
# TOP-RISK CUSTOMERS
# =========================================================

top_risk_columns = [
    "customerID",
    "ModelSplit",
    "Segment",
    "RetentionPriority",
    "RiskBand",
    "ChurnProbability",
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
    "ChurnProbability",
    ascending=False,
)


top_risk_customers.head(100).to_csv(
    TABLES_DIR / "top_100_risk_customers.csv",
    index=False,
)


# =========================================================
# COMPLETE RISK REGISTER
# =========================================================

risk_register_columns = [
    "customerID",
    "ModelSplit",
    "Cluster",
    "Segment",
    "ChurnProbability",
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


risk_register = df[risk_register_columns].copy()


priority_rank = {
    "Critical": 0,
    "High": 1,
    "Medium": 2,
    "Low": 3,
}


risk_register["_PriorityRank"] = risk_register["RetentionPriority"].map(priority_rank)


risk_register = risk_register.sort_values(
    [
        "_PriorityRank",
        "ChurnProbability",
    ],
    ascending=[
        True,
        False,
    ],
).drop(columns=["_PriorityRank"])


risk_register.to_csv(
    TABLES_DIR / "final_customer_risk_register.csv",
    index=False,
)


# =========================================================
# FIGURE 32
# FULL RISK-BAND DISTRIBUTION
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


plt.title("Cross-Fitted Customer Distribution by Churn Risk Band")

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
# FIGURE 33
# INDEPENDENT HOLDOUT CHURN BY RISK BAND
# =========================================================

plt.figure(figsize=(10, 6))


ax = sns.barplot(
    data=holdout_risk_band_summary,
    x="RiskBand",
    y="ActualChurnRate",
    order=risk_order,
)


plt.title("Independent Holdout Churn Rate by Predicted Risk Band")

plt.xlabel("Predicted Risk Band")

plt.ylabel("Actual Holdout Churn Rate (%)")

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
# FIGURE 34
# FULL PRIORITY DISTRIBUTION
# =========================================================

plt.figure(figsize=(9, 6))


ax = sns.barplot(
    data=priority_summary,
    x="RetentionPriority",
    y="Customers",
    order=priority_order,
)


plt.title("Cross-Fitted Customer Retention Priority Distribution")

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
# FIGURE 35
# INDEPENDENT HOLDOUT SEGMENT × RISK HEATMAP
# =========================================================

holdout_segment_risk_percent = (
    pd.crosstab(
        holdout_results["Segment"],
        holdout_results["RiskBand"],
        normalize="index",
    )
    * 100
)


holdout_segment_risk_percent = holdout_segment_risk_percent.reindex(
    columns=risk_order,
    fill_value=0,
)


plt.figure(figsize=(12, 6))


sns.heatmap(
    holdout_segment_risk_percent,
    annot=True,
    fmt=".1f",
    cmap="YlOrRd",
)


plt.title("Independent Holdout Risk-Band Distribution Within Customer Segments (%)")

plt.xlabel("Predicted Risk Band")

plt.ylabel("Training-Defined Customer Segment")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "35_segment_risk_heatmap.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# =========================================================
# REFIT FINAL CALIBRATED DEPLOYMENT MODEL
# =========================================================
#
# The architecture, calibration method and threshold are
# already locked.
#
# This model is fitted to all available customers only
# AFTER independent holdout evaluation has been completed.
# =========================================================

print("\nRefitting final calibrated deployment model using all available customers...")


final_model = clone(retention_model)


final_model.fit(
    X,
    y,
)


joblib.dump(
    final_model,
    FINAL_MODEL_PATH,
)


# =========================================================
# MACHINE-READABLE FINAL SUMMARY
# =========================================================

final_summary_json = {
    "model_type": threshold_summary["selected_probability_model"],
    "retention_threshold": RETENTION_THRESHOLD,
    "default_threshold": DEFAULT_THRESHOLD,
    "independent_holdout_customers": int(len(holdout_results)),
    "independent_holdout_roc_auc": float(holdout_auc),
    "independent_holdout_brier_score": float(holdout_brier),
    "independent_holdout_log_loss": float(holdout_logloss),
    "independent_holdout_precision": float(holdout_precision),
    "independent_holdout_recall": float(holdout_recall),
    "independent_holdout_f1": float(holdout_f1),
    "independent_holdout_f2": float(holdout_f2),
    "full_cross_fitted_roc_auc": float(full_oof_auc),
    "full_cross_fitted_brier_score": float(full_oof_brier),
    "full_cross_fitted_log_loss": float(full_oof_logloss),
}


with open(
    TABLES_DIR / "final_risk_system_summary.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        final_summary_json,
        file,
        indent=4,
    )


# =========================================================
# FINAL TEXT SUMMARY
# =========================================================

summary_path = TABLES_DIR / "final_risk_scoring_summary.txt"


with open(
    summary_path,
    "w",
    encoding="utf-8",
) as file:
    file.write("TELCO CUSTOMER CHURN - FINAL CALIBRATED RISK SCORING SUMMARY\n")

    file.write("=" * 76 + "\n\n")

    file.write("METHODOLOGY\n")

    file.write("Classifier family selected using training-only cross-validation.\n")

    file.write(
        "Probability calibration selected using training-only "
        "out-of-fold Brier Score.\n"
    )

    file.write(
        "Retention threshold selected using training-only out-of-fold F2 score.\n"
    )

    file.write("Customer segments fitted and named using training customers only.\n")

    file.write(
        "The final retention-priority policy was evaluated "
        "on the independent holdout set before the deployment "
        "model was refitted on all customers.\n\n"
    )

    file.write("INDEPENDENT HOLDOUT PERFORMANCE\n")

    file.write(f"Customers: {len(holdout_results)}\n")

    file.write(f"ROC-AUC: {holdout_auc:.4f}\n")

    file.write(f"Brier Score: {holdout_brier:.4f}\n")

    file.write(f"Log Loss: {holdout_logloss:.4f}\n")

    file.write(f"Retention threshold: {RETENTION_THRESHOLD:.2f}\n")

    file.write(f"Precision: {holdout_precision:.4f}\n")

    file.write(f"Recall: {holdout_recall:.4f}\n")

    file.write(f"F1: {holdout_f1:.4f}\n")

    file.write(f"F2: {holdout_f2:.4f}\n\n")

    file.write("INDEPENDENT HOLDOUT RISK-BAND VALIDATION\n")

    file.write(holdout_risk_band_summary.to_string(index=False))

    file.write("\n\nINDEPENDENT HOLDOUT PRIORITY VALIDATION\n")

    file.write(holdout_priority_summary.to_string(index=False))

    file.write("\n\nFULL CROSS-FITTED ANALYTICAL RISK REGISTER\n")

    file.write(f"Customers scored: {len(df)}\n")

    file.write(f"Cross-fitted ROC-AUC: {full_oof_auc:.4f}\n")

    file.write(f"Cross-fitted Brier Score: {full_oof_brier:.4f}\n")

    file.write(f"Cross-fitted Log Loss: {full_oof_logloss:.4f}\n\n")

    file.write("FULL RISK-BAND SUMMARY\n")

    file.write(risk_band_summary.to_string(index=False))

    file.write("\n\nFULL PRIORITY SUMMARY\n")

    file.write(priority_summary.to_string(index=False))

    file.write("\n\nFULL SEGMENT RISK PROFILE\n")

    file.write(segment_risk_profile.to_string(index=False))


# =========================================================
# COMPLETE
# =========================================================

print("\n" + "=" * 80)
print("FINAL CALIBRATED RISK SYSTEM COMPLETED")
print("=" * 80)

print("\nIndependent holdout ROC-AUC:")
print(f"{holdout_auc:.4f}")

print("\nIndependent holdout Brier Score:")
print(f"{holdout_brier:.4f}")

print("\nIndependent holdout recall:")
print(f"{holdout_recall:.4f}")

print("\nIndependent holdout risk bands:")
print(holdout_risk_band_summary)

print("\nIndependent holdout priorities:")
print(holdout_priority_summary)

print("\nRisk register:")
print(TABLES_DIR / "final_customer_risk_register.csv")

print("\nFinal calibrated deployment model:")
print(FINAL_MODEL_PATH)

print("\nFinal summary:")
print(summary_path)
