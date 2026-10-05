from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
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
    train_test_split,
)


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "telco_churn_cleaned.csv"

MODEL_PATH = PROJECT_ROOT / "models" / "best_churn_model.joblib"

TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

TABLES_DIR.mkdir(parents=True, exist_ok=True)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# LOAD DATA AND MODEL
# =========================================================

df = pd.read_csv(DATA_PATH)

best_model = joblib.load(MODEL_PATH)

print("=" * 78)
print("TELCO CHURN - THRESHOLD OPTIMIZATION AND ERROR ANALYSIS")
print("=" * 78)


# =========================================================
# FEATURES
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
# RECREATE ORIGINAL SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

test_indices = y_test.index


print("\nTraining observations:")
print(len(X_train))

print("\nTesting observations:")
print(len(X_test))


# =========================================================
# OUT-OF-FOLD TRAINING PROBABILITIES
# =========================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

print("\nGenerating out-of-fold training probabilities...")

oof_probabilities = cross_val_predict(
    clone(best_model),
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1,
)[:, 1]


print("\nOOF ROC-AUC:")
print(f"{roc_auc_score(y_train, oof_probabilities):.4f}")


# =========================================================
# THRESHOLD SEARCH
# =========================================================

thresholds = np.arange(
    0.20,
    0.81,
    0.01,
)

threshold_results = []

for threshold in thresholds:
    predictions = (oof_probabilities >= threshold).astype(int)

    threshold_results.append(
        {
            "Threshold": threshold,
            "Accuracy": accuracy_score(
                y_train,
                predictions,
            ),
            "BalancedAccuracy": balanced_accuracy_score(
                y_train,
                predictions,
            ),
            "Precision": precision_score(
                y_train,
                predictions,
                zero_division=0,
            ),
            "Recall": recall_score(
                y_train,
                predictions,
                zero_division=0,
            ),
            "F1": f1_score(
                y_train,
                predictions,
                zero_division=0,
            ),
            "F2": fbeta_score(
                y_train,
                predictions,
                beta=2,
                zero_division=0,
            ),
        }
    )


threshold_df = pd.DataFrame(threshold_results)

threshold_df.to_csv(
    TABLES_DIR / "threshold_analysis.csv",
    index=False,
)


# =========================================================
# BEST F2 THRESHOLD
# =========================================================

best_threshold_row = threshold_df.sort_values(
    [
        "F2",
        "BalancedAccuracy",
    ],
    ascending=False,
).iloc[0]

optimal_threshold = float(best_threshold_row["Threshold"])


print("\nBest training threshold based on F2:")
print(f"{optimal_threshold:.2f}")

print("\nOOF metrics at selected threshold:")
print(best_threshold_row.round(4))


# =========================================================
# THRESHOLD FIGURE
# =========================================================

threshold_plot = threshold_df[
    [
        "Threshold",
        "Precision",
        "Recall",
        "F1",
        "F2",
        "BalancedAccuracy",
    ]
]

threshold_long = threshold_plot.melt(
    id_vars="Threshold",
    var_name="Metric",
    value_name="Score",
)

plt.figure(figsize=(11, 7))

sns.lineplot(
    data=threshold_long,
    x="Threshold",
    y="Score",
    hue="Metric",
)

plt.axvline(
    x=optimal_threshold,
    linestyle="--",
    label=(f"Selected Threshold ({optimal_threshold:.2f})"),
)

plt.title("Classification Threshold Optimization")

plt.xlabel("Churn Probability Threshold")

plt.ylabel("Metric Score")

plt.ylim(
    0,
    1,
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "20_threshold_optimization.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# =========================================================
# TEST PROBABILITIES
# =========================================================

test_probabilities = best_model.predict_proba(X_test)[:, 1]


default_predictions = (test_probabilities >= 0.50).astype(int)

optimized_predictions = (test_probabilities >= optimal_threshold).astype(int)


# =========================================================
# METRIC FUNCTION
# =========================================================


def calculate_metrics(
    actual,
    predicted,
    probabilities,
):

    return {
        "ROC_AUC": roc_auc_score(
            actual,
            probabilities,
        ),
        "Accuracy": accuracy_score(
            actual,
            predicted,
        ),
        "BalancedAccuracy": balanced_accuracy_score(
            actual,
            predicted,
        ),
        "Precision": precision_score(
            actual,
            predicted,
            zero_division=0,
        ),
        "Recall": recall_score(
            actual,
            predicted,
            zero_division=0,
        ),
        "F1": f1_score(
            actual,
            predicted,
            zero_division=0,
        ),
        "F2": fbeta_score(
            actual,
            predicted,
            beta=2,
            zero_division=0,
        ),
    }


default_metrics = calculate_metrics(
    y_test,
    default_predictions,
    test_probabilities,
)

optimized_metrics = calculate_metrics(
    y_test,
    optimized_predictions,
    test_probabilities,
)


comparison = pd.DataFrame(
    [
        {
            "Strategy": "Default threshold 0.50",
            "Threshold": 0.50,
            **default_metrics,
        },
        {
            "Strategy": "Optimized F2 threshold",
            "Threshold": optimal_threshold,
            **optimized_metrics,
        },
    ]
)


print("\n" + "=" * 78)
print("DEFAULT vs OPTIMIZED THRESHOLD")
print("=" * 78)

print(comparison.round(4))

comparison.to_csv(
    TABLES_DIR / "threshold_test_comparison.csv",
    index=False,
)


# =========================================================
# OPTIMIZED CONFUSION MATRIX
# =========================================================

optimized_cm = confusion_matrix(
    y_test,
    optimized_predictions,
)

optimized_cm_df = pd.DataFrame(
    optimized_cm,
    index=[
        "Actual No Churn",
        "Actual Churn",
    ],
    columns=[
        "Predicted No Churn",
        "Predicted Churn",
    ],
)

print("\nOptimized threshold confusion matrix:")

print(optimized_cm_df)

optimized_cm_df.to_csv(TABLES_DIR / "optimized_threshold_confusion_matrix.csv")


plt.figure(figsize=(7, 6))

sns.heatmap(
    optimized_cm_df,
    annot=True,
    fmt="d",
    cmap="Blues",
)

plt.title("Confusion Matrix - Optimized Threshold")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "21_optimized_threshold_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# =========================================================
# CUSTOMER-LEVEL ERROR TABLE
# =========================================================

test_results = df.loc[
    test_indices,
    [
        "customerID",
        "Churn",
        "ChurnValue",
        "tenure",
        "TenureGroup",
        "Contract",
        "InternetService",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "SeniorCitizen",
        "AutomaticPayment",
    ],
].copy()


test_results["ChurnProbability"] = test_probabilities

test_results["Prediction"] = optimized_predictions


def classify_error(row):

    actual = row["ChurnValue"]
    predicted = row["Prediction"]

    if actual == 1 and predicted == 1:
        return "True Positive"

    if actual == 0 and predicted == 0:
        return "True Negative"

    if actual == 0 and predicted == 1:
        return "False Positive"

    return "False Negative"


test_results["PredictionType"] = test_results.apply(
    classify_error,
    axis=1,
)


test_results.to_csv(
    TABLES_DIR / "optimized_test_predictions_detailed.csv",
    index=False,
)


# =========================================================
# ERROR COUNTS
# =========================================================

error_counts = (
    test_results["PredictionType"]
    .value_counts()
    .rename_axis("PredictionType")
    .reset_index(name="Customers")
)

print("\nPrediction outcome counts:")
print(error_counts)

error_counts.to_csv(
    TABLES_DIR / "prediction_outcome_counts.csv",
    index=False,
)


plt.figure(figsize=(9, 6))

ax = sns.barplot(
    data=error_counts,
    x="PredictionType",
    y="Customers",
)

plt.title("Prediction Outcomes at Optimized Threshold")

plt.xlabel("Prediction Outcome")

plt.ylabel("Customers")

plt.xticks(
    rotation=20,
)

for container in ax.containers:
    ax.bar_label(container)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "22_prediction_outcome_counts.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# =========================================================
# FALSE NEGATIVE CUSTOMERS
# =========================================================

false_negatives = test_results[
    test_results["PredictionType"] == "False Negative"
].sort_values(
    "ChurnProbability",
    ascending=False,
)

false_positives = test_results[
    test_results["PredictionType"] == "False Positive"
].sort_values(
    "ChurnProbability",
    ascending=False,
)


false_negatives.to_csv(
    TABLES_DIR / "false_negative_customers.csv",
    index=False,
)

false_positives.to_csv(
    TABLES_DIR / "false_positive_customers.csv",
    index=False,
)


print("\nFalse negatives:")
print(len(false_negatives))

print("\nFalse positives:")
print(len(false_positives))


# =========================================================
# SUBGROUP PERFORMANCE FUNCTION
# =========================================================


def subgroup_performance(column):

    records = []

    for category, group in test_results.groupby(
        column,
        observed=True,
    ):
        actual = group["ChurnValue"]

        predicted = group["Prediction"]

        actual_churners = int(actual.sum())

        true_positives = int(((actual == 1) & (predicted == 1)).sum())

        false_negatives_count = int(((actual == 1) & (predicted == 0)).sum())

        if actual_churners > 0:
            subgroup_recall = true_positives / actual_churners

        else:
            subgroup_recall = np.nan

        records.append(
            {
                column: category,
                "Customers": len(group),
                "ActualChurners": actual_churners,
                "TruePositives": true_positives,
                "FalseNegatives": false_negatives_count,
                "Recall": subgroup_recall,
                "AveragePredictedRisk": group["ChurnProbability"].mean(),
            }
        )

    return pd.DataFrame(records)


# =========================================================
# SUBGROUP ERROR ANALYSIS
# =========================================================

subgroup_columns = [
    "TenureGroup",
    "Contract",
    "InternetService",
    "PaymentMethod",
    "SeniorCitizen",
]


for column in subgroup_columns:
    subgroup_table = subgroup_performance(column)

    subgroup_table["Recall"] = subgroup_table["Recall"].round(4)

    subgroup_table["AveragePredictedRisk"] = subgroup_table[
        "AveragePredictedRisk"
    ].round(4)

    subgroup_table.to_csv(
        TABLES_DIR / f"error_analysis_{column}.csv",
        index=False,
    )

    print(f"\nPerformance by {column}:")

    print(subgroup_table)


# =========================================================
# CONTRACT RECALL FIGURE
# =========================================================

contract_performance = subgroup_performance("Contract")

contract_performance["RecallPercent"] = contract_performance["Recall"] * 100


plt.figure(figsize=(8, 6))

ax = sns.barplot(
    data=contract_performance,
    x="Contract",
    y="RecallPercent",
)

plt.title("Churn Detection Recall by Contract Type")

plt.xlabel("Contract")

plt.ylabel("Recall (%)")

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.1f",
    )

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "23_recall_by_contract.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# =========================================================
# FINAL SUMMARY
# =========================================================

summary_path = TABLES_DIR / "threshold_error_summary.txt"

with open(
    summary_path,
    "w",
    encoding="utf-8",
) as file:
    file.write("TELCO CHURN - THRESHOLD AND ERROR ANALYSIS\n")

    file.write("=" * 65 + "\n\n")

    file.write(f"Selected F2 threshold: {optimal_threshold:.2f}\n\n")

    file.write("Default threshold metrics:\n")

    for key, value in default_metrics.items():
        file.write(f"{key}: {value:.4f}\n")

    file.write("\nOptimized threshold metrics:\n")

    for key, value in optimized_metrics.items():
        file.write(f"{key}: {value:.4f}\n")

    file.write("\nOptimized confusion matrix:\n")

    file.write(optimized_cm_df.to_string())

    file.write("\n\nPrediction outcome counts:\n")

    file.write(error_counts.to_string(index=False))


print("\n" + "=" * 78)
print("THRESHOLD AND ERROR ANALYSIS COMPLETED")
print("=" * 78)

print("\nSelected threshold:")

print(f"{optimal_threshold:.2f}")

print("\nTotal figures available:")

print(len(list(FIGURES_DIR.glob("*.png"))))

print("\nSummary saved to:")

print(summary_path)
