from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.base import clone
from sklearn.calibration import (
    CalibratedClassifierCV,
    calibration_curve,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
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
    train_test_split,
)


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "telco_churn_cleaned.csv"

BASE_MODEL_PATH = PROJECT_ROOT / "models" / "best_churn_model.joblib"

RETENTION_MODEL_PATH = PROJECT_ROOT / "models" / "retention_churn_model.joblib"

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
# LOAD DATA AND LOCKED MODEL
# =========================================================

df = pd.read_csv(DATA_PATH)

base_model = joblib.load(BASE_MODEL_PATH)


print("=" * 80)
print("TELCO CHURN - CALIBRATION, THRESHOLD AND ERROR ANALYSIS")
print("=" * 80)


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
# RECREATE LOCKED TRAIN / TEST SPLIT
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

print("\nFinal holdout observations:")
print(len(X_test))


# =========================================================
# OUTER CV
# =========================================================

outer_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


# =========================================================
# RAW OOF PROBABILITIES
# =========================================================

print("\nGenerating RAW training out-of-fold probabilities...")

raw_oof_probabilities = cross_val_predict(
    clone(base_model),
    X_train,
    y_train,
    cv=outer_cv,
    method="predict_proba",
    n_jobs=-1,
)[:, 1]


# =========================================================
# CALIBRATED MODEL TEMPLATE
# =========================================================
#
# Sigmoid calibration is selected as the candidate
# calibration approach because it is relatively stable
# on moderate-size datasets.
#
# Calibration itself is fitted only inside training folds.
# =========================================================

calibrated_template = CalibratedClassifierCV(
    estimator=clone(base_model),
    method="sigmoid",
    cv=5,
)


print("\nGenerating CALIBRATED training out-of-fold probabilities...")

calibrated_oof_probabilities = cross_val_predict(
    calibrated_template,
    X_train,
    y_train,
    cv=outer_cv,
    method="predict_proba",
    n_jobs=-1,
)[:, 1]


# =========================================================
# CALIBRATION COMPARISON
# =========================================================


def probability_metrics(
    actual,
    probabilities,
):

    return {
        "ROC_AUC": roc_auc_score(
            actual,
            probabilities,
        ),
        "Brier_Score": brier_score_loss(
            actual,
            probabilities,
        ),
        "Log_Loss": log_loss(
            actual,
            probabilities,
        ),
        "Mean_Predicted_Risk": probabilities.mean(),
        "Actual_Churn_Rate": actual.mean(),
    }


raw_probability_metrics = probability_metrics(
    y_train,
    raw_oof_probabilities,
)


calibrated_probability_metrics = probability_metrics(
    y_train,
    calibrated_oof_probabilities,
)


calibration_comparison = pd.DataFrame(
    [
        {
            "ProbabilityModel": "Raw selected classifier",
            **raw_probability_metrics,
        },
        {
            "ProbabilityModel": "Sigmoid calibrated classifier",
            **calibrated_probability_metrics,
        },
    ]
)


calibration_comparison["Brier_Improvement_vs_Raw"] = (
    raw_probability_metrics["Brier_Score"] - calibration_comparison["Brier_Score"]
)


calibration_comparison["LogLoss_Improvement_vs_Raw"] = (
    raw_probability_metrics["Log_Loss"] - calibration_comparison["Log_Loss"]
)


print("\n" + "=" * 80)
print("TRAINING-ONLY PROBABILITY CALIBRATION COMPARISON")
print("=" * 80)

print(calibration_comparison.round(5))


calibration_comparison.to_csv(
    TABLES_DIR / "probability_calibration_comparison.csv",
    index=False,
)


# =========================================================
# SELECT PROBABILITY MODEL
# =========================================================
#
# Lower Brier score is better.
#
# This selection uses ONLY training OOF predictions.
# =========================================================

if (
    calibrated_probability_metrics["Brier_Score"]
    < raw_probability_metrics["Brier_Score"]
):
    selected_probability_model_name = "Sigmoid calibrated classifier"

    selected_oof_probabilities = calibrated_oof_probabilities

    selected_probability_model = CalibratedClassifierCV(
        estimator=clone(base_model),
        method="sigmoid",
        cv=5,
    )

    calibration_selected = True

else:
    selected_probability_model_name = "Raw selected classifier"

    selected_oof_probabilities = raw_oof_probabilities

    selected_probability_model = clone(base_model)

    calibration_selected = False


print("\nSelected probability model:")

print(selected_probability_model_name)

print("\nSelection criterion:")

print("Lowest training OOF Brier Score")


# =========================================================
# CALIBRATION CURVE
# =========================================================

raw_fraction_positive, raw_mean_predicted = calibration_curve(
    y_train,
    raw_oof_probabilities,
    n_bins=10,
    strategy="quantile",
)


cal_fraction_positive, cal_mean_predicted = calibration_curve(
    y_train,
    calibrated_oof_probabilities,
    n_bins=10,
    strategy="quantile",
)


plt.figure(figsize=(8, 7))


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration",
)


plt.plot(
    raw_mean_predicted,
    raw_fraction_positive,
    marker="o",
    label="Raw classifier",
)


plt.plot(
    cal_mean_predicted,
    cal_fraction_positive,
    marker="o",
    label="Sigmoid calibrated",
)


plt.title("Training OOF Probability Calibration")

plt.xlabel("Mean Predicted Probability")

plt.ylabel("Observed Churn Fraction")

plt.xlim(
    0,
    1,
)

plt.ylim(
    0,
    1,
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "36_probability_calibration_curve.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# =========================================================
# THRESHOLD SEARCH
# =========================================================
#
# Threshold is selected using ONLY training OOF
# probabilities from the selected probability model.
# =========================================================

thresholds = np.arange(
    0.05,
    0.81,
    0.01,
)


threshold_results = []


for threshold in thresholds:
    predictions = (selected_oof_probabilities >= threshold).astype(int)

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
# SELECT F2 THRESHOLD
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

print("\nTraining OOF metrics at selected threshold:")
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


plt.title("Retention Threshold Optimization Using Training OOF Predictions")

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
# FIT SELECTED PROBABILITY MODEL
# TRAINING DATA ONLY
# =========================================================

print("\nFitting selected probability model on full training data...")


selected_probability_model.fit(
    X_train,
    y_train,
)


joblib.dump(
    selected_probability_model,
    RETENTION_MODEL_PATH,
)


# =========================================================
# FINAL HOLDOUT PROBABILITIES
# =========================================================

test_probabilities = selected_probability_model.predict_proba(X_test)[:, 1]


default_predictions = (test_probabilities >= 0.50).astype(int)


optimized_predictions = (test_probabilities >= optimal_threshold).astype(int)


# =========================================================
# METRIC FUNCTION
# =========================================================


def classification_metrics(
    actual,
    predicted,
    probabilities,
):

    return {
        "ROC_AUC": roc_auc_score(
            actual,
            probabilities,
        ),
        "Brier_Score": brier_score_loss(
            actual,
            probabilities,
        ),
        "Log_Loss": log_loss(
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


default_metrics = classification_metrics(
    y_test,
    default_predictions,
    test_probabilities,
)


optimized_metrics = classification_metrics(
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


print("\n" + "=" * 80)
print("FINAL HOLDOUT: DEFAULT vs OPTIMIZED THRESHOLD")
print("=" * 80)

print(comparison.round(4))


comparison.to_csv(
    TABLES_DIR / "threshold_test_comparison.csv",
    index=False,
)


# =========================================================
# HOLDOUT CALIBRATION SUMMARY
# =========================================================

holdout_calibration_summary = pd.DataFrame(
    [
        {
            "ProbabilityModel": selected_probability_model_name,
            "ROC_AUC": roc_auc_score(
                y_test,
                test_probabilities,
            ),
            "Brier_Score": brier_score_loss(
                y_test,
                test_probabilities,
            ),
            "Log_Loss": log_loss(
                y_test,
                test_probabilities,
            ),
            "Mean_Predicted_Risk": test_probabilities.mean(),
            "Actual_Churn_Rate": y_test.mean(),
        }
    ]
)


holdout_calibration_summary.to_csv(
    TABLES_DIR / "holdout_probability_calibration.csv",
    index=False,
)


print("\nFinal holdout probability quality:")

print(holdout_calibration_summary.round(5))


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


plt.title("Confusion Matrix - Optimized Retention Threshold")

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
# CUSTOMER-LEVEL HOLDOUT ERROR TABLE
# =========================================================

test_results = df.loc[
    test_indices,
    [
        "customerID",
        "gender",
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


def classify_error(
    row,
):

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
# FALSE NEGATIVE / POSITIVE TABLES
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
# SUBGROUP PERFORMANCE
# =========================================================


def subgroup_performance(
    column,
):

    records = []

    for (
        category,
        group,
    ) in test_results.groupby(
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
    "gender",
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
# MACHINE-READABLE SUMMARY
# =========================================================

summary_json = {
    "selected_probability_model": selected_probability_model_name,
    "calibration_selected": calibration_selected,
    "raw_training_oof_roc_auc": float(raw_probability_metrics["ROC_AUC"]),
    "raw_training_oof_brier": float(raw_probability_metrics["Brier_Score"]),
    "calibrated_training_oof_roc_auc": float(calibrated_probability_metrics["ROC_AUC"]),
    "calibrated_training_oof_brier": float(
        calibrated_probability_metrics["Brier_Score"]
    ),
    "selected_threshold": optimal_threshold,
    "holdout_roc_auc": float(optimized_metrics["ROC_AUC"]),
    "holdout_brier": float(optimized_metrics["Brier_Score"]),
    "holdout_log_loss": float(optimized_metrics["Log_Loss"]),
    "holdout_default_recall": float(default_metrics["Recall"]),
    "holdout_optimized_recall": float(optimized_metrics["Recall"]),
    "holdout_optimized_precision": float(optimized_metrics["Precision"]),
    "holdout_optimized_f2": float(optimized_metrics["F2"]),
}


with open(
    TABLES_DIR / "calibration_threshold_summary.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        summary_json,
        file,
        indent=4,
    )


# =========================================================
# TEXT SUMMARY
# =========================================================

summary_path = TABLES_DIR / "threshold_error_summary.txt"


with open(
    summary_path,
    "w",
    encoding="utf-8",
) as file:
    file.write("TELCO CHURN - CALIBRATION, THRESHOLD AND ERROR ANALYSIS\n")

    file.write("=" * 72 + "\n\n")

    file.write("PROBABILITY MODEL SELECTION\n")

    file.write(f"Selected probability model: {selected_probability_model_name}\n")

    file.write(f"Calibration selected: {calibration_selected}\n\n")

    file.write("TRAINING OOF CALIBRATION\n")

    file.write(f"Raw ROC-AUC: {raw_probability_metrics['ROC_AUC']:.4f}\n")

    file.write(f"Raw Brier Score: {raw_probability_metrics['Brier_Score']:.4f}\n")

    file.write(f"Raw Log Loss: {raw_probability_metrics['Log_Loss']:.4f}\n")

    file.write(f"Calibrated ROC-AUC: {calibrated_probability_metrics['ROC_AUC']:.4f}\n")

    file.write(
        f"Calibrated Brier Score: {calibrated_probability_metrics['Brier_Score']:.4f}\n"
    )

    file.write(
        f"Calibrated Log Loss: {calibrated_probability_metrics['Log_Loss']:.4f}\n\n"
    )

    file.write(f"Selected F2 threshold: {optimal_threshold:.2f}\n\n")

    file.write("FINAL HOLDOUT DEFAULT THRESHOLD METRICS\n")

    for (
        key,
        value,
    ) in default_metrics.items():
        file.write(f"{key}: {value:.4f}\n")

    file.write("\nFINAL HOLDOUT OPTIMIZED THRESHOLD METRICS\n")

    for (
        key,
        value,
    ) in optimized_metrics.items():
        file.write(f"{key}: {value:.4f}\n")

    file.write("\nOPTIMIZED CONFUSION MATRIX\n")

    file.write(optimized_cm_df.to_string())

    file.write("\n\nPREDICTION OUTCOME COUNTS\n")

    file.write(error_counts.to_string(index=False))


# =========================================================
# FINAL OUTPUT
# =========================================================

print("\n" + "=" * 80)
print("CALIBRATION + THRESHOLD ANALYSIS COMPLETED")
print("=" * 80)

print("\nSelected probability model:")

print(selected_probability_model_name)

print("\nSelected F2 threshold:")

print(f"{optimal_threshold:.2f}")

print("\nSaved retention model:")

print(RETENTION_MODEL_PATH)

print("\nCalibration figure:")

print(FIGURES_DIR / "36_probability_calibration_curve.png")

print("\nSummary:")

print(summary_path)
