from pathlib import Path
import importlib.util
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
SRC_DIR = PROJECT_ROOT / "src"

PREDICTION_SCRIPT = SRC_DIR / "10_predict_customer.py"


print("=" * 82)
print("TELCO CUSTOMER CHURN CAPSTONE - FINAL PROJECT VALIDATION")
print("=" * 82)


# =========================================================
# VALIDATION STATUS
# =========================================================

validation_checks = {}

warnings = []


def record_check(
    name,
    condition,
    failure_message,
):

    validation_checks[name] = bool(condition)

    if condition:
        print(f"PASS  {name}")

    else:
        print(f"FAIL  {name}")

        warnings.append(failure_message)


# =========================================================
# REQUIRED FILES
# =========================================================

required_files = [
    # Raw / processed data
    DATA_DIR / "Telco-Customer-Churn.csv",
    DATA_DIR / "telco_churn_cleaned.csv",
    DATA_DIR / "telco_churn_segmented.csv",
    DATA_DIR / "telco_churn_segmented_labeled.csv",
    # Core outputs
    TABLES_DIR / "data_validation_report.txt",
    TABLES_DIR / "data_cleaning_summary.txt",
    TABLES_DIR / "eda_summary.txt",
    # Supervised model selection
    TABLES_DIR / "model_comparison.csv",
    TABLES_DIR / "model_summary.json",
    TABLES_DIR / "selected_model_test_metrics.csv",
    # Calibration / thresholding
    TABLES_DIR / "probability_calibration_comparison.csv",
    TABLES_DIR / "holdout_probability_calibration.csv",
    TABLES_DIR / "calibration_threshold_summary.json",
    TABLES_DIR / "threshold_error_summary.txt",
    # Segmentation
    TABLES_DIR / "clustering_summary.txt",
    TABLES_DIR / "cluster_stability_analysis.csv",
    TABLES_DIR / "cluster_training_profile.csv",
    TABLES_DIR / "cluster_holdout_profile.csv",
    # Business segments
    TABLES_DIR / "segment_business_summary.txt",
    TABLES_DIR / "business_segment_holdout_validation.csv",
    TABLES_DIR / "segment_mapping.json",
    # Final independent validation
    TABLES_DIR / "holdout_risk_band_validation.csv",
    TABLES_DIR / "holdout_retention_priority_validation.csv",
    TABLES_DIR / "holdout_customer_priority_results.csv",
    TABLES_DIR / "final_risk_system_summary.json",
    # Final operational outputs
    TABLES_DIR / "final_customer_risk_register.csv",
    TABLES_DIR / "final_risk_scoring_summary.txt",
    # Models
    MODELS_DIR / "best_churn_model.joblib",
    MODELS_DIR / "retention_churn_model.joblib",
    MODELS_DIR / "customer_segmentation_pipeline.joblib",
    MODELS_DIR / "final_churn_model_full.joblib",
    # Inference
    PREDICTION_SCRIPT,
]


print("\n" + "=" * 82)
print("1. REQUIRED FILE VALIDATION")
print("=" * 82)


missing_files = []


for file_path in required_files:
    if file_path.exists():
        print(f"OK       {file_path.relative_to(PROJECT_ROOT)}")

    else:
        print(f"MISSING  {file_path.relative_to(PROJECT_ROOT)}")

        missing_files.append(file_path)


record_check(
    "All required project files exist",
    len(missing_files) == 0,
    ("One or more required project files are missing."),
)


# =========================================================
# FIGURE VALIDATION
# =========================================================

print("\n" + "=" * 82)
print("2. FIGURE VALIDATION")
print("=" * 82)


figure_files = sorted(FIGURES_DIR.glob("*.png"))


print(
    "\nFigure count:",
    len(figure_files),
)


for figure in figure_files:
    print(
        "-",
        figure.name,
    )


record_check(
    "At least 36 analytical figures generated",
    len(figure_files) >= 36,
    ("Expected at least 36 figures after adding probability calibration."),
)


calibration_figure = FIGURES_DIR / "36_probability_calibration_curve.png"


record_check(
    "Probability calibration figure exists",
    calibration_figure.exists(),
    ("Probability calibration curve is missing."),
)


# =========================================================
# CLEAN DATA VALIDATION
# =========================================================

print("\n" + "=" * 82)
print("3. CLEANED DATA VALIDATION")
print("=" * 82)


cleaned_df = pd.read_csv(DATA_DIR / "telco_churn_cleaned.csv")


print(
    "\nShape:",
    cleaned_df.shape,
)

print(
    "Missing values:",
    cleaned_df.isna().sum().sum(),
)

print(
    "Duplicate rows:",
    cleaned_df.duplicated().sum(),
)

print(
    "Duplicate customer IDs:",
    cleaned_df["customerID"].duplicated().sum(),
)

print(
    "Churn rate:",
    f"{cleaned_df['ChurnValue'].mean() * 100:.2f}%",
)


record_check(
    "Cleaned dataset contains 7,043 customers",
    len(cleaned_df) == 7043,
    ("Cleaned dataset does not contain exactly 7,043 customers."),
)


record_check(
    "Cleaned dataset has no missing values",
    cleaned_df.isna().sum().sum() == 0,
    ("Missing values remain in the cleaned dataset."),
)


record_check(
    "Cleaned dataset has no duplicate rows",
    cleaned_df.duplicated().sum() == 0,
    ("Duplicate rows remain in the cleaned dataset."),
)


record_check(
    "Customer IDs are unique",
    (cleaned_df["customerID"].duplicated().sum() == 0),
    ("Duplicate customer IDs detected."),
)


# =========================================================
# MODEL SELECTION VALIDATION
# =========================================================

print("\n" + "=" * 82)
print("4. MODEL SELECTION VALIDATION")
print("=" * 82)


model_results = pd.read_csv(TABLES_DIR / "model_comparison.csv")


with open(
    TABLES_DIR / "model_summary.json",
    "r",
    encoding="utf-8",
) as file:
    model_summary = json.load(file)


print("\nTraining-only comparison:")
print(model_results.round(4))


print("\nSelected model:")
print(model_summary["selected_model"])


print("\nSelection method:")
print(model_summary["selection_method"])


print(
    "\nFinal holdout ROC-AUC:",
    round(
        model_summary["test_roc_auc"],
        4,
    ),
)


record_check(
    "Model selection is training-CV based",
    (model_summary.get("selection_metric") == "GridSearch_CV_ROC_AUC"),
    ("Model summary does not confirm training-only CV model selection."),
)


record_check(
    "Selected model is Logistic Regression",
    (model_summary["selected_model"] == "Logistic Regression"),
    ("Unexpected supervised model selected."),
)


# =========================================================
# CALIBRATION VALIDATION
# =========================================================

print("\n" + "=" * 82)
print("5. PROBABILITY CALIBRATION VALIDATION")
print("=" * 82)


with open(
    TABLES_DIR / "calibration_threshold_summary.json",
    "r",
    encoding="utf-8",
) as file:
    calibration_summary = json.load(file)


calibration_comparison = pd.read_csv(
    TABLES_DIR / "probability_calibration_comparison.csv"
)


print("\nCalibration comparison:")

print(calibration_comparison.round(5))


print("\nSelected probability model:")

print(calibration_summary["selected_probability_model"])


print("\nSelected retention threshold:")

print(
    round(
        calibration_summary["selected_threshold"],
        2,
    )
)


print("\nHoldout Brier score:")

print(
    round(
        calibration_summary["holdout_brier"],
        5,
    )
)


raw_brier = float(calibration_summary["raw_training_oof_brier"])


calibrated_brier = float(calibration_summary["calibrated_training_oof_brier"])


record_check(
    "Calibration improves training OOF Brier Score",
    calibrated_brier < raw_brier,
    ("Calibrated probabilities do not improve the training OOF Brier Score."),
)


record_check(
    "Sigmoid calibration selected",
    calibration_summary["calibration_selected"] is True,
    ("Calibration summary does not indicate that the calibrated model was selected."),
)


record_check(
    "Retention threshold is approximately 0.16",
    abs(float(calibration_summary["selected_threshold"]) - 0.16) < 0.001,
    ("Unexpected retention threshold."),
)


# =========================================================
# SEGMENTATION VALIDATION
# =========================================================

print("\n" + "=" * 82)
print("6. SEGMENTATION VALIDATION")
print("=" * 82)


segmented_df = pd.read_csv(DATA_DIR / "telco_churn_segmented_labeled.csv")


print(
    "\nShape:",
    segmented_df.shape,
)


print("\nCluster counts:")

print(segmented_df["Cluster"].value_counts().sort_index())


print("\nSegment counts:")

print(segmented_df["Segment"].value_counts())


stability_df = pd.read_csv(TABLES_DIR / "cluster_stability_analysis.csv")


mean_ari = stability_df["AdjustedRandIndex"].mean()


print(
    "\nMean cluster stability ARI:",
    round(
        mean_ari,
        4,
    ),
)


record_check(
    "Exactly three clusters are present",
    segmented_df["Cluster"].nunique() == 3,
    ("Expected exactly three customer clusters."),
)


record_check(
    "Cluster stability is high",
    mean_ari >= 0.90,
    ("Cluster stability across random initializations is weak."),
)


# =========================================================
# SEGMENT HOLDOUT VALIDATION
# =========================================================

print("\n" + "=" * 82)
print("7. INDEPENDENT SEGMENT VALIDATION")
print("=" * 82)


segment_validation = pd.read_csv(TABLES_DIR / "business_segment_holdout_validation.csv")


print("\nTraining vs holdout segment performance:")

print(segment_validation)


segment_lookup = segment_validation.set_index("Segment")


required_segments = [
    "Low-Cost Stable Customers",
    "High-Risk Short-Tenure Customers",
    "High-Spend Loyal Customers",
]


record_check(
    "All three business segments exist",
    all(segment in segment_lookup.index for segment in required_segments),
    ("One or more expected business segments are missing."),
)


if all(segment in segment_lookup.index for segment in required_segments):
    low_cost_holdout = float(
        segment_lookup.loc[
            "Low-Cost Stable Customers",
            "HoldoutChurnRate",
        ]
    )

    high_risk_holdout = float(
        segment_lookup.loc[
            "High-Risk Short-Tenure Customers",
            "HoldoutChurnRate",
        ]
    )

    high_spend_holdout = float(
        segment_lookup.loc[
            "High-Spend Loyal Customers",
            "HoldoutChurnRate",
        ]
    )

    record_check(
        "Holdout segment risk ordering is valid",
        (high_risk_holdout > high_spend_holdout > low_cost_holdout),
        ("Holdout segment churn rates do not follow the expected ordering."),
    )


# =========================================================
# FINAL RISK SYSTEM VALIDATION
# =========================================================

print("\n" + "=" * 82)
print("8. FINAL RISK SYSTEM VALIDATION")
print("=" * 82)


with open(
    TABLES_DIR / "final_risk_system_summary.json",
    "r",
    encoding="utf-8",
) as file:
    risk_system_summary = json.load(file)


print(
    "\nIndependent holdout ROC-AUC:",
    round(
        risk_system_summary["independent_holdout_roc_auc"],
        4,
    ),
)


print(
    "Independent holdout Brier:",
    round(
        risk_system_summary["independent_holdout_brier_score"],
        4,
    ),
)


print(
    "Independent holdout recall:",
    round(
        risk_system_summary["independent_holdout_recall"],
        4,
    ),
)


record_check(
    "Independent holdout ROC-AUC >= 0.80",
    (risk_system_summary["independent_holdout_roc_auc"] >= 0.80),
    ("Independent holdout ROC-AUC is below 0.80."),
)


record_check(
    "Independent holdout recall >= 0.90",
    (risk_system_summary["independent_holdout_recall"] >= 0.90),
    ("Independent holdout recall is below 0.90."),
)


# =========================================================
# HOLDOUT RISK BAND VALIDATION
# =========================================================

holdout_risk_df = pd.read_csv(TABLES_DIR / "holdout_risk_band_validation.csv")


print("\nIndependent holdout risk bands:")

print(holdout_risk_df)


risk_lookup = holdout_risk_df.set_index("RiskBand")


expected_risk_bands = [
    "High Risk",
    "Retention Candidate",
    "Below Retention Threshold",
]


if all(band in risk_lookup.index for band in expected_risk_bands):
    high_risk_rate = float(
        risk_lookup.loc[
            "High Risk",
            "ActualChurnRate",
        ]
    )

    candidate_rate = float(
        risk_lookup.loc[
            "Retention Candidate",
            "ActualChurnRate",
        ]
    )

    below_rate = float(
        risk_lookup.loc[
            "Below Retention Threshold",
            "ActualChurnRate",
        ]
    )

    record_check(
        "Risk bands show monotonic holdout churn separation",
        (high_risk_rate > candidate_rate > below_rate),
        ("Holdout risk bands do not show monotonic churn separation."),
    )

else:
    record_check(
        "Risk bands show monotonic holdout churn separation",
        False,
        ("One or more required risk bands are absent."),
    )


# =========================================================
# HOLDOUT PRIORITY VALIDATION
# =========================================================

holdout_priority_df = pd.read_csv(
    TABLES_DIR / "holdout_retention_priority_validation.csv"
)


print("\nIndependent holdout retention priorities:")

print(holdout_priority_df)


priority_lookup = holdout_priority_df.set_index("RetentionPriority")


expected_priorities = [
    "Critical",
    "High",
    "Medium",
    "Low",
]


if all(priority in priority_lookup.index for priority in expected_priorities):
    critical_rate = float(
        priority_lookup.loc[
            "Critical",
            "ActualChurnRate",
        ]
    )

    high_rate = float(
        priority_lookup.loc[
            "High",
            "ActualChurnRate",
        ]
    )

    medium_rate = float(
        priority_lookup.loc[
            "Medium",
            "ActualChurnRate",
        ]
    )

    low_rate = float(
        priority_lookup.loc[
            "Low",
            "ActualChurnRate",
        ]
    )

    record_check(
        "Retention priorities show monotonic holdout churn separation",
        (critical_rate > high_rate > medium_rate > low_rate),
        ("Critical > High > Medium > Low holdout churn ordering failed."),
    )

else:
    record_check(
        "Retention priorities show monotonic holdout churn separation",
        False,
        ("One or more retention-priority groups are missing."),
    )


# =========================================================
# RISK REGISTER VALIDATION
# =========================================================

print("\n" + "=" * 82)
print("9. CUSTOMER RISK REGISTER VALIDATION")
print("=" * 82)


risk_df = pd.read_csv(TABLES_DIR / "final_customer_risk_register.csv")


print(
    "\nShape:",
    risk_df.shape,
)


print(
    "Unique customers:",
    risk_df["customerID"].nunique(),
)


print(
    "Missing churn probabilities:",
    risk_df["ChurnProbability"].isna().sum(),
)


print(
    "Minimum probability:",
    round(
        risk_df["ChurnProbability"].min(),
        4,
    ),
)


print(
    "Maximum probability:",
    round(
        risk_df["ChurnProbability"].max(),
        4,
    ),
)


record_check(
    "Risk register contains all 7,043 customers",
    (len(risk_df) == 7043 and risk_df["customerID"].nunique() == 7043),
    ("Final customer risk register does not contain exactly 7,043 unique customers."),
)


record_check(
    "Risk register has no missing probabilities",
    (risk_df["ChurnProbability"].isna().sum() == 0),
    ("Risk register contains missing probabilities."),
)


record_check(
    "All churn probabilities are between 0 and 1",
    risk_df["ChurnProbability"]
    .between(
        0,
        1,
    )
    .all(),
    ("Invalid churn probability found."),
)


# =========================================================
# MODEL ARTIFACT VALIDATION
# =========================================================

print("\n" + "=" * 82)
print("10. SAVED MODEL VALIDATION")
print("=" * 82)


model_files = {
    "best_churn_model.joblib": MODELS_DIR / "best_churn_model.joblib",
    "retention_churn_model.joblib": MODELS_DIR / "retention_churn_model.joblib",
    "customer_segmentation_pipeline.joblib": MODELS_DIR
    / "customer_segmentation_pipeline.joblib",
    "final_churn_model_full.joblib": MODELS_DIR / "final_churn_model_full.joblib",
}


loaded_models = {}


for (
    name,
    path,
) in model_files.items():
    try:
        loaded_models[name] = joblib.load(path)

        print(f"{name}: OK")

    except Exception as exc:
        print(f"{name}: FAILED")

        print(exc)


record_check(
    "All serialized model artifacts load successfully",
    len(loaded_models) == len(model_files),
    ("One or more serialized model artifacts failed to load."),
)


# =========================================================
# SEGMENTATION MODEL CONTENT VALIDATION
# =========================================================

segmentation_bundle = loaded_models.get("customer_segmentation_pipeline.joblib")


if isinstance(
    segmentation_bundle,
    dict,
):
    required_bundle_keys = {
        "preprocessor",
        "kmeans",
        "cluster_features",
        "selected_k",
        "segment_names",
        "segment_strategy",
    }

    record_check(
        "Segmentation bundle contains inference components",
        required_bundle_keys.issubset(segmentation_bundle.keys()),
        ("Saved segmentation bundle is missing required inference components."),
    )

else:
    record_check(
        "Segmentation bundle contains inference components",
        False,
        ("Segmentation bundle could not be validated."),
    )


# =========================================================
# NEW-CUSTOMER INFERENCE SMOKE TEST
# =========================================================

print("\n" + "=" * 82)
print("11. NEW-CUSTOMER INFERENCE TEST")
print("=" * 82)


inference_passed = False
demo_result = None


try:
    spec = importlib.util.spec_from_file_location(
        "predict_customer_module",
        PREDICTION_SCRIPT,
    )

    prediction_module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(prediction_module)

    demo_result = prediction_module.predict_customer(prediction_module.DEMO_CUSTOMER)

    print("\nDemo inference result:")

    print(
        json.dumps(
            demo_result,
            indent=4,
        )
    )

    required_result_keys = {
        "ChurnProbability",
        "ChurnProbabilityPercent",
        "RetentionThreshold",
        "Cluster",
        "Segment",
        "RiskBand",
        "RetentionPriority",
        "RecommendedAction",
    }

    inference_passed = (
        required_result_keys.issubset(demo_result.keys())
        and 0 <= demo_result["ChurnProbability"] <= 1
        and demo_result["Segment"] in required_segments
        and demo_result["RetentionPriority"] in expected_priorities
    )


except Exception as exc:
    print("\nInference test failed:")

    print(exc)


record_check(
    "End-to-end new-customer inference works",
    inference_passed,
    ("New-customer inference smoke test failed."),
)


# =========================================================
# KEY BUSINESS RESULTS
# =========================================================

print("\n" + "=" * 82)
print("12. KEY VALIDATED RESULTS")
print("=" * 82)


overall_churn = cleaned_df["ChurnValue"].mean() * 100


print(f"\nOverall churn rate: {overall_churn:.2f}%")


print("\nIndependent holdout priority churn rates:")


if all(priority in priority_lookup.index for priority in expected_priorities):
    for priority in expected_priorities:
        print(f"{priority:8}: {priority_lookup.loc[priority, 'ActualChurnRate']:.2f}%")


print("\nIndependent holdout segment churn rates:")


if all(segment in segment_lookup.index for segment in required_segments):
    for segment in required_segments:
        print(f"{segment}: {segment_lookup.loc[segment, 'HoldoutChurnRate']:.2f}%")


# =========================================================
# FINAL PROJECT STATUS
# =========================================================

print("\n" + "=" * 82)
print("13. FINAL PROJECT STATUS")
print("=" * 82)


all_passed = all(validation_checks.values())


passed_count = sum(validation_checks.values())


total_count = len(validation_checks)


print(f"\nValidation checks passed: {passed_count}/{total_count}")


if all_passed:
    final_status = "PASSED"

    print("\nFINAL PROJECT VALIDATION: PASSED")

else:
    final_status = "CHECK WARNINGS"

    print("\nFINAL PROJECT VALIDATION: CHECK WARNINGS BELOW")


if warnings:
    print("\nWarnings:")

    for warning in warnings:
        print(
            "-",
            warning,
        )


print("=" * 82)


# =========================================================
# SAVE VALIDATION SUMMARY
# =========================================================

validation_path = TABLES_DIR / "final_project_validation.txt"


with open(
    validation_path,
    "w",
    encoding="utf-8",
) as file:
    file.write("TELCO CUSTOMER CHURN CAPSTONE - FINAL VALIDATION\n")

    file.write("=" * 76 + "\n\n")

    file.write(f"Final status: {final_status}\n")

    file.write(f"Validation checks passed: {passed_count}/{total_count}\n\n")

    file.write("DATA VALIDATION\n")

    file.write(f"Cleaned dataset shape: {cleaned_df.shape}\n")

    file.write(f"Missing values: {cleaned_df.isna().sum().sum()}\n")

    file.write(f"Duplicate rows: {cleaned_df.duplicated().sum()}\n")

    file.write(
        f"Duplicate customer IDs: {cleaned_df['customerID'].duplicated().sum()}\n"
    )

    file.write(f"Overall churn: {overall_churn:.2f}%\n\n")

    file.write("MODEL VALIDATION\n")

    file.write(f"Selected model: {model_summary['selected_model']}\n")

    file.write(
        f"Training CV ROC-AUC: {model_summary['selected_model_cv_roc_auc']:.4f}\n"
    )

    file.write(
        f"Independent holdout ROC-AUC: "
        f"{risk_system_summary['independent_holdout_roc_auc']:.4f}\n"
    )

    file.write(
        f"Independent holdout Brier Score: "
        f"{risk_system_summary['independent_holdout_brier_score']:.4f}\n"
    )

    file.write(
        f"Independent holdout recall: "
        f"{risk_system_summary['independent_holdout_recall']:.4f}\n"
    )

    file.write(
        f"Validated retention threshold: "
        f"{float(calibration_summary['selected_threshold']):.2f}\n\n"
    )

    file.write("SEGMENTATION VALIDATION\n")

    file.write(f"Clusters: {segmented_df['Cluster'].nunique()}\n")

    file.write(f"Mean stability ARI: {mean_ari:.4f}\n\n")

    file.write("INDEPENDENT HOLDOUT PRIORITY RESULTS\n")

    if all(priority in priority_lookup.index for priority in expected_priorities):
        for priority in expected_priorities:
            file.write(
                f"{priority}: "
                f"{priority_lookup.loc[priority, 'ActualChurnRate']:.2f}% churn\n"
            )

    file.write("\n")

    file.write("PROJECT ARTIFACTS\n")

    file.write(f"Figures generated: {len(figure_files)}\n")

    file.write(f"Risk-register customers: {risk_df['customerID'].nunique()}\n")

    file.write(f"Required files missing: {len(missing_files)}\n")

    file.write(f"New-customer inference: {'PASS' if inference_passed else 'FAIL'}\n\n")

    file.write("INDIVIDUAL CHECKS\n")

    for (
        check,
        result,
    ) in validation_checks.items():
        file.write(f"{'PASS' if result else 'FAIL'} - {check}\n")

    if warnings:
        file.write("\nWARNINGS\n")

        for warning in warnings:
            file.write(f"- {warning}\n")


print("\nValidation report saved to:")

print(validation_path)
