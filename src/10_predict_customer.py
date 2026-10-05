from pathlib import Path
import argparse
import json

import joblib
import pandas as pd


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FINAL_MODEL_PATH = PROJECT_ROOT / "models" / "final_churn_model_full.joblib"

SEGMENTATION_MODEL_PATH = (
    PROJECT_ROOT / "models" / "customer_segmentation_pipeline.joblib"
)

THRESHOLD_SUMMARY_PATH = (
    PROJECT_ROOT / "outputs" / "tables" / "calibration_threshold_summary.json"
)


# =========================================================
# MODEL INPUT FEATURES
# =========================================================

NUMERIC_MODEL_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "NumServices",
    "HasInternetService",
    "ContractValue",
    "AutomaticPayment",
]


CATEGORICAL_MODEL_FEATURES = [
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


MODEL_FEATURES = NUMERIC_MODEL_FEATURES + CATEGORICAL_MODEL_FEATURES


# =========================================================
# REQUIRED RAW INPUT
# =========================================================

REQUIRED_INPUT_FIELDS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


# =========================================================
# SERVICE COLUMNS
# =========================================================

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


# =========================================================
# FEATURE ENGINEERING
# =========================================================


def prepare_customer(
    customer,
):

    missing_fields = [field for field in REQUIRED_INPUT_FIELDS if field not in customer]

    if missing_fields:
        raise ValueError(
            "Missing required customer fields: " + ", ".join(missing_fields)
        )

    row = customer.copy()

    # -----------------------------------------------------
    # Basic numeric validation
    # -----------------------------------------------------

    row["SeniorCitizen"] = int(row["SeniorCitizen"])

    row["tenure"] = int(row["tenure"])

    row["MonthlyCharges"] = float(row["MonthlyCharges"])

    row["TotalCharges"] = float(row["TotalCharges"])

    # -----------------------------------------------------
    # New customer TotalCharges rule
    # -----------------------------------------------------

    if row["tenure"] == 0:
        row["TotalCharges"] = 0.0

    # -----------------------------------------------------
    # NumServices
    # -----------------------------------------------------

    def has_service(
        value,
    ):

        return 1 if str(value).strip().lower() == "yes" else 0

    row["NumServices"] = sum(has_service(row[column]) for column in SERVICE_COLUMNS)

    # -----------------------------------------------------
    # Internet indicator
    # -----------------------------------------------------

    row["HasInternetService"] = int(row["InternetService"] != "No")

    # -----------------------------------------------------
    # Contract encoding
    # -----------------------------------------------------

    contract_mapping = {
        "Month-to-month": 0,
        "One year": 1,
        "Two year": 2,
    }

    if row["Contract"] not in contract_mapping:
        raise ValueError(
            "Invalid Contract. Expected one of: "
            "'Month-to-month', "
            "'One year', "
            "'Two year'."
        )

    row["ContractValue"] = contract_mapping[row["Contract"]]

    # -----------------------------------------------------
    # Automatic payment indicator
    # -----------------------------------------------------

    automatic_payment_methods = {
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    }

    row["AutomaticPayment"] = int(row["PaymentMethod"] in automatic_payment_methods)

    return row


# =========================================================
# RISK BAND
# =========================================================


def risk_band(
    probability,
    retention_threshold,
):

    if probability >= 0.50:
        return "High Risk"

    if probability >= retention_threshold:
        return "Retention Candidate"

    return "Below Retention Threshold"


# =========================================================
# RETENTION PRIORITY
# =========================================================


def retention_priority(
    probability,
    segment,
    retention_threshold,
):

    high_risk_segment = segment == "High-Risk Short-Tenure Customers"

    if high_risk_segment and probability >= 0.50:
        return "Critical"

    if probability >= 0.50:
        return "High"

    if high_risk_segment and probability >= retention_threshold:
        return "High"

    if probability >= retention_threshold:
        return "Medium"

    if high_risk_segment:
        return "Medium"

    return "Low"


# =========================================================
# RECOMMENDED ACTION
# =========================================================


def recommended_action(
    priority,
):

    if priority == "Critical":
        return (
            "Immediate retention outreach; investigate "
            "service issues; provide onboarding/support "
            "intervention and a personalized contract "
            "or service incentive."
        )

    if priority == "High":
        return (
            "Proactive retention contact; personalized "
            "offer; promote a suitable longer-term "
            "contract and automatic payment."
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
# PREDICT ONE CUSTOMER
# =========================================================


def predict_customer(
    customer,
):

    # -----------------------------------------------------
    # Load trained artifacts
    # -----------------------------------------------------

    churn_model = joblib.load(FINAL_MODEL_PATH)

    segmentation_bundle = joblib.load(SEGMENTATION_MODEL_PATH)

    with open(
        THRESHOLD_SUMMARY_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        threshold_summary = json.load(file)

    retention_threshold = float(threshold_summary["selected_threshold"])

    # -----------------------------------------------------
    # Feature engineering
    # -----------------------------------------------------

    prepared = prepare_customer(customer)

    # -----------------------------------------------------
    # Churn model dataframe
    # -----------------------------------------------------

    model_input = pd.DataFrame(
        [{feature: prepared[feature] for feature in MODEL_FEATURES}]
    )

    # -----------------------------------------------------
    # Calibrated churn probability
    # -----------------------------------------------------

    churn_probability = float(churn_model.predict_proba(model_input)[0, 1])

    # -----------------------------------------------------
    # Customer segmentation
    # -----------------------------------------------------

    cluster_features = segmentation_bundle["cluster_features"]

    cluster_input = pd.DataFrame(
        [{feature: prepared[feature] for feature in cluster_features}]
    )

    cluster_processed = segmentation_bundle["preprocessor"].transform(cluster_input)

    cluster = int(segmentation_bundle["kmeans"].predict(cluster_processed)[0])

    segment_names = segmentation_bundle["segment_names"]

    segment = segment_names[cluster]

    # -----------------------------------------------------
    # Risk policy
    # -----------------------------------------------------

    band = risk_band(
        churn_probability,
        retention_threshold,
    )

    priority = retention_priority(
        churn_probability,
        segment,
        retention_threshold,
    )

    action = recommended_action(priority)

    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------

    result = {
        "ChurnProbability": round(
            churn_probability,
            4,
        ),
        "ChurnProbabilityPercent": round(
            churn_probability * 100,
            2,
        ),
        "RetentionThreshold": round(
            retention_threshold,
            2,
        ),
        "Cluster": cluster,
        "Segment": segment,
        "RiskBand": band,
        "RetentionPriority": priority,
        "RecommendedAction": action,
    }

    return result


# =========================================================
# DEMO CUSTOMER
# =========================================================

DEMO_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.50,
    "TotalCharges": 447.50,
}


# =========================================================
# CLI
# =========================================================


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Predict calibrated churn probability, "
            "customer segment and retention priority "
            "for a new Telco customer."
        )
    )

    parser.add_argument(
        "--json",
        type=str,
        help=("Path to a JSON file containing one customer record."),
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help=("Run the included demonstration customer."),
    )

    args = parser.parse_args()

    if args.json:
        json_path = Path(args.json)

        with open(
            json_path,
            "r",
            encoding="utf-8",
        ) as file:
            customer = json.load(file)

    elif args.demo:
        customer = DEMO_CUSTOMER

    else:
        print("No customer supplied. Running built-in demo customer.")

        customer = DEMO_CUSTOMER

    result = predict_customer(customer)

    print("\n" + "=" * 70)

    print("NEW CUSTOMER CHURN ASSESSMENT")

    print("=" * 70)

    print("\nCalibrated churn probability:")

    print(f"{result['ChurnProbabilityPercent']:.2f}%")

    print("\nCustomer cluster:")

    print(result["Cluster"])

    print("\nBusiness segment:")

    print(result["Segment"])

    print("\nRisk band:")

    print(result["RiskBand"])

    print("\nRetention priority:")

    print(result["RetentionPriority"])

    print("\nRecommended action:")

    print(result["RecommendedAction"])

    print("\nMachine-readable result:")

    print(
        json.dumps(
            result,
            indent=4,
        )
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
