from pathlib import Path
import pandas as pd


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "Telco-Customer-Churn.csv"

CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "telco_churn_cleaned.csv"

TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

TABLES_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(RAW_DATA_PATH)

print("=" * 70)
print("TELCO CUSTOMER CHURN - DATA CLEANING")
print("=" * 70)

print("\nOriginal dataset shape:")
print(df.shape)


# =========================================================
# CLEAN STRING COLUMNS
# =========================================================

string_columns = [col for col in df.columns if pd.api.types.is_string_dtype(df[col])]

for column in string_columns:
    df[column] = df[column].str.strip()


# =========================================================
# TOTAL CHARGES INVESTIGATION
# =========================================================

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

invalid_total_charges = df[df["TotalCharges"].isna()].copy()

print("\nRows with invalid TotalCharges:")
print(len(invalid_total_charges))

if not invalid_total_charges.empty:
    print(
        invalid_total_charges[
            ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]
        ]
    )

    invalid_total_charges.to_csv(
        TABLES_DIR / "invalid_totalcharges_rows.csv", index=False
    )


# =========================================================
# HANDLE TOTAL CHARGES
# =========================================================

zero_tenure_missing = df["TotalCharges"].isna() & (df["tenure"] == 0)

print("\nBlank TotalCharges with tenure = 0:")
print(zero_tenure_missing.sum())


# New customers with tenure = 0 have not accumulated
# historical charges yet, therefore TotalCharges = 0.
df.loc[zero_tenure_missing, "TotalCharges"] = 0.0


# Safety check:
remaining_missing_totalcharges = df["TotalCharges"].isna().sum()

print("\nRemaining missing TotalCharges:")
print(remaining_missing_totalcharges)

if remaining_missing_totalcharges > 0:
    raise ValueError("Unexpected missing TotalCharges values remain.")


# =========================================================
# TARGET ENCODING
# =========================================================

df["ChurnValue"] = df["Churn"].map({"No": 0, "Yes": 1})

if df["ChurnValue"].isna().any():
    raise ValueError("Unexpected value found in Churn column.")


# =========================================================
# TENURE GROUP
# =========================================================

df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 36, 48, 60, 72],
    labels=[
        "0-12 months",
        "13-24 months",
        "25-36 months",
        "37-48 months",
        "49-60 months",
        "61-72 months",
    ],
)


# =========================================================
# SERVICE COUNT FEATURE
# =========================================================

service_columns = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def has_service(value):
    return 1 if value == "Yes" else 0


df["NumServices"] = df[service_columns].map(has_service).sum(axis=1)


# =========================================================
# INTERNET SERVICE FLAG
# =========================================================

df["HasInternetService"] = (df["InternetService"] != "No").astype(int)


# =========================================================
# CONTRACT COMMITMENT FEATURE
# =========================================================

contract_mapping = {"Month-to-month": 0, "One year": 1, "Two year": 2}

df["ContractValue"] = df["Contract"].map(contract_mapping)


# =========================================================
# AUTOMATIC PAYMENT FEATURE
# =========================================================

automatic_payment_methods = ["Bank transfer (automatic)", "Credit card (automatic)"]

df["AutomaticPayment"] = df["PaymentMethod"].isin(automatic_payment_methods).astype(int)


# =========================================================
# CHECK DUPLICATES
# =========================================================

duplicate_rows = df.duplicated().sum()

duplicate_customer_ids = df["customerID"].duplicated().sum()

print("\nDuplicate rows:")
print(duplicate_rows)

print("\nDuplicate customer IDs:")
print(duplicate_customer_ids)


# =========================================================
# FINAL MISSING VALUE CHECK
# =========================================================

missing_values = df.isnull().sum()

print("\nMissing values after cleaning:")
print(missing_values[missing_values > 0])


# =========================================================
# FINAL DATA TYPES
# =========================================================

print("\nSelected data types:")

print(
    df[
        [
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "ChurnValue",
            "NumServices",
            "HasInternetService",
            "ContractValue",
            "AutomaticPayment",
        ]
    ].dtypes
)


# =========================================================
# CHURN SUMMARY
# =========================================================

churn_summary = (
    df.groupby("Churn", observed=True)
    .agg(
        Customers=("customerID", "count"),
        AvgTenure=("tenure", "mean"),
        AvgMonthlyCharges=("MonthlyCharges", "mean"),
        AvgTotalCharges=("TotalCharges", "mean"),
        AvgServices=("NumServices", "mean"),
    )
    .round(2)
)

print("\nChurn group summary:")
print(churn_summary)

churn_summary.to_csv(TABLES_DIR / "churn_group_summary.csv")


# =========================================================
# TENURE GROUP CHURN ANALYSIS
# =========================================================

tenure_churn = df.groupby("TenureGroup", observed=True).agg(
    Customers=("customerID", "count"),
    Churned=("ChurnValue", "sum"),
    ChurnRate=("ChurnValue", "mean"),
)

tenure_churn["ChurnRate"] = (tenure_churn["ChurnRate"] * 100).round(2)

print("\nChurn by tenure group:")
print(tenure_churn)

tenure_churn.to_csv(TABLES_DIR / "tenure_group_churn.csv")


# =========================================================
# CLEANING SUMMARY
# =========================================================

summary_path = TABLES_DIR / "data_cleaning_summary.txt"

with open(summary_path, "w", encoding="utf-8") as file:
    file.write("TELCO CUSTOMER CHURN - DATA CLEANING SUMMARY\n")

    file.write("=" * 60 + "\n\n")

    file.write(f"Original rows: {len(df)}\n")

    file.write(f"Original columns: 21\n")

    file.write(f"Invalid TotalCharges detected: {len(invalid_total_charges)}\n")

    file.write(
        f"TotalCharges filled for tenure-zero customers: {zero_tenure_missing.sum()}\n"
    )

    file.write(f"Remaining TotalCharges missing: {df['TotalCharges'].isna().sum()}\n")

    file.write(f"Duplicate rows: {duplicate_rows}\n")

    file.write(f"Duplicate customer IDs: {duplicate_customer_ids}\n")

    file.write(f"Final rows: {df.shape[0]}\n")

    file.write(f"Final columns: {df.shape[1]}\n")

    file.write("\nEngineered features:\n")

    file.write("- ChurnValue\n")

    file.write("- TenureGroup\n")

    file.write("- NumServices\n")

    file.write("- HasInternetService\n")

    file.write("- ContractValue\n")

    file.write("- AutomaticPayment\n")


# =========================================================
# SAVE CLEANED DATASET
# =========================================================

df.to_csv(CLEANED_DATA_PATH, index=False)


# =========================================================
# FINAL VALIDATION
# =========================================================

print("\n" + "=" * 70)
print("CLEANING COMPLETED")
print("=" * 70)

print("\nFinal dataset shape:")
print(df.shape)

print("\nTotal missing values:")
print(df.isnull().sum().sum())

print("\nCleaned dataset saved to:")
print(CLEANED_DATA_PATH)

print("\nCleaning summary saved to:")
print(summary_path)
