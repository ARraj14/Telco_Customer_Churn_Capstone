from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "Telco-Customer-Churn.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("TELCO CUSTOMER CHURN - DATA VALIDATION")
print("=" * 70)


# ---------------------------------------------------------
# Basic information
# ---------------------------------------------------------

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
for i, column in enumerate(df.columns, start=1):
    print(f"{i:02d}. {column}")


# ---------------------------------------------------------
# First rows
# ---------------------------------------------------------

print("\nFirst five rows:")
print(df.head())


# ---------------------------------------------------------
# Data types
# ---------------------------------------------------------

print("\nData types:")
print(df.dtypes)


# ---------------------------------------------------------
# Missing values
# ---------------------------------------------------------

print("\nStandard missing values:")
print(df.isnull().sum())


# ---------------------------------------------------------
# Check blank TotalCharges values
# ---------------------------------------------------------

if "TotalCharges" in df.columns:
    total_charges_numeric = pd.to_numeric(df["TotalCharges"], errors="coerce")

    invalid_total_charges = total_charges_numeric.isna().sum()

    print("\nInvalid / blank TotalCharges values:")
    print(invalid_total_charges)


# ---------------------------------------------------------
# Duplicate rows
# ---------------------------------------------------------

duplicate_rows = df.duplicated().sum()

print("\nDuplicate rows:")
print(duplicate_rows)


# ---------------------------------------------------------
# Duplicate customer IDs
# ---------------------------------------------------------

if "customerID" in df.columns:
    duplicate_customer_ids = df["customerID"].duplicated().sum()

    print("\nDuplicate customer IDs:")
    print(duplicate_customer_ids)


# ---------------------------------------------------------
# Churn distribution
# ---------------------------------------------------------

if "Churn" in df.columns:
    print("\nChurn counts:")
    print(df["Churn"].value_counts())

    print("\nChurn percentages:")
    print(df["Churn"].value_counts(normalize=True).mul(100).round(2))


# ---------------------------------------------------------
# Numerical summary
# ---------------------------------------------------------

print("\nNumerical summary:")
print(df.describe())


# ---------------------------------------------------------
# Unique values
# ---------------------------------------------------------

categorical_columns = [
    col for col in df.columns if pd.api.types.is_string_dtype(df[col])
]

print("\nCategorical unique-value counts:")

for column in categorical_columns:
    print(f"{column}: {df[column].nunique()} unique values")


# ---------------------------------------------------------
# Create column summary table
# ---------------------------------------------------------

summary = pd.DataFrame(
    {
        "Column": df.columns,
        "Data_Type": df.dtypes.astype(str).values,
        "Missing_Values": df.isnull().sum().values,
        "Unique_Values": df.nunique(dropna=False).values,
    }
)

summary.to_csv(OUTPUT_DIR / "data_validation_summary.csv", index=False)


# ---------------------------------------------------------
# Save validation report
# ---------------------------------------------------------

report_path = OUTPUT_DIR / "data_validation_report.txt"

with open(report_path, "w", encoding="utf-8") as file:
    file.write("TELCO CUSTOMER CHURN - DATA VALIDATION\n")
    file.write("=" * 60 + "\n\n")

    file.write(f"Dataset shape: {df.shape}\n")
    file.write(f"Rows: {df.shape[0]}\n")
    file.write(f"Columns: {df.shape[1]}\n")
    file.write(f"Duplicate rows: {duplicate_rows}\n")

    if "customerID" in df.columns:
        file.write(f"Duplicate customer IDs: {df['customerID'].duplicated().sum()}\n")

    if "TotalCharges" in df.columns:
        file.write(f"Invalid/blank TotalCharges values: {invalid_total_charges}\n")

    if "Churn" in df.columns:
        file.write("\nChurn distribution:\n")

        churn_counts = df["Churn"].value_counts()

        churn_percentages = df["Churn"].value_counts(normalize=True).mul(100).round(2)

        for category in churn_counts.index:
            file.write(
                f"{category}: "
                f"{churn_counts[category]} "
                f"({churn_percentages[category]}%)\n"
            )


print("\n" + "=" * 70)
print("VALIDATION COMPLETED")
print("=" * 70)

print("\nSaved:")
print(OUTPUT_DIR / "data_validation_summary.csv")
print(report_path)
