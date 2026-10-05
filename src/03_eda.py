from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "telco_churn_cleaned.csv"

FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(DATA_PATH)

print("=" * 75)
print("TELCO CUSTOMER CHURN - EXPLORATORY DATA ANALYSIS")
print("=" * 75)

print("\nDataset shape:")
print(df.shape)

print("\nOverall churn rate:")
print(f"{df['ChurnValue'].mean() * 100:.2f}%")


# =========================================================
# HELPER FUNCTIONS
# =========================================================


def save_figure(filename):
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def churn_rate_table(column):
    table = (
        df.groupby(column, observed=True)
        .agg(
            Customers=("customerID", "count"),
            Churned=("ChurnValue", "sum"),
            ChurnRate=("ChurnValue", "mean"),
        )
        .reset_index()
    )

    table["ChurnRate"] = (table["ChurnRate"] * 100).round(2)

    return table


# =========================================================
# 1. TARGET DISTRIBUTION
# =========================================================

churn_counts = df["Churn"].value_counts().reset_index()

churn_counts.columns = ["Churn", "Customers"]

churn_counts["Percentage"] = (churn_counts["Customers"] / len(df) * 100).round(2)

print("\nChurn distribution:")
print(churn_counts)

churn_counts.to_csv(TABLES_DIR / "eda_churn_distribution.csv", index=False)

plt.figure(figsize=(7, 5))

ax = sns.barplot(data=churn_counts, x="Churn", y="Customers")

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

for container in ax.containers:
    ax.bar_label(container)

save_figure("01_churn_distribution.png")


# =========================================================
# 2. TENURE DISTRIBUTION BY CHURN
# =========================================================

plt.figure(figsize=(9, 6))

sns.histplot(data=df, x="tenure", hue="Churn", bins=24, kde=True, element="step")

plt.title("Customer Tenure Distribution by Churn")
plt.xlabel("Tenure (Months)")
plt.ylabel("Customers")

save_figure("02_tenure_distribution_by_churn.png")


# =========================================================
# 3. MONTHLY CHARGES BY CHURN
# =========================================================

plt.figure(figsize=(8, 6))

sns.boxplot(data=df, x="Churn", y="MonthlyCharges")

plt.title("Monthly Charges by Churn Status")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")

save_figure("03_monthly_charges_by_churn.png")


# =========================================================
# 4. TOTAL CHARGES BY CHURN
# =========================================================

plt.figure(figsize=(8, 6))

sns.boxplot(data=df, x="Churn", y="TotalCharges")

plt.title("Total Charges by Churn Status")
plt.xlabel("Churn")
plt.ylabel("Total Charges")

save_figure("04_total_charges_by_churn.png")


# =========================================================
# 5. TENURE GROUP CHURN RATE
# =========================================================

tenure_table = churn_rate_table("TenureGroup")

tenure_order = [
    "0-12 months",
    "13-24 months",
    "25-36 months",
    "37-48 months",
    "49-60 months",
    "61-72 months",
]

tenure_table["TenureGroup"] = pd.Categorical(
    tenure_table["TenureGroup"], categories=tenure_order, ordered=True
)

tenure_table = tenure_table.sort_values("TenureGroup")

print("\nChurn rate by tenure group:")
print(tenure_table)

tenure_table.to_csv(TABLES_DIR / "eda_tenure_group_churn.csv", index=False)

plt.figure(figsize=(10, 6))

ax = sns.barplot(data=tenure_table, x="TenureGroup", y="ChurnRate")

plt.title("Churn Rate by Customer Tenure Group")
plt.xlabel("Tenure Group")
plt.ylabel("Churn Rate (%)")

plt.xticks(rotation=30, ha="right")

for container in ax.containers:
    ax.bar_label(container, fmt="%.1f")

save_figure("05_tenure_group_churn_rate.png")


# =========================================================
# 6. CONTRACT CHURN
# =========================================================

contract_table = churn_rate_table("Contract")

contract_table = contract_table.sort_values("ChurnRate", ascending=False)

print("\nChurn by contract:")
print(contract_table)

contract_table.to_csv(TABLES_DIR / "eda_contract_churn.csv", index=False)

plt.figure(figsize=(8, 6))

ax = sns.barplot(data=contract_table, x="Contract", y="ChurnRate")

plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract")
plt.ylabel("Churn Rate (%)")

for container in ax.containers:
    ax.bar_label(container, fmt="%.1f")

save_figure("06_contract_churn_rate.png")


# =========================================================
# 7. INTERNET SERVICE CHURN
# =========================================================

internet_table = churn_rate_table("InternetService")

internet_table = internet_table.sort_values("ChurnRate", ascending=False)

print("\nChurn by internet service:")
print(internet_table)

internet_table.to_csv(TABLES_DIR / "eda_internet_service_churn.csv", index=False)

plt.figure(figsize=(8, 6))

ax = sns.barplot(data=internet_table, x="InternetService", y="ChurnRate")

plt.title("Churn Rate by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Churn Rate (%)")

for container in ax.containers:
    ax.bar_label(container, fmt="%.1f")

save_figure("07_internet_service_churn_rate.png")


# =========================================================
# 8. PAYMENT METHOD CHURN
# =========================================================

payment_table = churn_rate_table("PaymentMethod")

payment_table = payment_table.sort_values("ChurnRate", ascending=False)

print("\nChurn by payment method:")
print(payment_table)

payment_table.to_csv(TABLES_DIR / "eda_payment_method_churn.csv", index=False)

plt.figure(figsize=(11, 6))

ax = sns.barplot(data=payment_table, x="PaymentMethod", y="ChurnRate")

plt.title("Churn Rate by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Churn Rate (%)")

plt.xticks(rotation=30, ha="right")

for container in ax.containers:
    ax.bar_label(container, fmt="%.1f")

save_figure("08_payment_method_churn_rate.png")


# =========================================================
# 9. SERVICES USED
# =========================================================

services_table = churn_rate_table("NumServices")

print("\nChurn by number of services:")
print(services_table)

services_table.to_csv(TABLES_DIR / "eda_services_churn.csv", index=False)

plt.figure(figsize=(9, 6))

ax = sns.barplot(data=services_table, x="NumServices", y="ChurnRate")

plt.title("Churn Rate by Number of Services")
plt.xlabel("Number of Active Services")
plt.ylabel("Churn Rate (%)")

for container in ax.containers:
    ax.bar_label(container, fmt="%.1f")

save_figure("09_services_churn_rate.png")


# =========================================================
# 10. SENIOR CITIZEN ANALYSIS
# =========================================================

senior_table = churn_rate_table("SeniorCitizen")

senior_table["CustomerType"] = senior_table["SeniorCitizen"].map(
    {0: "Non-Senior", 1: "Senior Citizen"}
)

print("\nSenior citizen churn:")
print(senior_table)

senior_table.to_csv(TABLES_DIR / "eda_senior_churn.csv", index=False)

plt.figure(figsize=(7, 5))

ax = sns.barplot(data=senior_table, x="CustomerType", y="ChurnRate")

plt.title("Churn Rate: Senior vs Non-Senior Customers")
plt.xlabel("Customer Type")
plt.ylabel("Churn Rate (%)")

for container in ax.containers:
    ax.bar_label(container, fmt="%.1f")

save_figure("10_senior_citizen_churn.png")


# =========================================================
# 11. AUTOMATIC PAYMENT ANALYSIS
# =========================================================

auto_table = churn_rate_table("AutomaticPayment")

auto_table["PaymentType"] = auto_table["AutomaticPayment"].map(
    {0: "Manual Payment", 1: "Automatic Payment"}
)

print("\nAutomatic payment churn:")
print(auto_table)

auto_table.to_csv(TABLES_DIR / "eda_automatic_payment_churn.csv", index=False)

plt.figure(figsize=(7, 5))

ax = sns.barplot(data=auto_table, x="PaymentType", y="ChurnRate")

plt.title("Churn Rate by Payment Automation")
plt.xlabel("Payment Type")
plt.ylabel("Churn Rate (%)")

for container in ax.containers:
    ax.bar_label(container, fmt="%.1f")

save_figure("11_automatic_payment_churn.png")


# =========================================================
# 12. MONTHLY CHARGE QUARTILES
# =========================================================

df["MonthlyChargeBand"] = pd.qcut(
    df["MonthlyCharges"], q=4, labels=["Low", "Medium-Low", "Medium-High", "High"]
)

charge_band_table = churn_rate_table("MonthlyChargeBand")

print("\nChurn by monthly charge band:")
print(charge_band_table)

charge_band_table.to_csv(TABLES_DIR / "eda_monthly_charge_band_churn.csv", index=False)

plt.figure(figsize=(8, 5))

ax = sns.barplot(data=charge_band_table, x="MonthlyChargeBand", y="ChurnRate")

plt.title("Churn Rate by Monthly Charge Band")
plt.xlabel("Monthly Charge Band")
plt.ylabel("Churn Rate (%)")

for container in ax.containers:
    ax.bar_label(container, fmt="%.1f")

save_figure("12_monthly_charge_band_churn.png")


# =========================================================
# 13. CONTRACT x INTERNET SERVICE
# =========================================================

contract_internet = (
    df.pivot_table(
        index="Contract",
        columns="InternetService",
        values="ChurnValue",
        aggfunc="mean",
        observed=True,
    )
    * 100
)

contract_internet = contract_internet.round(2)

print("\nContract x Internet Service churn rates:")
print(contract_internet)

contract_internet.to_csv(TABLES_DIR / "eda_contract_internet_churn_matrix.csv")

plt.figure(figsize=(9, 6))

sns.heatmap(contract_internet, annot=True, fmt=".1f", cmap="YlOrRd")

plt.title("Churn Rate (%) by Contract and Internet Service")

plt.xlabel("Internet Service")
plt.ylabel("Contract")

save_figure("13_contract_internet_heatmap.png")


# =========================================================
# 14. CONTRACT x PAYMENT METHOD
# =========================================================

contract_payment = (
    df.pivot_table(
        index="Contract",
        columns="PaymentMethod",
        values="ChurnValue",
        aggfunc="mean",
        observed=True,
    )
    * 100
)

contract_payment = contract_payment.round(2)

print("\nContract x Payment Method churn rates:")
print(contract_payment)

contract_payment.to_csv(TABLES_DIR / "eda_contract_payment_churn_matrix.csv")

plt.figure(figsize=(12, 6))

sns.heatmap(contract_payment, annot=True, fmt=".1f", cmap="YlOrRd")

plt.title("Churn Rate (%) by Contract and Payment Method")

plt.xlabel("Payment Method")
plt.ylabel("Contract")

plt.xticks(rotation=30, ha="right")

save_figure("14_contract_payment_heatmap.png")


# =========================================================
# 15. NUMERICAL CORRELATION ANALYSIS
# =========================================================

correlation_columns = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "ChurnValue",
    "NumServices",
    "HasInternetService",
    "ContractValue",
    "AutomaticPayment",
]

correlation_matrix = df[correlation_columns].corr().round(3)

print("\nCorrelation matrix:")
print(correlation_matrix)

correlation_matrix.to_csv(TABLES_DIR / "eda_correlation_matrix.csv")

churn_correlations = (
    correlation_matrix["ChurnValue"]
    .drop("ChurnValue")
    .sort_values(key=abs, ascending=False)
)

print("\nCorrelations with ChurnValue:")
print(churn_correlations)

churn_correlations.to_csv(
    TABLES_DIR / "eda_churn_correlations.csv", header=["Correlation"]
)

plt.figure(figsize=(11, 8))

sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)

plt.title("Correlation Heatmap of Numerical and Engineered Features")

save_figure("15_correlation_heatmap.png")


# =========================================================
# 16. CHURNED vs RETAINED NUMERICAL PROFILE
# =========================================================

numeric_profile = (
    df.groupby("Churn", observed=True)
    .agg(
        Customers=("customerID", "count"),
        AvgTenure=("tenure", "mean"),
        MedianTenure=("tenure", "median"),
        AvgMonthlyCharges=("MonthlyCharges", "mean"),
        MedianMonthlyCharges=("MonthlyCharges", "median"),
        AvgTotalCharges=("TotalCharges", "mean"),
        AvgServices=("NumServices", "mean"),
    )
    .round(2)
)

print("\nNumerical customer profile by churn:")
print(numeric_profile)

numeric_profile.to_csv(TABLES_DIR / "eda_churn_numeric_profile.csv")


# =========================================================
# 17. HIGH-RISK CUSTOMER SEGMENTS
# =========================================================

risk_segments = (
    df.groupby(["Contract", "InternetService", "PaymentMethod"], observed=True)
    .agg(
        Customers=("customerID", "count"),
        Churned=("ChurnValue", "sum"),
        ChurnRate=("ChurnValue", "mean"),
    )
    .reset_index()
)

risk_segments["ChurnRate"] = (risk_segments["ChurnRate"] * 100).round(2)

# Avoid interpreting extremely tiny groups
risk_segments_filtered = risk_segments[risk_segments["Customers"] >= 50].sort_values(
    ["ChurnRate", "Customers"], ascending=[False, False]
)

print("\nTop high-risk customer segments:")
print(risk_segments_filtered.head(15))

risk_segments_filtered.to_csv(TABLES_DIR / "eda_high_risk_segments.csv", index=False)


# =========================================================
# 18. CATEGORY SUMMARY
# =========================================================

categorical_features = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "InternetService",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

category_summary_list = []

for feature in categorical_features:
    temp = churn_rate_table(feature)

    temp.insert(0, "Feature", feature)

    temp = temp.rename(columns={feature: "Category"})

    category_summary_list.append(temp)

category_summary = pd.concat(category_summary_list, ignore_index=True)

category_summary = category_summary.sort_values("ChurnRate", ascending=False)

category_summary.to_csv(TABLES_DIR / "eda_category_churn_summary.csv", index=False)


# =========================================================
# FINAL EDA SUMMARY
# =========================================================

summary_path = TABLES_DIR / "eda_summary.txt"

highest_tenure_risk = tenure_table.sort_values("ChurnRate", ascending=False).iloc[0]

highest_contract_risk = contract_table.iloc[0]

highest_internet_risk = internet_table.iloc[0]

highest_payment_risk = payment_table.iloc[0]

with open(summary_path, "w", encoding="utf-8") as file:
    file.write("TELCO CUSTOMER CHURN - EDA SUMMARY\n")

    file.write("=" * 65 + "\n\n")

    file.write(f"Customers analysed: {len(df)}\n")

    file.write(f"Overall churn rate: {df['ChurnValue'].mean() * 100:.2f}%\n\n")

    file.write("Average profile by churn status:\n")

    file.write(numeric_profile.to_string())

    file.write("\n\n")

    file.write("Highest-risk tenure group:\n")

    file.write(
        f"{highest_tenure_risk['TenureGroup']} - "
        f"{highest_tenure_risk['ChurnRate']:.2f}%\n\n"
    )

    file.write("Highest-risk contract:\n")

    file.write(
        f"{highest_contract_risk['Contract']} - "
        f"{highest_contract_risk['ChurnRate']:.2f}%\n\n"
    )

    file.write("Highest-risk internet service:\n")

    file.write(
        f"{highest_internet_risk['InternetService']} - "
        f"{highest_internet_risk['ChurnRate']:.2f}%\n\n"
    )

    file.write("Highest-risk payment method:\n")

    file.write(
        f"{highest_payment_risk['PaymentMethod']} - "
        f"{highest_payment_risk['ChurnRate']:.2f}%\n\n"
    )

    file.write("Correlations with churn:\n")

    file.write(churn_correlations.to_string())

    file.write("\n\n")

    file.write("Top high-risk customer segments:\n")

    file.write(risk_segments_filtered.head(10).to_string(index=False))


# =========================================================
# COMPLETE
# =========================================================

print("\n" + "=" * 75)
print("EDA COMPLETED")
print("=" * 75)

print("\nFigures generated:")
print(len(list(FIGURES_DIR.glob("*.png"))))

print("\nEDA tables saved to:")
print(TABLES_DIR)

print("\nEDA summary saved to:")
print(summary_path)
