from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "telco_churn_segmented.csv"

OUTPUT_DATA_PATH = PROJECT_ROOT / "data" / "telco_churn_segmented_labeled.csv"

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
# LOAD DATA
# =========================================================

df = pd.read_csv(DATA_PATH)

print("=" * 78)
print("TELCO CHURN - CUSTOMER SEGMENT BUSINESS ANALYSIS")
print("=" * 78)


# =========================================================
# SEGMENT LABELS
# =========================================================

segment_names = {
    0: "Low-Cost Stable Customers",
    1: "High-Value Loyal Customers",
    2: "High-Risk Short-Tenure Customers",
}

df["Segment"] = df["Cluster"].map(segment_names)


# =========================================================
# SEGMENT STRATEGIES
# =========================================================

segment_strategy = {
    0: (
        "Maintain affordable service and explore gentle "
        "cross-sell opportunities without increasing price sensitivity."
    ),
    1: (
        "Protect these high-value customers through loyalty rewards, "
        "priority support, service-quality monitoring, and personalized offers."
    ),
    2: (
        "Prioritize proactive retention, early-tenure onboarding, "
        "contract-upgrade incentives, automatic-payment adoption, "
        "technical-support offers, and fiber-service quality monitoring."
    ),
}

df["RetentionStrategy"] = df["Cluster"].map(segment_strategy)


# =========================================================
# SEGMENT PROFILE
# =========================================================

segment_profile = (
    df.groupby(
        [
            "Cluster",
            "Segment",
        ],
        observed=True,
    )
    .agg(
        Customers=(
            "customerID",
            "count",
        ),
        AvgTenure=(
            "tenure",
            "mean",
        ),
        MedianTenure=(
            "tenure",
            "median",
        ),
        AvgMonthlyCharges=(
            "MonthlyCharges",
            "mean",
        ),
        AvgTotalCharges=(
            "TotalCharges",
            "mean",
        ),
        AvgServices=(
            "NumServices",
            "mean",
        ),
        ChurnRate=(
            "ChurnValue",
            "mean",
        ),
        AutomaticPaymentRate=(
            "AutomaticPayment",
            "mean",
        ),
    )
    .reset_index()
)


segment_profile["Percentage"] = segment_profile["Customers"] / len(df) * 100


segment_profile["ChurnRate"] *= 100

segment_profile["AutomaticPaymentRate"] *= 100


segment_profile = segment_profile.round(2)


print("\nSegment profile:")
print(segment_profile)

segment_profile.to_csv(
    TABLES_DIR / "business_segment_profile.csv",
    index=False,
)


# =========================================================
# CONTRACT DISTRIBUTION
# =========================================================

contract_distribution = (
    pd.crosstab(
        df["Segment"],
        df["Contract"],
        normalize="index",
    )
    * 100
).round(2)

print("\nContract distribution (%):")
print(contract_distribution)

contract_distribution.to_csv(TABLES_DIR / "business_segment_contract_distribution.csv")


# =========================================================
# INTERNET DISTRIBUTION
# =========================================================

internet_distribution = (
    pd.crosstab(
        df["Segment"],
        df["InternetService"],
        normalize="index",
    )
    * 100
).round(2)

print("\nInternet service distribution (%):")
print(internet_distribution)

internet_distribution.to_csv(TABLES_DIR / "business_segment_internet_distribution.csv")


# =========================================================
# PAYMENT DISTRIBUTION
# =========================================================

payment_distribution = (
    pd.crosstab(
        df["Segment"],
        df["PaymentMethod"],
        normalize="index",
    )
    * 100
).round(2)

print("\nPayment distribution (%):")
print(payment_distribution)

payment_distribution.to_csv(TABLES_DIR / "business_segment_payment_distribution.csv")


# =========================================================
# SEGMENT RECOMMENDATION TABLE
# =========================================================

recommendations = pd.DataFrame(
    [
        {
            "Segment": "Low-Cost Stable Customers",
            "PrimaryRisk": "Low churn but relatively low service adoption",
            "RecommendedAction": (
                "Preserve affordability and introduce careful cross-sell offers."
            ),
            "Priority": "Low",
        },
        {
            "Segment": "High-Value Loyal Customers",
            "PrimaryRisk": ("Loss of a customer would carry high revenue impact"),
            "RecommendedAction": (
                "Loyalty rewards, premium support, "
                "service-quality monitoring and personalized offers."
            ),
            "Priority": "Medium",
        },
        {
            "Segment": "High-Risk Short-Tenure Customers",
            "PrimaryRisk": (
                "High churn, short tenure, month-to-month "
                "contracts and lower automatic-payment adoption"
            ),
            "RecommendedAction": (
                "Early retention outreach, onboarding support, "
                "contract incentives, automatic-payment offers "
                "and technical-support intervention."
            ),
            "Priority": "High",
        },
    ]
)


recommendations.to_csv(
    TABLES_DIR / "segment_retention_recommendations.csv",
    index=False,
)


# =========================================================
# FIGURE 29 - SEGMENT SIZE
# =========================================================

segment_order = segment_profile.sort_values(
    "Customers",
    ascending=False,
)["Segment"].tolist()

plt.figure(figsize=(11, 6))

ax = sns.barplot(
    data=segment_profile,
    x="Segment",
    y="Customers",
    order=segment_order,
)

plt.title("Customer Population by Business Segment")

plt.xlabel("Customer Segment")

plt.ylabel("Number of Customers")

plt.xticks(
    rotation=20,
    ha="right",
)

for container in ax.containers:
    ax.bar_label(container)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "29_business_segment_sizes.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# =========================================================
# FIGURE 30 - CHURN RATE BY NAMED SEGMENT
# =========================================================

risk_order = segment_profile.sort_values(
    "ChurnRate",
    ascending=False,
)["Segment"].tolist()

plt.figure(figsize=(11, 6))

ax = sns.barplot(
    data=segment_profile,
    x="Segment",
    y="ChurnRate",
    order=risk_order,
)

plt.title("Observed Churn Rate by Customer Segment")

plt.xlabel("Customer Segment")

plt.ylabel("Churn Rate (%)")

plt.xticks(
    rotation=20,
    ha="right",
)

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.1f",
    )

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "30_business_segment_churn_rate.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# =========================================================
# FIGURE 31 - CUSTOMER VALUE / RISK MAP
# =========================================================

plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=segment_profile,
    x="AvgMonthlyCharges",
    y="ChurnRate",
    size="Customers",
    hue="Segment",
    sizes=(300, 1000),
)

for _, row in segment_profile.iterrows():
    plt.annotate(
        row["Segment"],
        (
            row["AvgMonthlyCharges"],
            row["ChurnRate"],
        ),
        xytext=(6, 6),
        textcoords="offset points",
    )


plt.title("Customer Segment Value-Risk Map")

plt.xlabel("Average Monthly Charges")

plt.ylabel("Observed Churn Rate (%)")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "31_segment_value_risk_map.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# =========================================================
# SEGMENT RISK MULTIPLIERS
# =========================================================

overall_churn_rate = df["ChurnValue"].mean() * 100

risk_comparison = segment_profile[
    [
        "Cluster",
        "Segment",
        "Customers",
        "ChurnRate",
    ]
].copy()

risk_comparison["OverallChurnRate"] = round(
    overall_churn_rate,
    2,
)

risk_comparison["RelativeRiskVsOverall"] = (
    risk_comparison["ChurnRate"] / overall_churn_rate
).round(2)


lowest_segment_rate = risk_comparison["ChurnRate"].min()

risk_comparison["RelativeRiskVsLowestSegment"] = (
    risk_comparison["ChurnRate"] / lowest_segment_rate
).round(2)


print("\nSegment churn-risk comparison:")
print(risk_comparison)

risk_comparison.to_csv(
    TABLES_DIR / "segment_risk_comparison.csv",
    index=False,
)


# =========================================================
# SAVE LABELED DATASET
# =========================================================

df.to_csv(
    OUTPUT_DATA_PATH,
    index=False,
)


# =========================================================
# BUSINESS SUMMARY
# =========================================================

summary_path = TABLES_DIR / "segment_business_summary.txt"


with open(
    summary_path,
    "w",
    encoding="utf-8",
) as file:
    file.write("TELCO CUSTOMER SEGMENT BUSINESS SUMMARY\n")

    file.write("=" * 65 + "\n\n")

    file.write(f"Overall churn rate: {overall_churn_rate:.2f}%\n\n")

    file.write(segment_profile.to_string(index=False))

    file.write("\n\nSEGMENT RECOMMENDATIONS\n")

    file.write(recommendations.to_string(index=False))

    file.write("\n\nSEGMENT RISK COMPARISON\n")

    file.write(risk_comparison.to_string(index=False))


# =========================================================
# COMPLETE
# =========================================================

print("\n" + "=" * 78)
print("SEGMENT BUSINESS ANALYSIS COMPLETED")
print("=" * 78)

print("\nLabeled dataset:")
print(OUTPUT_DATA_PATH)

print("\nTotal figures available:")
print(len(list(FIGURES_DIR.glob("*.png"))))

print("\nBusiness summary:")
print(summary_path)
