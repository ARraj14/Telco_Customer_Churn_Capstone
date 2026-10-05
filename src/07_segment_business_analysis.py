from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "telco_churn_segmented.csv"

OUTPUT_DATA_PATH = PROJECT_ROOT / "data" / "telco_churn_segmented_labeled.csv"

SEGMENTATION_MODEL_PATH = (
    PROJECT_ROOT / "models" / "customer_segmentation_pipeline.joblib"
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
# LOAD DATA
# =========================================================

df = pd.read_csv(DATA_PATH)

segmentation_bundle = joblib.load(SEGMENTATION_MODEL_PATH)


print("=" * 80)
print("TELCO CHURN - LEAKAGE-SAFE SEGMENT BUSINESS ANALYSIS")
print("=" * 80)


print("\nDataset shape:")
print(df.shape)


# =========================================================
# VALIDATE SPLIT INFORMATION
# =========================================================

if "ModelSplit" not in df.columns:
    raise ValueError(
        "ModelSplit column is missing. Run src/06_customer_segmentation.py first."
    )


training_df = df[df["ModelSplit"] == "Training"].copy()


holdout_df = df[df["ModelSplit"] == "Holdout"].copy()


print("\nTraining customers:")
print(len(training_df))

print("\nHoldout customers:")
print(len(holdout_df))


# =========================================================
# PROFILE FUNCTION
# =========================================================


def create_segment_profile(
    data,
):

    profile = (
        data.groupby(
            "Cluster",
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
            SeniorCitizenRate=(
                "SeniorCitizen",
                "mean",
            ),
        )
        .reset_index()
    )

    profile["ChurnRate"] *= 100

    profile["AutomaticPaymentRate"] *= 100

    profile["SeniorCitizenRate"] *= 100

    return profile.round(2)


# =========================================================
# TRAINING PROFILE
# =========================================================
#
# CRITICAL:
# Business segment names are derived using TRAINING DATA
# ONLY.
#
# Holdout churn is not used to determine segment names.
# =========================================================

training_profile = create_segment_profile(training_df)


print("\nTraining cluster profile:")
print(training_profile)


# =========================================================
# TRAINING-DEFINED SEGMENT NAMING
# =========================================================
#
# Naming logic:
#
# 1. Lowest average monthly charges
#       -> Low-Cost Stable Customers
#
# 2. Highest training churn rate among remaining clusters
#       -> High-Risk Short-Tenure Customers
#
# 3. Remaining / highest-spend behavioral cluster
#       -> High-Spend Loyal Customers
#
# The labels are frozen BEFORE looking at holdout outcomes.
# =========================================================

low_cost_cluster = int(
    training_profile.sort_values("AvgMonthlyCharges").iloc[0]["Cluster"]
)


remaining_after_low_cost = training_profile[
    training_profile["Cluster"] != low_cost_cluster
]


high_risk_cluster = int(
    remaining_after_low_cost.sort_values(
        "ChurnRate",
        ascending=False,
    ).iloc[0]["Cluster"]
)


remaining_clusters = [
    int(cluster)
    for cluster in training_profile["Cluster"].tolist()
    if int(cluster)
    not in [
        low_cost_cluster,
        high_risk_cluster,
    ]
]


if len(remaining_clusters) != 1:
    raise ValueError("Expected exactly three clusters for business-segment naming.")


high_spend_cluster = remaining_clusters[0]


segment_names = {
    low_cost_cluster: "Low-Cost Stable Customers",
    high_risk_cluster: "High-Risk Short-Tenure Customers",
    high_spend_cluster: "High-Spend Loyal Customers",
}


print("\nTraining-defined segment mapping:")

for (
    cluster,
    segment,
) in segment_names.items():
    print(f"Cluster {cluster}: {segment}")


# =========================================================
# SEGMENT STRATEGIES
# =========================================================

segment_strategy = {
    "Low-Cost Stable Customers": (
        "Maintain affordable service and use careful "
        "cross-sell offers that do not increase "
        "price sensitivity."
    ),
    "High-Spend Loyal Customers": (
        "Protect long-tenure, high-spend customers "
        "through loyalty rewards, priority support, "
        "service-quality monitoring and personalized "
        "offers."
    ),
    "High-Risk Short-Tenure Customers": (
        "Prioritize early retention outreach, "
        "onboarding support, contract incentives, "
        "automatic-payment adoption, technical-support "
        "offers and fiber-service quality monitoring."
    ),
}


# =========================================================
# APPLY FROZEN LABELS
# =========================================================

df["Segment"] = df["Cluster"].map(segment_names)


if df["Segment"].isna().any():
    raise ValueError(
        "One or more clusters could not be mapped to business segment names."
    )


df["RetentionStrategy"] = df["Segment"].map(segment_strategy)


training_df = df[df["ModelSplit"] == "Training"].copy()


holdout_df = df[df["ModelSplit"] == "Holdout"].copy()


# =========================================================
# NAMED PROFILE FUNCTION
# =========================================================


def create_named_profile(
    data,
):

    profile = (
        data.groupby(
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

    profile["Percentage"] = profile["Customers"] / len(data) * 100

    profile["ChurnRate"] *= 100

    profile["AutomaticPaymentRate"] *= 100

    return profile.round(2)


# =========================================================
# TRAINING / HOLDOUT / FULL PROFILES
# =========================================================

training_named_profile = create_named_profile(training_df)


holdout_named_profile = create_named_profile(holdout_df)


full_named_profile = create_named_profile(df)


print("\nTraining-defined segment profile:")

print(training_named_profile)


print("\nIndependent holdout segment profile:")

print(holdout_named_profile)


print("\nFull descriptive segment profile:")

print(full_named_profile)


training_named_profile.to_csv(
    TABLES_DIR / "business_segment_training_profile.csv",
    index=False,
)


holdout_named_profile.to_csv(
    TABLES_DIR / "business_segment_holdout_profile.csv",
    index=False,
)


full_named_profile.to_csv(
    TABLES_DIR / "business_segment_profile.csv",
    index=False,
)


# =========================================================
# HOLDOUT VALIDATION
# =========================================================

training_churn_lookup = training_named_profile.set_index("Segment")["ChurnRate"]


holdout_validation = holdout_named_profile[
    [
        "Cluster",
        "Segment",
        "Customers",
        "ChurnRate",
        "AvgTenure",
        "AvgMonthlyCharges",
    ]
].copy()


holdout_validation["TrainingChurnRate"] = holdout_validation["Segment"].map(
    training_churn_lookup
)


holdout_validation["ChurnRateDifference"] = (
    holdout_validation["ChurnRate"] - holdout_validation["TrainingChurnRate"]
)


holdout_validation = holdout_validation.rename(
    columns={"ChurnRate": "HoldoutChurnRate"}
)


holdout_validation["ChurnRateDifference"] = holdout_validation[
    "ChurnRateDifference"
].round(2)


print("\nTraining vs holdout segment validation:")

print(holdout_validation)


holdout_validation.to_csv(
    TABLES_DIR / "business_segment_holdout_validation.csv",
    index=False,
)


# =========================================================
# HOLDOUT RISK COMPARISON
# =========================================================

holdout_overall_churn = holdout_df["ChurnValue"].mean() * 100


holdout_risk_comparison = holdout_named_profile[
    [
        "Cluster",
        "Segment",
        "Customers",
        "ChurnRate",
    ]
].copy()


holdout_risk_comparison["HoldoutOverallChurnRate"] = round(
    holdout_overall_churn,
    2,
)


holdout_risk_comparison["RelativeRiskVsHoldoutOverall"] = (
    holdout_risk_comparison["ChurnRate"] / holdout_overall_churn
).round(2)


lowest_holdout_rate = holdout_risk_comparison["ChurnRate"].min()


holdout_risk_comparison["RelativeRiskVsLowestSegment"] = (
    holdout_risk_comparison["ChurnRate"] / lowest_holdout_rate
).round(2)


print("\nIndependent holdout segment risk comparison:")

print(holdout_risk_comparison)


holdout_risk_comparison.to_csv(
    TABLES_DIR / "segment_risk_comparison.csv",
    index=False,
)


# =========================================================
# CATEGORY DISTRIBUTIONS
# TRAINING-DEFINED DESCRIPTIVE PROFILE
# =========================================================

contract_distribution = (
    pd.crosstab(
        training_df["Segment"],
        training_df["Contract"],
        normalize="index",
    )
    * 100
).round(2)


internet_distribution = (
    pd.crosstab(
        training_df["Segment"],
        training_df["InternetService"],
        normalize="index",
    )
    * 100
).round(2)


payment_distribution = (
    pd.crosstab(
        training_df["Segment"],
        training_df["PaymentMethod"],
        normalize="index",
    )
    * 100
).round(2)


contract_distribution.to_csv(TABLES_DIR / "business_segment_contract_distribution.csv")


internet_distribution.to_csv(TABLES_DIR / "business_segment_internet_distribution.csv")


payment_distribution.to_csv(TABLES_DIR / "business_segment_payment_distribution.csv")


print("\nTraining contract distribution (%):")

print(contract_distribution)


print("\nTraining internet distribution (%):")

print(internet_distribution)


print("\nTraining payment distribution (%):")

print(payment_distribution)


# =========================================================
# SEGMENT RECOMMENDATIONS
# =========================================================

recommendations = pd.DataFrame(
    [
        {
            "Segment": "Low-Cost Stable Customers",
            "PrimaryRisk": (
                "Low churn and low spending, with limited service adoption."
            ),
            "RecommendedAction": (
                "Preserve affordability and introduce "
                "careful, relevant cross-sell offers."
            ),
            "Priority": "Low",
        },
        {
            "Segment": "High-Spend Loyal Customers",
            "PrimaryRisk": (
                "Churn is relatively low, but losing "
                "these long-tenure high-spend customers "
                "would reduce recurring revenue."
            ),
            "RecommendedAction": (
                "Use loyalty rewards, priority support, "
                "service-quality monitoring and "
                "personalized retention offers."
            ),
            "Priority": "Medium",
        },
        {
            "Segment": "High-Risk Short-Tenure Customers",
            "PrimaryRisk": (
                "High churn, short tenure, strong "
                "month-to-month concentration and lower "
                "automatic-payment adoption."
            ),
            "RecommendedAction": (
                "Use early retention outreach, improved "
                "onboarding, contract-upgrade incentives, "
                "automatic-payment offers and technical "
                "support intervention."
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
# FIGURE 29
# FULL CUSTOMER POPULATION BY FROZEN SEGMENT
# =========================================================

segment_order = full_named_profile.sort_values(
    "Customers",
    ascending=False,
)["Segment"].tolist()


plt.figure(figsize=(11, 6))


ax = sns.barplot(
    data=full_named_profile,
    x="Segment",
    y="Customers",
    order=segment_order,
)


plt.title("Customer Population by Training-Defined Business Segment")

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
# FIGURE 30
# INDEPENDENT HOLDOUT CHURN BY SEGMENT
# =========================================================

risk_order = holdout_named_profile.sort_values(
    "ChurnRate",
    ascending=False,
)["Segment"].tolist()


plt.figure(figsize=(11, 6))


ax = sns.barplot(
    data=holdout_named_profile,
    x="Segment",
    y="ChurnRate",
    order=risk_order,
)


plt.axhline(
    holdout_overall_churn,
    linestyle="--",
    label=(f"Holdout overall churn ({holdout_overall_churn:.2f}%)"),
)


plt.title("Independent Holdout Churn Rate by Training-Defined Segment")

plt.xlabel("Customer Segment")

plt.ylabel("Holdout Churn Rate (%)")

plt.xticks(
    rotation=20,
    ha="right",
)


for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.1f",
    )


plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "30_business_segment_churn_rate.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# =========================================================
# FIGURE 31
# TRAINING PROFILE vs HOLDOUT RISK
# =========================================================

training_spend_lookup = training_named_profile.set_index("Segment")["AvgMonthlyCharges"]


validation_map = holdout_named_profile[
    [
        "Segment",
        "Customers",
        "ChurnRate",
    ]
].copy()


validation_map["TrainingAvgMonthlyCharges"] = validation_map["Segment"].map(
    training_spend_lookup
)


plt.figure(figsize=(10, 7))


sns.scatterplot(
    data=validation_map,
    x="TrainingAvgMonthlyCharges",
    y="ChurnRate",
    size="Customers",
    hue="Segment",
    sizes=(
        300,
        1000,
    ),
)


for _, row in validation_map.iterrows():
    plt.annotate(
        row["Segment"],
        (
            row["TrainingAvgMonthlyCharges"],
            row["ChurnRate"],
        ),
        xytext=(
            6,
            6,
        ),
        textcoords="offset points",
    )


plt.title("Training-Defined Spend Profile vs Holdout Churn Risk")

plt.xlabel("Training Average Monthly Charges")

plt.ylabel("Independent Holdout Churn Rate (%)")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "31_segment_value_risk_map.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# =========================================================
# UPDATE SEGMENTATION MODEL BUNDLE
# =========================================================
#
# Adds the frozen business mapping and strategies so
# brand-new customers can later receive a segment name.
# =========================================================

segmentation_bundle["segment_names"] = segment_names


segmentation_bundle["segment_strategy"] = segment_strategy


segmentation_bundle["segment_naming_method"] = (
    "Business segment labels were derived using training "
    "customer profiles only and then frozen before "
    "independent holdout evaluation."
)


joblib.dump(
    segmentation_bundle,
    SEGMENTATION_MODEL_PATH,
)


# =========================================================
# SAVE MAPPING JSON
# =========================================================

mapping_json = {
    str(cluster): segment
    for (
        cluster,
        segment,
    ) in segment_names.items()
}


with open(
    TABLES_DIR / "segment_mapping.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        mapping_json,
        file,
        indent=4,
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

    file.write("=" * 72 + "\n\n")

    file.write("METHODOLOGY\n")

    file.write(
        "Business segment names were determined from training customer profiles only.\n"
    )

    file.write(
        "The resulting mapping was frozen before holdout "
        "churn outcomes were evaluated.\n\n"
    )

    file.write("FROZEN SEGMENT MAPPING\n")

    for (
        cluster,
        segment,
    ) in segment_names.items():
        file.write(f"Cluster {cluster}: {segment}\n")

    file.write("\nTRAINING SEGMENT PROFILE\n")

    file.write(training_named_profile.to_string(index=False))

    file.write("\n\nINDEPENDENT HOLDOUT SEGMENT PROFILE\n")

    file.write(holdout_named_profile.to_string(index=False))

    file.write("\n\nTRAINING vs HOLDOUT VALIDATION\n")

    file.write(holdout_validation.to_string(index=False))

    file.write("\n\nHOLDOUT RISK COMPARISON\n")

    file.write(holdout_risk_comparison.to_string(index=False))

    file.write("\n\nSEGMENT RECOMMENDATIONS\n")

    file.write(recommendations.to_string(index=False))


# =========================================================
# COMPLETE
# =========================================================

print("\n" + "=" * 80)
print("LEAKAGE-SAFE SEGMENT BUSINESS ANALYSIS COMPLETED")
print("=" * 80)


print("\nFrozen segment mapping:")

for (
    cluster,
    segment,
) in segment_names.items():
    print(f"Cluster {cluster}: {segment}")


print("\nIndependent holdout validation:")

print(holdout_validation)


print("\nUpdated segmentation model:")

print(SEGMENTATION_MODEL_PATH)


print("\nLabeled dataset:")

print(OUTPUT_DATA_PATH)


print("\nBusiness summary:")

print(summary_path)
