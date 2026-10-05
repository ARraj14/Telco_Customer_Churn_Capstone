from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "telco_churn_cleaned.csv"

SEGMENTED_DATA_PATH = PROJECT_ROOT / "data" / "telco_churn_segmented.csv"

FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TABLES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(DATA_PATH)

print("=" * 78)
print("TELCO CUSTOMER CHURN - CUSTOMER SEGMENTATION")
print("=" * 78)

print("\nDataset shape:")
print(df.shape)


# =========================================================
# CLUSTERING FEATURES
# =========================================================

numeric_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "NumServices",
]

categorical_features = [
    "Contract",
    "InternetService",
    "PaymentMethod",
    "OnlineSecurity",
    "TechSupport",
    "PaperlessBilling",
]

cluster_features = numeric_features + categorical_features

X = df[cluster_features].copy()

print("\nClustering features:")
for feature in cluster_features:
    print(f"- {feature}")


# =========================================================
# PREPROCESSING
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features,
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
            categorical_features,
        ),
    ],
    remainder="drop",
)

X_processed = preprocessor.fit_transform(X)

print("\nProcessed clustering shape:")
print(X_processed.shape)


# =========================================================
# EVALUATE K VALUES
# =========================================================

cluster_results = []

for k in range(2, 9):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20,
    )

    labels = model.fit_predict(X_processed)

    silhouette = silhouette_score(
        X_processed,
        labels,
    )

    davies_bouldin = davies_bouldin_score(
        X_processed,
        labels,
    )

    calinski_harabasz = calinski_harabasz_score(
        X_processed,
        labels,
    )

    cluster_results.append(
        {
            "K": k,
            "Inertia": model.inertia_,
            "Silhouette": silhouette,
            "DaviesBouldin": davies_bouldin,
            "CalinskiHarabasz": calinski_harabasz,
        }
    )


metrics_df = pd.DataFrame(cluster_results)

print("\nClustering evaluation:")
print(metrics_df.round(4))

metrics_df.to_csv(
    TABLES_DIR / "clustering_k_evaluation.csv",
    index=False,
)


# =========================================================
# SELECT K
# =========================================================

best_row = metrics_df.sort_values(
    "Silhouette",
    ascending=False,
).iloc[0]

best_k = int(best_row["K"])

print("\nBest K by silhouette score:")
print(best_k)

print("\nBest silhouette score:")
print(f"{best_row['Silhouette']:.4f}")


# =========================================================
# FIGURE 24 - ELBOW METHOD
# =========================================================

plt.figure(figsize=(8, 6))

sns.lineplot(
    data=metrics_df,
    x="K",
    y="Inertia",
    marker="o",
)

plt.title("K-Means Elbow Method")

plt.xlabel("Number of Clusters (K)")

plt.ylabel("Within-Cluster Inertia")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "24_kmeans_elbow_method.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# =========================================================
# FIGURE 25 - CLUSTER QUALITY
# =========================================================

plt.figure(figsize=(8, 6))

sns.lineplot(
    data=metrics_df,
    x="K",
    y="Silhouette",
    marker="o",
)

plt.axvline(
    best_k,
    linestyle="--",
)

plt.title("Silhouette Score by Number of Clusters")

plt.xlabel("Number of Clusters (K)")

plt.ylabel("Silhouette Score")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "25_kmeans_silhouette_scores.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# =========================================================
# FINAL K-MEANS MODEL
# =========================================================

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=20,
)

cluster_labels = kmeans.fit_predict(X_processed)

df["Cluster"] = cluster_labels


# =========================================================
# PCA
# =========================================================

pca = PCA(
    n_components=2,
    random_state=42,
)

pca_coordinates = pca.fit_transform(X_processed)

df["PCA1"] = pca_coordinates[:, 0]

df["PCA2"] = pca_coordinates[:, 1]

explained_variance = pca.explained_variance_ratio_

total_variance = explained_variance.sum()

print("\nPCA explained variance:")
print(f"PC1: {explained_variance[0] * 100:.2f}%")

print(f"PC2: {explained_variance[1] * 100:.2f}%")

print(f"Total 2D variance: {total_variance * 100:.2f}%")


# =========================================================
# FIGURE 26 - PCA CLUSTER VISUALIZATION
# =========================================================

plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=df,
    x="PCA1",
    y="PCA2",
    hue="Cluster",
    palette="tab10",
    alpha=0.65,
    s=35,
)

plt.title("Customer Segments Visualized with PCA")

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.legend(title="Cluster")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "26_customer_segments_pca.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# =========================================================
# CLUSTER SIZES
# =========================================================

cluster_sizes = (
    df["Cluster"]
    .value_counts()
    .sort_index()
    .rename_axis("Cluster")
    .reset_index(name="Customers")
)

cluster_sizes["Percentage"] = (cluster_sizes["Customers"] / len(df) * 100).round(2)

print("\nCluster sizes:")
print(cluster_sizes)

cluster_sizes.to_csv(
    TABLES_DIR / "cluster_sizes.csv",
    index=False,
)


# =========================================================
# NUMERICAL CLUSTER PROFILE
# =========================================================

cluster_profile = (
    df.groupby(
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
        SeniorCitizenRate=(
            "SeniorCitizen",
            "mean",
        ),
        AutomaticPaymentRate=(
            "AutomaticPayment",
            "mean",
        ),
    )
    .reset_index()
)

cluster_profile["ChurnRate"] = cluster_profile["ChurnRate"] * 100

cluster_profile["SeniorCitizenRate"] = cluster_profile["SeniorCitizenRate"] * 100

cluster_profile["AutomaticPaymentRate"] = cluster_profile["AutomaticPaymentRate"] * 100

cluster_profile = cluster_profile.round(2)

print("\nNumerical cluster profile:")
print(cluster_profile)

cluster_profile.to_csv(
    TABLES_DIR / "cluster_numerical_profile.csv",
    index=False,
)


# =========================================================
# DOMINANT CATEGORY HELPER
# =========================================================


def dominant_category(
    group,
    column,
):
    counts = group[column].value_counts(normalize=True)

    category = counts.index[0]

    percentage = counts.iloc[0] * 100

    return (
        category,
        percentage,
    )


# =========================================================
# CATEGORY PROFILE
# =========================================================

category_records = []

profile_columns = [
    "Contract",
    "InternetService",
    "PaymentMethod",
    "OnlineSecurity",
    "TechSupport",
    "PaperlessBilling",
]

for cluster, group in df.groupby(
    "Cluster",
    observed=True,
):
    record = {
        "Cluster": cluster,
    }

    for column in profile_columns:
        category, percentage = dominant_category(
            group,
            column,
        )

        record[f"{column}_Dominant"] = category

        record[f"{column}_Share"] = round(
            percentage,
            2,
        )

    category_records.append(record)


category_profile = pd.DataFrame(category_records)

print("\nDominant categorical characteristics:")

print(category_profile)

category_profile.to_csv(
    TABLES_DIR / "cluster_category_profile.csv",
    index=False,
)


# =========================================================
# FIGURE 27 - CHURN RATE BY CLUSTER
# =========================================================

cluster_churn = cluster_profile[
    [
        "Cluster",
        "Customers",
        "ChurnRate",
    ]
].copy()

plt.figure(figsize=(8, 6))

ax = sns.barplot(
    data=cluster_churn,
    x="Cluster",
    y="ChurnRate",
)

plt.title("Observed Churn Rate by Customer Segment")

plt.xlabel("Cluster")

plt.ylabel("Churn Rate (%)")

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.1f",
    )

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "27_cluster_churn_rates.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# =========================================================
# FIGURE 28 - STANDARDIZED PROFILE HEATMAP
# =========================================================

heatmap_columns = [
    "AvgTenure",
    "AvgMonthlyCharges",
    "AvgTotalCharges",
    "AvgServices",
    "ChurnRate",
    "AutomaticPaymentRate",
]

heatmap_data = cluster_profile.set_index("Cluster")[heatmap_columns]

standardized_heatmap = (heatmap_data - heatmap_data.mean()) / heatmap_data.std(ddof=0)

plt.figure(figsize=(11, 6))

sns.heatmap(
    standardized_heatmap,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
)

plt.title("Standardized Customer Segment Profiles")

plt.xlabel("Customer Characteristics")

plt.ylabel("Cluster")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "28_cluster_profile_heatmap.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# =========================================================
# CONTRACT DISTRIBUTION BY CLUSTER
# =========================================================

contract_distribution = (
    pd.crosstab(
        df["Cluster"],
        df["Contract"],
        normalize="index",
    )
    * 100
).round(2)

contract_distribution.to_csv(TABLES_DIR / "cluster_contract_distribution.csv")


# =========================================================
# INTERNET DISTRIBUTION BY CLUSTER
# =========================================================

internet_distribution = (
    pd.crosstab(
        df["Cluster"],
        df["InternetService"],
        normalize="index",
    )
    * 100
).round(2)

internet_distribution.to_csv(TABLES_DIR / "cluster_internet_distribution.csv")


# =========================================================
# PAYMENT DISTRIBUTION BY CLUSTER
# =========================================================

payment_distribution = (
    pd.crosstab(
        df["Cluster"],
        df["PaymentMethod"],
        normalize="index",
    )
    * 100
).round(2)

payment_distribution.to_csv(TABLES_DIR / "cluster_payment_distribution.csv")


# =========================================================
# SAVE SEGMENTED DATASET
# =========================================================

df.to_csv(
    SEGMENTED_DATA_PATH,
    index=False,
)


# =========================================================
# SUMMARY
# =========================================================

summary_path = TABLES_DIR / "clustering_summary.txt"

with open(
    summary_path,
    "w",
    encoding="utf-8",
) as file:
    file.write("TELCO CUSTOMER SEGMENTATION SUMMARY\n")

    file.write("=" * 65 + "\n\n")

    file.write(f"Customers analysed: {len(df)}\n")

    file.write(f"Selected K: {best_k}\n")

    file.write(f"Silhouette score: {best_row['Silhouette']:.4f}\n")

    file.write(f"Davies-Bouldin score: {best_row['DaviesBouldin']:.4f}\n")

    file.write(f"Calinski-Harabasz score: {best_row['CalinskiHarabasz']:.2f}\n")

    file.write(f"PCA 2D explained variance: {total_variance * 100:.2f}%\n\n")

    file.write("Cluster sizes:\n")

    file.write(cluster_sizes.to_string(index=False))

    file.write("\n\nNumerical profiles:\n")

    file.write(cluster_profile.to_string(index=False))

    file.write("\n\nDominant categories:\n")

    file.write(category_profile.to_string(index=False))


# =========================================================
# COMPLETE
# =========================================================

print("\n" + "=" * 78)
print("CUSTOMER SEGMENTATION COMPLETED")
print("=" * 78)

print("\nSelected K:")
print(best_k)

print("\nSegmented dataset saved to:")
print(SEGMENTED_DATA_PATH)

print("\nTotal figures available:")
print(len(list(FIGURES_DIR.glob("*.png"))))

print("\nClustering summary:")
print(summary_path)
