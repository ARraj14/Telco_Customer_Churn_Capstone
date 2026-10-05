from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
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

SEGMENTATION_MODEL_PATH = (
    PROJECT_ROOT / "models" / "customer_segmentation_pipeline.joblib"
)

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

SEGMENTATION_MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(DATA_PATH)

print("=" * 80)
print("TELCO CUSTOMER CHURN - LEAKAGE-SAFE CUSTOMER SEGMENTATION")
print("=" * 80)

print("\nDataset shape:")
print(df.shape)


# =========================================================
# CLUSTERING FEATURES
# =========================================================
#
# IMPORTANT:
# Churn / ChurnValue are NOT clustering inputs.
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


print("\nClustering features:")

for feature in cluster_features:
    print(f"- {feature}")


# =========================================================
# RECREATE TRAIN / HOLDOUT PARTITION
# =========================================================
#
# The same random_state / stratification used by the
# supervised model is reused here.
#
# Clustering itself does not use the churn target.
# The split simply ensures that all clustering decisions
# are made before looking at holdout customers.
# =========================================================

all_indices = df.index

train_indices, test_indices = train_test_split(
    all_indices,
    test_size=0.20,
    random_state=42,
    stratify=df["ChurnValue"],
)


df["ModelSplit"] = "Training"

df.loc[
    test_indices,
    "ModelSplit",
] = "Holdout"


X_train = df.loc[
    train_indices,
    cluster_features,
].copy()


X_holdout = df.loc[
    test_indices,
    cluster_features,
].copy()


X_full = df[cluster_features].copy()


print("\nTraining customers used to build clusters:")
print(len(X_train))

print("\nHoldout customers:")
print(len(X_holdout))


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


# Fit preprocessing on TRAINING CUSTOMERS ONLY.

X_train_processed = preprocessor.fit_transform(X_train)


X_holdout_processed = preprocessor.transform(X_holdout)


X_full_processed = preprocessor.transform(X_full)


print("\nProcessed training shape:")
print(X_train_processed.shape)

print("\nProcessed holdout shape:")
print(X_holdout_processed.shape)


# =========================================================
# EVALUATE K VALUES USING TRAINING DATA ONLY
# =========================================================

cluster_results = []


for k in range(
    2,
    9,
):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20,
    )

    labels = model.fit_predict(X_train_processed)

    cluster_results.append(
        {
            "K": k,
            "Inertia": model.inertia_,
            "Silhouette": silhouette_score(
                X_train_processed,
                labels,
            ),
            "DaviesBouldin": davies_bouldin_score(
                X_train_processed,
                labels,
            ),
            "CalinskiHarabasz": calinski_harabasz_score(
                X_train_processed,
                labels,
            ),
        }
    )


metrics_df = pd.DataFrame(cluster_results)


print("\nTraining-only clustering evaluation:")
print(metrics_df.round(4))


metrics_df.to_csv(
    TABLES_DIR / "clustering_k_evaluation.csv",
    index=False,
)


# =========================================================
# SELECT K USING TRAINING SILHOUETTE SCORE
# =========================================================

best_row = metrics_df.sort_values(
    [
        "Silhouette",
        "CalinskiHarabasz",
    ],
    ascending=[
        False,
        False,
    ],
).iloc[0]


best_k = int(best_row["K"])


print("\nSelected K:")
print(best_k)

print("\nTraining silhouette score:")
print(f"{best_row['Silhouette']:.4f}")

print("\nTraining Davies-Bouldin score:")
print(f"{best_row['DaviesBouldin']:.4f}")

print("\nTraining Calinski-Harabasz score:")
print(f"{best_row['CalinskiHarabasz']:.2f}")


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


plt.axvline(
    best_k,
    linestyle="--",
)


plt.title("Training-Only K-Means Elbow Method")

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
# FIGURE 25 - SILHOUETTE SCORE
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


plt.title("Training Silhouette Score by Number of Clusters")

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
#
# Fit ONLY on training customers.
# =========================================================

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=20,
)


training_cluster_labels = kmeans.fit_predict(X_train_processed)


# Predict clusters for HOLDOUT customers using frozen model.

holdout_cluster_labels = kmeans.predict(X_holdout_processed)


# Predict clusters for all customers using same frozen model.

full_cluster_labels = kmeans.predict(X_full_processed)


df["Cluster"] = full_cluster_labels


# =========================================================
# CLUSTER STABILITY ANALYSIS
# =========================================================
#
# Adjusted Rand Index is invariant to cluster label
# permutations and measures whether clustering is stable
# across random initializations.
# =========================================================

stability_records = []


stability_seeds = [
    7,
    21,
    42,
    99,
    123,
]


reference_labels = training_cluster_labels


for seed in stability_seeds:
    stability_model = KMeans(
        n_clusters=best_k,
        random_state=seed,
        n_init=20,
    )

    stability_labels = stability_model.fit_predict(X_train_processed)

    ari = adjusted_rand_score(
        reference_labels,
        stability_labels,
    )

    stability_records.append(
        {
            "RandomState": seed,
            "AdjustedRandIndex": ari,
        }
    )


stability_df = pd.DataFrame(stability_records)


mean_stability_ari = stability_df["AdjustedRandIndex"].mean()


print("\nCluster stability across random seeds:")
print(stability_df.round(4))

print("\nMean stability ARI:")
print(f"{mean_stability_ari:.4f}")


stability_df.to_csv(
    TABLES_DIR / "cluster_stability_analysis.csv",
    index=False,
)


# =========================================================
# PCA
# =========================================================
#
# PCA is fitted on training processed data and then
# applied to the full dataset for visualization.
# =========================================================

pca = PCA(
    n_components=2,
    random_state=42,
)


pca.fit(X_train_processed)


full_pca_coordinates = pca.transform(X_full_processed)


df["PCA1"] = full_pca_coordinates[
    :,
    0,
]


df["PCA2"] = full_pca_coordinates[
    :,
    1,
]


explained_variance = pca.explained_variance_ratio_


total_variance = explained_variance.sum()


print("\nTraining PCA explained variance:")

print(f"PC1: {explained_variance[0] * 100:.2f}%")

print(f"PC2: {explained_variance[1] * 100:.2f}%")

print(f"Total 2D variance: {total_variance * 100:.2f}%")


# =========================================================
# FIGURE 26 - PCA VISUALIZATION
# =========================================================

plt.figure(figsize=(10, 7))


sns.scatterplot(
    data=df,
    x="PCA1",
    y="PCA2",
    hue="Cluster",
    style="ModelSplit",
    palette="tab10",
    alpha=0.65,
    s=35,
)


plt.title("Customer Segments from Training-Fitted K-Means")

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.legend(
    title="Cluster / Split",
    bbox_to_anchor=(
        1.02,
        1,
    ),
    loc="upper left",
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "26_customer_segments_pca.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# =========================================================
# CLUSTER SIZE TABLE
# =========================================================

cluster_sizes = (
    df["Cluster"]
    .value_counts()
    .sort_index()
    .rename_axis("Cluster")
    .reset_index(name="Customers")
)


cluster_sizes["Percentage"] = (cluster_sizes["Customers"] / len(df) * 100).round(2)


print("\nFull-dataset cluster sizes:")
print(cluster_sizes)


cluster_sizes.to_csv(
    TABLES_DIR / "cluster_sizes.csv",
    index=False,
)


# =========================================================
# PROFILE FUNCTION
# =========================================================


def create_cluster_profile(
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

    percentage_columns = [
        "ChurnRate",
        "SeniorCitizenRate",
        "AutomaticPaymentRate",
    ]

    for column in percentage_columns:
        profile[column] *= 100

    return profile.round(2)


# =========================================================
# TRAINING / HOLDOUT / FULL PROFILES
# =========================================================

training_profile = create_cluster_profile(df[df["ModelSplit"] == "Training"])


holdout_profile = create_cluster_profile(df[df["ModelSplit"] == "Holdout"])


full_profile = create_cluster_profile(df)


print("\nTraining cluster profile:")
print(training_profile)

print("\nHoldout cluster profile:")
print(holdout_profile)

print("\nFull descriptive cluster profile:")
print(full_profile)


training_profile.to_csv(
    TABLES_DIR / "cluster_training_profile.csv",
    index=False,
)


holdout_profile.to_csv(
    TABLES_DIR / "cluster_holdout_profile.csv",
    index=False,
)


full_profile.to_csv(
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

    return (
        counts.index[0],
        counts.iloc[0] * 100,
    )


# =========================================================
# TRAINING CATEGORY PROFILE
# =========================================================
#
# Business segment names will later be derived from this
# TRAINING profile only.
# =========================================================

training_df = df[df["ModelSplit"] == "Training"]


profile_columns = [
    "Contract",
    "InternetService",
    "PaymentMethod",
    "OnlineSecurity",
    "TechSupport",
    "PaperlessBilling",
]


category_records = []


for (
    cluster,
    group,
) in training_df.groupby(
    "Cluster",
    observed=True,
):
    record = {
        "Cluster": cluster,
    }

    for column in profile_columns:
        (
            category,
            percentage,
        ) = dominant_category(
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


print("\nTraining dominant categorical characteristics:")

print(category_profile)


category_profile.to_csv(
    TABLES_DIR / "cluster_category_profile.csv",
    index=False,
)


# =========================================================
# FIGURE 27 - HOLDOUT CHURN BY CLUSTER
# =========================================================
#
# This is genuinely out-of-sample because the clustering
# model was fitted before holdout customers were assigned.
# =========================================================

plt.figure(figsize=(8, 6))


ax = sns.barplot(
    data=holdout_profile,
    x="Cluster",
    y="ChurnRate",
)


plt.title("Holdout Churn Rate by Training-Defined Cluster")

plt.xlabel("Cluster")

plt.ylabel("Holdout Churn Rate (%)")


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
# FIGURE 28 - TRAINING CLUSTER PROFILE HEATMAP
# =========================================================

heatmap_columns = [
    "AvgTenure",
    "AvgMonthlyCharges",
    "AvgTotalCharges",
    "AvgServices",
    "ChurnRate",
    "AutomaticPaymentRate",
]


heatmap_data = training_profile.set_index("Cluster")[heatmap_columns]


standardized_heatmap = (heatmap_data - heatmap_data.mean()) / heatmap_data.std(ddof=0)


plt.figure(figsize=(11, 6))


sns.heatmap(
    standardized_heatmap,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
)


plt.title("Training-Defined Standardized Cluster Profiles")

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
# DISTRIBUTION TABLES
# =========================================================

contract_distribution = (
    pd.crosstab(
        training_df["Cluster"],
        training_df["Contract"],
        normalize="index",
    )
    * 100
).round(2)


internet_distribution = (
    pd.crosstab(
        training_df["Cluster"],
        training_df["InternetService"],
        normalize="index",
    )
    * 100
).round(2)


payment_distribution = (
    pd.crosstab(
        training_df["Cluster"],
        training_df["PaymentMethod"],
        normalize="index",
    )
    * 100
).round(2)


contract_distribution.to_csv(TABLES_DIR / "cluster_contract_distribution.csv")


internet_distribution.to_csv(TABLES_DIR / "cluster_internet_distribution.csv")


payment_distribution.to_csv(TABLES_DIR / "cluster_payment_distribution.csv")


# =========================================================
# SAVE SEGMENTATION MODEL BUNDLE
# =========================================================
#
# This allows a completely new customer to be transformed
# and assigned a cluster later.
# =========================================================

segmentation_bundle = {
    "preprocessor": preprocessor,
    "kmeans": kmeans,
    "pca": pca,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "cluster_features": cluster_features,
    "selected_k": best_k,
    "training_silhouette": float(best_row["Silhouette"]),
    "training_davies_bouldin": float(best_row["DaviesBouldin"]),
    "training_calinski_harabasz": float(best_row["CalinskiHarabasz"]),
    "mean_stability_ari": float(mean_stability_ari),
}


joblib.dump(
    segmentation_bundle,
    SEGMENTATION_MODEL_PATH,
)


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

    file.write("=" * 72 + "\n\n")

    file.write("METHODOLOGY\n")

    file.write(
        "K selection, preprocessing, PCA and K-Means fitting "
        "were performed using training customers only.\n"
    )

    file.write(
        "Holdout customers were assigned using the frozen "
        "training-fitted clustering model.\n"
    )

    file.write("Churn / ChurnValue were not clustering inputs.\n\n")

    file.write(f"Total customers: {len(df)}\n")

    file.write(f"Training customers: {len(train_indices)}\n")

    file.write(f"Holdout customers: {len(test_indices)}\n")

    file.write(f"Selected K: {best_k}\n")

    file.write(f"Training silhouette score: {best_row['Silhouette']:.4f}\n")

    file.write(f"Training Davies-Bouldin score: {best_row['DaviesBouldin']:.4f}\n")

    file.write(
        f"Training Calinski-Harabasz score: {best_row['CalinskiHarabasz']:.2f}\n"
    )

    file.write(f"Mean cluster stability ARI: {mean_stability_ari:.4f}\n")

    file.write(f"PCA 2D explained variance: {total_variance * 100:.2f}%\n\n")

    file.write("TRAINING CLUSTER PROFILE\n")

    file.write(training_profile.to_string(index=False))

    file.write("\n\nHOLDOUT CLUSTER PROFILE\n")

    file.write(holdout_profile.to_string(index=False))

    file.write("\n\nTRAINING DOMINANT CATEGORIES\n")

    file.write(category_profile.to_string(index=False))


# =========================================================
# COMPLETE
# =========================================================

print("\n" + "=" * 80)
print("LEAKAGE-SAFE CUSTOMER SEGMENTATION COMPLETED")
print("=" * 80)

print("\nSelected K:")
print(best_k)

print("\nMean cluster stability ARI:")
print(f"{mean_stability_ari:.4f}")

print("\nSaved segmentation pipeline:")
print(SEGMENTATION_MODEL_PATH)

print("\nSegmented dataset:")
print(SEGMENTED_DATA_PATH)

print("\nSummary:")
print(summary_path)
