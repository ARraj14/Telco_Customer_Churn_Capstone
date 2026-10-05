from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "telco_churn_cleaned.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(DATA_PATH)

print("=" * 75)
print("TELCO CUSTOMER CHURN - SUPERVISED MODELING")
print("=" * 75)

print("\nDataset shape:")
print(df.shape)


# =========================================================
# FEATURES AND TARGET
# =========================================================

target = "ChurnValue"

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
y = df[target].copy()


print("\nFeatures used:")
print(len(feature_columns))

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget percentages:")
print(y.value_counts(normalize=True).mul(100).round(2))


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTrain shape:")
print(X_train.shape)

print("\nTest shape:")
print(X_test.shape)

print("\nTraining churn rate:")
print(f"{y_train.mean() * 100:.2f}%")

print("\nTesting churn rate:")
print(f"{y_test.mean() * 100:.2f}%")


# =========================================================
# PREPROCESSING
# =========================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features,
        ),
    ]
)


# =========================================================
# CROSS-VALIDATION
# =========================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


# =========================================================
# MODEL DEFINITIONS
# =========================================================

models = {
    "Logistic Regression": (
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42,
        ),
        {
            "model__C": [
                0.01,
                0.1,
                1.0,
                10.0,
            ]
        },
    ),
    "Decision Tree": (
        DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42,
        ),
        {
            "model__max_depth": [
                3,
                5,
                7,
                None,
            ],
            "model__min_samples_leaf": [
                2,
                5,
                10,
            ],
        },
    ),
    "Random Forest": (
        RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        {
            "model__max_depth": [
                None,
                8,
                12,
            ],
            "model__min_samples_leaf": [
                2,
                5,
                10,
            ],
            "model__max_features": [
                "sqrt",
                0.7,
            ],
        },
    ),
}


# =========================================================
# TRAIN AND TUNE MODELS
# =========================================================

results = []
trained_models = {}

for model_name, (
    classifier,
    param_grid,
) in models.items():
    print("\n" + "=" * 75)
    print(f"TRAINING: {model_name}")
    print("=" * 75)

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                classifier,
            ),
        ]
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )

    search.fit(
        X_train,
        y_train,
    )

    best_model = search.best_estimator_

    trained_models[model_name] = best_model

    probabilities = best_model.predict_proba(X_test)[:, 1]

    predictions = best_model.predict(X_test)

    metrics = {
        "Model": model_name,
        "CV_ROC_AUC": (search.best_score_),
        "Test_ROC_AUC": (
            roc_auc_score(
                y_test,
                probabilities,
            )
        ),
        "Accuracy": (
            accuracy_score(
                y_test,
                predictions,
            )
        ),
        "Balanced_Accuracy": (
            balanced_accuracy_score(
                y_test,
                predictions,
            )
        ),
        "Precision": (
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "Recall": (
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "F1": (
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
    }

    results.append(metrics)

    print("\nBest parameters:")
    print(search.best_params_)

    print("\nBest CV ROC-AUC:")
    print(f"{search.best_score_:.4f}")

    print("\nTest metrics:")

    for metric, value in metrics.items():
        if metric != "Model":
            print(f"{metric}: {value:.4f}")


# =========================================================
# MODEL COMPARISON
# =========================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "Test_ROC_AUC",
    ascending=False,
).reset_index(drop=True)

print("\n" + "=" * 75)
print("MODEL COMPARISON")
print("=" * 75)

print(results_df.round(4))

results_df.to_csv(
    TABLES_DIR / "model_comparison.csv",
    index=False,
)


# =========================================================
# SELECT BEST MODEL
# =========================================================

best_model_name = results_df.loc[
    0,
    "Model",
]

best_model = trained_models[best_model_name]

print("\nSelected best model:")
print(best_model_name)


# =========================================================
# FINAL PREDICTIONS
# =========================================================

best_probabilities = best_model.predict_proba(X_test)[:, 1]

best_predictions = best_model.predict(X_test)


prediction_output = pd.DataFrame(
    {
        "Actual": y_test.values,
        "Predicted": best_predictions,
        "ChurnProbability": (best_probabilities),
    },
    index=y_test.index,
)

prediction_output.insert(
    0,
    "customerID",
    df.loc[
        y_test.index,
        "customerID",
    ],
)

prediction_output = prediction_output.sort_values(
    "ChurnProbability",
    ascending=False,
)

prediction_output.to_csv(
    TABLES_DIR / "test_predictions.csv",
    index=False,
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_test,
    best_predictions,
)

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual No Churn",
        "Actual Churn",
    ],
    columns=[
        "Predicted No Churn",
        "Predicted Churn",
    ],
)

print("\nConfusion matrix:")
print(cm_df)

cm_df.to_csv(TABLES_DIR / "best_model_confusion_matrix.csv")


plt.figure(figsize=(7, 6))

sns.heatmap(
    cm_df,
    annot=True,
    fmt="d",
    cmap="Blues",
)

plt.title(f"Confusion Matrix - {best_model_name}")

plt.ylabel("Actual")
plt.xlabel("Predicted")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "16_best_model_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# =========================================================
# ROC CURVES FOR ALL MODELS
# =========================================================

plt.figure(figsize=(8, 7))

roc_table = []

for model_name, model in trained_models.items():
    probability = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(
        y_test,
        probability,
    )

    auc = roc_auc_score(
        y_test,
        probability,
    )

    plt.plot(
        fpr,
        tpr,
        label=(f"{model_name} (AUC={auc:.3f})"),
    )

    for (
        current_fpr,
        current_tpr,
        current_threshold,
    ) in zip(
        fpr,
        tpr,
        thresholds,
    ):
        roc_table.append(
            {
                "Model": model_name,
                "FPR": current_fpr,
                "TPR": current_tpr,
                "Threshold": (current_threshold),
            }
        )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier",
)

plt.title("ROC Curves - Churn Prediction Models")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "17_model_roc_curves.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


pd.DataFrame(roc_table).to_csv(
    TABLES_DIR / "model_roc_curve_points.csv",
    index=False,
)


# =========================================================
# MODEL COMPARISON FIGURE
# =========================================================

comparison_long = results_df[
    [
        "Model",
        "Test_ROC_AUC",
        "Balanced_Accuracy",
        "Precision",
        "Recall",
        "F1",
    ]
].melt(
    id_vars="Model",
    var_name="Metric",
    value_name="Score",
)


plt.figure(figsize=(12, 7))

sns.barplot(
    data=comparison_long,
    x="Metric",
    y="Score",
    hue="Model",
)

plt.ylim(
    0,
    1,
)

plt.title("Predictive Model Performance Comparison")

plt.xlabel("Evaluation Metric")

plt.ylabel("Score")

plt.xticks(
    rotation=20,
    ha="right",
)

plt.legend(title="Model")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "18_model_performance_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# =========================================================
# FEATURE IMPORTANCE / COEFFICIENT ANALYSIS
# =========================================================

fitted_preprocessor = best_model.named_steps["preprocessor"]

feature_names = fitted_preprocessor.get_feature_names_out()

classifier = best_model.named_steps["model"]


if hasattr(
    classifier,
    "feature_importances_",
):
    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Effect": (classifier.feature_importances_),
        }
    )

    importance_df["Importance"] = importance_df["Effect"]


elif hasattr(
    classifier,
    "coef_",
):
    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Effect": (classifier.coef_[0]),
        }
    )

    importance_df["Importance"] = importance_df["Effect"].abs()


else:
    importance_df = None


if importance_df is not None:
    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False,
    )

    importance_df.to_csv(
        TABLES_DIR / "best_model_feature_importance.csv",
        index=False,
    )

    top_features = importance_df.head(20).sort_values(
        "Importance",
        ascending=True,
    )

    plt.figure(figsize=(10, 8))

    sns.barplot(
        data=top_features,
        x="Importance",
        y="Feature",
    )

    plt.title(f"Top Predictive Features - {best_model_name}")

    plt.xlabel("Feature Importance")

    plt.ylabel("Feature")

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "19_best_model_feature_importance.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# =========================================================
# SAVE BEST MODEL
# =========================================================

model_path = MODELS_DIR / "best_churn_model.joblib"

joblib.dump(
    best_model,
    model_path,
)


# =========================================================
# MODEL SUMMARY JSON
# =========================================================

best_row = results_df.iloc[0]

summary = {
    "dataset_rows": int(len(df)),
    "training_rows": int(len(X_train)),
    "testing_rows": int(len(X_test)),
    "training_churn_rate": float(y_train.mean()),
    "testing_churn_rate": float(y_test.mean()),
    "selected_model": (best_model_name),
    "test_roc_auc": float(best_row["Test_ROC_AUC"]),
    "accuracy": float(best_row["Accuracy"]),
    "balanced_accuracy": float(best_row["Balanced_Accuracy"]),
    "precision": float(best_row["Precision"]),
    "recall": float(best_row["Recall"]),
    "f1": float(best_row["F1"]),
}

with open(
    TABLES_DIR / "model_summary.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        summary,
        file,
        indent=4,
    )


# =========================================================
# FINAL OUTPUT
# =========================================================

print("\n" + "=" * 75)
print("SUPERVISED MODELING COMPLETED")
print("=" * 75)

print("\nBest model:")
print(best_model_name)

print("\nFinal metrics:")
print(results_df.iloc[0])

print("\nSaved model:")
print(model_path)

print("\nFigures now available:")
print(len(list(FIGURES_DIR.glob("*.png"))))
