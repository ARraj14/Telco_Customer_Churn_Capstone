from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.base import clone
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
    cross_val_predict,
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

TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

MODELS_DIR = PROJECT_ROOT / "models"

TABLES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(DATA_PATH)

print("=" * 80)
print("TELCO CUSTOMER CHURN - SUPERVISED MODELING")
print("TRAINING-ONLY MODEL SELECTION + FINAL HOLDOUT EVALUATION")
print("=" * 80)

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
# TRAIN / FINAL TEST SPLIT
# =========================================================
#
# IMPORTANT:
# The holdout test set is created once here.
#
# Model-family selection and hyperparameter selection are
# performed ONLY using X_train / y_train.
#
# X_test / y_test are used only after the winning model
# has already been selected.
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


print("\nTraining shape:")
print(X_train.shape)

print("\nFinal holdout test shape:")
print(X_test.shape)

print("\nTraining churn rate:")
print(f"{y_train.mean() * 100:.2f}%")

print("\nHoldout churn rate:")
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
# TRAINING-ONLY MODEL COMPARISON
# =========================================================

comparison_records = []

trained_models = {}

oof_probabilities_by_model = {}

oof_predictions_by_model = {}

best_params_by_model = {}


for model_name, (
    classifier,
    param_grid,
) in models.items():
    print("\n" + "=" * 80)
    print(f"TRAINING / TUNING: {model_name}")
    print("=" * 80)

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

    # -----------------------------------------------------
    # Hyperparameter tuning:
    # TRAINING DATA ONLY
    # -----------------------------------------------------

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

    best_params_by_model[model_name] = search.best_params_

    # -----------------------------------------------------
    # Training out-of-fold predictions
    #
    # Used for diagnostic comparison only.
    # The final holdout test set is NOT touched here.
    # -----------------------------------------------------

    oof_probabilities = cross_val_predict(
        clone(best_model),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    oof_predictions = (oof_probabilities >= 0.50).astype(int)

    oof_probabilities_by_model[model_name] = oof_probabilities

    oof_predictions_by_model[model_name] = oof_predictions

    grid_cv_auc = float(search.best_score_)

    oof_auc = roc_auc_score(
        y_train,
        oof_probabilities,
    )

    record = {
        "Model": model_name,
        "GridSearch_CV_ROC_AUC": grid_cv_auc,
        "OOF_ROC_AUC": oof_auc,
        "OOF_Accuracy": accuracy_score(
            y_train,
            oof_predictions,
        ),
        "OOF_Balanced_Accuracy": balanced_accuracy_score(
            y_train,
            oof_predictions,
        ),
        "OOF_Precision": precision_score(
            y_train,
            oof_predictions,
            zero_division=0,
        ),
        "OOF_Recall": recall_score(
            y_train,
            oof_predictions,
            zero_division=0,
        ),
        "OOF_F1": f1_score(
            y_train,
            oof_predictions,
            zero_division=0,
        ),
        "BestParameters": json.dumps(
            search.best_params_,
            sort_keys=True,
        ),
    }

    comparison_records.append(record)

    print("\nBest parameters:")
    print(search.best_params_)

    print("\nGrid-search CV ROC-AUC:")
    print(f"{grid_cv_auc:.4f}")

    print("\nTraining OOF ROC-AUC:")
    print(f"{oof_auc:.4f}")

    print("\nTraining OOF metrics:")
    print(f"Accuracy: {record['OOF_Accuracy']:.4f}")

    print(f"Balanced Accuracy: {record['OOF_Balanced_Accuracy']:.4f}")

    print(f"Precision: {record['OOF_Precision']:.4f}")

    print(f"Recall: {record['OOF_Recall']:.4f}")

    print(f"F1: {record['OOF_F1']:.4f}")


# =========================================================
# MODEL SELECTION
# =========================================================
#
# CRITICAL METHODOLOGY:
#
# The winning model is selected ONLY from training CV.
#
# No holdout-test metric is involved in this decision.
# =========================================================

comparison_df = pd.DataFrame(comparison_records)


comparison_df = comparison_df.sort_values(
    [
        "GridSearch_CV_ROC_AUC",
        "OOF_ROC_AUC",
    ],
    ascending=False,
).reset_index(drop=True)


best_model_name = comparison_df.loc[
    0,
    "Model",
]


comparison_df["Selected"] = comparison_df["Model"] == best_model_name


print("\n" + "=" * 80)
print("TRAINING-ONLY MODEL COMPARISON")
print("=" * 80)

print(comparison_df.round(4))


comparison_df.to_csv(
    TABLES_DIR / "model_comparison.csv",
    index=False,
)


print("\nSelected model based ONLY on training CV:")
print(best_model_name)

print("\nSelection criterion:")
print("Highest GridSearch 5-fold CV ROC-AUC")


best_model = trained_models[best_model_name]


# =========================================================
# FIGURE 17
# TRAINING OOF ROC CURVES
# =========================================================

plt.figure(figsize=(8, 7))

roc_table = []


for (
    model_name,
    probabilities,
) in oof_probabilities_by_model.items():
    fpr, tpr, thresholds = roc_curve(
        y_train,
        probabilities,
    )

    auc = roc_auc_score(
        y_train,
        probabilities,
    )

    plt.plot(
        fpr,
        tpr,
        label=(f"{model_name} (OOF AUC={auc:.3f})"),
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
                "Threshold": current_threshold,
            }
        )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier",
)

plt.title("Training Out-of-Fold ROC Curves")

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
# FIGURE 18
# TRAINING-ONLY MODEL COMPARISON
# =========================================================

comparison_long = comparison_df[
    [
        "Model",
        "OOF_ROC_AUC",
        "OOF_Balanced_Accuracy",
        "OOF_Precision",
        "OOF_Recall",
        "OOF_F1",
    ]
].melt(
    id_vars="Model",
    var_name="Metric",
    value_name="Score",
)


metric_name_mapping = {
    "OOF_ROC_AUC": "ROC-AUC",
    "OOF_Balanced_Accuracy": "Balanced Accuracy",
    "OOF_Precision": "Precision",
    "OOF_Recall": "Recall",
    "OOF_F1": "F1",
}


comparison_long["Metric"] = comparison_long["Metric"].map(metric_name_mapping)


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

plt.title("Training Out-of-Fold Model Performance Comparison")

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
# FINAL HOLDOUT TEST
# =========================================================
#
# The winning model has already been fixed.
# This is the FIRST use of the test set for model
# performance evaluation.
# =========================================================

print("\n" + "=" * 80)
print("FINAL HOLDOUT TEST EVALUATION")
print("=" * 80)

print("\nLocked model:")

print(best_model_name)


test_probabilities = best_model.predict_proba(X_test)[:, 1]


test_predictions = (test_probabilities >= 0.50).astype(int)


test_metrics = {
    "Model": best_model_name,
    "Test_ROC_AUC": roc_auc_score(
        y_test,
        test_probabilities,
    ),
    "Accuracy": accuracy_score(
        y_test,
        test_predictions,
    ),
    "Balanced_Accuracy": balanced_accuracy_score(
        y_test,
        test_predictions,
    ),
    "Precision": precision_score(
        y_test,
        test_predictions,
        zero_division=0,
    ),
    "Recall": recall_score(
        y_test,
        test_predictions,
        zero_division=0,
    ),
    "F1": f1_score(
        y_test,
        test_predictions,
        zero_division=0,
    ),
}


print("\nFinal holdout metrics:")

for key, value in test_metrics.items():
    if key == "Model":
        print(f"{key}: {value}")

    else:
        print(f"{key}: {value:.4f}")


pd.DataFrame([test_metrics]).to_csv(
    TABLES_DIR / "selected_model_test_metrics.csv",
    index=False,
)


# =========================================================
# TEST PREDICTIONS
# =========================================================

prediction_output = pd.DataFrame(
    {
        "Actual": y_test.values,
        "Predicted": test_predictions,
        "ChurnProbability": test_probabilities,
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
    test_predictions,
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


print("\nFinal holdout confusion matrix:")
print(cm_df)


cm_df.to_csv(TABLES_DIR / "best_model_confusion_matrix.csv")


plt.figure(figsize=(7, 6))

sns.heatmap(
    cm_df,
    annot=True,
    fmt="d",
    cmap="Blues",
)

plt.title(f"Final Holdout Confusion Matrix - {best_model_name}")

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
# FEATURE EFFECT / IMPORTANCE
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
            "Effect": classifier.feature_importances_,
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
            "Effect": classifier.coef_[0],
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

    if hasattr(
        classifier,
        "coef_",
    ):
        plt.title(f"Top Absolute Model Coefficients - {best_model_name}")

        plt.xlabel("Absolute Standardized Coefficient")

    else:
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
# SAVE LOCKED MODEL
# =========================================================

model_path = MODELS_DIR / "best_churn_model.joblib"


joblib.dump(
    best_model,
    model_path,
)


# =========================================================
# MODEL SUMMARY JSON
# =========================================================

selected_training_row = comparison_df[comparison_df["Model"] == best_model_name].iloc[0]


summary = {
    "dataset_rows": int(len(df)),
    "training_rows": int(len(X_train)),
    "testing_rows": int(len(X_test)),
    "training_churn_rate": float(y_train.mean()),
    "testing_churn_rate": float(y_test.mean()),
    "selection_method": (
        "Model family and hyperparameters selected "
        "using training-only 5-fold stratified "
        "cross-validation."
    ),
    "selection_metric": "GridSearch_CV_ROC_AUC",
    "selected_model": best_model_name,
    "selected_model_cv_roc_auc": float(selected_training_row["GridSearch_CV_ROC_AUC"]),
    "selected_model_oof_roc_auc": float(selected_training_row["OOF_ROC_AUC"]),
    "best_parameters": best_params_by_model[best_model_name],
    "test_roc_auc": float(test_metrics["Test_ROC_AUC"]),
    "accuracy": float(test_metrics["Accuracy"]),
    "balanced_accuracy": float(test_metrics["Balanced_Accuracy"]),
    "precision": float(test_metrics["Precision"]),
    "recall": float(test_metrics["Recall"]),
    "f1": float(test_metrics["F1"]),
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

print("\n" + "=" * 80)
print("SUPERVISED MODELING COMPLETED")
print("=" * 80)

print("\nSelection was based ONLY on training CV.")

print("\nSelected model:")
print(best_model_name)

print("\nSelected model CV ROC-AUC:")
print(f"{summary['selected_model_cv_roc_auc']:.4f}")

print("\nFinal holdout ROC-AUC:")
print(f"{summary['test_roc_auc']:.4f}")

print("\nFinal holdout recall:")
print(f"{summary['recall']:.4f}")

print("\nSaved model:")
print(model_path)

print("\nModel comparison:")
print(TABLES_DIR / "model_comparison.csv")

print("\nFinal test metrics:")
print(TABLES_DIR / "selected_model_test_metrics.csv")
