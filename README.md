# Telco Customer Churn – End-to-End Data Science Capstone

## Project Overview

This project presents an end-to-end customer churn analysis using the IBM Telco Customer Churn dataset. The objective is not only to predict which customers are likely to churn, but also to understand the factors associated with churn, identify meaningful customer segments, prioritize customers for retention, and translate analytical findings into practical business recommendations.

The project combines data cleaning, exploratory data analysis, feature engineering, supervised machine learning, hyperparameter tuning, threshold optimization, error analysis, customer segmentation using K-Means, PCA visualization, out-of-fold risk scoring, and business-oriented retention prioritization.

The final dataset contains **7,043 customers**, with an overall churn rate of **26.54%**.

---

## Dataset

The original dataset contains:

| Item | Value |
|---|---:|
| Customers | 7,043 |
| Original Features | 21 |
| Churned Customers | 1,869 |
| Retained Customers | 5,174 |
| Overall Churn Rate | 26.54% |
| Duplicate Rows | 0 |
| Duplicate Customer IDs | 0 |

During validation, `TotalCharges` appeared to contain no standard missing values because the column was stored as text. After numerical conversion, **11 blank values** were discovered.

All 11 records belonged to customers with `tenure = 0`, indicating new customers who had not accumulated historical charges. Their `TotalCharges` values were therefore assigned `0.0` instead of deleting valid customer records.

The final cleaned dataset contains **7,043 rows and 27 columns with zero missing values**.

---

## Feature Engineering

Several additional features were created to improve analysis and modeling:

`ChurnValue` converts the churn target to 0/1.

`TenureGroup` divides customers into tenure ranges.

`NumServices` measures the number of active services.

`HasInternetService` identifies customers with internet service.

`ContractValue` represents increasing contract commitment.

`AutomaticPayment` identifies customers using automatic payment methods.

These engineered features were used for exploratory analysis, customer segmentation, and predictive modeling where appropriate.

---

## Exploratory Data Analysis

The exploratory analysis generated **15 dedicated EDA figures** and several supporting tables.

Important findings include:

| Customer Characteristic | Observed Churn Rate |
|---|---:|
| Overall | 26.54% |
| 0–12 month tenure | 47.44% |
| 61–72 month tenure | 6.61% |
| Month-to-month contract | 42.71% |
| One-year contract | 11.27% |
| Two-year contract | 2.83% |
| Fiber optic | 41.89% |
| Electronic check | 45.29% |
| Automatic payment | 15.98% |
| Manual payment | 34.67% |
| Senior citizens | 41.68% |

The highest-risk combined customer group discovered during multivariable analysis was:

**Month-to-month contract + Fiber optic internet + Electronic check**

This group contained **1,307 customers** and showed a churn rate of **60.37%**.

### Correlation Analysis

Among the numerical and engineered variables, the strongest correlations with churn were:

| Feature | Correlation with Churn |
|---|---:|
| ContractValue | -0.397 |
| tenure | -0.352 |
| HasInternetService | +0.228 |
| AutomaticPayment | -0.210 |
| TotalCharges | -0.198 |
| MonthlyCharges | +0.193 |
| SeniorCitizen | +0.151 |

These relationships are treated as associations rather than evidence of causation.

---

## Supervised Machine Learning

Three classification algorithms were trained and tuned using **5-fold Stratified Cross-Validation**:

| Model | CV ROC-AUC | Test ROC-AUC | Accuracy | Balanced Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.8462 | **0.8414** | 0.7509 | **0.7647** | 0.5201 | **0.7941** | **0.6286** |
| Logistic Regression | **0.8464** | 0.8407 | 0.7388 | 0.7531 | 0.5052 | 0.7834 | 0.6143 |
| Decision Tree | 0.8298 | 0.8322 | **0.7551** | 0.7556 | **0.5270** | 0.7567 | 0.6213 |

### Selected Model

The final selected model is:

**Random Forest Classifier**

Best hyperparameters:

```text
max_depth = 8
max_features = sqrt
min_samples_leaf = 10
n_estimators = 300
class_weight = balanced
```

The Random Forest achieved:

```text
Test ROC-AUC        0.8414
Accuracy            0.7509
Balanced Accuracy   0.7647
Precision           0.5201
Recall              0.7941
F1 Score            0.6286
```

Its cross-validation ROC-AUC and test ROC-AUC are very similar, providing evidence that the selected model generalizes reasonably well to unseen data.

### Confusion Matrix at 0.50 Threshold

```text
                       Predicted
                    No          Churn
Actual No           761           274
Actual Churn         77           297
```

The model correctly detected **297 of 374 churners** in the held-out test dataset.

---

## Classification Threshold Optimization

Because customer retention places greater importance on identifying potential churners, the default probability threshold of `0.50` was compared with a recall-oriented threshold.

Threshold optimization was performed using **out-of-fold predictions on the training dataset**, avoiding direct threshold tuning on the held-out test set.

The selected F2-oriented threshold was:

**0.31**

| Metric | Threshold 0.50 | Threshold 0.31 |
|---|---:|---:|
| Accuracy | 0.7509 | 0.6480 |
| Balanced Accuracy | 0.7647 | 0.7305 |
| Precision | 0.5201 | 0.4238 |
| Recall | 0.7941 | **0.9064** |
| F1 | 0.6286 | 0.5775 |
| F2 | 0.7184 | **0.7382** |

At the optimized threshold, false negatives decreased from **77 to 35**.

The `0.31` threshold is therefore treated as a **retention-oriented operating threshold**, while `0.50` remains useful when higher precision is preferred.

---

## Customer Segmentation

K-Means clustering was used to discover behavioral customer segments.

The clustering inputs did **not** include `Churn` or `ChurnValue`. Churn was analyzed only after the clusters were created.

Values of `K = 2` through `K = 8` were evaluated using Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score, and the Elbow Method.

The strongest solution was:

**K = 3**

```text
Silhouette Score        0.3003
Davies-Bouldin Index    1.2683
Calinski-Harabasz       3345.74
```

PCA was used for two-dimensional visualization, with the first two components explaining **64.37%** of total variation.

### Final Customer Segments

| Segment | Customers | Share | Avg Tenure | Avg Monthly Charges | Avg Services | Churn Rate |
|---|---:|---:|---:|---:|---:|---:|
| Low-Cost Stable Customers | 1,549 | 21.99% | 30.75 | 21.29 | 1.24 | **7.36%** |
| High-Value Loyal Customers | 2,303 | 32.70% | 57.10 | 89.15 | 5.58 | **14.63%** |
| High-Risk Short-Tenure Customers | 3,191 | 45.31% | 15.31 | 68.26 | 2.79 | **44.44%** |

The **High-Risk Short-Tenure Customers** segment contains almost half of the customer base and has approximately **6.04 times the churn rate** of the Low-Cost Stable segment.

Within this segment:

```text
Month-to-month contracts     87.78%
Electronic check             50.27%
Automatic payment            29.08%
Observed churn               44.44%
```

---

## Final Customer Risk Scoring System

The final stage combines predictive churn probabilities with customer segmentation.

Rather than generating risk scores from predictions made on the same observations used to train a model, **5-fold out-of-fold probabilities** were generated for all 7,043 customers.

Full-dataset OOF ROC-AUC:

**0.8450**

At the `0.31` retention threshold:

```text
Precision     0.4236
Recall        0.9197
F1            0.5801
F2            0.7452
```

The system identified:

```text
Actual churners detected    1,719
Actual churners missed        150
Total actual churners       1,869
```

### Risk Bands

| Risk Band | Customers | Actual Churn Rate |
|---|---:|---:|
| High Risk | 2,785 | **52.78%** |
| Retention Candidate | 1,273 | 19.56% |
| Below Retention Threshold | 2,985 | **5.03%** |

### Retention Priorities

| Priority | Customers | Actual Churn Rate |
|---|---:|---:|
| Critical | 2,293 | **56.00%** |
| High | 1,053 | 27.64% |
| Medium | 1,049 | 16.49% |
| Low | 2,648 | **4.57%** |

This strong separation between the Critical and Low-priority groups demonstrates that the combined segmentation and prediction system successfully concentrates observed churn risk.

---

## Business Recommendations

### High-Risk Short-Tenure Customers

This segment should receive the highest retention attention. Recommended interventions include improved early-tenure onboarding, proactive customer support, fiber-service quality monitoring, incentives for moving from month-to-month to longer contracts, automatic-payment incentives, and personalized retention offers.

### High-Value Loyal Customers

These customers have the highest average spending and service usage. Recommended strategies include loyalty benefits, premium support, personalized offers, and proactive service-quality monitoring.

### Low-Cost Stable Customers

These customers exhibit very low churn but relatively low service adoption. Retention spending can remain limited, while carefully selected cross-sell opportunities may increase customer value without damaging price sensitivity.

---

## Project Workflow

```text
Raw Dataset
      |
      v
Data Validation
      |
      v
Data Cleaning
      |
      v
Feature Engineering
      |
      v
Exploratory Data Analysis
      |
      v
Correlation & Multivariable Analysis
      |
      v
Supervised Model Training
      |
      v
5-Fold Cross-Validation
      |
      v
Hyperparameter Optimization
      |
      v
Random Forest Selection
      |
      v
Threshold Optimization
      |
      v
Error & Subgroup Analysis
      |
      v
K-Means Customer Segmentation
      |
      v
PCA Visualization
      |
      v
Business Segment Profiling
      |
      v
Out-of-Fold Risk Scoring
      |
      v
Retention Priority System
      |
      v
Final Deployment Model
```

---

## Project Structure

```text
Telco_Customer_Churn_Capstone/
|
├── data/
│   ├── Telco-Customer-Churn.csv
│   ├── telco_churn_cleaned.csv
│   ├── telco_churn_segmented.csv
│   └── telco_churn_segmented_labeled.csv
|
├── models/
│   ├── best_churn_model.joblib
│   └── final_churn_model_full.joblib
|
├── outputs/
│   ├── figures/
│   │   └── 35 generated analytical figures
│   └── tables/
│       └── validation, EDA, model, clustering and risk-analysis outputs
|
├── src/
│   ├── 01_validate_data.py
│   ├── 02_clean_data.py
│   ├── 03_eda.py
│   ├── 04_train_models.py
│   ├── 05_threshold_error_analysis.py
│   ├── 06_customer_segmentation.py
│   ├── 07_segment_business_analysis.py
│   ├── 08_final_risk_scoring.py
│   └── 09_final_validation.py
|
├── report/
├── README.md
├── pyproject.toml
└── uv.lock
```

---

## Technologies Used

```text
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Joblib
uv
Git / GitHub
```

---

## Key Deliverables

The completed project includes **35 analytical figures**, detailed numerical output tables, cleaned and segmented datasets, customer-level risk scores, false-positive and false-negative analysis, subgroup performance analysis, clustering evaluation, two serialized machine-learning models, and a final customer retention risk register containing all **7,043 customers**.

The final validation script confirms:

```text
Missing values          0
Duplicate rows          0
Duplicate customer IDs  0
Figures generated       35
Risk-register customers 7,043
Selected model          Random Forest
Test ROC-AUC            0.8414
Final validation        PASSED
```

---

## Conclusion

The analysis shows that customer churn is particularly concentrated among short-tenure customers, month-to-month subscribers, fiber-optic users, electronic-check users, and customers without automatic payment.

The project goes beyond churn prediction by combining supervised learning, unsupervised customer segmentation, threshold optimization, subgroup error analysis, and customer-level retention prioritization.

The resulting system can help a telecommunications business identify customers who require immediate retention attention while separating stable customers from high-value customers who may benefit from loyalty-focused strategies.