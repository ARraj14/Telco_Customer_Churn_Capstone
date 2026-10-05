# Telco Customer Churn – End-to-End Data Science Capstone

## Project Overview

This project presents an end-to-end customer churn analysis and retention-prioritization system using the IBM Telco Customer Churn dataset.

The objective is not only to predict whether a customer is likely to churn, but also to:

- understand the major factors associated with churn;
- clean and validate real-world customer data;
- compare multiple supervised machine-learning models;
- generate calibrated churn probabilities;
- optimize a business-oriented retention threshold;
- analyze false positives, false negatives, and subgroup performance;
- discover behavioral customer segments using unsupervised learning;
- independently validate segment risk on unseen customers;
- combine churn probabilities and customer segments into actionable retention priorities;
- score completely new customers through a reusable inference pipeline.

The final system was validated through a dedicated project-validation script, with all major checks passing.

---

## Dataset

The project uses the public **IBM Telco Customer Churn** sample dataset.

IBM describes the Telco Customer Churn sample as data representing a fictional telecommunications company and customer churn behavior.

### Original Dataset

| Item | Value |
|---|---:|
| Customers | 7,043 |
| Original Columns | 21 |
| Churned Customers | 1,869 |
| Retained Customers | 5,174 |
| Overall Churn Rate | 26.54% |
| Duplicate Rows | 0 |
| Duplicate Customer IDs | 0 |

During validation, `TotalCharges` initially appeared to contain no standard missing values because the column was stored as text.

After converting the column to numeric format, **11 blank values** were discovered.

All 11 records belonged to customers with:

```text
tenure = 0
```

These customers had not yet accumulated historical charges. Their `TotalCharges` values were therefore assigned:

```text
0.0
```

instead of deleting valid customer records.

The final cleaned dataset contains:

```text
7,043 rows
27 columns
0 missing values
0 duplicate rows
0 duplicate customer IDs
```

---

## Feature Engineering

Several additional variables were created to improve analysis, segmentation, and predictive modeling.

### ChurnValue

Converts the churn target into binary numerical form:

```text
No  -> 0
Yes -> 1
```

### TenureGroup

Groups customers into tenure ranges:

```text
0-12 months
13-24 months
25-36 months
37-48 months
49-60 months
61-72 months
```

### NumServices

Counts the number of active customer services.

### HasInternetService

Indicates whether the customer has an internet service.

### ContractValue

Represents increasing contract commitment:

```text
Month-to-month -> 0
One year       -> 1
Two year       -> 2
```

### AutomaticPayment

Identifies customers using automatic bank transfer or automatic credit-card payment.

---

## Exploratory Data Analysis

The project contains **15 dedicated EDA visualizations** and numerous supporting numerical tables.

Important churn patterns include:

| Customer Characteristic | Churn Rate |
|---|---:|
| Overall | 26.54% |
| 0–12 month tenure | 47.44% |
| 61–72 month tenure | 6.61% |
| Month-to-month contract | 42.71% |
| One-year contract | 11.27% |
| Two-year contract | 2.83% |
| Fiber optic internet | 41.89% |
| Electronic check | 45.29% |
| Automatic payment | 15.98% |
| Manual payment | 34.67% |
| Senior citizens | 41.68% |

Customers in their first 12 months showed approximately **7.2 times** the churn rate of customers with 61–72 months of tenure.

### Multivariable Risk Analysis

The highest-risk combination identified during exploratory analysis was:

```text
Month-to-month contract
+ Fiber optic internet
+ Electronic check
```

This group contained:

```text
1,307 customers
789 churners
60.37% churn rate
```

This demonstrates that combining customer characteristics can reveal substantially stronger risk patterns than analyzing variables independently.

---

## Correlation Analysis

The strongest numerical and engineered correlations with churn were:

| Feature | Correlation with Churn |
|---|---:|
| ContractValue | -0.397 |
| tenure | -0.352 |
| HasInternetService | +0.228 |
| AutomaticPayment | -0.210 |
| TotalCharges | -0.198 |
| MonthlyCharges | +0.193 |
| SeniorCitizen | +0.151 |

Important relationships among predictors also included:

```text
tenure ↔ TotalCharges          0.826
MonthlyCharges ↔ NumServices  0.802
TotalCharges ↔ NumServices    0.796
```

These relationships are interpreted as **associations rather than causal effects**.

For example, longer contracts are strongly associated with lower churn, but the analysis does not prove that changing a customer's contract alone would causally prevent churn.

---

# Supervised Machine Learning

## Train / Holdout Design

The dataset was divided using a stratified split:

```text
Training customers: 5,634
Holdout customers:  1,409
```

The churn rate was preserved at approximately **26.54%** in both partitions.

The holdout dataset was excluded from model-family and hyperparameter selection.

---

## Model Comparison

Three supervised classifiers were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest

Hyperparameter optimization used:

```text
5-fold Stratified Cross-Validation
GridSearchCV
ROC-AUC scoring
```

### Training-Only Model Selection

| Model | GridSearch CV ROC-AUC | Training OOF ROC-AUC |
|---|---:|---:|
| Logistic Regression | **0.8464** | **0.8458** |
| Random Forest | 0.8462 | 0.8455 |
| Decision Tree | 0.8298 | 0.8286 |

The final model family was selected **only from training cross-validation results**.

### Selected Base Model

**Logistic Regression**

Best hyperparameter:

```text
C = 10.0
class_weight = balanced
```

The model achieved the highest GridSearch CV ROC-AUC:

```text
0.8464
```

Only after the model was locked was it evaluated on the independent holdout dataset.

### Independent Holdout Performance

```text
ROC-AUC            0.8407
Accuracy           0.7388
Balanced Accuracy  0.7531
Precision          0.5052
Recall             0.7834
F1                 0.6143
```

The close CV and holdout ROC-AUC values indicate stable discrimination on unseen customers.

---

# Probability Calibration

A classification model can rank customers correctly while still producing poorly calibrated probability estimates.

Because the selected Logistic Regression used class balancing, probability calibration was explicitly evaluated.

Two probability models were compared using **training-only out-of-fold predictions**.

| Probability Model | ROC-AUC | Brier Score | Log Loss | Mean Predicted Risk |
|---|---:|---:|---:|---:|
| Raw classifier | 0.84580 | 0.16524 | 0.49049 | 41.37% |
| Sigmoid calibrated | 0.84579 | **0.13519** | **0.41660** | 26.60% |

Actual training churn rate:

```text
26.54%
```

Sigmoid calibration substantially improved probability reliability while preserving model discrimination.

### Selected Probability Model

```text
Sigmoid-calibrated Logistic Regression
```

### Independent Holdout Calibration

```text
ROC-AUC                 0.8407
Brier Score             0.1387
Log Loss                0.4217
Mean Predicted Risk     26.80%
Actual Churn Rate       26.54%
```

The very close agreement between average predicted risk and observed churn provides evidence that the calibrated probabilities are meaningful for risk stratification.

---

# Classification Threshold Optimization

Customer retention places greater importance on identifying likely churners than a standard classification task.

Therefore, the default `0.50` threshold was compared against alternative thresholds.

Threshold selection was performed using **training-only out-of-fold predictions**.

The optimization metric was:

```text
F2 Score
```

F2 gives more weight to recall than precision.

### Selected Retention Threshold

```text
0.16
```

### Training OOF Performance at 0.16

```text
Accuracy            0.6709
Balanced Accuracy   0.7474
Precision           0.4417
Recall              0.9104
F1                  0.5948
F2                  0.7510
```

### Independent Holdout Performance at 0.16

```text
Precision  0.4452
Recall     0.9118
F1         0.5982
F2         0.7538
```

Confusion matrix:

```text
                       Predicted
                    No          Churn

Actual No           610           425
Actual Churn         33           341
```

The optimized retention policy detected:

```text
341 of 374 holdout churners
```

and missed only:

```text
33 churners
```

The lower threshold intentionally trades precision for higher recall and is therefore appropriate when the cost of retention outreach is lower than the cost of losing an undetected churner.

The `0.16` threshold is **not universally superior** to `0.50`; businesses with strict outreach budgets may prefer a higher threshold.

---

# Error and Subgroup Analysis

Performance was evaluated across several customer groups.

Examples of holdout recall at the retention threshold include:

### Contract

| Contract | Recall |
|---|---:|
| Month-to-month | 97.26% |
| One year | 58.33% |
| Two year | 0.00% |

The very low number of churners within the two-year holdout group makes its subgroup recall unstable and highlights an important limitation of subgroup metrics with small positive-class counts.

### Internet Service

| Service | Recall |
|---|---:|
| Fiber optic | 95.24% |
| DSL | 85.57% |
| No internet | 72.00% |

### Senior Citizen

| Group | Recall |
|---|---:|
| Non-senior | 88.77% |
| Senior | 97.96% |

### Gender

| Gender | Recall |
|---|---:|
| Female | 89.64% |
| Male | 92.82% |

These results make model weaknesses visible instead of relying only on aggregate performance.

---

# Customer Segmentation

K-Means clustering was used to identify behavioral customer groups.

The clustering inputs did **not** contain:

```text
Churn
ChurnValue
```

Therefore, the target was not used to form the clusters.

## Leakage-Safe Clustering Design

The same training/holdout partition used for supervised modeling was retained.

Clustering preprocessing, `K` selection, PCA, and final K-Means fitting were performed using **training customers only**.

The frozen clustering pipeline was then applied to the unseen holdout customers.

---

## Choosing K

Values from:

```text
K = 2 through K = 8
```

were evaluated using:

- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score
- Elbow Method

The strongest training solution was:

```text
K = 3

Silhouette Score       0.2995
Davies-Bouldin         1.2698
Calinski-Harabasz      2675.17
```

The silhouette score indicates **moderate rather than perfect cluster separation**, so the segments are interpreted as useful behavioral groupings rather than naturally isolated populations.

---

## Cluster Stability

The K=3 solution was refitted using several random seeds.

Adjusted Rand Index:

```text
Seed 7     1.0000
Seed 21    1.0000
Seed 42    1.0000
Seed 99    1.0000
Seed 123   1.0000
```

Mean stability:

```text
ARI = 1.0000
```

This indicates that the selected clustering solution is highly stable across the tested initializations.

---

## PCA Visualization

PCA was used only for **two-dimensional visualization**, not for creating the clusters.

Training PCA explained variance:

```text
PC1    43.83%
PC2    20.54%
Total  64.37%
```

---

# Training-Defined Business Segments

Business segment names were derived from **training customer profiles only**.

The labels were then frozen before holdout churn outcomes were analyzed.

The final mapping is:

```text
Cluster 0 -> Low-Cost Stable Customers
Cluster 1 -> High-Risk Short-Tenure Customers
Cluster 2 -> High-Spend Loyal Customers
```

The name **High-Spend Loyal Customers** is used instead of "High-Value" because the dataset contains customer charges but does not contain customer profitability or lifetime economic value.

---

## Segment Profiles

### Full Descriptive Population

| Segment | Customers | Avg Tenure | Avg Monthly Charges | Avg Services | Churn Rate |
|---|---:|---:|---:|---:|---:|
| Low-Cost Stable | 1,549 | 30.75 | 21.29 | 1.24 | 7.36% |
| High-Risk Short-Tenure | 3,185 | 15.27 | 68.26 | 2.79 | 44.52% |
| High-Spend Loyal | 2,309 | 57.06 | 89.10 | 5.58 | 14.60% |

---

# Independent Segment Validation

The segment labels were defined from training data before holdout outcomes were examined.

### Training vs Holdout Churn

| Segment | Training Churn | Holdout Churn |
|---|---:|---:|
| Low-Cost Stable | 7.22% | **7.89%** |
| High-Risk Short-Tenure | 44.67% | **43.95%** |
| High-Spend Loyal | 14.95% | **13.00%** |

The segment behavior remained highly consistent on unseen customers.

On the independent holdout set, the High-Risk Short-Tenure segment had:

```text
43.95% churn
1.66x overall holdout churn risk
5.57x the churn rate of the Low-Cost Stable segment
```

This provides out-of-sample evidence that the behavioral segmentation captures meaningful differences in churn risk.

---

# Final Customer Risk and Retention System

The final policy combines:

```text
Calibrated churn probability
+
Validated 0.16 retention threshold
+
Training-defined customer segment
```

The decision policy was defined without using holdout outcomes.

It was then evaluated on the independent holdout dataset.

---

## Independent Holdout Risk Bands

| Risk Band | Customers | Actual Churn Rate | Avg Predicted Risk |
|---|---:|---:|---:|
| High Risk | 315 | **66.03%** | 63.69% |
| Retention Candidate | 451 | **29.49%** | 31.75% |
| Below Retention Threshold | 643 | **5.13%** | 5.25% |

The observed churn rate decreases consistently with the model-generated risk bands.

---

## Independent Holdout Retention Priorities

| Priority | Customers | Actual Churn Rate | Avg Predicted Risk |
|---|---:|---:|---:|
| Critical | 299 | **67.56%** | 64.09% |
| High | 289 | **32.18%** | 35.98% |
| Medium | 275 | **18.55%** | 20.52% |
| Low | 546 | **5.13%** | 4.68% |

This monotonic decline:

```text
Critical -> High -> Medium -> Low

67.56% -> 32.18% -> 18.55% -> 5.13%
```

provides independent evidence that the final combined prioritization system successfully concentrates churn risk.

---

# Full Customer Risk Register

After independent validation was complete, cross-fitted calibrated probabilities were generated for all **7,043 customers** to produce the analytical risk register.

Full cross-fitted performance:

```text
ROC-AUC      0.8450
Brier Score  0.1357
Log Loss     0.4175
```

### Full Risk Bands

| Risk Band | Customers | Actual Churn Rate | Avg Predicted Risk |
|---|---:|---:|---:|
| High Risk | 1,550 | 66.06% | 63.68% |
| Retention Candidate | 2,309 | 29.36% | 30.97% |
| Below Retention Threshold | 3,184 | 5.24% | 5.25% |

### Full Retention Priorities

| Priority | Customers | Actual Churn Rate | Avg Predicted Risk |
|---|---:|---:|---:|
| Critical | 1,448 | 66.64% | 64.14% |
| High | 1,423 | 33.59% | 35.88% |
| Medium | 1,404 | 20.87% | 21.41% |
| Low | 2,768 | 4.80% | 4.68% |

The full risk register is intended for analytical prioritization, while the **independent holdout results remain the primary evidence of generalization**.

---

# Business Recommendations

## High-Risk Short-Tenure Customers

This segment should receive the strongest retention attention.

Recommended actions include:

- improved early-tenure onboarding;
- proactive customer support;
- service-quality monitoring;
- incentives for moving away from month-to-month contracts;
- automatic-payment incentives;
- personalized retention offers.

## High-Spend Loyal Customers

These customers have long tenure, high monthly charges, and relatively low churn.

Recommended actions include:

- loyalty benefits;
- priority support;
- proactive service-quality monitoring;
- personalized offers;
- avoiding unnecessary interventions that could reduce satisfaction.

## Low-Cost Stable Customers

These customers show very low churn and relatively low service adoption.

Recommended actions include:

- maintaining affordability;
- limiting unnecessary retention expenditure;
- offering careful and relevant cross-sell opportunities.

---

# New-Customer Inference System

The project includes a reusable inference script capable of assessing a completely new customer.

The pipeline automatically performs:

```text
Raw customer information
        |
        v
Feature engineering
        |
        v
Calibrated Logistic Regression
        |
        v
Churn probability
        |
        v
Frozen K-Means segmentation pipeline
        |
        v
Business segment
        |
        v
Validated retention threshold
        |
        v
Risk band
        |
        v
Retention priority
        |
        v
Recommended retention action
```

Example demonstration result:

```text
Calibrated churn probability: 75.50%

Cluster:
1

Segment:
High-Risk Short-Tenure Customers

Risk Band:
High Risk

Retention Priority:
Critical
```

---

# Reproducibility / Quick Start

## 1. Clone the Repository

```powershell
git clone https://github.com/ARraj14/Telco_Customer_Churn_Capstone.git
cd Telco_Customer_Churn_Capstone
```

## 2. Install Dependencies

The project uses `uv`.

```powershell
uv sync
```

## 3. Run the Complete Analysis

Run the scripts in this order:

```powershell
uv run python src\01_validate_data.py
uv run python src\02_clean_data.py
uv run python src\03_eda.py
uv run python src\04_train_models.py
uv run python src\05_threshold_error_analysis.py
uv run python src\06_customer_segmentation.py
uv run python src\07_segment_business_analysis.py
uv run python src\08_final_risk_scoring.py
uv run python src\09_final_validation.py
```

## 4. Test New-Customer Prediction

```powershell
uv run python src\10_predict_customer.py --demo
```

A customer stored in a JSON file can also be assessed using:

```powershell
uv run python src\10_predict_customer.py --json path\to\customer.json
```

---

# Project Workflow

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
Stratified Train / Holdout Split
      |
      v
Training-Only Model Selection
      |
      v
Logistic Regression
      |
      v
Probability Calibration
      |
      v
Training-OOF Threshold Optimization
      |
      v
Independent Holdout Evaluation
      |
      v
Training-Fitted K-Means Segmentation
      |
      v
Cluster Stability Analysis
      |
      v
Frozen Business Segment Mapping
      |
      v
Independent Segment Validation
      |
      v
Combined Risk + Segment Policy
      |
      v
Independent Priority Validation
      |
      v
Cross-Fitted Full Customer Risk Register
      |
      v
Final Deployment Model
      |
      v
New-Customer Inference
```

---

# Project Structure

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
│   ├── retention_churn_model.joblib
│   ├── customer_segmentation_pipeline.joblib
│   └── final_churn_model_full.joblib
|
├── outputs/
│   ├── figures/
│   │   └── 36 analytical figures
│   └── tables/
│       └── validation, EDA, calibration, modeling,
│           clustering, holdout and risk-analysis outputs
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
│   ├── 09_final_validation.py
│   └── 10_predict_customer.py
|
├── README.md
├── pyproject.toml
├── uv.lock
└── .python-version
```

---

# Technologies Used

```text
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Joblib
uv
Git
GitHub
```

---

# Key Deliverables

The completed project includes:

```text
7,043 customer records
36 analytical figures
Detailed EDA and supporting tables
Training-only model-family selection
Independent holdout evaluation
Probability calibration analysis
F2-based retention threshold optimization
False-positive / false-negative analysis
Subgroup performance analysis
Training-fitted K-Means clustering
Cluster stability testing
PCA visualization
Independent segment validation
Independent retention-policy validation
Customer-level risk register
4 serialized model artifacts
Reusable new-customer inference script
Automated final validation
```

---

# Final Validation

The final automated project validation confirms:

```text
Cleaned customers              7,043
Missing values                     0
Duplicate rows                      0
Duplicate customer IDs             0

Selected model            Logistic Regression
Training CV ROC-AUC                 0.8464
Independent ROC-AUC                 0.8407
Independent Brier Score             0.1387
Independent retention recall        0.9118

Selected K                               3
Mean cluster stability ARI          1.0000

Figures generated                         36
Risk-register customers                7,043
New-customer inference                  PASS

Validation checks                     26/26
Final validation                      PASSED
```

---

# Challenges and Solutions

## Hidden Missing Values in TotalCharges

**Challenge:** Missing values were stored as blank strings rather than standard null values.

**Solution:** The column was converted to numeric format, exposing 11 invalid entries. All belonged to zero-tenure customers and were logically assigned zero historical charges.

## Class Imbalance

**Challenge:** Only 26.54% of customers churned.

**Solution:** Stratified sampling, balanced class weighting, ROC-AUC, balanced accuracy, precision, recall, F1, and F2 were used instead of relying on accuracy alone.

## Test-Set Contamination Risk

**Challenge:** Comparing model families using the final test set would weaken the independence of the reported test result.

**Solution:** Final model selection was changed to use training-only cross-validation. The holdout dataset is evaluated only after the model family and hyperparameters are locked.

## Probability Reliability

**Challenge:** Raw class-balanced probabilities substantially overestimated absolute churn risk.

**Solution:** Probability calibration was explicitly evaluated. Sigmoid calibration reduced training OOF Brier Score from `0.16524` to `0.13519`.

## Business-Oriented Threshold

**Challenge:** The standard 0.50 threshold missed too many potential churners.

**Solution:** The threshold was optimized using training-only out-of-fold F2 score, producing a retention threshold of `0.16` and holdout recall of `91.18%`.

## Circular Segment Validation

**Challenge:** Naming and evaluating segments on the same outcomes could exaggerate business conclusions.

**Solution:** K-Means fitting and business segment naming were performed using training customers only. Segment churn patterns were then independently confirmed on holdout customers.

## Reusable Deployment

**Challenge:** Analysis outputs alone could not score a completely new customer end-to-end.

**Solution:** The clustering preprocessing pipeline, K-Means model, calibrated churn model, threshold, and business mapping are serialized and used by `10_predict_customer.py`.

---

# Limitations

This analysis should be interpreted within several limitations.

The dataset represents a static historical sample rather than a temporal production stream. Therefore, model performance may change as customer behavior or business conditions evolve.

Although probability calibration substantially improved reliability, calibration should be monitored on future production data.

The 0.16 retention threshold is optimized for a recall-oriented F2 objective. It is not automatically the economically optimal threshold because actual campaign costs, customer lifetime value, offer costs, and retention success rates are unavailable.

K-Means uses Euclidean distance after scaling numerical variables and one-hot encoding categorical variables. This is a practical clustering baseline, but mixed-data methods such as K-Prototypes or Gower-distance clustering could also be investigated.

Observed relationships should not be interpreted as causal effects. For example, customers with longer contracts churn less frequently, but the project does not establish that forcing a customer into a longer contract would itself cause retention.

Some subgroup metrics are based on small numbers of churners and can therefore be unstable.

---

# Future Improvements

Possible future extensions include:

```text
Temporal or rolling validation
Production calibration monitoring
Retention campaign A/B testing
Customer lifetime-value integration
Contact-cost and intervention-cost optimization
Permutation or SHAP-based model interpretation
Alternative mixed-data clustering methods
Automated API or web-interface deployment
Model-drift monitoring
Additional behavioral and service-quality variables
```

The most important production extension would be to evaluate whether customers identified by the system actually respond positively to specific retention interventions.

---

# Conclusion

The project demonstrates a complete churn analytics workflow extending beyond simple classification.

The final system combines:

```text
data validation
data cleaning
feature engineering
exploratory analysis
supervised learning
training-only model selection
probability calibration
threshold optimization
error analysis
customer segmentation
cluster stability testing
independent segment validation
independent retention-priority validation
customer-level risk scoring
new-customer inference
```

The final independently validated retention priorities show:

```text
Critical   67.56% churn
High       32.18% churn
Medium     18.55% churn
Low         5.13% churn
```

This separation indicates that the final system can meaningfully rank and prioritize customers according to churn risk while maintaining a clear distinction between predictive modeling, customer segmentation, and business decision logic.

The result is a reproducible end-to-end customer churn and retention-prioritization framework rather than only a standalone machine-learning model.