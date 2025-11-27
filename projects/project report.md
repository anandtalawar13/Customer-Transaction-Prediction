# **Customer Transaction Prediction**

## **1. Project Overview**

This project focuses on predicting whether a bank customer will perform a transaction in the future. With an extremely **imbalanced dataset** (only ~8% positive cases), the challenge centers on building models that can correctly detect the minority class without being overwhelmed by the majority class.

The project includes:
* Full **Exploratory Data Analysis (EDA)**
* Multiple **Machine Learning Models**
* **Hyperparameter Tuning** using RandomizedSearchCV
* **Imbalance Handling** using class weights / scale_pos_weight
* **Threshold Optimization** for F1-Score maximization
* Final **Model Comparison & Conclusion**

---

## **2. Dataset Summary**
* **ID_code** — unique identifier
* **target** — 0 (no transaction), 1 (transaction)
* **var_0 … var_199** — 200 anonymized continuous features

### **Target Imbalance**
* Class 0: **92%**
* Class 1: **8%**

This imbalance makes the problem ideal for **weighted models** rather than oversampling.

---

## **3. Exploratory Data Analysis (EDA)**

### **Key Insights**
* No missing values were found.
* All features (var_0 to var_199) are continuous numeric values.
* The target variable is heavily imbalanced.
* Due to anonymized features, feature-level interpretation is limited, but variance and outlier checks confirm usable distributions.

---

## **4. Handling Class Imbalance**

Instead of using SMOTE / SMOTEENN (which distorts distributions in high-dimensional data), the project uses:
* `class_weight='balanced'` → Logistic Regression, Random Forest
* `scale_pos_weight` → XGBoost, LightGBM
* `class_weights` → CatBoost

This preserves the **original data distribution** and provides **mathematically consistent weighting** for heavily skewed data.

---

## **5. Models Trained & Tuned**

You trained and tuned the following models:
1. **Logistic Regression**
2. **Random Forest**
3. **XGBoost**
4. **LightGBM**
5. **CatBoost**
6. **Stacking Classifier** (XGB + LGB + CAT → Logistic Regression)

Hyperparameters were optimized using **RandomizedSearchCV**.

---

## **6. Model Evaluation (Before Threshold Optimization)**

| Model               | Train Accuracy | Test Accuracy | Test F1-Score |
| ------------------- | -------------- | ------------- | ------------- |
| Logistic Regression | 0.782          | 0.784         | 0.419         |
| Random Forest       | 0.934          | 0.884         | 0.426         |
| XGBoost             | 0.939          | 0.874         | 0.478         |
| LightGBM            | 0.912          | 0.867         | 0.504         |
| CatBoost            | 0.912          | 0.868         | **0.514**     |
| Stacking Model      | 0.856          | 0.813         | 0.463         |

**CatBoost** is the best model before threshold tuning.

---

## **7. Threshold Tuning Results**

Threshold tuning was applied to maximize **F1-score for class 1**.

| Model               | Best Threshold | Best F1   |
| ------------------- | -------------- | --------- |
| Logistic Regression | 0.730          | 0.493     |
| Random Forest       | 0.494          | 0.435     |
| XGBoost             | 0.528          | 0.484     |
| LightGBM            | 0.592          | 0.531     |
| CatBoost            | 0.610          | 0.543     |
| **Stacking Model**  | **0.804**      | **0.548** |

* **Final Winner: Stacking Classifier** (Best F1 = **0.548**)
* CatBoost close second (F1 = **0.543**)

---

## **8. Visualizations Created**

Notebook includes:
* ROC-AUC Curves for all models
* Precision-Recall Curves
* F1 vs Threshold Curves
* Precision vs Threshold
* Recall vs Threshold
* Bar chart comparing F1 scores

These visuals clearly show the impact of threshold tuning and model comparisons.

---

## **9. Final Conclusion**

This project successfully built a predictive system for customer transaction forecasting using multiple machine learning models. Due to the highly imbalanced nature of the dataset, **class weighting** and **threshold optimization** were essential for improving minority-class performance.

Key takeaways:
* Boosting models (XGBoost, LightGBM, CatBoost) outperformed classical models.
* **CatBoost** provided the best base performance.
* After threshold tuning, the **Stacking Classifier emerged as the best overall model**, achieving:

  * Highest F1-score for minority class (0.548)
  * Strong recall, essential for identifying potential transaction customers
* Traditional oversampling (SMOTE/SMOTEENN) was intentionally avoided to preserve data integrity.

**Final recommended model for deployment: Stacking Classifier with threshold = 0.804**

---