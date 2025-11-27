#  **Customer Transaction Prediction — Machine Learning Project**

<p align="center">
  <img src="https://img.shields.io/badge/ML-Classification-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Imbalanced-Learning-red?style=for-the-badge">
  <img src="https://img.shields.io/badge/Python-3.10+-yellow?style=for-the-badge">
  <img src="https://img.shields.io/badge/Models-LR|RF|XGB|LGBM|CAT|STACKING-green?style=for-the-badge">
</p>

A powerful end-to-end Machine Learning pipeline designed to predict **customer transactions** from a highly **imbalanced financial dataset** (only 8% positive class).
This project demonstrates **advanced modeling**, **class imbalance handling**, **threshold tuning**, and **ensemble learning** to maximize real-world performance.

---

# **Project Highlights**

### Built 6 Machine Learning Models

* Logistic Regression
* Random Forest
* XGBoost
* LightGBM
* CatBoost
* Stacking Ensemble (XGB + LGBM + CAT → Logistic Regression)

### Smart Imbalance Handling

No SMOTE — instead, **model-specific class weights** such as:
* `class_weight='balanced'`
* `scale_pos_weight` (XGB/LGBM)
* `class_weights` (CatBoost)

### Threshold Tuning

Optimized decision thresholds using F1-maximization to significantly improve minority-class performance.

### Comprehensive Visualizations
* F1 vs Threshold
* Precision vs Threshold
* Recall vs Threshold
* Precision–Recall Curves
* ROC Curves
* Model Comparison Charts

---

# **Installation**

### **Clone the repository**

```bash
git clone https://github.com/anandtalawar13/Customer-Transaction-Prediction.git
cd Customer-Transaction-Prediction
```

### **Install dependencies**

```bash
pip install -r requirements.txt
```
---

# **Data Preprocessing**
* No missing values
* Only numerical features
* Feature scaling (StandardScaler)
* Train-test split
* No SMOTE/SMOTEENN — preserves true distribution
* Class imbalance handled using model-weighted strategies

---

# **Model Training Summary**

## **Before Threshold Tuning**

| Model               | Test Accuracy | Test F1-Score |
| ------------------- | ------------- | ------------- |
| Logistic Regression | 0.784         | 0.419         |
| Random Forest       | 0.884         | 0.426         |
| XGBoost             | 0.874         | 0.478         |
| LightGBM            | 0.867         | 0.504         |
| CatBoost            | 0.868         | **0.514**     |
| Stacking Classifier | 0.813         | 0.463         |

---

#  **Threshold Tuning (F1-Maximization)**

| Model                   | Best Threshold | Best F1-Score |
| ----------------------- | -------------- | ------------- |
| Logistic Regression     | 0.730          | 0.493         |
| Random Forest           | 0.494          | 0.435         |
| XGBoost                 | 0.528          | 0.484         |
| LightGBM                | 0.592          | 0.531         |
| CatBoost                | 0.610          | 0.543         |
| **Stacking Classifier** | **0.804**      | **0.548**     |

* **Final Best Model: Stacking Classifier**
*  **Runner-Up: CatBoost**

The stacking model delivers the strongest minority-class performance **after threshold optimization**.

---

# **Final Conclusion**

This project demonstrates a full machine-learning workflow for imbalanced financial prediction.
After evaluating six models and applying threshold optimization, the **Stacking Classifier** achieved the highest minority-class F1-score (**0.548**), outperforming all other models.

Boosting models like CatBoost and LightGBM also delivered strong performance, proving highly effective for noisy, large-feature datasets.
This study highlights the importance of **class weighting**, **proper evaluation metrics**, and **threshold tuning** in real-world classification tasks.

---

#  **Model Saving**

```python
joblib.dump(stack_model, "saved_models/stacking_classifier.pkl")
```

All models are stored inside the **saved_models** directory for deployment.

---
