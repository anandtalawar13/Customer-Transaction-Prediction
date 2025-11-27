import os
import joblib
from preprocessing import preprocess_dataset

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier


def compute_scale_pos_weight(y):
    pos = sum(y == 1)
    neg = sum(y == 0)
    return neg / pos


def train_all_models(data_path, save_dir="saved_models"):

    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_dataset(data_path)

    # Class imbalance handling
    scale_pos_weight = compute_scale_pos_weight(y_train)

    # Models
    model_lr = LogisticRegression(class_weight='balanced', max_iter=3000)
    model_rf = RandomForestClassifier(class_weight='balanced', n_estimators=150)
    model_xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        n_estimators=200
    )
    model_lgb = LGBMClassifier(scale_pos_weight=scale_pos_weight, n_estimators=200)
    model_cat = CatBoostClassifier(
        verbose=0,
        loss_function="Logloss",
        class_weights=[1, scale_pos_weight],
        iterations=300
    )

    # Stacking model
    final_estimator = LogisticRegression(class_weight="balanced", max_iter=2000)
    stack_model = StackingClassifier(
        estimators=[
            ("xgb", model_xgb),
            ("lgb", model_lgb),
            ("cat", model_cat)
        ],
        final_estimator=final_estimator,
        stack_method="predict_proba"
    )

    # Train models
    model_lr.fit(X_train, y_train)
    model_rf.fit(X_train, y_train)
    model_xgb.fit(X_train, y_train)
    model_lgb.fit(X_train, y_train)
    model_cat.fit(X_train, y_train)
    stack_model.fit(X_train, y_train)

    # Save directory
    os.makedirs(save_dir, exist_ok=True)

    # Save models
    joblib.dump(scaler, f"{save_dir}/scaler.pkl")
    joblib.dump(model_lr, f"{save_dir}/logistic_regression.pkl")
    joblib.dump(model_rf, f"{save_dir}/random_forest.pkl")
    joblib.dump(model_xgb, f"{save_dir}/xgboost.pkl")
    joblib.dump(model_lgb, f"{save_dir}/lightgbm.pkl")
    joblib.dump(model_cat, f"{save_dir}/catboost.pkl")
    joblib.dump(stack_model, f"{save_dir}/stacking_classifier.pkl")

    print("All models trained and saved successfully!")

    return X_train, X_test, y_train, y_test
