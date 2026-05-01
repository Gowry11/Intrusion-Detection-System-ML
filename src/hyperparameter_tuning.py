"""
Hyperparameter tuning using RandomizedSearchCV.
Tunes Random Forest, XGBoost, LightGBM.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, make_scorer
import joblib
from pathlib import Path

from .utils import MODELS_DIR, RANDOM_STATE, ensure_dirs


# Parameter grids for RandomizedSearchCV
RF_PARAMS = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [5, 10, 15, 20, 25, None],
    "min_samples_split": [2, 5, 10, 15],
}

XGB_PARAMS = {
    "n_estimators": [100, 200, 300, 500],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7, 9, 11],
}

LGB_PARAMS = {
    "n_estimators": [100, 200, 300, 500],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7, 9, 11],
}


def tune_model(
    model,
    param_distributions,
    X_train,
    y_train,
    n_iter=20,
    cv=3,
    random_state=RANDOM_STATE,
):
    """Run RandomizedSearchCV and return best estimator."""
    search = RandomizedSearchCV(
        model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="accuracy",
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


def tune_all_models(
    X_train,
    y_train,
    X_test,
    y_test,
    n_iter=15,
    cv=3,
    save_models=True,
    models_dir=None,
):
    """
    Tune Random Forest, XGBoost, LightGBM.
    Returns dict of best models and their scores.
    """
    models_dir = Path(models_dir) if models_dir else MODELS_DIR
    ensure_dirs()

    tuned = {}

    # Random Forest
    print("\n--- Tuning Random Forest ---")
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    best_rf, params_rf, score_rf = tune_model(rf, RF_PARAMS, X_train, y_train, n_iter=n_iter, cv=cv)
    acc_rf = accuracy_score(y_test, best_rf.predict(X_test))
    tuned["random_forest"] = {"model": best_rf, "params": params_rf, "cv_score": score_rf, "test_accuracy": acc_rf}
    if save_models:
        joblib.dump(best_rf, models_dir / "random_forest_tuned.joblib")

    # XGBoost
    try:
        import xgboost as xgb
        print("\n--- Tuning XGBoost ---")
        xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE)
        best_xgb, params_xgb, score_xgb = tune_model(xgb_clf, XGB_PARAMS, X_train, y_train, n_iter=n_iter, cv=cv)
        acc_xgb = accuracy_score(y_test, best_xgb.predict(X_test))
        tuned["xgboost"] = {"model": best_xgb, "params": params_xgb, "cv_score": score_xgb, "test_accuracy": acc_xgb}
        if save_models:
            joblib.dump(best_xgb, models_dir / "xgboost_tuned.joblib")
    except ImportError:
        print("XGBoost not available, skipping tuning.")

    # LightGBM
    try:
        import lightgbm as lgb
        print("\n--- Tuning LightGBM ---")
        lgb_clf = lgb.LGBMClassifier(verbose=-1, random_state=RANDOM_STATE)
        best_lgb, params_lgb, score_lgb = tune_model(lgb_clf, LGB_PARAMS, X_train, y_train, n_iter=n_iter, cv=cv)
        acc_lgb = accuracy_score(y_test, best_lgb.predict(X_test))
        tuned["lightgbm"] = {"model": best_lgb, "params": params_lgb, "cv_score": score_lgb, "test_accuracy": acc_lgb}
        if save_models:
            joblib.dump(best_lgb, models_dir / "lightgbm_tuned.joblib")
    except ImportError:
        print("LightGBM not available, skipping tuning.")

    return tuned
