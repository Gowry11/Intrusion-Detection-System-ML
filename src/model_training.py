"""
Model training: Logistic Regression, Random Forest, XGBoost, LightGBM, SVM.
Class imbalance handled via SMOTE (training set only).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path

from .utils import MODELS_DIR, RANDOM_STATE, ensure_dirs


def print_class_distribution(y, prefix=""):
    """Print class distribution."""
    unique, counts = np.unique(y, return_counts=True)
    print(f"{prefix}Class distribution:")
    for u, c in zip(unique, counts):
        print(f"  Class {u}: {c} ({100 * c / len(y):.1f}%)")


def apply_smote(X_train, y_train, random_state=RANDOM_STATE, k_neighbors=5):
    """
    Apply SMOTE only to training data.
    """
    print_class_distribution(y_train, "Before SMOTE - ")
    min_class = np.bincount(y_train).min()
    k = min(k_neighbors, min_class - 1) if min_class > 1 else 1
    smote = SMOTE(random_state=random_state, k_neighbors=max(1, k))
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print_class_distribution(y_resampled, "After SMOTE - ")
    return X_resampled, y_resampled, smote


def train_and_evaluate(
    model,
    name,
    X_train,
    y_train,
    X_test,
    y_test,
    class_names,
    binary_task=False,
):
    """Train model and compute metrics."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm.tolist(),
    }

    if binary_task:
        try:
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = model.decision_function(X_test)
            metrics["roc_auc"] = roc_auc_score(y_test, y_prob, average="macro")
            metrics["roc_curve"] = roc_curve(y_test, y_prob)
        except Exception:
            metrics["roc_auc"] = None
            metrics["roc_curve"] = None
    else:
        try:
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)
                metrics["roc_auc"] = roc_auc_score(
                    y_test, y_prob, multi_class="ovr", average="macro"
                )
            else:
                metrics["roc_auc"] = None
        except Exception:
            metrics["roc_auc"] = None

    return model, metrics, y_pred


def train_all_models(
    X_train,
    y_train,
    X_test,
    y_test,
    class_names,
    task="multiclass",
    use_smote=True,
    save_models=True,
    models_dir=None,
):
    """
    Train Logistic Regression, Random Forest, XGBoost, LightGBM, SVM.
    Apply SMOTE to training data if use_smote=True.
    """
    models_dir = Path(models_dir) if models_dir else MODELS_DIR
    ensure_dirs()

    binary_task = task == "binary"

    if use_smote:
        X_tr, y_tr, _ = apply_smote(X_train, y_train)
    else:
        X_tr, y_tr = X_train, y_train

    models_config = [
        ("logistic_regression", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ("random_forest", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
        ("xgboost", None),  # Will use xgboost.XGBClassifier
        ("lightgbm", None),
        ("svm", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
    ]

    try:
        import xgboost as xgb
        models_config[2] = ("xgboost", xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE))
    except ImportError:
        models_config[2] = ("xgboost", None)

    try:
        import lightgbm as lgb
        models_config[3] = ("lightgbm", lgb.LGBMClassifier(verbose=-1, random_state=RANDOM_STATE))
    except ImportError:
        models_config[3] = ("lightgbm", None)

    results = []
    trained_models = {}

    for name, model in models_config:
        if model is None:
            continue
        print(f"\n--- Training {name} ---")
        model, metrics, y_pred = train_and_evaluate(
            model, name, X_tr, y_tr, X_test, y_test, class_names, binary_task
        )
        metrics["model_name"] = name
        results.append(metrics)
        trained_models[name] = model

        if save_models:
            joblib.dump(model, models_dir / f"{name}.joblib")

    return {
        "results": results,
        "models": trained_models,
        "class_names": class_names,
    }
