"""
Hybrid Stacking Model: Tuned Random Forest + Tuned XGBoost -> Meta Logistic Regression.
"""

import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib
from pathlib import Path

from .utils import MODELS_DIR, RANDOM_STATE, ensure_dirs


def build_stacking_model(tuned_rf, tuned_xgb):
    """
    Build StackingClassifier with:
    - Base: Tuned Random Forest, Tuned XGBoost
    - Meta: Logistic Regression
    """
    estimators = [
        ("rf", tuned_rf),
        ("xgb", tuned_xgb),
    ]
    meta = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=meta,
        cv=5,
        stack_method="predict_proba",
    )
    return stack


def train_and_evaluate_stacking(
    stacking_model,
    X_train,
    y_train,
    X_test,
    y_test,
    class_names,
    save_model=True,
    models_dir=None,
):
    """Train stacking model and evaluate."""
    models_dir = Path(models_dir) if models_dir else MODELS_DIR
    ensure_dirs()

    print("\n--- Training Stacking Model ---")
    stacking_model.fit(X_train, y_train)
    y_pred = stacking_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)

    print(f"Stacking Accuracy: {acc:.4f}")
    print(f"Stacking F1 (weighted): {f1:.4f}")
    print("\nClassification Report:\n", report)

    result = {
        "model": stacking_model,
        "accuracy": acc,
        "f1_score": f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "y_pred": y_pred,
    }

    if save_model:
        joblib.dump(stacking_model, models_dir / "stacking_model.joblib")

    return result


def compare_with_individuals(stacking_result, individual_results, class_names):
    """Compare stacking model with individual base models."""
    print("\n--- Model Comparison ---")
    rows = []
    for r in individual_results:
        rows.append((r["model_name"], r["accuracy"], r.get("f1_score", 0)))
    rows.append(("Stacking", stacking_result["accuracy"], stacking_result["f1_score"]))
    for name, acc, f1 in rows:
        print(f"  {name}: Accuracy={acc:.4f}, F1={f1:.4f}")
    return rows
