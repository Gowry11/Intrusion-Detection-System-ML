"""
Explainability using SHAP: summary plot, feature importance, force plot.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .utils import PLOTS_DIR, ensure_dirs


def shap_explain(
    model,
    X,
    feature_names,
    class_names=None,
    model_name="model",
    max_samples=500,
    save_dir=None,
):
    """
    Generate SHAP explanations for a tree-based or compatible model.
    """
    save_dir = Path(save_dir) if save_dir else PLOTS_DIR
    ensure_dirs()

    try:
        import shap
    except ImportError:
        print("SHAP not installed. Run: pip install shap")
        return None

    # Limit samples for speed
    n_samples = min(max_samples, X.shape[0])
    X_sample = X[:n_samples] if X.shape[0] > n_samples else X

    try:
        explainer = shap.TreeExplainer(model, X_sample)
    except Exception:
        try:
            explainer = shap.KernelExplainer(model.predict_proba, X_sample[:100])
        except Exception as e:
            print(f"SHAP explainer failed for {model_name}: {e}")
            return None

    shap_values = explainer.shap_values(X_sample)

    # Handle multi-output (multi-class) vs single output
    if isinstance(shap_values, list):
        # Multi-class: use class 0 or mean across classes for summary
        shap_sum = np.mean(np.abs(shap_values), axis=0)
    else:
        shap_sum = np.abs(shap_values)

    # Summary plot
    try:
        plt.figure(figsize=(12, 8))
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[0], X_sample, feature_names=feature_names, show=False)
        else:
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.title(f"SHAP Summary - {model_name}")
        plt.tight_layout()
        plt.savefig(save_dir / f"shap_summary_{model_name}.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"SHAP summary plot failed: {e}")

    # Feature importance (mean |SHAP|)
    mean_abs_shap = np.mean(shap_sum, axis=0)
    if mean_abs_shap.ndim > 1:
        mean_abs_shap = np.mean(mean_abs_shap, axis=-1)
    feature_names = list(feature_names)
    imp_order = np.argsort(mean_abs_shap)[::-1].flatten()
    top_k = min(15, len(feature_names))
    top_features = [feature_names[int(i)] for i in imp_order[:top_k]]
    top_importance = mean_abs_shap[imp_order[:top_k]]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_k), top_importance[::-1])
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(top_features[::-1])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Feature Importance (SHAP) - {model_name}")
    plt.tight_layout()
    plt.savefig(save_dir / f"shap_feature_importance_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Top 10 features
    top10 = list(zip(top_features[:10], top_importance[:10]))
    print(f"\nTop 10 Important Features ({model_name}):")
    for f, v in top10:
        print(f"  {f}: {v:.4f}")

    # Force plot for one sample (binary or first class)
    try:
        if isinstance(shap_values, list):
            sv = shap_values[0]
        else:
            sv = shap_values
        shap.force_plot(
            explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
            sv[0],
            X_sample[0],
            feature_names=feature_names,
            matplotlib=True,
            show=False,
        )
        plt.title(f"SHAP Force Plot (Sample 0) - {model_name}")
        plt.tight_layout()
        plt.savefig(save_dir / f"shap_force_{model_name}.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"SHAP force plot failed: {e}")

    return {"top_features": top10, "mean_abs_shap": mean_abs_shap}
