"""
Robustness testing: add Gaussian noise and compare performance.
"""

import numpy as np
from sklearn.metrics import accuracy_score
from pathlib import Path

from .utils import REPORTS_DIR, RANDOM_STATE, ensure_dirs


def add_gaussian_noise(X, noise_std=0.01, random_state=RANDOM_STATE):
    """Add Gaussian noise to features."""
    rng = np.random.default_rng(random_state)
    noise = rng.normal(0, noise_std, X.shape)
    return X + noise.astype(X.dtype)


def robustness_test(
    model,
    X_original,
    y_true,
    noise_std=0.01,
    model_name="model",
):
    """
    Evaluate model on original and noisy data.
    Returns dict with accuracy_original, accuracy_noisy, performance_drop_pct.
    """
    y_pred_orig = model.predict(X_original)
    acc_orig = accuracy_score(y_true, y_pred_orig)

    X_noisy = add_gaussian_noise(X_original, noise_std=noise_std)
    y_pred_noisy = model.predict(X_noisy)
    acc_noisy = accuracy_score(y_true, y_pred_noisy)

    drop = (acc_orig - acc_noisy) / acc_orig * 100 if acc_orig > 0 else 0

    return {
        "model_name": model_name,
        "accuracy_original": acc_orig,
        "accuracy_noisy": acc_noisy,
        "performance_drop_pct": drop,
    }


def run_robustness_tests(
    models_dict,
    X_test,
    y_test,
    noise_std=0.01,
    save_report=True,
    reports_dir=None,
):
    """
    Run robustness test for multiple models.
    models_dict: {name: model}
    """
    reports_dir = Path(reports_dir) if reports_dir else REPORTS_DIR
    ensure_dirs()

    results = []
    for name, model in models_dict.items():
        r = robustness_test(model, X_test, y_test, noise_std=noise_std, model_name=name)
        results.append(r)
        print(f"{name}: Orig={r['accuracy_original']:.4f}, Noisy={r['accuracy_noisy']:.4f}, Drop={r['performance_drop_pct']:.2f}%")

    if save_report:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(reports_dir / "robustness_comparison.csv", index=False)

    return results
