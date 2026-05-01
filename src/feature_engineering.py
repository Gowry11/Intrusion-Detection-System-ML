"""
Feature engineering: variance filter, correlation removal, SelectKBest.
"""

from functools import partial
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif

from .utils import RANDOM_STATE


def remove_low_variance(X, feature_names, threshold=0.01):
    """
    Remove features with variance below threshold.
    """
    selector = VarianceThreshold(threshold=threshold)
    X_selected = selector.fit_transform(X)
    kept = selector.get_support()
    selected_features = [f for f, k in zip(feature_names, kept) if k]
    return X_selected, selected_features, selector


def remove_correlated_features(X, feature_names, correlation_threshold=0.95):
    """
    Remove highly correlated features, keeping one from each correlated pair.
    """
    n_features = X.shape[1]
    if n_features <= 1:
        return X, feature_names

    corr_matrix = np.corrcoef(X.T)
    np.fill_diagonal(corr_matrix, 0)

    to_drop = set()
    for i in range(n_features):
        for j in range(i + 1, n_features):
            if abs(corr_matrix[i, j]) >= correlation_threshold:
                # Drop the feature with higher index (arbitrary)
                to_drop.add(j)

    keep_idx = [i for i in range(n_features) if i not in to_drop]
    X_selected = X[:, keep_idx]
    selected_features = [feature_names[i] for i in keep_idx]
    return X_selected, selected_features


def select_k_best(X, y, feature_names, k=20, random_state=RANDOM_STATE):
    """
    Select top k features using mutual information.
    """
    n_features = min(k, X.shape[1])
    score_func = partial(mutual_info_classif, random_state=random_state)
    selector = SelectKBest(score_func=score_func, k=n_features)
    X_selected = selector.fit_transform(X, y)
    scores = selector.scores_
    kept = selector.get_support()
    selected_features = [f for f, k in zip(feature_names, kept) if k]
    return X_selected, selected_features, selector


def apply_feature_engineering(
    X_train,
    y_train,
    X_test,
    feature_names,
    variance_threshold=0.01,
    correlation_threshold=0.95,
    k_best=20,
    save_artifacts=False,
    output_dir=None,
):
    """
    Full feature engineering pipeline:
    1. Remove low variance
    2. Remove highly correlated
    3. SelectKBest (mutual information)

    Returns transformed X_train, X_test, selected_feature_names, selectors.
    Does not serialize; result dict is pickle-safe for downstream use.
    save_artifacts, output_dir: ignored (no joblib dump).
    """

    X_tr, feat1, var_sel = remove_low_variance(
        X_train, feature_names, threshold=variance_threshold
    )
    X_te = X_test[:, [feature_names.index(f) for f in feat1]]

    X_tr, feat2 = remove_correlated_features(
        X_tr, feat1, correlation_threshold=correlation_threshold
    )
    X_te = X_te[:, [feat1.index(f) for f in feat2]]

    X_tr, feat3, kb_sel = select_k_best(X_tr, y_train, feat2, k=k_best)
    X_te = kb_sel.transform(X_te)

    selected_features = feat3

    result = {
        "X_train": X_tr,
        "X_test": X_te,
        "feature_names": selected_features,
        "variance_selector": var_sel,
        "select_kbest": kb_sel,
        "feat_after_var": feat1,
        "feat_after_corr": feat2,
    }

    return result
