"""
Main script: Complete pipeline for Hybrid Intrusion Detection System.
Runs preprocessing, feature engineering, training, tuning, stacking,
evaluation, explainability, and robustness testing.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import preprocess
from src.feature_engineering import apply_feature_engineering
from src.model_training import train_all_models, apply_smote
from src.hyperparameter_tuning import tune_all_models
from src.stacking_model import build_stacking_model, train_and_evaluate_stacking
from src.evaluation import generate_evaluation_report
from src.explainability import shap_explain
from src.robustness import run_robustness_tests
from src.utils import (
    DATA_DIR,
    MODELS_DIR,
    PLOTS_DIR,
    REPORTS_DIR,
    ensure_dirs,
)


def main():
    ensure_dirs()

    print("=" * 60)
    print("Explainable & Optimized Hybrid Intrusion Detection System")
    print("NSL-KDD Dataset")
    print("=" * 60)

    # 1. Preprocess
    print("\n[1/8] Preprocessing...")
    prep = preprocess(
        data_dir=DATA_DIR,
        test_size=0.2,
        task="multiclass",
        save_artifacts=True,
        output_dir=DATA_DIR,
    )
    X_train, X_test = prep["X_train"], prep["X_test"]
    y_train, y_test = prep["y_train"], prep["y_test"]
    feature_names = prep["feature_names"]
    class_names = prep["class_names"]
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}, Classes: {class_names}")

    # 2. Feature engineering
    print("\n[2/8] Feature Engineering...")
    fe_result = apply_feature_engineering(
        X_train, y_train, X_test, feature_names,
        variance_threshold=0.01,
        correlation_threshold=0.95,
        k_best=20,
        save_artifacts=True,
        output_dir=DATA_DIR,
    )
    X_train_fe = fe_result["X_train"]
    X_test_fe = fe_result["X_test"]
    selected_features = fe_result["feature_names"]
    print(f"  Selected features: {len(selected_features)}")

    # 3. SMOTE on training data
    print("\n[3/8] Applying SMOTE...")
    X_train_smote, y_train_smote, _ = apply_smote(X_train_fe, y_train)

    # 4. Train all models
    print("\n[4/8] Training base models...")
    train_result = train_all_models(
        X_train_smote, y_train_smote, X_test_fe, y_test,
        class_names=class_names,
        task="multiclass",
        use_smote=False,  # Already applied above
        save_models=True,
        models_dir=MODELS_DIR,
    )
    base_models = train_result["models"]
    base_results = train_result["results"]

    # 5. Hyperparameter tuning
    print("\n[5/8] Hyperparameter Tuning (RandomizedSearchCV)...")
    tuned = tune_all_models(
        X_train_smote, y_train_smote, X_test_fe, y_test,
        n_iter=15,
        cv=3,
        save_models=True,
        models_dir=MODELS_DIR,
    )

    # 6. Stacking model
    print("\n[6/8] Building Stacking Model...")
    tuned_rf = tuned.get("random_forest", {}).get("model")
    tuned_xgb = tuned.get("xgboost", {}).get("model")

    if tuned_rf is not None and tuned_xgb is not None:
        stack = build_stacking_model(tuned_rf, tuned_xgb)
        stack_result = train_and_evaluate_stacking(
            stack, X_train_smote, y_train_smote, X_test_fe, y_test,
            class_names=class_names,
            save_model=True,
            models_dir=MODELS_DIR,
        )
        stacking_model = stack_result["model"]
        stacking_acc = stack_result["accuracy"]
    else:
        stacking_model = None
        stacking_acc = 0.0
        print("  Skipping stacking: tuned RF or XGB not available.")

    # 7. Evaluation
    print("\n[7/8] Evaluation...")
    eval_results = []
    for r in base_results:
        name = r["model_name"]
        model = base_models.get(name)
        if model is None:
            continue
        y_prob = model.predict_proba(X_test_fe) if hasattr(model, "predict_proba") else None
        eval_results.append({
            "model_name": name,
            "y_true": y_test,
            "y_pred": model.predict(X_test_fe),
            "y_prob": y_prob,
        })

    if stacking_model is not None:
        eval_results.append({
            "model_name": "stacking",
            "y_true": y_test,
            "y_pred": stacking_model.predict(X_test_fe),
            "y_prob": stacking_model.predict_proba(X_test_fe),
        })

    report, df_comp = generate_evaluation_report(
        eval_results,
        class_names,
        plots_dir=PLOTS_DIR,
        reports_dir=REPORTS_DIR,
    )

    # U2R and R2L class-wise recall
    u2r_r2l = report.get("u2r_r2l_recall", {})
    print("\nClass-wise Recall (U2R, R2L):")
    for name, d in u2r_r2l.items():
        print(f"  {name}: {d}")

    # 8. Explainability (SHAP)
    print("\n[8/8] Explainability (SHAP)...")
    xgb_model = base_models.get("xgboost") or tuned.get("xgboost", {}).get("model")
    if xgb_model is not None:
        shap_explain(
            xgb_model, X_test_fe[:500], selected_features,
            class_names=class_names,
            model_name="xgboost",
            save_dir=PLOTS_DIR,
        )
    if stacking_model is not None:
        try:
            shap_explain(
                stacking_model, X_test_fe[:200], selected_features,
                class_names=class_names,
                model_name="stacking",
                max_samples=200,
                save_dir=PLOTS_DIR,
            )
        except Exception as e:
            print(f"  SHAP for stacking skipped: {e}")

    # Robustness testing
    print("\nRobustness Testing...")
    models_for_robustness = dict(base_models)
    if stacking_model is not None:
        models_for_robustness["stacking"] = stacking_model
    robustness_results = run_robustness_tests(
        models_for_robustness,
        X_test_fe,
        y_test,
        noise_std=0.01,
        save_report=True,
        reports_dir=REPORTS_DIR,
    )

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    all_accs = [(r["model_name"], r["accuracy"]) for r in base_results]
    if stacking_model is not None:
        all_accs.append(("stacking", stacking_acc))
    all_accs.append(("tuned_rf", tuned.get("random_forest", {}).get("test_accuracy", 0)))
    all_accs.append(("tuned_xgb", tuned.get("xgboost", {}).get("test_accuracy", 0)))
    all_accs.append(("tuned_lgb", tuned.get("lightgbm", {}).get("test_accuracy", 0)))

    best_name = max(all_accs, key=lambda x: x[1])
    print(f"\nBest Model: {best_name[0]}")
    print(f"Final Accuracy: {best_name[1]:.4f} ({best_name[1]*100:.2f}%)")

    print("\nAccuracy Comparison Table:")
    print(df_comp.to_string(index=False))

    print("\nRobustness Comparison:")
    for r in robustness_results:
        print(f"  {r['model_name']}: Orig={r['accuracy_original']:.4f}, Noisy={r['accuracy_noisy']:.4f}, Drop={r['performance_drop_pct']:.2f}%")

    print("\nOutputs saved to:")
    print(f"  Plots: {PLOTS_DIR}")
    print(f"  Reports: {REPORTS_DIR}")
    print(f"  Models: {MODELS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
