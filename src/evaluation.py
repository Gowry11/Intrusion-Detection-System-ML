"""
Evaluation module: confusion matrix, ROC curve, class-wise F1, comparison tables.
Saves plots to outputs/plots and reports to outputs/reports.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
    f1_score,
    classification_report,
)
from pathlib import Path

from .utils import PLOTS_DIR, REPORTS_DIR, ensure_dirs


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    title="Confusion Matrix",
    save_path=None,
):
    """Plot and save confusion matrix."""
    ensure_dirs()
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Count"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_roc_curve(
    y_true,
    y_prob,
    class_names,
    title="ROC Curve",
    save_path=None,
):
    """Plot One-vs-Rest ROC curve for multi-class (or binary)."""
    ensure_dirs()
    n_classes = len(class_names)

    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    else:
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true == i, y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure(figsize=(8, 6))
        for i in range(n_classes):
            plt.plot(
                fpr[i],
                tpr[i],
                lw=2,
                label=f"{class_names[i]} (AUC = {roc_auc[i]:.3f})",
            )
        mean_auc = np.mean(list(roc_auc.values()))
        plt.plot([0, 1], [0, 1], "k--", lw=1, label=f"Mean AUC = {mean_auc:.3f}")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def class_wise_f1(y_true, y_pred, class_names):
    """Compute class-wise F1 scores."""
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    return dict(zip(class_names, f1))


def generate_evaluation_report(
    results_list,
    class_names,
    plots_dir=None,
    reports_dir=None,
    model_name="all",
):
    """
    Generate full evaluation: confusion matrices, ROC, comparison table, report.
    results_list: list of dicts with keys: model_name, y_true, y_pred, y_prob (optional)
    """
    plots_dir = Path(plots_dir) if plots_dir else PLOTS_DIR
    reports_dir = Path(reports_dir) if reports_dir else REPORTS_DIR
    ensure_dirs()

    comparison = []
    class_wise_recalls = {}

    for r in results_list:
        name = r.get("model_name", "unknown")
        y_true = r["y_true"]
        y_pred = r["y_pred"]
        y_prob = r.get("y_prob")

        plot_confusion_matrix(
            y_true, y_pred, class_names,
            title=f"Confusion Matrix - {name}",
            save_path=plots_dir / f"confusion_matrix_{name}.png",
        )

        if y_prob is not None:
            try:
                plot_roc_curve(
                    y_true, y_prob, class_names,
                    title=f"ROC Curve - {name}",
                    save_path=plots_dir / f"roc_curve_{name}.png",
                )
            except Exception as e:
                print(f"ROC plot failed for {name}: {e}")

        f1_per_class = class_wise_f1(y_true, y_pred, class_names)
        class_wise_recalls[name] = f1_per_class

        acc = (y_pred == y_true).mean()
        f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        comparison.append({"Model": name, "Accuracy": acc, "F1 (weighted)": f1_w})

    # Accuracy comparison table
    df_comp = pd.DataFrame(comparison)
    df_comp = df_comp.sort_values("Accuracy", ascending=False)
    df_comp.to_csv(reports_dir / "accuracy_comparison.csv", index=False)
    print("\nAccuracy Comparison:\n", df_comp.to_string(index=False))

    # Class-wise recall for U2R and R2L
    u2r_r2l = {}
    for name in class_wise_recalls:
        u2r_r2l[name] = {
            k: v for k, v in class_wise_recalls[name].items()
            if k in ("U2R", "R2L")
        }

    report = {
        "accuracy_comparison": df_comp.to_dict("records"),
        "class_wise_f1": class_wise_recalls,
        "u2r_r2l_recall": u2r_r2l,
    }

    with open(reports_dir / "evaluation_report.txt", "w") as f:
        f.write("EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write("Accuracy Comparison:\n")
        f.write(df_comp.to_string(index=False) + "\n\n")
        f.write("Class-wise F1 (U2R, R2L):\n")
        for name, d in u2r_r2l.items():
            f.write(f"  {name}: {d}\n")

    return report, df_comp
