"""
evaluate_rf_model.py
--------------------

Evaluate a trained Random Forest classifier against held-out test data,
report core metrics, and persist diagnostic plots.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = Path("models/rf_model.joblib")
X_TEST_PATH = Path("data/X_test.npy")
Y_TEST_PATH = Path("data/y_test.npy")
CONF_MATRIX_FIG = Path("confusion_matrix.png")
ROC_CURVE_FIG = Path("roc_curve.png")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def load_artifacts(
    model_path: Union[str, Path], x_path: Union[str, Path], y_path: Union[str, Path]
) -> Tuple[object, np.ndarray, np.ndarray]:
    """
    Load model and test arrays from disk.

    Raises:
        FileNotFoundError: if any file is missing.
        Exception: for other loading errors.
    """
    model = joblib.load(model_path)
    X_test = np.load(x_path)
    y_test = np.load(y_path)
    return model, X_test, y_test


def get_score_matrix(model, X: np.ndarray) -> np.ndarray:
    """
    Return probability/score matrix for ROC + AUC calculations.

    Prefers predict_proba; falls back to decision_function. Ensures the
    returned array is 2D for consistent downstream processing.
    """
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
    else:
        raise AttributeError(
            "Model must implement predict_proba or decision_function."
        )

    # Normalize to 2D
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    return scores


def compute_macro_auc(
    y_true: np.ndarray, scores: np.ndarray, classes: np.ndarray
) -> float:
    """
    Compute macro-average AUC score.

    Handles binary and multi-class by using the macro-average one-vs-rest
    strategy for multi-class settings.
    """
    if len(classes) == 2:
        # Select positive class column if available
        column = 1 if scores.shape[1] == 2 else 0
        return roc_auc_score(y_true, scores[:, column])
    return roc_auc_score(y_true, scores, multi_class="ovr", average="macro")


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path) -> None:
    """Generate and persist confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_roc(
    y_true: np.ndarray, scores: np.ndarray, classes: np.ndarray, save_path: Path
) -> None:
    """
    Plot ROC curve(s) for binary or multi-class classifications.

    For multi-class, one curve per class is drawn using a one-vs-rest approach.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    if len(classes) == 2:
        column = 1 if scores.shape[1] == 2 else 0
        fpr, tpr, _ = roc_curve(y_true, scores[:, column], pos_label=classes[1])
        auc_val = roc_auc_score(y_true, scores[:, column])
        ax.plot(fpr, tpr, label=f"ROC (AUC = {auc_val:.3f})", color="tab:blue")
    else:
        y_bin = label_binarize(y_true, classes=classes)
        if scores.shape[1] != len(classes):
            raise ValueError("Score matrix columns must match number of classes.")
        for idx, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, idx], scores[:, idx])
            auc_val = roc_auc_score(y_bin[:, idx], scores[:, idx])
            ax.plot(fpr, tpr, label=f"Class {cls} (AUC = {auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def print_summary(
    accuracy: float,
    precision: float,
    recall: float,
    f1: float,
    auc_score: float,
    conf_path: Path,
    roc_path: Path,
    report: str,
) -> None:
    """Pretty-print metrics table, saved figure paths, and classification report."""
    summary = pd.DataFrame(
        [
            {
                "accuracy": accuracy,
                "precision_weighted": precision,
                "recall_weighted": recall,
                "f1_weighted": f1,
                "auc_macro": auc_score,
            }
        ]
    )
    print("=== Evaluation Metrics ===")
    print(summary.round(4).to_string(index=False))
    print("\nSaved figures:")
    print(f" - Confusion Matrix: {conf_path.resolve()}")
    print(f" - ROC Curve: {roc_path.resolve()}")
    print("\n=== Classification Report ===")
    print(report)


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------
def main() -> None:
    """Main evaluation workflow."""
    try:
        model, X_test, y_test = load_artifacts(MODEL_PATH, X_TEST_PATH, Y_TEST_PATH)
    except FileNotFoundError as exc:
        sys.exit(f"Missing file: {exc}")
    except Exception as exc:  # pylint: disable=broad-except
        sys.exit(f"Failed to load artifacts: {exc}")

    # Generate predictions
    y_pred = model.predict(X_test)

    # Core metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_test, y_pred, digits=4)

    # Probability/decision scores for ROC + AUC
    classes = np.unique(y_test)
    try:
        scores = get_score_matrix(model, X_test)
        auc_score = compute_macro_auc(y_test, scores, classes)
    except Exception as exc:  # pylint: disable=broad-except
        scores = None
        auc_score = float("nan")
        print(f"Warning: could not compute AUC ({exc}).")

    # Visualizations
    CONF_MATRIX_FIG.parent.mkdir(parents=True, exist_ok=True)
    ROC_CURVE_FIG.parent.mkdir(parents=True, exist_ok=True)

    plot_confusion(y_test, y_pred, CONF_MATRIX_FIG)
    if scores is not None:
        try:
            plot_roc(y_test, scores, classes, ROC_CURVE_FIG)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Warning: ROC curve generation failed ({exc}).")
    else:
        print("Skipping ROC curve: no score matrix available.")

    # Reporting
    print_summary(
        accuracy,
        precision,
        recall,
        f1,
        auc_score,
        CONF_MATRIX_FIG,
        ROC_CURVE_FIG,
        report,
    )


if __name__ == "__main__":
    main()

