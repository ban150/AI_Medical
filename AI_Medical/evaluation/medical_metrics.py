"""
Métriques pour un cadre binaire NORMAL (0) vs PNEUMONIA (1).

Conformes aux usages en évaluation de modèles médicaux : précision, rappel, F1,
matrice de confusion. Les fonctions de tracé ne s'exécutent que si appelées explicitement.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
    pos_label: int = 1,
) -> dict[str, float]:
    """
    Calcule accuracy, precision, recall, F1 (binaire).

    y_true, y_pred : entiers {0, 1}.
    y_score : probabilités de la classe positive (pour AUC-ROC).
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)

    out: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
    }
    if y_score is not None:
        y_score = np.asarray(y_score).ravel().astype(np.float64)
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            out["roc_auc"] = float("nan")
    return out


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: tuple[str, str] = ("NORMAL", "PNEUMONIA"),
    save_path: str | None = None,
    show: bool = False,
) -> Any:
    """
    Trace la matrice de confusion (normalisée et brute).

    Ne fait aucun affichage si show=False et save_path=None (appel minimal pour tests sans GUI).
    """
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    axes[0].set_title("Matrice de confusion (effectifs)")
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(labels)
    axes[0].set_yticklabels(labels)
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.colorbar(im0, ax=axes[0])

    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm.astype(float) / row_sums
    im1 = axes[1].imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Greens, vmin=0, vmax=1)
    axes[1].set_title("Par ligne (rappel par vraie classe)")
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(labels)
    axes[1].set_yticklabels(labels)
    fig.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return cm


def print_medical_report(metrics: dict[str, float], title: str = "Métriques test") -> None:
    """Affichage structuré pour inclusion dans un rapport de mémoire."""
    print(f"\n--- {title} ---")
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        if k in metrics:
            v = metrics[k]
            print(f"  {k:12s}: {v:.4f}" if not np.isnan(v) else f"  {k:12s}: n/a")
