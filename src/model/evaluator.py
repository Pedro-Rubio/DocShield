"""
Módulo de evaluación de modelos de detección de fraude.

Genera métricas detalladas y visualizaciones para evaluar
el rendimiento del modelo: curvas ROC, PR, matriz de confusión.
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
GOLD_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "gold"

FEATURE_COLUMNS = [
    "blur_score",
    "ela_score",
    "ocr_confidence",
    "ocr_field_count",
    "moire_score",
    "dct_anomaly",
    "reflection_score",
    "edge_density",
    "brightness",
    "contrast",
    "noise_ratio",
    "symmetry_score",
    "color_variance",
    "ip_risk_score",
    "emulator_detected",
    "tor_detected",
    "vpn_detected",
    "repeated_attempts",
    "liveness_passed",
    "device_fingerprint_score",
]


def evaluate_model(
    model_path: Optional[Path] = None,
    dataset_path: Optional[Path] = None,
    n_splits: int = 5,
) -> dict:
    """
    Evalúa un modelo con validación cruzada y genera métricas.

    Args:
        model_path: Ruta al modelo serializado.
        dataset_path: Ruta al dataset Gold.
        n_splits: Número de folds.

    Returns:
        Diccionario con todas las métricas.
    """
    if model_path is None:
        model_path = MODELS_DIR / "fraud_detector.pkl"

    if dataset_path is None:
        dataset_path = GOLD_DIR / "gold_dataset.parquet"

    model = joblib.load(model_path)
    df = pd.read_parquet(dataset_path)

    X = df[FEATURE_COLUMNS].values
    y = df["is_fraud"].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_metrics = {
        "roc_auc": [],
        "pr_auc": [],
        "recall": [],
        "precision": [],
        "f1": [],
        "threshold": 35.0,
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Re-entrenar en cada fold para evaluación justa
        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_val)[:, 1]

        # Threshold 35/100 = 0.35
        y_pred = (y_pred_proba > 0.35).astype(int)

        all_metrics["roc_auc"].append(roc_auc_score(y_val, y_pred_proba))
        all_metrics["pr_auc"].append(average_precision_score(y_val, y_pred_proba))
        all_metrics["recall"].append(recall_score(y_val, y_pred, zero_division=0))
        all_metrics["precision"].append(precision_score(y_val, y_pred, zero_division=0))

        from sklearn.metrics import f1_score
        all_metrics["f1"].append(f1_score(y_val, y_pred, zero_division=0))

    # Promediar
    results = {}
    for key in ["roc_auc", "pr_auc", "recall", "precision", "f1"]:
        values = all_metrics[key]
        results[f"{key}_mean"] = np.mean(values)
        results[f"{key}_std"] = np.std(values)
    results["threshold"] = all_metrics["threshold"]

    logger.info("Métricas de evaluación:")
    for key, value in results.items():
        logger.info(f"  {key}: {value:.4f}")

    return results


def plot_roc_curve(
    model_path: Optional[Path] = None,
    dataset_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> None:
    """
    Genera y guarda la curva ROC.

    Args:
        model_path: Ruta al modelo.
        dataset_path: Ruta al dataset.
        output_path: Ruta de salida para la imagen.
    """
    if model_path is None:
        model_path = MODELS_DIR / "fraud_detector.pkl"
    if dataset_path is None:
        dataset_path = GOLD_DIR / "gold_dataset.parquet"
    if output_path is None:
        output_path = MODELS_DIR / "roc_curve.png"

    model = joblib.load(model_path)
    df = pd.read_parquet(dataset_path)

    X = df[FEATURE_COLUMNS].values
    y = df["is_fraud"].values

    model.fit(X, y)
    y_pred_proba = model.predict_proba(X)[:, 1]

    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    auc = roc_auc_score(y, y_pred_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f"ROC AUC = {auc:.4f}", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Curva ROC — DocShield")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Curva ROC guardada en {output_path}")


def plot_pr_curve(
    model_path: Optional[Path] = None,
    dataset_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> None:
    """
    Genera y guarda la curva Precision-Recall.

    Args:
        model_path: Ruta al modelo.
        dataset_path: Ruta al dataset.
        output_path: Ruta de salida para la imagen.
    """
    if model_path is None:
        model_path = MODELS_DIR / "fraud_detector.pkl"
    if dataset_path is None:
        dataset_path = GOLD_DIR / "gold_dataset.parquet"
    if output_path is None:
        output_path = MODELS_DIR / "pr_curve.png"

    model = joblib.load(model_path)
    df = pd.read_parquet(dataset_path)

    X = df[FEATURE_COLUMNS].values
    y = df["is_fraud"].values

    model.fit(X, y)
    y_pred_proba = model.predict_proba(X)[:, 1]

    precision, recall, _ = precision_recall_curve(y, y_pred_proba)
    avg_pr = average_precision_score(y, y_pred_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, label=f"PR AUC = {avg_pr:.4f}", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curva Precision-Recall — DocShield")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Curva PR guardada en {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = evaluate_model()
    plot_roc_curve()
    plot_pr_curve()
    print(f"\nResultados: {results}")
