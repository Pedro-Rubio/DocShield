import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    classification_report, confusion_matrix
)
import os

def evaluate_model_metrics(model, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Calcula métricas detalladas y curvas para evaluación del modelo.

    Args:
        model: Modelo entrenado.
        X: Features de evaluación.
        y: Labels verdaderos.

    Returns:
        Diccionario con métricas y curvas.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    
    FEATURE_COLS = [
        "blur_score", "edge_density", "brightness", "contrast", 
        "noise_ratio", "symmetry_score", "color_variance", "ela_score",
        "moire_score", "dct_score", "reflection_score", "ocr_confidence",
        "ip_risk_score", "emulator_detected", "tor_detected", 
        "vpn_detected", "repeated_attempts", "liveness_passed"
    ]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_proba = cross_val_predict(model, X[FEATURE_COLS], y, cv=skf, method='predict_proba')[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    roc_auc = roc_auc_score(y, y_pred_proba)
    pr_auc = average_precision_score(y, y_pred_proba)
    recall = recall_score(y, y_pred)
    precision = precision_score(y, y_pred)
    
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    precision_curve, recall_curve, _ = precision_recall_curve(y, y_pred_proba)
    
    cm = confusion_matrix(y, y_pred)
    
    return {
        'ROC-AUC': roc_auc,
        'PR-AUC': pr_auc,
        'Recall (fraud)': recall,
        'Precision (fraud)': precision,
        'fpr': fpr,
        'tpr': tpr,
        'precision_curve': precision_curve,
        'recall_curve': recall_curve,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, output_path: str = "reports/roc_curve.png") -> None:
    """
    Genera y guarda la curva ROC.

    Args:
        fpr: False positive rate.
        tpr: True positive rate.
        roc_auc: Área bajo la curva ROC.
        output_path: Ruta donde guardar la imagen.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()
    print(f"Curva ROC guardada en {output_path}")

def plot_pr_curve(precision: np.ndarray, recall: np.ndarray, pr_auc: float, output_path: str = "reports/pr_curve.png") -> None:
    """
    Genera y guarda la curva Precision-Recall.

    Args:
        precision: Precision values.
        recall: Recall values.
        pr_auc: Área bajo la curva PR.
        output_path: Ruta donde guardar la imagen.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()
    print(f"Curva PR guardada en {output_path}")

def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Imprime el reporte de clasificación detallado.

    Args:
        y_true: Labels verdaderos.
        y_pred: Labels predichos.
    """
    print("\nReporte de clasificación:")
    print(classification_report(y_true, y_pred, target_names=['Legit', 'Fraud']))

def plot_confusion_matrix(cm: np.ndarray, output_path: str = "reports/confusion_matrix.png") -> None:
    """
    Genera y guarda la matriz de confusión.

    Args:
        cm: Matriz de confusión.
        output_path: Ruta donde guardar la imagen.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Legit', 'Fraud'], rotation=45)
    plt.yticks(tick_marks, ['Legit', 'Fraud'])
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(output_path)
    plt.close()
    print(f"Matriz de confusión guardada en {output_path}")
