"""
Módulo de interpretabilidad de modelos usando SHAP.

Genera explicaciones para las predicciones del modelo,
permitiendo entender qué features contribuyen a cada
decisión de fraude.
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

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

FEATURE_NAMES_ES = {
    "blur_score": "Nitidez de imagen",
    "ela_score": "Anomalía ELA",
    "ocr_confidence": "Confianza OCR",
    "ocr_field_count": "Campos de texto detectados",
    "moire_score": "Patrón de Moiré (pantalla)",
    "dct_anomaly": "Inconsistencia DCT",
    "reflection_score": "Reflexión especular",
    "edge_density": "Densidad de bordes",
    "brightness": "Brillo promedio",
    "contrast": "Contraste",
    "noise_ratio": "Ratio de ruido",
    "symmetry_score": "Simetría del documento",
    "color_variance": "Varianza de color",
    "ip_risk_score": "Riesgo de IP",
    "emulator_detected": "Emulador detectado",
    "tor_detected": "Tor detectado",
    "vpn_detected": "VPN detectada",
    "repeated_attempts": "Intentos repetidos",
    "liveness_passed": "Liveness superado",
    "device_fingerprint_score": "Fingerprint del dispositivo",
}


def explain_prediction(
    model_path: Optional[Path] = None,
    features: Optional[np.ndarray] = None,
    feature_names: Optional[list[str]] = None,
) -> dict:
    """
    Explica una predicción individual usando SHAP.

    Args:
        model_path: Ruta al modelo serializado.
        features: Array de features del documento a explicar (1xN).
        feature_names: Lista de nombres de features.

    Returns:
        Diccionario con:
        - feature_importances: Dict de feature -> valor SHAP
        - top_signals: Lista de señales principales (en español)
        - base_value: Valor base del modelo
        - predicted_proba: Probabilidad predicha
    """
    import shap

    if model_path is None:
        model_path = MODELS_DIR / "fraud_detector.pkl"
    if feature_names is None:
        feature_names = FEATURE_COLUMNS

    model = joblib.load(model_path)

    # Si no se pasan features, usar una muestra del dataset
    if features is None:
        df = pd.read_parquet(GOLD_DIR / "gold_dataset.parquet")
        features = df[feature_names].values[:1]

    # Asegurar formato 2D
    if features.ndim == 1:
        features = features.reshape(1, -1)

    # Crear explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)

    # Para clasificación binaria, tomar la clase positiva (fraude)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Crear dict de importancias
    feature_importances = {}
    for name, value in zip(feature_names, shap_values[0]):
        feature_importances[name] = float(value)

    # Ordenar por importancia absoluta
    sorted_features = sorted(
        feature_importances.items(), key=lambda x: abs(x[1]), reverse=True
    )

    # Top señales en español
    top_signals = []
    for name, value in sorted_features[:5]:
        if abs(value) > 0.01:  # Solo señales significativas
            direction = "↑" if value > 0 else "↓"
            top_signals.append(f"{FEATURE_NAMES_ES.get(name, name)} ({direction})")

    # Probabilidad predicha
    predicted_proba = model.predict_proba(features)[0, 1]

    return {
        "feature_importances": dict(sorted_features),
        "top_signals": top_signals,
        "base_value": float(explainer.expected_value),
        "predicted_proba": float(predicted_proba),
    }


def compute_global_importance(
    model_path: Optional[Path] = None,
    dataset_path: Optional[Path] = None,
    n_samples: int = 500,
) -> dict:
    """
    Calcula la importancia global de features usando SHAP.

    Args:
        model_path: Ruta al modelo.
        dataset_path: Ruta al dataset.
        n_samples: Número de muestras para el análisis.

    Returns:
        Diccionario con feature -> importancia media absoluta.
    """
    import shap

    if model_path is None:
        model_path = MODELS_DIR / "fraud_detector.pkl"
    if dataset_path is None:
        dataset_path = GOLD_DIR / "gold_dataset.parquet"

    model = joblib.load(model_path)
    df = pd.read_parquet(dataset_path)

    X = df[FEATURE_COLUMNS].values[:n_samples]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Importancia media absoluta
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    importance = {}
    for name, value in zip(FEATURE_COLUMNS, mean_abs_shap):
        importance[name] = float(value)

    # Ordenar por importancia
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    importance = compute_global_importance()
    print("\nImportancia global de features (SHAP):")
    for name, value in importance.items():
        print(f"  {FEATURE_NAMES_ES.get(name, name):30s} {value:.4f}")
