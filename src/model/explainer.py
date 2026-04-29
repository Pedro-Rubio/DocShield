import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import os

def generate_shap_explanation(model, X_sample: pd.DataFrame, feature_names: List[str] = None) -> Dict[str, Any]:
    """
    Genera explicaciones SHAP para el modelo.

    Args:
        model: Modelo entrenado.
        X_sample: Muestra de features para explicar.
        feature_names: Nombres de las features.

    Returns:
        Diccionario con valores SHAP y datos para visualización.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP no está instalado. Instalar con: pip install shap")
    
    if feature_names is None:
        feature_names = [
            "blur_score", "edge_density", "brightness", "contrast", 
            "noise_ratio", "symmetry_score", "color_variance", "ela_score",
            "moire_score", "dct_score", "reflection_score", "ocr_confidence",
            "ip_risk_score", "emulator_detected", "tor_detected", 
            "vpn_detected", "repeated_attempts", "liveness_passed"
        ]
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample[feature_names])
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Para clasificación binaria, tomar la clase positiva
    
    return {
        'shap_values': shap_values,
        'feature_names': feature_names,
        'base_value': explainer.expected_value,
        'X_sample': X_sample[feature_names].values
    }

def plot_shap_summary(shap_values: np.ndarray, X_sample: pd.DataFrame, 
                      feature_names: List[str], output_path: str = "reports/shap_summary.png") -> None:
    """
    Genera y guarda un gráfico de resumen SHAP.

    Args:
        shap_values: Valores SHAP calculados.
        X_sample: Datos de muestra.
        feature_names: Nombres de las features.
        output_path: Ruta donde guardar la imagen.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP no está instalado.")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample[feature_names], show=False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Gráfico SHAP guardado en {output_path}")

def plot_shap_waterfall(shap_dict: Dict[str, Any], sample_idx: int = 0, 
                        output_path: str = "reports/shap_waterfall.png") -> None:
    """
    Genera y guarda un gráfico de cascada SHAP para una muestra individual.

    Args:
        shap_dict: Diccionario retornado por generate_shap_explanation.
        sample_idx: Índice de la muestra a explicar.
        output_path: Ruta donde guardar la imagen.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP no está instalado.")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    shap_values = shap_dict['shap_values'][sample_idx]
    features = shap_dict['X_sample'][sample_idx]
    feature_names = shap_dict['feature_names']
    
    shap.waterfall_plot(shap.Explanation(
        values=shap_values,
        base_values=shap_dict['base_value'],
        data=features,
        feature_names=feature_names
    ), show=False)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Gráfico de cascada SHAP guardado en {output_path}")
