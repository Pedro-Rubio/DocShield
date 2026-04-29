import pandas as pd
import numpy as np
from typing import List, Optional

def assemble_gold_layer(feature_files: List[str], output_path: str = "data/gold/dataset.parquet") -> pd.DataFrame:
    """
    Ensambla la Gold Layer combinando múltiples archivos de la Silver Layer.

    Args:
        feature_files: Lista de rutas a archivos JSON de la Silver Layer.
        output_path: Ruta donde guardar el dataset final en formato Parquet.

    Returns:
        DataFrame con el dataset Gold ensamblado.
    """
    import json
    import os
    
    all_features = []
    
    for file_path in feature_files:
        if not os.path.exists(file_path):
            print(f"Advertencia: {file_path} no existe, se omite.")
            continue
        
        with open(file_path, 'r') as f:
            data = json.load(f)
            all_features.extend(data if isinstance(data, list) else [data])
    
    if not all_features:
        raise ValueError("No se encontraron datos para ensamblar.")
    
    df = pd.DataFrame(all_features)
    
    numeric_cols = ["blur_score", "edge_density", "brightness", "contrast", 
                   "noise_ratio", "symmetry_score", "color_variance", "ela_score",
                   "moire_score", "dct_score", "reflection_score", "ocr_confidence",
                   "ip_risk_score", "emulator_detected", "tor_detected", 
                   "vpn_detected", "repeated_attempts", "liveness_passed"]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=["blur_score", "ela_score"])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    print(f"Gold Layer ensamblada: {len(df)} muestras guardadas en {output_path}")
    return df

def create_feature_vector(features: dict) -> np.ndarray:
    """
    Crea un vector de features numérico para el modelo.

    Args:
        features: Diccionario con las features extraídas.

    Returns:
        Array numpy con el vector de features.
    """
    feature_order = [
        "blur_score", "edge_density", "brightness", "contrast", 
        "noise_ratio", "symmetry_score", "color_variance", "ela_score",
        "moire_score", "dct_score", "reflection_score", "ocr_confidence",
        "ip_risk_score", "emulator_detected", "tor_detected", 
        "vpn_detected", "repeated_attempts", "liveness_passed"
    ]
    
    return np.array([features.get(feat, 0) for feat in feature_order])
