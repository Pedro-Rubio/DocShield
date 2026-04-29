import pandas as pd
import numpy as np
import joblib
import os
from typing import Tuple, Dict, Any
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import xgboost as xgb
import lightgbm as lgb

FEATURE_COLS = [
    "blur_score", "edge_density", "brightness", "contrast", 
    "noise_ratio", "symmetry_score", "color_variance", "ela_score",
    "moire_score", "dct_score", "reflection_score", "ocr_confidence",
    "ip_risk_score", "emulator_detected", "tor_detected", 
    "vpn_detected", "repeated_attempts", "liveness_passed"
]

def train_xgboost(X: pd.DataFrame, y: pd.Series, params: Dict = None) -> xgb.XGBClassifier:
    """
    Entrena un modelo XGBoost para clasificación de fraude.

    Args:
        X: Features de entrenamiento.
        y: Labels (is_fraud).
        params: Parámetros opcionales para XGBClassifier.

    Returns:
        Modelo XGBoost entrenado.
    """
    if params is None:
        params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',
            'random_state': 42,
            'n_jobs': -1
        }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X[FEATURE_COLS], y)
    return model

def train_lightgbm(X: pd.DataFrame, y: pd.Series, params: Dict = None) -> lgb.LGBMClassifier:
    """
    Entrena un modelo LightGBM para clasificación de fraude.

    Args:
        X: Features de entrenamiento.
        y: Labels (is_fraud).
        params: Parámetros opcionales para LGBMClassifier.

    Returns:
        Modelo LightGBM entrenado.
    """
    if params is None:
        params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'objective': 'binary',
            'metric': 'auc',
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X[FEATURE_COLS], y)
    return model

def evaluate_model(model, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
    """
    Evalúa el modelo usando StratifiedKFold y métricas PR-AUC, ROC-AUC.

    Args:
        model: Modelo entrenado.
        X: Features.
        y: Labels.
        cv: Número de folds para cross-validation.

    Returns:
        Diccionario con métricas de evaluación.
    """
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    y_pred_proba = cross_val_predict(model, X[FEATURE_COLS], y, cv=skf, method='predict_proba')[:, 1]
    
    roc_auc = roc_auc_score(y, y_pred_proba)
    pr_auc = average_precision_score(y, y_pred_proba)
    
    y_pred = (y_pred_proba >= 0.5).astype(int)
    recall = recall_score(y, y_pred)
    precision = precision_score(y, y_pred)
    
    return {
        'ROC-AUC': roc_auc,
        'PR-AUC': pr_auc,
        'Recall (fraud)': recall,
        'Precision (fraud)': precision
    }

def save_model(model, path: str = "models/docshield_model.pkl") -> None:
    """
    Guarda el modelo entrenado usando joblib.

    Args:
        model: Modelo entrenado.
        path: Ruta donde guardar el modelo.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Modelo guardado en {path}")

def load_model(path: str = "models/docshield_model.pkl"):
    """
    Carga un modelo guardado.

    Args:
        path: Ruta al modelo guardado.

    Returns:
        Modelo cargado.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modelo no encontrado en {path}")
    return joblib.load(path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Entrenar modelo DocShield')
    parser.add_argument('--data', type=str, default='data/gold/dataset.parquet',
                        help='Ruta al dataset Gold')
    parser.add_argument('--model', type=str, default='xgb', choices=['xgb', 'lgb'],
                        help='Tipo de modelo')
    parser.add_argument('--output', type=str, default='models/docshield_model.pkl',
                        help='Ruta de salida del modelo')
    args = parser.parse_args()
    
    df = pd.read_parquet(args.data)
    print(f"Dataset cargado: {len(df)} muestras")
    print(f"Distribución de clases:\n{df['is_fraud'].value_counts()}")
    
    if args.model == 'xgb':
        model = train_xgboost(df, df['is_fraud'])
    else:
        model = train_lightgbm(df, df['is_fraud'])
    
    metrics = evaluate_model(model, df, df['is_fraud'])
    print("\nMétricas de evaluación:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    save_model(model, args.output)
