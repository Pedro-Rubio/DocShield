import pandas as pd
import numpy as np
import joblib
import os
from typing import Tuple, Dict, Any
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, recall_score, precision_score
import xgboost as xgb
import lightgbm as lgb

from src.model.constants import FEATURE_COLS

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

def evaluate_model(model, X_train: pd.DataFrame, y_train: pd.Series, 
                  X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evalúa el modelo en un test set separado (no visto en entrenamiento).

    Args:
        model: Modelo entrenado.
        X_train: Features de entrenamiento.
        y_train: Labels de entrenamiento.
        X_test: Features de prueba.
        y_test: Labels de prueba.

    Returns:
        Diccionario con métricas de evaluación.
    """
    # Evaluar en train (para detectar overfitting)
    y_train_pred_proba = model.predict_proba(X_train[FEATURE_COLS])[:, 1]
    train_roc_auc = roc_auc_score(y_train, y_train_pred_proba)
    train_pr_auc = average_precision_score(y_train, y_train_pred_proba)
    
    # Evaluar en test (métrica real)
    y_test_pred_proba = model.predict_proba(X_test[FEATURE_COLS])[:, 1]
    test_roc_auc = roc_auc_score(y_test, y_test_pred_proba)
    test_pr_auc = average_precision_score(y_test, y_test_pred_proba)
    
    y_test_pred = (y_test_pred_proba >= 0.5).astype(int)
    from sklearn.metrics import recall_score, precision_score
    recall = recall_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    
    return {
        'Train_ROC-AUC': train_roc_auc,
        'Train_PR-AUC': train_pr_auc,
        'Test_ROC-AUC': test_roc_auc,
        'Test_PR-AUC': test_pr_auc,
        'Recall (fraud)': recall,
        'Precision (fraud)': precision
    }

def evaluate_on_test_set(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evalúa el modelo en un conjunto de prueba separado (no visto en entrenamiento).
    
    Args:
        model: Modelo entrenado.
        X_test: Features de prueba.
        y_test: Labels de prueba.
    
    Returns:
        Diccionario con métricas en test set.
    """
    y_pred_proba = model.predict_proba(X_test[FEATURE_COLS])[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    return {
        'Test_ROC-AUC': roc_auc,
        'Test_PR-AUC': pr_auc
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
    from sklearn.model_selection import train_test_split
    
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
    
    # Split train/test
    X = df[FEATURE_COLS]
    y = df['is_fraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    if args.model == 'xgb':
        model = train_xgboost(X_train, y_train)
    else:
        model = train_lightgbm(X_train, y_train)
    
    # Evaluar en test set (no visto)
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
    print("\nMétricas de evaluación (Train/Test):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    save_model(model, args.output)
