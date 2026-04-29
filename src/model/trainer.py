"""
Módulo de entrenamiento de modelos de detección de fraude.

Entrena modelos XGBoost y LightGBM con el dataset Gold,
utilizando StratifiedKFold para evaluación robusta.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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

TARGET_COLUMN = "is_fraud"


def train_models(
    dataset_path: Path | None = None,
    n_splits: int = 5,
    random_seed: int = 42,
) -> dict:
    """
    Entrena modelos XGBoost y LightGBM con validación cruzada.

    Args:
        dataset_path: Ruta al dataset Gold (.parquet).
        n_splits: Número de folds para StratifiedKFold.
        random_seed: Seed para reproducibilidad.

    Returns:
        Diccionario con los modelos entrenados y métricas.
    """
    # Cargar dataset
    if dataset_path is None:
        dataset_path = GOLD_DIR / "gold_dataset.parquet"

    if not dataset_path.exists():
        logger.info("Dataset no encontrado, generando...")
        from src.dataset.assembler import assemble_gold_dataset

        df = assemble_gold_dataset(synthetic=True)
    else:
        df = pd.read_parquet(dataset_path)

    # Preparar features y target
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values

    logger.info(f"Dataset: {X.shape[0]} muestras, {X.shape[1]} features")
    logger.info(f"Distribución: {np.bincount(y)}")

    # Cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    xgb_metrics = []
    lgbm_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Entrenar XGBoost
        xgb_model = _train_xgboost(X_train, y_train)
        xgb_fold_metrics = _evaluate_fold(xgb_model, X_val, y_val, fold, "XGBoost")
        xgb_metrics.append(xgb_fold_metrics)

        # Entrenar LightGBM
        lgbm_model = _train_lightgbm(X_train, y_train)
        lgbm_fold_metrics = _evaluate_fold(lgbm_model, X_val, y_val, fold, "LightGBM")
        lgbm_metrics.append(lgbm_fold_metrics)

    # Promediar métricas
    avg_xgb = _average_metrics(xgb_metrics)
    avg_lgbm = _average_metrics(lgbm_metrics)

    logger.info(f"\nXGBoost promedio:\n{avg_xgb}")
    logger.info(f"\nLightGBM promedio:\n{avg_lgbm}")

    # Entrenar modelo final con todos los datos (mejor de los dos)
    if avg_xgb["pr_auc"] >= avg_lgbm["pr_auc"]:
        best_model_name = "XGBoost"
        best_model = _train_xgboost(X, y)
    else:
        best_model_name = "LightGBM"
        best_model = _train_lightgbm(X, y)

    logger.info(f"Modelo final: {best_model_name}")

    # Guardar modelos
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Modelo final (usado en producción)
    joblib.dump(best_model, MODELS_DIR / "fraud_detector.pkl")

    # Modelos de CV (para comparación)
    xgb_final = _train_xgboost(X, y)
    lgbm_final = _train_lightgbm(X, y)
    joblib.dump(xgb_final, MODELS_DIR / "xgboost_model.pkl")
    joblib.dump(lgbm_final, MODELS_DIR / "lightgbm_model.pkl")

    # Guardar feature names
    joblib.dump(FEATURE_COLUMNS, MODELS_DIR / "feature_names.pkl")

    logger.info(f"Modelos guardados en {MODELS_DIR}")

    return {
        "xgb_metrics": avg_xgb,
        "lgbm_metrics": avg_lgbm,
        "best_model": best_model_name,
        "feature_names": FEATURE_COLUMNS,
    }


def _train_xgboost(X: np.ndarray, y: np.ndarray):
    """Entrena un modelo XGBoost."""
    import xgboost as xgb

    scale_pos_weight = np.sum(y == 0) / max(np.sum(y == 1), 1)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    model.fit(X, y)
    return model


def _train_lightgbm(X: np.ndarray, y: np.ndarray):
    """Entrena un modelo LightGBM."""
    import lightgbm as lgb

    scale_pos_weight = np.sum(y == 0) / max(np.sum(y == 1), 1)

    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(X, y)
    return model


def _evaluate_fold(model, X_val: np.ndarray, y_val: np.ndarray, fold: int, name: str) -> dict:
    """Evalúa un modelo en un fold de validación."""
    from sklearn.metrics import (
        average_precision_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)

    return {
        "fold": fold,
        "model": name,
        "roc_auc": roc_auc_score(y_val, y_pred_proba),
        "pr_auc": average_precision_score(y_val, y_pred_proba),
        "recall": recall_score(y_val, y_pred, zero_division=0),
        "precision": precision_score(y_val, y_pred, zero_division=0),
    }


def _average_metrics(metrics: list[dict]) -> dict:
    """Promedia métricas de todos los folds."""
    avg = {}
    for key in metrics[0].keys():
        if key in ("fold", "model"):
            continue
        values = [m[key] for m in metrics]
        avg[f"{key}_mean"] = np.mean(values)
        avg[f"{key}_std"] = np.std(values)
    return avg


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = train_models()
    print("\n" + "=" * 50)
    print("RESULTADOS DEL ENTRENAMIENTO")
    print("=" * 50)
    for key, value in results.items():
        if key != "feature_names":
            print(f"{key}: {value}")
