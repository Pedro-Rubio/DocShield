"""
Módulo de etiquetado heurístico para detección de fraude documental.

Aplica reglas basadas en umbrales para determinar si un documento
es fraudulento. Se usa para generar labels en el dataset sintético
y como baseline para comparar con el modelo de ML.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Umbrales para detección de fraude
ELA_THRESHOLD = 25.0  # ela_score > umbral
OCR_THRESHOLD = 0.55  # ocr_confidence < umbral
IP_THRESHOLD = 0.70  # ip_risk_score > umbral
BLUR_THRESHOLD = 50.0  # blur_score < umbral
EMULATOR_FLAG = True  # emulator_detected == 1
TOR_VPN_FLAG = True  # tor_detected + vpn_detected > 0
REPEAT_THRESHOLD = 3  # repeated_attempts >= umbral


def apply_heuristic_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica reglas heurísticas para etiquetar documentos como fraude.

    Un documento se etiqueta como fraude si 2 o más de estas
    condiciones se cumplen:
    - ELA alto
    - OCR confidence bajo
    - IP risk alto
    - Blur score bajo
    - Emulador detectado
    - Tor/VPN detectado
    - Muchos intentos repetidos

    Args:
        df: DataFrame con las features extraídas.

    Returns:
        DataFrame con columnas 'is_fraud' y 'fraud_signals' añadidas.
    """
    conditions = []

    # 1. ELA alto
    cond_ela = df["ela_score"] > ELA_THRESHOLD
    conditions.append(cond_ela)

    # 2. OCR confidence bajo
    cond_ocr = df["ocr_confidence"] < OCR_THRESHOLD
    conditions.append(cond_ocr)

    # 3. IP risk alto
    cond_ip = df["ip_risk_score"] > IP_THRESHOLD
    conditions.append(cond_ip)

    # 4. Blur score bajo (imagen borrosa)
    cond_blur = df["blur_score"] < BLUR_THRESHOLD
    conditions.append(cond_blur)

    # 5. Emulador detectado
    cond_emulator = df["emulator_detected"] == 1
    conditions.append(cond_emulator)

    # 6. Tor o VPN detectado
    cond_tor_vpn = (df["tor_detected"] + df["vpn_detected"]) > 0
    conditions.append(cond_tor_vpn)

    # 7. Muchos intentos repetidos
    cond_repeat = df["repeated_attempts"] >= REPEAT_THRESHOLD
    conditions.append(cond_repeat)

    # Contar cuántas condiciones se cumplen
    condition_sum = sum(conditions)
    df = df.copy()
    df["is_fraud"] = (condition_sum >= 2).astype(int)

    # Generar columna con señales detectadas
    df["fraud_signals"] = _build_signal_strings(df, conditions)

    logger.info(
        f"Etiquetado heurístico aplicado: {df['is_fraud'].sum()} fraudes "
        f"de {len(df)} ({df['is_fraud'].mean():.1%})"
    )

    return df


def _build_signal_strings(
    df: pd.DataFrame,
    conditions: list[pd.Series],
) -> pd.Series:
    """
    Construye una lista de señales detectadas por fila.

    Args:
        df: DataFrame con las features.
        conditions: Lista de Series booleanos (una por condición).

    Returns:
        Series con strings de señales detectadas.
    """
    signal_names = [
        "ela_anomalia_alta",
        "ocr_confidence_baja",
        "ip_risk_alto",
        "blur_score_bajo",
        "emulador_detectado",
        "tor_vpn_detectado",
        "intentos_repetidos",
    ]

    signals = []
    for idx in range(len(df)):
        row_signals = []
        for cond, name in zip(conditions, signal_names):
            if cond.iloc[idx]:
                row_signals.append(name)
        signals.append(",".join(row_signals))

    return pd.Series(signals, index=df.index)


def compute_fraud_score(row: pd.Series) -> float:
    """
    Calcula el score de fraude para una fila individual.

    Usa los pesos definidos en el scoring model para calcular
    un score entre 0 y 100.

    Args:
        row: Serie con las features de un documento.

    Returns:
        Score de fraude entre 0 y 100.
    """
    from src.api.anti_spoofing_api import WEIGHTS

    weights = WEIGHTS

    # Normalizar cada feature a 0-1 y multiplicar por su peso
    ela_norm = min(row.get("ela_score", 0) / 100.0, 1.0)
    moire_norm = min(row.get("moire_score", 0) / 20.0, 1.0)
    dct_norm = min(row.get("dct_anomaly", 0) / 1.0, 1.0)
    blur_norm = 1.0 - min(row.get("blur_score", 180) / 200.0, 1.0)
    ocr_norm = 1.0 - min(row.get("ocr_confidence", 0.8), 1.0)
    reflection_norm = min(row.get("reflection_score", 0) / 20.0, 1.0)

    score = (
        weights["ela"] * ela_norm
        + weights["moire"] * moire_norm
        + weights["dct"] * dct_norm
        + weights["blur"] * blur_norm
        + weights["ocr"] * ocr_norm
        + weights["reflection"] * reflection_norm
    )

    # Convertir a 0-100
    score = score * 100.0

    # Penalización por liveness
    if row.get("liveness_passed", 1) == 0:
        score += 20.0

    return min(max(score, 0), 100)
