import pandas as pd
import numpy as np
from typing import List, Dict

ELA_THRESHOLD = 25.0
OCR_THRESHOLD = 0.55
IP_THRESHOLD = 0.70
BLUR_THRESHOLD = 50.0
EMULATOR_FLAG = True
TOR_VPN_FLAG = True
REPEAT_THRESHOLD = 3

def apply_heuristics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica reglas heurísticas para etiquetar muestras como fraude.

    Reglas para generar is_fraud = 1:
    fraude si 2 o más de estas condiciones se cumplen:
    - ela_score > ELA_THRESHOLD
    - ocr_confidence < OCR_THRESHOLD
    - ip_risk_score > IP_THRESHOLD
    - blur_score < BLUR_THRESHOLD
    - emulator_detected == 1 (si EMULATOR_FLAG)
    - tor_detected + vpn_detected > 0 (si TOR_VPN_FLAG)
    - repeated_attempts >= REPEAT_THRESHOLD

    Args:
        df: DataFrame con las features extraídas.

    Returns:
        DataFrame con la columna 'is_fraud' agregada.
    """
    conditions = pd.DataFrame()

    conditions["ela"] = df["ela_score"] > ELA_THRESHOLD
    conditions["ocr"] = df["ocr_confidence"] < OCR_THRESHOLD
    conditions["ip"] = df["ip_risk_score"] > IP_THRESHOLD
    conditions["blur"] = df["blur_score"] < BLUR_THRESHOLD
    conditions["emulator"] = df["emulator_detected"] == 1 if EMULATOR_FLAG else False
    conditions["tor_vpn"] = (df["tor_detected"] + df["vpn_detected"]) > 0 if TOR_VPN_FLAG else False
    conditions["repeat"] = df["repeated_attempts"] >= REPEAT_THRESHOLD

    df = df.copy()
    df["fraud_signals"] = conditions.sum(axis=1)
    df["is_fraud"] = (df["fraud_signals"] >= 2).astype(int)

    return df

def get_fraud_signals(row: pd.Series) -> List[str]:
    """
    Obtiene la lista de señales de fraude detectadas para una muestra.

    Args:
        row: Serie con los datos de una muestra.

    Returns:
        Lista de strings describiendo las señales detectadas.
    """
    signals = []

    if row["ela_score"] > ELA_THRESHOLD:
        signals.append(f"ELA anomalía alta ({row['ela_score']:.1f})")
    if row["ocr_confidence"] < OCR_THRESHOLD:
        signals.append(f"OCR confidence baja ({row['ocr_confidence']:.2f})")
    if row["ip_risk_score"] > IP_THRESHOLD:
        signals.append(f"IP de alto riesgo ({row['ip_risk_score']:.2f})")
    if row["blur_score"] < BLUR_THRESHOLD:
        signals.append(f"Blur bajo ({row['blur_score']:.1f})")
    if row["emulator_detected"] == 1:
        signals.append("Emulador detectado")
    if (row["tor_detected"] + row["vpn_detected"]) > 0:
        signals.append("Tor/VPN detectado")
    if row["repeated_attempts"] >= REPEAT_THRESHOLD:
        signals.append(f"Múltiples intentos ({row['repeated_attempts']})")
    if row.get("moire_score", 0) > 8.5:
        signals.append(f"Posible screen capture (Moiré: {row['moire_score']:.1f})")
    if row.get("liveness_passed", 1) == 0:
        signals.append("Liveness check fallido")

    return signals
