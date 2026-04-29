"""
Generador de dataset sintético para entrenamiento del modelo de detección de fraude.

Genera datos con distribuciones calibradas para documentos legítimos y fraudulentos
de 4 subtipos diferentes.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Directorio por defecto para datos gold
GOLD_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "gold"


def generate_fraud_dataset(
    n_legit: int = 4000,
    n_fraud: int = 600,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Genera un dataset sintético con documentos legítimos y fraudulentos.

    Args:
        n_legit: Número de muestras legítimas.
        n_fraud: Número de muestras fraudulentas.
        random_seed: Seed para reproducibilidad.

    Returns:
        DataFrame con todas las features y columna 'is_fraud'.
    """
    rng = np.random.default_rng(random_seed)

    # Generar muestras legítimas
    legit = _generate_legit(n_legit, rng)

    # Generar muestras fraudulentas (4 subtipos)
    fraud_per_type = n_fraud // 4
    fraud_edited = _generate_fraud_edited(fraud_per_type, rng)
    fraud_screen = _generate_fraud_screen(fraud_per_type, rng)
    fraud_printed = _generate_fraud_printed(fraud_per_type, rng)
    fraud_stolen = _generate_fraud_stolen(n_fraud - 3 * fraud_per_type, rng)

    fraud = pd.concat([fraud_edited, fraud_screen, fraud_printed, fraud_stolen], ignore_index=True)

    dataset = pd.concat([legit, fraud], ignore_index=True)
    dataset = dataset.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    logger.info(
        f"Dataset generado: {len(legit)} legítimos, {len(fraud)} fraudulentos "
        f"({len(dataset)} total)"
    )

    return dataset


def _generate_legit(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Genera documentos legítimos con distribuciones características.

    Documentos legítimos típicos tienen:
    - Alta nitidez (blur_score ~ Normal(180, 40))
    - Poca recompresión (ela_score ~ Exponential(8))
    - Alta confianza OCR (ocr_confidence ~ Beta(9, 1.5))
    - IP de bajo riesgo (ip_risk_score ~ Beta(1.5, 12))
    """
    data = {
        "blur_score": rng.normal(180, 40, n),
        "ela_score": rng.exponential(8, n),
        "ocr_confidence": rng.beta(9, 1.5, n),
        "ocr_field_count": rng.integers(8, 20, n).astype(float),
        "moire_score": rng.normal(3.0, 1.5, n),
        "dct_anomaly": rng.normal(0.3, 0.15, n),
        "reflection_score": rng.normal(2.0, 1.0, n),
        "edge_density": rng.beta(5, 20, n),
        "brightness": rng.normal(180, 30, n),
        "contrast": rng.normal(60, 15, n),
        "noise_ratio": rng.normal(5.0, 2.0, n),
        "symmetry_score": rng.beta(8, 2, n),
        "color_variance": rng.normal(500, 200, n),
        "ip_risk_score": rng.beta(1.5, 12, n),
        "emulator_detected": rng.integers(0, 2, n).astype(float) * 0,
        "tor_detected": rng.integers(0, 2, n).astype(float) * 0,
        "vpn_detected": rng.integers(0, 2, n).astype(float) * 0,
        "repeated_attempts": rng.poisson(0.5, n).astype(float),
        "liveness_passed": np.ones(n).astype(float),
        "device_fingerprint_score": rng.beta(8, 2, n),
        "is_fraud": np.zeros(n).astype(int),
        "fraud_type": ["none"] * n,
    }

    df = pd.DataFrame(data)
    # Asegurar rangos válidos
    df["ocr_confidence"] = df["ocr_confidence"].clip(0, 1)
    df["blur_score"] = df["blur_score"].clip(10, 500)
    df["ela_score"] = df["ela_score"].clip(0, 100)
    df["symmetry_score"] = df["symmetry_score"].clip(0, 1)
    df["edge_density"] = df["edge_density"].clip(0, 1)
    df["brightness"] = df["brightness"].clip(0, 255)
    df["contrast"] = df["contrast"].clip(0, 200)
    df["noise_ratio"] = df["noise_ratio"].clip(0, 50)
    df["color_variance"] = df["color_variance"].clip(0, 5000)
    df["ip_risk_score"] = df["ip_risk_score"].clip(0, 1)
    df["device_fingerprint_score"] = df["device_fingerprint_score"].clip(0, 1)

    return df


def _generate_fraud_edited(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Fraude tipo edición de foto: ELA alto, ruido de edición."""
    data = {
        "blur_score": rng.normal(150, 50, n),
        "ela_score": rng.exponential(30, n) + 15,
        "ocr_confidence": rng.beta(5, 4, n),
        "ocr_field_count": rng.integers(5, 15, n).astype(float),
        "moire_score": rng.normal(3.0, 1.5, n),
        "dct_anomaly": rng.normal(0.7, 0.2, n),
        "reflection_score": rng.normal(2.5, 1.2, n),
        "edge_density": rng.beta(4, 20, n),
        "brightness": rng.normal(170, 35, n),
        "contrast": rng.normal(55, 18, n),
        "noise_ratio": rng.normal(10, 4, n),
        "symmetry_score": rng.beta(5, 4, n),
        "color_variance": rng.normal(800, 300, n),
        "ip_risk_score": rng.beta(3, 8, n),
        "emulator_detected": rng.integers(0, 2, n, p=[0.9, 0.1]).astype(float),
        "tor_detected": rng.integers(0, 2, n, p=[0.85, 0.15]).astype(float),
        "vpn_detected": rng.integers(0, 2, n, p=[0.8, 0.2]).astype(float),
        "repeated_attempts": rng.poisson(1.5, n).astype(float),
        "liveness_passed": (rng.random(n) > 0.3).astype(float),
        "device_fingerprint_score": rng.beta(4, 5, n),
        "is_fraud": np.ones(n).astype(int),
        "fraud_type": ["edited_photo"] * n,
    }

    df = pd.DataFrame(data)
    df["ocr_confidence"] = df["ocr_confidence"].clip(0, 1)
    df["blur_score"] = df["blur_score"].clip(10, 500)
    df["ela_score"] = df["ela_score"].clip(0, 150)
    df["symmetry_score"] = df["symmetry_score"].clip(0, 1)
    df["edge_density"] = df["edge_density"].clip(0, 1)
    df["brightness"] = df["brightness"].clip(0, 255)
    df["contrast"] = df["contrast"].clip(0, 200)
    df["noise_ratio"] = df["noise_ratio"].clip(0, 50)
    df["color_variance"] = df["color_variance"].clip(0, 5000)
    df["ip_risk_score"] = df["ip_risk_score"].clip(0, 1)
    df["device_fingerprint_score"] = df["device_fingerprint_score"].clip(0, 1)

    return df


def _generate_fraud_screen(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Fraude tipo screen capture: blur bajo, patrón de Moiré alto."""
    data = {
        "blur_score": rng.normal(80, 30, n),
        "ela_score": rng.exponential(15, n),
        "ocr_confidence": rng.beta(4, 5, n),
        "ocr_field_count": rng.integers(3, 12, n).astype(float),
        "moire_score": rng.normal(12.0, 4.0, n),
        "dct_anomaly": rng.normal(0.5, 0.25, n),
        "reflection_score": rng.normal(12.0, 5.0, n),
        "edge_density": rng.beta(3, 15, n),
        "brightness": rng.normal(140, 40, n),
        "contrast": rng.normal(45, 20, n),
        "noise_ratio": rng.normal(15, 5, n),
        "symmetry_score": rng.beta(3, 5, n),
        "color_variance": rng.normal(1200, 500, n),
        "ip_risk_score": rng.beta(4, 6, n),
        "emulator_detected": rng.integers(0, 2, n, p=[0.7, 0.3]).astype(float),
        "tor_detected": rng.integers(0, 2, n, p=[0.9, 0.1]).astype(float),
        "vpn_detected": rng.integers(0, 2, n, p=[0.75, 0.25]).astype(float),
        "repeated_attempts": rng.poisson(2.0, n).astype(float),
        "liveness_passed": (rng.random(n) > 0.5).astype(float),
        "device_fingerprint_score": rng.beta(3, 6, n),
        "is_fraud": np.ones(n).astype(int),
        "fraud_type": ["screen_capture"] * n,
    }

    df = pd.DataFrame(data)
    df["ocr_confidence"] = df["ocr_confidence"].clip(0, 1)
    df["blur_score"] = df["blur_score"].clip(10, 500)
    df["ela_score"] = df["ela_score"].clip(0, 150)
    df["symmetry_score"] = df["symmetry_score"].clip(0, 1)
    df["edge_density"] = df["edge_density"].clip(0, 1)
    df["brightness"] = df["brightness"].clip(0, 255)
    df["contrast"] = df["contrast"].clip(0, 200)
    df["noise_ratio"] = df["noise_ratio"].clip(0, 50)
    df["color_variance"] = df["color_variance"].clip(0, 5000)
    df["ip_risk_score"] = df["ip_risk_score"].clip(0, 1)
    df["device_fingerprint_score"] = df["device_fingerprint_score"].clip(0, 1)

    return df


def _generate_fraud_printed(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Fraude tipo documento impreso: características intermedias."""
    data = {
        "blur_score": rng.normal(100, 35, n),
        "ela_score": rng.exponential(20, n) + 5,
        "ocr_confidence": rng.beta(6, 3, n),
        "ocr_field_count": rng.integers(6, 16, n).astype(float),
        "moire_score": rng.normal(6.0, 2.5, n),
        "dct_anomaly": rng.normal(0.45, 0.2, n),
        "reflection_score": rng.normal(5.0, 2.5, n),
        "edge_density": rng.beta(3, 12, n),
        "brightness": rng.normal(160, 35, n),
        "contrast": rng.normal(50, 15, n),
        "noise_ratio": rng.normal(12, 4, n),
        "symmetry_score": rng.beta(6, 3, n),
        "color_variance": rng.normal(600, 250, n),
        "ip_risk_score": rng.beta(3, 7, n),
        "emulator_detected": rng.integers(0, 2, n, p=[0.85, 0.15]).astype(float),
        "tor_detected": rng.integers(0, 2, n, p=[0.88, 0.12]).astype(float),
        "vpn_detected": rng.integers(0, 2, n, p=[0.82, 0.18]).astype(float),
        "repeated_attempts": rng.poisson(1.2, n).astype(float),
        "liveness_passed": (rng.random(n) > 0.4).astype(float),
        "device_fingerprint_score": rng.beta(5, 4, n),
        "is_fraud": np.ones(n).astype(int),
        "fraud_type": ["printed_fake"] * n,
    }

    df = pd.DataFrame(data)
    df["ocr_confidence"] = df["ocr_confidence"].clip(0, 1)
    df["blur_score"] = df["blur_score"].clip(10, 500)
    df["ela_score"] = df["ela_score"].clip(0, 150)
    df["symmetry_score"] = df["symmetry_score"].clip(0, 1)
    df["edge_density"] = df["edge_density"].clip(0, 1)
    df["brightness"] = df["brightness"].clip(0, 255)
    df["contrast"] = df["contrast"].clip(0, 200)
    df["noise_ratio"] = df["noise_ratio"].clip(0, 50)
    df["color_variance"] = df["color_variance"].clip(0, 5000)
    df["ip_risk_score"] = df["ip_risk_score"].clip(0, 1)
    df["device_fingerprint_score"] = df["device_fingerprint_score"].clip(0, 1)

    return df


def _generate_fraud_stolen(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Fraude tipo documento real robado: pocas señales visuales, detectar por metadatos."""
    data = {
        "blur_score": rng.normal(170, 40, n),
        "ela_score": rng.exponential(10, n),
        "ocr_confidence": rng.beta(8, 2, n),
        "ocr_field_count": rng.integers(8, 18, n).astype(float),
        "moire_score": rng.normal(3.5, 1.5, n),
        "dct_anomaly": rng.normal(0.35, 0.15, n),
        "reflection_score": rng.normal(2.5, 1.0, n),
        "edge_density": rng.beta(5, 18, n),
        "brightness": rng.normal(175, 30, n),
        "contrast": rng.normal(58, 15, n),
        "noise_ratio": rng.normal(6, 2.5, n),
        "symmetry_score": rng.beta(7, 3, n),
        "color_variance": rng.normal(550, 220, n),
        "ip_risk_score": rng.beta(8, 3, n),
        "emulator_detected": rng.integers(0, 2, n, p=[0.6, 0.4]).astype(float),
        "tor_detected": rng.integers(0, 2, n, p=[0.65, 0.35]).astype(float),
        "vpn_detected": rng.integers(0, 2, n, p=[0.55, 0.45]).astype(float),
        "repeated_attempts": rng.poisson(3.0, n).astype(float),
        "liveness_passed": (rng.random(n) > 0.6).astype(float),
        "device_fingerprint_score": rng.beta(5, 5, n),
        "is_fraud": np.ones(n).astype(int),
        "fraud_type": ["stolen_real"] * n,
    }

    df = pd.DataFrame(data)
    df["ocr_confidence"] = df["ocr_confidence"].clip(0, 1)
    df["blur_score"] = df["blur_score"].clip(10, 500)
    df["ela_score"] = df["ela_score"].clip(0, 150)
    df["symmetry_score"] = df["symmetry_score"].clip(0, 1)
    df["edge_density"] = df["edge_density"].clip(0, 1)
    df["brightness"] = df["brightness"].clip(0, 255)
    df["contrast"] = df["contrast"].clip(0, 200)
    df["noise_ratio"] = df["noise_ratio"].clip(0, 50)
    df["color_variance"] = df["color_variance"].clip(0, 5000)
    df["ip_risk_score"] = df["ip_risk_score"].clip(0, 1)
    df["device_fingerprint_score"] = df["device_fingerprint_score"].clip(0, 1)

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = generate_fraud_dataset(n_legit=4000, n_fraud=600)

    # Guardar en directorio gold
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    output_path = GOLD_DIR / "synthetic_dataset.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"Dataset guardado en {output_path}")
    logger.info(f"Distribución:\n{df['is_fraud'].value_counts()}")
    logger.info(f"Tipos de fraude:\n{df['fraud_type'].value_counts()}")
