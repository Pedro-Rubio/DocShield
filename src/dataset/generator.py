import numpy as np
import pandas as pd
from faker import Faker
from typing import Dict, List, Tuple

fake = Faker()

def _generate_legit_samples(n: int) -> List[Dict]:
    """Genera muestras de documentos legítimos."""
    samples = []
    for _ in range(n):
        sample = {
            "blur_score": np.random.normal(180, 40),
            "edge_density": np.random.uniform(0.05, 0.25),
            "brightness": np.random.normal(160, 30),
            "contrast": np.random.normal(50, 15),
            "noise_ratio": np.random.exponential(2),
            "symmetry_score": np.random.uniform(0.7, 0.95),
            "color_variance": np.random.normal(500, 150),
            "ela_score": np.random.exponential(8),
            "moire_score": np.random.exponential(1),
            "dct_score": np.random.normal(0.5, 0.1),
            "reflection_score": np.random.exponential(0.5),
            "ocr_confidence": np.random.beta(9, 1.5),
            "ip_risk_score": np.random.beta(1.5, 12),
            "emulator_detected": 0,
            "tor_detected": 0,
            "vpn_detected": 0,
            "repeated_attempts": np.random.randint(0, 2),
            "liveness_passed": 1,
            "fraud_type": "legit"
        }
        samples.append(sample)
    return samples

def _generate_edited_photo_samples(n: int) -> List[Dict]:
    """Genera muestras de fotos editadas (alta ELA, ruido)."""
    samples = []
    for _ in range(n):
        sample = {
            "blur_score": np.random.normal(150, 30),
            "edge_density": np.random.uniform(0.08, 0.3),
            "brightness": np.random.normal(140, 40),
            "contrast": np.random.normal(60, 20),
            "noise_ratio": np.random.exponential(5),
            "symmetry_score": np.random.uniform(0.6, 0.9),
            "color_variance": np.random.normal(600, 200),
            "ela_score": np.random.exponential(30) + 20,
            "moire_score": np.random.exponential(1.5),
            "dct_score": np.random.normal(0.7, 0.15),
            "reflection_score": np.random.exponential(1),
            "ocr_confidence": np.random.beta(7, 3),
            "ip_risk_score": np.random.beta(2, 10),
            "emulator_detected": 0,
            "tor_detected": np.random.choice([0, 1], p=[0.9, 0.1]),
            "vpn_detected": np.random.choice([0, 1], p=[0.8, 0.2]),
            "repeated_attempts": np.random.randint(0, 4),
            "liveness_passed": np.random.choice([0, 1], p=[0.3, 0.7]),
            "fraud_type": "edited_photo"
        }
        samples.append(sample)
    return samples

def _generate_screen_capture_samples(n: int) -> List[Dict]:
    """Genera muestras de capturas de pantalla (bajo blur, patrón Moiré)."""
    samples = []
    for _ in range(n):
        sample = {
            "blur_score": np.random.normal(80, 20),
            "edge_density": np.random.uniform(0.1, 0.35),
            "brightness": np.random.normal(120, 30),
            "contrast": np.random.normal(70, 25),
            "noise_ratio": np.random.exponential(1.5),
            "symmetry_score": np.random.uniform(0.65, 0.95),
            "color_variance": np.random.normal(400, 100),
            "ela_score": np.random.exponential(5),
            "moire_score": np.random.exponential(15) + 8,
            "dct_score": np.random.normal(0.6, 0.12),
            "reflection_score": np.random.exponential(3) + 1,
            "ocr_confidence": np.random.beta(8, 2),
            "ip_risk_score": np.random.beta(3, 8),
            "emulator_detected": np.random.choice([0, 1], p=[0.7, 0.3]),
            "tor_detected": np.random.choice([0, 1], p=[0.85, 0.15]),
            "vpn_detected": np.random.choice([0, 1], p=[0.75, 0.25]),
            "repeated_attempts": np.random.randint(0, 5),
            "liveness_passed": np.random.choice([0, 1], p=[0.6, 0.4]),
            "fraud_type": "screen_capture"
        }
        samples.append(sample)
    return samples

def _generate_printed_fake_samples(n: int) -> List[Dict]:
    """Genera muestras de documentos impresos falsos (características intermedias)."""
    samples = []
    for _ in range(n):
        sample = {
            "blur_score": np.random.normal(120, 25),
            "edge_density": np.random.uniform(0.06, 0.28),
            "brightness": np.random.normal(150, 35),
            "contrast": np.random.normal(55, 18),
            "noise_ratio": np.random.exponential(3),
            "symmetry_score": np.random.uniform(0.6, 0.88),
            "color_variance": np.random.normal(550, 180),
            "ela_score": np.random.exponential(15) + 10,
            "moire_score": np.random.exponential(3),
            "dct_score": np.random.normal(0.55, 0.1),
            "reflection_score": np.random.exponential(1.5),
            "ocr_confidence": np.random.beta(7, 2.5),
            "ip_risk_score": np.random.beta(2.5, 10),
            "emulator_detected": np.random.choice([0, 1], p=[0.8, 0.2]),
            "tor_detected": np.random.choice([0, 1], p=[0.9, 0.1]),
            "vpn_detected": np.random.choice([0, 1], p=[0.8, 0.2]),
            "repeated_attempts": np.random.randint(0, 4),
            "liveness_passed": np.random.choice([0, 1], p=[0.4, 0.6]),
            "fraud_type": "printed_fake"
        }
        samples.append(sample)
    return samples

def _generate_stolen_real_samples(n: int) -> List[Dict]:
    """Genera muestras de documentos reales robados (difícil de detectar visualmente)."""
    samples = []
    for _ in range(n):
        sample = {
            "blur_score": np.random.normal(175, 35),
            "edge_density": np.random.uniform(0.05, 0.22),
            "brightness": np.random.normal(155, 28),
            "contrast": np.random.normal(52, 14),
            "noise_ratio": np.random.exponential(2.2),
            "symmetry_score": np.random.uniform(0.68, 0.93),
            "color_variance": np.random.normal(520, 140),
            "ela_score": np.random.exponential(9),
            "moire_score": np.random.exponential(1.2),
            "dct_score": np.random.normal(0.52, 0.08),
            "reflection_score": np.random.exponential(0.6),
            "ocr_confidence": np.random.beta(8.5, 1.8),
            "ip_risk_score": np.random.beta(4, 6),
            "emulator_detected": np.random.choice([0, 1], p=[0.6, 0.4]),
            "tor_detected": np.random.choice([0, 1], p=[0.7, 0.3]),
            "vpn_detected": np.random.choice([0, 1], p=[0.6, 0.4]),
            "repeated_attempts": np.random.randint(2, 6),
            "liveness_passed": 0,
            "fraud_type": "stolen_real"
        }
        samples.append(sample)
    return samples

def generate_fraud_dataset(n_legit: int = 4000, n_fraud: int = 600) -> pd.DataFrame:
    """
    Generar dataset sintético con distribuciones calibradas.

    Args:
        n_legit: Número de muestras legítimas.
        n_fraud: Número total de muestras fraudulentas (se dividen equitativamente entre 4 subtipos).

    Returns:
        DataFrame con el dataset sintético.
    """
    samples = []
    
    # Generar muestras legítimas
    samples.extend(_generate_legit_samples(n_legit))
    
    # Generar muestras fraudulentas (divididas equitativamente entre 4 subtipos)
    n_per_type = n_fraud // 4
    remainder = n_fraud % 4
    
    samples.extend(_generate_edited_photo_samples(n_per_type + (1 if remainder > 0 else 0)))
    samples.extend(_generate_screen_capture_samples(n_per_type + (1 if remainder > 1 else 0)))
    samples.extend(_generate_printed_fake_samples(n_per_type + (1 if remainder > 2 else 0)))
    samples.extend(_generate_stolen_real_samples(n_per_type))
    
    df = pd.DataFrame(samples)
    
    # Asegurar que los valores estén en rangos válidos
    df["blur_score"] = df["blur_score"].clip(lower=0)
    df["brightness"] = df["brightness"].clip(lower=0, upper=255)
    df["contrast"] = df["contrast"].clip(lower=0)
    df["noise_ratio"] = df["noise_ratio"].clip(lower=0)
    df["symmetry_score"] = df["symmetry_score"].clip(lower=0, upper=1)
    df["color_variance"] = df["color_variance"].clip(lower=0)
    df["ela_score"] = df["ela_score"].clip(lower=0)
    df["moire_score"] = df["moire_score"].clip(lower=0)
    df["dct_score"] = df["dct_score"].clip(lower=0)
    df["reflection_score"] = df["reflection_score"].clip(lower=0)
    df["ocr_confidence"] = df["ocr_confidence"].clip(lower=0, upper=1)
    df["ip_risk_score"] = df["ip_risk_score"].clip(lower=0, upper=1)
    
    return df
