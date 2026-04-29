"""
Módulo de ensamblado del dataset Gold.

Combina features de diferentes fuentes (visuales, OCR, metadatos)
en un dataset tabular final listo para entrenamiento.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

SILVER_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "silver"
GOLD_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "gold"


def assemble_gold_dataset(
    visual_features_path: Optional[Path] = None,
    ocr_features_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
    synthetic: bool = True,
) -> pd.DataFrame:
    """
    Ensambla el dataset Gold combinando diferentes fuentes de features.

    Args:
        visual_features_path: Ruta al archivo de features visuales.
        ocr_features_path: Ruta al archivo de features OCR.
        metadata_path: Ruta al archivo de metadatos.
        synthetic: Si usar datos sintéticos como fallback.

    Returns:
        DataFrame Gold ensamblado.
    """
    if synthetic:
        from src.dataset.generator import generate_fraud_dataset

        logger.info("Generando dataset sintético para Gold layer")
        df = generate_fraud_dataset(n_legit=4000, n_fraud=600)

        # Aplicar etiquetado heurístico
        from src.dataset.labeler import apply_heuristic_labels

        df = apply_heuristic_labels(df)

        # Guardar en Gold layer
        GOLD_DIR.mkdir(parents=True, exist_ok=True)
        output_path = GOLD_DIR / "gold_dataset.parquet"
        df.to_parquet(output_path, index=False)
        logger.info(f"Dataset Gold guardado en {output_path}")

        return df

    # Enfoque con archivos reales (cuando se tengan datos reales)
    frames = []

    if visual_features_path and visual_features_path.exists():
        visual_df = pd.read_parquet(visual_features_path)
        frames.append(visual_df)
        logger.info(f"Features visuales cargadas: {len(visual_df)} registros")

    if ocr_features_path and ocr_features_path.exists():
        ocr_df = pd.read_parquet(ocr_features_path)
        frames.append(ocr_df)
        logger.info(f"Features OCR cargadas: {len(ocr_df)} registros")

    if metadata_path and metadata_path.exists():
        meta_df = pd.read_parquet(metadata_path)
        frames.append(meta_df)
        logger.info(f"Metadatos cargados: {len(meta_df)} registros")

    if not frames:
        raise FileNotFoundError("No se encontraron archivos de features")

    # Combinar todas las features
    df = pd.concat(frames, axis=1)
    logger.info(f"Dataset Gold ensamblado: {df.shape}")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = assemble_gold_dataset(synthetic=True)
    print(f"Dataset Gold: {df.shape}")
    print(f"Distribución de clases:\n{df['is_fraud'].value_counts()}")
