"""
DocShield — Herramienta de análisis de fraude para analistas.

Interfaz web construida con Streamlit que permite:
1. Subir imagen de DNI (solo en entorno analista)
2. Ver score de fraude con gauge visual
3. Ver señales detectadas con peso relativo
4. Ver mapa ELA superpuesto
5. Ver features numéricas en tabla
6. Exportar reporte en PDF
"""

import base64
import io
import logging
import tempfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from src.pipeline.anti_spoofing import (
    analyze_dct_blocks,
    analyze_reflection,
    detect_moire,
)
from src.pipeline.visual_extractor import compute_ela, extract_visual_features
from src.api.anti_spoofing_api import (
    compute_fraud_score,
    FRAUD_THRESHOLD,
    WEIGHTS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="DocShield — Fraud Analysis Tool",
    page_icon="🛡️",
    layout="wide",
)


def main():
    st.title("🛡️ DocShield — Herramienta de Análisis de Fraude")
    st.caption("Sistema de detección de fraude documental para LATAM fintech")

    # Sidebar
    with st.sidebar:
        st.header("Configuración")
        threshold = st.slider(
            "Umbral de fraude",
            min_value=10.0,
            max_value=80.0,
            value=FRAUD_THRESHOLD,
            step=1.0,
        )

        st.markdown("---")
        st.markdown("**Pesos del modelo:**")
        for feature, weight in WEIGHTS.items():
            st.text(f"  {feature}: {weight:.2f}")

    # Upload
    uploaded_file = st.file_uploader(
        "Subir imagen de documento (DNI, pasaporte, cédula)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        st.info("Subí una imagen para analizar")
        show_sample_analysis()
        return

    # Process image
    image = Image.open(uploaded_file)
    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Imagen del documento")
        st.image(image, use_container_width=True)

    # Extract features
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name, "JPEG")
        tmp_path = tmp.name

        visual_features = extract_visual_features(tmp_path)
        ela_score = compute_ela(tmp_path)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    moire_score = detect_moire(gray)
    dct_anomaly = analyze_dct_blocks(gray)
    reflection_score = analyze_reflection(bgr)

    # Combine all features
    all_features = {
        **visual_features,
        "ela_score": ela_score,
        "moire_score": moire_score,
        "dct_anomaly": dct_anomaly,
        "reflection_score": reflection_score,
        "ip_risk_score": 0.1,
        "emulator_detected": 0,
        "tor_detected": 0,
        "vpn_detected": 0,
        "repeated_attempts": 0,
        "liveness_passed": 1,
        "device_fingerprint_score": 0.8,
    }

    # Compute fraud score
    fraud_score, signals = compute_fraud_score(all_features, liveness_passed=True)
    is_fraud = fraud_score > threshold

    # Clean up
    del tmp_path
    Path(tmp.name).unlink(missing_ok=True)

    with col2:
        st.subheader("Resultado del análisis")

        # Gauge
        gauge_color = "green" if fraud_score < threshold else ("orange" if fraud_score < 60 else "red")
        gauge_emoji = "🟢" if fraud_score < threshold else ("🟡" if fraud_score < 60 else "🔴")

        st.metric(
            "Score de fraude",
            f"{fraud_score:.0f}/100 {gauge_emoji}",
            delta=None,
        )

        if is_fraud:
            st.error(f"⚠️ Documento sospechoso (score > {threshold})")
        else:
            st.success(f"✅ Documento aparentemente legítimo (score < {threshold})")

        # Signals
        st.subheader("Señales detectadas")
        if signals:
            for signal in signals:
                st.warning(f"• {signal}")
        else:
            st.info("No se detectaron señales de fraude")

    # Feature importance
    st.subheader("Contribución de cada feature al score")
    fig = create_feature_contribution_chart(all_features)
    st.plotly_chart(fig, use_container_width=True)

    # ELA overlay
    st.subheader("Mapa ELA (Error Level Analysis)")
    ela_fig = create_ela_overlay(image, ela_score)
    st.pyplot(ela_fig)

    # Feature table
    st.subheader("Features numéricas")
    feature_df = pd.DataFrame({
        "Feature": list(all_features.keys()),
        "Valor": [f"{v:.4f}" if isinstance(v, float) else v for v in all_features.values()],
    })
    st.dataframe(feature_df, use_container_width=True, hide_index=True)

    # Export
    if st.button("Exportar reporte PDF"):
        export_report(fraud_score, signals, all_features, is_fraud)


def show_sample_analysis():
    """Muestra un análisis de ejemplo con datos ficticios."""
    st.subheader("Datos de ejemplo")

    sample_data = {
        "blur_score": 180.5,
        "ela_score": 12.3,
        "ocr_confidence": 0.92,
        "moire_score": 3.2,
        "dct_anomaly": 0.25,
        "reflection_score": 2.1,
    }

    st.dataframe(pd.DataFrame({
        "Feature": list(sample_data.keys()),
        "Valor": [f"{v:.4f}" for v in sample_data.values()],
    }), use_container_width=True, hide_index=True)

    st.info("Subí una imagen para ver el análisis completo")


def create_feature_contribution_chart(features: dict) -> go.Figure:
    """Crea un gráfico de barras con la contribución de cada feature."""
    weights = WEIGHTS
    contributions = []
    labels = []

    ela_norm = min(features.get("ela_score", 0) / 100.0, 1.0)
    moire_norm = min(features.get("moire_score", 0) / 20.0, 1.0)
    dct_norm = min(features.get("dct_anomaly", 0) / 1.0, 1.0)
    blur_norm = 1.0 - min(features.get("blur_score", 180) / 200.0, 1.0)
    ocr_norm = 1.0 - min(features.get("ocr_confidence", 0.8), 1.0)
    reflection_norm = min(features.get("reflection_score", 0) / 20.0, 1.0)

    contributions = [
        weights["ela"] * ela_norm * 100,
        weights["moire"] * moire_norm * 100,
        weights["dct"] * dct_norm * 100,
        weights["blur"] * blur_norm * 100,
        weights["ocr"] * ocr_norm * 100,
        weights["reflection"] * reflection_norm * 100,
    ]

    labels = ["ELA", "Moiré", "DCT", "Blur", "OCR", "Reflexión"]

    colors = ["red" if c > 10 else "orange" if c > 5 else "green" for c in contributions]

    fig = go.Figure(go.Bar(
        x=labels,
        y=contributions,
        marker_color=colors,
        text=[f"{c:.1f}" for c in contributions],
        textposition="auto",
    ))

    fig.update_layout(
        title="Contribución de cada feature al score de fraude",
        yaxis_title="Puntos de score",
        showlegend=False,
    )

    return fig


def create_ela_overlay(image: Image.Image, ela_score: float) -> plt.Figure:
    """Crea una visualización del mapa ELA superpuesto."""
    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Aplicar Gaussian Blur para simular recompresión
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Diferencia entre original y blur
    diff = cv2.absdiff(gray, blurred)

    # Normalizar para visualización
    diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    ax1.set_title("Imagen original")
    ax1.axis("off")

    ax2.imshow(diff_norm, cmap="hot")
    ax2.set_title(f"ELA overlay (score: {ela_score:.1f})")
    ax2.axis("off")

    plt.tight_layout()
    return fig


def export_report(
    fraud_score: float,
    signals: list[str],
    features: dict,
    is_fraud: bool,
):
    """Exporta un reporte PDF del análisis."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "DocShield — Reporte de Análisis", ln=True, align="C")

    pdf.ln(5)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, f"Score de fraude: {fraud_score:.1f}/100", ln=True)
    pdf.cell(0, 10, f"Resultado: {'FRAUDE DETECTADO' if is_fraud else 'LEGÍTIMO'}", ln=True)

    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Señales detectadas:", ln=True)

    pdf.set_font("Helvetica", "", 10)
    for signal in signals:
        pdf.cell(0, 8, f"  - {signal}", ln=True)

    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Features numéricas:", ln=True)

    pdf.set_font("Helvetica", "", 9)
    for key, value in features.items():
        pdf.cell(0, 6, f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}", ln=True)

    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(0, 6, "Este reporte es generado automáticamente por DocShield.", ln=True, align="C")
    pdf.cell(0, 6, "No reemplaza la revisión humana por un analista.", ln=True, align="C")

    pdf.output("docshield_report.pdf")
    st.success("Reporte exportado: docshield_report.pdf")


if __name__ == "__main__":
    main()
