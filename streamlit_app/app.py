import streamlit as st
import numpy as np
import cv2
import base64
import io
import os
import sys
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import time

sys.path.append('..')

from src.pipeline.risk_engine import calculate_risk_score
from src.pipeline.anti_spoofing import detect_moire, analyze_dct_blocks, analyze_reflection
from src.pipeline.ocr_extractor import extract_ocr_features
from src.dataset.labeler import get_fraud_signals

st.set_page_config(page_title="DocShield — Fraud Analysis Tool", layout="wide")

st.title("DocShield — Fraud Analysis Tool")
st.markdown("Herramienta para analistas: subí una imagen de documento para evaluar su riesgo de fraude.")

with st.sidebar:
    st.header("Configuración")
    analyst_mode = st.checkbox("Modo analista", value=True)
    fraud_threshold = st.slider("Umbral de fraude", 0.0, 100.0, 35.0)
    st.markdown("---")
    st.markdown("**Leyenda:**")
    st.markdown("- 🟢 Bajo riesgo (< 35)")
    st.markdown("- 🟡 Riesgo medio (35-70)")
    st.markdown("- 🔴 Alto riesgo (> 70)")

uploaded_file = st.file_uploader("Subí una imagen de documento (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Documento")
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_column_width=True)
        
        # Usar archivo temporal único para evitar condiciones de carrera
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        image.save(temp_path)
    
    with col2:
        start_time = time.time()
        
        with st.spinner("Extrayendo features..."):
            try:
                visual_feats = extract_visual_features(temp_path)
                ela_score = compute_ela(temp_path)
                
                bgr = cv2.imread(temp_path)
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                
                moire_score = detect_moire(gray)
                dct_score = analyze_dct_blocks(gray)
                reflection_score = analyze_reflection(bgr)
                
                try:
                    ocr_feats = extract_ocr_features(temp_path)
                except Exception:
                    ocr_feats = {"ocr_confidence": 0.0}
                
                all_features = {**visual_feats, "ela_score": ela_score, "moire_score": moire_score,
                              "dct_score": dct_score, "reflection_score": reflection_score,
                              **ocr_feats}
                
                del bgr, gray
                
            except Exception as e:
                st.error(f"Error procesando imagen: {e}")
                st.stop()
        
        processing_ms = int((time.time() - start_time) * 1000)
        
        from src.api.anti_spoofing_api import WEIGHTS, calculate_fraud_score
        fraud_score, signals = calculate_fraud_score(all_features, {})
        
        if fraud_score >= 70:
            risk_emoji = "🔴"
            risk_color = "red"
        elif fraud_score >= 35:
            risk_emoji = "🟡"
            risk_color = "orange"
        else:
            risk_emoji = "🟢"
            risk_color = "green"
        
        st.subheader("Resultado")
        st.metric("Score de Fraude", f"{fraud_score:.1f}/100", delta=None)
        st.markdown(f"**Nivel de riesgo:** {risk_emoji} {risk_color.upper()}")
        st.markdown(f"**Tiempo de procesamiento:** {processing_ms} ms")
        
        if signals:
            st.markdown("**Señales detectadas:**")
            for sig in signals:
                st.markdown(f"- ⚠️ {sig}")
        else:
            st.markdown("✅ No se detectaron señales de fraude")
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["Features Numéricas", "Mapa ELA", "Importancia de Features"])
    
    with tab1:
        st.subheader("Features Extraídas")
        feat_df = pd.DataFrame([all_features]).T
        feat_df.columns = ["Valor"]
        st.dataframe(feat_df.style.format("{:.4f}"), use_container_width=True)
    
    with tab2:
        st.subheader("Error Level Analysis (ELA)")
        try:
            orig = np.array(Image.open(temp_path).convert('RGB'), dtype=np.int16)
            buffer = io.BytesIO()
            Image.fromarray(orig.astype(np.uint8)).save(buffer, format='JPEG', quality=90)
            buffer.seek(0)
            compressed = np.array(Image.open(buffer).convert('RGB'), dtype=np.int16)
            ela_img = np.abs(orig - compressed).astype(np.uint8)
            
            st.image(ela_img, caption="Mapa ELA (diferencia con recompresión)", use_column_width=True)
        except Exception as e:
            st.error(f"Error generando ELA: {e}")
    
    with tab3:
        st.subheader("Pesos del Modelo")
        weight_df = pd.DataFrame(list(WEIGHTS.items()), columns=["Feature", "Peso"])
        st.bar_chart(weight_df.set_index("Feature"))
    
    if uploaded_file is not None and 'temp_path' in locals() and os.path.exists(temp_path):
        os.unlink(temp_path)
else:
    st.info("Subí una imagen para comenzar el análisis.")

st.markdown("---")
st.caption("DocShield v1.0.0 — Las imágenes se procesan en memoria y nunca se almacenan.")
