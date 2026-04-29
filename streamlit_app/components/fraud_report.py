import streamlit as st
import pandas as pd

def render_fraud_report(fraud_score: float, signals: list, processing_ms: int):
    """
    Renderiza el reporte de fraude en Streamlit.
    
    Args:
        fraud_score: Score de fraude (0-100)
        signals: Lista de señales detectadas
        processing_ms: Tiempo de procesamiento en ms
    """
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if fraud_score >= 70:
            st.error(f"🔴 Score: {fraud_score:.1f}/100")
        elif fraud_score >= 35:
            st.warning(f"🟡 Score: {fraud_score:.1f}/100")
        else:
            st.success(f"🟢 Score: {fraud_score:.1f}/100")
    
    with col2:
        st.metric("Señales detectadas", len(signals))
    
    with col3:
        st.metric("Procesamiento", f"{processing_ms} ms")
    
    if signals:
        st.markdown("### Señales detectadas:")
        for signal in signals:
            st.markdown(f"- ⚠️ {signal}")
    else:
        st.markdown("### ✅ No se detectaron señales de fraude")

def render_feature_table(features: dict):
    """
    Renderiza tabla de features en Streamlit.
    
    Args:
        features: Diccionario de features extraídas
    """
    st.markdown("### Features Numéricas")
    df = pd.DataFrame([features]).T
    df.columns = ["Valor"]
    st.dataframe(df.style.format("{:.4f}"), use_container_width=True)
