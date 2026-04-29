import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io

def render_ela_overlay(original_image, ela_image):
    """
    Renderiza el mapa ELA superpuesto.
    
    Args:
        original_image: Imagen original (PIL)
        ela_image: Imagen ELA (numpy array)
    """
    st.markdown("### Mapa ELA")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(original_image, caption="Original", use_column_width=True)
    
    with col2:
        st.image(ela_image, caption="ELA (diferencia)", use_column_width=True)

def render_feature_importance_chart(weights: dict):
    """
    Renderiza gráfico de barras con importancia de features.
    
    Args:
        weights: Diccionario con pesos de features
    """
    st.markdown("### Importancia de Features (Pesos)")
    import pandas as pd
    
    df = pd.DataFrame(list(weights.items()), columns=["Feature", "Peso"])
    st.bar_chart(df.set_index("Feature"))

def render_gauge(score: float, threshold: float = 35.0):
    """
    Renderiza un gauge visual del score de fraude.
    
    Args:
        score: Score de fraude (0-100)
        threshold: Umbral de fraude
    """
    st.markdown("### Score de Fraude")
    
    if score >= 70:
        color = "red"
        emoji = "🔴"
    elif score >= threshold:
        color = "orange"
        emoji = "🟡"
    else:
        color = "green"
        emoji = "🟢"
    
    st.markdown(f"## {emoji} {score:.1f}/100")
    
    import plotly.graph_objects as go
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Score de Fraude"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, threshold], 'color': "lightgreen"},
                {'range': [threshold, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "red"}
            ]
        }
    ))
    
    st.plotly_chart(fig, use_container_width=True)
