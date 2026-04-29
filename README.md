# DocShield

Open source document fraud detection for LATAM fintech

## Demo

<!-- TODO: agregar gif o screenshot de la interfaz Streamlit -->

## Quick Start

```bash
git clone https://github.com/tu-org/docshield.git
cd docshield
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```

## Architecture

```
Bronze Layer  → imágenes crudas + metadatos EXIF (solo durante dev)
Silver Layer  → features numéricas extraídas (blur, ELA, OCR confidence...)
Gold Layer    → dataset tabular final para entrenamiento (.parquet)
```

## Features detectadas

| Señal | Descripción | Tipo |
|-------|-------------|------|
| blur_score | Varianza del Laplaciano | Visual |
| edge_density | Densidad de bordes Canny | Visual |
| brightness | Brillo medio del canal gris | Visual |
| contrast | Contraste (std del canal gris) | Visual |
| noise_ratio | Relación de ruido vs GaussianBlur | Visual |
| symmetry_score | Correlación izquierda/derecha | Visual |
| color_variance | Varianza entre canales RGB | Visual |
| ela_score | Error Level Analysis (diferencia de recompresión) | Forense |
| moire_score | Detección de patrón Moiré vía FFT | Anti-spoofing |
| dct_score | Análisis de bloques DCT 8x8 | Anti-spoofing |
| reflection_score | Análisis de reflexión especular | Anti-spoofing |
| ocr_confidence | Confianza del OCR (EasyOCR/Tesseract) | OCR |
| ip_risk_score | Riesgo de IP (GeoIP, Tor, VPN) | Metadatos |
| emulator_detected | Detección de emulador Android | Metadatos |
| tor_detected | Uso de red Tor | Metadatos |
| vpn_detected | Uso de VPN | Metadatos |
| repeated_attempts | Intentos repetidos en sesión | Metadatos |
| liveness_passed | Verificación de liveness activa | Liveness |

## Legal & Ethics

- Las imágenes se procesan en memoria y nunca se almacenan.
- El usuario es responsable de obtener consentimiento de los titulares de los documentos.
- Esta herramienta no reemplaza la revisión humana.
- Cumplimiento con Ley 25.326 (Argentina), LGPD (Brasil) y regulaciones locales de privacidad.

## Roadmap

- **Fase 1** — Base del proyecto y pipeline de datos
- **Fase 2** — Dataset sintético y entrenamiento de modelo baseline
- **Fase 3** — API FastAPI y endpoint de verificación
- **Fase 4** — Interfaz Streamlit para analistas y componente React Native
- **Fase 5** — Pruebas de carga, monitoreo y MLOps con MLflow

## Contributing

1. Fork el repositorio
2. Crea una branch para tu feature (`git checkout -b feature/nueva-feature`)
3. Commit tus cambios (`git commit -m 'Agrega nueva feature'`)
4. Push a la branch (`git push origin feature/nueva-feature`)
5. Abre un Pull Request

## License

MIT

**Disclaimer:** DocShield se proporciona «tal cual», sin garantía de ningún tipo. El uso de esta herramienta para verificación de identidad es responsabilidad exclusiva del usuario. Los autores no se hacen responsables por falsos positivos, falsos negativos o cualquier daño derivado de su uso.
