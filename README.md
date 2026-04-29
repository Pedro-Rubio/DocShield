# DocShield

**Open source document fraud detection for LATAM fintech**

Sistema multimodal de detección de fraude en documentos de identidad (DNI, pasaporte, cédula) para procesos de onboarding digital. Diseñado para fintechs de 5-50 personas en LATAM que necesitan una solución que corra on-premise sin depender de proveedores enterprise costosos.

## Demo

```
┌──────────────────────────────────────────────────┐
│  DocShield — Fraud Analysis Tool                 │
├─────────────────────┬────────────────────────────┤
│  [Subir imagen]     │  Score: 72/100 🔴          │
│  [Preview doc]      │  Señales detectadas: 3     │
│                     │  ├─ ELA anomalía alta      │
│                     │  ├─ Posible screen capture │
│                     │  └─ OCR confidence baja    │
├─────────────────────┴────────────────────────────┤
│  Feature importance (SHAP bar chart)             │
│  ELA overlay map                                 │
└──────────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Generar dataset sintético y entrenar modelo
python -m src.dataset.generator
python -m src.model.trainer

# 3. Iniciar API
uvicorn src.api.main:app --reload
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      MOBILE APP                              │
│  react-native-vision-camera + expo-sensors (liveness)        │
└──────────────────────┬──────────────────────────────────────┘
                       │ POST /api/v1/verify-document
                       │ (base64 + capture_meta)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                       FastAPI API                            │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐ │
│  │  Visual     │  │  Anti-       │  │  OCR                │ │
│  │  Extractor  │  │  Spoofing    │  │  Extractor          │ │
│  │  (OpenCV)   │  │  (FFT/DCT)   │  │  (EasyOCR/Tesseract)│ │
│  └──────┬──────┘  └──────┬───────┘  └──────────┬──────────┘ │
│         └────────────────┼─────────────────────┘            │
│                          ▼                                  │
│              ┌───────────────────────┐                      │
│              │  Feature Vector       │                      │
│              │  (memory only — DEL)  │                      │
│              └──────────┬────────────┘                      │
│                         ▼                                   │
│              ┌───────────────────────┐                      │
│              │  XGBoost / LightGBM   │                      │
│              │  Fraud Score 0-100    │                      │
│              └──────────┬────────────┘                      │
│                         ▼                                   │
│              ┌───────────────────────┐                      │
│              │  SHAP Explanation     │                      │
│              │  + Human Signals      │                      │
│              └───────────────────────┘                      │
└─────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Medallion Data Layer                      │
│                                                              │
│  Bronze Layer  →  imágenes crudas + EXIF (solo en dev)      │
│       │                                                      │
│       ▼                                                      │
│  Silver Layer  →  features numéricas extraídas               │
│       │                                                      │
│       ▼                                                      │
│  Gold Layer    →  dataset tabular final (.parquet)           │
└─────────────────────────────────────────────────────────────┘
```

## Features Detectadas

| Feature | Descripción | Método |
|---------|-------------|--------|
| `blur_score` | Nitidez de la imagen | Varianza del Laplaciano |
| `edge_density` | Densidad de bordes | Canny edge detection |
| `brightness` | Brillo promedio | Media del canal gris |
| `contrast` | Contraste de la imagen | Std del canal gris |
| `noise_ratio` | Ratio de ruido | Diferencia con GaussianBlur |
| `symmetry_score` | Simetría del documento | Correlación izq/der |
| `color_variance` | Varianza entre canales | Varianza RGB |
| `ela_score` | Error Level Analysis | Recompr JPEG + diff |
| `moire_score` | Patrón de Moiré (screen replay) | FFT 2D + ratio pico |
| `dct_anomaly` | Inconsistencias DCT | Energía AC bloques 8x8 |
| `reflection_score` | Reflexión especular | Gradiente en zonas brillantes |
| `ocr_confidence` | Confianza del OCR | EasyOCR / Tesseract |
| `ocr_field_count` | Campos de texto detectados | Conteo de regiones OCR |
| `ip_risk_score` | Riesgo de la IP | Geolocalización + proxy |
| `device_fingerprint` | Huella del dispositivo | User-Agent + resolución |
| `emulator_detected` | Detección de emulador | Sensores + hardware info |
| `liveness_passed` | Prueba de liveness | Acelerómetro + rotación |
| `repeated_attempts` | Intentos repetidos | Conteo de sesión |

## Legal & Ethics

### Privacidad
- **Las imágenes se procesan exclusivamente en memoria y nunca se almacenan en disco ni en bases de datos.**
- Solo se persisten los vectores numéricos de features extraídas (valores float, sin datos visuales recuperables).
- Este principio de Privacy-Preserving ML es la base de toda la arquitectura.

### Responsabilidad del usuario
- **El usuario es responsable de obtener consentimiento informado de los titulares de los documentos** antes de procesarlos.
- Esta herramienta no reemplaza la revisión humana por analistas capacitados.
- Las decisiones automatizadas deben ser revisables y apelables.

### Cumplimiento normativo
- **Ley 25.326** (Argentina) — Protección de datos personales
- **LGPD** (Brasil) — Lei Geral de Proteção de Dados
- **Regulaciones locales** de cada país de LATAM aplicables a fintech y KYC

### Licencia
- Licenciado bajo MIT License con el siguiente disclaimer:
  > THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND. THE AUTHORS SHALL NOT BE LIABLE FOR ANY DAMAGES ARISING FROM THE USE OF THIS SOFTWARE, INCLUDING BUT NOT LIMITED TO FALSE POSITIVES/NEGATIVES IN FRAUD DETECTION. USERS ARE RESPONSIBLE FOR COMPLIANCE WITH APPLICABLE LAWS AND REGULATIONS.

## Roadmap

### Fase 1 — MVP (actual)
- [x] Pipeline de extracción de features forenses
- [x] Dataset sintético para entrenamiento
- [x] Modelo baseline XGBoost/LightGBM
- [x] API REST FastAPI
- [x] Herramienta de análisis Streamlit

### Fase 2 — Mejora del modelo
- [ ] Integración con datos reales (con consentimiento)
- [ ] Fine-tuning con transfer learning
- [ ] Detección específica por tipo de documento (DNI AR, DNI CL, etc.)
- [ ] Modelo deep learning para features visuales (EfficientNet)

### Fase 3 — Producción
- [ ] Docker + Kubernetes deployment
- [ ] Rate limiting avanzado + WAF
- [ ] Dashboard de métricas en tiempo real
- [ ] A/B testing de modelos

### Fase 4 — Expansión regional
- [ ] Soporte para documentos de 15+ países LATAM
- [ ] Templates de validación por país
- [ ] Comunidad de contribuidores regionales

## Contributing

1. Fork el repositorio
2. Crear una rama feature (`git checkout -b feature/nueva-feature`)
3. Commit de cambios (`git commit -m 'feat: agregar nueva feature'`)
4. Push a la rama (`git push origin feature/nueva-feature`)
5. Abrir un Pull Request

### Setup de desarrollo

```bash
python -m venv .venv
source .venv/bin/activate  # o .venv\Scripts\activate en Windows
pip install -r requirements.txt
pip install -r requirements-dev.txt
pytest tests/ -v
```

### Convenciones
- Type hints en todas las funciones públicas
- Docstrings en español
- Tests para todas las funciones del pipeline
- Commits siguiendo Conventional Commits

## License

MIT License — Copyright (c) 2026 DocShield Contributors

Ver archivo LICENSE para el texto completo.

> **DISCLAIMER:** This software is provided for educational and research purposes. Users are solely responsible for ensuring compliance with applicable laws and regulations when deploying this system in production environments.
