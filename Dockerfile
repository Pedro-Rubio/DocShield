# Multi-stage build para producción
FROM python:3.12-slim as builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Imagen final de producción
FROM python:3.12-slim

WORKDIR /app

# Copiar solo las dependencias instaladas
COPY --from=builder /root/.local /root/.local

# Asegurar que los scripts estén en PATH
ENV PATH=/root/.local/bin:$PATH

# Copiar el resto de la aplicación (sin requirements-dev.txt)
COPY . .

# Exponer el puerto de la API
EXPOSE 8000

# Comando por defecto
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
