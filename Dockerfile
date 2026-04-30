FROM python:3.12-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos de dependencias
COPY requirements.txt .
COPY requirements-dev.txt .

# Instalar Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copiar el resto de la aplicación
COPY . .

# Exponer el puerto de la API
EXPOSE 8000

# Comando por defecto
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
