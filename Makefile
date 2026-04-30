.PHONY: test lint install clean docker-build docker-run

# Instalar dependencias
install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install .

# Ejecutar tests
test:
	python -m pytest tests/ -v --cov=src/ --cov-report=term-missing

# Linting
lint:
	pylint src/ --disable=C0114,C0115,C0116,R0801

# Generar dataset y entrenar modelo
train:
	python -m src.dataset.generator
	python -m src.model.trainer

# Iniciar API
run-api:
	uvicorn src.api.main:app --reload

# Iniciar Streamlit
run-streamlit:
	streamlit run streamlit_app/app.py

# Limpiar archivos temporales
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -f *.log
	rm -rf .pytest_cache
	rm -rf mlruns/

# Construir imagen Docker
docker-build:
	docker build -t docshield:latest .

# Ejecutar contenedor Docker
docker-run:
	docker run -p 8000:8000 --env-file .env docshield:latest
