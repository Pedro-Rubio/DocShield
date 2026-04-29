"""Configuración de pytest para el proyecto DocShield."""

import sys
import os

# Agregar el directorio raíz al path para poder importar src
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
