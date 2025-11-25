"""
Configuración global del proyecto
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Rutas del proyecto
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Crear directorios si no existen
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# API Keys (opcional)
FRED_API_KEY = os.getenv("FRED_API_KEY", None)

# Configuración de logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Parámetros del modelo
RATE_SCENARIOS = [-50, -25, 0, 25, 50]  # Cambios posibles en bps
DEFAULT_FED_RATE = 5.25  # Tasa actual por defecto (actualizar según contexto)

# Fechas históricas importantes (para análisis)
FINANCIAL_CRISIS_START = "2008-01-01"
COVID_CRISIS_START = "2020-03-01"
POST_COVID_START = "2022-01-01"

# Configuración de features
FRED_TICKERS = {
    "PCEPILFE": "Core PCE",
    "UNRATE": "Unemployment Rate",
    "PAYEMS": "Non-Farm Payrolls",
    "GDPC1": "Real GDP",
    "FEDFUNDS": "Fed Funds Rate",
    "T10Y2Y": "10Y-2Y Spread",
    "CPIAUCSL": "CPI",
    "ICSA": "Initial Claims",
    "RSAFS": "Retail Sales",
    "INDPRO": "Industrial Production",
}

# ETFs para análisis de duration
BOND_ETFS = {
    "SHY": "1-3Y Treasury",
    "IEF": "7-10Y Treasury",
    "TLT": "20+Y Treasury",
    "AGG": "Aggregate Bond",
}

# Duraciones aproximadas (años)
DURATIONS = {
    "SHY": 1.8,
    "IEF": 7.5,
    "TLT": 17.0,
    "AGG": 6.0,
}
