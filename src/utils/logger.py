"""
Sistema de logging para el proyecto
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from .config import LOGS_DIR, LOG_LEVEL


def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Configura y retorna un logger

    Args:
        name: Nombre del logger
        log_file: Nombre del archivo de log (opcional)

    Returns:
        Logger configurado
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))

    # Evitar duplicar handlers
    if logger.handlers:
        return logger

    # Formato
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Handler para consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler para archivo (si se especifica)
    if log_file:
        file_path = LOGS_DIR / log_file
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Obtiene o crea un logger con configuración por defecto

    Args:
        name: Nombre del módulo (__name__)

    Returns:
        Logger
    """
    # Crear nombre de archivo de log basado en la fecha
    today = datetime.now().strftime("%Y%m%d")
    log_file = f"tasas_{today}.log"

    return setup_logger(name, log_file)
