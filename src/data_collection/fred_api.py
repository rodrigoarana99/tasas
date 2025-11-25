"""
Descarga de datos macroeconómicos desde FRED (Federal Reserve Economic Data)
"""
import pandas as pd
from typing import Dict, Optional, List
from datetime import datetime
import pickle
from pathlib import Path

from ..utils.config import FRED_API_KEY, FRED_TICKERS, RAW_DATA_DIR
from ..utils.logger import get_logger

logger = get_logger(__name__)


class FREDData:
    """Cliente para descargar datos de FRED"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Inicializa cliente FRED

        Args:
            api_key: API key de FRED (opcional si está en .env)
        """
        self.api_key = api_key or FRED_API_KEY
        self.cache_dir = RAW_DATA_DIR / "fred"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.api_key:
            try:
                from fredapi import Fred
                self.fred = Fred(api_key=self.api_key)
                logger.info("FRED API inicializada correctamente")
            except ImportError:
                logger.warning("fredapi no instalado. Usando modo caché solamente.")
                self.fred = None
        else:
            logger.warning("FRED_API_KEY no configurada. Usando modo caché solamente.")
            self.fred = None

    def download_series(
        self,
        ticker: str,
        start_date: str = "1990-01-01",
        end_date: Optional[str] = None
    ) -> pd.Series:
        """
        Descarga una serie de tiempo de FRED

        Args:
            ticker: Ticker de FRED (ej: 'PCEPILFE')
            start_date: Fecha de inicio
            end_date: Fecha de fin (default: hoy)

        Returns:
            Serie de pandas con los datos
        """
        cache_file = self.cache_dir / f"{ticker}.pkl"

        # Intentar cargar desde caché
        if cache_file.exists():
            try:
                data = pd.read_pickle(cache_file)
                logger.info(f"Cargado {ticker} desde caché")
                return data
            except Exception as e:
                logger.warning(f"Error cargando caché de {ticker}: {e}")

        # Si no hay API, retornar vacío
        if self.fred is None:
            logger.error(f"No se puede descargar {ticker}: API no disponible")
            return pd.Series(dtype=float)

        # Descargar desde FRED
        try:
            logger.info(f"Descargando {ticker} desde FRED...")
            data = self.fred.get_series(ticker, start_date, end_date)

            # Guardar en caché
            data.to_pickle(cache_file)
            logger.info(f"Descargado y guardado {ticker} en caché")

            return data

        except Exception as e:
            logger.error(f"Error descargando {ticker}: {e}")
            return pd.Series(dtype=float)

    def download_all_indicators(
        self,
        start_date: str = "1990-01-01",
        end_date: Optional[str] = None
    ) -> Dict[str, pd.Series]:
        """
        Descarga todos los indicadores configurados

        Args:
            start_date: Fecha de inicio
            end_date: Fecha de fin

        Returns:
            Diccionario {ticker: serie}
        """
        logger.info(f"Descargando {len(FRED_TICKERS)} indicadores...")

        data = {}
        for ticker, description in FRED_TICKERS.items():
            logger.info(f"Procesando {ticker} ({description})...")
            series = self.download_series(ticker, start_date, end_date)

            if not series.empty:
                data[ticker] = series
            else:
                logger.warning(f"No se pudo obtener datos para {ticker}")

        logger.info(f"Descarga completada: {len(data)}/{len(FRED_TICKERS)} series")
        return data

    def get_latest_features(self) -> Dict[str, float]:
        """
        Obtiene los valores más recientes de todos los indicadores

        Returns:
            Diccionario {ticker: último_valor}
        """
        data = self.download_all_indicators()

        latest = {}
        for ticker, series in data.items():
            if not series.empty:
                latest[ticker] = series.iloc[-1]

        return latest

    def create_features_dataframe(
        self,
        start_date: str = "1990-01-01",
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Crea DataFrame con todos los features macroeconómicos

        Args:
            start_date: Fecha de inicio
            end_date: Fecha de fin

        Returns:
            DataFrame con todos los indicadores
        """
        data = self.download_all_indicators(start_date, end_date)

        # Convertir a DataFrame
        df = pd.DataFrame(data)

        # Forward fill para fechas faltantes (muchos indicadores son mensuales)
        df = df.fillna(method='ffill')

        logger.info(f"DataFrame creado: {df.shape}")
        return df


def main():
    """Función principal para testing"""
    import argparse

    parser = argparse.ArgumentParser(description="Descargar datos de FRED")
    parser.add_argument("--start-date", default="1990-01-01", help="Fecha de inicio")
    parser.add_argument("--end-date", default=None, help="Fecha de fin")
    args = parser.parse_args()

    # Crear cliente y descargar
    fred = FREDData()
    data = fred.download_all_indicators(args.start_date, args.end_date)

    print(f"\nDescargadas {len(data)} series:")
    for ticker, series in data.items():
        print(f"  {ticker}: {len(series)} observaciones "
              f"({series.index[0].date()} a {series.index[-1].date()})")

    # Mostrar últimos valores
    print("\nÚltimos valores:")
    latest = fred.get_latest_features()
    for ticker, value in latest.items():
        desc = FRED_TICKERS.get(ticker, ticker)
        print(f"  {desc:30s}: {value:.2f}")


if __name__ == "__main__":
    main()
