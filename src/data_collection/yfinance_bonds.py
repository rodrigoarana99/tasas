"""
Descarga de datos de ETFs de bonos vía yfinance (sin API key requerida)
"""
import pandas as pd
import yfinance as yf
from typing import Dict, Optional, List
from datetime import datetime, timedelta

from ..utils.config import BOND_ETFS, DURATIONS, RAW_DATA_DIR
from ..utils.logger import get_logger

logger = get_logger(__name__)


class BondETFData:
    """Cliente para descargar datos de ETFs de bonos"""

    def __init__(self):
        """Inicializa cliente de yfinance"""
        self.cache_dir = RAW_DATA_DIR / "yfinance"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Cliente yfinance inicializado")

    def download_etf(
        self,
        ticker: str,
        start_date: str = "2003-01-01",
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Descarga datos históricos de un ETF

        Args:
            ticker: Ticker del ETF (ej: 'TLT')
            start_date: Fecha de inicio
            end_date: Fecha de fin (default: hoy)

        Returns:
            DataFrame con OHLCV + Adj Close
        """
        cache_file = self.cache_dir / f"{ticker}.pkl"

        # Intentar cargar desde caché (si es reciente)
        if cache_file.exists():
            try:
                data = pd.read_pickle(cache_file)
                last_date = data.index[-1]
                days_old = (datetime.now() - last_date).days

                # Si el caché tiene menos de 1 día, usarlo
                if days_old < 1:
                    logger.info(f"Usando caché reciente para {ticker}")
                    return data
            except Exception as e:
                logger.warning(f"Error cargando caché de {ticker}: {e}")

        # Descargar desde yfinance
        try:
            logger.info(f"Descargando {ticker} desde yfinance...")

            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")

            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False
            )

            if data.empty:
                logger.error(f"No se obtuvieron datos para {ticker}")
                return pd.DataFrame()

            # Guardar en caché
            data.to_pickle(cache_file)
            logger.info(f"Descargado {ticker}: {len(data)} días")

            return data

        except Exception as e:
            logger.error(f"Error descargando {ticker}: {e}")
            return pd.DataFrame()

    def download_all_etfs(
        self,
        start_date: str = "2003-01-01",
        end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Descarga todos los ETFs configurados

        Args:
            start_date: Fecha de inicio
            end_date: Fecha de fin

        Returns:
            Diccionario {ticker: DataFrame}
        """
        logger.info(f"Descargando {len(BOND_ETFS)} ETFs...")

        data = {}
        for ticker, description in BOND_ETFS.items():
            logger.info(f"Procesando {ticker} ({description})...")
            df = self.download_etf(ticker, start_date, end_date)

            if not df.empty:
                data[ticker] = df
            else:
                logger.warning(f"No se pudo obtener datos para {ticker}")

        logger.info(f"Descarga completada: {len(data)}/{len(BOND_ETFS)} ETFs")
        return data

    def calculate_returns(
        self,
        ticker: str,
        window_days: int = 5,
        start_date: str = "2003-01-01"
    ) -> pd.Series:
        """
        Calcula retornos de un ETF

        Args:
            ticker: Ticker del ETF
            window_days: Ventana para calcular retornos
            start_date: Fecha de inicio

        Returns:
            Serie de retornos
        """
        df = self.download_etf(ticker, start_date)

        if df.empty:
            return pd.Series(dtype=float)

        # Usar Adj Close para ajustar por dividendos
        prices = df['Adj Close']
        returns = prices.pct_change(periods=window_days)

        return returns

    def get_etf_response_to_rate_change(
        self,
        fomc_dates: List[datetime],
        rate_changes: Dict[datetime, float]
    ) -> pd.DataFrame:
        """
        Analiza respuesta de ETFs a cambios de tasas en meetings FOMC

        Args:
            fomc_dates: Lista de fechas de meetings FOMC
            rate_changes: Dict {fecha: cambio_en_bps}

        Returns:
            DataFrame con análisis de respuesta
        """
        results = []

        for ticker in BOND_ETFS.keys():
            df = self.download_etf(ticker)

            if df.empty:
                continue

            for meeting_date in fomc_dates:
                if meeting_date not in rate_changes:
                    continue

                rate_change = rate_changes[meeting_date]

                # Calcular retorno del ETF en ventana alrededor del meeting
                try:
                    # 1 día antes y 1 día después
                    pre_date = meeting_date - timedelta(days=1)
                    post_date = meeting_date + timedelta(days=1)

                    # Buscar precios más cercanos
                    pre_price = df.loc[df.index <= pre_date, 'Adj Close'].iloc[-1]
                    post_price = df.loc[df.index >= post_date, 'Adj Close'].iloc[0]

                    etf_return = (post_price - pre_price) / pre_price

                    results.append({
                        'date': meeting_date,
                        'etf': ticker,
                        'rate_change_bps': rate_change,
                        'etf_return': etf_return,
                        'duration': DURATIONS.get(ticker, 0)
                    })

                except Exception as e:
                    logger.warning(f"Error procesando {ticker} en {meeting_date}: {e}")

        return pd.DataFrame(results)

    def get_current_prices(self) -> Dict[str, float]:
        """
        Obtiene precios actuales de todos los ETFs

        Returns:
            Diccionario {ticker: precio}
        """
        prices = {}

        for ticker in BOND_ETFS.keys():
            df = self.download_etf(ticker)
            if not df.empty:
                prices[ticker] = df['Adj Close'].iloc[-1]

        return prices


def main():
    """Función principal para testing"""
    import argparse

    parser = argparse.ArgumentParser(description="Descargar datos de ETFs de bonos")
    parser.add_argument("--start-date", default="2003-01-01", help="Fecha de inicio")
    parser.add_argument("--end-date", default=None, help="Fecha de fin")
    args = parser.parse_args()

    # Crear cliente y descargar
    bond_data = BondETFData()
    data = bond_data.download_all_etfs(args.start_date, args.end_date)

    print(f"\nDescargados {len(data)} ETFs:")
    for ticker, df in data.items():
        desc = BOND_ETFS.get(ticker, ticker)
        duration = DURATIONS.get(ticker, 0)
        print(f"  {ticker:5s} ({desc:20s}): {len(df):5d} días, "
              f"Duration: {duration:.1f}Y, "
              f"{df.index[0].date()} a {df.index[-1].date()}")

    # Mostrar precios actuales
    print("\nPrecios actuales:")
    prices = bond_data.get_current_prices()
    for ticker, price in prices.items():
        desc = BOND_ETFS.get(ticker, ticker)
        print(f"  {ticker:5s} ({desc:20s}): ${price:.2f}")


if __name__ == "__main__":
    main()
