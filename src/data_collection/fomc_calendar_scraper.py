"""
Obtención de calendario FOMC y decisiones históricas
"""
import pandas as pd
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from datetime import datetime

from ..utils.config import RAW_DATA_DIR
from ..utils.date_utils import HISTORICAL_FOMC_MEETINGS
from ..utils.logger import get_logger

logger = get_logger(__name__)


class FOMCCalendar:
    """Manejo de calendario y decisiones FOMC"""

    def __init__(self):
        """Inicializa calendario FOMC"""
        self.cache_dir = RAW_DATA_DIR / "fomc_calendar"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("FOMCCalendar inicializado")

    def get_historical_meetings(self) -> List[datetime]:
        """
        Retorna lista de fechas históricas de meetings FOMC

        Returns:
            Lista de fechas
        """
        meetings = [pd.to_datetime(date) for date in HISTORICAL_FOMC_MEETINGS]
        return sorted(meetings)

    def load_historical_decisions(self) -> pd.DataFrame:
        """
        Carga datos históricos de decisiones FOMC

        Returns:
            DataFrame con decisiones históricas
        """
        # Por ahora, crear un DataFrame de ejemplo
        # En producción, esto vendría de un CSV o base de datos
        cache_file = self.cache_dir / "historical_decisions.csv"

        if cache_file.exists():
            logger.info("Cargando decisiones históricas desde caché")
            return pd.read_csv(cache_file, parse_dates=['date'])

        # Crear datos de ejemplo para desarrollo
        logger.warning("Creando datos de ejemplo - Reemplazar con datos reales")

        # Datos de ejemplo (2023-2024)
        example_data = {
            'date': [
                '2023-02-01', '2023-03-22', '2023-05-03', '2023-06-14',
                '2023-07-26', '2023-09-20', '2023-11-01', '2023-12-13',
                '2024-01-31', '2024-03-20', '2024-05-01', '2024-06-12',
                '2024-07-31', '2024-09-18', '2024-11-07',
            ],
            'rate_change_bps': [
                25, 25, 25, 0, 25, 0, 0, 0,
                0, 0, 0, 0, 0, -50, 0,
            ],
            'new_rate': [
                4.75, 5.00, 5.25, 5.25, 5.50, 5.50, 5.50, 5.50,
                5.50, 5.50, 5.50, 5.50, 5.50, 5.00, 5.00,
            ]
        }

        df = pd.DataFrame(example_data)
        df['date'] = pd.to_datetime(df['date'])

        # Guardar en caché
        df.to_csv(cache_file, index=False)
        logger.info(f"Datos de ejemplo guardados en {cache_file}")

        return df

    def add_decision(
        self,
        date: str,
        rate_change_bps: int,
        new_rate: float
    ):
        """
        Agrega una nueva decisión al registro histórico

        Args:
            date: Fecha del meeting (YYYY-MM-DD)
            rate_change_bps: Cambio en bps
            new_rate: Nueva tasa
        """
        df = self.load_historical_decisions()

        new_row = pd.DataFrame([{
            'date': pd.to_datetime(date),
            'rate_change_bps': rate_change_bps,
            'new_rate': new_rate
        }])

        df = pd.concat([df, new_row], ignore_index=True)
        df = df.drop_duplicates(subset=['date'], keep='last')
        df = df.sort_values('date')

        # Guardar
        cache_file = self.cache_dir / "historical_decisions.csv"
        df.to_csv(cache_file, index=False)
        logger.info(f"Decisión agregada: {date}, {rate_change_bps:+d}bps → {new_rate}%")

    def get_rate_changes(self) -> Dict[datetime, int]:
        """
        Retorna diccionario de cambios de tasa por fecha

        Returns:
            Dict {fecha: cambio_en_bps}
        """
        df = self.load_historical_decisions()
        return dict(zip(df['date'], df['rate_change_bps']))

    def get_current_rate(self) -> float:
        """
        Retorna la tasa actual (última decisión)

        Returns:
            Tasa actual
        """
        df = self.load_historical_decisions()
        if df.empty:
            return 5.0  # Default

        return df.iloc[-1]['new_rate']


def main():
    """Función principal para testing"""
    fomc = FOMCCalendar()

    # Cargar decisiones históricas
    decisions = fomc.load_historical_decisions()
    print(f"\nDecisiones históricas: {len(decisions)}")
    print(decisions.tail(10))

    # Tasa actual
    current_rate = fomc.get_current_rate()
    print(f"\nTasa actual: {current_rate}%")

    # Próximos meetings
    meetings = fomc.get_historical_meetings()
    today = datetime.now()
    future_meetings = [m for m in meetings if m > today]

    print(f"\nPróximos {min(5, len(future_meetings))} meetings:")
    for meeting in future_meetings[:5]:
        print(f"  {meeting.strftime('%Y-%m-%d (%A)')}")


if __name__ == "__main__":
    main()
