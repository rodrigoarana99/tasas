"""
Utilidades para manejo de fechas y calendario FOMC
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional


# Fechas históricas de meetings FOMC (actualizar según necesidad)
# Formato: YYYY-MM-DD
HISTORICAL_FOMC_MEETINGS = [
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025 (proyectadas)
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17",
]


def get_fomc_meetings(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> List[datetime]:
    """
    Retorna lista de fechas de meetings FOMC en un rango

    Args:
        start_date: Fecha de inicio (formato YYYY-MM-DD)
        end_date: Fecha de fin (formato YYYY-MM-DD)

    Returns:
        Lista de fechas de meetings
    """
    meetings = [pd.to_datetime(date) for date in HISTORICAL_FOMC_MEETINGS]

    if start_date:
        start = pd.to_datetime(start_date)
        meetings = [m for m in meetings if m >= start]

    if end_date:
        end = pd.to_datetime(end_date)
        meetings = [m for m in meetings if m <= end]

    return sorted(meetings)


def get_next_fomc_meeting(reference_date: Optional[datetime] = None) -> Optional[datetime]:
    """
    Retorna la fecha del próximo meeting FOMC

    Args:
        reference_date: Fecha de referencia (default: hoy)

    Returns:
        Fecha del próximo meeting o None si no hay
    """
    if reference_date is None:
        reference_date = datetime.now()

    meetings = get_fomc_meetings()
    future_meetings = [m for m in meetings if m > reference_date]

    return future_meetings[0] if future_meetings else None


def get_previous_fomc_meeting(reference_date: Optional[datetime] = None) -> Optional[datetime]:
    """
    Retorna la fecha del meeting FOMC anterior

    Args:
        reference_date: Fecha de referencia (default: hoy)

    Returns:
        Fecha del meeting anterior o None si no hay
    """
    if reference_date is None:
        reference_date = datetime.now()

    meetings = get_fomc_meetings()
    past_meetings = [m for m in meetings if m < reference_date]

    return past_meetings[-1] if past_meetings else None


def days_until_next_meeting(reference_date: Optional[datetime] = None) -> int:
    """
    Calcula días hasta el próximo meeting FOMC

    Args:
        reference_date: Fecha de referencia (default: hoy)

    Returns:
        Número de días hasta el próximo meeting
    """
    if reference_date is None:
        reference_date = datetime.now()

    next_meeting = get_next_fomc_meeting(reference_date)

    if next_meeting is None:
        return -1

    return (next_meeting - reference_date).days


def is_business_day(date: datetime) -> bool:
    """
    Verifica si una fecha es día hábil (no sábado/domingo)

    Args:
        date: Fecha a verificar

    Returns:
        True si es día hábil
    """
    return date.weekday() < 5


def get_trading_days(start_date: datetime, end_date: datetime) -> List[datetime]:
    """
    Retorna lista de días hábiles entre dos fechas

    Args:
        start_date: Fecha de inicio
        end_date: Fecha de fin

    Returns:
        Lista de días hábiles
    """
    date_range = pd.date_range(start_date, end_date, freq='B')  # B = business days
    return list(date_range)


def get_data_window(
    meeting_date: datetime,
    days_before: int = 30,
    days_after: int = 5
) -> tuple[datetime, datetime]:
    """
    Retorna ventana de datos relevante para un meeting

    Args:
        meeting_date: Fecha del meeting
        days_before: Días antes del meeting
        days_after: Días después del meeting

    Returns:
        Tupla (fecha_inicio, fecha_fin)
    """
    start = meeting_date - timedelta(days=days_before)
    end = meeting_date + timedelta(days=days_after)

    return start, end
