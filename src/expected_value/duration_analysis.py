"""
Análisis de duration para instrumentos de renta fija
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional

from ..utils.config import DURATIONS
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DurationAnalyzer:
    """Análisis de duration y sensibilidad a tasas"""

    def __init__(self):
        """Inicializa analizador"""
        logger.info("DurationAnalyzer inicializado")

    def calculate_modified_duration(
        self,
        price: float,
        coupon_rate: float,
        ytm: float,
        years_to_maturity: float,
        frequency: int = 2
    ) -> float:
        """
        Calcula modified duration de un bono

        Args:
            price: Precio del bono
            coupon_rate: Tasa de cupón anual
            ytm: Yield to maturity
            years_to_maturity: Años hasta vencimiento
            frequency: Frecuencia de cupones por año

        Returns:
            Modified duration
        """
        # Macaulay duration
        periods = int(years_to_maturity * frequency)
        coupon = coupon_rate / frequency
        ytm_period = ytm / frequency

        # Calcular Macaulay duration
        pv_cash_flows = 0
        weighted_pv = 0

        for t in range(1, periods + 1):
            cf = coupon if t < periods else (1 + coupon)
            pv = cf / (1 + ytm_period) ** t
            pv_cash_flows += pv
            weighted_pv += t * pv / frequency

        macaulay_duration = weighted_pv / pv_cash_flows

        # Modified duration
        modified_duration = macaulay_duration / (1 + ytm / frequency)

        return modified_duration

    def estimate_price_change(
        self,
        duration: float,
        rate_change_bps: int,
        convexity: Optional[float] = None
    ) -> float:
        """
        Estima cambio de precio dado cambio en tasas

        Args:
            duration: Duration del instrumento
            rate_change_bps: Cambio en tasas (bps)
            convexity: Convexidad (opcional)

        Returns:
            Cambio porcentual en precio
        """
        rate_change_pct = rate_change_bps / 10000

        # Efecto de duration (lineal)
        price_change = -duration * rate_change_pct

        # Efecto de convexidad (cuadrático)
        if convexity is not None:
            price_change += 0.5 * convexity * (rate_change_pct ** 2)

        return price_change

    def calculate_dv01(
        self,
        duration: float,
        price: float = 100.0
    ) -> float:
        """
        Calcula DV01 (Dollar Value of 01 basis point)

        Args:
            duration: Duration del instrumento
            price: Precio del instrumento

        Returns:
            DV01
        """
        # DV01 = Duration × Price × 0.0001
        dv01 = duration * price * 0.0001
        return dv01

    def analyze_etf_duration(
        self,
        etf_ticker: str,
        rate_scenarios: list = None
    ) -> pd.DataFrame:
        """
        Analiza sensibilidad de un ETF a diferentes escenarios de tasas

        Args:
            etf_ticker: Ticker del ETF
            rate_scenarios: Lista de cambios de tasa en bps

        Returns:
            DataFrame con análisis
        """
        if rate_scenarios is None:
            from ..utils.config import RATE_SCENARIOS
            rate_scenarios = RATE_SCENARIOS

        duration = DURATIONS.get(etf_ticker, 0)

        if duration == 0:
            logger.warning(f"Duration no disponible para {etf_ticker}")
            return pd.DataFrame()

        analysis_data = []

        for scenario in rate_scenarios:
            price_change = self.estimate_price_change(duration, scenario)
            dv01 = self.calculate_dv01(duration)

            analysis_data.append({
                'rate_change_bps': scenario,
                'estimated_price_change_pct': price_change,
                'estimated_return': price_change,
                'dv01': dv01
            })

        df = pd.DataFrame(analysis_data)

        logger.info(f"Análisis completado para {etf_ticker} (duration={duration:.1f})")
        return df

    def compare_etfs(
        self,
        rate_change_bps: int
    ) -> pd.DataFrame:
        """
        Compara todos los ETFs configurados para un escenario

        Args:
            rate_change_bps: Cambio en tasas (bps)

        Returns:
            DataFrame comparativo
        """
        from ..utils.config import BOND_ETFS

        comparison_data = []

        for ticker, description in BOND_ETFS.items():
            duration = DURATIONS.get(ticker, 0)

            if duration == 0:
                continue

            price_change = self.estimate_price_change(duration, rate_change_bps)
            dv01 = self.calculate_dv01(duration)

            comparison_data.append({
                'etf': ticker,
                'description': description,
                'duration': duration,
                'price_change_pct': price_change,
                'dv01': dv01
            })

        df = pd.DataFrame(comparison_data)
        df = df.sort_values('duration')

        return df

    def optimal_duration_for_view(
        self,
        expected_rate_change: float,
        max_risk: float = 0.05
    ) -> Dict[str, any]:
        """
        Sugiere duration óptima dado un view de tasas

        Args:
            expected_rate_change: Cambio esperado en bps
            max_risk: Máximo riesgo aceptable (como % de precio)

        Returns:
            Dict con recomendación
        """
        from ..utils.config import BOND_ETFS

        # Calcular duration máxima permitida por riesgo
        rate_change_pct = abs(expected_rate_change) / 10000
        max_duration = max_risk / rate_change_pct if rate_change_pct > 0 else 100

        # Encontrar ETF más cercano
        best_etf = None
        best_duration = 0
        min_diff = float('inf')

        for ticker in BOND_ETFS.keys():
            duration = DURATIONS.get(ticker, 0)

            if duration <= max_duration:
                diff = abs(duration - max_duration)
                if diff < min_diff:
                    min_diff = diff
                    best_etf = ticker
                    best_duration = duration

        # Calcular payoff esperado
        expected_return = self.estimate_price_change(
            best_duration,
            expected_rate_change
        )

        recommendation = {
            'view': 'dovish' if expected_rate_change < 0 else 'hawkish',
            'expected_rate_change_bps': expected_rate_change,
            'max_duration': max_duration,
            'recommended_etf': best_etf,
            'etf_duration': best_duration,
            'expected_return': expected_return,
            'risk_at_worst_case': self.estimate_price_change(
                best_duration,
                -expected_rate_change  # Worst case
            )
        }

        return recommendation


def main():
    """Función principal para testing"""
    analyzer = DurationAnalyzer()

    # Análisis de ETFs para diferentes escenarios
    print("="*80)
    print("ANÁLISIS DE DURATION - ETFs DE BONOS")
    print("="*80)

    for scenario in [-50, -25, 0, 25, 50]:
        print(f"\nEscenario: {scenario:+d} bps")
        print("-" * 60)

        comparison = analyzer.compare_etfs(scenario)

        for _, row in comparison.iterrows():
            print(f"{row['etf']:5s} ({row['description']:20s}): "
                  f"Duration={row['duration']:5.1f}Y, "
                  f"Retorno={row['price_change_pct']:+7.2%}, "
                  f"DV01=${row['dv01']:.4f}")

    # Recomendación basada en view
    print("\n" + "="*80)
    print("RECOMENDACIONES BASADAS EN VIEW")
    print("="*80)

    views = [
        {'change': -50, 'desc': 'Muy dovish (corte 50bps)'},
        {'change': -25, 'desc': 'Dovish (corte 25bps)'},
        {'change': 25, 'desc': 'Hawkish (suba 25bps)'},
    ]

    for view in views:
        print(f"\n{view['desc']}:")
        rec = analyzer.optimal_duration_for_view(view['change'], max_risk=0.05)

        print(f"  ETF Recomendado:    {rec['recommended_etf']}")
        print(f"  Duration:           {rec['etf_duration']:.1f}Y")
        print(f"  Retorno Esperado:   {rec['expected_return']:+.2%}")
        print(f"  Riesgo (worst case): {rec['risk_at_worst_case']:+.2%}")


if __name__ == "__main__":
    main()
