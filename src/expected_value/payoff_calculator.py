"""
Calculadora de Expected Value para armados de trading
"""
import pandas as pd
import numpy as np
from typing import Dict, Callable, Optional, Tuple

from ..utils.config import RATE_SCENARIOS
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PayoffCalculator:
    """Calcula Expected Value de trades basados en probabilidades"""

    def __init__(self, scenarios: list = None):
        """
        Inicializa calculadora

        Args:
            scenarios: Lista de escenarios posibles (default: [-50, -25, 0, 25, 50])
        """
        self.scenarios = scenarios or RATE_SCENARIOS
        logger.info(f"PayoffCalculator inicializado con {len(self.scenarios)} escenarios")

    def calculate_ev(
        self,
        probabilities: Dict[int, float],
        payoff_function: Callable[[int], float]
    ) -> Tuple[float, pd.DataFrame]:
        """
        Calcula Expected Value de un trade

        Args:
            probabilities: Dict {scenario_bps: probability}
            payoff_function: Función que mapea scenario → payoff

        Returns:
            Tupla (expected_value, breakdown_df)
        """
        ev = 0
        breakdown_data = []

        for scenario in self.scenarios:
            prob = probabilities.get(scenario, 0)
            payoff = payoff_function(scenario)
            contribution = prob * payoff

            ev += contribution

            breakdown_data.append({
                'scenario_bps': scenario,
                'probability': prob,
                'payoff': payoff,
                'contribution': contribution
            })

        breakdown_df = pd.DataFrame(breakdown_data)

        logger.info(f"EV calculado: {ev:.4f}")
        return ev, breakdown_df

    def calculate_risk_metrics(
        self,
        probabilities: Dict[int, float],
        payoff_function: Callable[[int], float]
    ) -> Dict[str, float]:
        """
        Calcula métricas de riesgo del trade

        Args:
            probabilities: Probabilidades por escenario
            payoff_function: Función de payoff

        Returns:
            Dict con métricas de riesgo
        """
        # Calcular EV y breakdown
        ev, breakdown = self.calculate_ev(probabilities, payoff_function)

        # Métricas
        payoffs = breakdown['payoff'].values
        probs = breakdown['probability'].values

        # Probability of profit
        prob_profit = probs[payoffs > 0].sum()

        # Expected gain/loss
        expected_gain = (probs[payoffs > 0] * payoffs[payoffs > 0]).sum()
        expected_loss = (probs[payoffs < 0] * payoffs[payoffs < 0]).sum()

        # Max gain/loss
        max_gain = payoffs.max()
        max_loss = payoffs.min()

        # Variance y Sharpe (simplificado)
        variance = ((payoffs - ev) ** 2 * probs).sum()
        std_dev = np.sqrt(variance)
        sharpe = ev / std_dev if std_dev > 0 else 0

        # Risk/Reward ratio
        risk_reward = abs(expected_gain / expected_loss) if expected_loss != 0 else np.inf

        metrics = {
            'expected_value': ev,
            'probability_of_profit': prob_profit,
            'expected_gain': expected_gain,
            'expected_loss': expected_loss,
            'max_gain': max_gain,
            'max_loss': max_loss,
            'std_dev': std_dev,
            'sharpe_ratio': sharpe,
            'risk_reward_ratio': risk_reward
        }

        return metrics

    def should_trade(
        self,
        probabilities: Dict[int, float],
        payoff_function: Callable[[int], float],
        min_ev: float = 0.005,
        max_prob_loss: float = 0.60,
        max_loss: float = -0.05,
        min_risk_reward: float = 2.0,
        min_confidence: float = 0.3
    ) -> Tuple[bool, str]:
        """
        Determina si se debe ejecutar el trade basado en criterios

        Args:
            probabilities: Probabilidades por escenario
            payoff_function: Función de payoff
            min_ev: EV mínimo requerido
            max_prob_loss: Probabilidad máxima de pérdida aceptable
            max_loss: Pérdida máxima aceptable
            min_risk_reward: Ratio mínimo risk/reward
            min_confidence: Confianza mínima del modelo (basada en entropía)

        Returns:
            Tupla (ejecutar: bool, razón: str)
        """
        metrics = self.calculate_risk_metrics(probabilities, payoff_function)

        # Check 1: EV mínimo
        if metrics['expected_value'] < min_ev:
            return False, f"EV too low: {metrics['expected_value']:.4f} < {min_ev}"

        # Check 2: Probabilidad de pérdida
        prob_loss = 1 - metrics['probability_of_profit']
        if prob_loss > max_prob_loss:
            return False, f"Prob of loss too high: {prob_loss:.2%} > {max_prob_loss:.0%}"

        # Check 3: Max loss
        if metrics['max_loss'] < max_loss:
            return False, f"Max loss exceeds limit: {metrics['max_loss']:.2%} < {max_loss:.0%}"

        # Check 4: Risk/Reward
        if metrics['risk_reward_ratio'] < min_risk_reward:
            return False, f"Risk/reward too low: {metrics['risk_reward_ratio']:.2f} < {min_risk_reward}"

        # Check 5: Confianza del modelo (entropía)
        probs = np.array(list(probabilities.values()))
        probs = probs[probs > 0]  # Eliminar zeros para log
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(len(probabilities))
        confidence = 1 - entropy / max_entropy

        if confidence < min_confidence:
            return False, f"Model too uncertain: confidence {confidence:.2%} < {min_confidence:.0%}"

        return True, "All checks passed"

    def compare_trades(
        self,
        probabilities: Dict[int, float],
        payoff_functions: Dict[str, Callable[[int], float]]
    ) -> pd.DataFrame:
        """
        Compara múltiples trades

        Args:
            probabilities: Probabilidades por escenario
            payoff_functions: Dict {trade_name: payoff_function}

        Returns:
            DataFrame con comparación de trades
        """
        comparison_data = []

        for trade_name, payoff_func in payoff_functions.items():
            metrics = self.calculate_risk_metrics(probabilities, payoff_func)
            should_execute, reason = self.should_trade(probabilities, payoff_func)

            metrics['trade_name'] = trade_name
            metrics['should_execute'] = should_execute
            metrics['decision_reason'] = reason

            comparison_data.append(metrics)

        df = pd.DataFrame(comparison_data)

        # Ordenar por EV
        df = df.sort_values('expected_value', ascending=False)

        return df


# Funciones de payoff predefinidas

def long_duration_payoff(duration: float = 17.0) -> Callable[[int], float]:
    """
    Payoff para Long Duration trade (ej: Long TLT)

    Args:
        duration: Duration del instrumento (años)

    Returns:
        Función de payoff
    """
    def payoff(rate_change_bps: int) -> float:
        rate_change_pct = rate_change_bps / 10000
        price_change = -duration * rate_change_pct
        return price_change

    return payoff


def short_duration_payoff(duration: float = 17.0) -> Callable[[int], float]:
    """
    Payoff para Short Duration trade

    Args:
        duration: Duration del instrumento (años)

    Returns:
        Función de payoff
    """
    def payoff(rate_change_bps: int) -> float:
        rate_change_pct = rate_change_bps / 10000
        price_change = duration * rate_change_pct  # Negativo de long
        return price_change

    return payoff


def steepener_payoff(
    long_duration: float = 8.5,
    short_duration: float = 2.0,
    curve_response: Dict[str, float] = None
) -> Callable[[int], float]:
    """
    Payoff para Steepener trade (Long 10Y, Short 2Y)

    Args:
        long_duration: Duration del leg largo
        short_duration: Duration del leg corto
        curve_response: Respuesta empírica de la curva por tipo de movimiento

    Returns:
        Función de payoff
    """
    # Respuesta por defecto (empírica)
    if curve_response is None:
        curve_response = {
            'cut': 10,      # bps de steepening en cortes
            'hike': -5,     # bps de flattening en subas
            'hold': 0       # sin cambio
        }

    def payoff(rate_change_bps: int) -> float:
        # Determinar tipo de movimiento
        if rate_change_bps <= -25:
            curve_change = curve_response['cut']
        elif rate_change_bps >= 25:
            curve_change = curve_response['hike']
        else:
            curve_change = curve_response['hold']

        # P&L del steepener
        dv01_diff = long_duration - short_duration
        pnl = curve_change * dv01_diff / 10000

        return pnl

    return payoff


def straddle_payoff(
    underlying_duration: float = 17.0,
    strike: float = 100.0,
    premium: float = 3.0,
    spot: float = 100.0
) -> Callable[[int], float]:
    """
    Payoff para Straddle (Long Call + Long Put)

    Args:
        underlying_duration: Duration del subyacente (ej: TLT)
        strike: Strike del straddle
        premium: Prima pagada
        spot: Precio spot actual

    Returns:
        Función de payoff
    """
    def payoff(rate_change_bps: int) -> float:
        # Movimiento del subyacente
        rate_change_pct = rate_change_bps / 10000
        price_pct_change = -underlying_duration * rate_change_pct
        new_price = spot * (1 + price_pct_change)

        # Payoff del straddle
        call_value = max(new_price - strike, 0)
        put_value = max(strike - new_price, 0)

        total_payoff = (call_value + put_value - premium) / spot

        return total_payoff

    return payoff


def main():
    """Función principal para testing"""
    # Probabilidades de ejemplo
    probabilities = {
        -50: 0.05,
        -25: 0.30,
        0: 0.45,
        25: 0.15,
        50: 0.05
    }

    print("Probabilidades:")
    for scenario, prob in probabilities.items():
        print(f"  {scenario:+4d} bps: {prob:5.1%}")

    # Crear calculadora
    calc = PayoffCalculator()

    # Definir varios trades
    trades = {
        'Long TLT': long_duration_payoff(duration=17.0),
        'Short TLT': short_duration_payoff(duration=17.0),
        'Steepener': steepener_payoff(),
        'Straddle': straddle_payoff()
    }

    # Comparar trades
    comparison = calc.compare_trades(probabilities, trades)

    print("\n" + "="*80)
    print("COMPARACIÓN DE TRADES")
    print("="*80)

    for _, row in comparison.iterrows():
        print(f"\n{row['trade_name']}:")
        print(f"  EV:                {row['expected_value']:+.4f} ({row['expected_value']*100:+.2f}%)")
        print(f"  Prob Profit:       {row['probability_of_profit']:.1%}")
        print(f"  Max Gain/Loss:     {row['max_gain']:+.4f} / {row['max_loss']:+.4f}")
        print(f"  Risk/Reward:       {row['risk_reward_ratio']:.2f}")
        print(f"  Sharpe:            {row['sharpe_ratio']:.2f}")
        print(f"  Ejecutar:          {'✅ SI' if row['should_execute'] else '❌ NO'}")
        print(f"  Razón:             {row['decision_reason']}")


if __name__ == "__main__":
    main()
