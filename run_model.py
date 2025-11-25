#!/usr/bin/env python3
"""
Script principal para ejecutar el modelo completo de probabilidades y EV
"""
import sys
import argparse
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import get_logger
from src.utils.config import RATE_SCENARIOS
from src.data_collection.fred_api import FREDData
from src.data_collection.yfinance_bonds import BondETFData
from src.data_collection.fomc_calendar_scraper import FOMCCalendar
from src.feature_engineering.macro_features import MacroFeatureEngineer
from src.probability_models.logistic_model import FedRateLogisticModel
from src.expected_value.payoff_calculator import (
    PayoffCalculator,
    long_duration_payoff,
    short_duration_payoff,
    steepener_payoff,
    straddle_payoff
)
from src.expected_value.duration_analysis import DurationAnalyzer

logger = get_logger(__name__)


def download_data(start_date: str = "2010-01-01"):
    """Descarga todos los datos necesarios"""
    print("\n" + "="*80)
    print("DESCARGANDO DATOS")
    print("="*80)

    # FRED data
    print("\n1. Descargando datos macroeconómicos (FRED)...")
    fred = FREDData()
    macro_data = fred.download_all_indicators(start_date=start_date)
    print(f"   ✓ Descargadas {len(macro_data)} series macroeconómicas")

    # Bond ETFs
    print("\n2. Descargando ETFs de bonos (yfinance)...")
    bonds = BondETFData()
    bond_data = bonds.download_all_etfs(start_date=start_date)
    print(f"   ✓ Descargados {len(bond_data)} ETFs")

    # FOMC calendar
    print("\n3. Cargando calendario FOMC...")
    fomc = FOMCCalendar()
    decisions = fomc.load_historical_decisions()
    print(f"   ✓ Cargadas {len(decisions)} decisiones históricas")

    print("\n✓ Descarga de datos completada")


def train_model():
    """Entrena el modelo de probabilidades"""
    print("\n" + "="*80)
    print("ENTRENANDO MODELO")
    print("="*80)

    # Cargar datos
    print("\n1. Cargando datos...")
    fred = FREDData()
    raw_data = fred.create_features_dataframe(start_date="2010-01-01")
    print(f"   ✓ Datos cargados: {raw_data.shape}")

    # Feature engineering
    print("\n2. Creando features...")
    engineer = MacroFeatureEngineer()
    features = engineer.create_all_features(raw_data)
    print(f"   ✓ Features creados: {features.shape}")

    # Cargar targets
    fomc = FOMCCalendar()
    decisions = fomc.load_historical_decisions()

    # Alinear features con decisiones
    features_at_meetings = features.loc[features.index.isin(decisions['date'])]
    targets = decisions.set_index('date')['rate_change_bps']
    targets = targets.loc[features_at_meetings.index]

    print(f"\n3. Datos de entrenamiento:")
    print(f"   Observaciones: {len(targets)}")
    print(f"   Distribución de targets:")
    for val, count in targets.value_counts().sort_index().items():
        print(f"     {val:+4d} bps: {count:3d} veces ({count/len(targets):.1%})")

    # Entrenar modelo
    print("\n4. Entrenando modelo logístico con cross-validation...")
    model = FedRateLogisticModel()

    # Seleccionar features clave
    key_features = [col for col in features_at_meetings.columns
                    if any(x in col for x in ['PCEPILFE', 'UNRATE', 'FEDFUNDS', 'T10Y2Y', 'taylor'])]

    if len(key_features) > 30:
        key_features = key_features[:30]

    print(f"   Usando {len(key_features)} features")

    # Cross-validation
    cv_metrics = model.cross_validate(features_at_meetings, targets, n_splits=3)

    print(f"\n5. Resultados:")
    print(f"   Mean CV Accuracy: {cv_metrics['cv_mean']:.3f} ± {cv_metrics['cv_std']:.3f}")

    # Guardar modelo
    model.save()
    print(f"\n✓ Modelo entrenado y guardado")


def predict_probabilities():
    """Predice probabilidades para próximo meeting"""
    print("\n" + "="*80)
    print("PREDICCIÓN DE PROBABILIDADES")
    print("="*80)

    # Cargar modelo
    try:
        model = FedRateLogisticModel.load()
        print("✓ Modelo cargado")
    except FileNotFoundError:
        print("❌ Modelo no encontrado. Ejecutar primero: python run_model.py --train")
        return None

    # Cargar datos más recientes
    fred = FREDData()
    raw_data = fred.create_features_dataframe(start_date="2010-01-01")

    engineer = MacroFeatureEngineer()
    features = engineer.create_all_features(raw_data)

    # Predecir con última observación
    last_features = features.iloc[[-1]]

    probabilities = model.predict_proba(last_features)

    # Mostrar resultados
    print("\nProbabilidades para próximo meeting FOMC:")
    print("-" * 60)
    for scenario, prob in sorted(probabilities.items()):
        bar_length = int(prob * 50)
        bar = "█" * bar_length
        print(f"  {scenario:+4d} bps: {prob:6.1%} {bar}")

    return probabilities


def analyze_trades(probabilities: dict = None):
    """Analiza expected value de diferentes trades"""
    print("\n" + "="*80)
    print("ANÁLISIS DE EXPECTED VALUE")
    print("="*80)

    # Si no hay probabilidades, usar ejemplo
    if probabilities is None:
        print("\nUsando probabilidades de ejemplo...")
        probabilities = {
            -50: 0.05,
            -25: 0.30,
            0: 0.45,
            25: 0.15,
            50: 0.05
        }

    # Mostrar probabilidades
    print("\nProbabilidades:")
    for scenario, prob in sorted(probabilities.items()):
        print(f"  {scenario:+4d} bps: {prob:5.1%}")

    # Definir trades
    trades = {
        'Long TLT (20Y)': long_duration_payoff(duration=17.0),
        'Long IEF (7Y)': long_duration_payoff(duration=7.5),
        'Long SHY (2Y)': long_duration_payoff(duration=1.8),
        'Steepener (10Y-2Y)': steepener_payoff(),
        'Straddle en TLT': straddle_payoff(underlying_duration=17.0, premium=3.0)
    }

    # Analizar
    calc = PayoffCalculator()
    comparison = calc.compare_trades(probabilities, trades)

    # Mostrar resultados
    print("\n" + "="*80)
    print("COMPARACIÓN DE TRADES")
    print("="*80)

    for _, row in comparison.iterrows():
        print(f"\n{row['trade_name']}:")
        print(f"  Expected Value:     {row['expected_value']:+.4f} ({row['expected_value']*100:+.2f}%)")
        print(f"  Prob. de Ganancia:  {row['probability_of_profit']:.1%}")
        print(f"  Max Gain / Loss:    {row['max_gain']:+.2%} / {row['max_loss']:+.2%}")
        print(f"  Expected Gain/Loss: {row['expected_gain']:+.2%} / {row['expected_loss']:+.2%}")
        print(f"  Risk/Reward:        {row['risk_reward_ratio']:.2f}x")
        print(f"  Sharpe Ratio:       {row['sharpe_ratio']:.2f}")
        print(f"  EJECUTAR:           {'✅ SI' if row['should_execute'] else '❌ NO'}")
        if not row['should_execute']:
            print(f"  Razón:              {row['decision_reason']}")


def analyze_duration():
    """Análisis de duration para diferentes escenarios"""
    print("\n" + "="*80)
    print("ANÁLISIS DE DURATION")
    print("="*80)

    analyzer = DurationAnalyzer()

    for scenario in [-50, -25, 0, 25, 50]:
        print(f"\nEscenario: {scenario:+d} bps")
        print("-" * 60)

        comparison = analyzer.compare_etfs(scenario)

        for _, row in comparison.iterrows():
            print(f"  {row['etf']:5s} ({row['description']:20s}): "
                  f"Duration={row['duration']:5.1f}Y → "
                  f"Retorno={row['price_change_pct']:+7.2%}")


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="Modelo de Probabilidades de Tasas Fed y Expected Value"
    )

    parser.add_argument(
        "--download",
        action="store_true",
        help="Descargar datos de FRED y yfinance"
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Entrenar modelo de probabilidades"
    )

    parser.add_argument(
        "--predict",
        action="store_true",
        help="Predecir probabilidades para próximo meeting"
    )

    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analizar expected value de trades"
    )

    parser.add_argument(
        "--duration",
        action="store_true",
        help="Análisis de duration"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Ejecutar workflow completo"
    )

    args = parser.parse_args()

    # Si no hay args, mostrar ayuda
    if not any(vars(args).values()):
        parser.print_help()
        return

    try:
        probabilities = None

        if args.download or args.all:
            download_data()

        if args.train or args.all:
            train_model()

        if args.predict or args.all:
            probabilities = predict_probabilities()

        if args.analyze or args.all:
            analyze_trades(probabilities)

        if args.duration or args.all:
            analyze_duration()

        print("\n" + "="*80)
        print("✓ PROCESO COMPLETADO")
        print("="*80)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
