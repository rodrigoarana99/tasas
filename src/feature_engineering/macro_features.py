"""
Ingeniería de features macroeconómicos
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)


class MacroFeatureEngineer:
    """Generación de features macroeconómicos avanzados"""

    def __init__(self):
        """Inicializa feature engineer"""
        logger.info("MacroFeatureEngineer inicializado")

    def create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features de momentum (cambios porcentuales)

        Args:
            df: DataFrame con series macroeconómicas

        Returns:
            DataFrame con features de momentum
        """
        momentum_df = pd.DataFrame(index=df.index)

        for col in df.columns:
            # 1-month momentum
            momentum_df[f'{col}_mom_1m'] = df[col].pct_change(periods=1)

            # 3-month momentum
            momentum_df[f'{col}_mom_3m'] = df[col].pct_change(periods=3)

            # 6-month momentum
            momentum_df[f'{col}_mom_6m'] = df[col].pct_change(periods=6)

            # 12-month momentum
            momentum_df[f'{col}_mom_12m'] = df[col].pct_change(periods=12)

            # Absolute change
            momentum_df[f'{col}_diff_1m'] = df[col].diff(periods=1)
            momentum_df[f'{col}_diff_3m'] = df[col].diff(periods=3)

        logger.info(f"Creados {len(momentum_df.columns)} features de momentum")
        return momentum_df

    def calculate_taylor_rule(
        self,
        df: pd.DataFrame,
        inflation_col: str = 'PCEPILFE',
        unemployment_col: str = 'UNRATE',
        current_rate_col: str = 'FEDFUNDS',
        inflation_target: float = 2.0,
        natural_rate: float = 2.5,
        nairu: float = 4.0
    ) -> pd.Series:
        """
        Calcula Taylor Rule y desviación

        Taylor Rule: r* = r_neutral + 1.5*(π - π_target) + 0.5*(y - y*)

        Args:
            df: DataFrame con datos macro
            inflation_col: Columna de inflación
            unemployment_col: Columna de desempleo
            current_rate_col: Columna de tasa actual
            inflation_target: Target de inflación (default 2%)
            natural_rate: Tasa neutral (default 2.5%)
            nairu: Tasa natural de desempleo (default 4%)

        Returns:
            Serie con desviación de Taylor Rule
        """
        # Calcular tasa sugerida por Taylor Rule
        inflation = df[inflation_col]
        unemployment = df[unemployment_col]
        current_rate = df[current_rate_col]

        # Output gap aproximado usando Okun's Law: 2% desempleo ≈ 1% output
        output_gap = -2 * (unemployment - nairu)

        taylor_rate = (
            natural_rate +
            1.5 * (inflation - inflation_target) +
            0.5 * output_gap
        )

        # Desviación: si positiva, Fed está siendo "dove" (tasa baja vs Taylor)
        taylor_deviation = current_rate - taylor_rate

        logger.info("Taylor Rule calculada")
        return taylor_deviation

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features de interacción entre variables

        Args:
            df: DataFrame con features base

        Returns:
            DataFrame con interacciones
        """
        interactions = pd.DataFrame(index=df.index)

        # Inflación x Desempleo (curva de Phillips)
        if 'PCEPILFE' in df.columns and 'UNRATE' in df.columns:
            interactions['inflation_x_unemployment'] = df['PCEPILFE'] * df['UNRATE']

        # Tasa actual x Yield Curve
        if 'FEDFUNDS' in df.columns and 'T10Y2Y' in df.columns:
            interactions['rate_x_curve'] = df['FEDFUNDS'] * df['T10Y2Y']

        # Desempleo x Claims (señal de deterioro laboral)
        if 'UNRATE' in df.columns and 'ICSA' in df.columns:
            interactions['unemployment_x_claims'] = df['UNRATE'] * df['ICSA']

        logger.info(f"Creados {len(interactions.columns)} features de interacción")
        return interactions

    def create_regime_features(
        self,
        df: pd.DataFrame,
        rate_col: str = 'FEDFUNDS'
    ) -> pd.DataFrame:
        """
        Crea features de régimen (alcista, bajista, estable)

        Args:
            df: DataFrame con datos
            rate_col: Columna de tasa de interés

        Returns:
            DataFrame con features de régimen
        """
        regime_df = pd.DataFrame(index=df.index)

        if rate_col not in df.columns:
            return regime_df

        rates = df[rate_col]

        # Tendencia (3-month change)
        rate_change_3m = rates.diff(3)

        # Clasificar régimen
        regime_df['regime_hiking'] = (rate_change_3m > 0.1).astype(int)
        regime_df['regime_cutting'] = (rate_change_3m < -0.1).astype(int)
        regime_df['regime_stable'] = (
            (rate_change_3m >= -0.1) & (rate_change_3m <= 0.1)
        ).astype(int)

        # Velocidad de cambio
        regime_df['rate_change_velocity'] = rate_change_3m / 3  # Por mes

        logger.info("Features de régimen creados")
        return regime_df

    def create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features de volatilidad

        Args:
            df: DataFrame con series

        Returns:
            DataFrame con volatilidades
        """
        vol_df = pd.DataFrame(index=df.index)

        for col in df.columns:
            # Rolling volatility (std)
            vol_df[f'{col}_vol_3m'] = df[col].rolling(window=3).std()
            vol_df[f'{col}_vol_6m'] = df[col].rolling(window=6).std()

            # Range (max - min)
            vol_df[f'{col}_range_3m'] = (
                df[col].rolling(window=3).max() -
                df[col].rolling(window=3).min()
            )

        logger.info(f"Creados {len(vol_df.columns)} features de volatilidad")
        return vol_df

    def create_all_features(
        self,
        raw_df: pd.DataFrame,
        add_taylor: bool = True
    ) -> pd.DataFrame:
        """
        Crea todos los features a partir de datos raw

        Args:
            raw_df: DataFrame con datos crudos de FRED
            add_taylor: Si incluir Taylor Rule

        Returns:
            DataFrame con todos los features
        """
        logger.info("Creando features completos...")

        # Empezar con datos raw
        features = raw_df.copy()

        # Momentum
        momentum = self.create_momentum_features(raw_df)
        features = pd.concat([features, momentum], axis=1)

        # Taylor Rule
        if add_taylor and 'PCEPILFE' in raw_df.columns:
            taylor_dev = self.calculate_taylor_rule(raw_df)
            features['taylor_deviation'] = taylor_dev

        # Interacciones
        interactions = self.create_interaction_features(raw_df)
        features = pd.concat([features, interactions], axis=1)

        # Régimen
        regime = self.create_regime_features(raw_df)
        features = pd.concat([features, regime], axis=1)

        # Volatilidad
        volatility = self.create_volatility_features(raw_df)
        features = pd.concat([features, volatility], axis=1)

        # Eliminar NaN (forward fill primero, luego drop)
        features = features.fillna(method='ffill')
        features = features.dropna()

        logger.info(f"Features creados: {features.shape}")
        return features


def main():
    """Función principal para testing"""
    from ..data_collection.fred_api import FREDData

    # Cargar datos
    fred = FREDData()
    raw_data = fred.create_features_dataframe(start_date="2010-01-01")

    print(f"\nDatos raw: {raw_data.shape}")
    print(raw_data.tail())

    # Crear features
    engineer = MacroFeatureEngineer()
    features = engineer.create_all_features(raw_data)

    print(f"\nFeatures creados: {features.shape}")
    print(f"Columnas: {list(features.columns[:20])}...")

    # Mostrar estadísticas
    print("\nEstadísticas de features:")
    print(features.describe().T[['mean', 'std', 'min', 'max']].head(20))


if __name__ == "__main__":
    main()
