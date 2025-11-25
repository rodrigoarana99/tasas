"""
Modelo logístico baseline para probabilidades de cambios de tasa
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, Optional, Tuple
import joblib

from ..utils.config import RATE_SCENARIOS, MODELS_DIR
from ..utils.logger import get_logger

logger = get_logger(__name__)


class FedRateLogisticModel:
    """Modelo logístico para predecir cambios en Fed Funds Rate"""

    def __init__(
        self,
        scenarios: list = None,
        max_iter: int = 1000,
        C: float = 1.0
    ):
        """
        Inicializa modelo

        Args:
            scenarios: Lista de escenarios posibles (default: [-50, -25, 0, 25, 50])
            max_iter: Máximo de iteraciones
            C: Parámetro de regularización
        """
        self.scenarios = scenarios or RATE_SCENARIOS
        self.max_iter = max_iter
        self.C = C

        # Modelo multinomial para múltiples clases
        self.model = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=max_iter,
            C=C,
            random_state=42
        )

        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False

        logger.info(f"FedRateLogisticModel inicializado con {len(self.scenarios)} escenarios")

    def prepare_data(
        self,
        features_df: pd.DataFrame,
        targets: pd.Series,
        feature_subset: Optional[list] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara datos para entrenamiento

        Args:
            features_df: DataFrame con features
            targets: Serie con targets (cambios de tasa en bps)
            feature_subset: Lista de features a usar (opcional)

        Returns:
            Tupla (X, y) con features escalados y targets codificados
        """
        if feature_subset:
            features_df = features_df[feature_subset]

        self.feature_names = list(features_df.columns)

        # Escalar features
        X = self.scaler.fit_transform(features_df)

        # Mapear targets a índices de escenarios
        y = targets.apply(lambda x: self._map_to_scenario(x)).values

        logger.info(f"Datos preparados: X={X.shape}, y={y.shape}")
        return X, y

    def _map_to_scenario(self, rate_change: float) -> int:
        """
        Mapea cambio de tasa al escenario más cercano

        Args:
            rate_change: Cambio de tasa en bps

        Returns:
            Índice del escenario
        """
        # Encontrar escenario más cercano
        distances = [abs(rate_change - s) for s in self.scenarios]
        return distances.index(min(distances))

    def train(
        self,
        features_df: pd.DataFrame,
        targets: pd.Series,
        feature_subset: Optional[list] = None
    ):
        """
        Entrena el modelo

        Args:
            features_df: DataFrame con features
            targets: Serie con targets (cambios de tasa en bps)
            feature_subset: Lista de features a usar (opcional)
        """
        logger.info("Entrenando modelo logístico...")

        X, y = self.prepare_data(features_df, targets, feature_subset)

        # Entrenar
        self.model.fit(X, y)
        self.is_fitted = True

        # Métricas de entrenamiento
        train_score = self.model.score(X, y)
        logger.info(f"Accuracy de entrenamiento: {train_score:.3f}")

        # Feature importance (coeficientes)
        self._log_feature_importance()

    def _log_feature_importance(self):
        """Log de features más importantes"""
        if not self.is_fitted or self.feature_names is None:
            return

        # Tomar coeficientes promedio sobre todas las clases
        coef_avg = np.abs(self.model.coef_).mean(axis=0)

        # Top 10 features
        top_indices = np.argsort(coef_avg)[-10:][::-1]

        logger.info("Top 10 features más importantes:")
        for idx in top_indices:
            feature = self.feature_names[idx]
            importance = coef_avg[idx]
            logger.info(f"  {feature:40s}: {importance:.4f}")

    def predict_proba(self, features_df: pd.DataFrame) -> Dict[int, float]:
        """
        Predice probabilidades para cada escenario

        Args:
            features_df: DataFrame con features (una fila)

        Returns:
            Diccionario {escenario_bps: probabilidad}
        """
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado. Llamar train() primero.")

        # Asegurar que tenemos los features correctos
        if self.feature_names:
            features_df = features_df[self.feature_names]

        # Escalar
        X = self.scaler.transform(features_df)

        # Predecir
        probs = self.model.predict_proba(X)[0]

        # Mapear a diccionario
        prob_dict = {
            scenario: prob
            for scenario, prob in zip(self.scenarios, probs)
        }

        return prob_dict

    def cross_validate(
        self,
        features_df: pd.DataFrame,
        targets: pd.Series,
        n_splits: int = 5
    ) -> Dict[str, float]:
        """
        Cross-validation con splits temporales

        Args:
            features_df: DataFrame con features
            targets: Serie con targets
            n_splits: Número de splits

        Returns:
            Diccionario con métricas
        """
        logger.info(f"Cross-validation con {n_splits} splits...")

        X, y = self.prepare_data(features_df, targets)

        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Entrenar en fold
            self.model.fit(X_train, y_train)

            # Evaluar
            score = self.model.score(X_test, y_test)
            scores.append(score)

            logger.info(f"  Fold {fold+1}: accuracy = {score:.3f}")

        # Re-entrenar con todos los datos
        self.model.fit(X, y)
        self.is_fitted = True

        metrics = {
            'cv_mean': np.mean(scores),
            'cv_std': np.std(scores),
            'cv_min': np.min(scores),
            'cv_max': np.max(scores)
        }

        logger.info(f"CV Mean Accuracy: {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")

        return metrics

    def save(self, filename: str = "logistic_model.pkl"):
        """
        Guarda el modelo entrenado

        Args:
            filename: Nombre del archivo
        """
        if not self.is_fitted:
            logger.warning("Modelo no entrenado, guardando de todas formas")

        filepath = MODELS_DIR / filename

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'scenarios': self.scenarios,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Modelo guardado en {filepath}")

    @classmethod
    def load(cls, filename: str = "logistic_model.pkl") -> 'FedRateLogisticModel':
        """
        Carga un modelo guardado

        Args:
            filename: Nombre del archivo

        Returns:
            Modelo cargado
        """
        filepath = MODELS_DIR / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {filepath}")

        model_data = joblib.load(filepath)

        # Crear instancia
        instance = cls(scenarios=model_data['scenarios'])
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_names = model_data['feature_names']
        instance.is_fitted = model_data['is_fitted']

        logger.info(f"Modelo cargado desde {filepath}")
        return instance


def main():
    """Función principal para testing"""
    from ..data_collection.fred_api import FREDData
    from ..data_collection.fomc_calendar_scraper import FOMCCalendar
    from ..feature_engineering.macro_features import MacroFeatureEngineer

    # Cargar datos
    logger.info("Cargando datos...")
    fred = FREDData()
    raw_data = fred.create_features_dataframe(start_date="2010-01-01")

    # Crear features
    engineer = MacroFeatureEngineer()
    features = engineer.create_all_features(raw_data)

    # Cargar targets (decisiones FOMC)
    fomc = FOMCCalendar()
    decisions = fomc.load_historical_decisions()

    # Alinear features con decisiones
    features_at_meetings = features.loc[
        features.index.isin(decisions['date'])
    ]
    targets = decisions.set_index('date')['rate_change_bps']
    targets = targets.loc[features_at_meetings.index]

    print(f"\nDatos para entrenamiento:")
    print(f"  Features: {features_at_meetings.shape}")
    print(f"  Targets: {targets.shape}")
    print(f"  Distribución de targets:\n{targets.value_counts().sort_index()}")

    # Entrenar modelo
    model = FedRateLogisticModel()

    # Seleccionar features clave (simplificado para demo)
    key_features = [col for col in features_at_meetings.columns
                    if any(x in col for x in ['PCEPILFE', 'UNRATE', 'FEDFUNDS', 'T10Y2Y'])]

    if len(key_features) > 20:
        key_features = key_features[:20]

    print(f"\nUsando {len(key_features)} features")

    # Cross-validation
    cv_metrics = model.cross_validate(
        features_at_meetings,
        targets,
        n_splits=3
    )

    print(f"\nResultados de Cross-Validation:")
    print(f"  Mean Accuracy: {cv_metrics['cv_mean']:.3f}")
    print(f"  Std Dev: {cv_metrics['cv_std']:.3f}")

    # Guardar modelo
    model.save()

    # Predecir para última observación
    last_features = features.iloc[[-1]]
    probs = model.predict_proba(last_features)

    print(f"\nProbabilidades para próximo meeting:")
    for scenario, prob in sorted(probs.items()):
        print(f"  {scenario:+4d} bps: {prob:6.1%}")


if __name__ == "__main__":
    main()
