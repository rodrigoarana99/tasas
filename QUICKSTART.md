# Quick Start Guide

## 🚀 Inicio Rápido

### 1. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 2. Configurar API Key de FRED (Opcional)

Si quieres descargar datos frescos de FRED:

```bash
# Copiar archivo de ejemplo
cp .env.example .env

# Editar y agregar tu API key (gratis en https://fred.stlouisfed.org/docs/api/api_key.html)
# FRED_API_KEY=tu_api_key_aqui
```

**Nota:** Si no configuras la API key, el sistema usará datos en caché.

### 3. Ejecutar el Modelo Completo

#### Opción A: Workflow Completo (Recomendado para primera vez)

```bash
python run_model.py --all
```

Esto ejecutará:
1. Descarga de datos (FRED + yfinance)
2. Entrenamiento del modelo
3. Predicción de probabilidades
4. Análisis de Expected Value
5. Análisis de Duration

#### Opción B: Paso por Paso

```bash
# 1. Descargar datos
python run_model.py --download

# 2. Entrenar modelo
python run_model.py --train

# 3. Predecir probabilidades
python run_model.py --predict

# 4. Analizar trades
python run_model.py --analyze

# 5. Análisis de duration
python run_model.py --duration
```

## 📊 Salida Esperada

### Probabilidades
```
Probabilidades para próximo meeting FOMC:
------------------------------------------------------------
  -50 bps:   5.0% ██
  -25 bps:  30.0% ███████████████
    0 bps:  45.0% ██████████████████████
  +25 bps:  15.0% ███████
  +50 bps:   5.0% ██
```

### Expected Value de Trades
```
Long TLT (20Y):
  Expected Value:     +0.0095 (+0.95%)
  Prob. de Ganancia:  80.0%
  Max Gain / Loss:    +8.50% / -8.50%
  Risk/Reward:        4.33x
  Sharpe Ratio:       0.25
  EJECUTAR:           ✅ SI
```

## 🔧 Uso Programático

```python
from src.probability_models.logistic_model import FedRateLogisticModel
from src.expected_value.payoff_calculator import PayoffCalculator, long_duration_payoff

# Cargar modelo
model = FedRateLogisticModel.load()

# Obtener features actuales y predecir
# ... (ver run_model.py para ejemplo completo)

# Calcular EV
probabilities = {-50: 0.05, -25: 0.30, 0: 0.45, 25: 0.15, 50: 0.05}
calc = PayoffCalculator()
ev, breakdown = calc.calculate_ev(probabilities, long_duration_payoff(duration=17))

print(f"Expected Value: {ev:.4f}")
```

## 📁 Estructura de Archivos Generados

```
data/
├── raw/
│   ├── fred/          # Datos de FRED en caché (*.pkl)
│   ├── yfinance/      # Datos de ETFs (*.pkl)
│   └── fomc_calendar/ # Decisiones históricas (*.csv)
└── processed/         # Features procesados

models/
└── logistic_model.pkl # Modelo entrenado

logs/
└── tasas_YYYYMMDD.log # Logs del día
```

## 🐛 Troubleshooting

### Error: "No module named 'fredapi'"
```bash
pip install fredapi
```

### Error: "FRED_API_KEY not found"
El sistema funcionará usando datos en caché. Para descargar datos frescos, configura la API key en `.env`.

### Error: "Modelo no encontrado"
Primero debes entrenar el modelo:
```bash
python run_model.py --train
```

### Los datos están desactualizados
Ejecuta la descarga nuevamente:
```bash
python run_model.py --download
```

## 📚 Próximos Pasos

1. **Explorar notebooks**: Ver `notebooks/` para análisis interactivos
2. **Personalizar trades**: Editar payoff functions en `src/expected_value/payoff_calculator.py`
3. **Mejorar modelo**: Agregar más features o probar XGBoost/Random Forest
4. **Dashboard**: (Próximamente) `streamlit run dashboard/streamlit_app.py`

## 💡 Tips

- **Datos en caché**: Los datos se guardan en `data/raw/` para evitar descargas repetidas
- **Actualizar decisiones FOMC**: Editar `src/utils/date_utils.py` con nuevas fechas
- **Agregar nueva decisión**:
  ```python
  from src.data_collection.fomc_calendar_scraper import FOMCCalendar
  fomc = FOMCCalendar()
  fomc.add_decision(date="2025-01-29", rate_change_bps=-25, new_rate=4.75)
  ```

## 🎯 Objetivos del Modelo

Este modelo te permite:

✅ **Predecir probabilidades** de cambios en Fed Funds Rate
✅ **Calcular Expected Value** de armados de trading
✅ **Comparar strategies** (duration, steepeners, straddles)
✅ **Optimizar allocation** basado en view de tasas
✅ **Backtesting** de decisiones (próximamente)

---

Para más información, ver `README.md` completo.
