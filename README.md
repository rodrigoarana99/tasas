# tasas

# Interest Rate Change Probability Model - Fed Funds Rate

## 📋 Descripción del Proyecto

Modelo cuantitativo para estimar la probabilidad de cambios en la tasa de interés de referencia de la Reserva Federal (Fed Funds Rate) y **calcular la esperanza matemática de distintos armados de trading** bajo diferentes escenarios de tasas.

El objetivo principal es **evaluar el payoff esperado** de estructuras como:
- Long/Short duration trades
- Steepener/Flattener spreads  
- Options strategies (straddles, strangles en tasas)
- Bond portfolios con diferentes duraciones
- FX carry trades sensibles a diferenciales de tasas

## 🎯 Objetivo Principal

**Calcular Expected Value de Trades:**

Dado un armado con payoff definido P(r) donde r es el cambio de tasa:

```
E[P] = Σ P(r_i) × Prob(r_i)
```

Donde:
- `r_i` ∈ {-50bps, -25bps, 0, +25bps, +50bps}
- `Prob(r_i)` = Probabilidad estimada por el modelo
- `P(r_i)` = Payoff del armado en el escenario i

**Ejemplo concreto:**
```python
# Armado: Long TLT (20Y Treasury ETF) si esperamos corte de tasas

scenarios = {
    -50: {"prob": 0.05, "tlt_return": 0.08},   # Corte 50bps → TLT sube 8%
    -25: {"prob": 0.25, "tlt_return": 0.04},   # Corte 25bps → TLT sube 4%
      0: {"prob": 0.50, "tlt_return": 0.00},   # Sin cambio
    +25: {"prob": 0.18, "tlt_return": -0.04},  # Suba 25bps → TLT cae 4%
    +50: {"prob": 0.02, "tlt_return": -0.08}   # Suba 50bps → TLT cae 8%
}

expected_return = sum(s["prob"] * s["tlt_return"] for s in scenarios.values())
# E[R] = 0.05*8% + 0.25*4% + 0.50*0% + 0.18*(-4%) + 0.02*(-8%) = 0.42%
```

## 📁 Estructura del Proyecto

```
interest-rate-probability-model/
│
├── data/
│   ├── raw/                        # Datos sin procesar
│   │   ├── fred/                   # Indicadores macroeconómicos FRED
│   │   ├── thetadata/              # Futuros de tasas (ZQ, ZN, ZB)
│   │   ├── yfinance/               # ETFs de bonos (TLT, IEF, SHY)
│   │   └── fomc_calendar/          # Calendario de decisiones FOMC
│   ├── processed/                  # Features procesados
│   └── historical_decisions.csv    # Base histórica de decisiones Fed
│
├── src/
│   ├── data_collection/
│   │   ├── fred_api.py                 # Descarga datos FRED
│   │   ├── thetadata_futures.py        # API Thetadata para futuros ZQ
│   │   ├── yfinance_bonds.py           # ETFs de bonos vía yfinance
│   │   └── fomc_calendar_scraper.py    # Scraping calendario Fed
│   │
│   ├── feature_engineering/
│   │   ├── macro_features.py           # Features macro (PCE, Unemployment, etc)
│   │   ├── futures_features.py         # Análisis de futuros Fed Funds
│   │   ├── yield_curve.py              # Construcción curva de tasas
│   │   └── fed_sentiment.py            # Sentiment FOMC statements (opcional)
│   │
│   ├── probability_models/
│   │   ├── implied_probabilities.py    # Prob implícitas desde futuros ZQ
│   │   ├── logistic_model.py           # Modelo logístico baseline
│   │   ├── tree_models.py              # Random Forest, XGBoost
│   │   ├── ensemble_model.py           # Ensemble de modelos
│   │   └── model_calibration.py        # Calibración de probabilidades
│   │
│   ├── expected_value/
│   │   ├── payoff_calculator.py        # Cálculo de payoffs por escenario
│   │   ├── duration_analysis.py        # Análisis de duration para bonds
│   │   ├── options_payoff.py           # Payoffs de opciones sobre tasas
│   │   ├── spread_trades.py            # Steepeners, flatteners, butterflies
│   │   └── portfolio_ev.py             # EV de portfolios completos
│   │
│   ├── backtesting/
│   │   ├── backtest_probabilities.py   # Backtest de predicciones
│   │   ├── backtest_trades.py          # Backtest de trades basados en EV
│   │   └── performance_metrics.py      # Métricas (Brier, Sharpe, etc)
│   │
│   └── utils/
│       ├── date_utils.py               # Manejo fechas FOMC
│       ├── config.py                   # Configuración global
│       └── logger.py                   # Logging system
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_futures_implied_probs.ipynb
│   ├── 03_macro_models.ipynb
│   ├── 04_probability_ensemble.ipynb
│   ├── 05_payoff_structures.ipynb          # ⭐ Análisis de armados
│   ├── 06_expected_value_analysis.ipynb    # ⭐ Cálculo de EV
│   └── 07_backtesting_results.ipynb
│
├── dashboard/
│   ├── streamlit_app.py                # Dashboard principal
│   ├── pages/
│   │   ├── probabilities.py            # Visualización de probabilidades
│   │   ├── ev_calculator.py            # ⭐ Calculadora de EV para trades
│   │   └── backtesting.py              # Resultados históricos
│   └── components/
│       ├── scenario_table.py
│       └── payoff_charts.py
│
├── tests/
│   ├── test_models.py
│   ├── test_payoffs.py
│   └── test_data_collection.py
│
├── requirements.txt
├── setup.py
├── .env.example
├── .gitignore
└── README.md
```

## 🔧 Metodología

### 1. **Probabilidades Implícitas desde Futuros (Thetadata)**

**Instrumentos a usar:**
- **30-Day Fed Funds Futures (ZQ)** - Principal instrumento
- **2-Year Treasury Futures (ZT)** - Validación cruzada
- **10-Year Treasury Futures (ZN)** - Análisis de curva

**Cálculo de probabilidad implícita:**

```python
# Para un meeting del FOMC en fecha T
# Contrato de futuros ZQ que expira después del meeting

current_rate = 5.25  # EFFR actual
futures_price = 94.80  # Implica tasa de 5.20%
implied_rate = 100 - futures_price

# Probabilidad de corte de 25bps
days_before_meeting = 10
days_after_meeting = 20
total_days = 30

# Ajuste por días del mes
weight_current = days_before_meeting / total_days
weight_new = days_after_meeting / total_days

# Si asumimos corte de 25bps:
expected_rate_if_cut = weight_current * current_rate + weight_new * (current_rate - 0.25)

# Probabilidad implícita
prob_cut = (current_rate - implied_rate) / 0.25
```

**Ajustes necesarios:**
- Convexity bias (especialmente en entornos de alta volatilidad)
- Term premium adjustment
- Microstructure effects (bid-ask, liquidity)

### 2. **Modelos Econométricos**

**Variables Predictoras (FRED):**

| Variable | FRED Ticker | Descripción | Relevancia |
|----------|-------------|-------------|------------|
| Core PCE | PCEPILFE | Inflación core (preferida por Fed) | ⭐⭐⭐⭐⭐ |
| Unemployment | UNRATE | Tasa de desempleo | ⭐⭐⭐⭐⭐ |
| Non-Farm Payrolls | PAYEMS | Empleo no agrícola | ⭐⭐⭐⭐ |
| Real GDP | GDPC1 | Crecimiento económico | ⭐⭐⭐⭐ |
| Fed Funds Rate | FEDFUNDS | Tasa actual | ⭐⭐⭐⭐⭐ |
| 10Y-2Y Spread | T10Y2Y | Curva de rendimientos | ⭐⭐⭐⭐ |
| CPI | CPIAUCSL | Inflación headline | ⭐⭐⭐ |
| Initial Claims | ICSA | Solicitudes de desempleo | ⭐⭐⭐ |
| Retail Sales | RSAFS | Consumo | ⭐⭐⭐ |
| Industrial Production | INDPRO | Producción industrial | ⭐⭐ |

**Features Derivados:**
```python
# Momentum y cambios
core_pce_mom = core_pce.pct_change(periods=3)  # 3-month momentum
unemployment_change = unemployment.diff()

# Taylor Rule deviation
taylor_rate = neutral_rate + 1.5*(core_pce - 2.0) + 0.5*(output_gap)
taylor_deviation = fed_funds - taylor_rate

# Curva de Phillips
wage_inflation = wages.pct_change(periods=12)
phillips_residual = unemployment - NAIRU
```

**Modelos a Implementar:**

1. **Logistic Regression (Baseline)**
   - Interpretable
   - Rápido para iterar
   - Probabilidades bien calibradas

2. **Ordered Logit/Probit**
   - Para predecir: {-50bps, -25bps, 0, +25bps, +50bps}
   - Respeta ordering natural de outcomes

3. **Random Forest**
   - Feature importance
   - Non-linear relationships
   - Robusto a outliers

4. **XGBoost**
   - Mejor performance generalmente
   - Regularización built-in
   - Hyperparameter tuning intensivo

5. **Ensemble Model**
   - Weighted average de modelos
   - Implícitas (30%) + XGBoost (40%) + Logit (30%)

### 3. **Datos de Mercado (yfinance)**

**ETFs de Bonos para análisis de payoffs:**
- **SHY**: 1-3Y Treasury (baja duration)
- **IEF**: 7-10Y Treasury (duration media)
- **TLT**: 20+ Y Treasury (alta duration)
- **AGG**: Aggregate Bond Market
- **HYG**: High Yield (sensible a ciclo económico)

**Análisis histórico:**
```python
# Calcular respuesta histórica de TLT ante cambios de tasas
meetings = get_fomc_meetings()
for meeting in meetings:
    rate_change = get_rate_change(meeting)
    tlt_return = tlt_prices[meeting+1] / tlt_prices[meeting-1] - 1
    
    historical_responses[rate_change] = tlt_return
```

## 🎯 Cálculo de Expected Value para Armados

### Framework General

```python
def calculate_expected_value(probabilities, payoff_function, scenarios):
    """
    Calcula EV de un trade dado probabilidades y payoffs
    
    Args:
        probabilities: dict {scenario: probability}
        payoff_function: función que mapea scenario → payoff
        scenarios: lista de escenarios posibles
    
    Returns:
        expected_value: float
        scenario_breakdown: dict con análisis por escenario
    """
    ev = 0
    breakdown = {}
    
    for scenario in scenarios:
        prob = probabilities[scenario]
        payoff = payoff_function(scenario)
        contribution = prob * payoff
        
        ev += contribution
        breakdown[scenario] = {
            "probability": prob,
            "payoff": payoff,
            "contribution": contribution
        }
    
    return ev, breakdown
```

### Ejemplo 1: Long Duration (TLT)

```python
# Escenarios de cambio de tasa
scenarios = [-50, -25, 0, 25, 50]  # bps

# Probabilidades del modelo
probabilities = {
    -50: 0.05,
    -25: 0.30,
      0: 0.45,
     25: 0.15,
     50: 0.05
}

# Payoff basado en duration de TLT (≈17 años)
# Aproximación: ΔPrice ≈ -Duration × ΔYield
def tlt_payoff(rate_change_bps):
    rate_change_pct = rate_change_bps / 10000  # bps to decimal
    duration = 17
    price_change = -duration * rate_change_pct
    return price_change

# Cálculo
ev_tlt = sum(probabilities[s] * tlt_payoff(s) for s in scenarios)
# EV = 0.05*0.085 + 0.30*0.0425 + 0.45*0 + 0.15*(-0.0425) + 0.05*(-0.085)
# EV ≈ 0.0095 = +0.95%

# Interpretación: Si las probabilidades son correctas, 
# esperamos ganar 0.95% en TLT ante el próximo meeting
```

### Ejemplo 2: Steepener Trade (Long 10Y, Short 2Y)

```python
def steepener_payoff(rate_change_bps):
    """
    Steepener: apostamos a que la curva se empina
    En cortes de tasas, típicamente el front-end baja más
    """
    # Respuesta histórica empírica
    if rate_change_bps <= -25:  # Corte
        curve_steepening = 10  # bps (10Y-2Y aumenta)
        dv01_10y = 8.5
        dv01_2y = 2.0
        pnl = curve_steepening * (dv01_10y - dv_01_2y) / 10000
        return pnl
    elif rate_change_bps >= 25:  # Suba
        curve_flattening = -5  # bps
        pnl = curve_flattening * (dv01_10y - dv_01_2y) / 10000
        return pnl
    else:
        return 0

ev_steepener = sum(probabilities[s] * steepener_payoff(s) for s in scenarios)
```

### Ejemplo 3: Straddle en Opciones sobre TLT

```python
def straddle_payoff(rate_change_bps):
    """
    Long straddle: ganas con volatilidad (grandes movimientos)
    """
    # Simplified: payoff aumenta con |rate_change|
    spot = 100
    strike = 100
    premium_paid = 3  # Costo del straddle
    
    # TLT movement
    tlt_move = tlt_payoff(rate_change_bps)
    new_price = spot * (1 + tlt_move)
    
    # Straddle payoff
    call_value = max(new_price - strike, 0)
    put_value = max(strike - new_price, 0)
    
    total_payoff = call_value + put_value - premium_paid
    return total_payoff

ev_straddle = sum(probabilities[s] * straddle_payoff(s) for s in scenarios)

# También calcular break-even probability
# ¿Qué tan grande debe ser el movimiento para justificar la prima?
```

### Ejemplo 4: Portfolio Rebalancing Decision

```python
# Portfolio actual: 60% stocks (SPY) / 40% bonds (AGG)
# Decisión: ¿Aumentar duration si esperamos cortes?

def portfolio_payoff(rate_change_bps, allocation):
    """
    allocation: dict {"TLT": 0.3, "IEF": 0.2, "SHY": 0.1, "SPY": 0.4}
    """
    returns = {}
    
    # Bond returns basados en duration
    returns["TLT"] = tlt_payoff(rate_change_bps)
    returns["IEF"] = -7 * (rate_change_bps / 10000)  # duration ≈7
    returns["SHY"] = -1.5 * (rate_change_bps / 10000)  # duration ≈1.5
    
    # SPY: correlación histórica con tasas
    # Empíricamente: cortes de tasas → SPY sube (si no es recesión)
    if rate_change_bps < 0:
        returns["SPY"] = 0.02  # Positivo en cortes
    else:
        returns["SPY"] = -0.01  # Negativo en subas
    
    portfolio_return = sum(allocation[asset] * returns[asset] 
                          for asset in allocation)
    return portfolio_return

# Comparar allocations
current_alloc = {"TLT": 0.1, "IEF": 0.2, "SHY": 0.1, "SPY": 0.6}
aggressive_alloc = {"TLT": 0.3, "IEF": 0.2, "SHY": 0.0, "SPY": 0.5}

ev_current = sum(probabilities[s] * portfolio_payoff(s, current_alloc) 
                 for s in scenarios)
ev_aggressive = sum(probabilities[s] * portfolio_payoff(s, aggressive_alloc) 
                    for s in scenarios)

# Decisión: cambiar a aggressive_alloc si EV mejora significativamente
if ev_aggressive > ev_current + threshold:
    print("REBALANCE TO AGGRESSIVE DURATION")
```

## 📊 Backtesting Framework

### Métricas para Probabilidades

1. **Brier Score**
   ```python
   # Mide accuracy de probabilidades
   brier = mean((prob_predicted - actual_outcome)^2)
   # Rango: [0, 1], menor es mejor
   ```

2. **Log Loss**
   ```python
   log_loss = -mean(actual * log(prob) + (1-actual) * log(1-prob))
   ```

3. **Calibration Plot**
   - Binear probabilidades (0-10%, 10-20%, etc.)
   - Comparar prob predicha vs. frecuencia observada

### Métricas para Trades basados en EV

```python
# Simular estrategia: operar solo si EV > threshold
threshold = 0.005  # 0.5% expected return

for meeting in historical_meetings:
    # Calcular probabilidades out-of-sample
    probs = model.predict_proba(meeting)
    
    # Calcular EV del trade
    ev = calculate_ev(probs, payoff_func)
    
    if ev > threshold:
        # Ejecutar trade
        actual_return = get_actual_return(meeting, trade)
        pnl.append(actual_return)
    else:
        pnl.append(0)  # No trade

# Performance
sharpe = mean(pnl) / std(pnl) * sqrt(8)  # 8 meetings/año
win_rate = sum(pnl > 0) / len(pnl)
avg_win = mean([p for p in pnl if p > 0])
avg_loss = mean([p for p in pnl if p < 0])
```

## 🚀 Instalación y Setup

### Requisitos:
```bash
Python 3.9+
pip
git
```

### Paso 1: Clonar repositorio
```bash
git clone https://github.com/rodrigo/interest-rate-probability-model.git
cd interest-rate-probability-model
```

### Paso 2: Ambiente virtual
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows
```

### Paso 3: Instalar dependencias
```bash
pip install -r requirements.txt
```

### Paso 4: Configurar API Keys

Crear archivo `.env` en la raíz:
```bash
# FRED API (gratis)
FRED_API_KEY=your_fred_api_key_here

# Thetadata (necesitas suscripción)
THETADATA_USERNAME=your_username
THETADATA_PASSWORD=your_password

# Opcional: para notificaciones
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

**Obtener API Keys:**
- FRED: https://fred.stlouisfed.org/docs/api/api_key.html (gratis)
- Thetadata: https://thetadata.net/ (de pago, pero tienes acceso)

### Paso 5: Descargar datos iniciales
```bash
python src/data_collection/fred_api.py --start-date 1990-01-01
python src/data_collection/thetadata_futures.py --contracts ZQ,ZT,ZN
python src/data_collection/fomc_calendar_scraper.py
```

### Paso 6: Entrenar modelos baseline
```bash
python src/probability_models/train_all_models.py
```

### Paso 7: Ejecutar dashboard
```bash
streamlit run dashboard/streamlit_app.py
```

## 📱 Uso del Dashboard

### Pantalla Principal: Probabilidades

- Visualización de probabilidades para próximo meeting
- Comparación: Implícitas (futuros) vs. Modelo vs. Ensemble
- Gráfico de evolución temporal de probabilidades
- Tabla de features importantes

### Calculadora de Expected Value

**Input:**
1. Seleccionar tipo de trade:
   - Long/Short single bond
   - Steepener/Flattener
   - Options strategy
   - Custom portfolio

2. Configurar parámetros:
   - Tamaño de posición
   - Duration / DV01
   - Strike para opciones
   - Fees / slippage

3. Ajustar probabilidades (opcional):
   - Usar probabilidades del modelo
   - Override manual
   - Stress testing

**Output:**
- Expected Value total
- Breakdown por escenario
- Gráfico de distribución de payoffs
- Max gain / Max loss
- Probability of profit
- Comparación risk/reward

### Backtest Explorer

- Seleccionar período histórico
- Elegir estrategia
- Ver equity curve
- Métricas de performance
- Trade log detallado

## 🧪 Uso desde Python

### Ejemplo completo de workflow:

```python
from src.probability_models import EnsembleModel
from src.expected_value import PayoffCalculator, PortfolioEV
from src.data_collection import FREDData, ThetadataFutures
import pandas as pd

# 1. Obtener datos actuales
fred = FREDData()
macro_features = fred.get_latest_features()

theta = ThetadataFutures()
futures_data = theta.get_zq_chain()

# 2. Calcular probabilidades
model = EnsembleModel.load("models/ensemble_v1.pkl")
probabilities = model.predict_proba(macro_features, futures_data)

print(f"Probabilidades próximo FOMC meeting:")
for change, prob in probabilities.items():
    print(f"  {change:+3d}bps: {prob:.1%}")

# 3. Definir trade
def my_trade_payoff(rate_change_bps):
    """
    Ejemplo: Long TLT con stop-loss
    """
    duration = 17
    rate_change_pct = rate_change_bps / 10000
    unrealized_pnl = -duration * rate_change_pct
    
    # Stop loss en -3%
    if unrealized_pnl < -0.03:
        return -0.03
    
    # Take profit en +5%
    if unrealized_pnl > 0.05:
        return 0.05
    
    return unrealized_pnl

# 4. Calcular Expected Value
calculator = PayoffCalculator()
ev, breakdown = calculator.calculate_ev(
    probabilities=probabilities,
    payoff_function=my_trade_payoff
)

print(f"\nExpected Value del trade: {ev:.2%}")
print("\nBreakdown por escenario:")
for scenario, data in breakdown.items():
    print(f"  {scenario:+3d}bps: "
          f"Prob={data['probability']:.1%}, "
          f"Payoff={data['payoff']:+.2%}, "
          f"Contrib={data['contribution']:+.3%}")

# 5. Decisión
THRESHOLD = 0.005  # 0.5% mínimo EV para ejecutar
RISK_LIMIT = 0.03  # 3% máximo riesgo

max_loss = min(breakdown[s]['payoff'] for s in breakdown)

if ev > THRESHOLD and abs(max_loss) <= RISK_LIMIT:
    print(f"\n✅ EJECUTAR TRADE - EV: {ev:.2%}, Max Loss: {max_loss:.2%}")
else:
    print(f"\n❌ NO TRADE - EV: {ev:.2%}, Max Loss: {max_loss:.2%}")
```

## 📈 Ejemplos de Armados Típicos

### 1. Bull Steepener
**Setup:** Esperamos cortes de tasas + curva empinándose
```python
position = {
    "long": {"instrument": "ZN", "quantity": 10, "duration": 8.5},
    "short": {"instrument": "ZT", "quantity": -40, "duration": 2.0}
}
# DV01 neutral, apuestas pura a steepening
```

### 2. Barbell vs. Bullet
**Setup:** Comparar barbell (2Y+10Y) vs. bullet (5Y)
```python
barbell = {"ZT": 0.5, "ZN": 0.5}  # 50% en cada extremo
bullet = {"ZF": 1.0}  # 100% en medio

# Calcular EV de cada uno
ev_barbell = calculate_portfolio_ev(barbell, probabilities)
ev_bullet = calculate_portfolio_ev(bullet, probabilities)

# Barbell típicamente mejor si esperamos volatilidad de tasas
```

### 3. Convexity Trade
**Setup:** Comprar MBS (mortgage-backed securities) con convexidad negativa
```python
# MBS tienen convexidad negativa: pierden más cuando tasas bajan
# Solo atractivo si implícitas sobreestiman probabilidad de cortes

if prob_model["cut"] < prob_futures["cut"] - 0.15:  # 15% edge
    print("Opportunity: Long MBS, prob de corte sobreestimada")
```

## 🔮 Roadmap / TODOs

### Fase 1: MVP (2-3 semanas)
- [x] Setup de proyecto y estructura
- [ ] Descarga automática FRED + Thetadata
- [ ] Probabilidades implícitas desde ZQ
- [ ] Modelo logístico baseline
- [ ] Calculadora simple de EV
- [ ] Dashboard básico en Streamlit

### Fase 2: ML Models (3-4 semanas)
- [ ] Feature engineering avanzado
- [ ] Random Forest + XGBoost
- [ ] Ensemble model
- [ ] Backtesting histórico
- [ ] Calibración de probabilidades
- [ ] Análisis de feature importance

### Fase 3: Payoff Structures (2-3 semanas)
- [ ] Librería de payoff functions
- [ ] Duration analysis preciso
- [ ] Options payoffs (greeks)
- [ ] Spread trades (todas las combinaciones)
- [ ] Portfolio optimizer
- [ ] Risk analytics (VaR, CVaR)

### Fase 4: Production (3-4 semanas)
- [ ] Automatización completa
- [ ] Alertas vía Telegram
- [ ] API REST para integración
- [ ] Real-time data feeds
- [ ] Paper trading integration
- [ ] Dashboard avanzado con scenarios

### Fase 5: Extensiones
- [ ] Trayectorias completas de tasas (no solo próximo meeting)
- [ ] Modelos de volatilidad de tasas
- [ ] Integración con otros bancos centrales
- [ ] NLP sobre FOMC statements
- [ ] Reinforcement learning para timing óptimo

## ⚠️ Consideraciones Importantes

### Limitaciones del Modelo

1. **Sample Size**: ~250 decisiones del FOMC desde 1990
   - Poco data para modelos complejos
   - Regímenes económicos cambian (no es stationary)
   
2. **Structural Breaks**:
   - Crisis 2008 cambió comportamiento Fed
   - COVID-19 → política no convencional
   - Quantitative Easing/Tightening

3. **Data Lag**:
   - GDP: quarterly, con 1 mes de delay
   - Employment: mensual, disponible primer viernes
   - CPI/PCE: mensual, 2 semanas después del mes

4. **Overfitting Risk**:
   - Con pocas observaciones, easy to overfit
   - Usar cross-validation riguroso
   - Preferir modelos simples e interpretables

### Risk Management para Trades

```python
# Nunca operar solo por EV alto
# Chequear también:

def should_trade(ev, probabilities, payoff_function):
    """
    Decision framework completo
    """
    # 1. EV mínimo
    if ev < 0.005:  # 0.5%
        return False, "EV too low"
    
    # 2. Probability of loss
    prob_loss = sum(p for s, p in probabilities.items() 
                    if payoff_function(s) < 0)
    if prob_loss > 0.60:
        return False, "Prob of loss too high"
    
    # 3. Max drawdown
    worst_case = min(payoff_function(s) for s in probabilities)
    if worst_case < -0.05:  # -5%
        return False, "Max loss exceeds limit"
    
    # 4. Risk/Reward ratio
    expected_gain = sum(p * payoff_function(s) 
                       for s, p in probabilities.items() 
                       if payoff_function(s) > 0)
    expected_loss = sum(p * payoff_function(s) 
                       for s, p in probabilities.items() 
                       if payoff_function(s) < 0)
    
    if abs(expected_gain / expected_loss) < 2:
        return False, "Risk/reward ratio < 2"
    
    # 5. Model confidence
    entropy = -sum(p * np.log(p) for p in probabilities.values() if p > 0)
    max_entropy = np.log(len(probabilities))
    confidence = 1 - entropy / max_entropy
    
    if confidence < 0.3:
        return False, "Model too uncertain"
    
    return True, "All checks passed"
```

## 📚 Referencias

### Papers Académicos
1. **Gürkaynak, Sack & Swanson (2005)**: "Do Actions Speak Louder Than Words?"
2. **Hamilton (2009)**: "Daily Monetary Policy Shocks and New Home Sales"
3. **Piazzesi & Swanson (2008)**: "Futures Prices as Risk-Adjusted Forecasts"
4. **Cieslak & Povala (2015)**: "Expected Returns in Treasury Bonds"

### Recursos Online
- **CME FedWatch Tool**: https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html
- **FRED Database**: https://fred.stlouisfed.org/
- **Fed Monetary Policy**: https://www.federalreserve.gov/monetarypolicy.htm
- **Thetadata Docs**: https://http-docs.thetadata.us/

### Libros Recomendados
- **"Fixed Income Securities" - Tuckman & Serrat**: Biblia de renta fija
- **"The Federal Reserve System" - Carlson**: Historia y funcionamiento de la Fed
- **"Trading and Pricing Financial Derivatives" - Joshi**: Para opciones
