# API Documentation

Complete API reference for PSO Portfolio Optimizer.

## Table of Contents

- [Core Modules](#core-modules)
  - [pso.py](#psopy)
  - [portfolio.py](#portfoliopy)
  - [visualization.py](#visualizationpy)
- [Application Modules](#application-modules)
  - [app/main.py](#appmainpy)
  - [app/utils.py](#apputilspy)

---

## Core Modules

### pso.py

#### Class: `PSOConfig`

Configuration dataclass for PSO algorithm parameters.

**Attributes:**

```python
n_particles: int = 30          # Number of particles in swarm
n_dimensions: int = 4          # Dimensionality of search space
n_iterations: int = 100        # Maximum iterations
w: float = 0.7                 # Inertia weight
c1: float = 1.5                # Cognitive parameter
c2: float = 1.5                # Social parameter
w_min: float = 0.4             # Minimum inertia (adaptive)
w_max: float = 0.9             # Maximum inertia (adaptive)
bounds: Optional[Tuple] = None # (lower_bounds, upper_bounds)
adaptive_inertia: bool = False # Enable adaptive inertia
verbose: bool = True           # Enable logging
```

---

#### Class: `Particle`

Represents a single particle in the swarm.

**Constructor:**

```python
Particle(n_dimensions: int, bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None)
```

**Parameters:**
- `n_dimensions`: Dimensionality of the search space
- `bounds`: Optional tuple of (lower_bounds, upper_bounds)

**Attributes:**
- `position`: Current position in search space (np.ndarray)
- `velocity`: Current velocity (np.ndarray)
- `best_position`: Personal best position (np.ndarray)
- `best_fitness`: Fitness at personal best (float)

**Methods:**

```python
update_velocity(global_best_position: np.ndarray, w: float, c1: float, c2: float) -> None
```
Update particle velocity using PSO equations.

```python
update_position(bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> None
```
Update particle position based on velocity.

```python
evaluate(fitness_func: Callable) -> float
```
Evaluate fitness and update personal best.

---

#### Class: `ParticleSwarmOptimizer`

Main PSO optimization algorithm.

**Constructor:**

```python
ParticleSwarmOptimizer(config: Optional[PSOConfig] = None, **kwargs)
```

**Parameters:**
- `config`: PSOConfig object with algorithm parameters
- `**kwargs`: Alternative way to provide parameters

**Attributes:**
- `config`: Configuration object
- `particles`: List of Particle objects
- `global_best_position`: Best position found by swarm
- `global_best_fitness`: Best fitness value
- `history`: Dictionary of convergence history

**Methods:**

```python
optimize(
    fitness_func: Callable[[np.ndarray], float],
    callback: Optional[Callable[[int, float, np.ndarray], None]] = None
) -> Tuple[np.ndarray, float, Dict[str, List]]
```

Run PSO optimization.

**Parameters:**
- `fitness_func`: Objective function to minimize
- `callback`: Optional callback function called each iteration

**Returns:**
- `best_position`: Optimal solution found
- `best_fitness`: Fitness at optimal solution
- `history`: Dictionary containing:
  - `best_fitness`: Best fitness per iteration
  - `mean_fitness`: Mean fitness per iteration
  - `std_fitness`: Standard deviation per iteration
  - `diversity`: Swarm diversity per iteration

**Example:**

```python
from src.pso import ParticleSwarmOptimizer, PSOConfig
import numpy as np

# Define objective function
def sphere(x):
    return np.sum(x**2)

# Configure PSO
config = PSOConfig(
    n_particles=30,
    n_dimensions=5,
    n_iterations=100,
    bounds=(np.array([-5.0]*5), np.array([5.0]*5)),
    adaptive_inertia=True
)

# Optimize
pso = ParticleSwarmOptimizer(config)
best_pos, best_fit, history = pso.optimize(sphere)

print(f"Best position: {best_pos}")
print(f"Best fitness: {best_fit}")
```

---

### portfolio.py

#### Class: `PortfolioOptimizer`

Portfolio optimization for PSO.

**Constructor:**

```python
PortfolioOptimizer(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    risk_free_rate: float = 0.02,
    risk_aversion: float = 0.5,
    min_weight: float = 0.0,
    max_weight: float = 1.0
)
```

**Parameters:**
- `expected_returns`: Expected return for each asset (annualized)
- `covariance_matrix`: Covariance matrix of returns (annualized)
- `risk_free_rate`: Risk-free rate (default: 0.02)
- `risk_aversion`: Risk aversion parameter 0-1 (default: 0.5)
- `min_weight`: Minimum allocation per asset (default: 0.0)
- `max_weight`: Maximum allocation per asset (default: 1.0)

**Methods:**

```python
normalize_weights(weights: np.ndarray) -> np.ndarray
```
Normalize weights to sum to 1 and respect bounds.

```python
portfolio_return(weights: np.ndarray) -> float
```
Calculate expected portfolio return.

```python
portfolio_volatility(weights: np.ndarray) -> float
```
Calculate portfolio volatility (standard deviation).

```python
sharpe_ratio(weights: np.ndarray) -> float
```
Calculate Sharpe ratio (risk-adjusted return).

```python
objective_function(weights: np.ndarray) -> float
```
Objective function to minimize (negative Sharpe ratio).

```python
get_portfolio_metrics(weights: np.ndarray) -> dict
```
Calculate comprehensive portfolio metrics.

**Returns dictionary:**
- `weights`: Normalized weights
- `expected_return`: Expected return
- `volatility`: Portfolio volatility
- `sharpe_ratio`: Sharpe ratio
- `max_drawdown`: Estimated max drawdown

**Example:**

```python
from src.portfolio import PortfolioOptimizer
from src.pso import ParticleSwarmOptimizer, PSOConfig
import numpy as np

# Define portfolio
expected_returns = np.array([0.12, 0.18, 0.10, 0.15])
cov_matrix = np.array([
    [0.04, 0.01, 0.02, 0.01],
    [0.01, 0.09, 0.03, 0.02],
    [0.02, 0.03, 0.06, 0.01],
    [0.01, 0.02, 0.01, 0.03]
])

portfolio = PortfolioOptimizer(
    expected_returns=expected_returns,
    covariance_matrix=cov_matrix,
    risk_free_rate=0.02
)

# Optimize with PSO
config = PSOConfig(
    n_particles=30,
    n_dimensions=4,
    n_iterations=100,
    bounds=(np.zeros(4), np.ones(4))
)

pso = ParticleSwarmOptimizer(config)
best_weights, best_fitness, history = pso.optimize(portfolio.objective_function)

# Get metrics
best_weights = portfolio.normalize_weights(best_weights)
metrics = portfolio.get_portfolio_metrics(best_weights)

print(f"Optimal allocation: {best_weights}")
print(f"Sharpe ratio: {metrics['sharpe_ratio']:.4f}")
```

---

#### Function: `generate_sample_data`

Generate synthetic stock price data for testing.

```python
generate_sample_data(
    n_assets: int = 5,
    n_days: int = 252,
    mean_return_range: Tuple[float, float] = (0.05, 0.20),
    volatility_range: Tuple[float, float] = (0.15, 0.40)
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]
```

**Returns:**
- `price_dataframe`: Historical prices
- `expected_returns`: Annualized expected returns
- `covariance_matrix`: Annualized covariance matrix

---

#### Function: `load_real_stock_data`

Load real stock data from Yahoo Finance.

```python
load_real_stock_data(
    tickers: List[str],
    start_date: str,
    end_date: str
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]
```

**Parameters:**
- `tickers`: List of stock ticker symbols
- `start_date`: Start date (YYYY-MM-DD)
- `end_date`: End date (YYYY-MM-DD)

**Returns:**
- `price_dataframe`: Historical adjusted close prices
- `expected_returns`: Annualized expected returns
- `covariance_matrix`: Annualized covariance matrix

**Example:**

```python
from src.portfolio import load_real_stock_data

tickers = ['AAPL', 'GOOGL', 'MSFT']
prices, returns, cov = load_real_stock_data(
    tickers=tickers,
    start_date='2020-01-01',
    end_date='2023-12-31'
)
```

---

### visualization.py

#### Function: `plot_convergence`

Plot PSO convergence history.

```python
plot_convergence(
    history: Dict[str, List],
    title: str = "PSO Convergence",
    show_std: bool = True,
    interactive: bool = False
) -> Optional[go.Figure]
```

**Parameters:**
- `history`: Dictionary containing convergence history
- `title`: Plot title
- `show_std`: Show standard deviation bands
- `interactive`: Return Plotly figure (True) or show matplotlib (False)

**Returns:**
- Plotly Figure if interactive=True, else None

---

#### Function: `plot_portfolio_allocation`

Plot portfolio allocation as pie and bar charts.

```python
plot_portfolio_allocation(
    weights: np.ndarray,
    asset_names: Optional[List[str]] = None,
    title: str = "Portfolio Allocation",
    interactive: bool = False
) -> Optional[go.Figure]
```

---

#### Function: `plot_efficient_frontier`

Plot efficient frontier with optimal portfolio.

```python
plot_efficient_frontier(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    optimal_weights: np.ndarray,
    n_portfolios: int = 1000,
    risk_free_rate: float = 0.02,
    interactive: bool = False
) -> Optional[go.Figure]
```

---

#### Function: `plot_portfolio_metrics_comparison`

Compare metrics across different strategies.

```python
plot_portfolio_metrics_comparison(
    metrics_dict: Dict[str, Dict],
    interactive: bool = False
) -> Optional[go.Figure]
```

**Parameters:**
- `metrics_dict`: Dictionary of {strategy_name: metrics_dict}

---

## Application Modules

### app/utils.py

#### Function: `format_percentage`

```python
format_percentage(value: float, decimals: int = 2) -> str
```

Format decimal as percentage string.

#### Function: `format_currency`

```python
format_currency(value: float, currency: str = "$", decimals: int = 2) -> str
```

Format value as currency string.

#### Function: `save_results_to_csv`

```python
save_results_to_csv(
    weights: np.ndarray,
    asset_names: List[str],
    metrics: Dict[str, Any]
) -> str
```

Save optimization results to CSV format.

#### Function: `validate_weights`

```python
validate_weights(weights: np.ndarray, tolerance: float = 1e-6) -> bool
```

Validate portfolio weights are valid.

---

## Error Handling

All functions include appropriate error handling:

- **ValueError**: Invalid parameters or dimensions
- **AssertionError**: Constraint violations
- **ImportError**: Missing optional dependencies

Example:

```python
try:
    portfolio = PortfolioOptimizer(returns, cov)
    pso = ParticleSwarmOptimizer(config)
    result = pso.optimize(portfolio.objective_function)
except ValueError as e:
    print(f"Invalid input: {e}")
except Exception as e:
    print(f"Optimization error: {e}")
```

---

## Type Hints

All modules use comprehensive type hints for better IDE support and code clarity.

```python
from typing import Tuple, List, Optional, Dict, Callable, Any
import numpy as np
import pandas as pd
```

---

## Logging

The package uses Python's logging module:

```python
import logging

# Configure logging level
logging.basicConfig(level=logging.INFO)

# Or per module
logger = logging.getLogger(__name__)
```
