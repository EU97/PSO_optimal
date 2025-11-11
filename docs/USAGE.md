# Usage Guide

Comprehensive guide for using the PSO Portfolio Optimizer.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Basic Usage](#basic-usage)
4. [Advanced Usage](#advanced-usage)
5. [Web Interface](#web-interface)
6. [Custom Optimization](#custom-optimization)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Installation

### Using Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/EU97/PSO_optimal.git
cd PSO_optimal

# Build and run
docker-compose up --build

# Access application at http://localhost:8501
```

### Local Installation

```bash
# Clone repository
git clone https://github.com/EU97/PSO_optimal.git
cd PSO_optimal

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app/main.py
```

---

## Quick Start

### 1. Generate Sample Data

```python
from src.portfolio import generate_sample_data

# Generate synthetic portfolio data
prices, expected_returns, cov_matrix = generate_sample_data(
    n_assets=5,
    n_days=252
)

print("Expected Returns:", expected_returns)
print("Covariance Matrix Shape:", cov_matrix.shape)
```

### 2. Create Portfolio Optimizer

```python
from src.portfolio import PortfolioOptimizer

portfolio = PortfolioOptimizer(
    expected_returns=expected_returns,
    covariance_matrix=cov_matrix,
    risk_free_rate=0.02,
    risk_aversion=0.5
)
```

### 3. Configure PSO

```python
from src.pso import ParticleSwarmOptimizer, PSOConfig
import numpy as np

config = PSOConfig(
    n_particles=30,
    n_dimensions=5,
    n_iterations=100,
    bounds=(np.zeros(5), np.ones(5)),
    adaptive_inertia=True
)

pso = ParticleSwarmOptimizer(config)
```

### 4. Optimize

```python
# Run optimization
best_weights, best_fitness, history = pso.optimize(
    portfolio.objective_function
)

# Normalize weights
best_weights = portfolio.normalize_weights(best_weights)

# Get metrics
metrics = portfolio.get_portfolio_metrics(best_weights)

print("Optimal Allocation:", best_weights)
print("Sharpe Ratio:", metrics['sharpe_ratio'])
```

---

## Basic Usage

### Portfolio Optimization from Scratch

```python
import numpy as np
from src.portfolio import PortfolioOptimizer
from src.pso import ParticleSwarmOptimizer, PSOConfig

# Define your portfolio
expected_returns = np.array([0.12, 0.18, 0.10, 0.15, 0.14])
covariance_matrix = np.array([
    [0.04, 0.01, 0.02, 0.01, 0.015],
    [0.01, 0.09, 0.03, 0.02, 0.025],
    [0.02, 0.03, 0.06, 0.01, 0.020],
    [0.01, 0.02, 0.01, 0.03, 0.015],
    [0.015, 0.025, 0.020, 0.015, 0.05]
])

# Create optimizer
portfolio = PortfolioOptimizer(
    expected_returns=expected_returns,
    covariance_matrix=covariance_matrix,
    risk_free_rate=0.02
)

# Configure PSO
config = PSOConfig(
    n_particles=30,
    n_dimensions=5,
    n_iterations=100,
    bounds=(np.zeros(5), np.ones(5))
)

# Optimize
pso = ParticleSwarmOptimizer(config)
best_weights, _, history = pso.optimize(portfolio.objective_function)

# Results
best_weights = portfolio.normalize_weights(best_weights)
print("\nOptimal Portfolio Allocation:")
for i, weight in enumerate(best_weights):
    print(f"Asset {i+1}: {weight*100:.2f}%")
```

### Using Real Stock Data

```python
from src.portfolio import load_real_stock_data

# Load data from Yahoo Finance
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
prices, expected_returns, cov_matrix = load_real_stock_data(
    tickers=tickers,
    start_date='2020-01-01',
    end_date='2024-01-01'
)

# Continue with optimization...
portfolio = PortfolioOptimizer(
    expected_returns=expected_returns,
    covariance_matrix=cov_matrix
)
```

### Visualizing Results

```python
from src.visualization import (
    plot_convergence,
    plot_portfolio_allocation,
    plot_efficient_frontier
)

# Plot convergence
plot_convergence(history, interactive=True)

# Plot allocation
asset_names = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
plot_portfolio_allocation(best_weights, asset_names, interactive=True)

# Plot efficient frontier
plot_efficient_frontier(
    expected_returns,
    cov_matrix,
    best_weights,
    interactive=True
)
```

---

## Advanced Usage

### Custom Constraints

```python
# Set minimum and maximum weights
portfolio = PortfolioOptimizer(
    expected_returns=expected_returns,
    covariance_matrix=cov_matrix,
    min_weight=0.05,  # At least 5% in each asset
    max_weight=0.40   # Maximum 40% in any asset
)
```

### Adaptive Inertia Weight

```python
# Enable adaptive inertia for better convergence
config = PSOConfig(
    n_particles=30,
    n_dimensions=5,
    n_iterations=100,
    adaptive_inertia=True,
    w_min=0.4,
    w_max=0.9
)
```

### Custom Callback Function

```python
def my_callback(iteration, best_fitness, best_position):
    if iteration % 10 == 0:
        print(f"Iteration {iteration}: Fitness = {best_fitness:.6f}")
        
        # Early stopping
        if best_fitness < -2.0:  # Very good Sharpe ratio
            print("Early stopping: Excellent solution found")
            return False  # Stop optimization
    
    return True  # Continue

pso.optimize(portfolio.objective_function, callback=my_callback)
```

### Multiple Strategies Comparison

```python
# Equal-weight portfolio
equal_weights = np.ones(n_assets) / n_assets
equal_metrics = portfolio.get_portfolio_metrics(equal_weights)

# Minimum variance portfolio
from scipy.optimize import minimize

def variance(weights):
    return portfolio.portfolio_volatility(weights)

result = minimize(
    variance,
    x0=equal_weights,
    bounds=[(0, 1)] * n_assets,
    constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
)
min_var_weights = result.x
min_var_metrics = portfolio.get_portfolio_metrics(min_var_weights)

# PSO optimal
pso_metrics = portfolio.get_portfolio_metrics(best_weights)

# Compare
from src.visualization import plot_portfolio_metrics_comparison

strategies = {
    'Equal-Weight': equal_metrics,
    'Min Variance': min_var_metrics,
    'PSO Optimal': pso_metrics
}

plot_portfolio_metrics_comparison(strategies, interactive=True)
```

### Parameter Sensitivity Analysis

```python
import pandas as pd

results = []

# Test different particle counts
for n_particles in [10, 20, 30, 50]:
    config = PSOConfig(
        n_particles=n_particles,
        n_dimensions=5,
        n_iterations=100,
        bounds=(np.zeros(5), np.ones(5))
    )
    
    pso = ParticleSwarmOptimizer(config)
    weights, fitness, history = pso.optimize(portfolio.objective_function)
    
    results.append({
        'n_particles': n_particles,
        'final_fitness': history['best_fitness'][-1],
        'iterations_to_converge': len(history['best_fitness'])
    })

# Analyze results
df_results = pd.DataFrame(results)
print(df_results)
```

---

## Web Interface

### Starting the Application

```bash
# With Docker
docker-compose up

# Local
streamlit run app/main.py
```

### Using the Interface

1. **Configure Data Source**
   - Choose between synthetic or demo stock data
   - Adjust number of assets (synthetic only)

2. **Set PSO Parameters**
   - Number of particles (10-100)
   - Number of iterations (50-300)
   - Enable adaptive inertia
   - Adjust cognitive/social parameters

3. **Configure Portfolio Settings**
   - Set risk-free rate
   - Adjust risk aversion
   - Set allocation constraints

4. **Run Optimization**
   - Click "Run Optimization"
   - Monitor progress in real-time
   - View results in multiple tabs

5. **Analyze Results**
   - View optimal allocation
   - Examine convergence plots
   - Compare strategies
   - Export results

### Exporting Results

The web interface allows you to download:
- **CSV**: Allocation and metrics data
- **TXT**: Formatted summary report
- **Images**: Save plots as PNG files

---

## Custom Optimization

### Implementing Custom Objective Functions

```python
class CustomPortfolioOptimizer(PortfolioOptimizer):
    def custom_objective(self, weights):
        """
        Custom objective: Maximize return/volatility ratio 
        with downside risk penalty
        """
        weights = self.normalize_weights(weights)
        
        # Calculate metrics
        ret = self.portfolio_return(weights)
        vol = self.portfolio_volatility(weights)
        
        # Downside deviation (simplified)
        downside_vol = vol * 1.5  # Penalize downside more
        
        # Custom metric
        custom_metric = ret / downside_vol
        
        # Minimize negative metric
        return -custom_metric

# Use custom objective
pso = ParticleSwarmOptimizer(config)
best_weights, _, history = pso.optimize(
    portfolio.custom_objective
)
```

### Multi-Objective Optimization

```python
def pareto_objective(weights):
    """
    Multi-objective: Balance return and risk
    """
    weights = portfolio.normalize_weights(weights)
    
    ret = portfolio.portfolio_return(weights)
    vol = portfolio.portfolio_volatility(weights)
    
    # Weighted combination
    alpha = 0.7  # Preference for return
    objective = -alpha * ret + (1 - alpha) * vol
    
    return objective

pso = ParticleSwarmOptimizer(config)
best_weights, _, _ = pso.optimize(pareto_objective)
```

---

## Best Practices

### 1. Data Preparation

- Use at least 1-2 years of historical data
- Handle missing data appropriately
- Consider market regimes (bull/bear)
- Adjust for corporate actions (splits, dividends)

### 2. Parameter Selection

**Particle Count:**
- Small problems (3-5 assets): 20-30 particles
- Medium problems (5-10 assets): 30-50 particles
- Large problems (10+ assets): 50-100 particles

**Iterations:**
- Start with 100 iterations
- Monitor convergence plots
- Increase if not converged
- Use early stopping for efficiency

**Inertia Weight:**
- Use adaptive inertia (recommended)
- w_max = 0.9, w_min = 0.4
- Or fixed w = 0.7 for stable problems

### 3. Constraint Handling

```python
# Diversification: Limit max position
portfolio = PortfolioOptimizer(
    expected_returns=returns,
    covariance_matrix=cov,
    max_weight=0.30  # No more than 30% in any asset
)

# Minimum allocation: Avoid tiny positions
portfolio = PortfolioOptimizer(
    expected_returns=returns,
    covariance_matrix=cov,
    min_weight=0.05  # At least 5% or nothing
)
```

### 4. Validation

```python
# Always validate final weights
from app.utils import validate_weights

if validate_weights(best_weights):
    print("✓ Valid portfolio")
else:
    print("✗ Invalid portfolio - check constraints")

# Back-test on out-of-sample data
# Calculate realized returns
# Compare to benchmark
```

### 5. Risk Management

```python
# Calculate comprehensive risk metrics
metrics = portfolio.get_portfolio_metrics(best_weights)

print(f"Expected Return: {metrics['expected_return']:.2%}")
print(f"Volatility: {metrics['volatility']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")

# Set risk limits
if metrics['volatility'] > 0.25:
    print("Warning: High volatility portfolio")

if metrics['sharpe_ratio'] < 0.5:
    print("Warning: Low risk-adjusted returns")
```

---

## Troubleshooting

### Common Issues

**1. PSO Not Converging**

```python
# Solution: Increase iterations or particles
config = PSOConfig(
    n_particles=50,  # More particles
    n_iterations=200,  # More iterations
    adaptive_inertia=True
)
```

**2. Invalid Weights (Don't Sum to 1)**

```python
# Always normalize final weights
best_weights = portfolio.normalize_weights(best_weights)
```

**3. Poor Sharpe Ratio**

```python
# Check input data quality
print("Returns:", expected_returns)
print("Mean return:", np.mean(expected_returns))
print("Volatility range:", np.sqrt(np.diag(cov_matrix)))

# Adjust risk-free rate
portfolio = PortfolioOptimizer(
    expected_returns=returns,
    covariance_matrix=cov,
    risk_free_rate=0.01  # Lower rate
)
```

**4. Docker Issues**

```bash
# Rebuild containers
docker-compose down
docker-compose up --build

# Check logs
docker-compose logs pso-app

# Access shell
docker-compose exec pso-app /bin/bash
```

**5. Import Errors**

```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# Check Python version
python --version  # Should be 3.9+

# Reinstall specific package
pip install --upgrade streamlit
```

### Getting Help

- **Documentation**: See [API.md](API.md) and [THEORY.md](THEORY.md)
- **Issues**: [GitHub Issues](https://github.com/EU97/PSO_optimal/issues)
- **Examples**: Check `notebooks/` directory

---

## Next Steps

- Read [THEORY.md](THEORY.md) for mathematical background
- Explore example notebooks
- Experiment with real stock data
- Customize for your specific use case
- Contribute improvements via Pull Requests

Happy optimizing! 🚀
