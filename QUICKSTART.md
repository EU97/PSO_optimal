# Quick Start Guide

Get up and running with PSO Portfolio Optimizer in minutes!

## 🚀 Fastest Way to Start

### Windows Users

1. Open PowerShell in the project directory
2. Run the startup script:
```powershell
.\start.ps1
```
3. Choose your setup method (Docker or Local)
4. Access the app at `http://localhost:8501`

### Docker Users (All Platforms)

```bash
# Build and start
docker-compose up --build

# Access at http://localhost:8501
```

### Python Users (All Platforms)

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app/main.py
```

## 📋 What's Included

✅ Complete PSO implementation  
✅ Portfolio optimization for stocks  
✅ Interactive web interface (Streamlit)  
✅ Real-time visualization  
✅ Docker containerization  
✅ Comprehensive documentation  
✅ Unit tests  
✅ Sample data  

## 🎯 First Steps

1. **Choose Data Source**
   - Use synthetic data to experiment
   - Or select demo stocks (AAPL, GOOGL, etc.)

2. **Configure Parameters**
   - Adjust PSO settings in sidebar
   - Set portfolio constraints
   - Choose risk preferences

3. **Run Optimization**
   - Click "Run Optimization"
   - Watch progress in real-time
   - Analyze results

4. **Export Results**
   - Download optimal portfolio
   - Save performance metrics
   - Export visualizations

## 📚 Learn More

- [Full README](README.md)
- [API Documentation](docs/API.md)
- [Usage Guide](docs/USAGE.md)
- [Theory Background](docs/THEORY.md)

## 🆘 Need Help?

- Check [Troubleshooting](docs/USAGE.md#troubleshooting)
- Open an [Issue](https://github.com/EU97/PSO_optimal/issues)
- Read the [Documentation](docs/)

## ⚡ Quick Example

```python
from src.pso import ParticleSwarmOptimizer, PSOConfig
from src.portfolio import generate_sample_data, PortfolioOptimizer
import numpy as np

# Generate data
prices, returns, cov = generate_sample_data(n_assets=5)

# Create portfolio
portfolio = PortfolioOptimizer(returns, cov)

# Configure PSO
config = PSOConfig(
    n_particles=30,
    n_dimensions=5,
    n_iterations=100,
    bounds=(np.zeros(5), np.ones(5))
)

# Optimize
pso = ParticleSwarmOptimizer(config)
weights, fitness, history = pso.optimize(portfolio.objective_function)

# Results
weights = portfolio.normalize_weights(weights)
metrics = portfolio.get_portfolio_metrics(weights)
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
```

Happy optimizing! 🐝📊
