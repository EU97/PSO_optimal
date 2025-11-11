# PSO Optimizer - Portfolio Optimization with Particle Swarm Optimization

A comprehensive Python implementation of Particle Swarm Optimization (PSO) for solving real-world optimization problems, featuring an interactive web interface and Docker containerization.

## 🎯 Project Overview

This project demonstrates PSO optimization applied to **Portfolio Optimization** - a real-world financial problem where we optimize asset allocation to maximize returns while minimizing risk.

### Real-World Use Case: Stock Portfolio Allocation

The application optimizes investment allocation across multiple stocks by:
- Maximizing expected returns
- Minimizing portfolio risk (variance)
- Respecting investment constraints (weights sum to 1, no negative allocations)

## ✨ Features

- 🐝 **Advanced PSO Implementation** with configurable parameters
- 📊 **Interactive Visualization** using Streamlit
- 🐳 **Docker Containerization** for isolated, reproducible environments
- 📈 **Real-time Convergence Plots** showing optimization progress
- 🎨 **Modern UI** with parameter controls and result visualization
- 📚 **Comprehensive Documentation** and examples
- 🧪 **Unit Tests** for reliability

## 🏗️ Project Structure

```
PSO_optimal/
├── app/
│   ├── __init__.py
│   ├── main.py              # Streamlit web interface
│   └── utils.py             # Utility functions
├── src/
│   ├── __init__.py
│   ├── pso.py               # Core PSO algorithm
│   ├── portfolio.py         # Portfolio optimization problem
│   └── visualization.py     # Plotting and visualization
├── data/
│   ├── stock_data.csv       # Sample stock price data
│   └── README.md            # Data description
├── tests/
│   ├── __init__.py
│   ├── test_pso.py
│   └── test_portfolio.py
├── notebooks/
│   └── PSO_Tutorial.ipynb   # Interactive tutorial
├── docs/
│   ├── API.md               # API documentation
│   ├── USAGE.md             # Usage guide
│   └── THEORY.md            # PSO theory and background
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
├── .gitignore
└── README.md
```

## 🚀 Quick Start

### Option 1: Using Docker (Recommended)

1. **Clone the repository:**
```bash
git clone https://github.com/EU97/PSO_optimal.git
cd PSO_optimal
```

2. **Build and run with Docker Compose:**
```bash
docker-compose up --build
```

3. **Access the application:**
Open your browser and navigate to `http://localhost:8501`

### Option 2: Local Installation

1. **Clone and navigate to the project:**
```bash
git clone https://github.com/EU97/PSO_optimal.git
cd PSO_optimal
```

2. **Create virtual environment:**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the application:**
```bash
streamlit run app/main.py
```

## 📖 Usage Examples

### Basic PSO Optimization

```python
from src.pso import ParticleSwarmOptimizer
from src.portfolio import PortfolioOptimizer

# Initialize portfolio optimizer
portfolio = PortfolioOptimizer(
    returns=[0.12, 0.18, 0.10, 0.15],  # Expected returns
    cov_matrix=cov_matrix,              # Covariance matrix
    risk_aversion=0.5                   # Risk tolerance
)

# Create PSO optimizer
pso = ParticleSwarmOptimizer(
    n_particles=30,
    n_dimensions=4,
    n_iterations=100,
    w=0.7,      # Inertia weight
    c1=1.5,     # Cognitive parameter
    c2=1.5      # Social parameter
)

# Optimize
best_position, best_fitness, history = pso.optimize(portfolio.objective_function)

print(f"Optimal allocation: {best_position}")
print(f"Sharpe ratio: {-best_fitness:.4f}")
```

### Web Interface

The Streamlit interface provides:
- **Parameter Controls:** Adjust PSO parameters in real-time
- **Stock Selection:** Choose stocks to include in portfolio
- **Visualization:** See convergence plots and allocation charts
- **Results Export:** Download optimal portfolios

## 🧮 PSO Algorithm Details

Particle Swarm Optimization is a population-based metaheuristic inspired by social behavior of bird flocking. Each particle represents a potential solution that moves through the search space influenced by:

1. **Inertia:** Particle's current velocity
2. **Cognitive Component:** Particle's best-known position
3. **Social Component:** Swarm's best-known position

**Update Equations:**
```
v(t+1) = w * v(t) + c1 * r1 * (pbest - x(t)) + c2 * r2 * (gbest - x(t))
x(t+1) = x(t) + v(t+1)
```

Where:
- `w` = inertia weight
- `c1`, `c2` = acceleration coefficients
- `r1`, `r2` = random numbers in [0,1]
- `pbest` = particle's best position
- `gbest` = global best position

## 📊 Portfolio Optimization Problem

### Objective Function

Maximize the Sharpe Ratio:
```
Sharpe Ratio = (Expected Return - Risk-Free Rate) / Portfolio Standard Deviation
```

### Constraints

1. Weights sum to 1: `Σ wi = 1`
2. Non-negative weights: `wi ≥ 0` (no short selling)
3. Optional: Maximum position size: `wi ≤ max_weight`

### Data

Sample data includes historical returns for major stocks (AAPL, GOOGL, MSFT, AMZN, TSLA) from 2020-2024.

## 🧪 Testing

Run unit tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## 📚 Documentation

- [API Documentation](docs/API.md) - Detailed API reference
- [Usage Guide](docs/USAGE.md) - Comprehensive usage examples
- [Theory](docs/THEORY.md) - PSO theory and mathematics

## 🔧 Configuration

### PSO Parameters

- **n_particles:** Number of particles in swarm (default: 30)
- **n_iterations:** Maximum iterations (default: 100)
- **w:** Inertia weight (default: 0.7, range: 0.4-0.9)
- **c1:** Cognitive parameter (default: 1.5, range: 0-2)
- **c2:** Social parameter (default: 1.5, range: 0-2)

### Portfolio Parameters

- **risk_aversion:** Risk tolerance (0=risk-neutral, 1=risk-averse)
- **min_weight:** Minimum allocation per asset (default: 0)
- **max_weight:** Maximum allocation per asset (default: 1)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PSO algorithm based on Kennedy & Eberhart (1995)
- Portfolio theory based on Modern Portfolio Theory (Markowitz, 1952)

## 📞 Contact

- GitHub: [@EU97](https://github.com/EU97)
- Project Link: [https://github.com/EU97/PSO_optimal](https://github.com/EU97/PSO_optimal)

## 🔮 Future Enhancements

- [ ] Multi-objective optimization (Pareto frontier)
- [ ] Real-time data integration (APIs)
- [ ] Machine learning for parameter tuning
- [ ] Alternative optimization algorithms comparison
- [ ] Advanced constraints (sector limits, ESG factors)
- [ ] Backtesting framework
