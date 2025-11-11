"""
Package initialization for PSO Optimizer

Author: EU97
Date: 2024
"""

__version__ = "1.0.0"
__author__ = "EU97"

from .pso import ParticleSwarmOptimizer, PSOConfig, Particle
from .portfolio import PortfolioOptimizer, generate_sample_data, load_real_stock_data
from .visualization import (
    plot_convergence,
    plot_portfolio_allocation,
    plot_efficient_frontier,
    plot_portfolio_metrics_comparison
)

__all__ = [
    'ParticleSwarmOptimizer',
    'PSOConfig',
    'Particle',
    'PortfolioOptimizer',
    'generate_sample_data',
    'load_real_stock_data',
    'plot_convergence',
    'plot_portfolio_allocation',
    'plot_efficient_frontier',
    'plot_portfolio_metrics_comparison'
]
