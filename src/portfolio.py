"""
Portfolio Optimization using PSO

This module implements portfolio optimization as an objective function for PSO.
The goal is to find optimal asset allocation that maximizes returns while
minimizing risk (Sharpe ratio optimization).

Author: EU97
Date: 2025
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PortfolioConfig:
    """Configuration for portfolio optimization."""
    expected_returns: np.ndarray  # Expected return for each asset
    covariance_matrix: np.ndarray  # Covariance matrix of returns
    risk_free_rate: float = 0.02  # Risk-free rate (e.g., treasury bonds)
    risk_aversion: float = 0.5  # Risk aversion parameter (0=risk-neutral, 1=risk-averse)
    min_weight: float = 0.0  # Minimum allocation per asset
    max_weight: float = 1.0  # Maximum allocation per asset
    allow_short: bool = False  # Allow short positions


class PortfolioOptimizer:
    """
    Portfolio optimization problem for PSO.
    
    This class defines the objective function for optimizing a portfolio of assets.
    The optimization seeks to maximize the Sharpe ratio (risk-adjusted returns).
    """
    
    def __init__(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_free_rate: float = 0.02,
        risk_aversion: float = 0.5,
        min_weight: float = 0.0,
        max_weight: float = 1.0
    ):
        """
        Initialize portfolio optimizer.
        
        Args:
            expected_returns: Array of expected returns for each asset
            covariance_matrix: Covariance matrix of asset returns
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            risk_aversion: Risk aversion parameter (0-1)
            min_weight: Minimum weight for each asset
            max_weight: Maximum weight for each asset
        """
        self.expected_returns = np.array(expected_returns)
        self.covariance_matrix = np.array(covariance_matrix)
        self.risk_free_rate = risk_free_rate
        self.risk_aversion = risk_aversion
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.n_assets = len(expected_returns)
        
        # Validate inputs
        assert len(self.expected_returns) == self.covariance_matrix.shape[0], \
            "Returns and covariance matrix dimensions mismatch"
        assert self.covariance_matrix.shape[0] == self.covariance_matrix.shape[1], \
            "Covariance matrix must be square"
    
    def normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """
        Normalize weights to sum to 1 and respect bounds.
        
        Args:
            weights: Raw weight vector
            
        Returns:
            Normalized weights
        """
        # Clip to bounds
        weights = np.clip(weights, self.min_weight, self.max_weight)
        
        # Normalize to sum to 1
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            # If all weights are zero, distribute equally
            weights = np.ones(self.n_assets) / self.n_assets
        
        return weights
    
    def portfolio_return(self, weights: np.ndarray) -> float:
        """
        Calculate expected portfolio return.
        
        Args:
            weights: Asset allocation weights
            
        Returns:
            Expected return
        """
        return np.dot(weights, self.expected_returns)
    
    def portfolio_volatility(self, weights: np.ndarray) -> float:
        """
        Calculate portfolio volatility (standard deviation).
        
        Args:
            weights: Asset allocation weights
            
        Returns:
            Portfolio volatility
        """
        variance = np.dot(weights.T, np.dot(self.covariance_matrix, weights))
        return np.sqrt(variance)
    
    def sharpe_ratio(self, weights: np.ndarray) -> float:
        """
        Calculate Sharpe ratio (risk-adjusted return).
        
        Args:
            weights: Asset allocation weights
            
        Returns:
            Sharpe ratio
        """
        portfolio_ret = self.portfolio_return(weights)
        portfolio_vol = self.portfolio_volatility(weights)
        
        if portfolio_vol == 0:
            return -np.inf
        
        return (portfolio_ret - self.risk_free_rate) / portfolio_vol
    
    def objective_function(self, weights: np.ndarray) -> float:
        """
        Objective function to minimize (negative Sharpe ratio with penalty).
        
        This function combines the Sharpe ratio with penalties for constraint violations.
        
        Args:
            weights: Asset allocation weights
            
        Returns:
            Objective value (to be minimized)
        """
        # Normalize weights
        weights = self.normalize_weights(weights)
        
        # Calculate Sharpe ratio
        sharpe = self.sharpe_ratio(weights)
        
        # We want to maximize Sharpe ratio, so minimize negative Sharpe ratio
        objective = -sharpe
        
        # Add penalty for constraint violations
        penalty = 0.0
        
        # Penalty for weights not summing to 1 (should be handled by normalization)
        weight_sum_penalty = abs(np.sum(weights) - 1.0) * 1000
        penalty += weight_sum_penalty
        
        # Penalty for weights outside bounds
        lower_bound_penalty = np.sum(np.maximum(self.min_weight - weights, 0)) * 1000
        upper_bound_penalty = np.sum(np.maximum(weights - self.max_weight, 0)) * 1000
        penalty += lower_bound_penalty + upper_bound_penalty
        
        return objective + penalty
    
    def get_portfolio_metrics(self, weights: np.ndarray) -> dict:
        """
        Calculate comprehensive portfolio metrics.
        
        Args:
            weights: Asset allocation weights
            
        Returns:
            Dictionary of portfolio metrics
        """
        weights = self.normalize_weights(weights)
        
        return {
            'weights': weights,
            'expected_return': self.portfolio_return(weights),
            'volatility': self.portfolio_volatility(weights),
            'sharpe_ratio': self.sharpe_ratio(weights),
            'max_drawdown': self._calculate_max_drawdown(weights)
        }
    
    def _calculate_max_drawdown(self, weights: np.ndarray) -> float:
        """
        Estimate maximum drawdown (simplified).
        
        Args:
            weights: Asset allocation weights
            
        Returns:
            Estimated max drawdown
        """
        # Simplified estimation based on volatility
        volatility = self.portfolio_volatility(weights)
        return -2.5 * volatility  # Rule of thumb: max drawdown ~ 2-3x volatility


def generate_sample_data(
    n_assets: int = 5,
    n_days: int = 252,
    mean_return_range: Tuple[float, float] = (0.05, 0.20),
    volatility_range: Tuple[float, float] = (0.15, 0.40)
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Generate synthetic stock price data for testing.
    
    Args:
        n_assets: Number of assets to generate
        n_days: Number of trading days to simulate
        mean_return_range: Range of mean annual returns
        volatility_range: Range of annual volatilities
        
    Returns:
        Tuple of (price_dataframe, expected_returns, covariance_matrix)
    """
    np.random.seed(42)
    
    # Generate random parameters for each asset
    annual_returns = np.random.uniform(*mean_return_range, n_assets)
    annual_volatilities = np.random.uniform(*volatility_range, n_assets)
    
    # Convert to daily parameters
    daily_returns = annual_returns / 252
    daily_volatilities = annual_volatilities / np.sqrt(252)
    
    # Generate correlation matrix
    correlation_matrix = np.random.uniform(0.3, 0.7, (n_assets, n_assets))
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
    np.fill_diagonal(correlation_matrix, 1.0)
    
    # Make it positive definite
    eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)
    eigenvalues = np.maximum(eigenvalues, 0.01)
    correlation_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    # Generate covariance matrix
    D = np.diag(daily_volatilities)
    covariance_matrix = D @ correlation_matrix @ D
    
    # Generate price paths
    prices = np.zeros((n_days, n_assets))
    prices[0] = 100  # Starting price
    
    for t in range(1, n_days):
        random_returns = np.random.multivariate_normal(daily_returns, covariance_matrix)
        prices[t] = prices[t-1] * (1 + random_returns)
    
    # Create DataFrame
    asset_names = [f"ASSET_{i+1}" for i in range(n_assets)]
    df = pd.DataFrame(prices, columns=asset_names)
    df.index.name = 'Day'
    
    return df, annual_returns, covariance_matrix * 252  # Return annualized values


def load_real_stock_data(tickers: List[str], start_date: str, end_date: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load real stock data from Yahoo Finance.
    
    Args:
        tickers: List of stock ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        Tuple of (price_dataframe, expected_returns, covariance_matrix)
    """
    try:
        import yfinance as yf
        
        # Download data
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        
        # Calculate returns
        returns = data.pct_change().dropna()
        
        # Calculate expected returns (annualized)
        expected_returns = returns.mean() * 252
        
        # Calculate covariance matrix (annualized)
        covariance_matrix = returns.cov() * 252
        
        return data, expected_returns.values, covariance_matrix.values
        
    except ImportError:
        logger.warning("yfinance not installed. Using synthetic data instead.")
        return generate_sample_data(len(tickers))


def test_portfolio_optimization():
    """Test portfolio optimization with synthetic data."""
    
    print("Generating sample portfolio data...")
    prices, expected_returns, cov_matrix = generate_sample_data(n_assets=5)
    
    print(f"\nExpected Annual Returns:")
    for i, ret in enumerate(expected_returns):
        print(f"  Asset {i+1}: {ret*100:.2f}%")
    
    print(f"\nCovariance Matrix:")
    print(cov_matrix)
    
    # Create portfolio optimizer
    portfolio = PortfolioOptimizer(
        expected_returns=expected_returns,
        covariance_matrix=cov_matrix,
        risk_free_rate=0.02
    )
    
    # Test with equal weights
    equal_weights = np.ones(5) / 5
    metrics = portfolio.get_portfolio_metrics(equal_weights)
    
    print(f"\nEqual-Weight Portfolio:")
    print(f"  Expected Return: {metrics['expected_return']*100:.2f}%")
    print(f"  Volatility: {metrics['volatility']*100:.2f}%")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    
    # Import PSO and optimize
    from src.pso import ParticleSwarmOptimizer, PSOConfig
    
    print("\n" + "="*50)
    print("Running PSO optimization...")
    
    config = PSOConfig(
        n_particles=30,
        n_dimensions=5,
        n_iterations=100,
        bounds=(np.zeros(5), np.ones(5)),
        adaptive_inertia=True,
        verbose=False
    )
    
    pso = ParticleSwarmOptimizer(config)
    best_weights, best_fitness, history = pso.optimize(portfolio.objective_function)
    
    # Normalize final weights
    best_weights = portfolio.normalize_weights(best_weights)
    optimized_metrics = portfolio.get_portfolio_metrics(best_weights)
    
    print(f"\nOptimized Portfolio:")
    print(f"  Weights:")
    for i, w in enumerate(best_weights):
        print(f"    Asset {i+1}: {w*100:.2f}%")
    print(f"  Expected Return: {optimized_metrics['expected_return']*100:.2f}%")
    print(f"  Volatility: {optimized_metrics['volatility']*100:.2f}%")
    print(f"  Sharpe Ratio: {optimized_metrics['sharpe_ratio']:.4f}")
    
    improvement = (optimized_metrics['sharpe_ratio'] - metrics['sharpe_ratio']) / metrics['sharpe_ratio'] * 100
    print(f"\nImprovement over equal-weight: {improvement:.2f}%")


if __name__ == "__main__":
    test_portfolio_optimization()
