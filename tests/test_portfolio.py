"""
Unit tests for portfolio optimization

Author: EU97
Date: 2024
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd
from src.portfolio import PortfolioOptimizer, generate_sample_data


class TestPortfolioOptimizer:
    """Test cases for PortfolioOptimizer class."""
    
    @pytest.fixture
    def sample_portfolio(self):
        """Create a sample portfolio for testing."""
        expected_returns = np.array([0.10, 0.15, 0.12, 0.08])
        cov_matrix = np.array([
            [0.04, 0.01, 0.02, 0.01],
            [0.01, 0.09, 0.03, 0.02],
            [0.02, 0.03, 0.06, 0.01],
            [0.01, 0.02, 0.01, 0.03]
        ])
        
        return PortfolioOptimizer(
            expected_returns=expected_returns,
            covariance_matrix=cov_matrix,
            risk_free_rate=0.02
        )
    
    def test_initialization(self, sample_portfolio):
        """Test portfolio optimizer initialization."""
        assert sample_portfolio.n_assets == 4
        assert sample_portfolio.risk_free_rate == 0.02
        assert sample_portfolio.expected_returns.shape == (4,)
        assert sample_portfolio.covariance_matrix.shape == (4, 4)
    
    def test_normalize_weights(self, sample_portfolio):
        """Test weight normalization."""
        weights = np.array([0.3, 0.4, 0.2, 0.1])
        normalized = sample_portfolio.normalize_weights(weights)
        
        assert np.isclose(np.sum(normalized), 1.0)
        assert np.all(normalized >= 0)
        assert np.all(normalized <= 1)
    
    def test_normalize_weights_out_of_bounds(self, sample_portfolio):
        """Test normalization with out-of-bounds weights."""
        weights = np.array([0.5, 0.6, 0.4, 0.3])  # Sum > 1
        normalized = sample_portfolio.normalize_weights(weights)
        
        assert np.isclose(np.sum(normalized), 1.0)
    
    def test_portfolio_return(self, sample_portfolio):
        """Test portfolio return calculation."""
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        portfolio_return = sample_portfolio.portfolio_return(weights)
        
        expected = np.mean(sample_portfolio.expected_returns)
        assert np.isclose(portfolio_return, expected)
    
    def test_portfolio_volatility(self, sample_portfolio):
        """Test portfolio volatility calculation."""
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        volatility = sample_portfolio.portfolio_volatility(weights)
        
        assert volatility > 0
        assert isinstance(volatility, float)
    
    def test_sharpe_ratio(self, sample_portfolio):
        """Test Sharpe ratio calculation."""
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        sharpe = sample_portfolio.sharpe_ratio(weights)
        
        assert isinstance(sharpe, float)
        # Sharpe should be positive for positive excess returns
        portfolio_ret = sample_portfolio.portfolio_return(weights)
        if portfolio_ret > sample_portfolio.risk_free_rate:
            assert sharpe > 0
    
    def test_objective_function(self, sample_portfolio):
        """Test objective function."""
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        objective = sample_portfolio.objective_function(weights)
        
        # Objective is negative Sharpe ratio (for minimization)
        assert isinstance(objective, float)
    
    def test_get_portfolio_metrics(self, sample_portfolio):
        """Test comprehensive metrics calculation."""
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        metrics = sample_portfolio.get_portfolio_metrics(weights)
        
        assert 'weights' in metrics
        assert 'expected_return' in metrics
        assert 'volatility' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        
        assert np.allclose(metrics['weights'], weights, atol=0.01)
    
    def test_min_max_weight_constraints(self):
        """Test minimum and maximum weight constraints."""
        expected_returns = np.array([0.10, 0.15, 0.12])
        cov_matrix = np.eye(3) * 0.04
        
        portfolio = PortfolioOptimizer(
            expected_returns=expected_returns,
            covariance_matrix=cov_matrix,
            min_weight=0.1,
            max_weight=0.5
        )
        
        weights = np.array([0.05, 0.6, 0.35])  # Violates constraints
        normalized = portfolio.normalize_weights(weights)
        
        assert np.all(normalized >= portfolio.min_weight - 1e-6)
        assert np.all(normalized <= portfolio.max_weight + 1e-6)


class TestGenerateSampleData:
    """Test cases for sample data generation."""
    
    def test_generate_sample_data_default(self):
        """Test sample data generation with defaults."""
        prices, returns, cov_matrix = generate_sample_data()
        
        assert isinstance(prices, pd.DataFrame)
        assert isinstance(returns, np.ndarray)
        assert isinstance(cov_matrix, np.ndarray)
        
        assert prices.shape == (252, 5)
        assert returns.shape == (5,)
        assert cov_matrix.shape == (5, 5)
    
    def test_generate_sample_data_custom(self):
        """Test sample data generation with custom parameters."""
        n_assets = 8
        n_days = 500
        
        prices, returns, cov_matrix = generate_sample_data(
            n_assets=n_assets,
            n_days=n_days
        )
        
        assert prices.shape == (n_days, n_assets)
        assert returns.shape == (n_assets,)
        assert cov_matrix.shape == (n_assets, n_assets)
    
    def test_positive_definite_covariance(self):
        """Test that covariance matrix is positive definite."""
        _, _, cov_matrix = generate_sample_data()
        
        # Check positive definiteness
        eigenvalues = np.linalg.eigvals(cov_matrix)
        assert np.all(eigenvalues > 0)
    
    def test_returns_in_expected_range(self):
        """Test that generated returns are in reasonable range."""
        _, returns, _ = generate_sample_data(
            mean_return_range=(0.05, 0.20)
        )
        
        assert np.all(returns >= 0.05)
        assert np.all(returns <= 0.20)
    
    def test_prices_start_at_100(self):
        """Test that all prices start at 100."""
        prices, _, _ = generate_sample_data()
        
        assert np.allclose(prices.iloc[0].values, 100.0)


class TestPortfolioOptimization:
    """Integration tests for portfolio optimization."""
    
    def test_full_optimization_pipeline(self):
        """Test complete optimization workflow."""
        from src.pso import ParticleSwarmOptimizer, PSOConfig
        
        # Generate data
        prices, expected_returns, cov_matrix = generate_sample_data(n_assets=4)
        
        # Create portfolio
        portfolio = PortfolioOptimizer(
            expected_returns=expected_returns,
            covariance_matrix=cov_matrix
        )
        
        # Configure PSO
        config = PSOConfig(
            n_particles=20,
            n_dimensions=4,
            n_iterations=50,
            bounds=(np.zeros(4), np.ones(4)),
            verbose=False
        )
        
        # Optimize
        pso = ParticleSwarmOptimizer(config)
        best_weights, best_fitness, history = pso.optimize(portfolio.objective_function)
        
        # Normalize and validate
        best_weights = portfolio.normalize_weights(best_weights)
        
        assert np.isclose(np.sum(best_weights), 1.0)
        assert np.all(best_weights >= 0)
        assert best_fitness < 0  # Should be negative (negative Sharpe)
    
    def test_optimization_improves_sharpe(self):
        """Test that optimization improves Sharpe ratio vs equal weights."""
        from src.pso import ParticleSwarmOptimizer, PSOConfig
        
        prices, expected_returns, cov_matrix = generate_sample_data(n_assets=5)
        
        portfolio = PortfolioOptimizer(
            expected_returns=expected_returns,
            covariance_matrix=cov_matrix
        )
        
        # Equal weight portfolio
        equal_weights = np.ones(5) / 5
        equal_sharpe = portfolio.sharpe_ratio(equal_weights)
        
        # Optimized portfolio
        config = PSOConfig(
            n_particles=30,
            n_dimensions=5,
            n_iterations=100,
            bounds=(np.zeros(5), np.ones(5)),
            adaptive_inertia=True,
            verbose=False
        )
        
        pso = ParticleSwarmOptimizer(config)
        best_weights, _, _ = pso.optimize(portfolio.objective_function)
        best_weights = portfolio.normalize_weights(best_weights)
        optimized_sharpe = portfolio.sharpe_ratio(best_weights)
        
        # Optimized should be better (or at least not worse)
        assert optimized_sharpe >= equal_sharpe * 0.95  # Allow 5% tolerance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
