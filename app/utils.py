"""
Utility functions for the Streamlit application

Author: EU97
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
import io


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a decimal value as a percentage string.
    
    Args:
        value: Decimal value (e.g., 0.15 for 15%)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def format_currency(value: float, currency: str = "$", decimals: int = 2) -> str:
    """
    Format a value as currency.
    
    Args:
        value: Numeric value
        currency: Currency symbol
        decimals: Number of decimal places
        
    Returns:
        Formatted currency string
    """
    return f"{currency}{value:,.{decimals}f}"


def calculate_portfolio_value(
    weights: np.ndarray,
    initial_value: float,
    returns: np.ndarray
) -> float:
    """
    Calculate portfolio value after returns.
    
    Args:
        weights: Portfolio weights
        initial_value: Starting portfolio value
        returns: Asset returns
        
    Returns:
        Final portfolio value
    """
    portfolio_return = np.dot(weights, returns)
    return initial_value * (1 + portfolio_return)


def save_results_to_csv(
    weights: np.ndarray,
    asset_names: List[str],
    metrics: Dict[str, Any]
) -> str:
    """
    Save optimization results to CSV format.
    
    Args:
        weights: Optimal portfolio weights
        asset_names: Names of assets
        metrics: Portfolio metrics dictionary
        
    Returns:
        CSV string
    """
    # Create allocation dataframe
    allocation_df = pd.DataFrame({
        'Asset': asset_names,
        'Weight': weights,
        'Weight_Percent': weights * 100
    })
    
    # Create metrics dataframe
    metrics_df = pd.DataFrame({
        'Metric': ['Expected Return', 'Volatility', 'Sharpe Ratio'],
        'Value': [
            metrics['expected_return'],
            metrics['volatility'],
            metrics['sharpe_ratio']
        ]
    })
    
    # Combine into CSV
    output = io.StringIO()
    
    output.write("Portfolio Allocation\n")
    allocation_df.to_csv(output, index=False)
    output.write("\n")
    
    output.write("Portfolio Metrics\n")
    metrics_df.to_csv(output, index=False)
    
    return output.getvalue()


def validate_weights(weights: np.ndarray, tolerance: float = 1e-6) -> bool:
    """
    Validate that portfolio weights are valid.
    
    Args:
        weights: Portfolio weights
        tolerance: Numerical tolerance
        
    Returns:
        True if valid, False otherwise
    """
    # Check sum to 1
    if abs(np.sum(weights) - 1.0) > tolerance:
        return False
    
    # Check non-negative
    if np.any(weights < -tolerance):
        return False
    
    # Check not exceeding 1
    if np.any(weights > 1.0 + tolerance):
        return False
    
    return True


def calculate_drawdown_series(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate drawdown series for each asset.
    
    Args:
        prices: DataFrame of asset prices
        
    Returns:
        DataFrame of drawdown percentages
    """
    cumulative_max = prices.expanding().max()
    drawdown = (prices - cumulative_max) / cumulative_max
    return drawdown


def calculate_rolling_sharpe(
    returns: pd.Series,
    window: int = 60,
    risk_free_rate: float = 0.02
) -> pd.Series:
    """
    Calculate rolling Sharpe ratio.
    
    Args:
        returns: Return series
        window: Rolling window size
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Rolling Sharpe ratio series
    """
    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf
    
    rolling_mean = excess_returns.rolling(window).mean()
    rolling_std = excess_returns.rolling(window).std()
    
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
    return rolling_sharpe


def rebalance_portfolio(
    current_weights: np.ndarray,
    target_weights: np.ndarray,
    threshold: float = 0.05
) -> np.ndarray:
    """
    Calculate rebalancing trades needed.
    
    Args:
        current_weights: Current portfolio weights
        target_weights: Target portfolio weights
        threshold: Minimum weight difference to trigger rebalance
        
    Returns:
        Array of weight changes needed
    """
    weight_diff = target_weights - current_weights
    
    # Only rebalance if difference exceeds threshold
    trades = np.where(np.abs(weight_diff) > threshold, weight_diff, 0)
    
    return trades


def calculate_transaction_costs(
    trades: np.ndarray,
    portfolio_value: float,
    cost_rate: float = 0.001
) -> float:
    """
    Calculate transaction costs for rebalancing.
    
    Args:
        trades: Weight changes (positive=buy, negative=sell)
        portfolio_value: Total portfolio value
        cost_rate: Transaction cost rate (e.g., 0.001 = 0.1%)
        
    Returns:
        Total transaction cost
    """
    trade_values = np.abs(trades) * portfolio_value
    total_cost = np.sum(trade_values) * cost_rate
    return total_cost


def generate_allocation_report(
    weights: np.ndarray,
    asset_names: List[str],
    expected_returns: np.ndarray,
    metrics: Dict[str, Any],
    portfolio_value: float = 100000
) -> str:
    """
    Generate a detailed text report of the portfolio allocation.
    
    Args:
        weights: Portfolio weights
        asset_names: Names of assets
        expected_returns: Expected return for each asset
        metrics: Portfolio metrics
        portfolio_value: Total portfolio value
        
    Returns:
        Formatted text report
    """
    report = []
    report.append("="*60)
    report.append("PORTFOLIO OPTIMIZATION REPORT")
    report.append("="*60)
    report.append("")
    
    report.append("PORTFOLIO METRICS")
    report.append("-"*60)
    report.append(f"Expected Annual Return:    {format_percentage(metrics['expected_return'])}")
    report.append(f"Annual Volatility:         {format_percentage(metrics['volatility'])}")
    report.append(f"Sharpe Ratio:              {metrics['sharpe_ratio']:.4f}")
    report.append(f"Total Portfolio Value:     {format_currency(portfolio_value)}")
    report.append("")
    
    report.append("ASSET ALLOCATION")
    report.append("-"*60)
    report.append(f"{'Asset':<15} {'Weight':<12} {'Value':<15} {'Exp. Return':<12}")
    report.append("-"*60)
    
    for i, (asset, weight, exp_ret) in enumerate(zip(asset_names, weights, expected_returns)):
        value = weight * portfolio_value
        report.append(
            f"{asset:<15} {format_percentage(weight):<12} "
            f"{format_currency(value):<15} {format_percentage(exp_ret):<12}"
        )
    
    report.append("="*60)
    
    return "\n".join(report)


def create_markdown_report(
    weights: np.ndarray,
    asset_names: List[str],
    metrics: Dict[str, Any],
    history: Dict[str, List]
) -> str:
    """
    Create a Markdown-formatted report.
    
    Args:
        weights: Portfolio weights
        asset_names: Names of assets
        metrics: Portfolio metrics
        history: Optimization history
        
    Returns:
        Markdown report string
    """
    md = []
    md.append("# Portfolio Optimization Report")
    md.append("")
    md.append("## Executive Summary")
    md.append("")
    md.append(f"- **Expected Return**: {format_percentage(metrics['expected_return'])}")
    md.append(f"- **Volatility**: {format_percentage(metrics['volatility'])}")
    md.append(f"- **Sharpe Ratio**: {metrics['sharpe_ratio']:.4f}")
    md.append("")
    
    md.append("## Optimal Allocation")
    md.append("")
    md.append("| Asset | Weight | Allocation % |")
    md.append("|-------|--------|--------------|")
    
    for asset, weight in zip(asset_names, weights):
        md.append(f"| {asset} | {weight:.4f} | {format_percentage(weight)} |")
    
    md.append("")
    md.append("## Optimization Statistics")
    md.append("")
    md.append(f"- **Initial Fitness**: {history['best_fitness'][0]:.6f}")
    md.append(f"- **Final Fitness**: {history['best_fitness'][-1]:.6f}")
    md.append(f"- **Improvement**: {(1 - history['best_fitness'][-1]/history['best_fitness'][0])*100:.2f}%")
    md.append(f"- **Iterations**: {len(history['best_fitness'])}")
    md.append("")
    
    return "\n".join(md)


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    print(f"\nFormat percentage: {format_percentage(0.1523)}")
    print(f"Format currency: {format_currency(123456.789)}")
    
    weights = np.array([0.2, 0.3, 0.25, 0.15, 0.1])
    print(f"\nValidate weights: {validate_weights(weights)}")
    
    print("\nUtility functions working correctly!")
