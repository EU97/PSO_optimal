"""
Visualization utilities for PSO optimization

This module provides functions for visualizing PSO convergence,
portfolio allocation, and performance metrics.

Author: EU97
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_convergence(
    history: Dict[str, List],
    title: str = "PSO Convergence",
    show_std: bool = True,
    interactive: bool = False
) -> Optional[go.Figure]:
    """
    Plot PSO convergence history.
    
    Args:
        history: Dictionary containing convergence history
        title: Plot title
        show_std: Whether to show standard deviation band
        interactive: Return Plotly figure instead of matplotlib
        
    Returns:
        Plotly figure if interactive=True, else None
    """
    iterations = range(len(history['best_fitness']))
    
    if interactive:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Best Fitness Over Time',
                'Mean Fitness Over Time',
                'Fitness Standard Deviation',
                'Swarm Diversity'
            )
        )
        
        # Best fitness
        fig.add_trace(
            go.Scatter(
                x=list(iterations),
                y=history['best_fitness'],
                mode='lines',
                name='Best Fitness',
                line=dict(color='green', width=2)
            ),
            row=1, col=1
        )
        
        # Mean fitness
        mean_fitness = history['mean_fitness']
        std_fitness = history['std_fitness']
        
        if show_std:
            upper_bound = np.array(mean_fitness) + np.array(std_fitness)
            lower_bound = np.array(mean_fitness) - np.array(std_fitness)
            
            fig.add_trace(
                go.Scatter(
                    x=list(iterations) + list(iterations)[::-1],
                    y=list(upper_bound) + list(lower_bound)[::-1],
                    fill='toself',
                    fillcolor='rgba(0,100,255,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='±1 Std Dev',
                    showlegend=True
                ),
                row=1, col=2
            )
        
        fig.add_trace(
            go.Scatter(
                x=list(iterations),
                y=mean_fitness,
                mode='lines',
                name='Mean Fitness',
                line=dict(color='blue', width=2)
            ),
            row=1, col=2
        )
        
        # Standard deviation
        fig.add_trace(
            go.Scatter(
                x=list(iterations),
                y=std_fitness,
                mode='lines',
                name='Std Dev',
                line=dict(color='orange', width=2)
            ),
            row=2, col=1
        )
        
        # Diversity
        fig.add_trace(
            go.Scatter(
                x=list(iterations),
                y=history['diversity'],
                mode='lines',
                name='Diversity',
                line=dict(color='purple', width=2)
            ),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text="Iteration", row=2, col=1)
        fig.update_xaxes(title_text="Iteration", row=2, col=2)
        fig.update_yaxes(title_text="Fitness", row=1, col=1)
        fig.update_yaxes(title_text="Fitness", row=1, col=2)
        
        fig.update_layout(
            title_text=title,
            height=800,
            showlegend=True
        )
        
        return fig
    
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Best fitness
        axes[0, 0].plot(iterations, history['best_fitness'], 'g-', linewidth=2, label='Best Fitness')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Best Fitness')
        axes[0, 0].set_title('Best Fitness Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Mean fitness with std
        mean_fitness = history['mean_fitness']
        std_fitness = history['std_fitness']
        
        axes[0, 1].plot(iterations, mean_fitness, 'b-', linewidth=2, label='Mean Fitness')
        if show_std:
            axes[0, 1].fill_between(
                iterations,
                np.array(mean_fitness) - np.array(std_fitness),
                np.array(mean_fitness) + np.array(std_fitness),
                alpha=0.3,
                label='±1 Std Dev'
            )
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Mean Fitness')
        axes[0, 1].set_title('Mean Fitness Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Standard deviation
        axes[1, 0].plot(iterations, std_fitness, 'orange', linewidth=2)
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Standard Deviation')
        axes[1, 0].set_title('Fitness Standard Deviation')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Diversity
        axes[1, 1].plot(iterations, history['diversity'], 'purple', linewidth=2)
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Diversity')
        axes[1, 1].set_title('Swarm Diversity')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return None


def plot_portfolio_allocation(
    weights: np.ndarray,
    asset_names: Optional[List[str]] = None,
    title: str = "Portfolio Allocation",
    interactive: bool = False
) -> Optional[go.Figure]:
    """
    Plot portfolio allocation as pie chart and bar chart.
    
    Args:
        weights: Array of portfolio weights
        asset_names: Optional list of asset names
        title: Plot title
        interactive: Return Plotly figure instead of matplotlib
        
    Returns:
        Plotly figure if interactive=True, else None
    """
    if asset_names is None:
        asset_names = [f"Asset {i+1}" for i in range(len(weights))]
    
    # Sort by weight for better visualization
    sorted_indices = np.argsort(weights)[::-1]
    sorted_weights = weights[sorted_indices]
    sorted_names = [asset_names[i] for i in sorted_indices]
    
    if interactive:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Allocation Distribution', 'Weight Comparison'),
            specs=[[{'type': 'pie'}, {'type': 'bar'}]]
        )
        
        # Pie chart
        fig.add_trace(
            go.Pie(
                labels=sorted_names,
                values=sorted_weights,
                textinfo='label+percent',
                hovertemplate='<b>%{label}</b><br>Weight: %{value:.2%}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Bar chart
        fig.add_trace(
            go.Bar(
                x=sorted_names,
                y=sorted_weights * 100,
                text=[f'{w:.1f}%' for w in sorted_weights * 100],
                textposition='auto',
                marker_color='lightblue'
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Asset", row=1, col=2)
        fig.update_yaxes(title_text="Weight (%)", row=1, col=2)
        
        fig.update_layout(
            title_text=title,
            showlegend=False,
            height=500
        )
        
        return fig
    
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Pie chart
        colors = plt.cm.Set3(range(len(weights)))
        ax1.pie(sorted_weights, labels=sorted_names, autopct='%1.1f%%',
                startangle=90, colors=colors)
        ax1.set_title('Allocation Distribution')
        
        # Bar chart
        bars = ax2.bar(sorted_names, sorted_weights * 100, color=colors)
        ax2.set_xlabel('Asset')
        ax2.set_ylabel('Weight (%)')
        ax2.set_title('Weight Comparison')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return None


def plot_efficient_frontier(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    optimal_weights: np.ndarray,
    n_portfolios: int = 1000,
    risk_free_rate: float = 0.02,
    interactive: bool = False
) -> Optional[go.Figure]:
    """
    Plot efficient frontier with optimal portfolio highlighted.
    
    Args:
        expected_returns: Expected returns for each asset
        cov_matrix: Covariance matrix
        optimal_weights: Optimal portfolio weights from PSO
        n_portfolios: Number of random portfolios to generate
        risk_free_rate: Risk-free rate
        interactive: Return Plotly figure instead of matplotlib
        
    Returns:
        Plotly figure if interactive=True, else None
    """
    n_assets = len(expected_returns)
    
    # Generate random portfolios
    portfolio_returns = []
    portfolio_volatilities = []
    portfolio_sharpe = []
    
    for _ in range(n_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)
        
        ret = np.dot(weights, expected_returns)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0
        
        portfolio_returns.append(ret)
        portfolio_volatilities.append(vol)
        portfolio_sharpe.append(sharpe)
    
    # Calculate optimal portfolio metrics
    optimal_return = np.dot(optimal_weights, expected_returns)
    optimal_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
    optimal_sharpe = (optimal_return - risk_free_rate) / optimal_volatility
    
    if interactive:
        fig = go.Figure()
        
        # Random portfolios
        fig.add_trace(go.Scatter(
            x=portfolio_volatilities,
            y=portfolio_returns,
            mode='markers',
            marker=dict(
                size=4,
                color=portfolio_sharpe,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Sharpe Ratio")
            ),
            text=[f'Sharpe: {s:.3f}' for s in portfolio_sharpe],
            hovertemplate='<b>Random Portfolio</b><br>Volatility: %{x:.2%}<br>Return: %{y:.2%}<br>%{text}<extra></extra>',
            name='Random Portfolios'
        ))
        
        # Optimal portfolio
        fig.add_trace(go.Scatter(
            x=[optimal_volatility],
            y=[optimal_return],
            mode='markers',
            marker=dict(size=15, color='red', symbol='star'),
            name='PSO Optimal',
            hovertemplate='<b>PSO Optimal Portfolio</b><br>Volatility: %{x:.2%}<br>Return: %{y:.2%}<br>Sharpe: ' + f'{optimal_sharpe:.3f}<extra></extra>'
        ))
        
        # Capital Market Line
        cml_x = np.linspace(0, max(portfolio_volatilities), 100)
        cml_y = risk_free_rate + optimal_sharpe * cml_x
        fig.add_trace(go.Scatter(
            x=cml_x,
            y=cml_y,
            mode='lines',
            line=dict(dash='dash', color='orange'),
            name='Capital Market Line'
        ))
        
        fig.update_layout(
            title='Efficient Frontier and Optimal Portfolio',
            xaxis_title='Volatility (Risk)',
            yaxis_title='Expected Return',
            xaxis=dict(tickformat='.1%'),
            yaxis=dict(tickformat='.1%'),
            hovermode='closest',
            height=600
        )
        
        return fig
    
    else:
        plt.figure(figsize=(12, 8))
        
        scatter = plt.scatter(portfolio_volatilities, portfolio_returns,
                            c=portfolio_sharpe, cmap='viridis', s=10, alpha=0.5)
        plt.colorbar(scatter, label='Sharpe Ratio')
        
        # Plot optimal portfolio
        plt.scatter(optimal_volatility, optimal_return,
                   c='red', s=200, marker='*', edgecolors='black', linewidths=2,
                   label=f'PSO Optimal (Sharpe: {optimal_sharpe:.3f})', zorder=5)
        
        # Plot Capital Market Line
        cml_x = np.linspace(0, max(portfolio_volatilities), 100)
        cml_y = risk_free_rate + optimal_sharpe * cml_x
        plt.plot(cml_x, cml_y, 'orange', linestyle='--', linewidth=2,
                label='Capital Market Line')
        
        plt.xlabel('Volatility (Risk)', fontsize=12)
        plt.ylabel('Expected Return', fontsize=12)
        plt.title('Efficient Frontier and Optimal Portfolio', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return None


def plot_portfolio_metrics_comparison(
    metrics_dict: Dict[str, Dict],
    interactive: bool = False
) -> Optional[go.Figure]:
    """
    Compare metrics across different portfolio strategies.
    
    Args:
        metrics_dict: Dictionary of {strategy_name: metrics_dict}
        interactive: Return Plotly figure instead of matplotlib
        
    Returns:
        Plotly figure if interactive=True, else None
    """
    strategies = list(metrics_dict.keys())
    returns = [metrics_dict[s]['expected_return'] for s in strategies]
    volatilities = [metrics_dict[s]['volatility'] for s in strategies]
    sharpe_ratios = [metrics_dict[s]['sharpe_ratio'] for s in strategies]
    
    if interactive:
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Expected Return', 'Volatility', 'Sharpe Ratio')
        )
        
        fig.add_trace(
            go.Bar(x=strategies, y=np.array(returns)*100, name='Return', marker_color='lightgreen'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=strategies, y=np.array(volatilities)*100, name='Volatility', marker_color='lightcoral'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=strategies, y=sharpe_ratios, name='Sharpe Ratio', marker_color='lightblue'),
            row=1, col=3
        )
        
        fig.update_yaxes(title_text="Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=1, col=2)
        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=3)
        
        fig.update_layout(title_text="Portfolio Strategy Comparison", showlegend=False, height=400)
        
        return fig
    
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].bar(strategies, np.array(returns)*100, color='lightgreen')
        axes[0].set_ylabel('Return (%)')
        axes[0].set_title('Expected Return')
        axes[0].tick_params(axis='x', rotation=45)
        
        axes[1].bar(strategies, np.array(volatilities)*100, color='lightcoral')
        axes[1].set_ylabel('Volatility (%)')
        axes[1].set_title('Volatility')
        axes[1].tick_params(axis='x', rotation=45)
        
        axes[2].bar(strategies, sharpe_ratios, color='lightblue')
        axes[2].set_ylabel('Sharpe Ratio')
        axes[2].set_title('Sharpe Ratio')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.suptitle('Portfolio Strategy Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return None


if __name__ == "__main__":
    # Test visualization with sample data
    print("Testing visualization functions...")
    
    # Sample convergence history
    history = {
        'best_fitness': list(np.exp(-np.linspace(0, 3, 100)) + np.random.random(100) * 0.05),
        'mean_fitness': list(np.exp(-np.linspace(0, 2, 100)) + 1 + np.random.random(100) * 0.1),
        'std_fitness': list(np.linspace(0.5, 0.1, 100)),
        'diversity': list(np.linspace(1.0, 0.2, 100))
    }
    
    print("\nPlotting convergence...")
    plot_convergence(history, interactive=False)
    
    # Sample portfolio weights
    weights = np.array([0.25, 0.30, 0.15, 0.20, 0.10])
    asset_names = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    
    print("\nPlotting portfolio allocation...")
    plot_portfolio_allocation(weights, asset_names, interactive=False)
