"""
Streamlit Web Interface for PSO Portfolio Optimizer

This is the main entry point for the interactive web application.

Author: EU97
Date: 2025
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.pso import ParticleSwarmOptimizer, PSOConfig
from src.portfolio import PortfolioOptimizer, generate_sample_data, load_real_stock_data
from src.visualization import (
    plot_convergence,
    plot_portfolio_allocation,
    plot_efficient_frontier,
    plot_portfolio_metrics_comparison
)
from app.utils import format_percentage, format_currency, save_results_to_csv

# Page configuration
st.set_page_config(
    page_title="PSO Portfolio Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'optimization_done' not in st.session_state:
        st.session_state.optimization_done = False
    if 'best_weights' not in st.session_state:
        st.session_state.best_weights = None
    if 'history' not in st.session_state:
        st.session_state.history = None
    if 'portfolio_metrics' not in st.session_state:
        st.session_state.portfolio_metrics = None


def main():
    """Main application function."""
    initialize_session_state()
    
    # Header
    st.markdown('<p class="main-header">🐝 PSO Portfolio Optimizer</p>', unsafe_allow_html=True)
    st.markdown("""
    <p style='text-align: center; font-size: 1.2rem; color: #666;'>
    Optimize your investment portfolio using Particle Swarm Optimization
    </p>
    """, unsafe_allow_html=True)
    
    # Sidebar - Configuration
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/python/python.png", width=100)
        st.title("⚙️ Configuration")
        
        # Data source selection
        st.markdown("### 📊 Data Source")
        data_source = st.radio(
            "Choose data source:",
            ["Synthetic Data", "Real Stock Data (Demo)"],
            help="Synthetic data generates random portfolio data. Real data uses predefined sample stocks."
        )
        
        if data_source == "Synthetic Data":
            n_assets = st.slider("Number of Assets", 3, 10, 5)
            n_days = st.slider("Historical Days", 100, 500, 252)
        else:
            st.info("Using sample data for: AAPL, GOOGL, MSFT, AMZN, TSLA")
            n_assets = 5
        
        st.markdown("---")
        
        # PSO Parameters
        st.markdown("### 🐝 PSO Parameters")
        
        n_particles = st.slider(
            "Number of Particles",
            10, 100, 30,
            help="Number of candidate solutions in the swarm"
        )
        
        n_iterations = st.slider(
            "Number of Iterations",
            50, 300, 100,
            help="Maximum number of optimization iterations"
        )
        
        with st.expander("Advanced PSO Settings"):
            adaptive_inertia = st.checkbox(
                "Adaptive Inertia",
                value=True,
                help="Dynamically adjust inertia weight during optimization"
            )
            
            if adaptive_inertia:
                w_min = st.slider("Min Inertia Weight", 0.1, 0.9, 0.4, 0.1)
                w_max = st.slider("Max Inertia Weight", 0.1, 0.9, 0.9, 0.1)
                w = (w_min + w_max) / 2
            else:
                w = st.slider("Inertia Weight (w)", 0.1, 0.9, 0.7, 0.1)
                w_min, w_max = w, w
            
            c1 = st.slider(
                "Cognitive Parameter (c1)",
                0.0, 3.0, 1.5, 0.1,
                help="Personal best influence"
            )
            
            c2 = st.slider(
                "Social Parameter (c2)",
                0.0, 3.0, 1.5, 0.1,
                help="Global best influence"
            )
        
        st.markdown("---")
        
        # Portfolio Parameters
        st.markdown("### 💼 Portfolio Settings")
        
        risk_free_rate = st.slider(
            "Risk-Free Rate",
            0.0, 0.10, 0.02, 0.01,
            format="%.2f",
            help="Annual risk-free rate (e.g., treasury bonds)"
        )
        
        risk_aversion = st.slider(
            "Risk Aversion",
            0.0, 1.0, 0.5, 0.1,
            help="0=risk-neutral, 1=highly risk-averse"
        )
        
        with st.expander("Allocation Constraints"):
            min_weight = st.slider(
                "Minimum Weight per Asset",
                0.0, 0.2, 0.0, 0.05,
                format="%.2f",
                help="Minimum allocation percentage"
            )
            
            max_weight = st.slider(
                "Maximum Weight per Asset",
                0.2, 1.0, 1.0, 0.05,
                format="%.2f",
                help="Maximum allocation percentage"
            )
        
        st.markdown("---")
        
        # Run button
        run_optimization = st.button("🚀 Run Optimization", type="primary")
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Optimization", "📊 Results", "📉 Analysis", "ℹ️ About"])
    
    with tab1:
        st.markdown('<p class="sub-header">Optimization Process</p>', unsafe_allow_html=True)
        
        if run_optimization:
            with st.spinner("🔄 Generating portfolio data..."):
                # Generate or load data
                if data_source == "Synthetic Data":
                    prices_df, expected_returns, cov_matrix = generate_sample_data(
                        n_assets=n_assets,
                        n_days=n_days
                    )
                    asset_names = [f"Asset {i+1}" for i in range(n_assets)]
                else:
                    # Use predefined sample data
                    prices_df, expected_returns, cov_matrix = generate_sample_data(
                        n_assets=5,
                        n_days=252
                    )
                    asset_names = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
                
                # Store in session state
                st.session_state.prices_df = prices_df
                st.session_state.expected_returns = expected_returns
                st.session_state.cov_matrix = cov_matrix
                st.session_state.asset_names = asset_names
            
            # Display data overview
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Expected Annual Returns")
                returns_df = pd.DataFrame({
                    'Asset': asset_names,
                    'Expected Return': [format_percentage(r) for r in expected_returns],
                    'Return (%)': expected_returns * 100
                })
                st.dataframe(returns_df[['Asset', 'Expected Return']], hide_index=True, use_container_width=True)
            
            with col2:
                st.markdown("#### Price History Preview")
                st.line_chart(prices_df.head(50))
            
            # Run optimization
            with st.spinner("🐝 Running PSO optimization... This may take a moment."):
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Configure PSO
                config = PSOConfig(
                    n_particles=n_particles,
                    n_dimensions=n_assets,
                    n_iterations=n_iterations,
                    w=w,
                    c1=c1,
                    c2=c2,
                    w_min=w_min,
                    w_max=w_max,
                    bounds=(np.zeros(n_assets), np.ones(n_assets)),
                    adaptive_inertia=adaptive_inertia,
                    verbose=False
                )
                
                # Create portfolio optimizer
                portfolio = PortfolioOptimizer(
                    expected_returns=expected_returns,
                    covariance_matrix=cov_matrix,
                    risk_free_rate=risk_free_rate,
                    risk_aversion=risk_aversion,
                    min_weight=min_weight,
                    max_weight=max_weight
                )
                
                # Callback for progress updates
                def update_progress(iteration, best_fitness, best_position):
                    progress = (iteration + 1) / n_iterations
                    progress_bar.progress(progress)
                    status_text.text(f"Iteration {iteration + 1}/{n_iterations} - Best Fitness: {best_fitness:.6f}")
                
                # Run PSO
                pso = ParticleSwarmOptimizer(config)
                best_weights, best_fitness, history = pso.optimize(
                    portfolio.objective_function,
                    callback=update_progress
                )
                
                # Normalize weights
                best_weights = portfolio.normalize_weights(best_weights)
                
                # Calculate metrics
                metrics = portfolio.get_portfolio_metrics(best_weights)
                
                # Calculate equal-weight portfolio for comparison
                equal_weights = np.ones(n_assets) / n_assets
                equal_metrics = portfolio.get_portfolio_metrics(equal_weights)
                
                # Store results in session state
                st.session_state.best_weights = best_weights
                st.session_state.history = history
                st.session_state.portfolio_metrics = metrics
                st.session_state.equal_metrics = equal_metrics
                st.session_state.portfolio = portfolio
                st.session_state.optimization_done = True
                
                progress_bar.progress(1.0)
                status_text.text("✅ Optimization complete!")
            
            st.success("🎉 Optimization completed successfully!")
            st.balloons()
        
        elif not st.session_state.optimization_done:
            st.info("👈 Configure parameters in the sidebar and click 'Run Optimization' to begin.")
            
            # Show example visualization
            st.markdown("#### 📚 Example: What to Expect")
            st.image("https://miro.medium.com/max/1400/1*9RKFmGfxVdXpGckhUYJZpg.gif", 
                    caption="PSO Algorithm Visualization", use_column_width=True)
    
    with tab2:
        st.markdown('<p class="sub-header">Optimization Results</p>', unsafe_allow_html=True)
        
        if st.session_state.optimization_done:
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            metrics = st.session_state.portfolio_metrics
            equal_metrics = st.session_state.equal_metrics
            
            with col1:
                st.metric(
                    "Expected Return",
                    format_percentage(metrics['expected_return']),
                    delta=format_percentage(metrics['expected_return'] - equal_metrics['expected_return'])
                )
            
            with col2:
                st.metric(
                    "Volatility",
                    format_percentage(metrics['volatility']),
                    delta=format_percentage(metrics['volatility'] - equal_metrics['volatility']),
                    delta_color="inverse"
                )
            
            with col3:
                st.metric(
                    "Sharpe Ratio",
                    f"{metrics['sharpe_ratio']:.4f}",
                    delta=f"{metrics['sharpe_ratio'] - equal_metrics['sharpe_ratio']:.4f}"
                )
            
            with col4:
                improvement = ((metrics['sharpe_ratio'] - equal_metrics['sharpe_ratio']) / 
                              equal_metrics['sharpe_ratio'] * 100)
                st.metric(
                    "Improvement",
                    f"{improvement:.2f}%",
                    delta="vs Equal-Weight"
                )
            
            st.markdown("---")
            
            # Portfolio allocation
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🥧 Optimal Portfolio Allocation")
                fig_allocation = plot_portfolio_allocation(
                    st.session_state.best_weights,
                    st.session_state.asset_names,
                    title="Optimal Portfolio Allocation",
                    interactive=True
                )
                st.plotly_chart(fig_allocation, use_container_width=True)
            
            with col2:
                st.markdown("#### 📋 Allocation Details")
                allocation_df = pd.DataFrame({
                    'Asset': st.session_state.asset_names,
                    'Weight': st.session_state.best_weights,
                    'Weight (%)': [format_percentage(w) for w in st.session_state.best_weights],
                    'Expected Return': [format_percentage(r) for r in st.session_state.expected_returns]
                })
                allocation_df = allocation_df.sort_values('Weight', ascending=False)
                st.dataframe(allocation_df[['Asset', 'Weight (%)', 'Expected Return']], 
                           hide_index=True, use_container_width=True)
            
            # Convergence plot
            st.markdown("#### 📈 Optimization Convergence")
            fig_convergence = plot_convergence(
                st.session_state.history,
                title="PSO Convergence History",
                interactive=True
            )
            st.plotly_chart(fig_convergence, use_container_width=True)
            
            # Download results
            st.markdown("#### 💾 Export Results")
            col1, col2 = st.columns(2)
            
            with col1:
                csv_data = save_results_to_csv(
                    st.session_state.best_weights,
                    st.session_state.asset_names,
                    st.session_state.portfolio_metrics
                )
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv_data,
                    file_name="portfolio_optimization_results.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Create summary text
                summary = f"""
# Portfolio Optimization Results

## Optimal Portfolio Metrics
- Expected Return: {format_percentage(metrics['expected_return'])}
- Volatility: {format_percentage(metrics['volatility'])}
- Sharpe Ratio: {metrics['sharpe_ratio']:.4f}

## Asset Allocation
"""
                for asset, weight in zip(st.session_state.asset_names, st.session_state.best_weights):
                    summary += f"- {asset}: {format_percentage(weight)}\n"
                
                st.download_button(
                    label="📥 Download Summary (TXT)",
                    data=summary,
                    file_name="portfolio_summary.txt",
                    mime="text/plain"
                )
        
        else:
            st.info("Run an optimization to see results.")
    
    with tab3:
        st.markdown('<p class="sub-header">Portfolio Analysis</p>', unsafe_allow_html=True)
        
        if st.session_state.optimization_done:
            # Efficient frontier
            st.markdown("#### 🎯 Efficient Frontier")
            fig_frontier = plot_efficient_frontier(
                st.session_state.expected_returns,
                st.session_state.cov_matrix,
                st.session_state.best_weights,
                n_portfolios=1000,
                risk_free_rate=st.session_state.portfolio.risk_free_rate,
                interactive=True
            )
            st.plotly_chart(fig_frontier, use_container_width=True)
            
            st.info("🔍 The red star shows the PSO-optimized portfolio on the efficient frontier. "
                   "The orange dashed line is the Capital Market Line, representing the best possible "
                   "risk-return trade-off.")
            
            # Strategy comparison
            st.markdown("#### 📊 Strategy Comparison")
            
            # Calculate different strategies
            strategies_metrics = {
                'Equal-Weight': st.session_state.equal_metrics,
                'PSO Optimal': st.session_state.portfolio_metrics
            }
            
            # Add minimum variance portfolio
            min_var_weights = np.linalg.inv(st.session_state.cov_matrix) @ np.ones(len(st.session_state.expected_returns))
            min_var_weights = min_var_weights / np.sum(min_var_weights)
            min_var_weights = st.session_state.portfolio.normalize_weights(min_var_weights)
            strategies_metrics['Min Variance'] = st.session_state.portfolio.get_portfolio_metrics(min_var_weights)
            
            fig_comparison = plot_portfolio_metrics_comparison(strategies_metrics, interactive=True)
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Detailed comparison table
            st.markdown("#### 📝 Detailed Strategy Comparison")
            comparison_df = pd.DataFrame({
                'Strategy': list(strategies_metrics.keys()),
                'Expected Return': [format_percentage(m['expected_return']) for m in strategies_metrics.values()],
                'Volatility': [format_percentage(m['volatility']) for m in strategies_metrics.values()],
                'Sharpe Ratio': [f"{m['sharpe_ratio']:.4f}" for m in strategies_metrics.values()]
            })
            st.dataframe(comparison_df, hide_index=True, use_container_width=True)
        
        else:
            st.info("Run an optimization to see analysis.")
    
    with tab4:
        st.markdown('<p class="sub-header">About PSO Portfolio Optimizer</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🐝 What is PSO?
            
            **Particle Swarm Optimization** is a computational method inspired by the social behavior 
            of bird flocking or fish schooling. It optimizes a problem by iteratively improving 
            candidate solutions (particles) with regard to a given measure of quality.
            
            #### How it works:
            1. **Initialize** a swarm of particles with random positions
            2. **Evaluate** each particle's fitness
            3. **Update** velocities based on personal and social knowledge
            4. **Move** particles to new positions
            5. **Repeat** until convergence or maximum iterations
            
            ### 💼 Portfolio Optimization
            
            The application uses PSO to solve the **Mean-Variance Portfolio Optimization** problem:
            
            **Objective**: Maximize Sharpe Ratio
            ```
            Sharpe Ratio = (Expected Return - Risk-Free Rate) / Portfolio Volatility
            ```
            
            **Constraints**:
            - Weights sum to 1
            - Non-negative weights (no short selling)
            - Optional: min/max weight per asset
            """)
        
        with col2:
            st.markdown("""
            ### 🎯 Key Features
            
            - **Real-time Visualization**: See optimization progress
            - **Interactive Controls**: Adjust parameters on the fly
            - **Multiple Strategies**: Compare different approaches
            - **Efficient Frontier**: Visualize risk-return trade-offs
            - **Export Results**: Download allocations and reports
            
            ### 📊 Metrics Explained
            
            - **Expected Return**: Average anticipated portfolio return
            - **Volatility**: Standard deviation of returns (risk measure)
            - **Sharpe Ratio**: Risk-adjusted return metric
            - **Diversification**: Spread of investments across assets
            
            ### 🚀 Getting Started
            
            1. Choose your data source (synthetic or demo stocks)
            2. Configure PSO parameters in the sidebar
            3. Set portfolio constraints
            4. Click "Run Optimization"
            5. Analyze results and export portfolio
            
            ### 📚 References
            
            - Kennedy & Eberhart (1995): PSO Algorithm
            - Markowitz (1952): Modern Portfolio Theory
            - Sharpe (1966): Sharpe Ratio
            
            ### 🔗 Links
            
            - [GitHub Repository](https://github.com/EU97/PSO_optimal)
            - [Documentation](https://github.com/EU97/PSO_optimal/docs)
            - [Report Issues](https://github.com/EU97/PSO_optimal/issues)
            """)
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; padding: 2rem;'>
            <p><strong>PSO Portfolio Optimizer v1.0.0</strong></p>
            <p>Built with ❤️ by EU97 | 2024</p>
            <p>Powered by Python, Streamlit, NumPy, and Plotly</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
