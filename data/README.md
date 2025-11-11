# Stock Data

This directory contains sample stock price data for portfolio optimization.

## Files

### stock_data.csv

Sample historical stock prices for five major technology companies:

- **AAPL**: Apple Inc.
- **GOOGL**: Alphabet Inc. (Google)
- **MSFT**: Microsoft Corporation
- **AMZN**: Amazon.com Inc.
- **TSLA**: Tesla Inc.

## Data Format

- **Date**: Trading date (YYYY-MM-DD)
- **Columns**: Stock ticker symbols
- **Values**: Adjusted closing prices (normalized to 100 on start date)

## Usage

The data is automatically loaded by the application when using the "Real Stock Data" option. You can also load custom data:

```python
import pandas as pd
from src.portfolio import PortfolioOptimizer

# Load data
df = pd.read_csv('data/stock_data.csv', index_col='Date', parse_dates=True)

# Calculate returns
returns = df.pct_change().dropna()

# Calculate statistics
expected_returns = returns.mean() * 252  # Annualized
cov_matrix = returns.cov() * 252  # Annualized

# Create portfolio optimizer
portfolio = PortfolioOptimizer(
    expected_returns=expected_returns.values,
    covariance_matrix=cov_matrix.values
)
```

## Data Sources

For real-world applications, consider using:

- **Yahoo Finance**: [yfinance library](https://pypi.org/project/yfinance/)
- **Alpha Vantage**: [API](https://www.alphavantage.co/)
- **Quandl**: [Financial data](https://www.quandl.com/)
- **Bloomberg Terminal**: Professional data service

## Notes

- Sample data is for demonstration purposes only
- Not intended for actual investment decisions
- Real market data should be used for production applications
- Consider survivorship bias when using historical data
