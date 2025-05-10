import numpy as np
import pandas as pd
import yfinance as yf
import vectorbt as vbt
import matplotlib.pyplot as plt
import itertools

# Function to calculate Zero Lag Moving Average (ZLMA)
def calculate_zlma(series, period=20):
    """
    Calculate Zero Lag Moving Average (ZLMA).
    """
    ema1 = series.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    zlma = 2 * ema1 - ema2
    return zlma

# Function to calculate Fisher Transform and its Signal line
def calculate_fisher_transform(df, period=10):
    """
    Calculate Fisher Transform.
    """
    high_rolling = df['High'].rolling(window=period).max()
    low_rolling = df['Low'].rolling(window=period).min()

    # Avoid division by zero
    range_rolling = high_rolling - low_rolling
    range_rolling[range_rolling == 0] = np.nan  # Replace 0 with NaN to avoid errors

    # Calculate X
    X = 2 * ((df['Close'] - low_rolling) / range_rolling - 0.5)

    # Fisher Transform
    fisher = 0.5 * np.log((1 + X) / (1 - X))

    # Signal line (Exponential Moving Average of Fisher)
    fisher_signal = fisher.ewm(span=9, adjust=False).mean()

    return fisher, fisher_signal

# Walk-forward optimization with ZLMA and Fisher Transform
def walk_forward_optimization_zlma_fisher(df, start_year, end_year):
    results = []

    # Define dynamic ranges for ZLMA and Fisher Transform periods
    zlma_period_range = range(1, 101)  # Range for ZLMA periods
    fisher_period_range = range(1, 101)  # Range for Fisher Transform periods

    for test_year in range(start_year + 2, end_year + 1):
        train_start = test_year - 2
        train_end = test_year - 1
        test_start = test_year

        train_data = df[(df.index.year >= train_start) & (df.index.year <= train_end)]
        test_data = df[df.index.year == test_year]

        best_params = None
        best_performance = -np.inf

        # Loop through all combinations of ZLMA and Fisher periods
        for params in itertools.product(zlma_period_range, fisher_period_range):
            zlma_period, fisher_period = params

            # Calculate ZLMA and Fisher Transform indicators on the training data
            train_data['ZLMA'] = calculate_zlma(train_data['Close'], zlma_period)
            train_data['Fisher'], train_data['Fisher_Signal'] = calculate_fisher_transform(train_data, fisher_period)

            # Generate entry and exit signals based on ZLMA and Fisher Transform
            entries = (train_data['Close'] > train_data['ZLMA']) & (train_data['Fisher'] > 0)
            exits = (train_data['Close'] < train_data['ZLMA']) & (train_data['Fisher'] < 0)

            # Backtest on training data
            portfolio = vbt.Portfolio.from_signals(
                close=train_data['Close'],
                entries=entries,
                exits=exits,
                init_cash=100_000,
                fees=0.001
            )

            performance = portfolio.total_return()
            if performance > best_performance:
                best_performance = performance
                best_params = (zlma_period, fisher_period)

        # Test with the best parameters on the test data
        yearly_data = df[(df.index.year >= test_year - 1) & (df.index.year <= test_year)]

        # Apply ZLMA and Fisher Transform indicators
        yearly_data['ZLMA'] = calculate_zlma(yearly_data['Close'], best_params[0])
        yearly_data['Fisher'], yearly_data['Fisher_Signal'] = calculate_fisher_transform(yearly_data, best_params[1])

        # Keep only the second year to avoid missing values from indicator calculation
        yearly_data = yearly_data[yearly_data.index.year == test_year]

        entries = (yearly_data['Close'] > yearly_data['ZLMA']) & (yearly_data['Fisher'] > 0)
        exits = (yearly_data['Close'] < yearly_data['ZLMA']) & (yearly_data['Fisher'] < 0)

        portfolio = vbt.Portfolio.from_signals(
            close=yearly_data['Close'],
            entries=entries,
            exits=exits,
            init_cash=100_000,
            fees=0.001
        )

        results.append({
            'Year': test_year,
            'Best_Params': best_params
        })

    return pd.DataFrame(results)

# Define the stock symbol and time period
symbol = 'HWM'
start_date = '2016-11-01'
end_date = '2025-01-01'

# Download the stock data
df = yf.download(symbol, start=start_date, end=end_date)
df.columns = ['Close', 'High', 'Low', 'Open', 'Volume']

# Perform walk-forward optimization
results = walk_forward_optimization_zlma_fisher(df, 2018, 2025)

# Display results
print("\nWalk-Forward Optimization Results:")
print(results)

# Combine signals into a single portfolio
combined_entries = pd.Series(False, index=df.index)
combined_exits = pd.Series(False, index=df.index)

for _, row in results.iterrows():
    year = row['Year']
    params = row['Best_Params']

    # Extend the data range to include the previous year for indicator calculation
    yearly_data = df[(df.index.year >= year - 1) & (df.index.year <= year)]

    # Apply ZLMA and Fisher Transform indicators
    yearly_data['ZLMA'] = calculate_zlma(yearly_data['Close'], params[0])
    yearly_data['Fisher'], yearly_data['Fisher_Signal'] = calculate_fisher_transform(yearly_data, params[1])

    # Keep only the second year to avoid missing values from indicator calculation
    yearly_data = yearly_data[yearly_data.index.year == year]

    # Define entry/exit conditions
    entries = (yearly_data['Close'] > yearly_data['ZLMA']) & (yearly_data['Fisher'] > 0)
    exits = (yearly_data['Close'] < yearly_data['ZLMA']) & (yearly_data['Fisher'] < 0)

    combined_entries.loc[entries.index] = entries
    combined_exits.loc[exits.index] = exits

# Filter data for testing period only
df = df[(df.index.year >= 2020) & (df.index.year <= 2025)]
combined_entries = combined_entries[(combined_entries.index.year >= 2020) & (combined_entries.index.year <= 2025)]
combined_exits = combined_exits[(combined_exits.index.year >= 2020) & (combined_exits.index.year <= 2025)]

# Backtest using the combined signals
portfolio = vbt.Portfolio.from_signals(
    close=df['Close'],
    entries=combined_entries,
    exits=combined_exits,
    init_cash=100_000,
    fees=0.001
)

# Display performance metrics
print(portfolio.stats())

# Plot equity curve
portfolio.plot().show()