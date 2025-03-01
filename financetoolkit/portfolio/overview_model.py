"""Overview Model"""

import numpy as np
import pandas as pd

# pylint: disable=too-many-locals

# Matches up with currency codes EUR, USD, JPY etc. This is used for
# Yahoo Finance's notation of currencies. E.g. EURUSD=X
CURRENCY_CODE_LENGTH = 3


def create_transactions_overview(
    portfolio_volume: pd.Series,
    portfolio_price: pd.Series,
    portfolio_costs: pd.Series,
    latest_returns: pd.Series,
):
    """
    Generate a detailed overview of transaction performance for individual assets in a portfolio.

    This function calculates key performance metrics for each asset in the portfolio, including:
    - "Invested Amount": The initial investment made for each asset, accounting for transaction costs.
    - "Current Value": The current value of the asset based on the latest returns.
    - "% Return": The percentage change in value from the invested amount to the current value.
    - "Return": The absolute change in value from the invested amount to the current value.

    Args:
        portfolio_volume (pd.Series): A Series containing the number of units (volumes) of each asset in the portfolio.
        portfolio_price (pd.Series): A Series containing the purchase price per unit for each asset.
        portfolio_costs (pd.Series): A Series containing the transaction costs associated with each asset.
        latest_returns (pd.Series): A Series containing the most recent market value per unit for each asset.

    Returns:
        pd.DataFrame: A DataFrame with the following columns for each asset:
            - "Invested Amount": The total cost of acquiring the asset, including transaction costs.
            - "Current Value": The current total value of the asset in the portfolio.
            - "% Return": The percentage return on investment.
            - "Return": The absolute profit or loss.

            The DataFrame is indexed by the asset tickers from the `portfolio_volume` index.
            Missing or invalid values are replaced with 0.0.
    """
    invested_amount = []
    current_value = []
    percentage_return = []
    value_return = []

    for row, (_, ticker) in enumerate(portfolio_volume.index):
        bought_value = portfolio_volume.iloc[row] * portfolio_price.iloc[row] - abs(
            portfolio_costs.iloc[row]
        )

        recent_value = (
            portfolio_volume.iloc[row] * latest_returns.loc[ticker]
            if portfolio_volume.iloc[row] > 0
            else np.nan
        )

        invested_amount.append(bought_value)
        current_value.append(recent_value)
        percentage_return.append((recent_value / bought_value) - 1)
        value_return.append(recent_value - bought_value)

    new_columns = pd.DataFrame(
        [invested_amount, current_value, percentage_return, value_return],
        columns=portfolio_volume.index,
        index=["Invested Amount", "Current Value", "% Return", "Return"],
    ).T

    new_columns = new_columns.fillna(0.0)

    return new_columns


def create_portfolio_overview(
    portfolio_name: pd.Series,
    portfolio_volume: pd.Series,
    portfolio_price: pd.Series,
    portfolio_costs: pd.Series,
    latest_returns: pd.Series,
    benchmark_prices: pd.Series,
    benchmark_latest_prices: pd.Series,
    volatilities: pd.Series,
    betas: pd.Series,
    include_portfolio: bool = True,
):
    """
    Generate a comprehensive overview of portfolio positions and related performance metrics.

    This function calculates key statistics, performance metrics, and portfolio weights
    for each asset, along with an aggregated portfolio summary if requested. Metrics
    include invested amounts, current values, returns, alpha, and weights.

    Args:
        portfolio_name (pd.Series): A Series containing the names or identifiers of the assets in the portfolio.
        portfolio_volume (pd.Series): A Series containing the number of units (volumes) held for each asset.
        portfolio_price (pd.Series): A Series containing the purchase price per unit for each asset.
        portfolio_costs (pd.Series): A Series containing transaction costs for each asset.
        latest_returns (pd.Series): A Series containing the latest market prices for each asset.
        benchmark_prices (pd.Series): A Series containing historical benchmark prices for comparison.
        benchmark_latest_prices (pd.Series): A Series containing the latest benchmark prices for each asset.
        include_portfolio (bool): A flag indicating whether to include an aggregated portfolio summary row.
            Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing the following metrics for each asset:
            - "Volume": The total number of units held.
            - "Costs": The total transaction costs.
            - "Price": The average purchase price per unit.
            - "Invested": The total initial investment amount, including transaction costs.
            - "Latest Price": The most recent market price per unit.
            - "Latest Value": The total current value based on the latest prices.
            - "Return": The percentage change in value since the initial investment.
            - "Return Value": The absolute change in value since the initial investment.
            - "Benchmark Return": The percentage return based on benchmark prices.
            - "Alpha": The difference between asset return and benchmark return.
            - "Weight": The proportion of the asset's current value relative to the total portfolio value.

        If `include_portfolio` is True, the DataFrame includes a "Portfolio" row with aggregated metrics
        for the entire portfolio.

    Notes:
        - The function automatically handles missing or invalid values, ensuring consistent outputs.
        - Metrics such as "Alpha" and "Weight" are calculated based on individual asset performance
          relative to benchmarks and the portfolio total, respectively.
        - All financial values are aggregated or weighted appropriately for the portfolio row.
    """
    portfolio_overview = pd.DataFrame(index=portfolio_name.index)

    portfolio_overview["Name"] = portfolio_name
    portfolio_overview["Volume"] = portfolio_volume
    portfolio_overview["Costs"] = portfolio_costs
    portfolio_overview["Invested"] = portfolio_volume * portfolio_price - abs(
        portfolio_costs
    )
    portfolio_overview["Benchmark Invested"] = (
        portfolio_volume * benchmark_prices.to_numpy()
    )

    portfolio_overview_grouped = portfolio_overview.groupby(
        "Identifier", observed=True
    ).agg(
        {
            "Volume": "sum",
            "Costs": "sum",
            "Invested": "sum",
            "Benchmark Invested": "sum",
        }
    )

    benchmark_volatility = volatilities.loc["Benchmark"] * np.sqrt(252)
    asset_volatilities = volatilities.drop("Benchmark") * np.sqrt(252)
    asset_volatilities = asset_volatilities.reindex(portfolio_overview_grouped.index)
    betas = betas.reindex(portfolio_overview_grouped.index)

    latest_price_values = []
    for ticker in portfolio_overview_grouped.index:
        latest_price_values.append(latest_returns.loc[ticker])

    portfolio_overview_grouped.insert(
        2,
        "Price",
        (
            portfolio_overview_grouped["Invested"]
            + abs(portfolio_overview_grouped["Costs"])
        )
        / portfolio_overview_grouped["Volume"],
    )

    portfolio_overview_grouped["Latest Price"] = latest_price_values
    portfolio_overview_grouped["Latest Value"] = (
        portfolio_overview_grouped["Volume"]
        * portfolio_overview_grouped["Latest Price"]
    )

    benchmark_latest_value = (
        portfolio_overview_grouped["Volume"] * benchmark_latest_prices
    )

    portfolio_overview_grouped["Return"] = (
        portfolio_overview_grouped["Latest Value"]
        / portfolio_overview_grouped["Invested"]
        - 1
    )
    portfolio_overview_grouped["Return Value"] = (
        portfolio_overview_grouped["Latest Value"]
        - portfolio_overview_grouped["Invested"]
    )
    portfolio_overview_grouped["Benchmark Return"] = (
        benchmark_latest_value / portfolio_overview_grouped["Benchmark Invested"] - 1
    )

    portfolio_overview_grouped["Volatility"] = asset_volatilities
    portfolio_overview_grouped["Benchmark Volatility"] = benchmark_volatility

    portfolio_overview_grouped["Alpha"] = (
        portfolio_overview_grouped["Return"]
        - portfolio_overview_grouped["Benchmark Return"]
    )

    portfolio_overview_grouped["Beta"] = betas
    portfolio_overview_grouped["Weight"] = (
        portfolio_overview_grouped["Latest Value"]
        / portfolio_overview_grouped["Latest Value"].sum()
    )

    if include_portfolio:
        # Compute aggregated totals for numeric columns
        totals = portfolio_overview_grouped.sum(numeric_only=True)
        vol = totals["Volume"]
        costs = totals["Costs"]
        invested = totals["Invested"]
        latest_value = totals["Latest Value"]
        benchmark_invested = totals["Benchmark Invested"]
        benchmark_latest_value_sum = benchmark_latest_value.sum()

        portfolio_volatility = sum(
            portfolio_overview_grouped["Volatility"]
            * portfolio_overview_grouped["Weight"]
        )
        portfolio_beta = sum(
            portfolio_overview_grouped["Beta"] * portfolio_overview_grouped["Weight"]
        )

        portfolio_summary = {
            "Volume": vol,
            "Costs": costs,
            "Price": np.nan,
            "Invested": invested,
            "Latest Price": np.nan,
            "Latest Value": latest_value,
            "Return": np.nan,
            "Return Value": np.nan,
            "Benchmark Return": np.nan,
            "Volatility": portfolio_volatility,
            "Benchmark Volatility": benchmark_volatility,
            "Alpha": np.nan,
            "Beta": portfolio_beta,
            "Weight": portfolio_overview_grouped["Weight"].sum(),
        }

        # Calculate derived metrics only if volume and invested are non-zero
        portfolio_summary["Price"] = (invested - abs(costs)) / vol if vol else np.nan
        portfolio_summary["Latest Price"] = latest_value / vol if vol else np.nan
        portfolio_summary["Return"] = (
            (latest_value / invested - 1) if invested else np.nan
        )
        portfolio_summary["Return Value"] = latest_value - invested
        portfolio_summary["Benchmark Return"] = (
            (benchmark_latest_value_sum / benchmark_invested - 1)
            if benchmark_invested
            else np.nan
        )
        portfolio_summary["Alpha"] = (
            portfolio_summary["Return"] - portfolio_summary["Benchmark Return"]
        )

        # Insert the portfolio summary row into the DataFrame
        portfolio_overview_grouped.loc["Portfolio"] = pd.Series(portfolio_summary)

    portfolio_overview_grouped = portfolio_overview_grouped.drop(
        columns=["Benchmark Invested"]
    )

    return portfolio_overview_grouped


def create_transactions_performance(
    portfolio_dataset: pd.DataFrame,
    ticker_column: str,
    date_column: str,
    volume_column: str,
    price_column: str,
    costs_column: str,
    period_prices: pd.DataFrame,
    period_string: str,
    benchmark_specific_prices: pd.Series,
    benchmark_period_prices: pd.DataFrame,
):
    """
    Calculate transaction performance metrics over a specified period.

    This function computes detailed transaction performance metrics for portfolio assets,
    including returns, invested amounts, current values, and benchmarks. The metrics
    are calculated based on historical data for the specified period.

    Args:
        portfolio_dataset (pd.DataFrame): A DataFrame containing portfolio transaction details,
            including asset tickers, transaction dates, volumes, prices, and costs.
        ticker_column (str): The name of the column representing asset tickers in `portfolio_dataset`.
        date_column (str): The name of the column representing transaction dates in `portfolio_dataset`.
        volume_column (str): The name of the column representing transaction volumes in `portfolio_dataset`.
        price_column (str): The name of the column representing transaction prices in `portfolio_dataset`.
        costs_column (str): The name of the column representing transaction costs in `portfolio_dataset`.
        period_prices (pd.DataFrame): A DataFrame containing historical price data for assets during
            the specified period, indexed by date and asset ticker.
        period_string (str): The time period for grouping transactions. Supported values are
            'yearly', 'quarterly', 'monthly', 'weekly', or 'daily'.
        benchmark_specific_prices (pd.Series): A Series containing specific benchmark prices for
            assets, aligned with the `portfolio_dataset`.
        benchmark_period_prices (pd.DataFrame): A DataFrame containing historical benchmark prices
            for the specified period, indexed by date.

    Returns:
        pd.DataFrame: A DataFrame containing the following metrics for each asset and period:
            - "Volume": Total transaction volume during the period.
            - "Price": Average transaction price during the period.
            - "Costs": Total transaction costs during the period.
            - "Invested Amount": The total invested value, including transaction costs.
            - "Current Value": The total current value based on end-of-period prices.
            - "Return": The percentage return on investment during the period.
            - "Benchmark Return": The benchmark return for the same period.
            - "Alpha": The difference between the asset return and benchmark return.

    Raises:
        ValueError: If an unsupported value is provided for `period_string`.

    Notes:
        - The function aligns transaction data with the specified period frequencies
          using the `date_column`.
        - Metrics such as "Return" and "Alpha" are calculated for each asset based on the
          period prices and benchmark prices.
        - The "Benchmark Price" column is excluded from the final output as it is used only
          for intermediate calculations.
    """
    portfolio_dataset["Benchmark Price"] = benchmark_specific_prices.to_numpy()

    period_performance = portfolio_dataset.reset_index()

    period_performance = period_performance.set_index([date_column, ticker_column])

    dates = period_performance.index.get_level_values(date_column).asfreq(period_string)

    period_performance_grouped = period_performance.groupby(
        [dates, ticker_column], observed=True
    ).agg(
        {
            volume_column: "sum",
            price_column: "mean",
            costs_column: "sum",
            "Benchmark Price": "mean",
        }
    )

    period_performance_grouped["Invested Amount"] = period_performance_grouped[
        volume_column
    ] * period_performance_grouped[price_column] + abs(
        period_performance_grouped[costs_column]
    )

    last_prices = []
    last_benchmark_prices = []

    for period, _ in period_performance_grouped.iterrows():
        last_prices.append(period_prices.loc[period[0], period[1]])
        last_benchmark_prices.append(benchmark_period_prices.loc[period[0]])

    period_performance_grouped["Current Value"] = (
        period_performance_grouped[volume_column] * last_prices
    )

    period_performance_grouped["Return"] = (
        period_performance_grouped["Current Value"]
        / period_performance_grouped["Invested Amount"]
        - 1
    )

    total_benchmark_invested = (
        period_performance_grouped[volume_column]
        * period_performance_grouped["Benchmark Price"]
    )
    end_of_period_benchmark_value = (
        period_performance_grouped[volume_column] * last_benchmark_prices
    )

    period_performance_grouped["Benchmark Return"] = (
        end_of_period_benchmark_value / total_benchmark_invested - 1
    )

    period_performance_grouped["Alpha"] = (
        period_performance_grouped["Return"]
        - period_performance_grouped["Benchmark Return"]
    )

    # Dropping Benchmark Prices since it has no meaning in the current layout
    period_performance_grouped = period_performance_grouped.drop(
        "Benchmark Price", axis=1
    )

    return period_performance_grouped


def create_positions_overview(
    portfolio_tickers: list[str],
    period_dates: pd.DatetimeIndex,
    portfolio_dataset: pd.DataFrame,
    historical_prices: pd.Series,
):
    """
    Calculate an overview of portfolio positions and associated performance metrics over a specified period.

    This function generates a detailed overview of the portfolio’s positions, including key statistics such as
    current value, cumulative return, invested weight, and current weight for each asset in the portfolio.
    The resulting DataFrame summarizes these metrics over the specified dates.

    Args:
        portfolio_tickers (list[str]): A list of tickers representing assets in the portfolio.
        period_dates (pd.DatetimeIndex): A datetime index representing the dates for which the position metrics
            should be calculated.
        portfolio_dataset (pd.DataFrame): The dataset containing portfolio positions, typically with multi-index
            levels like tickers and dates, and columns such as 'Volume', 'Costs', 'Invested Amount', and others.
        historical_prices (pd.Series): A Series containing the historical prices of the assets, typically with
            a 'Adj Close' column used for calculating the 'Current Value' of positions.

    Returns:
        pd.DataFrame: A DataFrame containing the following metrics for each asset and date:
            - "Volume": The cumulative volume of positions for each asset.
            - "Costs": The cumulative transaction costs for each asset.
            - "Invested Amount": The cumulative invested amount for each asset.
            - "Current Value": The cumulative current value of each asset, calculated using historical prices.
            - "Cumulative Return": The return of each asset based on the 'Current Value' and 'Invested Amount'.
            - "Invested Weight": The weight of each asset relative to the total invested amount in the portfolio.
            - "Current Weight": The weight of each asset relative to the total current value of the portfolio.

    Notes:
        - The function assumes that `portfolio_dataset` has a multi-level index with tickers and dates.
        - Missing values in the dataset are filled with zeros, and cumulative sums are calculated where applicable.
        - The weights are normalized based on the total invested amount and current value across all
        positions in the portfolio.
    """
    columns = ["Volume", "Costs", "Invested Amount"]

    positions = {}
    for column in columns:
        positions[column] = pd.DataFrame(index=period_dates, columns=portfolio_tickers)

        for ticker in portfolio_tickers:
            positions[column].loc[:, ticker] = (
                portfolio_dataset.loc[:, ticker, :][column].groupby(level=0).sum()
            )

        positions[column] = positions[column].astype(float).fillna(0.0).cumsum()

    positions["Current Value"] = positions["Volume"] * historical_prices["Adj Close"]
    positions["Cumulative Return"] = (
        positions["Current Value"] / positions["Invested Amount"]
    )
    positions["Invested Weight"] = positions["Invested Amount"].div(
        positions["Invested Amount"].sum(axis=1), axis=0
    )
    positions["Current Weight"] = positions["Current Value"].div(
        positions["Current Value"].sum(axis=1), axis=0
    )

    positions_df = pd.concat(positions, axis=1).fillna(0)

    return positions_df


def create_portfolio_performance(
    positions_dataset: pd.DataFrame,
    date_column: str,
    ticker_column: str,
    period_string: str,
):
    """
    Calculate portfolio performance metrics based on the positions dataset.

    This function calculates various portfolio performance metrics, such as returns, for the specified period.
    It aggregates and calculates metrics for each date and ticker combination, providing insights into portfolio
    performance over different time periods (e.g., yearly, quarterly, etc.).

    Args:
        positions_dataset (pd.DataFrame): The dataset containing portfolio positions, typically with multi-level
            indices (date and ticker). It includes columns like 'Volume', 'Costs', 'Invested Amount', 'Current Value',
            'Invested Weight', and 'Current Weight'.
        date_column (str): The name of the column representing the dates in `positions_dataset`.
        ticker_column (str): The name of the column representing the tickers in `positions_dataset`.
        period_string (str): The time period for which portfolio performance metrics should be calculated. This can be
            'yearly', 'quarterly', 'monthly', 'weekly', or 'daily'.

    Returns:
        pd.DataFrame: A DataFrame containing portfolio performance metrics aggregated by date and ticker.
            The metrics include:
            - 'Volume': The portfolio volume at the end of the period.
            - 'Costs': The transaction costs for the period.
            - 'Invested Amount': The total amount invested in each asset at the end of the period.
            - 'Current Value': The value of each asset at the end of the period.
            - 'Invested Weight': The proportion of the portfolio invested in each asset.
            - 'Current Weight': The proportion of the portfolio's current value allocated to each asset.
            - 'Return': The return on each asset based on its 'Current Value' and 'Invested Amount'.

    Raises:
        ValueError: If an invalid or unsupported `period_string` is provided.

    Notes:
        - The function aggregates data at the specified frequency using the `.asfreq()` method.
        - Missing values in the resulting dataset are filled with zeros.
    """
    positions_dataset_stacked = positions_dataset.stack(future_stack=True)  # noqa
    positions_dataset_stacked.index.names = [date_column, ticker_column]

    dates = positions_dataset_stacked.index.get_level_values(date_column).asfreq(
        period_string
    )
    tickers = positions_dataset_stacked.index.get_level_values(ticker_column)

    positions_dataset_grouped = positions_dataset_stacked.groupby(
        [dates, tickers], observed=True
    ).agg(
        {
            "Volume": "last",
            "Costs": "last",
            "Invested Amount": "last",
            "Current Value": "last",
            "Invested Weight": "last",
            "Current Weight": "last",
        }
    )

    positions_dataset_grouped["Return"] = (
        positions_dataset_grouped["Current Value"]
        / positions_dataset_grouped["Invested Amount"]
        - 1
    )

    positions_dataset_grouped = positions_dataset_grouped.fillna(0.0)

    return positions_dataset_grouped


def create_profit_and_loss_overview(
    transactions_overview: pd.DataFrame,
    ticker_column: str,
    volume_column: str,
    price_column: str,
    method="FIFO",
):
    """
    Calculate Profit and Loss (PnL) using FIFO, LIFO, or Average Price methods for multiple tickers.

    This function calculates the realized PnL for each transaction based on the chosen inventory
    valuation method (FIFO, LIFO, or Average Price). It computes both the individual PnL per transaction
    and the cumulative PnL for each ticker.

    Args:
        transactions_overview (pd.DataFrame): DataFrame containing transaction data, with columns for tickers,
            transaction volumes, and transaction prices.
        ticker_column (str): The column name representing the tickers (e.g., asset names).
        volume_column (str): The column name representing the transaction volumes
            (positive for buys, negative for sells).
        price_column (str): The column name representing the transaction prices.
        method (str, optional): Method for inventory valuation.
            Can be 'FIFO', 'LIFO', or 'AVERAGE'. Default is 'FIFO'.

    Returns:
        pd.DataFrame: DataFrame with additional columns for 'PnL' (realized profit and loss per transaction)
        and 'Cumulative PnL' (cumulative profit and loss).

    Notes:
        - FIFO (First-In, First-Out): Sells the earliest purchased assets first.
        - LIFO (Last-In, First-Out): Sells the most recently purchased assets first.
        - AVERAGE: Uses the average cost of all assets in inventory for PnL calculation.
    """

    def process_group(group):
        inventory = []  # For FIFO or LIFO methods
        avg_cost = 0  # For Average method
        total_volume = 0  # For Average method
        pnl = []  # Store PnL for each row in the group

        for _, row in group.iterrows():
            volume = row[volume_column]
            price = row[price_column]

            if volume > 0:  # Buy order
                if method == "AVERAGE":
                    # Update average cost
                    total_cost = avg_cost * total_volume
                    total_cost += volume * price
                    total_volume += volume
                    avg_cost = total_cost / total_volume
                else:
                    # Add to inventory for FIFO or LIFO
                    inventory.append((volume, price))
                pnl.append(0)  # No PnL for buy orders

            elif volume < 0:  # Sell order
                sell_volume = -volume
                realized_pnl = 0

                if method == "FIFO":
                    # Process FIFO: Sell the earliest assets first
                    while sell_volume > 0 and inventory:
                        fifo_volume, fifo_price = inventory[0]
                        if fifo_volume <= sell_volume:
                            realized_pnl += fifo_volume * (price - fifo_price)
                            sell_volume -= fifo_volume
                            inventory.pop(0)
                        else:
                            realized_pnl += sell_volume * (price - fifo_price)
                            inventory[0] = (fifo_volume - sell_volume, fifo_price)
                            sell_volume = 0

                elif method == "LIFO":
                    # Process LIFO: Sell the most recently purchased assets first
                    while sell_volume > 0 and inventory:
                        lifo_volume, lifo_price = inventory.pop()
                        if lifo_volume <= sell_volume:
                            realized_pnl += lifo_volume * (price - lifo_price)
                            sell_volume -= lifo_volume
                        else:
                            realized_pnl += sell_volume * (price - lifo_price)
                            inventory.append((lifo_volume - sell_volume, lifo_price))
                            sell_volume = 0

                elif method == "AVERAGE":
                    # Calculate PnL based on average cost
                    if total_volume >= sell_volume:
                        realized_pnl = sell_volume * (price - avg_cost)
                        total_volume -= sell_volume
                    else:
                        realized_pnl = total_volume * (price - avg_cost)
                        total_volume = 0

                pnl.append(realized_pnl)

            else:
                pnl.append(0)  # No action for zero volume

        group["PnL"] = pnl
        return group

    # Process each ticker group independently
    transactions_overview = transactions_overview.groupby(
        ticker_column, group_keys=False, observed=True
    ).apply(process_group)
    transactions_overview["Cumulative PnL"] = transactions_overview["PnL"].cumsum()

    return transactions_overview[["PnL", "Cumulative PnL"]]
