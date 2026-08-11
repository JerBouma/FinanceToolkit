"""Overview Model"""

import numpy as np
import pandas as pd

from financetoolkit.utilities import logger_model

logger = logger_model.get_logger()

# pylint: disable=too-many-locals

# Matches currency codes (EUR, USD) in Yahoo Finance notation, e.g. EURUSD=X.
CURRENCY_CODE_LENGTH = 3

# Trading days used to annualize, matching the Risk module's convention.
TRADING_DAYS_PER_YEAR = 252


def _safe_divide(
    numerator: pd.Series | pd.DataFrame, denominator: pd.Series | pd.DataFrame
) -> pd.Series | pd.DataFrame:
    """
    Divide two objects, returning NaN wherever the denominator is zero or negative.

    A return is only defined against capital that was actually deployed. Dividing by a
    zero or negative denominator produces an infinite or sign-flipped number that reads
    as a real result, so those cases are reported as missing instead.

    Args:
        numerator (pd.Series | pd.DataFrame): The numerator of the division.
        denominator (pd.Series | pd.DataFrame): The denominator of the division.

    Returns:
        pd.Series | pd.DataFrame: The element-wise division with non-positive
            denominators replaced by NaN.
    """
    result = numerator / denominator.where(denominator > 0)

    return result.replace([np.inf, -np.inf], np.nan)


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
        # Transaction costs are part of what the position cost to acquire, so they are
        # added to the invested amount rather than netted off it.
        bought_value = portfolio_volume.iloc[row] * portfolio_price.iloc[row] + abs(
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


def _calculate_portfolio_volatility(
    weights: pd.Series,
    fallback_volatilities: pd.Series,
    asset_returns: pd.DataFrame | None,
) -> float:
    """
    Calculate the annualized portfolio volatility using the full covariance matrix of
    asset returns, i.e. Var_p = w^T * Cov * w (Markowitz, 1952).

    This is the theoretically correct way to combine individual asset volatilities into a
    single portfolio-level volatility because variance is not linear in weights -- unlike a
    weighted average, it captures diversification benefits from imperfectly correlated
    (or negatively correlated) assets. A naive weighted average of individual volatilities
    (sum_i w_i * sigma_i) is only equal to the true portfolio volatility when every pair of
    assets is perfectly correlated (rho_ij = 1); in every other case it overstates the true
    portfolio volatility.

    Args:
        weights (pd.Series): The portfolio weight of each asset (e.g. based on latest value),
            indexed by ticker.
        fallback_volatilities (pd.Series): Individual annualized asset volatilities, indexed by
            ticker. Used only when `asset_returns` is not available, as a documented approximation.
        asset_returns (pd.DataFrame | None): Daily returns for each asset, indexed by date/period,
            with one column per ticker. When available, the most recent calendar year of returns is
            used to build the covariance matrix, consistent with how individual asset volatilities
            are annualized elsewhere (see risk_model.get_volatility with period="yearly").

    Returns:
        float: The annualized portfolio volatility (standard deviation).

    References:
        Markowitz, H. (1952). Portfolio Selection. The Journal of Finance, 7(1), 77-91.
    """
    if asset_returns is not None and not asset_returns.empty:
        returns_for_covariance = asset_returns.reindex(columns=weights.index)

        # The most recent calendar year, matching the individual Volatility column.
        periods = returns_for_covariance.index.asfreq("Y")
        recent_returns = returns_for_covariance[periods == periods[-1]]

        # Columns that are entirely missing would propagate NaNs across the matrix.
        recent_returns = recent_returns.dropna(axis=1, how="all")

        if not recent_returns.empty and len(recent_returns.columns) > 0:
            annualized_covariance = recent_returns.cov() * TRADING_DAYS_PER_YEAR
            aligned_weights = weights.reindex(annualized_covariance.index).fillna(0.0)

            portfolio_variance = (
                aligned_weights.to_numpy()
                @ annualized_covariance.to_numpy()
                @ aligned_weights.to_numpy()
            )

            if portfolio_variance >= 0:
                return float(np.sqrt(portfolio_variance))

    # Ignores correlation and so overstates it; used only when no series exist.
    return float(sum(fallback_volatilities * weights))


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
    asset_returns: pd.DataFrame | None = None,
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
        asset_returns (pd.DataFrame | None, optional): Daily returns for each portfolio ticker, indexed by
            date/period. When provided, the portfolio-level "Volatility" is computed from the full
            covariance matrix of these returns (Markowitz, 1952), which correctly accounts for
            diversification across imperfectly-correlated assets. When omitted, the portfolio
            volatility falls back to a weighted average of individual asset volatilities, which
            ignores cross-asset correlation and therefore overstates true portfolio volatility unless
            all assets are perfectly correlated. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing the following metrics for each asset:
            - "Volume": The net number of units held, i.e. all buys minus all sells.
            - "Costs": The total transaction costs, reported with the sign found in the transactions.
            - "Price": The volume-weighted average purchase price per unit, i.e. the total value of
              every buy divided by the total number of units bought. Sells do not enter this figure.
            - "Invested": The total capital deployed, i.e. the value of every buy plus the absolute
              transaction costs. Sale proceeds are not netted off, so this figure never turns negative
              and always represents money that was actually put in.
            - "Latest Price": The most recent market price per unit.
            - "Latest Value": The market value of the position still held (Volume * Latest Price).
            - "Return": The total return on capital deployed, i.e. Return Value / Invested. This
              includes both realized and unrealized results.
            - "Return Value": The absolute profit or loss, i.e. Latest Value plus all sale proceeds
              minus Invested. This equals realized PnL + unrealized PnL - transaction costs.
            - "Benchmark Return": The return the identical cash flows would have produced in the
              benchmark, i.e. every buy and sell is replicated in benchmark units at the benchmark
              price of that transaction date, and the resulting position is valued at the latest
              benchmark price.
            - "Alpha": The difference between asset return and benchmark return.
            - "Weight": The proportion of the asset's current market value relative to the total
              current market value of the portfolio.

        If `include_portfolio` is True, the DataFrame includes a "Portfolio" row with aggregated metrics
        for the entire portfolio.

    Notes:
        - The cost basis convention is total capital deployed rather than an inventory method: neither
          FIFO, LIFO nor average cost is applied here because "Return" measures the result of every
          euro put into the position, not the basis of the units that happen to remain. The inventory
          methods are used for the realized PnL in `create_profit_and_loss_overview`.
        - The decomposition Return Value = realized PnL + unrealized PnL - transaction costs holds
          exactly, where realized PnL is the FIFO/LIFO/average figure of
          `create_profit_and_loss_overview` and unrealized PnL is the remaining position marked to
          the latest price against its own basis. Transaction costs appear once, inside "Invested",
          and are deliberately not also deducted from the realized PnL.
        - "Return" and "Benchmark Return" are NaN whenever no capital was deployed (Invested <= 0),
          since a return against a zero denominator is undefined.
        - The benchmark is dollar-matched rather than share-matched: it receives exactly the cash
          flows of the position on exactly the same dates, which makes "Alpha" a like-for-like
          comparison. Matching share counts instead would weight the benchmark by the number of units
          traded, which has no meaning across assets at different price levels.
        - The portfolio "Return" is computed from aggregated totals, which is identical to an
          invested-amount weighted average of the individual asset returns.
        - The portfolio "Beta" is a weighted average of individual betas, which is mathematically exact
          because beta is linear in portfolio weights: Cov(sum_i w_i * R_i, R_m) / Var(R_m)
          = sum_i w_i * Beta_i.
        - The portfolio "Volatility" is fundamentally different: variance is NOT linear in weights, so a
          weighted average of individual volatilities is only an upper-bound approximation (exact only
          if all assets are perfectly correlated). The correct formula is the full covariance matrix
          form Var_p = w^T * Cov * w (Markowitz, H. (1952). "Portfolio Selection". The Journal of
          Finance, 7(1), 77-91.). This is used whenever `asset_returns` is supplied.
        - The portfolio row's "Volume", "Price" and "Latest Price" sum or average units of different
          assets and therefore carry no economic meaning; they are reported only for completeness.

    References:
        Markowitz, H. (1952). Portfolio Selection. The Journal of Finance, 7(1), 77-91.
    """
    portfolio_overview = pd.DataFrame(index=portfolio_name.index)

    transaction_value = portfolio_volume * portfolio_price

    portfolio_overview["Name"] = portfolio_name
    portfolio_overview["Volume"] = portfolio_volume
    portfolio_overview["Costs"] = portfolio_costs
    # Buys and sells are tracked separately so that sale proceeds never shrink the amount
    # of capital that was deployed. Netting them results in a denominator that can reach
    # zero or turn negative once a position has returned more cash than was put in, which
    # in turn flips the sign of the reported return.
    portfolio_overview["Buy Volume"] = portfolio_volume.where(portfolio_volume > 0, 0.0)
    portfolio_overview["Buy Value"] = transaction_value.where(portfolio_volume > 0, 0.0)
    portfolio_overview["Proceeds"] = (-transaction_value).where(
        portfolio_volume < 0, 0.0
    )
    # Transaction costs are part of what the position cost to acquire, so they are added
    # to the invested amount rather than netted off it.
    portfolio_overview["Invested"] = portfolio_overview["Buy Value"] + abs(
        portfolio_costs
    )
    # The benchmark receives the identical cash flow on the identical date, so it is
    # matched in money rather than in share count.
    portfolio_overview["Benchmark Volume"] = (
        transaction_value / benchmark_prices.to_numpy()
    )

    portfolio_overview_grouped = portfolio_overview.groupby(
        "Identifier", observed=True
    ).agg(
        {
            "Volume": "sum",
            "Costs": "sum",
            "Buy Volume": "sum",
            "Buy Value": "sum",
            "Proceeds": "sum",
            "Invested": "sum",
            "Benchmark Volume": "sum",
        }
    )

    # volatilities is already annualized, so no annualization happens here.
    benchmark_volatility = volatilities.loc["Benchmark"]
    asset_volatilities = volatilities.drop("Benchmark")
    asset_volatilities = asset_volatilities.reindex(portfolio_overview_grouped.index)
    betas = betas.reindex(portfolio_overview_grouped.index)

    portfolio_overview_grouped.insert(
        2,
        "Price",
        # The volume-weighted price actually paid for the units that were bought.
        _safe_divide(
            portfolio_overview_grouped["Buy Value"],
            portfolio_overview_grouped["Buy Volume"],
        ),
    )

    portfolio_overview_grouped["Latest Price"] = latest_returns.reindex(
        portfolio_overview_grouped.index
    )
    portfolio_overview_grouped["Latest Value"] = (
        portfolio_overview_grouped["Volume"]
        * portfolio_overview_grouped["Latest Price"]
    )

    benchmark_latest_value = (
        portfolio_overview_grouped["Benchmark Volume"] * benchmark_latest_prices
    )

    portfolio_overview_grouped["Return Value"] = (
        portfolio_overview_grouped["Latest Value"]
        + portfolio_overview_grouped["Proceeds"]
        - portfolio_overview_grouped["Invested"]
    )
    portfolio_overview_grouped["Return"] = _safe_divide(
        portfolio_overview_grouped["Return Value"],
        portfolio_overview_grouped["Invested"],
    )
    portfolio_overview_grouped["Benchmark Return"] = _safe_divide(
        benchmark_latest_value
        + portfolio_overview_grouped["Proceeds"]
        - portfolio_overview_grouped["Invested"],
        portfolio_overview_grouped["Invested"],
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
        buy_volume = totals["Buy Volume"]
        buy_value = totals["Buy Value"]
        costs = totals["Costs"]
        proceeds = totals["Proceeds"]
        invested = totals["Invested"]
        latest_value = totals["Latest Value"]
        benchmark_latest_value_sum = benchmark_latest_value.sum()

        portfolio_volatility = _calculate_portfolio_volatility(
            weights=portfolio_overview_grouped["Weight"],
            fallback_volatilities=portfolio_overview_grouped["Volatility"],
            asset_returns=asset_returns,
        )
        portfolio_beta = sum(
            portfolio_overview_grouped["Beta"] * portfolio_overview_grouped["Weight"]
        )

        return_value = latest_value + proceeds - invested

        portfolio_summary = {
            "Volume": vol,
            "Costs": costs,
            "Price": buy_value / buy_volume if buy_volume else np.nan,
            "Buy Volume": buy_volume,
            "Buy Value": buy_value,
            "Proceeds": proceeds,
            "Invested": invested,
            "Benchmark Volume": totals["Benchmark Volume"],
            "Latest Price": latest_value / vol if vol else np.nan,
            "Latest Value": latest_value,
            "Return Value": return_value,
            "Return": return_value / invested if invested > 0 else np.nan,
            "Benchmark Return": (
                (benchmark_latest_value_sum + proceeds - invested) / invested
                if invested > 0
                else np.nan
            ),
            "Volatility": portfolio_volatility,
            "Benchmark Volatility": benchmark_volatility,
            "Alpha": np.nan,
            "Beta": portfolio_beta,
            "Weight": portfolio_overview_grouped["Weight"].sum(),
        }

        portfolio_summary["Alpha"] = (
            portfolio_summary["Return"] - portfolio_summary["Benchmark Return"]
        )

        # Insert the portfolio summary row into the DataFrame
        portfolio_overview_grouped.loc["Portfolio"] = pd.Series(portfolio_summary)

    portfolio_overview_grouped = portfolio_overview_grouped[
        [
            "Volume",
            "Costs",
            "Price",
            "Invested",
            "Latest Price",
            "Latest Value",
            "Return",
            "Return Value",
            "Benchmark Return",
            "Volatility",
            "Benchmark Volatility",
            "Alpha",
            "Beta",
            "Weight",
        ]
    ]

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
            - "Volume": Net transaction volume during the period, i.e. buys minus sells.
            - "Price": The volume-weighted average price paid for the units bought during the period.
            - "Costs": Total transaction costs during the period.
            - "Invested Amount": The capital deployed during the period, i.e. the value of every buy
              plus the absolute transaction costs. Sale proceeds are not netted off.
            - "Realized Proceeds": The cash received from sales during the period.
            - "Current Value": The value of the net position at the end-of-period price.
            - "Return": The total return on the capital deployed during the period, i.e.
              (Current Value + Realized Proceeds - Invested Amount) / Invested Amount.
            - "Benchmark Return": The return the identical cash flows would have produced in the
              benchmark over the same window.
            - "Alpha": The difference between the asset return and benchmark return.

    Raises:
        ValueError: If an unsupported value is provided for `period_string`.

    Notes:
        - The function aligns transaction data with the specified period frequencies
          using the `date_column`.
        - The invested amount is the sum of volume times price per transaction rather than the summed
          volume times the average price, because the latter is only equal when every transaction in
          the period is of the same size.
        - The benchmark is dollar-matched: each transaction buys or sells benchmark units for the
          exact cash amount of that transaction on that date, which makes "Alpha" a like-for-like
          comparison rather than a comparison of equal share counts.
        - "Return" is NaN for a period in which nothing was bought, since there is no capital
          deployed in that period to earn a return on.
        - The "Benchmark Price" column is excluded from the final output as it is used only
          for intermediate calculations.
    """
    # A copy is taken because the caller's transaction dataset is reused elsewhere and must
    # not silently gain a "Benchmark Price" column.
    portfolio_dataset = portfolio_dataset.copy()
    portfolio_dataset["Benchmark Price"] = benchmark_specific_prices.to_numpy()

    period_performance = portfolio_dataset.reset_index()

    period_performance = period_performance.set_index([date_column, ticker_column])

    transaction_volume = period_performance[volume_column]
    transaction_value = transaction_volume * period_performance[price_column]

    period_performance["Buy Volume"] = transaction_volume.where(
        transaction_volume > 0, 0.0
    )
    period_performance["Buy Value"] = transaction_value.where(
        transaction_volume > 0, 0.0
    )
    period_performance["Realized Proceeds"] = (-transaction_value).where(
        transaction_volume < 0, 0.0
    )
    period_performance["Benchmark Volume"] = (
        transaction_value / period_performance["Benchmark Price"]
    )

    dates = period_performance.index.get_level_values(date_column).asfreq(period_string)

    period_performance_grouped = period_performance.groupby(
        [dates, ticker_column], observed=True
    ).agg(
        {
            volume_column: "sum",
            costs_column: "sum",
            "Buy Volume": "sum",
            "Buy Value": "sum",
            "Realized Proceeds": "sum",
            "Benchmark Volume": "sum",
        }
    )

    period_performance_grouped.insert(
        1,
        price_column,
        _safe_divide(
            period_performance_grouped["Buy Value"],
            period_performance_grouped["Buy Volume"],
        ),
    )

    period_performance_grouped["Invested Amount"] = period_performance_grouped[
        "Buy Value"
    ] + abs(period_performance_grouped[costs_column])

    last_prices = []
    last_benchmark_prices = []

    for period in period_performance_grouped.index:
        # A transaction predating the ticker's price history is unknown, not an error.
        try:
            last_prices.append(period_prices.loc[period[0], period[1]])
        except KeyError:
            last_prices.append(np.nan)
        try:
            last_benchmark_prices.append(benchmark_period_prices.loc[period[0]])
        except KeyError:
            last_benchmark_prices.append(np.nan)

    period_performance_grouped["Current Value"] = (
        period_performance_grouped[volume_column] * last_prices
    )

    # A period in which nothing was bought deployed no capital, so there is no return to
    # report on it even though the absolute costs would form a non-zero denominator.
    bought_in_period = period_performance_grouped["Buy Value"] > 0

    period_performance_grouped["Return"] = _safe_divide(
        period_performance_grouped["Current Value"]
        + period_performance_grouped["Realized Proceeds"]
        - period_performance_grouped["Invested Amount"],
        period_performance_grouped["Invested Amount"],
    ).where(bought_in_period)

    end_of_period_benchmark_value = (
        period_performance_grouped["Benchmark Volume"] * last_benchmark_prices
    )

    period_performance_grouped["Benchmark Return"] = _safe_divide(
        end_of_period_benchmark_value
        + period_performance_grouped["Realized Proceeds"]
        - period_performance_grouped["Invested Amount"],
        period_performance_grouped["Invested Amount"],
    ).where(bought_in_period)

    period_performance_grouped["Alpha"] = (
        period_performance_grouped["Return"]
        - period_performance_grouped["Benchmark Return"]
    )

    # Dropping the intermediate columns since they have no meaning in the current layout
    period_performance_grouped = period_performance_grouped.drop(
        columns=["Buy Volume", "Buy Value", "Benchmark Volume"]
    )

    return period_performance_grouped[
        [
            volume_column,
            price_column,
            costs_column,
            "Invested Amount",
            "Realized Proceeds",
            "Current Value",
            "Return",
            "Benchmark Return",
            "Alpha",
        ]
    ]


def create_positions_overview(
    portfolio_tickers: list[str],
    period_dates: pd.PeriodIndex,
    portfolio_dataset: pd.DataFrame,
    historical_prices: pd.DataFrame,
    volume_column: str = "Volume",
    price_column: str = "Price",
    costs_column: str = "Costs",
):
    """
    Calculate an overview of portfolio positions and associated performance metrics over a specified period.

    This function generates a detailed overview of the portfolio’s positions, including key statistics such as
    current value, cumulative return, invested weight, and current weight for each asset in the portfolio.
    The resulting DataFrame summarizes these metrics over the specified dates.

    Args:
        portfolio_tickers (list[str]): A list of tickers representing assets in the portfolio.
        period_dates (pd.PeriodIndex): The dates for which the position metrics should be calculated,
            i.e. the dates on which a price is available.
        portfolio_dataset (pd.DataFrame): The dataset containing portfolio transactions, indexed by
            date and ticker, with columns such as 'Volume', 'Price' and 'Costs'.
        historical_prices (pd.DataFrame): The historical prices of the assets, containing an
            'Adj Close' column used for calculating the 'Current Value' of positions.
        volume_column (str): The name of the column holding the transaction volume. Defaults to "Volume".
        price_column (str): The name of the column holding the transaction price. Defaults to "Price".
        costs_column (str): The name of the column holding the transaction costs. Defaults to "Costs".

    Returns:
        pd.DataFrame: A DataFrame containing the following metrics for each asset and date:
            - "Volume": The cumulative net volume of each asset, i.e. all buys minus all sells.
            - "Costs": The cumulative transaction costs for each asset.
            - "Invested Amount": The cumulative capital deployed, i.e. the value of every buy plus the
              absolute transaction costs. Sale proceeds are not netted off.
            - "Realized Proceeds": The cumulative cash received from selling units of each asset.
            - "Current Value": The market value of the position still held, i.e. Volume times the
              adjusted closing price on that date.
            - "Cumulative Return": The total return on capital deployed, i.e.
              (Current Value + Realized Proceeds - Invested Amount) / Invested Amount.
            - "Invested Weight": The weight of each asset relative to the total capital deployed
              across the portfolio.
            - "Current Weight": The weight of each asset relative to the total current market value
              of the portfolio.

    Notes:
        - Transactions dated on a day without a price, such as a weekend or an exchange holiday, are
          carried forward to the first following date that does have a price. Aligning them on the
          date itself would drop the transaction from every position metric without any warning.
        - Positions are marked at the adjusted closing price rather than the raw closing price, so
          "Cumulative Return" is a total return that includes reinvested dividends. As a consequence
          the historical "Current Value" is a total-return equivalent value rather than the market
          value quoted on that date; the two coincide on the most recent date.
        - "Cumulative Return" is NaN before the first purchase, since a return against zero deployed
          capital is undefined.
        - The weights are normalized based on the total invested amount and current value across all
          positions in the portfolio.
    """
    transaction_volume = portfolio_dataset[volume_column]
    transaction_value = transaction_volume * portfolio_dataset[price_column]

    transactions = pd.DataFrame(index=portfolio_dataset.index)
    transactions["Volume"] = transaction_volume
    transactions["Costs"] = portfolio_dataset[costs_column]
    transactions["Invested Amount"] = transaction_value.where(
        transaction_volume > 0, 0.0
    ) + abs(portfolio_dataset[costs_column])
    transactions["Realized Proceeds"] = (-transaction_value).where(
        transaction_volume < 0, 0.0
    )

    # A transaction booked on a day without a price, such as a weekend or an exchange
    # holiday, has no row to land on. Aligning on the date alone silently drops it from
    # every position metric, so it is carried forward to the next date that does exist.
    price_dates = pd.Index(period_dates).unique().sort_values()
    transaction_dates = transactions.index.get_level_values(0)
    date_locations = price_dates.get_indexer(transaction_dates, method="bfill")

    if (date_locations == -1).any():
        logger.warning(
            "%s transaction(s) are dated after the last available price and are therefore "
            "excluded from the positions overview.",
            int((date_locations == -1).sum()),
        )

    matched = date_locations != -1
    transactions = transactions[matched]
    transactions.index = pd.MultiIndex.from_arrays(
        [
            price_dates[date_locations[matched]],
            portfolio_dataset.index.get_level_values(1)[matched],
        ],
        names=portfolio_dataset.index.names,
    )

    columns = ["Volume", "Costs", "Invested Amount", "Realized Proceeds"]

    positions = {}
    for column in columns:
        positions[column] = pd.DataFrame(index=price_dates, columns=portfolio_tickers)

        for ticker in portfolio_tickers:
            positions[column].loc[:, ticker] = (
                transactions.xs(ticker, level=1)[column].groupby(level=0).sum()
            )

        positions[column] = positions[column].astype(float).fillna(0.0).cumsum()

    positions["Current Value"] = positions["Volume"] * historical_prices["Adj Close"]
    positions["Cumulative Return"] = _safe_divide(
        positions["Current Value"]
        + positions["Realized Proceeds"]
        - positions["Invested Amount"],
        positions["Invested Amount"],
    )
    positions["Invested Weight"] = positions["Invested Amount"].div(
        positions["Invested Amount"].sum(axis=1), axis=0
    )
    positions["Current Weight"] = positions["Current Value"].div(
        positions["Current Value"].sum(axis=1), axis=0
    )

    positions_df = pd.concat(positions, axis=1).replace([np.inf, -np.inf], np.nan)

    return positions_df.fillna(0)


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
            - 'Volume': The cumulative net volume of each asset at the end of the period.
            - 'Costs': The cumulative transaction costs at the end of the period.
            - 'Invested Amount': The cumulative capital deployed in each asset at the end of the period.
            - 'Realized Proceeds': The cumulative cash received from sales at the end of the period.
            - 'Current Value': The market value of each asset at the end of the period.
            - 'Invested Weight': The proportion of the portfolio invested in each asset.
            - 'Current Weight': The proportion of the portfolio's current value allocated to each asset.
            - 'Return': The total return on capital deployed, i.e.
              (Current Value + Realized Proceeds - Invested Amount) / Invested Amount.

    Raises:
        ValueError: If an invalid or unsupported `period_string` is provided.

    Notes:
        - Every column is the cumulative position at the end of the period, not a flow during it,
          because `positions_dataset` is itself cumulative and the aggregation takes the last
          observation of each period.
        - The function aggregates data at the specified frequency using the `.asfreq()` method.
        - Missing values in the resulting dataset are filled with zeros.
    """
    positions_dataset_stacked = positions_dataset.stack()
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
            "Realized Proceeds": "last",
            "Current Value": "last",
            "Invested Weight": "last",
            "Current Weight": "last",
        }
    )

    positions_dataset_grouped["Return"] = _safe_divide(
        positions_dataset_grouped["Current Value"]
        + positions_dataset_grouped["Realized Proceeds"]
        - positions_dataset_grouped["Invested Amount"],
        positions_dataset_grouped["Invested Amount"],
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
        pd.DataFrame: DataFrame with the columns 'PnL' (realized profit and loss per transaction)
        and 'Cumulative PnL' (cumulative realized profit and loss per ticker). The rows are returned
        in exactly the order they were supplied in.

    Notes:
        - FIFO (First-In, First-Out): Sells the earliest purchased assets first.
        - LIFO (Last-In, First-Out): Sells the most recently purchased assets first.
        - AVERAGE: Uses the average cost of all assets in inventory for PnL calculation.
        - The transactions must be supplied in chronological order, since the inventory is built up
          in the order the rows appear.
        - Transaction costs are deliberately excluded from the realized PnL. They are carried by the
          invested amount instead, so that a cost is counted once rather than in both places.
        - Selling more units than the inventory holds realizes the PnL of the units that were held
          and ignores the excess; short positions are not modelled.
        - 'Cumulative PnL' restarts at zero for every ticker rather than running across the whole
          portfolio.
    """

    def process_group(group: pd.DataFrame) -> list[float]:
        inventory: list[tuple[float, float]] = []  # For FIFO or LIFO methods
        avg_cost = 0.0  # For Average method
        total_volume = 0.0  # For Average method
        pnl: list[float] = []  # Store PnL for each row in the group

        for volume, price in zip(group[volume_column], group[price_column]):
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
                pnl.append(0.0)  # No PnL for buy orders

            elif volume < 0:  # Sell order
                sell_volume = -volume
                realized_pnl = 0.0

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
                pnl.append(0.0)  # No action for zero volume

        return pnl

    # The rows are addressed by position rather than by label. A groupby-apply returns the
    # groups concatenated in group order whenever the index is not unique, which would
    # attach every PnL figure to the wrong transaction once two trades share a date.
    ordered_transactions = transactions_overview.reset_index()

    profit_and_loss = pd.DataFrame(
        0.0, index=ordered_transactions.index, columns=["PnL", "Cumulative PnL"]
    )

    # Process each ticker group independently
    for _, group in ordered_transactions.groupby(
        ticker_column, observed=True, sort=False
    ):
        group_pnl = process_group(group)

        profit_and_loss.loc[group.index, "PnL"] = group_pnl
        # Accumulated within the group so the running total restarts per ticker.
        profit_and_loss.loc[group.index, "Cumulative PnL"] = np.cumsum(group_pnl)

    profit_and_loss.index = transactions_overview.index

    return profit_and_loss
