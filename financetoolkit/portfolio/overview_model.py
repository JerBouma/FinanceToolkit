"""Overview Model"""

import numpy as np
import pandas as pd

from financetoolkit.utilities.statistics_model import to_period_index

# pylint: disable=too-many-locals

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


def _per_transaction(values: pd.Series | None, length: int) -> np.ndarray:
    """
    Values per transaction, such as the dividends per unit since each transaction, as an array.
    A missing value is no dividend at all rather than a missing return, so it counts as zero,
    as do all of them when no values are given.
    """
    if values is None:
        return np.zeros(length)
    return np.nan_to_num(np.asarray(values, dtype=float))


def create_transactions_overview(
    portfolio_volume: pd.Series,
    portfolio_price: pd.Series,
    portfolio_costs: pd.Series,
    latest_returns: pd.Series,
    dividends: pd.Series | None = None,
):
    """
    Generate a detailed overview of transaction performance for individual assets in a portfolio.

    This function calculates key performance metrics for each asset in the portfolio, including:
    - "Invested Amount": The initial investment made for each asset, accounting for transaction costs.
    - "Current Value": The current market value of the units bought, at the latest closing price.
    - "Dividends": The cash dividends the units bought have received since the transaction.
    - "% Return": The total return of the units bought, dividends included.
    - "Return": The absolute profit or loss of the units bought, dividends included.

    Args:
        portfolio_volume (pd.Series): A Series containing the number of units (volumes) of each asset in the portfolio.
        portfolio_price (pd.Series): A Series containing the purchase price per unit for each asset.
        portfolio_costs (pd.Series): A Series containing the transaction costs associated with each asset.
        latest_returns (pd.Series): A Series containing the latest closing price per unit for each asset,
            indexed by ticker.
        dividends (pd.Series | None, optional): The cash dividends per unit paid between each transaction
            and the latest date, aligned with `portfolio_volume`. Defaults to None, which means no dividends.

    Returns:
        pd.DataFrame: A DataFrame with the following columns for each asset:
            - "Invested Amount": The total cost of acquiring the asset, including transaction costs.
            - "Current Value": The current market value of the units bought.
            - "Dividends": The cash dividends received on the units bought.
            - "% Return": The percentage return on investment, dividends included.
            - "Return": The absolute profit or loss, dividends included.

            The DataFrame is indexed by the asset tickers from the `portfolio_volume` index.
            Missing or invalid values are replaced with 0.0. Sell transactions have no current
            value, dividends or return, since their result is the realized PnL instead.
    """
    dividends_per_unit = _per_transaction(dividends, len(portfolio_volume))

    invested_amount = []
    current_value = []
    dividends_received = []
    percentage_return = []
    value_return = []

    for row, (_, ticker) in enumerate(portfolio_volume.index):
        # Transaction costs are part of what the position cost to acquire, so they are added to the invested amount rather than netted off it.  # noqa: E501
        bought_value = portfolio_volume.iloc[row] * portfolio_price.iloc[row] + abs(
            portfolio_costs.iloc[row]
        )

        if portfolio_volume.iloc[row] > 0:
            # The units are valued at the traded close and their dividends are added as cash, so the return is a total return on the same basis as the price that was paid.  # noqa: E501
            recent_value = portfolio_volume.iloc[row] * latest_returns.loc[ticker]
            received = portfolio_volume.iloc[row] * dividends_per_unit[row]
        else:
            recent_value = np.nan
            received = np.nan

        invested_amount.append(bought_value)
        current_value.append(recent_value)
        dividends_received.append(received)
        percentage_return.append(((recent_value + received) / bought_value) - 1)
        value_return.append(recent_value + received - bought_value)

    new_columns = pd.DataFrame(
        [
            invested_amount,
            current_value,
            dividends_received,
            percentage_return,
            value_return,
        ],
        columns=portfolio_volume.index,
        index=["Invested Amount", "Current Value", "Dividends", "% Return", "Return"],
    ).T

    return new_columns.fillna(0.0)


def calculate_trailing_volatility(
    returns: pd.Series | pd.DataFrame, window: int = TRADING_DAYS_PER_YEAR
) -> float | pd.Series:
    """
    Calculate the annualized volatility over the most recent trading days.

    The volatility is the standard deviation of the last `window` daily returns, multiplied by
    the square root of 252 trading days. A trailing window always holds a full year of returns,
    whereas the current calendar year holds only a handful of returns early in January.

    Args:
        returns (pd.Series | pd.DataFrame): Daily returns, one column per asset.
        window (int, optional): The number of most recent trading days to use. Defaults to 252.

    Returns:
        float | pd.Series: The annualized volatility, per column for a DataFrame.
    """
    recent_returns = returns.iloc[-window:]

    return recent_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def calculate_trailing_beta(
    asset_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    window: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """
    Calculate the beta of every asset against the benchmark over the most recent trading days,
    i.e. Cov(R_i, R_m) / Var(R_m) over the last `window` daily returns.

    The covariance and the variance are taken over the same non-missing sample per asset, since
    an asset with a shorter history would otherwise be divided by a variance measured over a
    longer window than its own covariance. The window equals the one of
    `calculate_trailing_volatility`, so the Volatility and Beta columns describe the same year.

    Args:
        asset_returns (pd.DataFrame): Daily returns, one column per asset.
        benchmark_returns (pd.Series): Daily returns of the benchmark.
        window (int, optional): The number of most recent trading days to use. Defaults to 252.

    Returns:
        pd.Series: The beta of every asset, NaN when the asset and the benchmark share no returns.
    """
    common_dates = asset_returns.index.intersection(benchmark_returns.index)
    recent_assets = asset_returns.loc[common_dates].iloc[-window:]
    recent_benchmark = benchmark_returns.loc[common_dates].iloc[-window:]

    betas = {}
    for ticker in recent_assets.columns:
        paired_returns = pd.concat(
            [recent_assets[ticker], recent_benchmark], axis=1
        ).dropna()

        if len(paired_returns) < 2:  # noqa: PLR2004
            betas[ticker] = float("nan")
            continue

        covariance = paired_returns.iloc[:, 0].cov(paired_returns.iloc[:, 1])
        variance = paired_returns.iloc[:, 1].var()
        betas[ticker] = covariance / variance if variance else float("nan")

    return pd.Series(betas, dtype=float)


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
            with one column per ticker. When available, the most recent 252 trading days of returns
            are used to build the covariance matrix, the same window as the individual asset
            volatilities (see `calculate_trailing_volatility`).

    Returns:
        float: The annualized portfolio volatility (standard deviation).

    References:
        Markowitz, H. (1952). Portfolio Selection. The Journal of Finance, 7(1), 77-91.
    """
    if asset_returns is not None and not asset_returns.empty:
        returns_for_covariance = asset_returns.reindex(columns=weights.index)

        # The most recent 252 trading days, matching the individual Volatility column.
        recent_returns = returns_for_covariance.iloc[-TRADING_DAYS_PER_YEAR:]

        # Columns that are entirely missing would propagate NaNs across the matrix.
        recent_returns = recent_returns.dropna(axis=1, how="all")

        if not recent_returns.empty:
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
    dividends: pd.Series | None = None,
    benchmark_dividends: pd.Series | None = None,
    contributed_capital: float | None = None,
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
        latest_returns (pd.Series): A Series containing the latest closing price for each asset.
        benchmark_prices (pd.Series): The closing price of the benchmark on the date of every transaction,
            aligned with `portfolio_volume`.
        benchmark_latest_prices (pd.Series): A Series containing the latest benchmark closing price for each asset.
        volatilities (pd.Series): The annualized volatility of every asset and, under "Benchmark", of the
            benchmark.
        betas (pd.Series): The beta of every asset against the benchmark.
        include_portfolio (bool): A flag indicating whether to include an aggregated portfolio summary row.
            Defaults to True.
        asset_returns (pd.DataFrame | None, optional): Daily returns for each portfolio ticker, indexed by
            date/period. When provided, the portfolio-level "Volatility" is computed from the full
            covariance matrix of these returns (Markowitz, 1952), which correctly accounts for
            diversification across imperfectly-correlated assets. When omitted, the portfolio
            volatility falls back to a weighted average of individual asset volatilities, which
            ignores cross-asset correlation and therefore overstates true portfolio volatility unless
            all assets are perfectly correlated. Defaults to None.
        dividends (pd.Series | None, optional): The cash dividends per unit of the asset paid between the
            date of every transaction and the latest date, aligned with `portfolio_volume`. Defaults to
            None, which means no dividends.
        benchmark_dividends (pd.Series | None, optional): The cash dividends per unit of the benchmark paid
            between the date of every transaction and the latest date, aligned with `portfolio_volume`.
            Defaults to None, which means no dividends.
        contributed_capital (float | None, optional): The capital the portfolio as a whole received from
            outside, as returned by `calculate_contributed_capital`. It is the denominator of the
            "Portfolio" row's "Return" and "Benchmark Return". Defaults to None, which means the
            portfolio's "Invested" is used.

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
            - "Dividends": The cash dividends received on the units held on every ex-dividend date.
            - "Return": The total return on capital deployed, i.e. Return Value / Invested. This
              includes realized and unrealized results and the dividends received.
            - "Return Value": The absolute profit or loss, i.e. Latest Value plus all sale proceeds
              plus Dividends minus Invested. This equals realized PnL + unrealized PnL + dividends -
              transaction costs.
            - "Benchmark Return": The return the identical cash flows would have produced in the
              benchmark, i.e. every buy and sell is replicated in benchmark units at the benchmark
              closing price of that transaction date, and the resulting position is valued at the
              latest benchmark closing price plus the benchmark dividends it received.
            - "Alpha": The difference between asset return and benchmark return.
            - "Weight": The proportion of the asset's current market value relative to the total
              current market value of the portfolio.

        If `include_portfolio` is True, the DataFrame includes a "Portfolio" row with aggregated metrics
        for the entire portfolio. Its "Return" and "Benchmark Return" divide by `contributed_capital`, the
        capital actually put into the portfolio, rather than by its "Invested".

    Notes:
        - The cost basis convention is total capital deployed rather than an inventory method: neither
          FIFO, LIFO nor average cost is applied here because "Return" measures the result of every
          euro put into the position, not the basis of the units that happen to remain. The inventory
          methods are used for the realized PnL in `create_profit_and_loss_overview`.
        - Positions are valued at the traded closing price, the basis of the transaction prices, and
          dividends are added as cash when they are paid rather than reinvested. Valuing them at the
          dividend-adjusted closing price instead would mark units bought at a traded price against a
          back-adjusted price and understate every return by the dividends paid after the purchase.
          The asset and the benchmark are treated identically, so "Alpha" compares like with like.
        - The decomposition Return Value = realized PnL + unrealized PnL + dividends - transaction
          costs holds exactly, where realized PnL is the FIFO/LIFO/average figure of
          `create_profit_and_loss_overview` and unrealized PnL is the remaining position marked to
          the latest price against its own basis. Transaction costs appear once, inside "Invested",
          and are deliberately not also deducted from the realized PnL.
        - "Return" and "Benchmark Return" are NaN whenever no capital was deployed (Invested <= 0),
          since a return against a zero denominator is undefined.
        - The benchmark is dollar-matched rather than share-matched: it receives exactly the cash
          flows of the position on exactly the same dates, which makes "Alpha" a like-for-like
          comparison. Matching share counts instead would weight the benchmark by the number of units
          traded, which has no meaning across assets at different price levels.
        - The portfolio "Return" is the portfolio's "Return Value" divided by the capital contributed to
          it. Within a portfolio, sale proceeds and dividends fund later purchases, so the buys of a
          rebalancing are largely paid for by the sells of the same date: dividing by "Invested", the sum
          of every buy, would count that recycled money as new capital and understate the return of a
          rebalanced portfolio more with every rebalancing. Without `contributed_capital` the
          portfolio "Return" is computed from aggregated totals, which is identical to an
          invested-amount weighted average of the individual asset returns. The "Benchmark Return"
          uses the same denominator, so the portfolio "Alpha" compares like with like.
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
    # Buys and sells are tracked separately so that sale proceeds never shrink the amount of capital that was deployed, since netting them results in a denominator that can reach zero or turn negative once a position has returned more cash than was put in, which in turn flips the sign of the reported return.  # noqa: E501
    portfolio_overview["Buy Volume"] = portfolio_volume.where(portfolio_volume > 0, 0.0)
    portfolio_overview["Buy Value"] = transaction_value.where(portfolio_volume > 0, 0.0)
    portfolio_overview["Proceeds"] = (-transaction_value).where(
        portfolio_volume < 0, 0.0
    )
    # Transaction costs are part of what the position cost to acquire, so they are added to the invested amount rather than netted off it.  # noqa: E501
    portfolio_overview["Invested"] = portfolio_overview["Buy Value"] + abs(
        portfolio_costs
    )
    # The benchmark receives the identical cash flow on the identical date, so it is matched in money rather than in share count.  # noqa: E501
    portfolio_overview["Benchmark Volume"] = (
        transaction_value / benchmark_prices.to_numpy()
    )
    # Every transaction adds or removes units that receive (or no longer receive) the dividends paid after its date, so the dividends of the position are the volume-weighted sum of the dividends per unit since each transaction.  # noqa: E501
    portfolio_overview["Dividends"] = portfolio_volume * _per_transaction(
        dividends, len(portfolio_volume)
    )
    portfolio_overview["Benchmark Dividends"] = portfolio_overview[
        "Benchmark Volume"
    ] * _per_transaction(benchmark_dividends, len(portfolio_volume))

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
            "Dividends": "sum",
            "Benchmark Dividends": "sum",
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
        + portfolio_overview_grouped["Benchmark Dividends"]
    )

    portfolio_overview_grouped["Return Value"] = (
        portfolio_overview_grouped["Latest Value"]
        + portfolio_overview_grouped["Proceeds"]
        + portfolio_overview_grouped["Dividends"]
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
        dividends_received = totals["Dividends"]
        benchmark_latest_value_sum = benchmark_latest_value.sum()

        portfolio_volatility = _calculate_portfolio_volatility(
            weights=portfolio_overview_grouped["Weight"],
            fallback_volatilities=portfolio_overview_grouped["Volatility"],
            asset_returns=asset_returns,
        )
        portfolio_beta = sum(
            portfolio_overview_grouped["Beta"] * portfolio_overview_grouped["Weight"]
        )

        return_value = latest_value + proceeds + dividends_received - invested
        # The portfolio's own capital, not the sum of its buys: the buys of a rebalancing are funded by the sells and dividends of the portfolio itself, so counting them again as capital understates its return.  # noqa: E501
        capital = (
            contributed_capital
            if contributed_capital is not None and contributed_capital > 0
            else invested
        )

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
            "Dividends": dividends_received,
            "Benchmark Dividends": totals["Benchmark Dividends"],
            "Return Value": return_value,
            "Return": return_value / capital if capital > 0 else np.nan,
            "Benchmark Return": (
                (benchmark_latest_value_sum + proceeds - invested) / capital
                if capital > 0
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
            "Dividends",
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


def calculate_contributed_capital(positions_overview: pd.DataFrame) -> float:
    """
    Calculate the capital contributed to a portfolio from outside, the denominator of its total return.

    The portfolio is treated as a cash account: sale proceeds and dividends are cash within the
    portfolio that funds later purchases. The net contribution on every date is the cumulative value of
    every buy plus the transaction costs, minus the sale proceeds and the dividends received up to that
    date, summed over all assets. Its peak is the most money that ever had to be put in: for a portfolio
    that invests 10,000 once and rebalances with its own sells and dividends, it stays 10,000 however
    many rebalancing buys follow.

    Args:
        positions_overview (pd.DataFrame): The positions overview of `create_positions_overview`, with the
            cumulative "Invested Amount", "Realized Proceeds" and "Dividends" of every asset per date.

    Returns:
        float: The peak of the cumulative net contribution. A portfolio of buys without a dividend
            between them returns its total "Invested Amount", every buy plus the costs.

    Notes:
        - A dividend received before a later buy pays for part of that buy, so it lowers the capital by
          at most the dividend. Dividends after the last buy leave it unchanged.
        - The figure moves continuously with the size of the sales: a small sale lowers it by at most its
          proceeds, so a portfolio of buys alone and the same portfolio with a tiny sale report nearly the
          same return.
        - The peak rather than the latest value is used, because money taken out later does not undo
          the capital that was at work before it.
    """
    if positions_overview.empty:
        return float("nan")

    invested = positions_overview["Invested Amount"].sum(axis=1)
    proceeds = positions_overview["Realized Proceeds"].sum(axis=1)
    dividends = (
        positions_overview["Dividends"].sum(axis=1)
        if "Dividends" in positions_overview.columns.get_level_values(0)
        else 0.0
    )

    return float((invested - proceeds - dividends).cummax().iloc[-1])


def _lookup_period_values(
    period_values: pd.DataFrame | pd.Series | None,
    periods: pd.PeriodIndex,
    tickers: pd.Index,
) -> np.ndarray:
    """
    The value of every (period, ticker) pair in a frame with one row per period and one column
    per ticker, or in a Series with one value per period. A pair that is not available, such as a
    transaction that predates the ticker's price history, is NaN rather than an error.
    """
    values = np.full(len(periods), np.nan)

    if period_values is None:
        return values

    for row, (period, ticker) in enumerate(zip(periods, tickers, strict=True)):
        try:
            value = (
                period_values.loc[period, ticker]
                if isinstance(period_values, pd.DataFrame)
                else period_values.loc[period]
            )
        except KeyError:
            continue
        values[row] = float(value)

    return values


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
    benchmark_period_prices: pd.DataFrame | pd.Series,
    period_dividends: pd.DataFrame | None = None,
    transaction_dividends: pd.Series | None = None,
    benchmark_period_dividends: pd.DataFrame | pd.Series | None = None,
    benchmark_transaction_dividends: pd.Series | None = None,
):
    """
    Calculate the performance of the units bought in every period up to the end of that period.

    This function computes per asset and period the capital deployed in the units bought in that
    period, their value and dividends at the end of the period and the return on them, next to
    the return the identical purchases would have made in the benchmark.

    Args:
        portfolio_dataset (pd.DataFrame): A DataFrame containing portfolio transaction details,
            including asset tickers, transaction dates, volumes, prices, and costs.
        ticker_column (str): The name of the column representing asset tickers in `portfolio_dataset`.
        date_column (str): The name of the column representing transaction dates in `portfolio_dataset`.
        volume_column (str): The name of the column representing transaction volumes in `portfolio_dataset`.
        price_column (str): The name of the column representing transaction prices in `portfolio_dataset`.
        costs_column (str): The name of the column representing transaction costs in `portfolio_dataset`.
        period_prices (pd.DataFrame): The closing price of every asset at the end of every period, one
            row per period and one column per ticker.
        period_string (str): The time period for grouping transactions. Supported values are
            'Y', 'Q', 'M', 'W' or 'D'.
        benchmark_specific_prices (pd.Series): The closing price of the benchmark on the date of every
            transaction, aligned with `portfolio_dataset`.
        benchmark_period_prices (pd.DataFrame | pd.Series): The closing price of the benchmark at the end
            of every period, as a Series or, when the benchmark is expressed in the currency of each
            ticker, as a DataFrame with one column per ticker.
        period_dividends (pd.DataFrame | None, optional): The cumulative dividends per unit of every asset
            at the end of every period, shaped as `period_prices`. Defaults to None, which means no
            dividends.
        transaction_dividends (pd.Series | None, optional): The cumulative dividends per unit of the asset
            up to and including the date of every transaction, aligned with `portfolio_dataset`.
            Defaults to None, which means no dividends.
        benchmark_period_dividends (pd.DataFrame | pd.Series | None, optional): The cumulative dividends
            per unit of the benchmark at the end of every period, shaped as `benchmark_period_prices`.
            Defaults to None, which means no dividends.
        benchmark_transaction_dividends (pd.Series | None, optional): The cumulative dividends per unit
            of the benchmark up to and including the date of every transaction, aligned with
            `portfolio_dataset`. Defaults to None, which means no dividends.

    Returns:
        pd.DataFrame: A DataFrame containing the following metrics for each asset and period:
            - "Volume": Net transaction volume during the period, i.e. buys minus sells.
            - "Price": The volume-weighted average price paid for the units bought during the period.
            - "Costs": Total transaction costs during the period, of buys and sells.
            - "Invested Amount": The capital deployed during the period, i.e. the value of every buy
              plus the absolute transaction costs of those buys.
            - "Realized Proceeds": The cash received from sales during the period.
            - "Current Value": The value of the units bought during the period at the closing price
              at the end of the period.
            - "Dividends": The cash dividends those units received up to the end of the period.
            - "Return": The total return on the units bought during the period, i.e.
              (Current Value + Dividends - Invested Amount) / Invested Amount.
            - "Benchmark Return": The return the identical purchases would have produced in the
              benchmark over the same window, dividends included.
            - "Alpha": The difference between the asset return and benchmark return.

    Raises:
        ValueError: If an unsupported value is provided for `period_string`.

    Notes:
        - "Return" follows the units bought in the period only. Sales are reported in "Realized
          Proceeds" but left out of the return, because the units sold in a period were usually
          bought in an earlier one; their proceeds would credit the purchases of this period with
          the gains of earlier ones.
        - The costs of sales are reported in "Costs" but not in "Invested Amount", since they are no
          part of what the units bought in the period cost.
        - Units are valued at the traded closing price, the basis of the transaction prices, and the
          dividends paid on them are added as cash. A unit bought on an ex-dividend date does not
          receive that dividend.
        - The invested amount is the sum of volume times price per transaction rather than the summed
          volume times the average price, because the latter is only equal when every transaction in
          the period is of the same size.
        - The benchmark is dollar-matched: each purchase buys benchmark units for the exact cash
          amount of that transaction on that date, which makes "Alpha" a like-for-like comparison
          rather than a comparison of equal share counts.
        - "Return" is NaN for a period in which nothing was bought, since there is no capital
          deployed in that period to earn a return on.
    """
    period_performance = portfolio_dataset.reset_index().set_index(
        [date_column, ticker_column]
    )

    transaction_volume = period_performance[volume_column]
    transaction_value = transaction_volume * period_performance[price_column]
    bought = transaction_volume > 0

    dates = to_period_index(
        period_performance.index.get_level_values(date_column)
    ).asfreq(period_string)
    tickers = period_performance.index.get_level_values(ticker_column)

    end_prices = _lookup_period_values(period_prices, dates, tickers)
    end_dividends = np.nan_to_num(
        _lookup_period_values(period_dividends, dates, tickers)
    )
    benchmark_end_prices = _lookup_period_values(
        benchmark_period_prices, dates, tickers
    )
    benchmark_end_dividends = np.nan_to_num(
        _lookup_period_values(benchmark_period_dividends, dates, tickers)
    )

    buy_volume = transaction_volume.where(bought, 0.0)
    buy_value = transaction_value.where(bought, 0.0)
    benchmark_volume = buy_value / np.asarray(benchmark_specific_prices, dtype=float)

    period_performance["Buy Volume"] = buy_volume
    period_performance["Buy Value"] = buy_value
    period_performance["Buy Costs"] = (
        period_performance[costs_column].abs().where(bought, 0.0)
    )
    period_performance["Realized Proceeds"] = (-transaction_value).where(
        transaction_volume < 0, 0.0
    )
    # Sales carry no value here, so a missing end-of-period price on a sale leaves the purchases intact.
    period_performance["Current Value"] = (buy_volume * end_prices).where(bought, 0.0)
    # A unit receives every dividend with an ex-date after the day it was bought, up to the end of the period.  # noqa: E501
    period_performance["Dividends"] = buy_volume * (
        end_dividends - _per_transaction(transaction_dividends, len(period_performance))
    )
    period_performance["Benchmark Value"] = (
        benchmark_volume
        * (
            benchmark_end_prices
            + benchmark_end_dividends
            - _per_transaction(benchmark_transaction_dividends, len(period_performance))
        )
    ).where(bought, 0.0)

    period_performance_grouped = period_performance.groupby(
        [dates, ticker_column], observed=True
    ).agg(
        {
            volume_column: "sum",
            costs_column: "sum",
            "Buy Volume": "sum",
            "Buy Value": "sum",
            "Buy Costs": "sum",
            "Realized Proceeds": "sum",
            "Current Value": lambda values: values.sum(min_count=1),
            "Dividends": "sum",
            "Benchmark Value": lambda values: values.sum(min_count=1),
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

    period_performance_grouped["Invested Amount"] = (
        period_performance_grouped["Buy Value"]
        + period_performance_grouped["Buy Costs"]
    )

    # A period in which nothing was bought deployed no capital, so there is no return to report on it.
    bought_in_period = period_performance_grouped["Buy Value"] > 0

    period_performance_grouped["Return"] = _safe_divide(
        period_performance_grouped["Current Value"]
        + period_performance_grouped["Dividends"]
        - period_performance_grouped["Invested Amount"],
        period_performance_grouped["Invested Amount"],
    ).where(bought_in_period)

    period_performance_grouped["Benchmark Return"] = _safe_divide(
        period_performance_grouped["Benchmark Value"]
        - period_performance_grouped["Invested Amount"],
        period_performance_grouped["Invested Amount"],
    ).where(bought_in_period)

    period_performance_grouped["Alpha"] = (
        period_performance_grouped["Return"]
        - period_performance_grouped["Benchmark Return"]
    )

    return period_performance_grouped[
        [
            volume_column,
            price_column,
            costs_column,
            "Invested Amount",
            "Realized Proceeds",
            "Current Value",
            "Dividends",
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
        historical_prices (pd.DataFrame): The daily historical data of the assets, containing a
            'Close' column used for the 'Current Value' of positions and, optionally, a 'Dividends'
            column with the cash dividend per unit on every ex-dividend date.
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
            - "Dividends": The cumulative cash dividends received, i.e. on every ex-dividend date the
              units held at the previous close times the dividend per unit.
            - "Current Value": The market value of the position still held, i.e. Volume times the
              closing price on that date.
            - "Cumulative Return": The total return on capital deployed, i.e.
              (Current Value + Realized Proceeds + Dividends - Invested Amount) / Invested Amount.
            - "Invested Weight": The weight of each asset relative to the total capital deployed
              across the portfolio.
            - "Current Weight": The weight of each asset relative to the total current market value
              of the portfolio.

    Notes:
        - Transactions dated on a day without a price, such as a weekend or an exchange holiday, are
          carried forward to the first following date that does have a price. Aligning them on the
          date itself would drop the transaction from every position metric without any warning.
          A transaction after the last date with a price, such as one of today before today's close
          is available, is placed on that last date, the date of the closing price it is priced at.
        - Positions are marked at the traded closing price, the basis of the transaction prices, and
          dividends are added as cash when they are paid, so "Cumulative Return" is a total return
          and "Current Value" is the market value quoted on that date. Marking units bought at a
          traded price at the dividend-adjusted closing price instead would understate the return by
          every dividend paid after the purchase.
        - A missing closing price, such as on a holiday of the asset's exchange, is carried forward
          from the previous trading day, so the position keeps its value rather than dropping to zero.
        - "Cumulative Return" is NaN before the first purchase, since a return against zero deployed
          capital is undefined. Every other column is zero before the first purchase.
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

    # A transaction booked on a day without a price, such as a weekend or an exchange holiday, has no row to land on, and aligning on the date alone silently drops it from every position metric, so it is carried forward to the next date that does exist. One after the last date lands on the last date, the close it is priced at.  # noqa: E501
    price_dates = pd.Index(period_dates).unique().sort_values()
    transaction_dates = transactions.index.get_level_values(0)
    date_locations = price_dates.get_indexer(transaction_dates, method="bfill")
    date_locations[date_locations == -1] = len(price_dates) - 1

    transactions.index = pd.MultiIndex.from_arrays(
        [
            price_dates[date_locations],
            portfolio_dataset.index.get_level_values(1),
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

    closing_prices = (
        historical_prices["Close"]
        .reindex(index=price_dates, columns=portfolio_tickers)
        .astype(float)
        .ffill()
    )

    if "Dividends" in historical_prices.columns.get_level_values(0):
        dividends_per_unit = (
            historical_prices["Dividends"]
            .reindex(index=price_dates, columns=portfolio_tickers)
            .astype(float)
            .fillna(0.0)
        )
    else:
        dividends_per_unit = pd.DataFrame(
            0.0, index=price_dates, columns=portfolio_tickers
        )

    # A dividend goes to whoever held the units at the close before the ex-dividend date, so a purchase on the ex-date itself does not receive it and a sale on that date still does.  # noqa: E501
    positions["Dividends"] = (
        (positions["Volume"].shift(1).fillna(0.0) * dividends_per_unit)
        .fillna(0.0)
        .cumsum()
    )
    positions["Current Value"] = positions["Volume"] * closing_prices
    positions["Cumulative Return"] = _safe_divide(
        positions["Current Value"]
        + positions["Realized Proceeds"]
        + positions["Dividends"]
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

    # The return stays NaN before the first purchase, since there is no capital to measure it against.
    other_columns = positions_df.columns.get_level_values(0) != "Cumulative Return"
    positions_df.loc[:, other_columns] = positions_df.loc[:, other_columns].fillna(0.0)

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
            indices (date and ticker). It includes columns like 'Volume', 'Costs', 'Invested Amount',
            'Realized Proceeds', 'Dividends', 'Current Value', 'Invested Weight', and 'Current Weight'.
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
            - 'Dividends': The cumulative cash dividends received at the end of the period.
            - 'Current Value': The market value of each asset at the end of the period.
            - 'Invested Weight': The proportion of the portfolio invested in each asset.
            - 'Current Weight': The proportion of the portfolio's current value allocated to each asset.
            - 'Return': The total return on capital deployed, i.e.
              (Current Value + Realized Proceeds + Dividends - Invested Amount) / Invested Amount.

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

    columns = [
        column
        for column in [
            "Volume",
            "Costs",
            "Invested Amount",
            "Realized Proceeds",
            "Dividends",
            "Current Value",
            "Invested Weight",
            "Current Weight",
        ]
        if column in positions_dataset_stacked.columns
    ]

    positions_dataset_grouped = positions_dataset_stacked.groupby(
        [dates, tickers], observed=True
    ).agg(dict.fromkeys(columns, "last"))

    dividends_received = (
        positions_dataset_grouped["Dividends"]
        if "Dividends" in positions_dataset_grouped.columns
        else 0.0
    )

    positions_dataset_grouped["Return"] = _safe_divide(
        positions_dataset_grouped["Current Value"]
        + positions_dataset_grouped["Realized Proceeds"]
        + dividends_received
        - positions_dataset_grouped["Invested Amount"],
        positions_dataset_grouped["Invested Amount"],
    )

    return positions_dataset_grouped.fillna(0.0)


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

    # The rows are addressed by position rather than by label, since a groupby-apply returns the groups concatenated in group order whenever the index is not unique, which would attach every PnL figure to the wrong transaction once two trades share a date.  # noqa: E501
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
