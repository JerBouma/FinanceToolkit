"""Overview Model"""

import numpy as np
import pandas as pd

from financetoolkit.portfolio.helpers import (
    divide_by_capital,
    get_trading_day_locations,
    get_transaction_order,
)
from financetoolkit.utilities import logger_model
from financetoolkit.utilities.error_model import check_option
from financetoolkit.utilities.statistics_model import drop_non_finite, to_period_index

logger = logger_model.get_logger()


# Trading days used to annualize, matching the Risk module's convention.
TRADING_DAYS_PER_YEAR = 252
# Units left to sell below this are floating point residue of the volumes, not an oversold position.
OVERSOLD_TOLERANCE = 1e-9
# Units left in a lot below this times the largest volume are the floating point residue of subtracting volumes
# (0.1 + 0.2 - 0.3 leaves 5.6e-17), which would otherwise stay open as a lot of next to nothing.
LOT_RESIDUE_TOLERANCE = 1e-12


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
    - "% Return": The total return of the units bought, dividends included. NaN for units
      bought at no cost (a gift, spin-off or stock dividend), which have no return of their own.
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
            value, dividends or return, since their result is the realized PnL instead. Units bought
            of an asset without a latest price are worth an unknown amount, not nothing, so their
            "Current Value", "% Return" and "Return" are NaN.
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
        # A buy that cost nothing (a gift, spin-off or stock dividend) has no return of its own.
        percentage_return.append(
            ((recent_value + received) / bought_value) - 1
            if bought_value > 0
            else np.nan
        )
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

    buys = (portfolio_volume > 0).to_numpy()
    zero_cost_buys = buys & (new_columns["Invested Amount"] <= 0).to_numpy()
    # Units bought without a latest price are worth an unknown amount, not nothing, so their value and return stay NaN.  # noqa: E501
    unpriced_buys = buys & new_columns["Current Value"].isna().to_numpy()
    new_columns = new_columns.fillna(0.0)
    new_columns.loc[unpriced_buys, ["Current Value", "% Return", "Return"]] = np.nan
    new_columns.loc[zero_cost_buys, "% Return"] = np.nan

    return new_columns


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
            volatilities (see `calculate_trailing_volatility`). The covariance matrix is estimated
            on the dates of that window on which every held asset has a price: from the date the
            last of them has its first return, with a return of 0 on a date an asset did not trade
            while another did (its price is carried forward, as in the value of the portfolio).
            Each pair estimated on the dates it shares and each variance on the asset's own dates
            would mix windows into a matrix that need not be a covariance matrix of any one set of
            returns, and can make the volatility of two opposite assets much larger than it was.

    Returns:
        float: The annualized portfolio volatility (standard deviation).

    References:
        Markowitz, H. (1952). Portfolio Selection. The Journal of Finance, 7(1), 77-91.
    """
    if asset_returns is not None and not asset_returns.empty:
        # Only the held assets count: a sold position has no returns in the window to restrict it by.  # noqa: E501
        held = weights.index[weights.notna() & (weights != 0)]
        returns_for_covariance = asset_returns.reindex(columns=held)

        # The most recent 252 trading days, matching the individual Volatility column.
        recent_returns = returns_for_covariance.iloc[-TRADING_DAYS_PER_YEAR:]

        # Columns that are entirely missing would propagate NaNs across the matrix.
        recent_returns = recent_returns.dropna(axis=1, how="all")

        # One set of dates for every pair: from the first date every held asset has a return, with 0 on the dates an asset
        # did not trade.
        recent_returns = recent_returns[
            recent_returns.notna().cummax().all(axis=1)
        ].fillna(0.0)

        if len(recent_returns) > 1:
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


def is_open_position(volume: pd.Series, largest_volume: pd.Series) -> pd.Series:
    """
    Determine which positions are still open, long or short, from their net volume.

    A position whose sales add up to its purchases is closed, but the sum of fractional volumes
    rarely lands on exactly zero: 1.1 + 2.2 - 3.3 leaves 4.4e-16 units. Such a remainder is
    floating point residue that grows with the volumes traded, so a position counts as open
    only when its net volume exceeds `OVERSOLD_TOLERANCE` times the largest volume traded in
    the asset (at least one unit), the scale the profit and loss overview uses for the same
    residue. A negative net volume beyond that is a short position, which is open as well.

    Args:
        volume (pd.Series): The net volume of every position, indexed by identifier or by
            period and identifier, with the identifier as the last index level.
        largest_volume (pd.Series): The largest absolute volume of a single transaction in
            every identifier, indexed by identifier.

    Returns:
        pd.Series: True for every open position, aligned with `volume`.
    """
    scale = np.maximum(
        1.0,
        largest_volume.reindex(volume.index.get_level_values(-1))
        .fillna(0.0)
        .to_numpy(dtype=float),
    )

    return pd.Series(
        volume.abs().to_numpy(dtype=float) > OVERSOLD_TOLERANCE * scale,
        index=volume.index,
    )


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
        - Units held of an asset without a latest price, such as a ticker without data, are worth an
          unknown amount rather than nothing. The value of the whole portfolio is then unknown, so every
          "Weight" and the "Portfolio" row's "Latest Price", "Latest Value", "Return", "Return Value",
          "Alpha", "Volatility", "Beta" and "Weight" are NaN, while the sums of known amounts
          ("Volume", "Costs", "Invested", "Dividends") and the "Benchmark Return" remain. A closed
          position is worth nothing, priced or not.

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
        divide_by_capital(
            portfolio_overview_grouped["Buy Value"],
            portfolio_overview_grouped["Buy Volume"],
        ),
    )

    portfolio_overview_grouped["Latest Price"] = latest_returns.reindex(
        portfolio_overview_grouped.index
    )
    # Only the open positions count, long or short: a sold position has no beta over the window and a weight of zero, also when float residue of its volumes leaves a sliver of a unit.  # noqa: E501
    held = is_open_position(
        portfolio_overview_grouped["Volume"],
        portfolio_volume.abs().groupby(level="Identifier", observed=True).max(),
    )
    # Units held without a price are worth an unknown amount, not nothing; a closed position is worth nothing, priced or not.  # noqa: E501
    portfolio_overview_grouped["Latest Value"] = (
        portfolio_overview_grouped["Volume"]
        * portfolio_overview_grouped["Latest Price"]
    ).where(held | portfolio_overview_grouped["Latest Price"].notna(), 0.0)
    unpriced_holding = bool(
        (held & portfolio_overview_grouped["Latest Value"].isna()).any()
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
    portfolio_overview_grouped["Return"] = divide_by_capital(
        portfolio_overview_grouped["Return Value"],
        portfolio_overview_grouped["Invested"],
    )
    portfolio_overview_grouped["Benchmark Return"] = divide_by_capital(
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
    # The value of the whole portfolio is unknown as long as one of its holdings has no price, and so is the share of every position in it.  # noqa: E501
    total_latest_value = (
        np.nan if unpriced_holding else portfolio_overview_grouped["Latest Value"].sum()
    )
    portfolio_overview_grouped["Weight"] = (
        portfolio_overview_grouped["Latest Value"] / total_latest_value
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
        latest_value = total_latest_value
        dividends_received = totals["Dividends"]
        benchmark_latest_value_sum = benchmark_latest_value.sum()

        # Without the weights of the holdings neither the volatility nor the beta of the portfolio can be combined from those of its assets.  # noqa: E501
        portfolio_volatility = (
            np.nan
            if unpriced_holding
            else _calculate_portfolio_volatility(
                weights=portfolio_overview_grouped["Weight"],
                fallback_volatilities=portfolio_overview_grouped["Volatility"],
                asset_returns=asset_returns,
            )
        )
        # A sold position's missing beta would otherwise turn the portfolio's beta into NaN.
        portfolio_beta = (
            (
                portfolio_overview_grouped.loc[held, "Beta"]
                * portfolio_overview_grouped.loc[held, "Weight"]
            ).sum(skipna=False)
            if held.any()
            else np.nan
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
            "Weight": (
                np.nan
                if unpriced_holding
                else portfolio_overview_grouped["Weight"].sum()
            ),
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
        period_string (str): The time period for grouping transactions. Options are: "Y" (yearly),
            "Q" (quarterly), "M" (monthly), "W" (weekly), "D" (daily).
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
            # One purchase without a price leaves the value of the period's purchases unknown, not smaller.
            "Current Value": lambda values: values.sum(skipna=False),
            "Dividends": "sum",
            "Benchmark Value": lambda values: values.sum(skipna=False),
        }
    )

    period_performance_grouped.insert(
        1,
        price_column,
        divide_by_capital(
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

    period_performance_grouped["Return"] = divide_by_capital(
        period_performance_grouped["Current Value"]
        + period_performance_grouped["Dividends"]
        - period_performance_grouped["Invested Amount"],
        period_performance_grouped["Invested Amount"],
    ).where(bought_in_period)

    period_performance_grouped["Benchmark Return"] = divide_by_capital(
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
        - Transactions dated on a day their asset did not trade, such as a weekend or an exchange
          holiday, are carried forward to the first following date the asset traded on, also when
          another asset (such as a cryptocurrency) has a price on that day. Aligning them on the
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
          positions in the portfolio. On a date a held position has no price the current value of the
          portfolio is unknown, so the "Current Weight" of every position is NaN on that date.
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

    # A transaction booked on a day without a price, such as a weekend or an exchange holiday, has no row to land on, and aligning on the date alone silently drops it from every position metric, so it is carried forward to the next date its ticker traded on: a date only another ticker traded (a Saturday next to a cryptocurrency) holds the ticker's carried-forward close, not the close it is priced at. One after the last date lands on the last date, the close it is priced at.  # noqa: E501
    price_dates = pd.Index(period_dates).unique().sort_values()
    date_locations = get_trading_day_locations(
        historical_prices.reindex(price_dates),
        transactions.index.get_level_values(0),
        portfolio_dataset.index.get_level_values(1),
    )

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
    # Units held before their ticker has a price are worth an unknown amount, not nothing; without units the position is worth nothing, priced or not.  # noqa: E501
    positions["Current Value"] = (positions["Volume"] * closing_prices).where(
        positions["Volume"] != 0, 0.0
    )
    positions["Cumulative Return"] = divide_by_capital(
        positions["Current Value"]
        + positions["Realized Proceeds"]
        + positions["Dividends"]
        - positions["Invested Amount"],
        positions["Invested Amount"],
    )
    positions["Invested Weight"] = positions["Invested Amount"].div(
        positions["Invested Amount"].sum(axis=1), axis=0
    )
    # On a date a holding has no price the value of the portfolio is unknown, and so is the weight of every position.  # noqa: E501
    total_current_value = positions["Current Value"].sum(axis=1, skipna=False)
    positions["Current Weight"] = positions["Current Value"].div(
        total_current_value, axis=0
    )

    positions_df = drop_non_finite(pd.concat(positions, axis=1))

    # The return stays NaN before the first purchase, since there is no capital to measure it against, and so do the value and return of units without a price yet.  # noqa: E501
    other_columns = ~positions_df.columns.get_level_values(0).isin(
        ["Cumulative Return", "Current Value"]
    )
    positions_df.loc[:, other_columns] = positions_df.loc[:, other_columns].fillna(0.0)
    positions_df.loc[total_current_value.isna(), "Current Weight"] = np.nan

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
        period_string (str): The time period for which portfolio performance metrics should be
            calculated. Options are: "Y" (yearly), "Q" (quarterly), "M" (monthly), "W" (weekly), "D"
            (daily).

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
        - The Return is NaN in a period without capital invested, and the Current Value and
          Return are NaN for units held before their ticker has a price. The Current Weight of
          every position is NaN in a period in which a held position has no price, since the value
          of the portfolio is then unknown. Other missing values are zero.
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

    positions_dataset_grouped["Return"] = divide_by_capital(
        positions_dataset_grouped["Current Value"]
        + positions_dataset_grouped["Realized Proceeds"]
        + dividends_received
        - positions_dataset_grouped["Invested Amount"],
        positions_dataset_grouped["Invested Amount"],
    )

    # The return is NaN without capital invested or without a price for the units held, and so is the value of those units.  # noqa: E501
    other_columns = ~positions_dataset_grouped.columns.isin(
        ["Return", "Current Value", "Current Weight"]
    )
    positions_dataset_grouped.loc[:, other_columns] = positions_dataset_grouped.loc[
        :, other_columns
    ].fillna(0.0)

    return positions_dataset_grouped


def match_lots(
    volumes: np.ndarray, method: str = "FIFO", short_selling: bool = False
) -> tuple[list[tuple[int, int, float]], list[tuple[int, float]], bool]:
    """
    Match the transactions of one asset, in the order they are given, against the lots they
    close.

    Every purchase or sale opens a lot, and a transaction in the other direction closes the
    oldest open lots first (FIFO) or the newest (LIFO). A transaction larger than the lots
    it closes opens a lot in its own direction with the units left, so a sale of more than
    is held opens a short lot (when short selling is on) that later purchases close.

    The matches are the round trips of the asset: each pairs the transaction that opened
    units with the one that closed them, so the realized profit and loss of a sale and the
    trade list both follow from them.

    Args:
        volumes (np.ndarray): The volume of every transaction, positive for a purchase and
            negative for a sale, in the order they are matched in.
        method (str, optional): Which lots a sale closes first. Options are: "FIFO" (first in, first
            out: the oldest lots close first), "LIFO" (last in, first out: the newest lots close
            first). Defaults to "FIFO".
        short_selling (bool, optional): Whether a sale of more units than are held opens a
            short lot. Defaults to False, which leaves the units sold beyond those held out.

    Returns:
        tuple[list[tuple[int, int, float]], list[tuple[int, float]], bool]: The matches as
        (opening row, closing row, units), the lots still open as (opening row, units), with
        negative units for a short lot, and whether units were sold beyond those held while
        short selling is off.

    Notes:
        - Units left by floating point residue of the volumes (0.1 + 0.2 - 0.3), on the scale
          of the largest volume traded, neither stay open as a lot nor count as an excess.
    """
    finite = np.abs(volumes[np.isfinite(volumes)])
    scale = max(1.0, finite.max() if finite.size else 0.0)
    lots: list[list] = []  # [opening row, units] of the open lots, all in one direction
    side = 0
    matches: list[tuple[int, int, float]] = []
    oversold = False

    for row, volume in enumerate(volumes):
        if not (volume > 0 or volume < 0):  # no units, or a missing volume
            continue
        direction = 1 if volume > 0 else -1
        remaining = abs(volume)

        if lots and direction != side:
            while remaining > 0 and lots:
                position = 0 if method == "FIFO" else -1
                opening, units = lots[position]
                closed = min(units, remaining)
                matches.append((opening, row, closed))
                remaining -= closed
                if units - closed > LOT_RESIDUE_TOLERANCE * scale:
                    lots[position][1] = units - closed
                else:
                    lots.pop(position)
            # Units left beyond the lots on the scale of the oversold tolerance are residue, not a sale.
            if remaining <= OVERSOLD_TOLERANCE * scale:
                remaining = 0

        if remaining > 0:
            if direction > 0 or short_selling:
                lots.append([row, remaining])
                side = direction
            else:
                oversold = True

    return matches, [(opening, side * units) for opening, units in lots], oversold


def create_profit_and_loss_overview(
    transactions_overview: pd.DataFrame,
    ticker_column: str,
    volume_column: str,
    price_column: str,
    method="FIFO",
    purchases_first: bool = True,
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
        method (str, optional): Method for inventory valuation. Options are: "FIFO" (first in, first
            out: the oldest lots close first), "LIFO" (last in, first out: the newest lots close
            first), "AVERAGE" (every sale at the average cost of the units held). Defaults to
            "FIFO".
        purchases_first (bool, optional): Whether the purchases of a day come before its sales.
            False keeps the order of the rows within a day, for transactions that are in the
            order they were made in (dated with a time of day). Defaults to True.

    Returns:
        pd.DataFrame: DataFrame with the columns 'PnL' (realized profit and loss per transaction)
        and 'Cumulative PnL' (cumulative realized profit and loss per ticker). The rows are returned
        in exactly the order they were supplied in.

    Notes:
        - FIFO (First-In, First-Out): Sells the earliest purchased assets first.
        - LIFO (Last-In, First-Out): Sells the most recently purchased assets first.
        - AVERAGE: Uses the average cost of all assets in inventory for PnL calculation.
        - The inventory is built up in date order. Within a day the purchases come before
          the sales, so a position bought and sold on the same day realizes its PnL whatever
          the order of the rows; otherwise the rows keep their order. When the purchases do
          not come first, as the Portfolio passes for transactions dated with a time of day,
          the rows keep their order within the day as well, so a sale in the morning is
          matched against the units held before a purchase in the afternoon.
        - Transaction costs are deliberately excluded from the realized PnL. They are carried by the
          invested amount instead, so that a cost is counted once rather than in both places.
        - Selling more units than the inventory holds realizes the PnL of the units that were held
          and ignores the excess, with a warning; short positions are not modelled.
        - 'Cumulative PnL' restarts at zero for every ticker rather than running across the whole
          portfolio.
    """

    def process_group(group: pd.DataFrame) -> tuple[np.ndarray, bool]:
        volumes = group[volume_column].to_numpy(dtype=float)
        prices = group[price_column].to_numpy(dtype=float)
        pnl = np.zeros(len(volumes))

        if method != "AVERAGE":
            matches, _, group_oversold = match_lots(volumes, method=method)
            # Every closed lot realizes its units times the difference between the sale and purchase price.
            for opening, closing, units in matches:
                pnl[closing] += units * (prices[closing] - prices[opening])
            return pnl, group_oversold

        avg_cost = 0.0
        total_volume = 0.0
        group_oversold = False
        tolerance = OVERSOLD_TOLERANCE * max(1.0, np.nanmax(np.abs(volumes)))
        for row, (volume, price) in enumerate(zip(volumes, prices, strict=True)):
            if volume > 0:
                total_cost = avg_cost * total_volume + volume * price
                total_volume += volume
                avg_cost = total_cost / total_volume
            elif volume < 0:
                sell_volume = min(-volume, total_volume)
                pnl[row] = sell_volume * (price - avg_cost)
                total_volume -= sell_volume
                # Units sold beyond those held have no purchase to realize a PnL against.
                if -volume - sell_volume > tolerance:
                    group_oversold = True

        return pnl, group_oversold

    # The rows are addressed by position rather than by label, since a groupby-apply returns the groups concatenated in group order whenever the index is not unique, which would attach every PnL figure to the wrong transaction once two trades share a date.  # noqa: E501
    ordered_transactions = transactions_overview.reset_index()

    ordered_transactions = ordered_transactions.iloc[
        get_transaction_order(
            transactions_overview.index.get_level_values(0),
            ordered_transactions[volume_column],
            purchases_first,
        )
    ]

    profit_and_loss = pd.DataFrame(
        0.0, index=ordered_transactions.index, columns=["PnL", "Cumulative PnL"]
    )
    oversold: set[str] = set()

    # Process each ticker group independently
    for ticker, group in ordered_transactions.groupby(
        ticker_column, observed=True, sort=False
    ):
        group_pnl, group_oversold = process_group(group)
        if group_oversold:
            oversold.add(str(ticker))

        profit_and_loss.loc[group.index, "PnL"] = group_pnl
        # Accumulated within the group so the running total restarts per ticker.
        profit_and_loss.loc[group.index, "Cumulative PnL"] = np.cumsum(group_pnl)

    if oversold:
        logger.warning(
            "More units are sold than held for %s. The excess has no purchase to realize a "
            "profit or loss against and is left out; short positions are not modelled.",
            ", ".join(sorted(oversold)),
        )

    profit_and_loss = profit_and_loss.sort_index()
    profit_and_loss.index = transactions_overview.index

    return profit_and_loss


TURNOVER_DENOMINATORS = ("gross_value", "net_value")


def get_turnover(
    traded_values: pd.Series,
    position_values: pd.DataFrame,
    denominator: str = "gross_value",
) -> pd.DataFrame:
    """
    The turnover of a portfolio per day, the value traded as a share of the value held, as
    defined in pyfolio (Quantopian) by get_turnover:

    - Traded Value_t = Σ |volume × price| of the day's transactions
    - Gross Value_t = Σ_i |value of position i| (pyfolio's "actual gross book")
    - Turnover_t = Traded Value_t / ((Gross Value_t + Gross Value_t-1) / 2), the average of
      the gross value of the day and the day before (half the gross value on the first
      day), or Traded Value_t / Net Value_t over the net value (pyfolio's
      "portfolio_value"), Net Value_t = Σ_i value of position i.

    A purchase of a whole portfolio in one day is a turnover of 2 on that day, as the
    gross value averages to half its size.

    Shares similar logic:
    - pyfolio, get_turnover: https://github.com/stefan-jansen/pyfolio-reloaded/blob/0.9.9/src/pyfolio/txn.py#L148

    Args:
        traded_values (pd.Series): The value traded per day, positive.
        position_values (pd.DataFrame): The value of every position (columns) per day.
        denominator (str, optional): Options are: "gross_value" (the average gross value of the day
            and the day before), "net_value" (the net value). Defaults to "gross_value".

    Returns:
        pd.DataFrame: The Traded Value, the Gross Value (or Net Value) and the Turnover per
        day, 0 on days without trades.
    """
    check_option("denominator", denominator, TURNOVER_DENOMINATORS)
    traded = traded_values.reindex(position_values.index).fillna(0)
    if denominator == "gross_value":
        value = position_values.abs().sum(axis=1)
        average = value.rolling(2).mean()
        average.iloc[0] = value.iloc[0] / 2
        name = "Gross Value"
    else:
        value = position_values.sum(axis=1)
        average = value
        name = "Net Value"
    turnover = (traded / average.where(average != 0)).fillna(0)
    return pd.DataFrame({"Traded Value": traded, name: value, "Turnover": turnover})


def get_gross_leverage(position_values: pd.DataFrame) -> pd.DataFrame:
    """
    The gross leverage of a portfolio per day: its long and short positions together as a
    multiple of its net value.

    It is defined in pyfolio (Quantopian) by gross_lev:

    - Gross Leverage_t = Σ_i |value of position i| / Σ_i value of position i

    A portfolio of long positions has a gross leverage of 1; short positions raise it (a
    130/30 portfolio has 1.6). The positions hold no cash, so the net value is that of the
    positions, where pyfolio adds the cash.

    Shares similar logic:
    - pyfolio, gross_lev: https://github.com/stefan-jansen/pyfolio-reloaded/blob/0.9.9/src/pyfolio/timeseries.py#L605

    Args:
        position_values (pd.DataFrame): The value of every position (columns) per day.

    Returns:
        pd.DataFrame: The Gross Value, Net Value and Gross Leverage per day, NaN on days
        without a positive net value.
    """
    gross = position_values.abs().sum(axis=1)
    net = position_values.sum(axis=1)
    return pd.DataFrame(
        {
            "Gross Value": gross,
            "Net Value": net,
            "Gross Leverage": gross / net.where(net > 0),
        }
    )
