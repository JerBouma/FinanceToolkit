"""Factor Portfolio Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd
from scipy import stats

from financetoolkit.performance.helpers import get_newey_west_lags
from financetoolkit.portfolio.construction.optimization.estimation_model import (
    get_bootstrap_indices,
)
from financetoolkit.utilities.error_model import InvalidArgumentError, check_option
from financetoolkit.utilities.statistics_model import newey_west_long_run_variance

# The examples' output tables are wider than the line length.
# ruff: noqa: E501

# The ways the tickers within a sorted portfolio are weighted.
WEIGHTINGS = ("equal", "value")


def assign_breakpoint_portfolios(
    characteristic: pd.DataFrame, breakpoints: list[float]
) -> pd.DataFrame:
    """
    Sort the tickers of each date into portfolios by percentile breakpoints of a characteristic, as in the sorts of
    Fama and French: the lowest values go into portfolio 1, the highest into portfolio len(breakpoints) + 1.

    The breakpoints are percentiles of the characteristic of the tickers on that date (linear interpolation between
    the sorted values). A value on a breakpoint belongs to the portfolio below it, so tickers with the same value
    always share a portfolio; with many equal values a portfolio can stay empty on a date. A date with fewer
    tickers with a value than portfolios gets no portfolios.

    The formula is as follows:

    - b(k, t) = the breakpoints[k] percentile of c(i, t) over the tickers i with a value on date t
    - Portfolio of i on date t = 1 + the number of breakpoints b(k, t) below c(i, t)

    For more information, see the following papers:

    - Fama, E.F. and French, K.R. (1992). "The Cross-Section of Expected Stock Returns." Journal of Finance,
      47(2), 427-465.
    - Fama, E.F. and French, K.R. (1993). "Common Risk Factors in the Returns on Stocks and Bonds." Journal of
      Financial Economics, 33(1), 3-56.

    Shares similar logic:
    - alphalens, quantize_factor: https://github.com/stefan-jansen/alphalens-reloaded/blob/v0.4.4/src/alphalens/utils.py#L91

    Args:
        characteristic (pd.DataFrame): The characteristic with the dates as index and the tickers as columns, NaN
            where a ticker has none.
        breakpoints (list[float]): The percentiles that separate the portfolios, as decimals between 0 and 1 in
            increasing order, e.g. [0.3, 0.7] for Fama and French's 30th and 70th percentiles or [0.2, 0.4,
            0.6, 0.8] for quintiles.

    Returns:
        pd.DataFrame: The portfolio (1 to len(breakpoints) + 1) of each ticker per date, NaN where the ticker has
        no characteristic or the date too few tickers.

    Notes:
        - With equally spaced breakpoints this equals pandas' qcut per date (which alphalens' quantize_factor
          uses) whenever the breakpoints differ from each other. Where equal values make two breakpoints
          coincide, qcut refuses the date and this sort keeps the ties together.
        - The breakpoints come from all tickers on the date, where Fama and French take them from the NYSE
          stocks only.

    As an example:

    ```python
    import pandas as pd
    from financetoolkit.performance import factor_portfolio_model

    characteristic = pd.DataFrame(
        {"A": [1.0, 5.0], "B": [2.0, 4.0], "C": [2.0, 3.0], "D": [4.0, 2.0], "E": [5.0, None]},
        index=pd.period_range("2024-01", periods=2, freq="M"),
    )

    factor_portfolio_model.assign_breakpoint_portfolios(characteristic, [0.3, 0.7])
    ```

    Which returns:

    |         |   A |   B |   C |   D |   E |
    |:--------|----:|----:|----:|----:|----:|
    | 2024-01 |   1 |   1 |   1 |   3 |   3 |
    | 2024-02 |   3 |   2 |   2 |   1 | nan |
    """
    edges = np.asarray(breakpoints, dtype=float)
    if (
        edges.size == 0
        or np.any((edges <= 0) | (edges >= 1))
        or np.any(np.diff(edges) <= 0)
    ):
        raise InvalidArgumentError(
            "breakpoints must be percentiles between 0 and 1 in increasing order, e.g. [0.3, 0.7]."
        )

    values = characteristic.to_numpy(dtype=float)
    count = np.sum(~np.isnan(values), axis=1)

    # All-NaN rows have no percentiles; they are left out below by their count.
    with np.errstate(all="ignore"):
        quantiles = (
            np.nanquantile(values[count > 0], edges, axis=1).T
            if np.any(count > 0)
            else np.empty((0, edges.size))
        )
    percentiles = np.full((len(values), edges.size), np.nan)
    percentiles[count > 0] = quantiles

    portfolio = 1 + np.sum(values[:, :, None] > percentiles[:, None, :], axis=2).astype(
        float
    )
    portfolio[np.isnan(values) | (count < edges.size + 1)[:, None]] = np.nan

    return pd.DataFrame(
        portfolio, index=characteristic.index, columns=characteristic.columns
    )


def get_sorted_portfolio_weights(
    assignment: pd.DataFrame,
    returns: pd.DataFrame,
    market_cap: pd.DataFrame | None = None,
    holding_period: int = 1,
    lag: int = 1,
) -> pd.DataFrame:
    """
    Calculate the weights of the tickers in each sorted portfolio per period: which tickers each portfolio holds,
    from a sort known before the period starts, and how much of each.

    The sort of a date applies lag periods later, so the return of period t never uses a characteristic of
    period t itself. The portfolios are formed on a rebalancing date and keep their tickers for holding_period
    periods, the first rebalancing date being the first period with a sort. Within a portfolio the tickers are
    weighted equally, or by their market cap at the end of the previous period (value weighted), and only the
    tickers with a return in the period count, so the weights of each portfolio add up to one.

    The formula is as follows:

    - Members of p in period t = the tickers sorted into p on the rebalancing date s ≤ t, with the sort of s − lag
    - Equal weight: w(i, t) = 1 / N(p, t), with N(p, t) the members with a return in period t
    - Value weight: w(i, t) = ME(i, t − 1) / Σ_j ME(j, t − 1) over the members j with a return in period t

    For more information, see the following papers:

    - Fama, E.F. and French, K.R. (1993). "Common Risk Factors in the Returns on Stocks and Bonds." Journal of
      Financial Economics, 33(1), 3-56.
    - Jegadeesh, N. and Titman, S. (1993). "Returns to Buying Winners and Selling Losers: Implications for Stock
      Market Efficiency." Journal of Finance, 48(1), 65-91, for holding periods longer than the rebalancing
      frequency of the data.

    Args:
        assignment (pd.DataFrame): The portfolio of each ticker per date (see assign_breakpoint_portfolios), on
            the dates of the returns.
        returns (pd.DataFrame): The return of each ticker per period, as decimals, with the periods as index and
            the tickers as columns, NaN where a ticker has none.
        market_cap (pd.DataFrame | None, optional): The market cap of each ticker at the end of each period, on
            the index of the returns. Defaults to None, which weights the tickers equally.
        holding_period (int, optional): The number of periods between rebalancing dates. Defaults to 1.
        lag (int, optional): The number of periods between the sort and the first period it applies to.
            Defaults to 1.

    Returns:
        pd.DataFrame: The weight per period (index) and portfolio and ticker (columns), NaN for a ticker
        outside the portfolio in that period.

    Notes:
        - A member without a return in a period (not yet listed, delisted or not traded) is left out of that
          period and the weights of the others rescaled. The return at delisting is therefore not counted.
        - Value weights at the previous period's market cap are the weights of a portfolio bought at the start
          of the period and held through it, as Fama and French weight their monthly value-weighted returns.
          Equal weights are restored every period.

    As an example:

    ```python
    import pandas as pd
    from financetoolkit.performance import factor_portfolio_model

    periods = pd.period_range("2024-01", periods=3, freq="M")
    assignment = pd.DataFrame({"A": [1.0, 2.0, 2.0], "B": [2.0, 1.0, 1.0]}, index=periods)
    returns = pd.DataFrame({"A": [0.01, 0.02, -0.01], "B": [0.03, -0.02, 0.01]}, index=periods)

    factor_portfolio_model.get_sorted_portfolio_weights(assignment, returns)
    ```

    Which returns:

    |         |   (1.0, 'A') |   (1.0, 'B') |   (2.0, 'A') |   (2.0, 'B') |
    |:--------|-------------:|-------------:|-------------:|-------------:|
    | 2024-01 |          nan |          nan |          nan |          nan |
    | 2024-02 |            1 |          nan |          nan |            1 |
    | 2024-03 |          nan |            1 |            1 |          nan |
    """
    if holding_period < 1 or lag < 0:
        raise InvalidArgumentError(
            "holding_period must be at least 1 and lag at least 0."
        )

    sorted_before = assignment.reindex(
        index=returns.index, columns=returns.columns
    ).shift(lag)

    # Every period holds the sort of its rebalancing date, counted from the first period with a sort.
    has_sort = sorted_before.notna().any(axis=1).to_numpy()
    members = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    if has_sort.any():
        first = int(np.argmax(has_sort))
        offsets = np.arange(len(returns) - first)
        rebalancing = first + (offsets // holding_period) * holding_period
        members.iloc[first:] = sorted_before.iloc[rebalancing].to_numpy()

    if market_cap is None:
        size = pd.DataFrame(1.0, index=returns.index, columns=returns.columns)
    else:
        size = market_cap.reindex(index=returns.index, columns=returns.columns).shift(1)
        size = size.where(size > 0)

    labels = sorted(pd.unique(members.stack().dropna()))

    weights = {}
    for label in labels:
        held = size.where((members == label) & returns.notna())
        weights[label] = held.div(held.sum(axis=1, min_count=1), axis=0)

    if not weights:
        return pd.DataFrame(index=returns.index)

    return pd.concat(weights, axis=1)


def get_sorted_portfolio_returns(
    weights: pd.DataFrame, returns: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate the return of each sorted portfolio per period: the weighted return of the tickers it holds.

    The formula is as follows:

    - R(p, t) = Σ_i w(i, t) × R(i, t) over the tickers i in portfolio p in period t

    Args:
        weights (pd.DataFrame): The weight per period and portfolio and ticker (see get_sorted_portfolio_weights).
        returns (pd.DataFrame): The return of each ticker per period, on the index of the weights.

    Returns:
        pd.DataFrame: The return per period (index) and portfolio (columns), NaN for a period in which the
        portfolio holds no tickers.

    As an example:

    ```python
    import pandas as pd
    from financetoolkit.performance import factor_portfolio_model

    periods = pd.period_range("2024-01", periods=2, freq="M")
    weights = pd.DataFrame(
        {(1, "A"): [0.5, 0.25], (1, "B"): [0.5, 0.75]}, index=periods
    )
    returns = pd.DataFrame({"A": [0.02, -0.04], "B": [0.04, 0.04]}, index=periods)

    factor_portfolio_model.get_sorted_portfolio_returns(weights, returns)
    ```

    Which returns:

    |         |    1 |
    |:--------|-----:|
    | 2024-01 | 0.03 |
    | 2024-02 | 0.02 |
    """
    portfolios = list(dict.fromkeys(weights.columns.get_level_values(0)))

    return pd.DataFrame(
        {
            portfolio: (
                weights[portfolio]
                * returns.reindex(
                    index=weights.index, columns=weights[portfolio].columns
                )
            ).sum(axis=1, min_count=1)
            for portfolio in portfolios
        },
        index=weights.index,
    )


def get_quantile_portfolios(
    characteristic: pd.DataFrame,
    returns: pd.DataFrame,
    quantiles: int = 5,
    market_cap: pd.DataFrame | None = None,
    holding_period: int = 1,
    lag: int = 1,
) -> pd.DataFrame:
    """
    Calculate the returns of quantile portfolios sorted on a characteristic and of their long-short spread: how much
    more the tickers with the highest values of the characteristic returned than those with the lowest.

    On each rebalancing date, the tickers are sorted into quantiles of the characteristic known lag periods
    earlier (equally spaced percentile breakpoints, see assign_breakpoint_portfolios), held for holding_period
    periods and weighted equally or by market cap (see get_sorted_portfolio_weights). The Top Minus Bottom
    spread is the return of a portfolio long the highest quantile and short the lowest, which costs nothing to
    set up, so it is an excess return.

    The formula is as follows:

    - Breakpoints: the 1 / n, 2 / n, ..., (n − 1) / n percentiles of the characteristic per date
    - R(q, t) = Σ_i w(i, t) × R(i, t) over the tickers i in quantile q in period t
    - Top Minus Bottom(t) = R(n, t) − R(1, t)

    For more information, see the following papers:

    - Fama, E.F. and French, K.R. (1992). "The Cross-Section of Expected Stock Returns." Journal of Finance,
      47(2), 427-465.
    - Fama, E.F. and French, K.R. (1993). "Common Risk Factors in the Returns on Stocks and Bonds." Journal of
      Financial Economics, 33(1), 3-56.
    - Jegadeesh, N. and Titman, S. (1993). "Returns to Buying Winners and Selling Losers: Implications for Stock
      Market Efficiency." Journal of Finance, 48(1), 65-91.

    Shares similar logic:
    - alphalens, quantize_factor: https://github.com/stefan-jansen/alphalens-reloaded/blob/v0.4.4/src/alphalens/utils.py#L91
    - alphalens, mean_return_by_quantile: https://github.com/stefan-jansen/alphalens-reloaded/blob/v0.4.4/src/alphalens/performance.py#L461

    Args:
        characteristic (pd.DataFrame): The characteristic with the periods as index and the tickers as columns,
            e.g. the P/E or the 12-1 month momentum at the end of each month, on the index of the returns.
        returns (pd.DataFrame): The return of each ticker per period, as decimals.
        quantiles (int, optional): The number of quantile portfolios. Defaults to 5.
        market_cap (pd.DataFrame | None, optional): The market cap of each ticker at the end of each period, for
            value weights. Defaults to None, which weights the tickers equally.
        holding_period (int, optional): The number of periods between rebalancing dates. Defaults to 1.
        lag (int, optional): The number of periods between the characteristic and the first return it sorts.
            Defaults to 1.

    Returns:
        pd.DataFrame: The return of "Quantile 1" (the lowest values) to "Quantile n" and the "Top Minus Bottom"
        spread (columns) per period (index).

    Notes:
        - Where alphalens averages the forward returns of every date over the horizon (overlapping holding
          periods), these are the returns of one portfolio per quantile held from one rebalancing date to the
          next, as in a backtest, and so a series of non-overlapping returns.
        - A period with fewer tickers with a characteristic than quantiles has no portfolios, and a quantile left
          empty by equal values has no return in that period.

    As an example:

    ```python
    import numpy as np
    import pandas as pd
    from financetoolkit.performance import factor_portfolio_model

    periods = pd.period_range("2024-01", periods=6, freq="M")
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(
        rng.normal(0.01, 0.05, (6, 6)), index=periods, columns=list("ABCDEF")
    )
    characteristic = returns.rolling(2).mean()

    factor_portfolio_model.get_quantile_portfolios(characteristic, returns, quantiles=3).round(4)
    ```

    Which returns:

    |         |   Quantile 1 |   Quantile 2 |   Quantile 3 |   Top Minus Bottom |
    |:--------|-------------:|-------------:|-------------:|-------------------:|
    | 2024-01 |          nan |          nan |          nan |                nan |
    | 2024-02 |          nan |          nan |          nan |                nan |
    | 2024-03 |      -0.0219 |      -0.0291 |      -0.0536 |            -0.0317 |
    | 2024-04 |       0.0409 |       0.0037 |       0.0449 |             0.0039 |
    | 2024-05 |       0.014  |       0.0041 |      -0.0107 |            -0.0247 |
    | 2024-06 |       0.0114 |       0.0324 |      -0.0205 |            -0.0319 |
    """
    if quantiles < 2:  # noqa: PLR2004
        raise InvalidArgumentError(
            "quantiles must be at least 2 to compare the top with the bottom."
        )

    assignment = assign_breakpoint_portfolios(
        characteristic, [number / quantiles for number in range(1, quantiles)]
    )
    weights = get_sorted_portfolio_weights(
        assignment, returns, market_cap, holding_period, lag
    )
    portfolio_returns = get_sorted_portfolio_returns(weights, returns).reindex(
        columns=[float(number) for number in range(1, quantiles + 1)]
    )
    portfolio_returns.columns = [
        f"Quantile {number}" for number in range(1, quantiles + 1)
    ]
    portfolio_returns["Top Minus Bottom"] = (
        portfolio_returns[f"Quantile {quantiles}"] - portfolio_returns["Quantile 1"]
    )

    return portfolio_returns


def get_double_sort_portfolios(
    characteristic: pd.DataFrame,
    market_cap: pd.DataFrame,
    returns: pd.DataFrame,
    size_breakpoints: list[float] | None = None,
    characteristic_breakpoints: list[float] | None = None,
    weighting: str = "value",
    holding_period: int = 1,
    lag: int = 1,
) -> pd.DataFrame:
    """
    Calculate the returns of portfolios double sorted on size and a characteristic, and the size and
    characteristic factors built from them, as Fama and French build SMB and HML: the return of small over big
    companies, and of high over low values of the characteristic, each with the other held constant.

    The tickers are sorted on size (the market cap) and, independently, on the characteristic, by percentile
    breakpoints of each per date (see assign_breakpoint_portfolios). Every combination is a portfolio, by default
    the six of Fama and French's 2 × 3 sort (the median for size, the 30th and 70th percentiles for the
    characteristic), held and weighted as in get_sorted_portfolio_weights. The size factor averages the smallest
    size group over the characteristic groups minus the same for the biggest; the characteristic factor averages
    the highest characteristic group over the size groups minus the same for the lowest.

    The formula for the 2 × 3 sort is as follows:

    - Size Factor (SMB) = 1/3 (Small Low + Small Neutral + Small High) − 1/3 (Big Low + Big Neutral + Big High)
    - Characteristic Factor (HML for the book-to-market) = 1/2 (Small High + Big High) − 1/2 (Small Low + Big Low)

    For more information, see the following papers:

    - Fama, E.F. and French, K.R. (1993). "Common Risk Factors in the Returns on Stocks and Bonds." Journal of
      Financial Economics, 33(1), 3-56.
    - Fama, E.F. and French, K.R. (2015). "A Five-Factor Asset Pricing Model." Journal of Financial Economics,
      116(1), 1-22.

    Args:
        characteristic (pd.DataFrame): The characteristic with the periods as index and the tickers as columns,
            e.g. the book-to-market ratio, on the index of the returns.
        market_cap (pd.DataFrame): The market cap of each ticker at the end of each period, the size that is
            sorted on and the value weights.
        returns (pd.DataFrame): The return of each ticker per period, as decimals.
        size_breakpoints (list[float] | None, optional): The size percentiles that separate the size groups.
            Defaults to None, which uses [0.5] (the median).
        characteristic_breakpoints (list[float] | None, optional): The characteristic percentiles that separate
            the characteristic groups. Defaults to None, which uses [0.3, 0.7].
        weighting (str, optional): How the tickers within a portfolio are weighted. Options are: "value" (by
            market cap at the end of the previous period), "equal" (equally). Defaults to "value".
        holding_period (int, optional): The number of periods between rebalancing dates, e.g. 12 on monthly
            returns for Fama and French's yearly rebalancing. Defaults to 1.
        lag (int, optional): The number of periods between the sort and the first return it applies to. Defaults
            to 1.

    Returns:
        pd.DataFrame: The return of each portfolio, the "Size Factor" and the "Characteristic Factor" (columns)
        per period (index). The portfolios are named "Small" and "Big" with two size groups ("Size 1" for the
        smallest otherwise) and "Low", "Neutral" and "High" with three characteristic groups ("Characteristic
        1" for the lowest otherwise).

    Notes:
        - The sorts are independent, so a portfolio can be empty in a period when size and the characteristic
          are strongly related; its factor then has no return in that period.
        - Fama and French sort once a year in June, on the market cap of June and the book-to-market ratio of
          the previous fiscal year, with NYSE breakpoints. With monthly data, holding_period=12 and a lag that
          reaches back to that fiscal year give their timing; the breakpoints here come from all tickers.

    As an example:

    ```python
    import numpy as np
    import pandas as pd
    from financetoolkit.performance import factor_portfolio_model

    periods = pd.period_range("2024-01", periods=4, freq="M")
    rng = np.random.default_rng(1)
    tickers = [f"T{number}" for number in range(12)]
    returns = pd.DataFrame(rng.normal(0.01, 0.05, (4, 12)), index=periods, columns=tickers)
    market_cap = pd.DataFrame(
        rng.lognormal(10, 1, (4, 12)), index=periods, columns=tickers
    )
    book_to_market = pd.DataFrame(rng.uniform(0.1, 2, (4, 12)), index=periods, columns=tickers)

    factor_portfolio_model.get_double_sort_portfolios(book_to_market, market_cap, returns).round(4)
    ```

    Which returns:

    |         |   Small Low |   Small Neutral |   Small High |   Big Low |   Big Neutral |   Big High |   Size Factor |   Characteristic Factor |
    |:--------|------------:|----------------:|-------------:|----------:|--------------:|-----------:|--------------:|------------------------:|
    | 2024-01 |    nan      |        nan      |     nan      |  nan      |      nan      |   nan      |      nan      |                nan      |
    | 2024-02 |      0.0747 |          0.0098 |       0.024  |    0.0079 |       -0.0281 |    -0.0141 |        0.0476 |                 -0.0364 |
    | 2024-03 |     -0.0923 |         -0.0103 |       0.033  |    0.0139 |        0.0084 |     0.0677 |       -0.0532 |                  0.0896 |
    | 2024-04 |     -0.0338 |         -0.0514 |      -0.0099 |    0.0112 |        0.0123 |    -0.0724 |       -0.0154 |                 -0.0298 |
    """
    check_option("weighting", weighting, WEIGHTINGS)
    size_breakpoints = [0.5] if size_breakpoints is None else size_breakpoints
    characteristic_breakpoints = (
        [0.3, 0.7] if characteristic_breakpoints is None else characteristic_breakpoints
    )

    size_groups = len(size_breakpoints) + 1
    characteristic_groups = len(characteristic_breakpoints) + 1
    size_names = (
        ["Small", "Big"]
        if size_groups == 2  # noqa: PLR2004
        else [f"Size {n}" for n in range(1, size_groups + 1)]
    )
    characteristic_names = (
        ["Low", "Neutral", "High"]
        if characteristic_groups == 3  # noqa: PLR2004
        else [f"Characteristic {n}" for n in range(1, characteristic_groups + 1)]
    )

    size_assignment = assign_breakpoint_portfolios(
        market_cap.reindex(index=characteristic.index), size_breakpoints
    )
    characteristic_assignment = assign_breakpoint_portfolios(
        characteristic, characteristic_breakpoints
    )
    # One number per combination (size group × the number of characteristic groups + characteristic group), so
    # the weights treat every combination as a portfolio of its own.
    assignment = (
        size_assignment - 1
    ) * characteristic_groups + characteristic_assignment

    weights = get_sorted_portfolio_weights(
        assignment,
        returns,
        market_cap if weighting == "value" else None,
        holding_period,
        lag,
    )
    portfolio_returns = get_sorted_portfolio_returns(weights, returns)

    names = {}
    for size_number, size_name in enumerate(size_names):
        for characteristic_number, characteristic_name in enumerate(
            characteristic_names
        ):
            names[
                float(size_number * characteristic_groups + characteristic_number + 1)
            ] = f"{size_name} {characteristic_name}"
    portfolio_returns = portfolio_returns.reindex(columns=list(names)).set_axis(
        list(names.values()), axis=1
    )

    portfolio_grid = portfolio_returns.to_numpy().reshape(
        len(portfolio_returns), size_groups, characteristic_groups
    )
    with np.errstate(invalid="ignore"):
        portfolio_returns["Size Factor"] = portfolio_grid[:, 0, :].mean(
            axis=1
        ) - portfolio_grid[:, -1, :].mean(axis=1)
        portfolio_returns["Characteristic Factor"] = portfolio_grid[:, :, -1].mean(
            axis=1
        ) - portfolio_grid[:, :, 0].mean(axis=1)

    return portfolio_returns


def get_portfolio_turnover(
    weights: pd.DataFrame, returns: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate the turnover of each sorted portfolio per period: the share of the portfolio that is traded at the
    start of the period to go from the weights the previous period drifted to, to the new weights.

    The weights of the previous period drift with the returns of its tickers; the turnover is half the sum of the
    absolute changes from those drifted weights, the one-way turnover (what is bought, which equals what is
    sold). A portfolio that keeps its tickers trades only to restore equal weights, and a value-weighted one
    trades nothing when the market caps move with the returns.

    The formula is as follows:

    - Drifted weight: w̃(i, t − 1) = w(i, t − 1) × (1 + R(i, t − 1)) / Σ_j w(j, t − 1) × (1 + R(j, t − 1))
    - Turnover(p, t) = 1/2 × Σ_i |w(i, t) − w̃(i, t − 1)| over the tickers in p in period t or t − 1

    For more information, see the following paper:

    - DeMiguel, V., Garlappi, L. and Uppal, R. (2009). "Optimal Versus Naive Diversification: How Inefficient is
      the 1/N Portfolio Strategy?" Review of Financial Studies, 22(5), 1915-1953, for turnover against the
      drifted weights.

    Shares similar logic:
    - alphalens, quantile_turnover: https://github.com/stefan-jansen/alphalens-reloaded/blob/v0.4.4/src/alphalens/performance.py#L575

    Args:
        weights (pd.DataFrame): The weight per period and portfolio and ticker (see get_sorted_portfolio_weights).
        returns (pd.DataFrame): The return of each ticker per period, on the index of the weights.

    Returns:
        pd.DataFrame: The turnover per period (index) and portfolio (columns) as a decimal, NaN for the period a
        portfolio is first formed (and after a period without tickers).

    Notes:
        - alphalens' quantile turnover counts the share of the tickers that are new to the quantile; this
          weighs the trades by their size, so it also counts the rebalancing to equal weights.

    As an example:

    ```python
    import pandas as pd
    from financetoolkit.performance import factor_portfolio_model

    periods = pd.period_range("2024-01", periods=3, freq="M")
    weights = pd.DataFrame(
        {(1, "A"): [0.5, 0.5, None], (1, "B"): [0.5, 0.5, 0.5], (1, "C"): [None, None, 0.5]}, index=periods
    )
    returns = pd.DataFrame({"A": [0.1, 0.0, 0.0], "B": [-0.1, 0.0, 0.0], "C": [0.0, 0.0, 0.0]}, index=periods)

    factor_portfolio_model.get_portfolio_turnover(weights, returns)
    ```

    Which returns:

    |         |    1 |
    |:--------|-----:|
    | 2024-01 |  nan |
    | 2024-02 | 0.05 |
    | 2024-03 | 0.5  |
    """
    portfolios = list(dict.fromkeys(weights.columns.get_level_values(0)))

    turnover = {}
    for portfolio in portfolios:
        current = weights[portfolio]
        grown = current * (
            1 + returns.reindex(index=current.index, columns=current.columns)
        )
        drifted = grown.div(grown.sum(axis=1, min_count=1), axis=0).shift(1)

        # A weight left empty is no position, so 0 in the difference; a period without either portfolio has none.
        both = current.notna().any(axis=1) & drifted.notna().any(axis=1)
        traded = (current.fillna(0) - drifted.fillna(0)).abs().sum(axis=1) / 2
        turnover[portfolio] = traded.where(both)

    return pd.DataFrame(turnover, index=weights.index)


def get_factor_statistics(
    factor_returns: pd.DataFrame | pd.Series,
    periods_per_year: float = 12,
    lags: int | None = None,
) -> pd.DataFrame:
    """
    Calculate whether a factor's returns differ from zero: the mean return per period, its Newey-West t-statistic
    and p-value, and the annualized Sharpe ratio of each column.

    The t-statistic divides the mean by its Newey-West standard error, which allows for autocorrelation and
    changing volatility in the returns (as for holding periods longer than a period), with Bartlett weights up to
    lags autocovariances. The p-value is two-sided from the normal distribution. The Sharpe ratio is the mean over
    the standard deviation (ddof=1), annualized with the square root of the periods per year; it is the Sharpe
    ratio of a long-short spread, which is an excess return, while for a long-only portfolio pass returns in
    excess of the risk-free rate.

    The formula is as follows:

    - Long-run variance: S = γ(0) + 2 × Σ_{l=1}^{L} (1 − l / (L + 1)) × γ(l), with γ(l) the autocovariance at lag l
    - Newey-West Standard Error = √(S / T)
    - t-Statistic = Mean Return / Newey-West Standard Error
    - Sharpe Ratio = Mean Return / Standard Deviation × √(periods per year)

    For more information, see the following papers:

    - Newey, W.K. and West, K.D. (1987). "A Simple, Positive Semi-Definite, Heteroskedasticity and
      Autocorrelation Consistent Covariance Matrix." Econometrica, 55(3), 703-708.
    - Newey, W.K. and West, K.D. (1994). "Automatic Lag Selection in Covariance Matrix Estimation." Review of
      Economic Studies, 61(4), 631-653, for the default number of lags.
    - Sharpe, W.F. (1966). "Mutual Fund Performance." Journal of Business, 39(1), 119-138.

    Referenced against:
    - statsmodels, OLS with HAC covariance: https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_hac.html

    Args:
        factor_returns (pd.DataFrame | pd.Series): The returns per period, as decimals, a column per factor or
            portfolio.
        periods_per_year (float, optional): The number of periods in a year, e.g. 12 for monthly and 252 for
            daily returns. Defaults to 12.
        lags (int | None, optional): The number of autocovariances in the Newey-West standard error. Defaults to
            None, which uses floor(4 × (T / 100) ^ (2 / 9)) for the T returns of each column.

    Returns:
        pd.DataFrame: The Mean Return, Annualized Mean Return (mean × periods per year), Standard Deviation,
        Newey-West Standard Error, t-Statistic, P-Value, Sharpe Ratio, Observations and Lags (columns) per
        factor (index).

    Notes:
        - Each column uses its own periods with a return; a column with fewer than three has no statistics.

    As an example:

    ```python
    import numpy as np
    import pandas as pd
    from financetoolkit.performance import factor_portfolio_model

    rng = np.random.default_rng(0)
    spread = pd.Series(
        rng.normal(0.005, 0.03, 60), index=pd.period_range("2020-01", periods=60, freq="M"), name="Spread"
    )

    factor_portfolio_model.get_factor_statistics(spread).round(4)
    ```

    Which returns:

    |        |   Mean Return |   Annualized Mean Return |   Standard Deviation |   Newey-West Standard Error |   t-Statistic |   P-Value |   Sharpe Ratio |   Observations |   Lags |
    |:-------|--------------:|-------------------------:|---------------------:|----------------------------:|--------------:|----------:|---------------:|---------------:|-------:|
    | Spread |        0.0073 |                   0.0878 |               0.0272 |                      0.0043 |        1.6941 |    0.0902 |         0.9338 |             60 |      3 |
    """
    frame = (
        factor_returns.to_frame()
        if isinstance(factor_returns, pd.Series)
        else factor_returns
    )

    statistics = {}
    for column, series in frame.items():
        values = series.dropna().to_numpy(dtype=float)
        observations = len(values)
        if observations < 3:  # noqa: PLR2004
            statistics[column] = dict.fromkeys(
                [
                    "Mean Return",
                    "Annualized Mean Return",
                    "Standard Deviation",
                    "Newey-West Standard Error",
                    "t-Statistic",
                    "P-Value",
                    "Sharpe Ratio",
                ],
                np.nan,
            ) | {"Observations": float(observations), "Lags": np.nan}
            continue

        lag_count = get_newey_west_lags(observations) if lags is None else int(lags)
        mean = values.mean()
        standard_deviation = values.std(ddof=1)
        standard_error = np.sqrt(
            newey_west_long_run_variance(values - mean, lag_count) / observations
        )
        t_statistic = mean / standard_error if standard_error > 0 else np.nan

        statistics[column] = {
            "Mean Return": mean,
            "Annualized Mean Return": mean * periods_per_year,
            "Standard Deviation": standard_deviation,
            "Newey-West Standard Error": standard_error,
            "t-Statistic": t_statistic,
            "P-Value": (
                2 * stats.norm.sf(abs(t_statistic))
                if not np.isnan(t_statistic)
                else np.nan
            ),
            "Sharpe Ratio": (
                mean / standard_deviation * np.sqrt(periods_per_year)
                if standard_deviation > 0
                else np.nan
            ),
            "Observations": float(observations),
            "Lags": float(lag_count),
        }

    return pd.DataFrame(statistics).T


def get_monotonicity_test(
    portfolio_returns: pd.DataFrame,
    bootstraps: int = 1000,
    block_size: float | None = None,
    seed: int | None = 0,
) -> pd.DataFrame:
    """
    Test whether the mean returns of sorted portfolios rise (or fall) across every step from the lowest to the
    highest portfolio, with Patton and Timmermann's monotonic relation (MR) test, and give the Spearman rank
    correlation between the portfolio order and their mean returns.

    A spread between the top and bottom portfolio can come from the extremes alone; the MR test asks whether every
    step up is a step up in mean return. The statistic is the smallest difference between the mean returns of
    neighbouring portfolios, and its p-value comes from a stationary bootstrap of the periods (which keeps the
    portfolios' returns of a period together and their autocorrelation), with the differences centred on their
    sample values so that the bootstrap holds the null of no rise. The decreasing test applies the same to the
    portfolios from the highest to the lowest.

    The formula is as follows:

    - Δ(k) = mean R(k + 1) − mean R(k) for the neighbouring portfolios k = 1, ..., n − 1
    - J = min_k Δ(k)
    - J*(b) = min_k (Δ*(k, b) − Δ(k)) for the bootstrap samples b = 1, ..., B
    - P-Value = share of the bootstrap samples with J*(b) > J
    - Spearman Rank Correlation = rank correlation of k and mean R(k)

    For more information, see the following papers:

    - Patton, A.J. and Timmermann, A. (2010). "Monotonicity in Asset Returns: New Tests with Applications to the
      Term Structure, the CAPM, and Portfolio Sorts." Journal of Financial Economics, 98(3), 605-625.
    - Politis, D.N. and Romano, J.P. (1994). "The Stationary Bootstrap." Journal of the American Statistical
      Association, 89(428), 1303-1313.

    Shares similar logic:
    - arch, StationaryBootstrap: https://bashtage.github.io/arch/bootstrap/generated/arch.bootstrap.StationaryBootstrap.html

    Args:
        portfolio_returns (pd.DataFrame): The returns per period of the portfolios as columns, from the lowest to
            the highest sort (e.g. "Quantile 1" to "Quantile n"). A "Top Minus Bottom" column is left out.
        bootstraps (int, optional): The number of bootstrap samples. Defaults to 1000.
        block_size (float | None, optional): The average block length of the stationary bootstrap, in periods.
            Defaults to None, which uses the cube root of the number of periods (at least 1).
        seed (int | None, optional): The seed of the random numbers, for reproducible results. Defaults to 0.

    Returns:
        pd.DataFrame: The Statistic and P-Value (columns) of the "Increasing" and "Decreasing" MR tests and the
        "Spearman Rank Correlation" (index).

    Notes:
        - Only the periods in which every portfolio has a return are used.
        - The Spearman p-value treats the n mean returns as n observations, so it means little for a handful of
          portfolios; the MR test uses the variation over time instead.
        - Patton and Timmermann also offer a studentized version; this is their test on the plain differences.

    As an example:

    ```python
    import numpy as np
    import pandas as pd
    from financetoolkit.performance import factor_portfolio_model

    rng = np.random.default_rng(0)
    periods = pd.period_range("2015-01", periods=120, freq="M")
    portfolio_returns = pd.DataFrame(
        rng.normal([0.0, 0.01, 0.02, 0.03, 0.04], 0.03, (120, 5)),
        index=periods,
        columns=[f"Quantile {number}" for number in range(1, 6)],
    )

    factor_portfolio_model.get_monotonicity_test(portfolio_returns).round(4)
    ```

    Which returns:

    |                           |   Statistic |   P-Value |
    |:--------------------------|------------:|----------:|
    | Increasing                |      0.0017 |         0 |
    | Decreasing                |     -0.0198 |         1 |
    | Spearman Rank Correlation |      1      |         0 |
    """
    values = (
        portfolio_returns.drop(columns="Top Minus Bottom", errors="ignore")
        .dropna()
        .to_numpy(dtype=float)
    )
    periods, portfolios = values.shape

    if portfolios < 2 or periods < 3:  # noqa: PLR2004
        raise ValueError(
            "The monotonicity test needs at least two portfolios with returns in three or more of the same periods."
        )
    if bootstraps < 1:
        raise InvalidArgumentError("bootstraps must be at least 1.")

    block = max(1.0, periods ** (1 / 3)) if block_size is None else float(block_size)
    indices = get_bootstrap_indices(
        periods,
        "stationary",
        min(block, periods - 1),
        np.random.default_rng(seed),
        bootstraps,
    )

    differences = np.diff(values, axis=1)
    sample = differences.mean(axis=0)
    # The mean of the differences in each bootstrap sample, one row per sample.
    resampled = differences[indices].mean(axis=1) - sample

    increasing = sample.min()
    decreasing = (-sample).min()
    mean_returns = values.mean(axis=0)
    spearman = stats.spearmanr(np.arange(portfolios), mean_returns)

    return pd.DataFrame(
        {
            "Statistic": [increasing, decreasing, spearman.statistic],
            "P-Value": [
                np.mean(resampled.min(axis=1) > increasing),
                np.mean((-resampled).min(axis=1) > decreasing),
                spearman.pvalue,
            ],
        },
        index=["Increasing", "Decreasing", "Spearman Rank Correlation"],
    )
