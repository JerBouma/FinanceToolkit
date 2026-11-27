"""Portfolio Attribution Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from financetoolkit.utilities.error_model import check_option

# Peak-to-trough dates of the S&P 500 in well-known market stress episodes, used as the
# default historical scenarios of a stress test.
HISTORICAL_SCENARIOS: dict[str, tuple[str, str]] = {
    "Dot-com Crash": ("2000-03-24", "2002-10-09"),
    "Global Financial Crisis": ("2007-10-09", "2009-03-09"),
    "Euro Area Debt Crisis": ("2011-04-29", "2011-10-03"),
    "2015-2016 Sell-off": ("2015-05-21", "2016-02-11"),
    "2018 Q4 Sell-off": ("2018-09-20", "2018-12-24"),
    "COVID-19 Crash": ("2020-02-19", "2020-03-23"),
    "2022 Rate Shock": ("2022-01-03", "2022-10-12"),
}

# The Select Sector SPDR funds, which track the sectors of the S&P 500 and stand in for
# the benchmark's sector returns in a sector attribution.
SECTOR_ETFS: dict[str, str] = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}


def get_brinson_fachler_attribution(
    portfolio_weights: pd.Series,
    portfolio_returns: pd.Series,
    benchmark_weights: pd.Series,
    benchmark_returns: pd.Series,
) -> pd.DataFrame:
    """
    The Brinson-Fachler attribution of a portfolio's excess return over its benchmark: how
    much the allocation and the selection per segment added.

    The excess return is split into allocation, selection and interaction effects per
    segment (Brinson and Fachler, 1985; Brinson, Hood and Beebower, 1986):

    - Allocation = (w_p - w_b) x (r_b - R_b): the reward for over- or underweighting
      segments that beat or lag the benchmark as a whole.
    - Selection = w_b x (r_p - r_b): the reward for picking better assets within a segment.
    - Interaction = (w_p - w_b) x (r_p - r_b): the joint effect of both decisions.

    Here w are segment weights, r segment returns and R_b the benchmark's total return. The
    three effects add up to the portfolio return minus the benchmark return. A segment held
    short has a negative portfolio weight and keeps its own return; a segment the portfolio
    does not hold gets the benchmark's segment return, so only its allocation effect
    remains, and has no Portfolio Return in the table.

    A benchmark segment without a return (NaN), such as a sector whose fund has no prices
    for the whole period, keeps its benchmark weight: its return is unknown, not zero, and
    leaving it out would hand its weight to the other segments. The benchmark's total
    return is then unknown, and with it every allocation effect, the selection and
    interaction effect of that segment and the Total row's effects; the other segments'
    selection and interaction effects remain.

    Args:
        portfolio_weights (pd.Series): The portfolio weight per segment.
        portfolio_returns (pd.Series): The portfolio's return per segment.
        benchmark_weights (pd.Series): The benchmark weight per segment.
        benchmark_returns (pd.Series): The benchmark's return per segment, NaN where it
            is unknown.

    Returns:
        pd.DataFrame: Per segment and in total the weights, returns and the Allocation,
        Selection, Interaction and Total Effect.
    """
    segments = (
        portfolio_weights.index.union(benchmark_weights.index)
        .union(portfolio_returns.index)
        .union(benchmark_returns.index)
    )
    wp = portfolio_weights.reindex(segments).fillna(0)
    wb = benchmark_weights.reindex(segments).fillna(0)

    if wb.sum() <= 0 or wp.sum() <= 0:
        raise ValueError("Both the portfolio and the benchmark need positive weights.")

    wp, wb = wp / wp.sum(), wb / wb.sum()
    rb = benchmark_returns.reindex(segments)
    # A segment the portfolio does not hold has no portfolio return; using the benchmark
    # segment return there leaves only its allocation effect, as the method intends. A
    # segment held short (a negative weight) keeps its own return, so the total stays
    # the portfolio's return, Σ w_p x r_p.
    rp = portfolio_returns.reindex(segments).where(wp != 0, rb)

    # A segment in the benchmark without a return makes the benchmark's total unknown.
    total_benchmark = float((wb * rb)[wb != 0].sum(skipna=False))
    # A segment outside the benchmark is compared with the benchmark as a whole, so it
    # adds no allocation effect and its result shows as interaction.
    rb = rb.where(wb != 0, rb.fillna(total_benchmark))
    rp = rp.fillna(rb)

    allocation = (wp - wb) * (rb - total_benchmark)
    selection = wb * (rp - rb)
    interaction = (wp - wb) * (rp - rb)
    total = allocation + selection + interaction

    result = pd.DataFrame(
        {
            "Portfolio Weight": wp,
            "Benchmark Weight": wb,
            "Portfolio Return": rp.where(wp != 0),
            "Benchmark Return": rb,
            "Allocation Effect": allocation,
            "Selection Effect": selection,
            "Interaction Effect": interaction,
            "Total Effect": total,
        }
    )
    result.loc["Total"] = [
        wp.sum(),
        wb.sum(),
        float((wp * rp)[wp != 0].sum()),
        total_benchmark,
        allocation.sum(skipna=False),
        selection.sum(skipna=False),
        interaction.sum(skipna=False),
        total.sum(skipna=False),
    ]

    return result


# The ways to link the attribution effects of sub-periods into effects over the whole range.
LINKING_METHODS = ("carino", "menchero")


def get_linking_coefficients(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    linking: str = "carino",
) -> pd.Series:
    """
    The coefficient per sub-period that turns the arithmetic attribution effects of that
    sub-period into its share of the excess return over the whole range.

    Returns compound over time while excess returns of sub-periods do not add up to the
    excess return over the whole range, so each sub-period's effects are scaled such that
    the linked effects sum to R - B, with R and B the portfolio and benchmark returns
    compounded over all T sub-periods:

    - Cariño (1999): coefficient_t = k_t / K, with
      k_t = [ln(1 + R_t) - ln(1 + B_t)] / (R_t - B_t) and
      K = [ln(1 + R) - ln(1 + B)] / (R - B), where k_t = 1 / (1 + R_t) when R_t = B_t
      and K = 1 / (1 + R) when R = B (the limits of both ratios).
    - Menchero (2000): coefficient_t = M + α_t, with
      M = [(R - B) / T] / [(1 + R)^(1 / T) - (1 + B)^(1 / T)] and
      α_t = [R - B - M x Σ (R_s - B_s)] / Σ (R_s - B_s)² x (R_t - B_t), where
      M = (1 + R)^((T - 1) / T) when R = B (its limit). M is the average scaling and
      α_t the smallest corrections (in the sum of squares) that make the effects add up.

    Both make Σ coefficient_t x (R_t - B_t) equal R - B. Cariño's scaling follows the
    logarithmic excess return of each sub-period; Menchero's stays as close as possible to
    one uniform scaling factor. The forms follow Bacon, C. (2004), Practical Portfolio
    Performance Measurement and Attribution, pp. 191-196.

    Args:
        portfolio_returns (pd.Series): The portfolio return per sub-period (R_t).
        benchmark_returns (pd.Series): The benchmark return per sub-period (B_t).
        linking (str, optional): The linking. Options are: "carino" (Cariño), "menchero" (Menchero).
            Defaults to "carino".

    Returns:
        pd.Series: The linking coefficient per sub-period.
    """
    check_option("linking", linking, LINKING_METHODS)

    returns = portfolio_returns.astype(float)
    benchmark = benchmark_returns.reindex(returns.index).astype(float)
    if returns.isna().any() or benchmark.isna().any():
        raise ValueError("Linking needs a portfolio and benchmark return per period.")

    total = float(np.prod(1 + returns) - 1)
    total_benchmark = float(np.prod(1 + benchmark) - 1)
    excess = returns - benchmark
    periods = len(returns)

    if linking == "carino":
        if (returns <= -1).any() or (benchmark <= -1).any():
            raise ValueError(
                "Cariño linking needs sub-period returns above -100%; use linking='menchero'."
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            k_t = np.where(
                excess == 0,
                1 / (1 + returns),
                (np.log1p(returns) - np.log1p(benchmark)) / excess,
            )
        k = (
            1 / (1 + total)
            if total == total_benchmark
            else (np.log1p(total) - np.log1p(total_benchmark))
            / (total - total_benchmark)
        )
        return pd.Series(k_t / k, index=returns.index)

    # The limit of M when R = B; the corrections α_t still apply, as Σ (R_t - B_t) need not be 0.
    m = (
        (1 + total) ** ((periods - 1) / periods)
        if total == total_benchmark
        else ((total - total_benchmark) / periods)
        / ((1 + total) ** (1 / periods) - (1 + total_benchmark) ** (1 / periods))
    )
    squares = float((excess**2).sum())
    alpha = (
        (total - total_benchmark - m * float(excess.sum())) / squares * excess
        if squares > 0
        else 0 * excess
    )
    return m + alpha


def get_linked_brinson_fachler_attribution(
    portfolio_weights: pd.DataFrame,
    portfolio_returns: pd.DataFrame,
    benchmark_weights: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
    linking: str = "carino",
) -> pd.DataFrame:
    """
    The Brinson-Fachler attribution over several sub-periods, linked into effects over the
    whole range.

    Every sub-period gets its own attribution with the weights at its start
    (get_brinson_fachler_attribution), and its effects are scaled with the linking
    coefficients of get_linking_coefficients and summed:

    - Linked Effect = Σ_t coefficient_t x Effect_t

    The linked Total Effects add up to the compounded portfolio return minus the compounded
    benchmark return over the whole range. A segment without a benchmark return in a
    sub-period (NaN) keeps its weight there, as in get_brinson_fachler_attribution: its
    compounded benchmark return is unknown, and so are the compounded benchmark return and,
    since every linking coefficient depends on it, all linked effects.

    Args:
        portfolio_weights (pd.DataFrame): The portfolio weight per sub-period (rows, at
            its start) and segment (columns).
        portfolio_returns (pd.DataFrame): The portfolio's return per sub-period and segment.
        benchmark_weights (pd.DataFrame): The benchmark weight per sub-period and segment.
        benchmark_returns (pd.DataFrame): The benchmark's return per sub-period and segment.
        linking (str, optional): The linking. Options are: "carino" (Cariño), "menchero" (Menchero).
            Defaults to "carino".

    Returns:
        pd.DataFrame: Per segment and in total the weights at the start of the first
        sub-period, the returns compounded over all sub-periods and the linked Allocation,
        Selection, Interaction and Total Effect.
    """
    if portfolio_weights.empty:
        raise ValueError("The linked attribution needs at least one sub-period.")

    tables = {
        period: get_brinson_fachler_attribution(
            portfolio_weights.loc[period].dropna(),
            portfolio_returns.loc[period].dropna(),
            benchmark_weights.loc[period].dropna(),
            benchmark_returns.loc[period].dropna(),
        )
        for period in portfolio_weights.index
    }
    totals = pd.DataFrame(
        {period: table.loc["Total"] for period, table in tables.items()}
    ).T
    # Every coefficient depends on the benchmark return compounded over all sub-periods.
    coefficients = (
        pd.Series(np.nan, index=totals.index)
        if totals["Benchmark Return"].isna().any()
        else get_linking_coefficients(
            totals["Portfolio Return"], totals["Benchmark Return"], linking
        )
    )

    # A segment missing from a sub-period (no weight in either) has no effect there.
    segments = pd.Index([])
    for table in tables.values():
        segments = segments.union(table.index.drop("Total"))
    segments = segments.append(pd.Index(["Total"]))
    effects = [
        "Allocation Effect",
        "Selection Effect",
        "Interaction Effect",
        "Total Effect",
    ]
    linked = sum(
        tables[period][effects].reindex(segments, fill_value=0) * coefficients[period]
        for period in portfolio_weights.index
    )

    # Returns compound over the sub-periods; a segment the portfolio never holds has none,
    # and a benchmark return unknown in one sub-period is unknown over the whole range.
    returns = {
        column: (
            pd.DataFrame({period: table[column] for period, table in tables.items()})
            .reindex(segments)
            .add(1)
            .prod(axis=1, min_count=1, skipna=column == "Portfolio Return")
            .sub(1)
        )
        for column in ["Portfolio Return", "Benchmark Return"]
    }
    first = tables[portfolio_weights.index[0]].reindex(segments)
    result = pd.DataFrame(
        {
            "Portfolio Weight": first["Portfolio Weight"].fillna(0),
            "Benchmark Weight": first["Benchmark Weight"].fillna(0),
            "Portfolio Return": returns["Portfolio Return"],
            "Benchmark Return": returns["Benchmark Return"],
        }
    )

    return pd.concat([result, linked], axis=1)


# The highest annual money-weighted return searched for. Annualized, a short holding period
# with an ordinary gain is an enormous rate (ten times the money in 14 days is about 1e26 a
# year), so the search reaches far beyond any realistic return.
MAXIMUM_MONEY_WEIGHTED_RETURN = 1e30

# The rates at which the net present value is checked for a change of sign when the cash
# flows allow more than one solution, evenly spaced in the growth factor 1 + r on a log scale:
# a step of about 1% in 1 + r from -99.99% to 1,000% a year, where returns that settle the
# same cash flows twice lie (10% and 12%, for instance), and coarser above it.
MONEY_WEIGHTED_RETURN_GRID = (
    np.concatenate(
        [
            np.geomspace(1e-4, 11, 1000, endpoint=False),
            np.geomspace(11, 1 + MAXIMUM_MONEY_WEIGHTED_RETURN, 200),
        ]
    )
    - 1
)


def get_money_weighted_return(cash_flows: pd.Series) -> float:
    """
    The money-weighted (internal) rate of return of dated cash flows, as an annual rate: the
    rate r for which the cash flows discounted with (1 + r)^(days / 365) sum to zero.

    Unlike the time-weighted return, it rewards investing more before good periods, so it
    measures the investor's own result rather than that of the assets.

    Cash flows that change sign more than once, such as a position bought, sold and bought
    again, or one that ends with a payment, can be settled by more than one rate (Descartes'
    rule of signs allows one per change of sign). None of these rates is the return of the
    investor, so the rates found are reported in the error rather than one of them returned.

    Args:
        cash_flows (pd.Series): The cash flows by date, from the investor's point of view:
            money invested is negative, money received (sales, dividends and the value
            still held at the end) positive.

    Returns:
        float: The annualized money-weighted return.

    Raises:
        ValueError: When the cash flows lack an investment or a receipt, when no rate
            between -99.99% and 1e30 a year makes them add up to zero, or when more than one
            rate does.
    """
    flows = cash_flows.groupby(level=0).sum()
    flows = flows[flows != 0]

    if (flows > 0).sum() == 0 or (flows < 0).sum() == 0:
        raise ValueError(
            "The money-weighted return needs money invested and money received."
        )

    dates = pd.to_datetime(flows.index.astype(str))
    years = ((dates - dates.min()).days / 365).to_numpy()
    amounts = flows.to_numpy(dtype=float)

    def net_present_value(rate):
        with np.errstate(over="ignore"):
            return float(np.sum(amounts / (1 + rate) ** years))

    # The number of solutions is at most the number of sign changes of the cash flows in time
    # (Descartes' rule of signs, which holds for any exponents), so with one change the
    # bracket from -99.99% to the maximum holds the only one.
    if (np.diff(np.sign(amounts)) != 0).sum() == 1:
        lower, upper = MONEY_WEIGHTED_RETURN_GRID[[0, -1]]
        solutions = (
            [brentq(net_present_value, lower, upper, maxiter=1000)]
            if np.sign(net_present_value(lower)) != np.sign(net_present_value(upper))
            else []
        )
    else:
        # Every change of sign between neighbouring rates of the grid holds a solution. A
        # rate near -100% over many years underflows (1 + r)^t, a high one overflows it, and
        # the values that are not finite are left out.
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            values = np.array(
                [
                    (amounts * np.exp(-np.log1p(block)[:, None] * years)).sum(axis=1)
                    for block in np.array_split(MONEY_WEIGHTED_RETURN_GRID, 12)
                ]
            ).ravel()
        finite = np.isfinite(values)
        rates, signs = MONEY_WEIGHTED_RETURN_GRID[finite], np.sign(values[finite])
        solutions = list(rates[signs == 0]) + [
            brentq(
                net_present_value, rates[position], rates[position + 1], maxiter=1000
            )
            for position in np.flatnonzero(signs[:-1] * signs[1:] < 0)
        ]

    if not solutions:
        raise ValueError(
            "No annual rate between -99.99% and 1e30 makes the cash flows add up to zero."
        )

    if len(solutions) > 1:
        raise ValueError(
            f"{len(solutions)} annual rates make the cash flows add up to zero "
            f"({', '.join(f'{solution:.2%}' for solution in sorted(solutions))}), as they "
            "change sign more than once, so the money-weighted return is not defined."
        )

    return float(solutions[0])


def get_scenario_returns(
    prices: pd.DataFrame, scenarios: dict[str, tuple[str, str]]
) -> pd.DataFrame:
    """
    The return of every asset over each scenario: from its last price on or before the
    scenario's start to its last price on or before the scenario's end.

    Args:
        prices (pd.DataFrame): The prices by date, one column per asset.
        scenarios (dict[str, tuple[str, str]]): The start and end date per scenario.

    Returns:
        pd.DataFrame: The return per scenario (rows) and asset (columns); NaN where an
        asset has no price at the start of a scenario.
    """
    prices = prices.sort_index()
    dates = pd.to_datetime(prices.index.astype(str))
    rows = {}

    for name, (start, end) in scenarios.items():
        start_prices = prices[dates <= pd.Timestamp(start)].ffill().iloc[-1:]
        end_prices = prices[dates <= pd.Timestamp(end)].ffill().iloc[-1:]
        if start_prices.empty or end_prices.empty:
            rows[name] = pd.Series(np.nan, index=prices.columns)
            continue
        rows[name] = end_prices.iloc[0] / start_prices.iloc[0] - 1

    return pd.DataFrame(rows).T


def get_stress_test(scenario_returns: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """
    The return of a buy-and-hold portfolio in each scenario, such as a past crisis: how much
    the portfolio would have gained or lost had it been held through it.

    Assets without data for a scenario (not yet listed) are left out and the remaining
    weights scaled up to a net weight of one; the Coverage column shows the share of the
    portfolio's gross exposure that had data, Σ |w_i| over the assets with data / Σ |w_i|,
    between 0 and 1. When the assets with data have a net weight of zero or less (only short
    positions, or a hedge without the positions it hedges), they cannot be scaled to the
    portfolio and the Portfolio Return is empty.

    Args:
        scenario_returns (pd.DataFrame): The return per scenario (rows) and asset (columns).
        weights (pd.Series): The weight per asset.

    Returns:
        pd.DataFrame: Per scenario the Portfolio Return and the Coverage.
    """
    weights = weights.reindex(scenario_returns.columns).fillna(0)
    available = scenario_returns.notna()
    coverage = available.mul(weights.abs(), axis=1).sum(axis=1) / weights.abs().sum()
    net_weight = available.mul(weights, axis=1).sum(axis=1)
    portfolio = scenario_returns.fillna(0).mul(weights, axis=1).sum(
        axis=1
    ) / net_weight.where(net_weight > 0)

    return pd.DataFrame({"Portfolio Return": portfolio, "Coverage": coverage})


def get_look_through_exposure(
    weights: pd.Series, exposures: dict[str, pd.Series]
) -> pd.Series:
    """
    The exposure of a portfolio after looking through its funds.

    Every asset's weight is spread over its own exposures (a fund's sector, country or
    holding weights, or the single sector or country of a stock) and summed.

    Args:
        weights (pd.Series): The weight per asset.
        exposures (dict[str, pd.Series]): Per asset, its exposure weights. Weights that sum
            to more than one are scaled down; a shortfall (e.g. a fund that lists only its
            largest holdings) is reported as "Other".

    Returns:
        pd.Series: The portfolio's exposure per sector, country or holding, largest first.
    """
    total = pd.Series(dtype=float)

    for asset, weight in weights.items():
        # An asset without weight adds nothing, not even an "Unclassified" row of 0.
        if weight == 0:
            continue
        exposure = exposures.get(str(asset))
        if exposure is None or exposure.empty:
            exposure = pd.Series({"Unclassified": 1.0})
        exposure = exposure[exposure > 0]
        covered = exposure.sum()
        if covered > 1:
            exposure = exposure / covered
        elif covered < 1 - 1e-3:
            # Holdings lists often show only the largest positions; the rest is kept apart
            # rather than spread over the positions shown.
            exposure = pd.concat([exposure, pd.Series({"Other": 1 - covered})])
        total = total.add(exposure * weight, fill_value=0)

    return total.sort_values(ascending=False).rename("Exposure")


def get_period_contributions(
    contributions: pd.DataFrame, frequency: str
) -> pd.DataFrame:
    """
    The contributions of the assets to the portfolio's return over longer periods, from
    their contributions per (shorter) period, c_i,t = w_i,t-1 r_i,t.

    Within each longer period the contributions are carried by the growth of the portfolio
    since the period began, C_i = Σ_t c_i,t Π_(s<t) (1 + R_s) with R_t = Σ_i c_i,t, so they
    are the contributions in money divided by the portfolio's value at the start of the
    period and add up to its compounded return, Π (1 + R_t) - 1.

    See Bacon, C. R. (2008), "Practical Portfolio Performance Measurement and Attribution",
    2nd edition, Wiley, chapter 5.

    Shares similar logic:
    - PerformanceAnalytics (R), to.period.contributions: https://github.com/cran/PerformanceAnalytics/blob/2.1.0/R/to.period.contributions.R#L45

    Args:
        contributions (pd.DataFrame): The contribution per asset (columns) per period, with
            a PeriodIndex.
        frequency (str): The frequency of the longer periods, e.g. "M" or "Q-DEC".

    Returns:
        pd.DataFrame: The contribution per asset and the Portfolio return per longer period.
    """
    portfolio = contributions.sum(axis=1)
    periods = contributions.index.asfreq(frequency)
    # The portfolio's growth from the start of its longer period up to each day before.
    growth = (1 + portfolio).groupby(periods).cumprod()
    carried = growth.groupby(periods).shift(1).fillna(1)
    table = contributions.mul(carried, axis=0).groupby(periods).sum()
    table["Portfolio"] = table.sum(axis=1)
    return table


def get_factor_attribution(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    loadings: pd.DataFrame,
    factor_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    The attribution of a portfolio's return per period to factors.

    How much of it the factor exposures earned and how much the assets earned on their own,
    the risk model attribution of Grinold and Kahn (2000). The portfolio's exposure to
    factor f is the weighted sum of the loadings of its assets, x_f,t = Σ_i w_i,t β_i,f, and

    - Contribution_f,t = x_f,t F_f,t, the return earned through factor f
    - Common Return_t = Σ_f Contribution_f,t
    - Specific Return_t = R_t - Common Return_t, with R_t = Σ_i w_i,t r_i,t
    - Tilt Return_t = Σ_f mean(x_f) F_f,t, the return of the average exposure
    - Timing Return_t = Common Return_t - Tilt Return_t, the return of moving the
      exposures around their average

    See Grinold, R. C. and Kahn, R. N. (2000), "Active Portfolio Management", 2nd edition,
    McGraw-Hill, chapter 17.

    Referenced against:
    - empyrical, perf_attrib: https://github.com/stefan-jansen/empyrical-reloaded/blob/0.7.3/src/empyrical/perf_attrib.py#L5

    Shares similar logic:
    - empyrical, compute_exposures: https://github.com/stefan-jansen/empyrical-reloaded/blob/0.7.3/src/empyrical/perf_attrib.py#L139

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset.
        weights (pd.DataFrame): The weight of every asset at the start of every period.
        loadings (pd.DataFrame): The loading of every asset (rows) on every factor
            (columns).
        factor_returns (pd.DataFrame): The factor returns per period.

    Returns:
        pd.DataFrame: Per period the contribution of every factor and the Common,
        Specific, Total, Tilt and Timing Return.
    """
    weights = weights.reindex(index=returns.index, columns=returns.columns).fillna(0)
    factor_returns = factor_returns.reindex(returns.index)
    exposures = weights @ loadings.reindex(
        index=returns.columns, columns=factor_returns.columns
    ).fillna(0)
    table = exposures * factor_returns
    total = (weights * returns.fillna(0)).sum(axis=1)
    common = table.sum(axis=1)
    tilt = factor_returns @ exposures.mean()
    table["Common Return"] = common
    table["Specific Return"] = total - common
    table["Total Return"] = total
    table["Tilt Return"] = tilt
    table["Timing Return"] = common - tilt
    return table
