"""Portfolio Attribution Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd
from scipy.optimize import brentq

# pylint: disable=too-many-arguments

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
    The Brinson-Fachler attribution of a portfolio's excess return over its benchmark into
    allocation, selection and interaction effects per segment (Brinson and Fachler, 1985;
    Brinson, Hood and Beebower, 1986):

    - Allocation = (w_p - w_b) x (r_b - R_b): the reward for over- or underweighting
      segments that beat or lag the benchmark as a whole.
    - Selection = w_b x (r_p - r_b): the reward for picking better assets within a segment.
    - Interaction = (w_p - w_b) x (r_p - r_b): the joint effect of both decisions.

    Here w are segment weights, r segment returns and R_b the benchmark's total return. The
    three effects add up to the portfolio return minus the benchmark return.

    Args:
        portfolio_weights (pd.Series): The portfolio weight per segment.
        portfolio_returns (pd.Series): The portfolio's return per segment.
        benchmark_weights (pd.Series): The benchmark weight per segment.
        benchmark_returns (pd.Series): The benchmark's return per segment.

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
    # segment return there leaves only its allocation effect, as the method intends.
    rp = portfolio_returns.reindex(segments).where(wp > 0, rb)

    if rb[wb > 0].isna().any():
        missing = ", ".join(rb[(wb > 0) & rb.isna()].index)
        raise ValueError(f"The benchmark has no return for: {missing}.")
    total_benchmark = float((wb * rb.fillna(0)).sum())
    # A segment outside the benchmark is compared with the benchmark as a whole, so it
    # adds no allocation effect and its result shows as interaction.
    rb = rb.fillna(total_benchmark)
    rp = rp.fillna(rb)

    allocation = (wp - wb) * (rb - total_benchmark)
    selection = wb * (rp - rb)
    interaction = (wp - wb) * (rp - rb)
    total = allocation + selection + interaction

    result = pd.DataFrame(
        {
            "Portfolio Weight": wp,
            "Benchmark Weight": wb,
            "Portfolio Return": rp.where(wp > 0),
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
        float((wp * rp).sum()),
        total_benchmark,
        allocation.sum(),
        selection.sum(),
        interaction.sum(),
        total.sum(),
    ]

    return result


def get_money_weighted_return(cash_flows: pd.Series) -> float:
    """
    The money-weighted (internal) rate of return of dated cash flows, as an annual rate:
    the rate r for which the cash flows discounted with (1 + r)^(days / 365) sum to zero.
    Unlike the time-weighted return, it rewards investing more before good periods, so it
    measures the investor's own result rather than that of the assets.

    Args:
        cash_flows (pd.Series): The cash flows by date, from the investor's point of view:
            money invested is negative, money received (sales, dividends and the value
            still held at the end) positive.

    Returns:
        float: The annualized money-weighted return.
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
        return float(np.sum(amounts / (1 + rate) ** years))

    return float(brentq(net_present_value, -0.9999, 100, maxiter=1000))


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
    The return of a buy-and-hold portfolio in each scenario. Assets without data for a
    scenario (not yet listed) are left out and the remaining weights scaled up; the
    Coverage column shows the share of the portfolio that had data.

    Args:
        scenario_returns (pd.DataFrame): The return per scenario (rows) and asset (columns).
        weights (pd.Series): The weight per asset.

    Returns:
        pd.DataFrame: Per scenario the Portfolio Return and the Coverage.
    """
    weights = weights.reindex(scenario_returns.columns).fillna(0)
    available = scenario_returns.notna()
    coverage = available.mul(weights, axis=1).sum(axis=1)
    portfolio = scenario_returns.fillna(0).mul(weights, axis=1).sum(axis=1) / coverage

    return pd.DataFrame(
        {"Portfolio Return": portfolio.where(coverage > 0), "Coverage": coverage}
    )


def get_look_through_exposure(
    weights: pd.Series, exposures: dict[str, pd.Series]
) -> pd.Series:
    """
    The exposure of a portfolio after looking through its funds: every asset's weight is
    spread over its own exposures (a fund's sector, country or holding weights, or the
    single sector or country of a stock) and summed.

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
        exposure = exposures.get(str(asset))
        if exposure is None or exposure.empty or weight == 0:
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
