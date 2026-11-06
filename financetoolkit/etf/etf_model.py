"""ETF Analysis Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd

# The valuation multiples of the look-through valuation, averaged as weighted harmonic means
# (the multiple of the fund as a whole), and the yields and returns, averaged arithmetically.
MULTIPLES = (
    "Price-to-Earnings",
    "Price-to-Book",
    "Price-to-Sales",
    "Price-to-Free-Cash-Flow",
)
YIELDS = ("Dividend Yield", "Net Profit Margin")
TOP_HOLDINGS = 10


def get_concentration(weights: pd.Series) -> pd.Series:
    """
    Calculate how concentrated a fund is from the weights of its holdings: the number of
    holdings, the weight of the largest holding and of the ten largest (shares of the fund),
    the Herfindahl-Hirschman index (the sum of the squared weights, after scaling the weights
    of the holdings to sum to one) and the effective number of holdings (one over that
    index). A fund of 500 holdings whose ten largest make up 40% of
    it behaves like a fund of far fewer stocks; the effective number says how many equally
    weighted holdings would give the same concentration.

    Args:
        weights (pd.Series): The weight of every holding, as decimals.

    Returns:
        pd.Series: The Number of Holdings, Largest Holding, Largest Weight, Top 10 Weight,
        Herfindahl-Hirschman Index and Effective Number of Holdings.
    """
    weights = weights.dropna()
    weights = weights[weights > 0].sort_values(ascending=False)
    if weights.empty:
        # A fund without weighted holdings has no concentration; the other funds still do.
        return pd.Series(
            {
                "Number of Holdings": np.nan,
                "Largest Holding": None,
                "Largest Weight": np.nan,
                "Top 10 Weight": np.nan,
                "Herfindahl-Hirschman Index": np.nan,
                "Effective Number of Holdings": np.nan,
            }
        )

    # The weights are scaled to the part of the fund the holdings cover, so cash and
    # holdings without a weight do not dilute the index.
    shares = weights / weights.sum()
    herfindahl = float((shares**2).sum())
    return pd.Series(
        {
            "Number of Holdings": float(len(weights)),
            "Largest Holding": weights.index[0],
            "Largest Weight": float(weights.iloc[0]),
            "Top 10 Weight": float(weights.iloc[:TOP_HOLDINGS].sum()),
            "Herfindahl-Hirschman Index": herfindahl,
            "Effective Number of Holdings": 1 / herfindahl,
        }
    )


def get_overlap(weights: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Calculate the overlap between every pair of funds: the sum, over the holdings they share,
    of the smaller of the two weights. Two funds with an overlap of 0.8 hold 80% of the same
    portfolio, so holding both adds little diversification.

    Args:
        weights (dict[str, pd.Series]): The weights of every fund's holdings, as decimals.

    Returns:
        pd.DataFrame: The overlap of every pair of funds, as decimals (1 on the diagonal for a
        fund whose weights add up to 1).
    """
    funds = list(weights)
    overlap = pd.DataFrame(np.nan, index=funds, columns=funds)
    for first in funds:
        for second in funds:
            # A fund without holdings data has an unknown overlap, not none.
            if weights[first].empty or weights[second].empty:
                continue
            common = weights[first].index.intersection(weights[second].index)
            overlap.loc[first, second] = float(
                np.minimum(weights[first][common], weights[second][common])
                .clip(lower=0)
                .sum()
            )
    return overlap


def get_active_share(weights: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Calculate the active share between every pair of funds: half the sum, over all holdings
    of either fund, of the absolute difference between their weights. It is the share of a
    fund's portfolio that differs from the other's, 0 for identical holdings and 1 for funds
    without a holding in common.

    The formula is as follows:

        - Active Share = 1/2 * SUM(|Weight Fund i - Weight Benchmark i|)

    For more information, see the following paper:

    - Cremers, K.J.M., & Petajisto, A. (2009). "How Active Is Your Fund Manager? A New
      Measure That Predicts Performance." Review of Financial Studies, 22(9), 3329-3365.

    Args:
        weights (dict[str, pd.Series]): The weights of every fund's holdings, as decimals.

    Returns:
        pd.DataFrame: The active share of every fund (rows) against every other fund
        (columns), as decimals. The weights of each fund are scaled to add up to one, so an
        incomplete list of holdings is compared by its proportions.
    """
    funds = list(weights)
    active_share = pd.DataFrame(np.nan, index=funds, columns=funds)
    for first in funds:
        for second in funds:
            first_total = weights[first].sum()
            second_total = weights[second].sum()
            # A fund without holdings data has an unknown active share, not none.
            if first_total <= 0 or second_total <= 0:
                continue
            first_weights, second_weights = (weights[first] / first_total).align(
                weights[second] / second_total, fill_value=0
            )
            active_share.loc[first, second] = float(
                (first_weights - second_weights).abs().sum() / 2
            )
    return active_share


def get_look_through_valuation(weights: pd.Series, ratios: pd.DataFrame) -> pd.Series:
    """
    Calculate the valuation of a fund from the valuation of its holdings, weighted by their
    weight in the fund. The multiples (price-to-earnings, price-to-book, price-to-sales
    and price-to-free-cash-flow) are weighted harmonic means, 1 / Σ w / multiple, which is the
    multiple of the fund as a whole (its price over its share of the holdings' earnings) and does not let one holding
    with a very high multiple dominate; holdings with negative earnings count with their
    negative earnings, as in the fund's own earnings. The dividend yield and net profit margin
    are weighted arithmetic means. The coverage is the share of the fund's
    weight whose holdings have a usable, non-zero price-to-earnings ratio. When the holdings' earnings
    add up to a loss, the fund's price-to-earnings ratio is negative (some data providers
    show none in that case).

    Args:
        weights (pd.Series): The weight of every holding, as decimals.
        ratios (pd.DataFrame): The ratios of the holdings, a row per holding and a column per
            ratio (see MULTIPLES and YIELDS).

    Returns:
        pd.Series: Every ratio of the fund, and its Coverage.
    """
    common = weights.index.intersection(ratios.index)
    weights = weights[common]
    result: dict[str, float] = {}
    coverage = np.nan
    for ratio in [*MULTIPLES, *YIELDS]:
        if ratio not in ratios:
            continue
        values = pd.to_numeric(ratios.loc[common, ratio], errors="coerce")
        # A multiple of zero cannot be inverted; a yield of zero (no dividend) counts.
        valid = values.notna() & ((values != 0) if ratio in MULTIPLES else True)
        covered = weights[valid]
        # The coverage is that of the first multiple available.
        if ratio in MULTIPLES and np.isnan(coverage) and weights.sum():
            coverage = float(covered.sum())
        if covered.sum() <= 0:
            result[ratio] = np.nan
            continue
        shares = covered / covered.sum()
        if ratio in MULTIPLES:
            inverse = float((shares / values[valid]).sum())
            result[ratio] = 1 / inverse if inverse else np.nan
        else:
            result[ratio] = float((shares * values[valid]).sum())
    result["Coverage"] = coverage
    return pd.Series(result)


def get_price_impact(
    returns: pd.Series, dollar_volume: pd.Series, amount: float = 1e8
) -> float:
    """
    The Amihud (2002) illiquidity, scaled to an amount traded: the average absolute daily
    return per dollar traded, times the amount, so the average price move that trading that
    amount in a day goes with. The lower, the more liquid.

    For more information, see the following paper:

    - Amihud, Y. (2002). "Illiquidity and Stock Returns: Cross-Section and Time-Series
      Effects." Journal of Financial Markets, 5(1), 31-56.

    Args:
        returns (pd.Series): The daily returns.
        dollar_volume (pd.Series): The daily dollar volume.
        amount (float, optional): The amount traded. Defaults to 100 million.

    Returns:
        float: The price impact as a decimal.
    """
    ratio = returns.abs() / dollar_volume.where(dollar_volume > 0)
    return float(ratio.mean() * amount)


def get_tracking(
    returns: pd.Series, benchmark_returns: pd.Series, periods_per_year: int = 252
) -> pd.Series:
    """
    Calculate how closely a fund tracks a benchmark (its index): the tracking difference
    (the annualized return of the fund minus the annualized return of the benchmark over the
    period, which for an index fund should be about minus its expense ratio), the annualized
    tracking error (the volatility of the per-period return difference), the correlation, the beta and
    the R Squared.

    Args:
        returns (pd.Series): The fund's returns.
        benchmark_returns (pd.Series): The benchmark's returns.
        periods_per_year (int, optional): The returns per year. Defaults to 252.

    Returns:
        pd.Series: The Tracking Difference, Tracking Error, Correlation, Beta and R Squared.
    """
    data = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(data) < 2:  # noqa: PLR2004
        raise ValueError("At least two common returns are needed to measure tracking.")
    fund, benchmark = data.iloc[:, 0], data.iloc[:, 1]
    difference = fund - benchmark
    correlation = float(fund.corr(benchmark))
    return pd.Series(
        {
            "Tracking Difference": float(
                (1 + fund).prod() ** (periods_per_year / len(data))
                - (1 + benchmark).prod() ** (periods_per_year / len(data))
            ),
            "Tracking Error": float(difference.std(ddof=1) * np.sqrt(periods_per_year)),
            "Correlation": correlation,
            "Beta": float(fund.cov(benchmark) / benchmark.var()),
            "R Squared": correlation**2,
        }
    )


def get_contribution(
    weights: pd.Series,
    start_prices: pd.Series,
    end_prices: pd.Series,
    weight_date_prices: pd.Series,
    fund_change: float,
) -> pd.DataFrame:
    """
    Calculate the contribution of every holding to a fund's return over a period: its weight
    at the start of the period times its return over the period, so the contributions add up
    to the return of the part of the fund the holdings cover.

    The weights are those of a later date (a fund publishes only its current holdings). They
    are rolled back to the start of the period with every holding's own price change between
    the two dates, relative to the fund's: w_start = w × (P_start / P_weights) × (F_weights /
    F_start), with F the fund's price. That is exact for a fund that kept the same number of
    shares of each holding (and the same cash), as a market capitalization weighted index does
    between its rebalancings.

    Args:
        weights (pd.Series): The weight of every holding on the date of the holdings, as decimals.
        start_prices (pd.Series): The price of every holding at the start of the period.
        end_prices (pd.Series): The price of every holding at the end of the period.
        weight_date_prices (pd.Series): The price of every holding on the date of the holdings.
        fund_change (float): The fund's price on the date of the holdings over its price at the
            start of the period, F_weights / F_start.

    Returns:
        pd.DataFrame: The Start Weight, Return and Contribution of every holding with a price on
        all three dates, as decimals.
    """
    prices = pd.concat(
        [start_prices, end_prices, weight_date_prices],
        axis=1,
        keys=["start", "end", "weights"],
    ).reindex(weights.index)
    priced = prices.notna().all(axis=1) & (prices > 0).all(axis=1)
    weights, prices = weights[priced], prices[priced]

    start_weights = weights * prices["start"] / prices["weights"] * fund_change
    returns = prices["end"] / prices["start"] - 1

    return pd.DataFrame(
        {
            "Start Weight": start_weights,
            "Return": returns,
            "Contribution": start_weights * returns,
        }
    )
