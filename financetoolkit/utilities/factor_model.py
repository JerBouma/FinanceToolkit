"""Factor Model"""

import inspect

import numpy as np
import pandas as pd
from scipy import stats

from financetoolkit.ratios.helpers import map_period_data_to_daily_data
from financetoolkit.utilities.logger_model import get_logger

logger = get_logger()

# The fewest tickers a period needs for its values to be ranked against each other.
MINIMUM_TICKERS = 3

# The labels of the forward return horizons, per period frequency.
HORIZON_UNITS = {
    "daily": "Day",
    "weekly": "Week",
    "monthly": "Month",
    "quarterly": "Quarter",
    "yearly": "Year",
}

# The forward return horizons per period frequency when none are given: about a period, a
# quarter and a year ahead.
DEFAULT_HORIZONS = {
    "daily": [1, 5, 21],
    "weekly": [1, 4, 13],
    "monthly": [1, 3, 12],
    "quarterly": [1, 2, 4],
    "yearly": [1, 2, 3],
}

# The parts of the analysis that detail= returns for a single metric.
DETAILS = (
    "information_coefficient",
    "quantile_returns",
    "quantile_turnover",
    "rank_autocorrelation",
)

# The price columns some indicators return next to their own values, never a factor.
PRICE_COLUMNS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")

# Pandas period frequencies of the evaluation periods.
PERIOD_FREQUENCIES = {
    "daily": "D",
    "weekly": "W",
    "monthly": "M",
    "quarterly": "Q",
    "yearly": "Y",
}


def get_horizon_label(horizon: int, period: str) -> str:
    """
    Returns the label of a forward return horizon, e.g. "1 Month" or "3 Months".

    Args:
        horizon (int): The number of periods the forward return covers.
        period (str): The period frequency: daily, weekly, monthly, quarterly or yearly.

    Returns:
        str: The label of the horizon.
    """
    unit = HORIZON_UNITS[period]

    return f"{horizon} {unit}" if horizon == 1 else f"{horizon} {unit}s"


def sample_by_period(data: pd.DataFrame, period: str) -> pd.DataFrame:
    """
    Takes the last value of each period per column, e.g. the factor or the price on the last
    trading day of each month. A column without a value in a period stays empty (NaN) for that
    period, so a ticker that is not yet listed or no longer listed is left out of it rather
    than carrying an older value.

    Args:
        data (pd.DataFrame): Daily values with the dates as index and the tickers as columns.
        period (str): The period frequency: daily, weekly, monthly, quarterly or yearly.

    Returns:
        pd.DataFrame: The value at the end of each period with a PeriodIndex.
    """
    index = data.index
    dates = (
        index.to_timestamp()
        if isinstance(index, pd.PeriodIndex)
        else pd.DatetimeIndex(index)
    )
    data = data.set_axis(dates.normalize(), axis=0)

    periods = data.index.to_period(PERIOD_FREQUENCIES[period])

    # groupby().last() takes the last non-missing value of each column within the period.
    return data.groupby(periods).last()


def get_forward_returns(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Calculates the return from the close of each period to the close horizon periods later,
    per ticker over its own periods with a price. The return of a period therefore starts
    after the factor of that period is known, so it can be compared with the factor without
    look-ahead.

    Args:
        prices (pd.DataFrame): The (dividend and split adjusted) prices with the periods as
            index and the tickers as columns.
        horizon (int): The number of periods the return covers.

    Returns:
        pd.DataFrame: The forward returns on the index of the prices, NaN for the last
        horizon periods of each ticker.
    """
    forward_returns = {}

    for ticker in prices.columns:
        # Each ticker counts its own periods with a price, so a period another ticker trades
        # in but it does not (a holiday on its exchange) does not cut its return short.
        price = prices[ticker].dropna()
        forward_returns[ticker] = (price.shift(-horizon) / price - 1).reindex(
            prices.index
        )

    return pd.DataFrame(forward_returns, index=prices.index, columns=prices.columns)


def get_row_correlation(
    first: pd.DataFrame, second: pd.DataFrame, minimum_observations: int = 3
) -> pd.Series:
    """
    Calculates the Pearson correlation between two DataFrames per row, over the columns
    where both have a value.

    Args:
        first (pd.DataFrame): The first values, with the dates as index.
        second (pd.DataFrame): The second values on the same index and columns.
        minimum_observations (int, optional): The fewest columns with both values for a row to
            get a correlation. Defaults to 3.

    Returns:
        pd.Series: The correlation per row, NaN for a row with too few values or without
        variation.
    """
    both = first.notna() & second.notna()
    first, second = first.where(both), second.where(both)

    first_deviation = first.sub(first.mean(axis=1), axis=0)
    second_deviation = second.sub(second.mean(axis=1), axis=0)

    covariance = (first_deviation * second_deviation).sum(axis=1, min_count=1)
    scale = np.sqrt(
        (first_deviation**2).sum(axis=1, min_count=1)
        * (second_deviation**2).sum(axis=1, min_count=1)
    )

    correlation = covariance / scale.replace(0, np.nan)

    return correlation.where(both.sum(axis=1) >= minimum_observations)


def get_information_coefficient(
    factor: pd.DataFrame, forward_returns: pd.DataFrame
) -> pd.Series:
    """
    Calculates the information coefficient (IC) per date: the Spearman rank correlation
    between the factor of the tickers on that date and their forward returns (Grinold and
    Kahn, Active Portfolio Management, 2000). Only tickers with both a factor and a forward
    return on a date are ranked, as in alphalens' factor_information_coefficient.

    Args:
        factor (pd.DataFrame): The factor with the dates as index and the tickers as columns.
        forward_returns (pd.DataFrame): The forward returns on the same index and columns.

    Returns:
        pd.Series: The information coefficient per date, NaN for a date with fewer than three
        tickers with both values.
    """
    both = factor.notna() & forward_returns.notna()

    # The Spearman correlation is the Pearson correlation of the ranks.
    factor_ranks = factor.where(both).rank(axis=1)
    return_ranks = forward_returns.where(both).rank(axis=1)

    return get_row_correlation(factor_ranks, return_ranks)


def summarize_information_coefficient(
    information_coefficient: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarizes the information coefficients per horizon as alphalens' information table
    does: the mean, the standard deviation (ddof=1), their ratio (the IC information ratio,
    the risk-adjusted IC), the t-statistic of the mean and its two-sided p-value, the share
    of dates with a positive IC and the number of dates.

    Args:
        information_coefficient (pd.DataFrame): The IC per date (index) and horizon (columns).

    Returns:
        pd.DataFrame: The statistics (index) per horizon (columns).
    """
    observations = information_coefficient.count()
    mean = information_coefficient.mean()
    standard_deviation = information_coefficient.std(ddof=1)
    # A constant IC (a perfect factor on every date) has no deviation, and no ratio.
    standard_deviation = standard_deviation.replace(0, np.nan)
    t_statistic = mean / (standard_deviation / np.sqrt(observations))
    p_value = pd.Series(
        2 * stats.t.sf(t_statistic.abs(), observations - 1),
        index=information_coefficient.columns,
    )
    positive_share = (information_coefficient > 0).sum() / observations.replace(
        0, np.nan
    )

    return pd.DataFrame(
        {
            "Mean IC": mean,
            "IC Standard Deviation": standard_deviation,
            "IC Information Ratio": mean / standard_deviation,
            "t-Statistic": t_statistic,
            "p-Value": p_value,
            "Positive Share": positive_share,
            "Observations": observations.astype(float),
        }
    ).T


def assign_quantiles(factor: pd.DataFrame, quantiles: int) -> pd.DataFrame:
    """
    Sorts the tickers of each date into equally sized quantiles of the factor, 1 for the
    lowest values and quantiles for the highest. Equal factor values are split by the order
    of the tickers, so each quantile holds as close to the same number of tickers as
    possible. A date with fewer tickers with a factor than quantiles gets no quantiles.

    Args:
        factor (pd.DataFrame): The factor with the dates as index and the tickers as columns.
        quantiles (int): The number of quantiles.

    Returns:
        pd.DataFrame: The quantile (1 to quantiles) of each ticker per date, NaN where the
        ticker has no factor or the date too few tickers.
    """
    ranks = factor.rank(axis=1, method="first")
    count = factor.notna().sum(axis=1)

    assignment = np.ceil(ranks.mul(quantiles).div(count.replace(0, np.nan), axis=0))

    return assignment.where(count >= quantiles, axis=0)


def get_quantile_returns(
    quantile_assignment: pd.DataFrame, forward_returns: pd.DataFrame
) -> pd.Series:
    """
    Calculates the mean forward return of each quantile, as the average over the dates of
    the equally weighted return of the tickers in the quantile, and the top minus bottom
    spread, the average over the dates of the highest quantile's return minus the lowest
    quantile's return (alphalens' mean_return_by_quantile and the long-short spread).

    Args:
        quantile_assignment (pd.DataFrame): The quantile of each ticker per date.
        forward_returns (pd.DataFrame): The forward returns on the same index and columns.

    Returns:
        pd.Series: The mean forward return of "Quantile 1" to "Quantile n" and the
        "Top Minus Bottom" spread.
    """
    quantile_numbers = sorted(
        int(value) for value in pd.unique(quantile_assignment.stack().dropna())
    )

    returns_per_date = pd.DataFrame(
        {
            f"Quantile {number}": forward_returns.where(
                quantile_assignment == number
            ).mean(axis=1)
            for number in quantile_numbers
        }
    )

    mean_returns = returns_per_date.mean()

    if quantile_numbers:
        spread = (
            returns_per_date[f"Quantile {quantile_numbers[-1]}"]
            - returns_per_date[f"Quantile {quantile_numbers[0]}"]
        )
        mean_returns["Top Minus Bottom"] = spread.mean()

    return mean_returns


def get_quantile_turnover(
    quantile_assignment: pd.DataFrame, lag: int = 1
) -> pd.DataFrame:
    """
    Calculates the turnover of each quantile per date: the share of the tickers in the
    quantile that were not in it lag dates before (alphalens' quantile_turnover). A date
    whose earlier date had no quantiles, such as the first lag dates, has no turnover.

    Args:
        quantile_assignment (pd.DataFrame): The quantile of each ticker per date.
        lag (int, optional): The number of dates to compare with. Defaults to 1.

    Returns:
        pd.DataFrame: The turnover per date (index) and quantile (columns).
    """
    quantile_numbers = sorted(
        int(value) for value in pd.unique(quantile_assignment.stack().dropna())
    )
    earlier_assigned = (
        quantile_assignment.notna().any(axis=1).shift(lag, fill_value=False)
    )

    turnover = {}

    for number in quantile_numbers:
        members = quantile_assignment == number
        earlier_members = members.shift(lag, fill_value=False)

        size = members.sum(axis=1)
        stayed = (members & earlier_members).sum(axis=1)

        turnover[f"Quantile {number}"] = (1 - stayed / size.replace(0, np.nan)).where(
            earlier_assigned
        )

    return pd.DataFrame(turnover, index=quantile_assignment.index)


def get_factor_rank_autocorrelation(factor: pd.DataFrame, lag: int = 1) -> pd.Series:
    """
    Calculates the factor rank autocorrelation per date: the correlation between the ranks
    of the tickers' factor on that date and lag dates before, over the tickers with a factor
    on both (alphalens' factor_rank_autocorrelation). A value close to one means a stable
    ranking and so a low turnover of a portfolio built on it.

    Args:
        factor (pd.DataFrame): The factor with the dates as index and the tickers as columns.
        lag (int, optional): The number of dates to compare with. Defaults to 1.

    Returns:
        pd.Series: The rank autocorrelation per date, NaN for the first lag dates.
    """
    ranks = factor.rank(axis=1)

    return get_row_correlation(ranks, ranks.shift(lag))


def get_horizons(period: str, horizons: list[int] | int | None) -> list[int]:
    """
    Returns the forward return horizons as a list of positive numbers of periods.

    Args:
        period (str): The period frequency, which sets the default horizons.
        horizons (list[int] | int | None): The horizons given, if any.

    Returns:
        list[int]: The horizons.
    """
    if period not in DEFAULT_HORIZONS:
        raise ValueError("Period must be daily, weekly, monthly, quarterly or yearly.")

    if horizons is None:
        return DEFAULT_HORIZONS[period]

    horizon_list = [horizons] if isinstance(horizons, int) else list(horizons)

    if not horizon_list or any(int(horizon) < 1 for horizon in horizon_list):
        raise ValueError("Horizons must be whole numbers of periods of at least 1.")

    return [int(horizon) for horizon in horizon_list]


def select_metrics(
    metrics: str | list[str], collections: dict[str, list[str]]
) -> list[str]:
    """
    Returns the names of the metrics to analyse: a collection is replaced by its metrics,
    and a text may name several metrics or collections separated by commas.

    Args:
        metrics (str | list[str]): The metric names and collections.
        collections (dict[str, list[str]]): The metrics of each collection of the module.

    Returns:
        list[str]: The metric names, each once, in the order given.
    """
    if isinstance(metrics, str):
        metrics = [metric.strip() for metric in metrics.split(",") if metric.strip()]

    names: list[str] = []
    for metric in metrics:
        for name in collections.get(metric, [metric]):
            if name not in names:
                names.append(name)

    if not names:
        raise ValueError("Please name a metric or a collection to analyse.")

    return names


def get_daily_factors(
    module: object,
    names: list[str],
    parameters: dict,
    defaults: dict,
    tickers: list[str],
    column: str | None = None,
    period_end_dates: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Calculates each named metric of a module and returns it per day and ticker, as the
    factor to analyse.

    Each metric is called unrounded with the parameters it takes: the module's defaults
    (such as show_daily for the ratios) and the parameters given, which must be taken by at least one
    of the metrics so that none is silently ignored. A metric that returns no data, such as
    one missing a statement line item, is left out with a warning in the log of its own.

    Args:
        module (object): The module (Ratios, Technicals, Models, Risk or Performance).
        names (list[str]): The metric (method) names.
        parameters (dict): The parameters given for the metrics, e.g. window=14.
        defaults (dict): The module's own parameters, passed to the metrics that take them.
        tickers (list[str]): The tickers to compare.
        column (str | None, optional): The output of a metric with several, see
            to_daily_factor. Defaults to None.
        period_end_dates (pd.DataFrame | None, optional): The day from which each statement
            period's figures apply per ticker. Defaults to None.

    Returns:
        dict[str, pd.DataFrame]: The daily factor of each metric that returned data.
    """
    methods = {}
    unused = set(parameters)
    for name in names:
        method = getattr(module, name, None)
        if (
            method is None
            or not name.startswith("get_")
            or name == "get_factor_analysis"
        ):
            raise ValueError(
                f"{name} is not a metric of the {type(module).__name__} module."
            )
        methods[name] = method
        unused -= set(inspect.signature(method).parameters)

    if unused:
        raise ValueError(
            f"{', '.join(sorted(unused))} is not a parameter of {', '.join(names)}."
        )

    # Unrounded values, as rounding ties small values (an Amihud illiquidity of 1e-10 rounds
    # to 0) and so loses the ranking the analysis is about.
    arguments = {"rounding": 15, **defaults, **parameters}

    factors = {}
    for name, method in methods.items():
        accepted = inspect.signature(method).parameters
        values = method(
            **{key: value for key, value in arguments.items() if key in accepted}
        )
        if values.empty:
            continue
        factors[name] = to_daily_factor(values, tickers, column, period_end_dates)

    return factors


def to_daily_factor(
    values: pd.DataFrame | pd.Series,
    tickers: list[str],
    column: str | None = None,
    period_end_dates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Turns the output of a metric into a factor per day (index) and ticker (columns), without
    look-ahead.

    A metric with the tickers as rows and statement periods as columns (a ratio or a model)
    applies from the day each period's figures are known: the fiscal period end, or the
    filing day with point_in_time (period_end_dates). A metric per period and ticker (a risk
    or performance metric per month) applies from the last day of its period, and a daily
    metric (a technical indicator) from its own day.

    A metric with several outputs gives one: column names it, e.g. "Lower Band" of the
    Bollinger Bands or "Altman Z-Score" of the Altman Z-Score model. Without it, an indicator
    gives its first output other than a price (e.g. "MACD Line") and a model its last row,
    which is its score.

    Args:
        values (pd.DataFrame | pd.Series): The output of the metric.
        tickers (list[str]): The tickers to compare; the Benchmark and Portfolio never are.
        column (str | None, optional): The output to use of a metric with several. Defaults
            to None.
        period_end_dates (pd.DataFrame | None, optional): The day from which each statement
            period's figures apply per ticker. Defaults to None, which uses the end of each
            labelled period.

    Returns:
        pd.DataFrame: The factor per day and ticker.
    """
    if isinstance(values, pd.Series):
        raise ValueError(
            "The metric gives one value per ticker for the whole range, which cannot be "
            "followed over time; choose a metric with a value per period."
        )

    if values.columns.nlevels > 1:
        outputs = list(dict.fromkeys(values.columns.get_level_values(0)))
        choice = column or next(
            (output for output in outputs if output not in PRICE_COLUMNS), outputs[0]
        )
        if choice not in outputs:
            raise ValueError(f"column must be one of {', '.join(map(str, outputs))}.")
        values = values[choice]
    elif values.index.nlevels > 1:
        outputs = list(dict.fromkeys(values.index.get_level_values(1)))
        choice = column or outputs[-1]
        if choice not in outputs:
            raise ValueError(f"column must be one of {', '.join(map(str, outputs))}.")
        values = values.xs(choice, level=1)

    tickers = [ticker for ticker in tickers if ticker not in ("Benchmark", "Portfolio")]

    if values.index.isin(tickers).any() and not values.columns.isin(tickers).any():
        periods = values.columns
        quarterly = isinstance(periods, pd.PeriodIndex) and periods.freqstr.startswith(
            "Q"
        )
        statement = values.loc[values.index.isin(tickers)]
        start = (
            periods.min().to_timestamp()
            if isinstance(periods, pd.PeriodIndex)
            else pd.Timestamp(periods.min())
        )
        values = map_period_data_to_daily_data(
            period_data=statement,
            daily_dates=pd.date_range(start, pd.Timestamp.today().normalize()),
            quarterly=quarterly,
            period_end_dates=period_end_dates,
        )
    elif isinstance(values.index, pd.PeriodIndex):
        # A value per period is known once its period has ended.
        values = values.set_axis(
            values.index.to_timestamp(how="end").normalize(), axis=0
        )

    values = values.loc[:, values.columns.isin(tickers)]
    values.index = pd.DatetimeIndex(values.index)

    return values.astype(float)


def analyse_factor(
    factor: pd.DataFrame,
    prices: pd.DataFrame,
    period: str,
    horizons: list[int],
    quantiles: int,
    periods_apart: int,
) -> dict[str, pd.DataFrame | pd.Series] | None:
    """
    Analyses one factor: the information coefficient per period and horizon, the mean
    forward return per quantile, the quantile turnover and the rank autocorrelation.
    A factor that no period has for at least three tickers can't be ranked and gives None.

    Args:
        factor (pd.DataFrame): The daily factor with the dates as index and tickers as columns.
        prices (pd.DataFrame): The daily (dividend and split adjusted) prices of the tickers.
        period (str): The frequency at which the factor is evaluated.
        horizons (list[int]): The number of periods the forward returns cover.
        quantiles (int): The number of quantiles.
        periods_apart (int): The periods between the dates the turnover and the rank
            autocorrelation compare.

    Returns:
        dict[str, pd.DataFrame | pd.Series] | None: The four parts, keyed as in DETAILS, or
            None when too few tickers have a value.
    """
    prices = prices.loc[:, prices.columns.isin(factor.columns)]
    sampled_prices = sample_by_period(prices, period)
    sampled_factor = sample_by_period(factor[prices.columns], period).reindex(
        sampled_prices.index
    )
    # Periods without any factor (before the start date) only serve as prices.
    sampled_factor = sampled_factor.loc[sampled_factor.notna().any(axis=1)]

    if sampled_factor.notna().sum(axis=1).max() < MINIMUM_TICKERS:
        return None

    information_coefficient = pd.DataFrame(
        {
            get_horizon_label(horizon, period): get_information_coefficient(
                sampled_factor,
                get_forward_returns(sampled_prices, horizon).reindex(
                    sampled_factor.index
                ),
            )
            for horizon in horizons
        },
        index=sampled_factor.index,
    )
    quantile_assignment = assign_quantiles(sampled_factor, quantiles)
    quantile_returns = pd.DataFrame(
        {
            get_horizon_label(horizon, period): get_quantile_returns(
                quantile_assignment,
                get_forward_returns(sampled_prices, horizon).reindex(
                    sampled_factor.index
                ),
            )
            for horizon in horizons
        }
    )

    return {
        "information_coefficient": information_coefficient,
        "quantile_returns": quantile_returns,
        "quantile_turnover": get_quantile_turnover(quantile_assignment, periods_apart),
        "rank_autocorrelation": get_factor_rank_autocorrelation(
            sampled_factor, periods_apart
        ),
    }


def get_factor_analysis(
    factors: dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    period: str = "monthly",
    horizons: list[int] | int | None = None,
    quantiles: int = 5,
    periods_apart: int = 1,
    detail: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Analyses the predictive power of each factor: a summary row per factor and horizon, or
    with detail one part of the analysis of a single factor.

    The summary has the mean information coefficient (IC), its standard deviation (ddof=1),
    the IC information ratio, the t-statistic of the mean with its p-value, the share of
    periods with a positive IC, the number of periods, the top minus bottom quantile spread,
    the mean turnover of the top and bottom quantiles and the mean rank autocorrelation. The
    turnover and the rank autocorrelation do not depend on the horizon.

    Args:
        factors (dict[str, pd.DataFrame]): The daily factor (dates × tickers) per name.
        prices (pd.DataFrame): The daily (dividend and split adjusted) prices of the tickers.
        period (str, optional): The frequency at which the factors are evaluated. Defaults
            to "monthly".
        horizons (list[int] | int | None, optional): The number of periods the forward
            returns cover. Defaults to None, which uses DEFAULT_HORIZONS.
        quantiles (int, optional): The number of quantiles. Defaults to 5.
        periods_apart (int, optional): The periods between the dates the turnover and the
            rank autocorrelation compare. Defaults to 1.
        detail (str | None, optional): One of DETAILS to return that part for a single
            factor. Defaults to None, which returns the summary.
        start_date (str | None, optional): The first date a factor is evaluated on. Defaults
            to None.
        end_date (str | None, optional): The last date a factor is evaluated on. Defaults
            to None.

    Returns:
        pd.DataFrame: The summary per factor and horizon (index), or the detail.
    """
    horizon_list = get_horizons(period, horizons)

    if detail is not None and detail not in DETAILS:
        raise ValueError(f"detail must be one of {', '.join(DETAILS)}.")
    if detail is not None and len(factors) != 1:
        raise ValueError(
            "detail shows one metric at a time; name a single metric instead of "
            f"{', '.join(factors) or 'none'}."
        )
    if not factors:
        raise ValueError("None of the metrics returned data to analyse.")

    analyses = {
        name: analyse_factor(
            factor.loc[start_date:end_date],
            prices,
            period,
            horizon_list,
            quantiles,
            periods_apart,
        )
        for name, factor in factors.items()
    }

    # A metric most tickers lack (e.g. inventory turnover for companies without inventory)
    # can't be ranked; it is left out of a collection rather than failing the others.
    thin = [name for name, analysis in analyses.items() if analysis is None]
    analyses = {
        name: analysis for name, analysis in analyses.items() if analysis is not None
    }

    if not analyses:
        raise ValueError(
            f"{', '.join(thin)} {'has' if len(thin) == 1 else 'have'} a value for fewer than "
            f"{MINIMUM_TICKERS} tickers in every period, so there is nothing to rank; twenty "
            "or more tickers give meaningful quantiles."
        )
    if thin:
        logger.warning(
            "%s left out of the factor analysis: fewer than %s tickers have a value in any "
            "period.",
            ", ".join(thin),
            MINIMUM_TICKERS,
        )

    if detail is not None:
        result = next(iter(analyses.values()))[detail]
        if detail == "rank_autocorrelation":
            result = result.to_frame("Rank Autocorrelation")
        if detail != "quantile_returns":
            result = result.dropna(how="all")
            result.index.name = "Date"
        return result

    summaries = {}
    for name, analysis in analyses.items():
        summary = summarize_information_coefficient(
            analysis["information_coefficient"]
        ).T
        summary["Top Minus Bottom"] = analysis["quantile_returns"].T["Top Minus Bottom"]
        turnover = analysis["quantile_turnover"]
        summary["Quantile Turnover"] = (
            turnover.iloc[:, [0, -1]].stack().mean() if not turnover.empty else np.nan
        )
        summary["Rank Autocorrelation"] = analysis["rank_autocorrelation"].mean()
        summaries[name] = summary

    result = pd.concat(summaries)
    result.index.names = ["Metric", "Horizon"]

    return result
