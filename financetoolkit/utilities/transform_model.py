"""Time Series Transform Module"""

__docformat__ = "google"

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

from financetoolkit.utilities import logger_model
from financetoolkit.utilities.error_model import InvalidArgumentError, join_options

logger = logger_model.get_logger()

# Each step with whether its window is required, optional (with its default) or not taken.
REQUIRED = "required"
NO_WINDOW = "none"
TRANSFORM_STEPS: dict[str, object] = {
    "sma": REQUIRED,
    "ema": REQUIRED,
    "median": REQUIRED,
    "sum": REQUIRED,
    "std": REQUIRED,
    "min": REQUIRED,
    "max": REQUIRED,
    "zscore": REQUIRED,
    "percentile": REQUIRED,
    "annualize": REQUIRED,
    "diff": 1,
    "pct_change": 1,
    "log_diff": 1,
    "yoy": NO_WINDOW,
    "log": NO_WINDOW,
    "cumsum": NO_WINDOW,
    "hp_trend": None,
    "hp_cycle": None,
}

# Ravn and Uhlig (2002): the smoothing of 1,600 for quarterly data scaled by the fourth power
# of the number of periods per quarter, which gives 6.25 for yearly and 129,600 for monthly data.
HP_QUARTERLY_LAMBDA = 1600


def parse_transform(
    transform: str | list[str] | None,
) -> list[tuple[str, float | None]]:
    """
    Read the steps of a transform: "sma:12", ["log", "diff"] or "pct_change:3,annualize:3".

    Args:
        transform (str | list[str] | None): The steps, as a list or one text separated by
            commas, each a step name optionally followed by a colon and its window.

    Returns:
        list[tuple[str, float | None]]: Each step with its window, or None where the step
        takes none or uses its default.

    Raises:
        InvalidArgumentError: When a step is unknown, misses its window or has an invalid one.
    """
    if transform is None:
        return []

    texts = transform.split(",") if isinstance(transform, str) else list(transform)
    steps: list[tuple[str, float | None]] = []

    for text in texts:
        name, _, window_text = str(text).strip().partition(":")
        name = name.strip().lower()

        if not name:
            continue

        if name not in TRANSFORM_STEPS:
            raise InvalidArgumentError(
                f"Unknown transform step {name!r}. Options are: "
                f"{join_options(TRANSFORM_STEPS)}, each optionally followed by a window, "
                "such as 'sma:12'."
            )

        rule = TRANSFORM_STEPS[name]
        window: float | None = None

        if window_text.strip():
            if rule == NO_WINDOW:
                raise InvalidArgumentError(
                    f"The transform step {name!r} takes no window, so pass it as {name!r}."
                )
            try:
                window = float(window_text)
            except ValueError as error:
                raise InvalidArgumentError(
                    f"The window of the transform step {name!r} must be a number, "
                    f"not {window_text.strip()!r}."
                ) from error
            # The Hodrick-Prescott smoothing is a positive number; every other window counts
            # periods.
            if window <= 0 or (
                name not in ("hp_trend", "hp_cycle") and window != int(window)
            ):
                raise InvalidArgumentError(
                    f"The window of the transform step {name!r} must be a positive "
                    f"{'number' if name.startswith('hp_') else 'whole number'}, not {window_text.strip()}."
                )
        elif rule == REQUIRED:
            raise InvalidArgumentError(
                f"The transform step {name!r} needs a window, such as '{name}:12'."
            )
        elif isinstance(rule, int):
            window = rule

        steps.append((name, window))

    return steps


def get_hodrick_prescott_trend(series: pd.Series, smoothing: float) -> pd.Series:
    """
    Calculate the Hodrick-Prescott trend: the smooth path that trades closeness to the series
    against changes in its slope.

    The trend τ minimises Σ(yₜ - τₜ)² + λ Σ((τₜ₊₁ - τₜ) - (τₜ - τₜ₋₁))², which is the solution
    of a linear system with a banded matrix. The filter is two-sided: every value of the trend
    uses observations after it, so the last values change as new data arrives.

    The formula is as follows:

    - (I + λ D'D) τ = y, with D the (n - 2) x n second-difference matrix

    Reference: Hodrick, R. J., & Prescott, E. C. (1997). "Postwar U.S. Business Cycles: An
    Empirical Investigation." Journal of Money, Credit and Banking, 29(1), 1-16. The smoothing
    per frequency follows Ravn, M. O., & Uhlig, H. (2002). "On Adjusting the Hodrick-Prescott
    Filter for the Frequency of Observations." The Review of Economics and Statistics, 84(2),
    371-376.

    Referenced against:
    - statsmodels, hpfilter: https://github.com/statsmodels/statsmodels/blob/v0.14.4/statsmodels/tsa/filters/hp_filter.py#L10

    Args:
        series (pd.Series): The values, in time order.
        smoothing (float): The smoothing λ (1,600 for quarterly data).

    Returns:
        pd.Series: The trend on the dates with a value; dates without a value stay missing,
        and the values around a gap are filtered as if they were adjacent.

    As an example:

    ```python
    import pandas as pd

    from financetoolkit.utilities import transform_model

    series = pd.Series([1.0, 2.0, 4.0, 3.0, 5.0, 6.0, 8.0, 7.0])

    transform_model.get_hodrick_prescott_trend(series, 1600).round(4)
    ```

    Which returns:

    |    |      0 |
    |---:|-------:|
    |  0 | 1.1662 |
    |  1 | 2.119  |
    |  2 | 3.0716 |
    |  3 | 4.024  |
    |  4 | 4.9766 |
    |  5 | 5.929  |
    |  6 | 6.881  |
    |  7 | 7.8326 |
    """
    values = series.dropna()
    trend = pd.Series(np.nan, index=series.index, dtype=float)

    # Fewer than three values have no second difference to penalise, so the trend is the data.
    if len(values) < 3:  # noqa: PLR2004
        trend.loc[values.index] = values.to_numpy(dtype=float)
        return trend

    count = len(values)
    second_difference = sparse.diags(
        [np.ones(count - 2), -2 * np.ones(count - 2), np.ones(count - 2)],
        [0, 1, 2],
        shape=(count - 2, count),
    )
    system = sparse.eye(count) + smoothing * (second_difference.T @ second_difference)
    trend.loc[values.index] = spsolve(system.tocsc(), values.to_numpy(dtype=float))

    return trend


def default_hp_smoothing(dataset: pd.Series | pd.DataFrame) -> float:
    """
    The Hodrick-Prescott smoothing for the frequency of a dataset's periods (Ravn and Uhlig).

    Args:
        dataset (pd.Series | pd.DataFrame): The dataset with periods as its index.

    Returns:
        float: 6.25 for yearly, 1,600 for quarterly, 129,600 for monthly and 45,697,600 for
        weekly periods.

    Raises:
        InvalidArgumentError: For daily or other periods, which need the smoothing given.
    """
    return HP_QUARTERLY_LAMBDA * (periods_in_a_year(dataset, "hp_trend") / 4) ** 4


def periods_in_a_year(dataset: pd.Series | pd.DataFrame, step: str) -> int:
    """
    The number of periods in a year of a dataset indexed by yearly, quarterly, monthly or
    weekly periods.

    Args:
        dataset (pd.Series | pd.DataFrame): The dataset with periods as its index.
        step (str): The transform step that needs it, named in the error.

    Returns:
        int: 1, 4, 12 or 52.

    Raises:
        InvalidArgumentError: For daily or other periods.
    """
    frequency = str(getattr(dataset.index, "freqstr", "") or "").upper()

    for prefix, count in (("Y", 1), ("A", 1), ("Q", 4), ("M", 12), ("W", 52)):
        if frequency.startswith(prefix):
            return count

    window_hint = (
        f" Pass the smoothing as '{step}:<lambda>'."
        if step.startswith("hp_")
        else (
            " Pass the number of periods instead, such as 'pct_change:252' for daily data."
            if step == "yoy"
            else ""
        )
    )
    raise InvalidArgumentError(
        f"The transform step {step!r} needs yearly, quarterly, monthly or weekly periods, "
        f"not {frequency or 'dates without a frequency'}.{window_hint}"
    )


ROLLING_STEPS = (
    "sma",
    "ema",
    "median",
    "sum",
    "std",
    "min",
    "max",
    "zscore",
    "percentile",
)
GAPS_SHOWN = 3


def log_interior_gaps(dataset: pd.DataFrame, name: str, size: int) -> None:
    """
    Say which series miss a value inside their history, as every window of a rolling step that
    covers it is then missing too (a month a statistics office skipped empties a 12-month
    average for a year).

    Args:
        dataset (pd.DataFrame): The values the step is applied to, dates as the index.
        name (str): The step.
        size (int): Its window.
    """
    for column in dataset.columns:
        values = dataset[column]
        valid = values.notna().to_numpy()

        if not valid.any():
            continue

        first, last = valid.argmax(), len(valid) - valid[::-1].argmax()
        gaps = values.index[first:last][~valid[first:last]]

        if len(gaps):
            shown = ", ".join(str(gap) for gap in gaps[:GAPS_SHOWN])
            more = (
                f" and {len(gaps) - GAPS_SHOWN} more" if len(gaps) > GAPS_SHOWN else ""
            )
            logger.warning(
                "%s has no value for %s%s, so every %s-period window of '%s:%s' that includes it "
                "is missing.",
                column,
                shown,
                more,
                size,
                name,
                size,
            )


def apply_step(dataset: pd.DataFrame, name: str, window: float | None) -> pd.DataFrame:
    """
    Apply one transform step to every column of a dataset indexed by date.

    Args:
        dataset (pd.DataFrame): The values, dates as the index, a series per column.
        name (str): The step (see TRANSFORM_STEPS).
        window (float | None): Its window, smoothing or default.

    Returns:
        pd.DataFrame: The transformed values, on the same dates and columns.
    """
    size = int(window) if window is not None and not name.startswith("hp_") else 0

    if name == "sma":
        return dataset.rolling(size).mean()
    if name == "ema":
        # Imported here: the technicals package imports the helpers, which import this module.
        from financetoolkit.technicals.overlap.overlap_model import (  # noqa: PLC0415
            get_exponential_moving_average,
        )

        return get_exponential_moving_average(dataset, size)
    if name == "median":
        return dataset.rolling(size).median()
    if name == "sum":
        return dataset.rolling(size).sum()
    if name == "std":
        return dataset.rolling(size).std(ddof=1)
    if name == "min":
        return dataset.rolling(size).min()
    if name == "max":
        return dataset.rolling(size).max()
    if name == "zscore":
        rolling = dataset.rolling(size)
        return (dataset - rolling.mean()) / rolling.std(ddof=1)
    if name == "percentile":
        return dataset.rolling(size).rank(pct=True)
    if name == "diff":
        return dataset - dataset.shift(size)
    if name in ("pct_change", "yoy"):
        lag = size if name == "pct_change" else periods_in_a_year(dataset, name)
        previous = dataset.shift(lag)
        return (dataset - previous) / previous.abs()
    if name == "log_diff":
        logs = np.log(dataset.where(dataset > 0))
        return logs - logs.shift(size)
    if name == "annualize":
        return (1 + dataset) ** (periods_in_a_year(dataset, name) / size) - 1
    if name == "log":
        return np.log(dataset.where(dataset > 0))
    if name == "cumsum":
        return dataset.cumsum(skipna=True).where(dataset.notna())

    smoothing = window if window is not None else default_hp_smoothing(dataset)
    trend = dataset.apply(get_hodrick_prescott_trend, smoothing=smoothing)

    return trend if name == "hp_trend" else dataset - trend


def apply_transform(
    dataset: pd.Series | pd.DataFrame,
    transform: str | list[str] | None,
    axis: str = "rows",
) -> pd.Series | pd.DataFrame:
    """
    Apply a sequence of time series steps to every series of a dataset, in the order given:
    smoothing (moving averages), rolling statistics, changes, logs, cumulation and the
    Hodrick-Prescott trend or cycle.

    Each series keeps its own history: a rolling step needs a full window of its own values,
    so the first window - 1 periods and the periods after a gap are missing, as are logs of
    values that are not positive.

    The steps are as follows:

    - sma:n, ema:n, median:n: the simple (equally weighted), exponential (seeded with the
      simple average of the first n values) and median moving average over n periods
    - sum:n, std:n, min:n, max:n: the rolling sum, standard deviation (ddof = 1), minimum and
      maximum over n periods
    - zscore:n: (x - rolling mean) / rolling standard deviation over n periods
    - percentile:n: the share of the last n values at or below the current one (0 to 1)
    - diff:n: x(t) - x(t - n), with n = 1 by default
    - pct_change:n: (x(t) - x(t - n)) / |x(t - n)|, with n = 1 by default
    - yoy: the pct_change over the number of periods in a year (12 for monthly data)
    - log_diff:n: ln x(t) - ln x(t - n), with n = 1 by default
    - annualize:n: (1 + x)^(periods per year / n) - 1, for a change over n periods
    - log: ln x
    - cumsum: the running total
    - hp_trend:λ, hp_cycle:λ: the Hodrick-Prescott trend and the series minus it, with λ by
      default 6.25, 1,600, 129,600 or 45,697,600 for yearly, quarterly, monthly or weekly data

    Reference: Hodrick, R. J., & Prescott, E. C. (1997). "Postwar U.S. Business Cycles: An
    Empirical Investigation." Journal of Money, Credit and Banking, 29(1), 1-16; Ravn, M. O.,
    & Uhlig, H. (2002). "On Adjusting the Hodrick-Prescott Filter for the Frequency of
    Observations." The Review of Economics and Statistics, 84(2), 371-376.

    Args:
        dataset (pd.Series | pd.DataFrame): The values.
        transform (str | list[str] | None): The steps, such as "sma:12", ["log", "diff"] or
            "pct_change:3,annualize:3". Defaults to None, which returns the dataset as it is.
        axis (str, optional): Where the dates are: "rows" (dates as the index, the default) or
            "columns" (periods as the columns, as in the ratios). Defaults to "rows".

    Returns:
        pd.Series | pd.DataFrame: The transformed values, in the shape of the dataset.

    Raises:
        InvalidArgumentError: For an unknown step or a missing or invalid window.

    As an example:

    ```python
    import pandas as pd

    from financetoolkit.utilities import transform_model

    index = pd.period_range("2024-01", periods=6, freq="M")
    dataset = pd.DataFrame({"United States": [3.0, 3.2, 3.1, 3.5, 3.4, 3.8]}, index=index)

    transform_model.apply_transform(dataset, "sma:3,diff").round(4)
    ```

    Which returns:

    |         |   United States |
    |:--------|----------------:|
    | 2024-01 |        nan      |
    | 2024-02 |        nan      |
    | 2024-03 |        nan      |
    | 2024-04 |          0.1667 |
    | 2024-05 |          0.0667 |
    | 2024-06 |          0.2333 |
    """
    steps = parse_transform(transform)

    if not steps:
        return dataset

    by_columns = isinstance(dataset, pd.DataFrame) and axis == "columns"
    frame = dataset.T if by_columns else dataset
    is_series = isinstance(frame, pd.Series)
    frame = frame.to_frame() if is_series else frame

    for name, window in steps:
        if name in ROLLING_STEPS and window is not None:
            log_interior_gaps(frame, name, int(window))
        frame = apply_step(frame, name, window)

    frame = frame.replace([np.inf, -np.inf], np.nan)

    if is_series:
        return frame.iloc[:, 0].rename(dataset.name)

    return frame.T if by_columns else frame
