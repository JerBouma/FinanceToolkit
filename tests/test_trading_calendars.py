"""Trading Calendar Tests

A ticker listed on another exchange than the benchmark (the Nikkei 225 against SPY) trades on
other days. Its measures use its own trading days, whatever the benchmark, and measures against
the benchmark use the returns between the dates both traded.
"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit import Toolkit, historical_model
from financetoolkit.helpers import enrich_historical_data

DAYS = pd.bdate_range("2021-01-01", "2023-12-29")


def prices(index: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    returns = np.random.default_rng(seed).normal(0.0004, 0.015, len(index))
    close = 100 * np.cumprod(1 + returns)
    frame = pd.DataFrame(
        {
            "Open": close * 0.995,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Adj Close": close,
            "Volume": 1e6,
            "Dividends": 0.0,
        },
        index=pd.PeriodIndex(index, freq="D"),
    )
    return enrich_historical_data(frame, return_column="Adj Close")


def toolkit(asset: pd.DataFrame, benchmark: pd.DataFrame) -> Toolkit:
    # The tickers joined as the downloads are, including the filling of the dates only one traded.
    historical = pd.concat({"X": asset, "Benchmark": benchmark}).unstack(level=0)
    historical = historical.reindex(["X", "Benchmark"], level=1, axis=1)
    historical = historical_model.fill_non_trading_days(historical).sort_index()

    instance = Toolkit(
        ["X"],
        historical=historical,
        convert_currency=False,
        sleep_timer=False,
        start_date="2021-01-01",
        end_date="2023-12-31",
    )
    # A constant risk free rate, so it does not depend on the dates either. It is set before
    # the historical data is first read, which would otherwise download the Treasury rates:
    # a network request in an offline test, whose parallel yfinance downloads can lock
    # yfinance's SQLite cache ("database is locked") when the whole suite runs.
    treasury = pd.read_pickle("tests/datasets/treasury_data.pickle")
    index = historical.index
    instance._daily_treasury_data = pd.DataFrame(
        [treasury.iloc[-1].to_numpy()] * len(index),
        index=index,
        columns=treasury.columns,
    )
    instance._daily_risk_free_rate = instance._daily_treasury_data.xs(
        "10 Year", level=1, axis=1
    )

    return instance


@pytest.fixture(scope="module")
def calendars():
    rng = np.random.default_rng(0)
    asset_days = DAYS.delete(rng.choice(len(DAYS), 40, replace=False))
    benchmark_days = DAYS.delete(rng.choice(len(DAYS), 30, replace=False))
    asset = prices(asset_days, 1)
    benchmark = prices(benchmark_days, 2)

    # The same asset with a benchmark on its own days, and both on the days both traded.
    shared = asset.index.intersection(benchmark.index)
    columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends"]

    def on_shared_days(frame):
        return enrich_historical_data(frame.loc[shared, columns].copy())

    return {
        "mixed": toolkit(asset, benchmark),
        "own": toolkit(asset, prices(asset_days, 2)),
        "shared": toolkit(on_shared_days(asset), on_shared_days(benchmark)),
    }


@pytest.mark.parametrize(
    ("module", "method", "kwargs"),
    [
        ("risk", "get_drawdown_table", {}),
        ("risk", "get_maximum_drawdown_duration", {}),
        ("risk", "get_ulcer_index", {}),
        ("risk", "get_value_at_risk", {}),
        ("risk", "get_value_at_risk", {"period": "daily", "rolling": 21}),
        ("risk", "get_volatility", {"method": "parkinson"}),
        ("risk", "get_ewma_volatility", {}),
        ("performance", "get_sharpe_ratio", {}),
        ("performance", "get_sortino_ratio", {}),
        ("performance", "get_sterling_ratio", {}),
    ],
)
def test_a_tickers_own_measures_do_not_depend_on_the_benchmarks_days(
    calendars, module, method, kwargs
):
    def of_x(instance):
        result = getattr(getattr(instance, module), method)(**kwargs)
        if isinstance(result, pd.Series) or (
            isinstance(result.index, pd.MultiIndex)
            and "X" in result.index.get_level_values(0)
        ):
            return pd.DataFrame(result.loc[["X"]])
        # A value per date is compared on the dates X traded: with the other benchmark,
        # the dates only the benchmark traded are there too, without a value for X.
        return result[["X"]].dropna()

    pd.testing.assert_frame_equal(of_x(calendars["mixed"]), of_x(calendars["own"]))


@pytest.mark.parametrize(
    "method",
    [
        "get_beta",
        "get_tracking_error",
        "get_information_ratio",
        "get_upside_capture_ratio",
    ],
)
def test_measures_against_the_benchmark_use_the_days_both_traded(calendars, method):
    mixed = getattr(calendars["mixed"].performance, method)()
    shared = getattr(calendars["shared"].performance, method)()

    pd.testing.assert_series_equal(mixed["X"], shared["X"])
