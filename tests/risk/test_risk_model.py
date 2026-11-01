"""Risk Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.risk import risk_model

# pylint: disable=missing-function-docstring


def test_get_max_drawdown(recorder):
    recorder.capture(
        risk_model.get_max_drawdown(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]),
        )
    )


def test_get_ui(recorder):
    recorder.capture(
        risk_model.get_ui(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]),
            rolling=2,
        )
    )


def test_get_skewness(recorder):
    recorder.capture(
        risk_model.get_skewness(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]),
        )
    )


def test_get_kurtosis(recorder):
    recorder.capture(
        round(
            risk_model.get_kurtosis(
                returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]), fisher=True
            ),
            4,
        )
    )

    recorder.capture(
        round(
            risk_model.get_kurtosis(
                returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]), fisher=False
            ),
            4,
        )
    )


def test_get_conditional_drawdown_at_risk(recorder):
    recorder.capture(
        round(
            risk_model.get_conditional_drawdown_at_risk(
                returns=pd.Series([0.3, 0.2, -0.1, -0.2, 0.06]), alpha=0.5
            ),
            4,
        )
    )


def test_get_conditional_drawdown_at_risk_dataframe(recorder):
    returns_df = pd.DataFrame(
        {
            "AAPL": [0.3, 0.2, -0.1, -0.2, 0.06],
            "MSFT": [0.1, -0.05, -0.1, 0.2, 0.03],
        }
    )
    recorder.capture(
        risk_model.get_conditional_drawdown_at_risk(
            returns=returns_df, alpha=0.5
        ).round(4)
    )


def test_get_tail_ratio(recorder):
    recorder.capture(
        round(
            risk_model.get_tail_ratio(
                returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06, -0.1, -0.15]), alpha=0.2
            ),
            4,
        )
    )


def test_get_tail_ratio_dataframe(recorder):
    returns_df = pd.DataFrame(
        {
            "AAPL": [0.3, 0.2, 0.1, 0, 0.06, -0.1, -0.15],
            "MSFT": [0.25, 0.15, 0.08, -0.02, 0.04, -0.08, -0.03],
        }
    )
    recorder.capture(risk_model.get_tail_ratio(returns=returns_df, alpha=0.2).round(4))


def test_get_max_drawdown_duration(recorder):
    recorder.capture(
        risk_model.get_max_drawdown_duration(
            returns=pd.Series([0.3, 0.2, -0.1, -0.2, 0.05, 0.4])
        )
    )


def test_get_max_drawdown_duration_dataframe(recorder):
    returns_df = pd.DataFrame(
        {
            "AAPL": [0.3, 0.2, -0.1, -0.2, 0.05, 0.4],
            "MSFT": [0.1, -0.05, -0.1, 0.2, 0.03, -0.01],
        }
    )
    recorder.capture(risk_model.get_max_drawdown_duration(returns=returns_df))


def test_get_max_drawdown_recovery_time(recorder):
    recorder.capture(
        risk_model.get_max_drawdown_recovery_time(
            returns=pd.Series([0.3, 0.2, -0.1, -0.2, 0.05, 0.4])
        )
    )

    # No recovery within the available data
    recorder.capture(
        risk_model.get_max_drawdown_recovery_time(
            returns=pd.Series([0.3, 0.2, -0.1, -0.2, 0.01])
        )
    )


def test_get_rolling_skewness(recorder):
    recorder.capture(
        risk_model.get_rolling_skewness(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06, -0.1, -0.15]),
            window_size=3,
        ).round(4)
    )


def test_get_rolling_kurtosis(recorder):
    recorder.capture(
        risk_model.get_rolling_kurtosis(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06, -0.1, -0.15]),
            window_size=3,
            fisher=True,
        ).round(4)
    )

    recorder.capture(
        risk_model.get_rolling_kurtosis(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06, -0.1, -0.15]),
            window_size=3,
            fisher=False,
        ).round(4)
    )


def test_get_downside_deviation(recorder):
    recorder.capture(
        round(
            risk_model.get_downside_deviation(
                returns=pd.Series([0.3, 0.2, -0.1, -0.2, 0.06]),
            ),
            4,
        )
    )

    recorder.capture(
        round(
            risk_model.get_downside_deviation(
                returns=pd.Series([0.3, 0.2, -0.1, -0.2, 0.06]),
                minimum_acceptable_return=0.05,
            ),
            4,
        )
    )


def test_get_downside_deviation_dataframe(recorder):
    returns_df = pd.DataFrame(
        {
            "AAPL": [0.3, 0.2, -0.1, -0.2, 0.06],
            "MSFT": [0.1, -0.05, -0.1, 0.2, 0.03],
        }
    )
    recorder.capture(risk_model.get_downside_deviation(returns=returns_df).round(4))


def test_get_rolling_downside_deviation(recorder):
    recorder.capture(
        risk_model.get_rolling_downside_deviation(
            returns=pd.Series([0.3, 0.2, -0.1, -0.2, 0.06, -0.15, 0.1]),
            window_size=3,
        ).round(4)
    )


def test_get_rolling_tail_ratio(recorder):
    recorder.capture(
        risk_model.get_rolling_tail_ratio(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06, -0.1, -0.15]),
            alpha=0.2,
            window_size=4,
        ).round(4)
    )


def test_get_rolling_conditional_drawdown_at_risk(recorder):
    recorder.capture(
        risk_model.get_rolling_conditional_drawdown_at_risk(
            returns=pd.Series([0.3, 0.2, -0.1, -0.2, 0.06, -0.15, 0.1]),
            alpha=0.5,
            window_size=4,
        ).round(4)
    )


def test_get_rolling_variance(recorder):
    recorder.capture(
        risk_model.get_rolling_variance(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06, -0.1, -0.15]),
            period="monthly",
            window_size=3,
        ).round(4)
    )


def test_get_rolling_volatility(recorder):
    recorder.capture(
        risk_model.get_rolling_volatility(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06, -0.1, -0.15]),
            period="monthly",
            window_size=3,
        ).round(4)
    )


def test_get_hill_estimator_pareto_recovery(recorder):
    # A Pareto(alpha=3) tail, whose index the Hill estimator should recover.
    rng = np.random.default_rng(42)
    alpha_true = 3.0
    n = 20000
    u = rng.uniform(size=n)
    pareto = (1 - u) ** (-1 / alpha_true)
    returns = pd.Series(-pareto * 0.001)

    result = risk_model.get_hill_estimator(returns, k=0.05, tail="left")
    assert abs(result["Hill Tail Index"] - alpha_true) < 1.0
    recorder.capture(result.round(4))


def test_get_hill_estimator_dataframe(recorder):
    rng = np.random.default_rng(42)
    returns_df = pd.DataFrame(
        {
            "AAPL": rng.standard_normal(2000) * 0.02,
            "MSFT": rng.standard_normal(2000) * 0.02,
        }
    )
    recorder.capture(
        risk_model.get_hill_estimator(returns_df, k=0.1, tail="left").round(4)
    )


def test_get_hill_estimator_right_tail(recorder):
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.standard_normal(2000) * 0.02)
    recorder.capture(
        risk_model.get_hill_estimator(returns, k=0.1, tail="right").round(4)
    )


def test_get_hill_estimator_k_too_large():
    returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
    result = risk_model.get_hill_estimator(returns, k=10, tail="left")
    assert pd.isna(result["Hill Tail Index"])


def test_get_hill_estimator_invalid_tail():
    returns = pd.Series([0.01, -0.02, 0.03])
    try:
        risk_model.get_hill_estimator(returns, tail="bad")
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_hill_estimator_invalid_type():
    try:
        risk_model.get_hill_estimator(123)
        raise AssertionError("Expected TypeError")
    except TypeError:
        pass


def test_get_rolling_excess_volatility(recorder):
    recorder.capture(
        risk_model.get_rolling_excess_volatility(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06, -0.1, -0.15]),
            risk_free_rate=pd.Series([0.01, 0.02, 0.01, 0.0, 0.006, 0.01, 0.02]),
            period="monthly",
            window_size=3,
        ).round(4)
    )


def test_drawdown_measures_are_nan_without_observations():
    returns = pd.DataFrame(
        {"A": [np.nan, np.nan, np.nan], "B": [0.01, -0.05, 0.02]},
    )

    for result in (
        risk_model.get_max_drawdown(returns),
        risk_model.get_conditional_drawdown_at_risk(returns, 0.05),
        risk_model.get_max_drawdown_duration(returns),
        risk_model.get_max_drawdown_recovery_time(returns),
        risk_model.get_ui(returns, rolling=14),
    ):
        assert np.isnan(result["A"])
    assert np.isclose(risk_model.get_max_drawdown(returns)["B"], -0.05)
    assert np.isnan(risk_model.get_max_drawdown(returns["A"]))
    assert np.isnan(risk_model.get_max_drawdown_duration(returns["A"]))
    assert np.isnan(risk_model.get_max_drawdown_recovery_time(returns["A"]))


def test_within_period_drawdown_is_nan_for_a_period_without_data():
    index = pd.MultiIndex.from_tuples(
        [("2020", 1), ("2020", 2), ("2021", 1), ("2021", 2)]
    )
    returns = pd.DataFrame(
        {"A": [np.nan, np.nan, 0.01, -0.02], "B": [0.01, -0.03, 0.02, -0.01]},
        index=index,
    )

    result = risk_model.get_max_drawdown(returns)

    assert np.isnan(result.loc["2020", "A"])
    assert np.isclose(result.loc["2021", "A"], -0.02)


def test_get_ui_short_period_uses_the_high_so_far():
    # Five observations with a 14-period window: the high is taken over the observations so far.
    returns = pd.Series([0.02, -0.03, -0.01, 0.015, -0.02])

    assert np.isfinite(risk_model.get_ui(returns, rolling=14))
    assert np.isclose(
        risk_model.get_ui(returns, rolling=14), risk_model.get_ui(returns, None)
    )


def test_get_hurst_exponent_includes_max_lag():
    data = pd.Series(np.cumsum(np.random.default_rng(0).normal(0, 1, 2000)))
    values = data.to_numpy()
    lags = np.arange(2, 11)
    standard_deviations = [np.std(values[lag:] - values[:-lag]) for lag in lags]
    expected = np.polyfit(np.log(lags), np.log(standard_deviations), 1)[0]

    assert np.isclose(risk_model.get_hurst_exponent(data, max_lag=10), expected)
    assert abs(risk_model.get_hurst_exponent(data) - 0.5) < 0.1  # noqa: PLR2004


def test_get_ui_invalid_method():
    with pytest.raises(ValueError, match="method"):
        risk_model.get_ui(pd.Series([0.01, -0.02]), method="bad")


def test_get_hill_estimator_without_data_uses_no_order_statistics():
    for returns in (pd.Series(dtype=float), pd.Series([np.nan, np.nan, 0.01])):
        result = risk_model.get_hill_estimator(returns, tail="left")

        assert pd.isna(result["Hill Tail Index"])
        assert result["Observations Used (k)"] == 0

    # With data, a fraction that rounds down to zero still uses one order statistic.
    result = risk_model.get_hill_estimator(pd.Series([-0.01, -0.02, -0.03]), k=0.1)
    assert result["Observations Used (k)"] == 1


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_hurst_exponent_bounds_lags_to_available_differences():
    data = pd.Series([0.0, 1.0, 0.5, 2.0, 1.0, 3.0, 2.5, 4.0])
    values = data.to_numpy()
    lags = np.arange(2, len(values) - 1)
    deviations = [np.std(values[lag:] - values[:-lag]) for lag in lags]
    expected = np.polyfit(np.log(lags), np.log(deviations), 1)[0]

    assert risk_model.get_hurst_exponent(data) == pytest.approx(expected)


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("values", [[], [1.0], [1.0, 2.0, 3.0, 4.0], [2.0] * 30])
def test_hurst_exponent_without_two_usable_lags(values):
    assert np.isnan(risk_model.get_hurst_exponent(pd.Series(values, dtype=float)))


def _rolling_cdar_per_window(returns, alpha, window_size, method):
    def window_cdar(window):
        if method == "level":
            drawdowns = window - np.maximum.accumulate(window)
        else:
            wealth = np.cumprod(1 + window)
            drawdowns = wealth / np.maximum(np.maximum.accumulate(wealth), 1) - 1
        return drawdowns[drawdowns <= np.percentile(drawdowns, alpha * 100)].mean()

    return returns.rolling(window_size).apply(window_cdar, raw=True)


@pytest.mark.parametrize("method", ["return", "level"])
def test_rolling_cdar_matches_the_cdar_of_every_window(method):
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(rng.normal(0, 0.01, size=(250, 2)), columns=["A", "B"])
    returns.iloc[100, 1] = np.nan
    data = returns if method == "return" else (1 + returns.fillna(0)).cumprod()

    pd.testing.assert_frame_equal(
        risk_model.get_rolling_conditional_drawdown_at_risk(data, 0.05, 20, method),
        _rolling_cdar_per_window(data, 0.05, 20, method),
        rtol=1e-12,
        atol=1e-15,
    )


@pytest.mark.parametrize("rolling", [14, None, 1])
@pytest.mark.parametrize("method", ["return", "level"])
def test_ulcer_index_per_period_matches_each_period_on_its_own(rolling, method):
    rng = np.random.default_rng(2)
    days = pd.bdate_range("2020-01-01", periods=300)
    returns = pd.DataFrame(rng.normal(0, 0.01, size=(300, 2)), columns=["A", "B"])
    returns.iloc[:70, 1] = np.nan
    data = returns if method == "return" else (1 + returns.fillna(0)).cumprod() * 100
    data.index = pd.MultiIndex.from_arrays([days.to_period("M"), days.to_period("D")])

    expected = pd.DataFrame(
        {
            period: risk_model.get_ui(data.loc[period], rolling=rolling, method=method)
            for period in data.index.get_level_values(0).unique()
        }
    ).T

    # The months before B's first return have no Ulcer Index rather than one of 0.
    pd.testing.assert_frame_equal(
        risk_model.get_ui(data, rolling=rolling, method=method),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("rolling", [None, 3])
@pytest.mark.parametrize("method", ["return", "level"])
def test_ulcer_index_per_period_with_interleaved_rows(rolling, method):
    rng = np.random.default_rng(0)
    periods = ["A", "A", "B", "B", "A", "A", "B", "B", "A", "B", "A", "B"]
    returns = pd.DataFrame(
        rng.normal(0, 0.02, (12, 2)),
        index=pd.MultiIndex.from_arrays(
            [periods, pd.date_range("2020-01-01", periods=12)]
        ),
        columns=["x", "y"],
    )

    expected = pd.DataFrame(
        {
            period: risk_model.get_ui(returns.loc[period], rolling, method)
            for period in ["A", "B"]
        }
    ).T

    pd.testing.assert_frame_equal(
        risk_model.get_ui(returns, rolling, method).sort_index(),
        expected,
        check_names=False,
    )
