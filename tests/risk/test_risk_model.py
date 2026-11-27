"""Risk Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.risk import risk_model
from financetoolkit.risk.helpers import (
    compound_into_shared_dates,
    compound_risk_free_rate,
    get_observations_per_year,
    get_risk_free_rate_per_date,
    get_risk_free_rate_per_observation,
    get_risk_free_rate_per_ticker,
)
from financetoolkit.risk.value_at_risk import cvar_model, var_model
from financetoolkit.utilities.error_model import InvalidArgumentError


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


def test_get_skewness_rolling(recorder):
    recorder.capture(
        risk_model.get_skewness(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06, -0.1, -0.15]),
            window=3,
        ).round(4)
    )


def test_get_kurtosis_rolling(recorder):
    recorder.capture(
        risk_model.get_kurtosis(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06, -0.1, -0.15]),
            # Kurtosis needs four observations, so a smaller window is all NaN.
            window=4,
            fisher=True,
        ).round(4)
    )

    recorder.capture(
        risk_model.get_kurtosis(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06, -0.1, -0.15]),
            # Kurtosis needs four observations, so a smaller window is all NaN.
            window=4,
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


def test_get_downside_deviation_rolling(recorder):
    recorder.capture(
        risk_model.get_downside_deviation(
            returns=pd.Series([0.3, 0.2, -0.1, -0.2, 0.06, -0.15, 0.1]),
            window=3,
        ).round(4)
    )


def test_get_tail_ratio_rolling(recorder):
    recorder.capture(
        risk_model.get_tail_ratio(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06, -0.1, -0.15]),
            alpha=0.2,
            window=4,
        ).round(4)
    )


def test_get_conditional_drawdown_at_risk_rolling(recorder):
    recorder.capture(
        risk_model.get_conditional_drawdown_at_risk(
            returns=pd.Series([0.3, 0.2, -0.1, -0.2, 0.06, -0.15, 0.1]),
            alpha=0.5,
            window=4,
        ).round(4)
    )


def test_get_variance_rolling(recorder):
    recorder.capture(
        risk_model.get_variance(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06, -0.1, -0.15]),
            period="monthly",
            window=3,
        ).round(4)
    )


def test_get_volatility_rolling(recorder):
    recorder.capture(
        risk_model.get_volatility(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06, -0.1, -0.15]),
            period="monthly",
            window=3,
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


def test_get_excess_volatility_rolling(recorder):
    recorder.capture(
        risk_model.get_excess_volatility(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06, -0.1, -0.15]),
            risk_free_rate=pd.Series([0.01, 0.02, 0.01, 0.0, 0.006, 0.01, 0.02]),
            period="monthly",
            window=3,
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

    # The missing return is a day B didn't trade: its windows skip it, as for B on its own.
    expected = pd.concat(
        [
            _rolling_cdar_per_window(data[[column]].dropna(), 0.05, 20, method)
            for column in data.columns
        ],
        axis=1,
    ).reindex(data.index)

    pd.testing.assert_frame_equal(
        risk_model.get_conditional_drawdown_at_risk(data, 0.05, method, window=20),
        expected,
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


def test_hill_estimator_of_a_constant_tail_is_nan():
    estimate = risk_model.get_hill_estimator(pd.Series([-0.01] * 50), tail="left")

    assert np.isnan(estimate["Hill Tail Index"])


def test_move_significance_only_compares_a_move_with_the_moves_before_it():
    dates = pd.period_range("2024-01-01", periods=8, freq="D")
    returns = pd.Series(
        [0.01, -0.02, 0.005, -0.01, 0.03, np.nan, -0.025, 0.002], index=dates
    )

    significance = risk_model.get_move_significance(returns, window=3)

    # The volatility of the three returns before the day, not including the day itself.
    expected = -0.01 / np.std([0.01, -0.02, 0.005], ddof=1)
    assert significance.loc["2024-01-04", "Volatility Multiple"] == pytest.approx(
        expected
    )
    # The gap is skipped: the window before 2024-01-07 is the three returns before the gap.
    expected = -0.025 / np.std([0.005, -0.01, 0.03], ddof=1)
    assert significance.loc["2024-01-07", "Volatility Multiple"] == pytest.approx(
        expected
    )
    assert significance.loc["2024-01-06"].isna().all()

    # Smaller in absolute value than four of the five earlier moves.
    assert significance.loc["2024-01-07", "Percentile"] == pytest.approx(0.8)
    assert significance.loc[:"2024-01-03", "Percentile"].isna().all()


def test_the_var_multiple_divides_a_move_by_the_var_of_the_moves_before_it():
    dates = pd.period_range("2024-01-01", periods=6, freq="D")
    returns = pd.Series([0.01, -0.02, 0.005, -0.04, 0.03, -0.06], index=dates)

    significance = risk_model.get_move_significance(returns, window=4, alpha=0.25)

    # The 25% quantile of the four returns before the day (linear interpolation), a loss.
    var = np.quantile([-0.02, 0.005, -0.04, 0.03], 0.25)
    assert var < 0
    assert significance.loc["2024-01-06", "VaR Multiple"] == pytest.approx(-0.06 / -var)
    # A rise keeps its sign.
    var = np.quantile([0.01, -0.02, 0.005, -0.04], 0.25)
    assert significance.loc["2024-01-05", "VaR Multiple"] == pytest.approx(0.03 / -var)
    assert significance.loc[:"2024-01-04", "VaR Multiple"].isna().all()


def test_largest_since_compares_falls_with_falls_and_rises_with_rises():
    dates = pd.period_range("2024-01-01", periods=6, freq="D")
    returns = pd.Series([-0.05, 0.04, -0.01, 0.02, -0.03, 0.05], index=dates)

    largest_since = risk_model.get_move_significance(returns, window=2)["Largest Since"]

    # A 3% fall: the last fall at least as deep was the 5% fall, the rises don't count.
    assert largest_since["2024-01-05"] == "2024-01-01"
    # A 2% rise: the 4% rise was larger.
    assert largest_since["2024-01-04"] == "2024-01-02"
    # The largest rise of the history has no earlier date.
    assert pd.isna(largest_since["2024-01-06"])
    assert pd.isna(largest_since["2024-01-01"])


def drawdown_returns():
    """A first close (no return), then A falls, recovers, peaks and falls; B lists later."""
    return pd.DataFrame(
        {
            "A": [np.nan, -0.1, 0.05, 0.06, 0.1, -0.2, 0.1, np.nan, 0.01, 0.02],
            "B": [np.nan, np.nan, np.nan, 0.1, -0.5, 0.2, 0.0, 0.0, 0.0, 0.1],
        },
        index=pd.period_range("2024-01-01", periods=10, freq="D"),
    )


def test_the_drawdown_series_starts_at_each_tickers_first_close():
    drawdowns = risk_model.get_drawdown_series(drawdown_returns())

    # B's first close is on the third row, the close its first return is measured from.
    assert drawdowns["B"].iloc[:2].isna().all()
    assert drawdowns["B"].iloc[2] == 0
    assert drawdowns["A"].iloc[0] == 0
    assert np.isclose(drawdowns["A"].iloc[1], -0.1)
    # 0.9 * 1.05 * 1.06 is back above the starting wealth: a new high.
    assert drawdowns["A"].iloc[3] == 0
    assert np.isclose(drawdowns["A"].iloc[5], -0.2)


def test_the_recovery_series_is_the_gain_needed_to_get_back_to_the_peak():
    recovery = risk_model.get_drawdown_series(drawdown_returns(), recovery=True)

    # A 10% fall needs 1 / 0.9 - 1 = 11.1%, a 20% fall 25%, a new high nothing.
    assert np.isclose(recovery["A"].iloc[1], 1 / 0.9 - 1)
    assert np.isclose(recovery["A"].iloc[5], 0.25)
    assert recovery["A"].iloc[3] == 0
    # B halves after its first gain: it needs 100% to be back at that high.
    assert np.isclose(recovery["B"].iloc[4], 1.0)
    assert recovery["B"].iloc[:2].isna().all()


def test_the_drawdown_table_lists_each_drawdown_deepest_first():
    returns = drawdown_returns()
    table = risk_model.get_drawdown_table(returns, top=5)

    first, second = table.loc["A"].iloc[0], table.loc["A"].iloc[1]
    assert np.isclose(first["Drawdown"], -0.2)
    assert np.isclose(first["Recovery Needed"], 0.25)
    assert first["Peak"] == pd.Period("2024-01-05", freq="D")
    assert first["Trough"] == pd.Period("2024-01-06", freq="D")
    # Not recovered by the end of the window.
    assert pd.isna(first["Recovery"]) and np.isnan(first["Length"])

    assert np.isclose(second["Drawdown"], -0.1)
    assert second["Peak"] == pd.Period("2024-01-01", freq="D")
    assert second["Recovery"] == pd.Period("2024-01-04", freq="D")
    assert (second["To Trough"], second["To Recovery"], second["Length"]) == (1, 2, 3)
    # Daily rows on consecutive days: the calendar days equal the periods.
    assert (
        second["To Trough (Days)"],
        second["To Recovery (Days)"],
        second["Length (Days)"],
    ) == (1, 2, 3)
    assert np.isclose(second["Length (Years)"], 3 / 365.25)
    assert np.isnan(first["Length (Days)"]) and np.isnan(first["To Recovery (Days)"])
    assert first["To Trough (Days)"] == 1

    # The deepest drawdown is the one the maximum drawdown measures.
    measured = returns.iloc[1:]
    assert np.allclose(
        table.xs(1, level="Rank")["Drawdown"], risk_model.get_max_drawdown(measured)
    )
    assert (
        table.xs(1, level="Rank")["To Trough"]
        == risk_model.get_max_drawdown_duration(measured)
    ).all()


def test_the_drawdown_table_keeps_the_requested_number_of_drawdowns():
    table = risk_model.get_drawdown_table(drawdown_returns(), top=1)

    assert table.index.get_level_values("Rank").unique().tolist() == [1]
    assert table.index.get_level_values("Ticker").tolist() == ["A", "B"]


def test_a_ticker_without_returns_has_no_drawdowns():
    returns = drawdown_returns().assign(C=np.nan)

    assert risk_model.get_drawdown_series(returns)["C"].isna().all()
    assert "C" not in risk_model.get_drawdown_table(returns).index.get_level_values(0)


def test_get_mean_absolute_deviation_per_period():
    returns = pd.Series(
        [1.0, 2.0, 3.0, 4.0, 10.0, 10.0],
        index=pd.to_datetime(
            [
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-02-01",
                "2024-02-02",
            ]
        ).to_period("D"),
    )

    result = risk_model.get_mean_absolute_deviation(returns, "monthly")

    # January: mean 2.5, absolute deviations 1.5, 0.5, 0.5, 1.5. February is constant.
    assert result.tolist() == pytest.approx([1.0, 0.0])
    assert [str(period) for period in result.index] == ["2024-01", "2024-02"]

    with pytest.raises(InvalidArgumentError, match="period must be one of"):
        risk_model.get_mean_absolute_deviation(returns, "daily")
    with pytest.raises(TypeError):
        risk_model.get_mean_absolute_deviation(returns.tolist(), "monthly")


def test_get_coefficient_of_variation_per_group():
    returns = pd.DataFrame({"A": [1.0, 2.0, 3.0, 2.0, 2.0, 2.0]})
    groups = np.array(["first"] * 3 + ["second"] * 3)

    result = risk_model.get_coefficient_of_variation(returns, "monthly", groups=groups)

    # The sample standard deviation (ddof=1) of 1, 2, 3 is 1, its mean 2.
    assert result.loc["first", "A"] == pytest.approx(0.5)
    assert result.loc["second", "A"] == pytest.approx(0.0)

    with pytest.raises(InvalidArgumentError, match="period must be one of"):
        risk_model.get_coefficient_of_variation(returns, "hourly", groups=groups)


def test_get_autocorrelation_uses_the_full_sample_variance():
    # Returns are indexed by date: a result indexed by lag is then never read as dates.
    result = risk_model.get_autocorrelation(
        pd.Series(
            [1.0, 2.0, np.nan, 3.0, 4.0],
            index=pd.period_range("2024-01-01", periods=5, freq="D"),
        ),
        lags=3,
    )

    # Mean 2.5 and a sum of squared deviations of 5 (the missing value is dropped);
    # the lagged cross products sum to 1.25, -1.5 and -2.25.
    pd.testing.assert_series_equal(
        result,
        pd.Series({1: 0.25, 2: -0.3, 3: -0.45}, name="Autocorrelation"),
    )

    with pytest.raises(TypeError):
        risk_model.get_autocorrelation(pd.DataFrame({"A": [1.0, 2.0]}))


def test_get_quantile_per_type():
    returns = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

    assert risk_model.get_quantile(returns, 0.25) == 2.0

    frame = pd.DataFrame({"A": returns, "B": returns * 10})
    pd.testing.assert_series_equal(
        risk_model.get_quantile(frame, 0.5), pd.Series({"A": 3.0, "B": 30.0})
    )

    index = pd.MultiIndex.from_tuples([("2020", 1), ("2020", 2), ("2021", 1)])
    stacked = pd.DataFrame({"A": [1.0, 3.0, 5.0]}, index=index)
    # A period with a single return has no distribution to take a quantile of.
    assert risk_model.get_quantile(stacked, 0.5)["A"].tolist() == pytest.approx(
        [2.0, np.nan], nan_ok=True
    )

    with pytest.raises(TypeError):
        risk_model.get_quantile([1.0, 2.0], 0.5)


def test_get_wealth_and_peak_starts_from_an_initial_wealth_of_one():
    wealth, peak = risk_model.get_wealth_and_peak(pd.Series([-0.1, 0.5, np.nan, -0.5]))

    # A missing return keeps the wealth unchanged; a loss on the first day is measured
    # against the starting wealth of 1, not against the lower first value.
    assert wealth.tolist() == pytest.approx([0.9, 1.35, 1.35, 0.675])
    assert peak.tolist() == pytest.approx([1.0, 1.35, 1.35, 1.35])


def test_the_drawdown_table_counts_calendar_days_over_market_holidays():
    # Trading days only: a peak on Friday, a trough on Monday, back on Wednesday a week later.
    dates = pd.PeriodIndex(
        ["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-17"],
        freq="D",
    )
    returns = pd.DataFrame({"A": [np.nan, 0.1, -0.2, 0.1, 0.2]}, index=dates)

    drawdown = risk_model.get_drawdown_table(returns).loc[("A", 1)]

    assert (drawdown["To Trough"], drawdown["To Recovery"], drawdown["Length"]) == (
        1,
        2,
        3,
    )
    assert drawdown["To Trough (Days)"] == 3
    assert drawdown["To Recovery (Days)"] == 9
    assert drawdown["Length (Days)"] == 12


def test_the_drawdown_table_counts_calendar_days_to_the_end_of_each_month():
    returns = pd.DataFrame(
        {"A": [np.nan, -0.1, 0.0, 0.2]},
        index=pd.period_range("2024-01", periods=4, freq="M"),
    )

    drawdown = risk_model.get_drawdown_table(returns).loc[("A", 1)]

    # From the end of January to the end of February (29 days, a leap year) and of April.
    assert drawdown["To Trough (Days)"] == 29
    assert drawdown["Length (Days)"] == 90


def test_deepest_since_is_the_last_date_as_deep_as_today():
    drawdowns = pd.DataFrame(
        {"A": [0.0, -0.3, -0.1, 0.0, -0.05, -0.2, -0.25, -0.35, np.nan]},
        index=pd.period_range("2024-01-01", periods=9, freq="D"),
    )

    since = risk_model.get_deepest_since(drawdowns)["A"]

    # At a high there is nothing to compare with.
    assert since.iloc[0] is None and since.iloc[3] is None
    # -0.1 was last as deep on 2 January (-0.3), -0.05 on 3 January (-0.1).
    assert since.iloc[2] == "2024-01-02"
    assert since.iloc[4] == "2024-01-03"
    # -0.25 skips the shallower -0.2 of 6 January.
    assert since.iloc[6] == "2024-01-02"
    # The deepest drawdown so far has no earlier date as deep.
    assert since.iloc[7] is None


def test_skewness_and_kurtosis_of_returns_without_variation_are_nan():
    returns = pd.DataFrame(
        {"Constant": [0.0] * 20, "Varying": np.linspace(-1, 1, 20) ** 3}
    )

    skewness = risk_model.get_skewness(returns)
    kurtosis = risk_model.get_kurtosis(returns, fisher=False)

    assert np.isnan(skewness["Constant"]) and np.isfinite(skewness["Varying"])
    assert np.isnan(kurtosis["Constant"]) and np.isfinite(kurtosis["Varying"])
    assert np.isnan(risk_model.get_skewness(pd.Series([0.01] * 5)))

    rolling_returns = pd.Series([0.0] * 10 + [0.01, -0.02, 0.03, 0.01])
    rolling_skewness = risk_model.get_skewness(rolling_returns, window=5)
    rolling_kurtosis = risk_model.get_kurtosis(rolling_returns, window=5)

    # Windows of only zero returns are undefined; once a window varies it has a value.
    assert rolling_skewness.iloc[:10].isna().all()
    assert rolling_kurtosis.iloc[:10].isna().all()
    assert rolling_skewness.iloc[10:].notna().all()
    assert rolling_kurtosis.iloc[10:].notna().all()


def test_the_var_multiple_is_nan_when_the_trailing_quantile_is_no_loss():
    dates = pd.period_range("2024-01-01", periods=7, freq="D")
    returns = pd.Series([0.01, 0.02, 0.015, 0.03, 0.012, 0.025, -0.01], index=dates)

    significance = risk_model.get_move_significance(returns, window=5)

    # Every trailing 5% quantile is a gain: no loss to scale the move by, so no multiple
    # (rather than a fall measured against a gain).
    assert significance["VaR Multiple"].isna().all()
    assert significance.loc["2024-01-07", "Volatility Multiple"] < 0


def test_the_move_percentile_counts_equal_moves_as_half():
    dates = pd.period_range("2024-01-01", periods=6, freq="D")
    returns = pd.Series([0.0, 0.0, 0.0, 0.0, 0.01, 0.0], index=dates)

    significance = risk_model.get_move_significance(returns, window=3)

    # An unchanged day among three unchanged days sits in the middle of its ties.
    assert significance.loc["2024-01-04", "Percentile"] == pytest.approx(0.5)
    # The 1% move is larger than all four earlier ones.
    assert significance.loc["2024-01-05", "Percentile"] == pytest.approx(1.0)
    # An unchanged day: smaller than one earlier move, tied with four.
    assert significance.loc["2024-01-06", "Percentile"] == pytest.approx(0.4)
    # A flat day has no direction and the largest rise has no earlier date: both NaN,
    # the same missing marker as the rows of a gap.
    largest_since = significance["Largest Since"]
    assert largest_since.isna().all()
    assert all(value is not None for value in largest_since)


def _late_listing_returns():
    rng = np.random.default_rng(3)
    index = pd.period_range("2020-01-01", periods=500, freq="D")
    later = rng.normal(0.0, 0.03, 500)
    later[:300] = np.nan

    return pd.DataFrame(
        {"Early": rng.normal(0.0005, 0.02, 500), "Later": later}, index=index
    )


def test_drawdown_measures_skip_the_rows_before_a_later_listing():
    returns = _late_listing_returns()
    alone = returns[["Later"]].dropna()

    # The rows before the listing are no drawdowns of 0 that dilute the tail or the mean.
    assert risk_model.get_conditional_drawdown_at_risk(returns, 0.05)[
        "Later"
    ] == pytest.approx(
        risk_model.get_conditional_drawdown_at_risk(alone, 0.05)["Later"]
    )
    for rolling in (None, 14):
        assert risk_model.get_ui(returns, rolling)["Later"] == pytest.approx(
            risk_model.get_ui(alone, rolling)["Later"]
        )

    within = returns.set_axis(
        pd.MultiIndex.from_arrays([returns.index.asfreq("Y"), returns.index])
    )
    first_year = returns.loc[returns.index.year == 2020, ["Later"]].dropna()

    assert risk_model.get_conditional_drawdown_at_risk(within, 0.05).iloc[0][
        "Later"
    ] == pytest.approx(
        risk_model.get_conditional_drawdown_at_risk(first_year, 0.05)["Later"]
    )
    assert risk_model.get_ui(within, 14).iloc[0]["Later"] == pytest.approx(
        risk_model.get_ui(first_year, 14)["Later"]
    )


def _stock_and_crypto_returns():
    rng = np.random.default_rng(7)
    index = pd.period_range("2022-01-01", "2023-12-31", freq="D")
    stock = np.where(index.dayofweek < 5, rng.normal(0, 0.02, len(index)), np.nan)

    return pd.DataFrame(
        {"Crypto": rng.normal(0, 0.03, len(index)), "Stock": stock}, index=index
    )


def test_volatility_of_a_ticker_trading_on_weekends_scales_with_365_days():
    returns = _stock_and_crypto_returns()
    in_2022 = returns.index.year == 2022  # noqa: PLR2004

    volatility = risk_model.get_volatility(returns, "yearly")
    variance = risk_model.get_variance(returns, "weekly")

    assert volatility.iloc[0]["Crypto"] == pytest.approx(
        returns["Crypto"][in_2022].std() * np.sqrt(365)
    )
    assert volatility.iloc[0]["Stock"] == pytest.approx(
        returns["Stock"][in_2022].std() * np.sqrt(252)
    )
    week = returns.index.asfreq("W") == returns.index.asfreq("W")[10]
    assert variance.loc[returns.index.asfreq("W")[10], "Crypto"] == pytest.approx(
        returns["Crypto"][week].var() * 365 / 52
    )
    # A Series keeps the scaling of its own calendar.
    assert risk_model.get_volatility(returns["Stock"], "yearly").iloc[
        0
    ] == pytest.approx(volatility.iloc[0]["Stock"])


def test_quantile_of_a_period_with_a_single_return_is_nan():
    index = pd.period_range("2021-12-31", periods=6, freq="D")
    returns = pd.DataFrame({"A": [-0.02, 0.01, -0.03, 0.02, 0.0, -0.01]}, index=index)
    within = returns.set_axis(pd.MultiIndex.from_arrays([index.asfreq("Y"), index]))

    quantile = risk_model.get_quantile(within, 0.05)

    assert np.isnan(quantile.loc[pd.Period("2021", "Y"), "A"])
    assert quantile.loc[pd.Period("2022", "Y"), "A"] == pytest.approx(
        returns["A"].iloc[1:].quantile(0.05)
    )


def test_compound_into_shared_dates_compounds_the_weekend_into_monday():
    index = pd.period_range("2024-01-04", periods=8, freq="D")
    returns = pd.DataFrame(
        {
            "Stock": [np.nan, 0.01, np.nan, np.nan, 0.02, 0.03, np.nan, np.nan],
            "Crypto": [0.1, 0.04, 0.01, 0.01, 0.01, 0.05, 0.5, np.nan],
        },
        index=index,
    )

    shared = compound_into_shared_dates(returns)

    # The Thursday before the first shared date and the dates after the last are left out,
    # and each shared date keeps its own returns, in the order of the dates.
    assert list(shared.index.astype(str)) == ["2024-01-05", "2024-01-08", "2024-01-09"]
    assert shared["Stock"].to_numpy() == pytest.approx([0.01, 0.02, 0.03])
    # Monday 8 January holds the Crypto returns of Saturday, Sunday and Monday.
    assert shared["Crypto"].to_numpy() == pytest.approx([0.04, 1.01**3 - 1, 0.05])


def test_observations_per_year_and_risk_free_rate_follow_each_calendar():
    returns = _stock_and_crypto_returns()
    annualized_rate = pd.Series(0.05, index=returns.index)

    assert get_observations_per_year(returns).to_dict() == {
        "Crypto": 365.0,
        "Stock": 252.0,
    }

    per_observation = get_risk_free_rate_per_observation(annualized_rate, returns)
    stock_dates = returns["Stock"].dropna().index

    # A weekday-only ticker earns a trading day of the rate with every return.
    assert np.allclose(per_observation.loc[stock_dates, "Stock"], 1.05 ** (1 / 252) - 1)
    assert per_observation["Stock"].isna().sum() == returns["Stock"].isna().sum()
    # A ticker trading every day earns the annualized rate over a year of returns.
    in_2023 = returns.index.year == 2023  # noqa: PLR2004
    assert (1 + per_observation.loc[in_2023, "Crypto"]).prod() == pytest.approx(1.05)

    # The rate of a period the data covers in part is that of the calendar days it covers,
    # from the date before its first return (Sunday 2 January) to its last return.
    stock_only = returns[["Stock"]].loc[:"2022-03-31"]
    yearly = compound_risk_free_rate(annualized_rate, stock_only, "yearly")
    assert yearly.iloc[0] == pytest.approx(1.05 ** (88 / 365) - 1)

    # A trading day of the rate per date of stock-only data, as for the daily returns.
    per_date = get_risk_free_rate_per_date(annualized_rate, stock_only)
    assert np.allclose(per_date, 1.05 ** (1 / 252) - 1)


def test_only_a_ticker_trading_on_every_weekend_counts_365_days_a_year():
    returns = _stock_and_crypto_returns()
    index = returns.index
    # An exchange open from Sunday to Thursday and a stock with a one-off Saturday session.
    returns["Sunday to Thursday"] = np.where(
        (index.dayofweek < 4) | (index.dayofweek == 6), 0.001, np.nan  # noqa: PLR2004
    )
    returns["One Saturday"] = returns["Stock"]
    returns.loc["2022-01-08", "One Saturday"] = 0.002

    assert get_observations_per_year(returns).to_dict() == {
        "Crypto": 365.0,
        "Stock": 252.0,
        "Sunday to Thursday": 252.0,
        "One Saturday": 252.0,
    }


def test_the_period_risk_free_rate_does_not_depend_on_the_tickers():
    returns = _stock_and_crypto_returns()
    annualized_rate = pd.Series(0.05, index=returns.index)

    mixed = compound_risk_free_rate(annualized_rate, returns, "monthly")
    stock = compound_risk_free_rate(annualized_rate, returns[["Stock"]], "monthly")
    days_in_month = returns.index.asfreq("M").value_counts().sort_index()

    # Every calendar day earns 1 / 365 of the year: a full year earns the annualized rate
    # and a month its days, whether the dates are the trading days of the stock or every
    # calendar day of the cryptocurrency. Only a weekend at the turn of a month moves to
    # the month of the next trading day.
    yearly = compound_risk_free_rate(annualized_rate, returns, "yearly")
    assert yearly[pd.Period("2023", "Y")] == pytest.approx(0.05, rel=1e-12)
    assert mixed.to_numpy() == pytest.approx(
        (1.05 ** (days_in_month / 365) - 1).to_numpy(), rel=1e-12
    )
    weekday_turns = [
        month
        for month in stock.index[1:]
        if month.end_time.dayofweek < 5  # noqa: PLR2004
        and (month - 1).end_time.dayofweek < 5  # noqa: PLR2004
    ]
    assert len(weekday_turns) > 3  # noqa: PLR2004
    assert stock[weekday_turns].to_numpy() == pytest.approx(
        mixed[weekday_turns].to_numpy(), rel=1e-12
    )


def test_the_risk_free_rate_per_ticker_is_in_date_order_with_a_later_listing_first():
    days = pd.period_range("2021-01-01", "2022-12-30", freq="B")
    rng = np.random.default_rng(4)
    returns = pd.DataFrame(
        {
            "Listed Later": np.where(
                days >= pd.Period("2022-01-03", "B"),
                rng.normal(0, 0.01, len(days)),
                np.nan,
            ),
            "Listed Earlier": rng.normal(0, 0.01, len(days)),
        },
        index=days.asfreq("D"),
    )
    historical_data = {
        "daily": pd.concat(
            {"Return": returns, "Adj Close": (1 + returns).cumprod()}, axis=1
        )
    }
    annualized_rate = {"daily": pd.Series(0.04, index=returns.index)}

    monthly = get_risk_free_rate_per_ticker(annualized_rate, historical_data, "monthly")
    swapped = get_risk_free_rate_per_ticker(
        annualized_rate,
        {"daily": historical_data["daily"].iloc[:, [1, 0, 3, 2]]},
        "monthly",
    )

    # The periods of 2021 only the second ticker has come first, not after those of 2022,
    # so rolling windows over the rows cover consecutive periods.
    assert monthly.index.is_monotonic_increasing
    assert len(monthly) == 24  # noqa: PLR2004
    pd.testing.assert_frame_equal(monthly, swapped[monthly.columns])


def test_the_pain_index_averages_the_drawdowns_over_every_day():
    # Wealth 0.9, 0.99, 1.0395, 0.93555: drawdowns of 10%, 1%, 0 (a new high) and 10%.
    returns = pd.Series([-0.1, 0.1, 0.05, -0.1])

    assert risk_model.get_pain_index(returns) == pytest.approx(
        (0.1 + 0.01 + 0 + 0.1) / 4
    )


def test_the_pain_index_restarts_every_period_and_skips_a_later_listing():
    index = pd.MultiIndex.from_tuples(
        [("2021", 1), ("2021", 2), ("2022", 3), ("2022", 4)]
    )
    returns = pd.DataFrame(
        {"A": [-0.1, 0.1, -0.2, 0.0], "B": [np.nan, -0.1, -0.2, 0.0]}, index=index
    )

    result = risk_model.get_pain_index(returns)

    assert result.loc["2021", "A"] == pytest.approx((0.1 + 0.01) / 2)
    # The row before B's first return is no observation of it.
    assert result.loc["2021", "B"] == pytest.approx(0.1)
    assert result.loc["2022", "A"] == pytest.approx(0.2)


def test_the_pain_index_matches_the_bacon_definition():
    # PerformanceAnalytics' PainIndex (Bacon, 2013): sum(|D_i|) / n with D_i the drawdown
    # from the highest wealth so far, written out with plain loops. The first return is a
    # loss from the starting wealth of 1.
    rng = np.random.default_rng(3)
    returns = pd.DataFrame(
        rng.normal(0.0003, 0.015, (300, 2)),
        columns=["A", "B"],
        index=pd.period_range("2022-01-03", periods=300, freq="D"),
    )

    expected = {}
    for column in returns:
        wealth, peak, total = 1.0, 1.0, 0.0
        for value in returns[column]:
            wealth *= 1 + value
            peak = max(peak, wealth)
            total += abs(wealth / peak - 1)
        expected[column] = total / len(returns)

    result = risk_model.get_pain_index(returns)

    assert result.to_dict() == pytest.approx(expected)
    # The Pain Index is never above the Ulcer Index on the same running peak.
    assert (result <= risk_model.get_ui(returns, None)).all()


def test_the_pain_index_of_a_stock_next_to_a_seven_day_ticker_is_the_stock_alone():
    returns = _stock_and_crypto_returns()
    within = returns.set_axis(
        pd.MultiIndex.from_arrays([returns.index.asfreq("Y"), returns.index])
    )
    stock = returns[["Stock"]].dropna()
    stock_within = stock.set_axis(
        pd.MultiIndex.from_arrays([stock.index.asfreq("Y"), stock.index])
    )

    assert risk_model.get_pain_index(returns)["Stock"] == pytest.approx(
        risk_model.get_pain_index(stock)["Stock"]
    )
    pd.testing.assert_series_equal(
        risk_model.get_pain_index(within)["Stock"],
        risk_model.get_pain_index(stock_within)["Stock"],
    )


def test_the_pain_index_skips_the_rows_before_a_later_listing():
    returns = _late_listing_returns()
    alone = returns[["Later"]].dropna()

    assert risk_model.get_pain_index(returns)["Later"] == pytest.approx(
        risk_model.get_pain_index(alone)["Later"]
    )
    assert np.isnan(risk_model.get_pain_index(pd.Series([np.nan, np.nan])))


def test_moves_that_differ_in_the_last_bits_are_ties():
    # Changes of a rate quoted to four decimals: 0.001 comes out as 0.000999...994 or
    # 0.001...009 depending on the levels, which must still rank as the same move.
    levels = pd.Series(
        [4.2513, 4.2523, 4.2533, 4.2543, 4.2553, 4.2443, 4.2453],
        index=pd.date_range("2024-01-01", periods=7),
    )
    returns = levels.diff().dropna()

    significance = risk_model.get_move_significance(returns, window=2)

    # Four earlier moves of +0.001 and one of -0.011: the last +0.001 is a tie with four.
    assert significance["Percentile"].iloc[-1] == pytest.approx((0 + 0.5 * 4) / 5)
    assert significance["Largest Since"].iloc[-1] == "2024-01-05"


def test_a_drawdown_back_at_exactly_its_old_high_is_recovered():
    # Closes of 100, 99, 95 and 100 compound to a wealth of 0.9999999999999999, which read
    # as below the peak left the drawdown open (to 101) or merged it with the next one.
    dates = pd.period_range("2024-01-01", periods=7, freq="D")
    closes = pd.DataFrame(
        {
            "New High": [100.0, 99.0, 95.0, 100.0, 101.0, 101.0, 101.0],
            "Falls Again": [100.0, 99.0, 95.0, 100.0, 98.0, 96.0, 100.0],
        },
        index=dates,
    )
    returns = closes.pct_change()
    assert (1 + returns["New High"].iloc[1:4]).prod() < 1

    table = risk_model.get_drawdown_table(returns)

    new_high = table.loc["New High"].iloc[0]
    assert new_high["Recovery"] == pd.Period("2024-01-04", freq="D")
    assert new_high["Length"] == 3
    assert len(table.loc["New High"]) == 1

    falls_again = table.loc["Falls Again"]
    assert len(falls_again) == 2
    assert falls_again["Recovery"].tolist() == [
        pd.Period("2024-01-04", freq="D"),
        pd.Period("2024-01-07", freq="D"),
    ]
    assert (risk_model.get_drawdown_series(returns).iloc[[0, 3]] == 0).all().all()

    # The recovery time of the deepest drawdown ends on the same row (one period after the
    # trough rather than two), per column and per series.
    measured = returns.iloc[1:]
    assert risk_model.get_max_drawdown_recovery_time(measured)["New High"] == 1
    assert risk_model.get_max_drawdown_recovery_time(measured["New High"]) == 1


def test_get_autocorrelation_is_missing_at_lags_without_pairs():
    # Four values have pairs up to lag 3 (as statsmodels' acf, which stops there); the
    # empty sums beyond gave 0.
    result = risk_model.get_autocorrelation(pd.Series([1.0, 2.0, 3.0, 4.0]), lags=6)

    np.testing.assert_allclose(result.loc[1:3], [0.25, -0.3, -0.45])
    assert result.loc[4:].isna().all()


def test_get_compounded_volatility_by_hand():
    dates = pd.PeriodIndex(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"], freq="D"
    )
    returns = pd.Series([0.01, -0.02, 0.03, 0.005], index=dates)
    days = 252 / 12
    growth = np.prod(1 + returns.to_numpy()) ** (1 / 4)
    expected = np.sqrt((returns.var(ddof=1) + growth**2) ** days - growth ** (2 * days))
    volatility = risk_model.get_compounded_volatility(returns, "monthly")
    assert volatility.iloc[0] == pytest.approx(expected)
    # Small returns compound to about the square root of time scaling.
    small = returns / 100
    assert risk_model.get_compounded_volatility(small, "monthly").iloc[
        0
    ] == pytest.approx(small.std() * np.sqrt(days), rel=0.01)


def test_a_window_of_no_observations_raises():
    returns = pd.Series(np.random.default_rng(1).normal(0, 0.01, 50))

    for window in (0, -1, 2.5):
        with pytest.raises(InvalidArgumentError, match="window is a number"):
            cvar_model.get_cvar_historic(returns, 0.05, window=window)
        with pytest.raises(InvalidArgumentError, match="window is a number"):
            risk_model.get_conditional_drawdown_at_risk(returns, 0.05, window=window)
    # A value at risk of one return is that return; a skewness needs more.
    assert var_model.get_var_historic(returns, 0.05, window=1).notna().all()
    with pytest.raises(InvalidArgumentError, match="at least 2"):
        risk_model.get_skewness(returns, window=1)
