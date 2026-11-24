"""Bubble Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.econometrics.stationarity import bubble_model


def window_adf(values: np.ndarray, lags: int) -> float:
    """The ADF t-statistic of one window by plain least squares, independent of the model."""
    changes = np.diff(values)
    rows = len(changes) - lags
    regressors = np.column_stack(
        [np.ones(rows), values[lags:-1]]
        + [changes[lags - lag : lags - lag + rows] for lag in range(1, lags + 1)]
    )
    target = changes[lags:]
    beta, *_ = np.linalg.lstsq(regressors, target, rcond=None)
    residuals = target - regressors @ beta
    covariance = (
        residuals
        @ residuals
        / (rows - regressors.shape[1])
        * np.linalg.inv(regressors.T @ regressors)
    )

    return beta[1] / np.sqrt(covariance[1, 1])


@pytest.fixture
def bubble_prices():
    # A random walk in logs with an explosive stretch from observation 110 to 139 and a
    # collapse after it.
    generator = np.random.default_rng(3)
    values = 4.6 + np.cumsum(0.01 * generator.standard_normal(200))
    values[110:140] += np.cumsum(np.full(30, 0.012)) * np.linspace(1, 3, 30)
    values[140:] += values[139] - values[140] - np.cumsum(np.full(60, 0.004))
    return pd.Series(values, index=pd.date_range("2010-01-01", periods=200, freq="W"))


def test_minimum_window_rule():
    # r0 = 0.01 + 1.8 / sqrt(T): 0.19 of 100 observations and 0.1 of 400, the windows of PSY (2015, Table 1).
    assert bubble_model.get_minimum_window(100) == 19
    assert bubble_model.get_minimum_window(400) == 40


@pytest.mark.parametrize("lags", [0, 2])
def test_recursive_statistics_match_a_loop(bubble_prices, lags):
    values = bubble_prices.to_numpy()[:90]
    window = 25
    forward, backward = bubble_model.get_recursive_adf_statistics(values, lags, window)

    assert np.isnan(forward[: window - 1]).all()
    for end in (window - 1, 50, 89):
        statistics = [
            window_adf(values[start : end + 1], lags)
            for start in range(end - window + 2)
        ]
        assert forward[end] == pytest.approx(statistics[0], rel=1e-8)
        assert backward[end] == pytest.approx(max(statistics), rel=1e-8)


def test_window_statistic_matches_statsmodels(bubble_prices):
    stattools = pytest.importorskip("statsmodels.tsa.stattools")
    values = bubble_prices.to_numpy()
    forward, _ = bubble_model.get_recursive_adf_statistics(values, 1, 30)

    expected = stattools.adfuller(values[:121], maxlag=1, regression="c", autolag=None)[
        0
    ]
    assert forward[120] == pytest.approx(expected, rel=1e-8)


def test_window_too_small_for_the_lags(bubble_prices):
    with pytest.raises(ValueError):
        bubble_model.get_recursive_adf_statistics(bubble_prices.to_numpy(), 10, 20)


def test_simulated_critical_values_are_near_the_published_ones():
    # Phillips, Shi and Yu (2015, Table 1): T = 100 and r0 = 0.4, 95% critical values 1.05
    # (SADF) and 1.48 (GSADF) from 5,000 replications. Between sets of 1,000 paths the 95%
    # quantile of the GSADF moves by about 0.1, so 2,000 paths are compared within 0.1.
    simulated = bubble_model.simulate_bubble_critical_values(
        100, 40, 0, 2000, 1, bubble_model.CONFIDENCE_LEVELS
    )

    assert simulated["SADF Critical Values"][1] == pytest.approx(1.05, abs=0.1)
    assert simulated["GSADF Critical Values"][1] == pytest.approx(1.48, abs=0.1)
    # The same seed gives the same simulation.
    again = bubble_model.simulate_bubble_critical_values.__wrapped__(
        100, 40, 0, 2000, 1, bubble_model.CONFIDENCE_LEVELS
    )
    np.testing.assert_array_equal(again["GSADF"], simulated["GSADF"])


def test_bubble_test_finds_the_explosive_stretch(bubble_prices):
    gsadf = bubble_model.get_bubble_test(bubble_prices, simulations=200)
    sadf = bubble_model.get_bubble_test(bubble_prices, method="sadf", simulations=200)

    assert gsadf["Explosive (5%)"]
    assert gsadf["P-Value"] < 0.05
    # GSADF searches every window SADF does, and more.
    assert gsadf["GSADF Statistic"] >= sadf["SADF Statistic"]
    assert gsadf["Minimum Window"] == bubble_model.get_minimum_window(200)


def test_bubble_test_of_a_random_walk():
    walk = pd.Series(np.cumsum(np.random.default_rng(11).standard_normal(150)))
    result = bubble_model.get_bubble_test(walk, simulations=200)

    assert not result["Explosive (5%)"]


def test_bubble_episodes_date_the_stretch(bubble_prices):
    episodes, sequence = bubble_model.get_bubble_episodes(
        bubble_prices, simulations=200
    )

    assert len(episodes) == 1
    episode = episodes.iloc[0]
    assert bubble_prices.index[110] <= episode["Start"] <= bubble_prices.index[139]
    assert episode["End"] >= bubble_prices.index[135]
    assert sequence["Bubble"].sum() >= episode["Duration"]
    assert sequence.index.equals(bubble_prices.index)


def test_chow_statistics_match_the_regression(bubble_prices):
    statistics = bubble_model.get_chow_dickey_fuller_statistics(bubble_prices)
    levels = bubble_prices.to_numpy() - bubble_prices.iloc[0]
    trimmed = int(np.floor(0.15 * len(levels)))

    assert statistics.iloc[:trimmed].isna().all()
    assert statistics.iloc[len(levels) - trimmed :].isna().all()
    for break_position in (trimmed, 100, len(levels) - 1 - trimmed):
        regressor = levels[:-1] * (np.arange(1, len(levels)) > break_position)
        changes = np.diff(levels)
        delta = regressor @ changes / (regressor @ regressor)
        residuals = changes - delta * regressor
        standard_error = np.sqrt(
            residuals @ residuals / (len(changes) - 1) / (regressor @ regressor)
        )
        assert statistics.iloc[break_position] == pytest.approx(
            delta / standard_error, rel=1e-10
        )


def test_chow_test_dates_the_break(bubble_prices):
    result = bubble_model.get_chow_dickey_fuller_test(bubble_prices, simulations=500)

    assert result["Explosive (5%)"]
    assert bubble_prices.index[90] <= result["Break Date"] <= bubble_prices.index[139]


def test_bubble_tests_give_no_p_value_without_a_statistic():
    # A constant level series has no ADF statistic in any window; its p-value is not 0
    # ("significant") but NaN, and the series is not explosive.
    constant = pd.Series(np.full(120, 4.0))
    result = bubble_model.get_bubble_test(constant, simulations=50)

    assert np.isnan(result["GSADF Statistic"]) and np.isnan(result["P-Value"])
    assert not result["Explosive (5%)"]

    chow = bubble_model.get_chow_dickey_fuller_test(constant, simulations=50)
    assert np.isnan(chow["SDFC Statistic"]) and np.isnan(chow["P-Value"])
    assert not chow["Explosive (5%)"]

    with pytest.raises(ValueError, match="too short"):
        bubble_model.get_chow_dickey_fuller_test(pd.Series([1.0, 2.0]))


def test_bubble_tests_default_to_2000_replications():
    series = pd.Series(np.cumsum(np.random.default_rng(3).standard_normal(60)))

    # Phillips, Shi and Yu (2015) simulate 2,000 random walks for the critical values.
    assert bubble_model.get_bubble_test(series, "sadf")["Replications"] == 2000
