"""Fractional Differentiation Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.econometrics.stationarity import fractional_model


@pytest.fixture
def log_prices():
    generator = np.random.default_rng(7)
    return pd.Series(
        np.log(100) + np.cumsum(0.01 * generator.standard_normal(800)),
        index=pd.bdate_range("2015-01-01", periods=800),
    )


def test_weights_follow_the_binomial_series():
    # (1 - L)^0.5 = 1 - 0.5L - 0.125L² - 0.0625L³ - ...
    np.testing.assert_allclose(
        fractional_model.get_fractional_weights(0.5, size=4),
        [1, -0.5, -0.125, -0.0625],
    )
    # A whole d ends after d + 1 weights: (1 - L)² = 1 - 2L + L².
    np.testing.assert_allclose(
        fractional_model.get_fractional_weights(2, threshold=1e-5), [1, -2, 1]
    )


def test_fixed_width_weights_stop_at_the_threshold():
    weights = fractional_model.get_fractional_weights(0.4, threshold=1e-4)
    assert np.abs(weights[-1]) >= 1e-4
    # The next weight is below the threshold.
    assert abs(weights[-1] * (len(weights) - 1 - 0.4) / len(weights)) < 1e-4
    # A size caps the window.
    assert (
        len(fractional_model.get_fractional_weights(0.4, size=5, threshold=1e-9)) == 5
    )


def test_whole_orders_give_the_series_and_its_differences(log_prices):
    pd.testing.assert_series_equal(
        fractional_model.get_fractional_difference(log_prices, d=0), log_prices
    )
    pd.testing.assert_series_equal(
        fractional_model.get_fractional_difference(log_prices, d=1),
        log_prices.diff(),
    )


def test_fixed_width_difference_is_the_weighted_window(log_prices):
    weights = fractional_model.get_fractional_weights(0.35, threshold=1e-3)
    result = fractional_model.get_fractional_difference(
        log_prices, 0.35, threshold=1e-3
    )

    assert result.iloc[: len(weights) - 1].isna().all()
    expected = np.dot(weights, log_prices.to_numpy()[::-1][: len(weights)])
    assert result.iloc[-1] == pytest.approx(expected, rel=1e-12)


def test_expanding_difference_skips_until_the_weight_loss_is_small(log_prices):
    d, threshold = 0.35, 0.01
    result = fractional_model.get_fractional_difference(
        log_prices, d, method="expanding", threshold=threshold
    )
    weights = np.abs(fractional_model.get_fractional_weights(d, size=len(log_prices)))
    weight_from = weights[::-1].cumsum()[::-1] / weights.sum()

    skip = int(np.sum(weight_from > threshold))
    assert result.iloc[:skip].isna().all()
    assert result.iloc[skip:].notna().all()
    position = len(log_prices) - 1
    expected = np.dot(
        fractional_model.get_fractional_weights(d, size=position + 1),
        log_prices.to_numpy()[::-1],
    )
    assert result.iloc[-1] == pytest.approx(expected, rel=1e-12)


def test_missing_values_are_left_out(log_prices):
    gaps = log_prices.copy()
    gaps.iloc[[100, 400]] = np.nan
    result = fractional_model.get_fractional_difference(gaps, 0.4, threshold=1e-3)

    assert result.index.equals(gaps.index)
    assert np.isnan(result.iloc[400])
    pd.testing.assert_series_equal(
        result.dropna(),
        fractional_model.get_fractional_difference(
            gaps.dropna(), 0.4, threshold=1e-3
        ).dropna(),
    )


def test_minimum_d_of_a_random_walk(log_prices):
    table, minimum_d = fractional_model.get_minimum_fractional_difference(log_prices)

    assert 0 < minimum_d <= 1
    assert table.loc[minimum_d, "Stationary"]
    # Every d tried below d* fails the test.
    assert not table.loc[table.index < minimum_d, "Stationary"].any()
    # The memory kept falls as d rises.
    assert table.loc[0, "Correlation"] == pytest.approx(1)
    assert table.loc[1, "Correlation"] < table.loc[minimum_d, "Correlation"]


def test_minimum_d_of_a_stationary_series_is_zero():
    noise = pd.Series(
        np.random.default_rng(1).standard_normal(300),
        index=pd.bdate_range("2020-01-01", periods=300),
    )
    _, minimum_d = fractional_model.get_minimum_fractional_difference(noise)

    assert minimum_d == 0


def test_invalid_inputs(log_prices):
    with pytest.raises(ValueError):
        fractional_model.get_fractional_difference(log_prices, d=0.4, method="other")
    with pytest.raises(ValueError):
        fractional_model.get_fractional_weights(-0.5, size=10)
    with pytest.raises(ValueError):
        fractional_model.get_minimum_fractional_difference(log_prices, confidence=95)


def test_fixed_width_window_longer_than_the_series_gives_no_values():
    # At d = 0.4 and 1e-5 the window spans about 1,460 weights: 757 observations hold no
    # full window, so no value is computed from a window cut short at the series length.
    log_prices = pd.Series(np.log(100) + np.cumsum(np.full(757, 0.001)))
    result = fractional_model.get_fractional_difference(log_prices, d=0.4)

    assert len(result) == len(log_prices) and result.isna().all()

    # A window that fits exactly gives exactly one value.
    width = len(fractional_model.get_fractional_weights(0.4, threshold=1e-3))
    fits = fractional_model.get_fractional_difference(
        log_prices.iloc[:width], d=0.4, threshold=1e-3
    )
    assert fits.notna().sum() == 1
