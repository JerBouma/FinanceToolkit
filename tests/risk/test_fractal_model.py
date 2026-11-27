"""Fractal Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.helpers import delay_embedding
from financetoolkit.risk.distribution import fractal_model


def returns() -> np.ndarray:
    """1,000 AR(1) returns with Student-t shocks, the series the reference values are of."""
    shocks = np.random.default_rng(0).standard_t(4, 1000) * 0.01
    values = np.zeros(1000)
    for index in range(1, 1000):
        values[index] = 0.2 * values[index - 1] + shocks[index]
    return values


def henon(number: int = 2000) -> np.ndarray:
    """The x coordinate of the Hénon map (a = 1.4, b = 0.3) after 100 transient steps."""
    x, y, values = 0.1, 0.1, []
    for step in range(number + 100):
        x, y = 1 - 1.4 * x * x + y, 0.3 * x
        if step >= 100:  # noqa: PLR2004
            values.append(x)
    return np.array(values)


def test_get_generalized_hurst_exponent_includes_max_lag():
    data = pd.Series(np.cumsum(np.random.default_rng(0).normal(0, 1, 2000)))
    values = data.to_numpy()
    lags = np.arange(2, 11)
    standard_deviations = [np.std(values[lag:] - values[:-lag]) for lag in lags]
    expected = np.polyfit(np.log(lags), np.log(standard_deviations), 1)[0]

    assert np.isclose(
        fractal_model.get_generalized_hurst_exponent(data, max_lag=10), expected
    )
    assert (
        abs(fractal_model.get_generalized_hurst_exponent(data) - 0.5) < 0.1
    )  # noqa: PLR2004
    assert (
        abs(fractal_model.get_generalized_hurst_exponent(data, q=1) - 0.5) < 0.1
    )  # noqa: PLR2004


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_generalized_hurst_exponent_bounds_lags_to_available_differences():
    data = pd.Series([0.0, 1.0, 0.5, 2.0, 1.0, 3.0, 2.5, 4.0])
    values = data.to_numpy()
    lags = np.arange(2, len(values) - 1)
    deviations = [np.std(values[lag:] - values[:-lag]) for lag in lags]
    expected = np.polyfit(np.log(lags), np.log(deviations), 1)[0]

    assert fractal_model.get_generalized_hurst_exponent(data) == pytest.approx(expected)


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("values", [[], [1.0], [1.0, 2.0, 3.0, 4.0], [2.0] * 30])
def test_generalized_hurst_exponent_without_two_usable_lags(values):
    assert np.isnan(
        fractal_model.get_generalized_hurst_exponent(pd.Series(values, dtype=float))
    )


def test_window_sizes_grow_by_a_fifth_up_to_a_tenth():
    np.testing.assert_array_equal(
        fractal_model.get_window_sizes(1000),
        [10, 12, 14, 17, 20, 24, 29, 35, 42, 51, 61, 74, 89],
    )
    assert len(fractal_model.get_window_sizes(50)) == 0


def test_expected_rescaled_range_switches_to_the_approximation_above_340():
    expected = fractal_model.get_expected_rescaled_range(np.array([10, 340, 341]))
    total = np.sum(np.sqrt((10 - np.arange(1, 10)) / np.arange(1, 10)))
    # Gamma(4.5) = 105 sqrt(pi) / 16 and Gamma(5) = 24.
    assert expected[0] == pytest.approx(
        9.5 / 10 * (105 * np.sqrt(np.pi) / 16) / 24 / np.sqrt(np.pi) * total
    )
    assert expected[2] / expected[1] == pytest.approx(1, abs=0.005)


# nolds 0.6.3 and antropy 0.2.3 on returns() (or their cumulative sum) with the same window
# sizes and least squares fits: hurst_rs(unbiased=False), dfa(overlap=False), lyap_r with
# lag=1, and lyap_e with matrix_dim equal to emb_dim (m = 1, where both multiply the same maps).
NOLDS_WINDOW_SIZES = [10, 12, 14, 17, 20, 24, 29, 35, 42, 51, 61, 74, 89]


@pytest.mark.parametrize(
    ("function", "kwargs", "expected"),
    [
        (
            fractal_model.get_rescaled_range_hurst_exponent,
            {"window_sizes": NOLDS_WINDOW_SIZES},
            0.5018591865590569,
        ),
        (
            fractal_model.get_detrended_fluctuation,
            {"window_sizes": NOLDS_WINDOW_SIZES},
            0.49349605925505724,
        ),
        (
            fractal_model.get_largest_lyapunov_exponent,
            {"embedding": 5, "min_separation": 5, "trajectory_length": 15},
            0.07006434095209663,
        ),
    ],
)
def test_measures_of_returns_equal_nolds(function, kwargs, expected):
    assert function(returns(), **kwargs) == pytest.approx(expected, rel=1e-10)


def test_lyapunov_spectrum_equals_nolds_with_a_step_of_one():
    np.testing.assert_allclose(
        fractal_model.get_lyapunov_spectrum(returns(), embedding=4).to_numpy(),
        [
            0.5694123246944884,
            0.1400490568326096,
            -0.09937019510446587,
            -0.6475838500190857,
        ],
        rtol=1e-8,
    )


@pytest.mark.parametrize(
    ("function", "kwargs", "expected"),
    [
        (fractal_model.get_higuchi_fractal_dimension, {"kmax": 8}, 1.431607890221108),
        (fractal_model.get_katz_fractal_dimension, {}, 1.8831130761767794),
        (fractal_model.get_petrosian_fractal_dimension, {}, 1.0239298490536877),
    ],
)
def test_fractal_dimensions_equal_antropy(function, kwargs, expected):
    assert function(np.cumsum(returns()), **kwargs) == pytest.approx(expected, rel=1e-9)


def test_correlation_dimension_equals_a_direct_count():
    values = returns()[:400]
    multiples = np.array([0.5, 0.7, 1.0, 1.4])
    vectors = np.array([values[i : i + 2] for i in range(len(values) - 1)])
    distances = np.sqrt(((vectors[:, None] - vectors[None]) ** 2).sum(axis=2))
    pairs = distances[np.triu_indices(len(vectors), k=1)]
    radii = multiples * np.std(values, ddof=1)
    sums = np.array([(pairs < radius).mean() for radius in radii])
    expected = np.polyfit(np.log(radii), np.log(sums), 1)[0]

    assert fractal_model.get_correlation_dimension(
        values, embedding=2, radii=multiples
    ) == pytest.approx(expected)


def test_chaos_measures_of_the_henon_map():
    values = henon()
    # The Hénon attractor has a correlation dimension of about 1.21 and Lyapunov exponents of
    # about 0.42 and -1.62 (Grassberger and Procaccia, 1983).
    assert fractal_model.get_correlation_dimension(
        values, embedding=2
    ) == pytest.approx(1.21, abs=0.05)
    assert fractal_model.get_largest_lyapunov_exponent(
        values, embedding=2, min_separation=10, trajectory_length=10
    ) == pytest.approx(0.42, abs=0.03)
    spectrum = fractal_model.get_lyapunov_spectrum(
        values, embedding=3, matrix_dimension=2
    )
    assert spectrum["Exponent 1"] == pytest.approx(0.42, abs=0.03)


def test_hurst_exponents_of_independent_returns_are_near_a_half():
    noise = np.random.default_rng(3).normal(size=4000)
    assert fractal_model.get_rescaled_range_hurst_exponent(noise) == pytest.approx(
        0.5, abs=0.05
    )
    assert fractal_model.get_detrended_fluctuation(noise) == pytest.approx(
        0.5, abs=0.05
    )
    assert fractal_model.get_higuchi_fractal_dimension(
        np.cumsum(noise)
    ) == pytest.approx(1.5, abs=0.05)


def test_lyapunov_spectrum_rejects_a_mismatched_embedding():
    with pytest.raises(ValueError, match="multiple"):
        fractal_model.get_lyapunov_spectrum(returns(), embedding=10, matrix_dimension=3)


def test_memory_measures_are_nan_without_variation():
    # Rounding noise around a constant return is no signal of memory.
    constant = np.full(500, 0.001)

    assert np.isnan(fractal_model.get_rescaled_range_hurst_exponent(constant))
    assert np.isnan(fractal_model.get_detrended_fluctuation(constant))
    assert np.isnan(fractal_model.get_rescaled_range_hurst_exponent(np.array([])))
    # A series without values has no Lyapunov exponent rather than an FFT error.
    assert fractal_model.get_mean_period(np.array([np.nan, np.nan])) == 1
    assert np.isnan(fractal_model.get_largest_lyapunov_exponent(np.full(300, np.nan)))


def _rosenstein_by_every_pair(values, embedding, separation, trajectory_length=20):
    """The largest Lyapunov exponent with the nearest neighbours from every pair."""
    vectors = delay_embedding(values, embedding, 1)
    starts = len(vectors) - trajectory_length + 1
    followed = vectors[:starts]
    distances = ((followed[:, None, :] - followed[None, :, :]) ** 2).sum(axis=2)
    origins = np.arange(starts)
    distances[np.abs(origins[:, None] - origins[None, :]) <= separation] = np.inf
    neighbours = distances.argmin(axis=1)
    divergence = [
        np.mean(np.log(gaps[gaps > 0]))
        for gaps in (
            np.linalg.norm(vectors[origins + k] - vectors[neighbours + k], axis=1)
            for k in range(trajectory_length)
        )
    ]
    return np.polyfit(np.arange(trajectory_length), divergence, 1)[0]


def test_the_lyapunov_neighbours_of_the_tree_are_those_of_every_pair():
    rng = np.random.default_rng(5)
    # Half of the values are 0, so many vectors are equally near each other.
    repeated = np.where(rng.random(1500) < 0.5, 0.0, rng.normal(size=1500))
    for values in (returns(), rng.normal(size=1500), repeated):
        for embedding, separation in ((2, 3), (4, 5), (10, 0)):
            assert fractal_model.get_largest_lyapunov_exponent(
                values, embedding=embedding, min_separation=separation
            ) == pytest.approx(
                _rosenstein_by_every_pair(values, embedding, separation), abs=1e-12
            )
