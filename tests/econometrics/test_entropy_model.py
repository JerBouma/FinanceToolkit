"""Entropy Model Tests"""

import math

import numpy as np
import pandas as pd
import pytest

from financetoolkit.econometrics.statistics import entropy_model


def returns() -> np.ndarray:
    """1,000 AR(1) returns with Student-t shocks, the series the reference values are of."""
    shocks = np.random.default_rng(0).standard_t(4, 1000) * 0.01
    values = np.zeros(1000)
    for index in range(1, 1000):
        values[index] = 0.2 * values[index - 1] + shocks[index]
    return values


# nolds 0.6.3 and antropy 0.2.3 on returns() with the same parameters (tolerance 0.2 times
# the standard deviation with ddof=1, nolds' sampen with closed=True).
@pytest.mark.parametrize(
    ("function", "kwargs", "expected"),
    [
        (entropy_model.get_sample_entropy, {}, 1.979090174788766),
        (entropy_model.get_approximate_entropy, {}, 1.6089792475859346),
        (
            entropy_model.get_permutation_entropy,
            {"order": 4, "delay": 2},
            0.9976312929853794,
        ),
        (entropy_model.get_spectral_entropy, {}, 0.9347441212001666),
        (
            entropy_model.get_spectral_entropy,
            {"estimator": "welch", "normalize": False},
            6.866799420979284,
        ),
        (entropy_model.get_svd_entropy, {"order": 4}, 0.9957591957605808),
    ],
)
def test_entropies_equal_nolds_and_antropy(function, kwargs, expected):
    assert function(returns(), **kwargs) == pytest.approx(expected, rel=1e-12)


def test_lempel_ziv_complexity_equals_antropy():
    # antropy takes the symbols as given; the first return is exactly 0, which the binary
    # encoding would leave out.
    symbols = (returns() > 0).astype(int)
    assert entropy_model.get_lempel_ziv_complexity(symbols) == pytest.approx(
        1.0563731341741813, rel=1e-12
    )


def test_lempel_ziv_complexity_of_a_periodic_message_is_small():
    symbols = np.tile([0, 1], 200)
    assert (
        entropy_model.get_lempel_ziv_complexity(symbols, normalize=False) == 3
    )  # noqa: PLR2004


def test_sample_entropy_counts_matches_by_hand():
    values = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 3.0, 1.0, 2.0])
    radius = 0.2 * np.std(values, ddof=1)
    templates = np.array([values[i : i + 3] for i in range(len(values) - 2)])
    pairs = [
        (i, j) for i in range(len(templates)) for j in range(len(templates)) if i != j
    ]
    matches = sum(
        np.abs(templates[i, :2] - templates[j, :2]).max() <= radius for i, j in pairs
    )
    longer = sum(np.abs(templates[i] - templates[j]).max() <= radius for i, j in pairs)

    assert entropy_model.get_sample_entropy(values) == pytest.approx(
        -np.log(longer / matches)
    )


def test_sample_entropy_without_matches_is_nan():
    assert np.isnan(
        entropy_model.get_sample_entropy(np.array([1.0, 5.0, 2.0, 8.0, 3.0]))
    )


def test_permutation_entropy_of_a_trend_is_zero_and_of_noise_near_one():
    assert entropy_model.get_permutation_entropy(np.arange(100.0)) == 0
    noise = np.random.default_rng(1).normal(size=5000)
    assert entropy_model.get_permutation_entropy(noise) == pytest.approx(1, abs=0.01)
    assert entropy_model.get_permutation_entropy(
        noise, normalize=False
    ) == pytest.approx(np.log2(math.factorial(3)), abs=0.02)


@pytest.mark.parametrize(
    ("encoding", "expected"),
    [
        ("binary", [1, 0, 1, 0]),
        ("quantile", [3, 0, 3, 2, 1]),
        ("sigma", [2, 0, 1, 0, 0]),
    ],
)
def test_encode_series(encoding, expected):
    values = np.array([0.02, -0.01, 0.01, 0.0, -0.005])
    np.testing.assert_array_equal(
        entropy_model.encode_series(values, encoding, bins=4), expected
    )


def test_encode_series_rejects_an_unknown_encoding():
    with pytest.raises(ValueError, match="encoding"):
        entropy_model.encode_series(np.array([0.1, -0.1]), "unknown")


# The plug-in entropy of a balanced binary message is one bit per symbol (every word counted,
# the last one included) and its Lempel-Ziv entropy the number of phrases per symbol.
def test_entropies_of_an_encoded_message():
    symbols = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1])
    shares = np.array([8, 8]) / 16

    assert entropy_model.get_shannon_entropy(symbols) == pytest.approx(
        -np.sum(shares * np.log2(shares))
    )
    # Phrases: 0, 1, 10, 100, 11, 101, 00, 01, the last stretch 1 is no new phrase.
    assert entropy_model.get_lempel_ziv_entropy(symbols) == pytest.approx(8 / 16)
    assert (
        entropy_model.get_shannon_entropy(np.zeros(20, dtype=int), word_length=2) == 0
    )


def test_kontoyiannis_entropy_by_hand():
    symbols = np.array([0, 1, 0, 1, 1, 0])
    # Position 1: "1" is new (L = 1); 2: "01" seen at 0 (L = 3); 3: "1" seen, "11" not (L = 2).
    expected = np.mean([np.log2(2) / 1, np.log2(3) / 3, np.log2(4) / 2])
    assert entropy_model.get_kontoyiannis_entropy(symbols) == pytest.approx(expected)


def test_hjorth_parameters():
    values = returns()
    parameters = entropy_model.get_hjorth_parameters(pd.Series(values))
    mobility = np.sqrt(np.var(np.diff(values), ddof=1) / np.var(values, ddof=1))

    assert parameters["Activity"] == pytest.approx(np.var(values, ddof=1))
    assert parameters["Mobility"] == pytest.approx(mobility)
    assert parameters["Zero Crossings"] == np.sum(
        np.diff(np.sign(values[values != 0])) != 0
    )
    zeros = entropy_model.get_hjorth_parameters(np.array([-1.0, 0.0, -1.0, 2.0, 0.0]))
    assert zeros["Zero Crossings"] == 1


def share_prices() -> np.ndarray:
    """80 closes of a random walk, the series pandas-ta-classic's entropy values are of."""
    rng = np.random.default_rng(7)
    return 100 + rng.normal(0, 1, 80).cumsum()


def test_share_entropy_matches_pandas_ta_classic():
    result = entropy_model.get_share_entropy(share_prices(), 10, 2)

    assert len(result) == 80 - 10 + 1
    assert result[-3:] == pytest.approx([3.321847, 3.321866, 3.321854], abs=1e-6)
    # Equal prices give the maximum, log2 of the window, or 1 when normalized.
    assert entropy_model.get_share_entropy(np.full(8, 4.0), 8)[-1] == pytest.approx(3)
    assert entropy_model.get_share_entropy(np.full(8, 4.0), 8, normalize=True)[
        -1
    ] == pytest.approx(1)
    assert np.isnan(entropy_model.get_share_entropy(np.array([1.0, -1.0, 2.0]), 3)[-1])


def test_share_entropy_of_the_whole_sample_equals_numpy():
    prices = share_prices()
    shares = prices / prices.sum()
    entropy = -(shares * np.log(shares)).sum()

    assert entropy_model.get_share_entropy(prices) == pytest.approx(
        entropy / np.log(2), abs=1e-12
    )
    assert entropy_model.get_share_entropy(prices, base=np.e) == pytest.approx(
        entropy, abs=1e-12
    )
    assert entropy_model.get_share_entropy(prices, normalize=True) == pytest.approx(
        entropy / np.log(len(prices)), abs=1e-12
    )
    assert np.isnan(entropy_model.get_share_entropy(np.array([])))
