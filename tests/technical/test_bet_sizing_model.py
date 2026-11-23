"""Bet Sizing Model Tests"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from financetoolkit.technicals.labeling import bet_sizing_model


def series(values: list[float]) -> pd.Series:
    """A Series on consecutive days."""
    return pd.Series(
        values,
        index=pd.period_range("2024-01-01", periods=len(values), freq="D"),
        dtype=float,
    )


# Three bets: a long one placed on day 0 for 3 days (open on days 0 to 2), a short one on day 1
# for 2 days (days 1 and 2) and a long one on day 3 for 2 days (days 3 and 4).
HOLDING_PERIODS = series([3, 2, np.nan, 2, np.nan, np.nan])
SIDES = series([1, -1, np.nan, 1, np.nan, np.nan])


def test_get_probability_bet_size_by_hand():
    probabilities = series([0.75, 0.75, 0.5, 0.9])
    sizes = bet_sizing_model.get_probability_bet_size(
        probabilities,
        predictions=series([1, 1, 1, 0]),
        sides=series([1, -1, 1, 1]),
    )
    # z = 0.25 / SQRT(0.1875) for a probability of 0.75 among two classes.
    expected = 2 * stats.norm.cdf(0.25 / np.sqrt(0.1875)) - 1
    assert sizes.iloc[0] == pytest.approx(expected)
    assert sizes.iloc[1] == pytest.approx(-expected)
    # A probability of 1 / K is a random guess and a prediction of 0 no bet.
    assert sizes.iloc[2] == 0
    assert sizes.iloc[3] == 0

    three_classes = bet_sizing_model.get_probability_bet_size(series([0.5]), classes=3)
    assert three_classes.iloc[0] == pytest.approx(
        2 * stats.norm.cdf((0.5 - 1 / 3) / 0.5) - 1
    )
    with pytest.raises(ValueError):
        bet_sizing_model.get_probability_bet_size(series([0.5]), classes=1)


def test_get_concurrent_sides_and_budget_by_hand():
    budget = bet_sizing_model.get_budget_bet_size(HOLDING_PERIODS, SIDES)

    assert budget["Long Bets"].tolist() == [1, 1, 1, 1, 1, 0]
    assert budget["Short Bets"].tolist() == [0, 1, 1, 0, 0, 0]
    # The most long bets open at once is 1, the most short bets 1.
    assert budget["Bet Size"].tolist() == [1, 0, 0, 1, 1, 0]

    # Without short bets the short side adds nothing.
    long_only = bet_sizing_model.get_budget_bet_size(HOLDING_PERIODS, SIDES.abs())
    assert long_only["Bet Size"].tolist() == [0.5, 1, 1, 0.5, 0.5, 0]


def test_get_average_active_bet_size_by_hand():
    sizes = series([0.6, -0.2, np.nan, 0.4, np.nan, np.nan])
    average = bet_sizing_model.get_average_active_bet_size(sizes, HOLDING_PERIODS)

    assert average.tolist() == pytest.approx([0.6, 0.2, 0.2, 0.4, 0.4, 0.0])


def test_get_discrete_bet_size():
    discrete = bet_sizing_model.get_discrete_bet_size(series([0.33, -0.76, 1.2]), 0.25)
    assert discrete.tolist() == [0.25, -0.75, 1.0]
    with pytest.raises(ValueError):
        bet_sizing_model.get_discrete_bet_size(series([0.5]), 0)


def test_get_hit_rate_by_hand():
    labels = series([1, 0, np.nan, 1, np.nan, np.nan])
    hit_rate = bet_sizing_model.get_hit_rate(labels, HOLDING_PERIODS, minimum_labels=1)

    # The first two bets close on day 3; the third closes on day 5.
    assert hit_rate.iloc[:3].isna().all()
    assert hit_rate.iloc[3:].tolist() == pytest.approx([0.5, 0.5, 2 / 3])
    assert (
        bet_sizing_model.get_hit_rate(labels, HOLDING_PERIODS, 3).iloc[:5].isna().all()
    )


def mixture_sample(size: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    first = generator.uniform(size=size) < 0.3
    return np.where(
        first, generator.normal(2.0, 0.5, size), generator.normal(-1.0, 1.0, size)
    )


def test_get_mixture_of_two_gaussians_matches_five_moments():
    values = mixture_sample(20_000, 1)
    fit = bet_sizing_model.get_mixture_of_two_gaussians(values)

    mixture = fit[["Mu 1", "Mu 2", "Sigma 1", "Sigma 2", "Probability 1"]].to_numpy()
    moments = bet_sizing_model._get_mixture_moments(mixture)
    sample = [np.mean(values**order) for order in range(1, 6)]
    assert moments == pytest.approx(sample, rel=1e-8)
    # A large sample of a mixture recovers its parameters.
    assert fit["Mu 1"] == pytest.approx(2.0, abs=0.1)
    assert fit["Mu 2"] == pytest.approx(-1.0, abs=0.1)
    assert fit["Probability 1"] == pytest.approx(0.3, abs=0.05)


def test_get_mixture_of_two_gaussians_does_not_depend_on_the_seed():
    values = mixture_sample(2_000, 2)
    fits = pd.DataFrame(
        {
            seed: bet_sizing_model.get_mixture_of_two_gaussians(values, seed=seed)
            for seed in range(5)
        }
    )
    parameters = fits.loc[["Mu 1", "Mu 2", "Sigma 1", "Sigma 2", "Probability 1"]]
    assert (parameters.max(axis=1) - parameters.min(axis=1)).max() < 1e-8

    assert bet_sizing_model.get_mixture_of_two_gaussians(np.ones(10)).isna().all()
    with pytest.raises(ValueError):
        bet_sizing_model.get_mixture_of_two_gaussians(values, runs=0)


def test_get_reserve_bet_size():
    generator = np.random.default_rng(3)
    count = 400
    holding_periods = pd.Series(
        generator.integers(1, 15, count).astype(float),
        index=pd.period_range("2020-01-01", periods=count, freq="D"),
    )
    sides = pd.Series(
        np.where(generator.uniform(size=count) < 0.55, 1.0, -1.0),
        index=holding_periods.index,
    )
    reserve = bet_sizing_model.get_reserve_bet_size(holding_periods, sides)

    net = reserve["Long Bets"] - reserve["Short Bets"]
    sizes = reserve["Bet Size"]
    assert (sizes[net == 0] == 0).all()
    assert (np.sign(sizes[net != 0]) == np.sign(net[net != 0])).all()
    assert sizes.abs().max() < 1
    # The size rises with the net number of open bets.
    by_net = sizes.groupby(net).first()
    assert by_net.is_monotonic_increasing


def test_dynamic_bet_size_by_hand():
    width = bet_sizing_model.get_sigmoid_width(10, 0.95)
    assert width == pytest.approx(100 * (1 / 0.95**2 - 1))
    assert bet_sizing_model.get_dynamic_bet_size(10.0, width) == pytest.approx(0.95)
    # The inverse price gives back the market price of a bet size.
    size = bet_sizing_model.get_dynamic_bet_size(115 - 108.0, width)
    assert bet_sizing_model.get_inverse_price(115, width, size) == pytest.approx(108)

    exponent = bet_sizing_model.get_power_width(0.4, 0.6)
    assert bet_sizing_model.get_dynamic_bet_size(
        0.4, exponent, "power"
    ) == pytest.approx(0.6)
    assert bet_sizing_model.get_inverse_price(
        0.8, exponent, 0.6, "power"
    ) == pytest.approx(0.4)

    # From 0 to 2 units of 100: the average of the prices at which 1 and 2 units are the size.
    limit = bet_sizing_model.get_limit_price(2, 0, 115, width, 100)
    expected = np.mean(
        [
            bet_sizing_model.get_inverse_price(115, width, units / 100)
            for units in (1, 2)
        ]
    )
    assert limit == pytest.approx(expected)
    # Reducing a short position from -3 to -1 averages the prices of -2 and -1 units.
    reduce = bet_sizing_model.get_limit_price(-1, -3, 115, width, 100)
    assert reduce == pytest.approx(
        np.mean(
            [
                bet_sizing_model.get_inverse_price(115, width, units / 100)
                for units in (-2, -1)
            ]
        )
    )
    assert np.isnan(bet_sizing_model.get_limit_price(5, 5, 115, width, 100))

    sizes = bet_sizing_model.get_target_bet_sizes(
        pd.Series([100.0, 120.0]), 115, current_position=pd.Series([0, 10])
    )
    assert sizes["Target Position"].tolist() == [
        int(100 * 15 / np.sqrt(width + 225)),
        int(-100 * 5 / np.sqrt(width + 25)),
    ]
    with pytest.raises(ValueError):
        bet_sizing_model.get_dynamic_bet_size(2.0, exponent, "power")
    with pytest.raises(ValueError):
        bet_sizing_model.get_target_bet_sizes(100.0, 115.0, method="unknown")
