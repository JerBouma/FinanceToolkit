"""Monte Carlo Model Tests"""

import numpy as np
import pytest

from financetoolkit.options import helpers, options_model
from financetoolkit.options.numerical import monte_carlo_model
from financetoolkit.options.pricing.american_model import get_qd_fixed_point
from financetoolkit.options.pricing.black_scholes_model import get_black_scholes
from financetoolkit.options.pricing.exotics_model import get_barrier_option
from financetoolkit.options.pricing.path_dependent_model import (
    get_discrete_monitoring_barrier,
)

EUROPEAN = (100.0, 100.0, 0.05, 0.2, 1.0, 0.02)


def test_brownian_bridge_increments_are_independent_standard_normals():
    increments = np.concatenate(
        list(
            helpers.draw_standard_normals(
                4096, 8, 2, sequence="sobol", seed=1, chunk_size=1000
            )
        )
    ).reshape(4096, 16)

    np.testing.assert_allclose(increments.mean(axis=0), 0, atol=0.01)
    np.testing.assert_allclose(np.cov(increments, rowvar=False), np.eye(16), atol=0.03)


def test_antithetic_draws_come_in_mirrored_pairs():
    draws = next(helpers.draw_standard_normals(10, 3, antithetic=True, seed=4))

    np.testing.assert_array_equal(draws[0::2], -draws[1::2])
    with pytest.raises(ValueError, match="even"):
        next(helpers.draw_standard_normals(9, 3, antithetic=True))


@pytest.mark.parametrize("put_option", [False, True])
@pytest.mark.parametrize(
    "variance_reduction, sequence",
    [
        ("antithetic", "pseudo_random"),
        ("control_variate", "pseudo_random"),
        (None, "sobol"),
        ("control_variate", "sobol"),
    ],
)
def test_variance_reduction_narrows_the_error_around_black_scholes(
    put_option, variance_reduction, sequence
):
    exact = get_black_scholes(*EUROPEAN, put_option)
    plain_price, plain_error = options_model.get_monte_carlo_option_price(
        *EUROPEAN, put_option, simulations=16_384, time_steps=16, seed=7
    )
    price, error = options_model.get_monte_carlo_option_price(
        *EUROPEAN,
        put_option,
        simulations=16_384,
        time_steps=16,
        seed=7,
        variance_reduction=variance_reduction,
        sequence=sequence,
    )

    # The Sobol standard error comes from 16 replications, so its estimate has the wider
    # tails of a t-distribution with 15 degrees of freedom.
    assert abs(plain_price - exact) < 4 * plain_error
    assert abs(price - exact) < 5 * error
    assert error < plain_error


def test_sobol_error_shrinks_faster_than_pseudo_random():
    exact = get_black_scholes(*EUROPEAN)
    errors = {}
    for sequence in ("pseudo_random", "sobol"):
        errors[sequence] = [
            np.mean(
                [
                    abs(
                        options_model.get_monte_carlo_option_price(
                            *EUROPEAN,
                            simulations=simulations,
                            time_steps=8,
                            seed=seed,
                            sequence=sequence,
                        )[0]
                        - exact
                    )
                    for seed in range(8)
                ]
            )
            for simulations in (2_048, 32_768)
        ]

    # Sixteen times the paths divide the pseudo-random error by about four (1 / √n) and the
    # Sobol error by more, while Sobol is far closer at either size.
    assert errors["sobol"][0] * 5 < errors["pseudo_random"][0]
    assert errors["sobol"][1] * 5 < errors["pseudo_random"][1]
    assert (
        errors["sobol"][0] / errors["sobol"][1]
        > errors["pseudo_random"][0] / errors["pseudo_random"][1]
    )


def test_correlated_paths_have_the_forward_and_the_correlation():
    paths = monte_carlo_model.simulate_gbm_paths(
        [100, 50],
        0.05,
        [0.2, 0.3],
        1.0,
        [0.0, 0.02],
        [[1, 0.6], [0.6, 1]],
        time_steps=4,
        simulations=50_000,
        seed=2,
    )
    log_returns = np.log(paths[:, -1] / paths[:, 0])

    assert paths.shape == (50_000, 5, 2)
    np.testing.assert_allclose(
        paths[:, -1].mean(axis=0),
        [100 * np.exp(0.05), 50 * np.exp(0.03)],
        rtol=4e-3,
    )
    assert np.corrcoef(log_returns, rowvar=False)[0, 1] == pytest.approx(0.6, abs=0.01)


@pytest.mark.parametrize(
    "volatility_of_variance, correlation, quantlib",
    [(0.5, -0.7, 10.055482967735774), (1.0, -0.9, 9.037154909065308)],
)
def test_heston_quadratic_exponential_paths_price_the_analytic_value(
    volatility_of_variance, correlation, quantlib
):
    prices, variances = monte_carlo_model.simulate_heston_paths(
        100,
        0.04,
        0.05,
        1.5,
        0.04,
        volatility_of_variance,
        correlation,
        1.0,
        time_steps=10,
        simulations=100_000,
        seed=3,
    )
    payoffs = np.exp(-0.05) * np.maximum(prices[:, -1] - 100, 0)

    # QuantLib's AnalyticHestonEngine; ten steps a year suffice for this scheme, while a
    # full truncation Euler scheme is off by 0.12 and 0.84 at that step.
    assert payoffs.mean() == pytest.approx(
        quantlib, abs=3.5 * payoffs.std() / np.sqrt(payoffs.size)
    )
    assert (variances >= 0).all()
    assert prices[:, -1].mean() == pytest.approx(100 * np.exp(0.05), rel=3e-3)


def test_discretely_monitored_barrier_matches_the_continuity_correction():
    corrected = get_barrier_option(
        100,
        100,
        get_discrete_monitoring_barrier(90, 100, 0.2, 1 / 52),
        0.05,
        0.2,
        1.0,
        0.02,
    )
    price, error = monte_carlo_model.get_monte_carlo_barrier_option(
        100,
        100,
        90,
        0.05,
        0.2,
        1.0,
        0.02,
        monitoring_dates=52,
        simulations=100_000,
        variance_reduction="control_variate",
        seed=1,
    )
    continuous = get_barrier_option(100, 100, 90, 0.05, 0.2, 1.0, 0.02)

    # The continuity correction of Broadie, Glasserman and Kou is itself within about 0.01
    # here; a weekly barrier is worth clearly more than a continuous one.
    assert price == pytest.approx(corrected, abs=4 * error + 0.01)
    assert price > continuous + 10 * error


def test_knock_in_and_knock_out_add_up_to_the_european_on_the_same_paths():
    european = options_model.get_monte_carlo_option_prices(
        100, np.array([100.0]), 0.05, 0.2, 1.0, 0.02, True, 20_000, 12, 5
    )
    knock_out, knock_in = (
        monte_carlo_model.get_monte_carlo_barrier_option(
            100,
            100,
            110,
            0.05,
            0.2,
            1.0,
            0.02,
            True,
            "up",
            knock_type,
            0.0,
            12,
            20_000,
            seed=5,
        )[0]
        for knock_type in ("out", "in")
    )

    assert knock_out + knock_in == pytest.approx(float(european[0][0]), abs=1e-10)


def test_least_squares_monte_carlo_matches_the_american_put():
    exact = get_qd_fixed_point(36, 40, 0.06, 0.2, 1.0, 0.0, True)
    price, error = monte_carlo_model.get_longstaff_schwartz_option(
        36,
        40,
        0.06,
        0.2,
        1.0,
        put_option=True,
        simulations=100_000,
        variance_reduction="antithetic",
    )

    # Longstaff and Schwartz (2001, table 1) report 4.472 with 50 exercise dates, FinancePy's
    # equity_lsmc gives 4.4727 to 4.4786 over three seeds and QuantLib's MCAmericanEngine
    # 4.4662 (it fits the exercise rule on separate paths, which biases it down).
    assert price == pytest.approx(4.472, abs=4 * error)
    assert price < exact + 4 * error
    assert error < 0.01


def test_least_squares_monte_carlo_bases_and_sequences_agree():
    prices = [
        monte_carlo_model.get_longstaff_schwartz_option(
            36,
            40,
            0.06,
            0.2,
            1.0,
            put_option=True,
            simulations=32_768,
            basis=basis,
            sequence=sequence,
            seed=11,
        )
        for basis in ("laguerre", "monomial")
        for sequence in ("pseudo_random", "sobol")
    ]

    for price, error in prices:
        assert price == pytest.approx(4.472, abs=4 * error + 0.005)


def test_an_american_call_without_dividends_is_worth_the_european():
    price, error = monte_carlo_model.get_longstaff_schwartz_option(
        100,
        100,
        0.05,
        0.2,
        1.0,
        simulations=50_000,
        variance_reduction="control_variate",
    )

    # The European floor of the continuation value rules out every exercise, so the control
    # variate returns the Black-Scholes price itself.
    assert price == pytest.approx(
        get_black_scholes(100, 100, 0.05, 0.2, 1.0), abs=4 * error + 1e-12
    )


def test_least_squares_monte_carlo_never_prices_below_the_european_put():
    # With r <= 0 early exercise is never optimal, so the American put is the European put;
    # without the European floor the fitted rule exercised by mistake and the price fell
    # 2.5 to 4.2 standard errors below it.
    for risk_free_rate in (-0.01, 0.0):
        european = float(
            get_black_scholes(100, 100, risk_free_rate, 0.2, 1.0, put_option=True)
        )
        for seed in (1, 2):
            price, error = monte_carlo_model.get_longstaff_schwartz_option(
                100,
                100,
                risk_free_rate,
                0.2,
                1.0,
                put_option=True,
                simulations=100_000,
                variance_reduction="control_variate",
                seed=seed,
            )
            assert price == pytest.approx(european, abs=4 * error + 1e-3)


def test_least_squares_monte_carlo_matches_longstaff_schwartz_table_1():
    # Longstaff and Schwartz (2001), Table 1: the finite difference values of American puts
    # with K = 40, r = 0.06 and 50 exercise dates a year.
    for stock_price, volatility, time, value in (
        (36, 0.2, 1, 4.478),
        (36, 0.2, 2, 4.840),
        (36, 0.4, 1, 7.101),
        (40, 0.2, 1, 2.314),
        (44, 0.2, 1, 1.110),
        (44, 0.4, 2, 5.647),
    ):
        price, error = monte_carlo_model.get_longstaff_schwartz_option(
            stock_price,
            40,
            0.06,
            volatility,
            time,
            put_option=True,
            simulations=100_000,
            time_steps=50 * time,
            variance_reduction="antithetic",
        )
        assert abs(price - value) <= 2 * error + 1e-9


def test_american_max_call_on_two_stocks():
    price, error = monte_carlo_model.get_longstaff_schwartz_basket_option(
        [100, 100],
        [1, 1],
        100,
        0.05,
        [0.2, 0.2],
        [[1, 0], [0, 1]],
        3.0,
        [0.1, 0.1],
        basket_type="maximum",
        simulations=100_000,
        time_steps=9,
    )

    # Broadie and Glasserman (1997) and Glasserman (2003, table 8.1) give 13.90 for nine
    # exercise dates; QuantLib's MCAmericanBasketEngine gives 13.73 +- 0.05 (biased down
    # by its separate calibration paths).
    assert price == pytest.approx(13.90, abs=4 * error)


def test_american_basket_put():
    price, error = monte_carlo_model.get_longstaff_schwartz_basket_option(
        [100, 100],
        [0.5, 0.5],
        100,
        0.05,
        [0.3, 0.3],
        [[1, 0.5], [0.5, 1]],
        1.0,
        put_option=True,
        simulations=50_000,
        variance_reduction="antithetic",
    )

    # QuantLib's MCAmericanBasketEngine (100,000 paths, 50 dates) gives 8.3083 +- 0.0292.
    assert price == pytest.approx(8.308339279329024, abs=4 * (error + 0.0292))


def test_prices_are_seeded_and_stable_under_tiny_input_changes():
    def price(stock_price):
        return monte_carlo_model.get_longstaff_schwartz_option(
            stock_price,
            40,
            0.06,
            0.2,
            1.0,
            put_option=True,
            simulations=4_096,
            sequence="sobol",
            seed=3,
        )[0]

    assert price(36.0) == price(36.0)
    assert price(36 + 1e-12) == pytest.approx(price(36.0), abs=1e-9)
    barrier = monte_carlo_model.get_monte_carlo_barrier_option
    assert barrier(100 + 1e-12, 100, 90, 0.05, 0.2, 1.0)[0] == pytest.approx(
        barrier(100.0, 100, 90, 0.05, 0.2, 1.0)[0], abs=1e-9
    )


def test_inputs_are_validated():
    with pytest.raises(ValueError, match="sequence"):
        monte_carlo_model.simulate_gbm_paths(100, 0.05, 0.2, 1.0, sequence="halton")
    with pytest.raises(ValueError, match="positive definite"):
        monte_carlo_model.simulate_gbm_paths(
            [100, 100], 0.05, [0.2, 0.2], 1.0, correlation_matrix=[[1, 2], [2, 1]]
        )
    with pytest.raises(ValueError, match="too large"):
        monte_carlo_model.simulate_gbm_paths(
            100, 0.05, 0.2, 1.0, time_steps=1_000, simulations=100_000
        )
    with pytest.raises(ValueError, match="scheme"):
        monte_carlo_model.simulate_heston_paths(
            100, 0.04, 0.05, 1.5, 0.04, 0.5, -0.7, 1.0, scheme="milstein"
        )
    with pytest.raises(ValueError, match="basis"):
        monte_carlo_model.get_longstaff_schwartz_option(
            36, 40, 0.06, 0.2, 1.0, basis="hermite"
        )
    with pytest.raises(ValueError, match="variance_reduction"):
        options_model.get_monte_carlo_option_price(
            *EUROPEAN, variance_reduction="importance"
        )


@pytest.mark.parametrize(
    ("variance_reduction", "sequence"),
    [
        (None, "pseudo_random"),
        ("antithetic", "pseudo_random"),
        ("control_variate", "pseudo_random"),
        (None, "sobol"),
    ],
)
@pytest.mark.parametrize("put_option", [False, True])
def test_all_expirations_from_one_set_of_paths_equal_a_seeded_price_per_expiration(
    variance_reduction, sequence, put_option
):
    """Scaling the same standardised shocks per expiration prices as simulating it with the seed."""
    strikes = np.array([70.0, 95.0, 100.0, 130.0])
    times = np.array([0.0, 1 / 365, 0.25, 1.0])
    inputs = {
        "stock_price": 100.0,
        "risk_free_rate": 0.05,
        "volatility": 0.25,
        "dividend_yield": 0.02,
        "put_option": put_option,
        "simulations": 2_000,
        "time_steps": 20,
        "seed": 7,
        "variance_reduction": variance_reduction,
        "sequence": sequence,
    }

    prices, errors = options_model.get_monte_carlo_option_prices(
        strike_prices=strikes, time_to_expiration=times, **inputs
    )

    assert prices.shape == errors.shape == (len(strikes), len(times))
    for column, time in enumerate(times):
        for row, strike in enumerate(strikes):
            price, error = options_model.get_monte_carlo_option_price(
                strike_price=float(strike), time_to_expiration=float(time), **inputs
            )
            assert prices[row, column] == pytest.approx(price, abs=1e-10)
            assert errors[row, column] == pytest.approx(error, abs=1e-10)


def test_the_standard_error_of_independent_paths_is_that_of_their_payoffs():
    """The cumulative sums give the sample standard deviation of the payoffs over √n."""
    rng = np.random.default_rng(1)
    terminal = 100 * np.exp(rng.normal(0, 0.3, 5_001))
    strikes = np.array([50.0, 99.5, 100.0, 140.0, 400.0])

    for put_option in (False, True):
        prices, errors = options_model._plain_monte_carlo_estimates(
            terminal, strikes, 0.97, put_option
        )
        for strike, price, error in zip(strikes, prices, errors, strict=True):
            payoffs = 0.97 * np.maximum(
                (strike - terminal) if put_option else (terminal - strike), 0
            )
            assert price == pytest.approx(payoffs.mean(), abs=1e-10)
            assert error == pytest.approx(
                payoffs.std(ddof=1) / np.sqrt(payoffs.size), abs=1e-10
            )


def test_least_squares_monte_carlo_refuses_a_strike_of_zero():
    # The regression on the stock price over the strike failed to converge at zero.
    with pytest.raises(ValueError, match="must be above 0"):
        monte_carlo_model.get_longstaff_schwartz_option(
            100.0, 0.0, 0.05, 0.2, 1.0, simulations=1_000
        )
    with pytest.raises(ValueError, match="must be above 0"):
        monte_carlo_model.get_longstaff_schwartz_basket_option(
            stock_prices=[100.0, 95.0],
            strike_price=0.0,
            risk_free_rate=0.05,
            volatilities=[0.2, 0.3],
            correlation_matrix=[[1, 0.5], [0.5, 1]],
            weights=[0.5, 0.5],
            time_to_expiration=1.0,
            simulations=1_000,
        )
