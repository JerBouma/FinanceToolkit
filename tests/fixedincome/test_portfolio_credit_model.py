"""Portfolio Credit Model Tests"""

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.stats import t as student_t

from financetoolkit.fixedincome.credit import cds_model, portfolio_credit_model

# FinancePy 1.1.2 references for p = 5%, ρ = 20% (β = √ρ), R = 40% and 125 names. FinancePy
# evaluates the normal distribution with an approximation good to about 1e-7 and integrates
# over the factor on 50 points, hence the tolerance of 5e-7.
P, RHO, R = 0.05, 0.2, 0.4
FINANCEPY = 5e-7


@pytest.mark.parametrize(
    "loss, cdf, density",
    [
        (0.01, 0.281603003207771, 27.144362412124458),
        (0.03, 0.6511019215929912, 11.957481435060116),
        (0.05, 0.8191199512335241, 5.722594862895157),
        (0.1, 0.9593473275546969, 1.1648813544371701),
    ],
)
def test_vasicek_distribution_matches_financepy(loss, cdf, density):
    assert portfolio_credit_model.get_vasicek_loss_distribution(
        loss, P, RHO, R
    ) == pytest.approx(cdf, abs=FINANCEPY)
    assert portfolio_credit_model.get_vasicek_loss_density(
        loss, P, RHO, R
    ) == pytest.approx(density, rel=FINANCEPY)


def test_vasicek_density_integrates_to_the_distribution():
    for loss in (0.01, 0.05, 0.2, 0.5):
        integral, _ = quad(
            portfolio_credit_model.get_vasicek_loss_density,
            0,
            loss,
            args=(P, RHO, R),
            limit=200,
        )
        assert integral == pytest.approx(
            portfolio_credit_model.get_vasicek_loss_distribution(loss, P, RHO, R),
            abs=1e-10,
        )
    assert portfolio_credit_model.get_vasicek_loss_distribution(0.6, P, RHO, R) == 1.0
    assert portfolio_credit_model.get_vasicek_loss_density(0.7, P, RHO, R) == 0.0


TRANCHES = [(0, 0.03), (0.03, 0.07), (0.07, 0.1), (0.1, 0.15), (0.15, 0.3), (0.3, 1.0)]


@pytest.mark.parametrize(
    "tranche, lhp, recursion",
    list(
        zip(
            TRANCHES,
            [
                0.37229677786288307,
                0.805315317645215,
                0.9351323839870193,
                0.9782649695837154,
                0.9976974767164958,
                0.9999952035029059,
            ],
            [
                0.39880654139014793,
                0.7964912661432564,
                0.9283834000005943,
                0.9750622818065086,
                0.9971768446317854,
                0.9999928832409305,
            ],
        )
    ),
)
def test_tranche_survival_matches_financepy(tranche, lhp, recursion):
    # FinancePy's tr_surv_prob_lhp and tranche_surv_prob_recursion give 1 - the expected loss.
    assert 1 - portfolio_credit_model.get_vasicek_tranche_loss(
        *tranche, P, RHO, R
    ) == pytest.approx(lhp, abs=FINANCEPY)
    distribution = portfolio_credit_model.get_portfolio_loss_distribution(
        [P] * 125, RHO, R
    )
    assert 1 - portfolio_credit_model.get_tranche_expected_loss(
        distribution, *tranche
    ) == pytest.approx(recursion, abs=FINANCEPY)


def test_recursion_matches_financepy():
    probabilities = [0.01, 0.02, 0.03, 0.05, 0.08]
    np.testing.assert_allclose(
        portfolio_credit_model.get_portfolio_loss_distribution(
            probabilities, 0.3, recovery_rates=0.0
        ),
        [
            0.8429081536588551,
            0.12954819632324813,
            0.02288441513395987,
            0.004009182182790223,
            0.000594802840516241,
            5.524754814864832e-05,
        ],
        atol=FINANCEPY,
    )
    # Different correlations and losses of 1, 2, 1, 3 and 2 units (loss_dbn_recursion_gcd).
    distribution = portfolio_credit_model.get_portfolio_loss_distribution(
        probabilities,
        [0.1, 0.2, 0.3, 0.4, 0.25],
        recovery_rates=0.0,
        notionals=[1, 2, 1, 3, 2],
    )
    np.testing.assert_allclose(
        distribution,
        [
            0.8406948334061711,
            0.02565998247741604,
            0.0745968649174054,
            0.03914301456177335,
            0.0064252140818841725,
            0.010259085517332468,
            0.0021217233950996638,
            0.0008192904546132111,
            0.0002643221498630555,
            1.5666725959170113e-05,
        ],
        atol=FINANCEPY,
    )
    np.testing.assert_allclose(distribution.index, np.arange(10) / 9)
    # homog_basket_loss_dbn: ten names, survival 95%, recovery 40%.
    np.testing.assert_allclose(
        portfolio_credit_model.get_portfolio_loss_distribution([0.05] * 10, 0.2),
        [
            0.6619661606470235,
            0.22780890060644726,
            0.0744987072896217,
            0.024402255707113955,
            0.007896087430488826,
            0.0024653546454883094,
            0.0007214732100208517,
            0.0001903703597421916,
            4.2679498860373585e-05,
            7.288306023920879e-06,
            7.199866867262296e-07,
        ],
        atol=FINANCEPY,
    )


def test_recursion_without_correlation_is_binomial():
    distribution = portfolio_credit_model.get_portfolio_loss_distribution(
        [0.03] * 20, 0.0, recovery_rates=0.0
    )
    from scipy.stats import binom

    np.testing.assert_allclose(
        distribution, binom.pmf(np.arange(21), 20, 0.03), atol=1e-15
    )


@pytest.mark.parametrize("correlation", [0.05, 0.3])
def test_tranches_add_up_to_the_portfolio_loss(correlation):
    distribution = portfolio_credit_model.get_portfolio_loss_distribution(
        [0.02, 0.04, 0.06] * 20, correlation, recovery_rates=[0.4, 0.25, 0.6] * 20
    )
    expected_loss = float(distribution.index.to_numpy() @ distribution.to_numpy())
    assert expected_loss == pytest.approx(
        np.mean([0.02 * 0.6, 0.04 * 0.75, 0.06 * 0.4]), rel=1e-12
    )
    for tranche_loss in (portfolio_credit_model.get_tranche_expected_loss, None):
        total = 0.0
        for attachment, detachment in TRANCHES:
            width = detachment - attachment
            if tranche_loss is None:
                total += width * portfolio_credit_model.get_vasicek_tranche_loss(
                    attachment, detachment, P, correlation, R
                )
            else:
                total += width * tranche_loss(distribution, attachment, detachment)
        portfolio = (1 - R) * P if tranche_loss is None else expected_loss
        assert total == pytest.approx(portfolio, rel=1e-12)


def test_large_portfolio_tends_to_the_vasicek_limit():
    # The finite portfolio's tranche losses close in on the limit as the names grow, at the
    # rate 1 / names of the granularity adjustment (Gordy, 2003).
    gaps = []
    names = np.array([25, 100, 400, 1000])
    for count in names:
        distribution = portfolio_credit_model.get_portfolio_loss_distribution(
            [P] * count, RHO, R
        )
        gaps.append(
            max(
                abs(
                    portfolio_credit_model.get_tranche_expected_loss(
                        distribution, *tranche
                    )
                    - portfolio_credit_model.get_vasicek_tranche_loss(
                        *tranche, P, RHO, R
                    )
                )
                for tranche in TRANCHES[:4]
            )
        )
    assert np.all(np.diff(gaps) < 0)
    assert gaps[-1] < 5e-3
    # The gap times the number of names settles: the error is of order 1 / names.
    scaled = np.array(gaps) * names
    assert scaled[-1] == pytest.approx(scaled[-2], rel=0.15)


def test_student_t_copula_matches_monte_carlo():
    probabilities = np.array([0.02, 0.03, 0.05, 0.05, 0.08, 0.1])
    correlation, degrees_of_freedom = 0.3, 4
    distribution = portfolio_credit_model.get_portfolio_loss_distribution(
        probabilities,
        correlation,
        recovery_rates=0.0,
        copula="student_t",
        degrees_of_freedom=degrees_of_freedom,
    )
    generator = np.random.default_rng(3)
    draws = 400_000
    latent = np.sqrt(correlation) * generator.standard_normal((draws, 1)) + np.sqrt(
        1 - correlation
    ) * generator.standard_normal((draws, len(probabilities)))
    latent = latent * np.sqrt(
        degrees_of_freedom / generator.chisquare(degrees_of_freedom, (draws, 1))
    )
    counts = (
        np.bincount(
            (latent <= student_t.ppf(probabilities, degrees_of_freedom)).sum(axis=1),
            minlength=len(probabilities) + 1,
        )
        / draws
    )
    errors = np.sqrt(counts * (1 - counts) / draws)
    assert np.all(np.abs(distribution.to_numpy() - counts) < 4 * errors + 1e-6)
    # Tail dependence: more joint defaults than the Gaussian copula at the same correlation.
    gaussian = portfolio_credit_model.get_portfolio_loss_distribution(
        probabilities, correlation, recovery_rates=0.0
    )
    assert distribution.iloc[3:].sum() > gaussian.iloc[3:].sum()


def test_nth_to_default_without_correlation():
    probabilities = np.array([0.01, 0.02, 0.03, 0.05, 0.08])
    assert portfolio_credit_model.get_nth_to_default_probability(
        probabilities, 0.0, 1
    ) == pytest.approx(1 - np.prod(1 - probabilities), rel=1e-14)
    assert portfolio_credit_model.get_nth_to_default_probability(
        probabilities, 0.0, 5
    ) == pytest.approx(np.prod(probabilities), rel=1e-12)
    # A first-to-default swap on independent names is a CDS on the sum of the hazard rates.
    curve = pd.Series([0.035, 0.035], index=[0.0, 60.0])
    hazards = np.array([[0.01], [0.015], [0.02]])
    basket = portfolio_credit_model.get_nth_to_default_swap(
        1, 5, curve, hazards, [5], 0.0
    )
    single = cds_model.get_credit_default_swap(5, curve, [hazards.sum()], [5])
    pd.testing.assert_series_equal(basket, single, rtol=1e-12)


@pytest.mark.parametrize(
    "correlation, n, spread",
    [
        (0.0001, 1, 0.06003723051848332),
        (0.0001, 2, 0.008756156484799587),
        (0.0001, 5, 6.446810453456868e-07),
        (0.3, 1, 0.04868483199325284),
        (0.3, 2, 0.013022786914080476),
        (0.3, 5, 0.00011104699442060618),
    ],
)
def test_nth_to_default_swap_matches_financepy(correlation, n, spread):
    # FinancePy's CDSBasket.value_1f_gaussian_homo on five names with 5-year CDS spreads of
    # 60 to 180 basis points (the flat hazard rates below), a flat 3.5% curve and 40%
    # recovery. FinancePy schedules on business days and ACT/360 dates; 0.5% covers that.
    hazards = [
        [0.01008190749714305],
        [0.015122461796255357],
        [0.02016274829999794],
        [0.02520276588708992],
        [0.030242513447373488],
    ]
    curve = pd.Series([0.035, 0.035], index=[0.0, 60.0])
    result = portfolio_credit_model.get_nth_to_default_swap(
        n, 5, curve, hazards, [5], correlation
    )
    assert result["Par Spread"] == pytest.approx(spread, rel=5e-3)


def test_correlation_moves_first_and_last_to_default_apart():
    probabilities = [0.02, 0.04, 0.06, 0.08]
    first = [
        portfolio_credit_model.get_nth_to_default_probability(probabilities, rho, 1)
        for rho in (0.1, 0.5)
    ]
    last = [
        portfolio_credit_model.get_nth_to_default_probability(probabilities, rho, 4)
        for rho in (0.1, 0.5)
    ]
    assert first[1] < first[0] and last[1] > last[0]


def test_results_are_stable_under_tiny_perturbations():
    base = portfolio_credit_model.get_portfolio_loss_distribution(
        [0.02, 0.05] * 30, 0.25, [0.4, 0.3] * 30
    )
    moved = portfolio_credit_model.get_portfolio_loss_distribution(
        [0.02 + 1e-12, 0.05] * 30, 0.25 + 1e-12, [0.4, 0.3 + 1e-12] * 30
    )
    np.testing.assert_allclose(moved.to_numpy(), base.to_numpy(), atol=1e-10)
    np.testing.assert_allclose(moved.index, base.index, atol=1e-10)
    assert portfolio_credit_model.get_vasicek_tranche_loss(
        0.03, 0.07, P + 1e-12, RHO, R
    ) == pytest.approx(
        portfolio_credit_model.get_vasicek_tranche_loss(0.03, 0.07, P, RHO, R),
        abs=1e-10,
    )


def test_inputs_are_checked():
    with pytest.raises(ValueError, match="correlation"):
        portfolio_credit_model.get_vasicek_loss_distribution(0.05, P, 1.0)
    with pytest.raises(ValueError, match="attachment"):
        portfolio_credit_model.get_vasicek_tranche_loss(0.07, 0.03, P, RHO)
    with pytest.raises(ValueError, match="degrees of freedom"):
        portfolio_credit_model.get_portfolio_loss_distribution(
            [P] * 3, RHO, copula="student_t"
        )
    with pytest.raises(ValueError, match="copula"):
        portfolio_credit_model.get_portfolio_loss_distribution(
            [P] * 3, RHO, copula="clayton"
        )
    with pytest.raises(ValueError, match="n must"):
        portfolio_credit_model.get_nth_to_default_probability([P] * 3, RHO, 4)
