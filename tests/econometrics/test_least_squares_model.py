"""Least Squares Model Tests

The least squares fit and the test statistics are written from their formulas. The
reference values below were computed once with statsmodels (0.14.6) on the same data
and are hard-coded, so these tests check the values without importing statsmodels.
"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.econometrics import least_squares_model
from financetoolkit.econometrics.diagnostics import diagnostics_model
from financetoolkit.econometrics.regression import regression_model
from financetoolkit.econometrics.stationarity import causality_model
from financetoolkit.econometrics.testing import (
    hypothesis_testing_model,
    specification_tests_model,
)

RTOL = 1e-10

# statsmodels' OLS(...).fit(cov_type=...).bse on the data of `_data`.
REFERENCE_STANDARD_ERRORS = {
    "nonrobust": [0.14682559686506846, 0.15695200645063792, 0.16263304139302953],
    "HC0": [0.13780698728005403, 0.22793774352954882, 0.14314811598113153],
    "HC1": [0.13956256107770876, 0.23084152611659436, 0.14497173237795047],
    "HC2": [0.1403635440586889, 0.23283164151013647, 0.14661635348485832],
    "HC3": [0.14299664326465056, 0.2378602178265169, 0.15019597021512512],
}


def _data() -> tuple[np.ndarray, np.ndarray]:
    """A regression with heteroskedastic, autocorrelated errors."""
    rng = np.random.default_rng(7)
    n = 120
    x = rng.standard_normal((n, 2))
    noise = rng.standard_normal(n) * (0.5 + np.abs(x[:, 0]))
    for t in range(1, n):
        noise[t] += 0.5 * noise[t - 1]
    y = 0.2 + 0.8 * x[:, 0] - 0.3 * x[:, 1] + noise
    return y, x


def test_ols_matches_the_reference_fit():
    y, x = _data()
    result = regression_model.get_ols(y, x)

    np.testing.assert_allclose(
        result["coefficients"],
        [0.1598054616562727, 1.5283796468465245, -0.18719966097749355],
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        result["p_values"],
        [2.786540138193181e-01, 9.038624631295546e-17, 2.520563223158186e-01],
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        [
            result["r_squared"],
            result["adjusted_r_squared"],
            result["log_likelihood"],
            result["sum_of_squared_residuals"],
        ],
        [
            0.44857184893435664,
            0.4391457266939184,
            -222.73019930486828,
            287.6615641875562,
        ],
        rtol=RTOL,
    )


@pytest.mark.parametrize("cov_type", list(REFERENCE_STANDARD_ERRORS))
def test_ols_covariances_match_the_reference(cov_type):
    y, x = _data()
    result = regression_model.get_ols(y, x, cov_type=cov_type)

    np.testing.assert_allclose(
        result["standard_errors"], REFERENCE_STANDARD_ERRORS[cov_type], rtol=RTOL
    )


def test_ols_newey_west_and_cluster_covariances_match_the_reference():
    y, x = _data()

    newey_west = regression_model.get_ols(y, x, cov_type="HAC", lags=3)
    np.testing.assert_allclose(
        newey_west["standard_errors"],
        [0.18941833812769732, 0.2771781719737447, 0.14091563585041475],
        rtol=RTOL,
    )

    clustered = regression_model.get_ols(
        y, x, cov_type="cluster", clusters=np.arange(120) // 6
    )
    np.testing.assert_allclose(
        clustered["standard_errors"],
        [0.20222320979423417, 0.3226320867131054, 0.11202512759025199],
        rtol=RTOL,
    )
    # Robust p-values come from the standard normal distribution.
    np.testing.assert_allclose(
        clustered["p_values"],
        [4.293859094631707e-01, 2.166683469406768e-06, 9.471167609984864e-02],
        rtol=RTOL,
    )
    assert clustered["inference_degrees_of_freedom"] == 19


def test_ols_without_constant_uses_the_uncentered_r_squared():
    y, x = _data()
    result = regression_model.get_ols(y, x, add_constant=False)

    np.testing.assert_allclose(
        [result["r_squared"], result["adjusted_r_squared"]],
        [0.44309521072340163, 0.4336561464983746],
        rtol=RTOL,
    )


def test_wls_and_gls_match_the_reference():
    y, x = _data()
    weights = 1 / (0.5 + np.abs(x[:, 0]))

    wls = regression_model.get_wls(y, x, weights)
    np.testing.assert_allclose(
        wls["coefficients"],
        [0.2019843230974602, 1.435752508347733, -0.2378161276817693],
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        wls["standard_errors"],
        [0.11592768530818758, 0.1700115374501859, 0.131734826690781],
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        [wls["r_squared"], wls["log_likelihood"]],
        [0.38468641011623605, -202.44858517352048],
        rtol=RTOL,
    )
    wls_robust = regression_model.get_wls(y, x, weights, cov_type="HC3")
    np.testing.assert_allclose(
        wls_robust["standard_errors"],
        [0.10382579438713145, 0.20974671830335714, 0.10926596362800119],
        rtol=RTOL,
    )

    omega = 0.5 ** np.abs(np.subtract.outer(np.arange(120), np.arange(120)))
    gls = regression_model.get_gls(y, x, omega)
    np.testing.assert_allclose(
        gls["coefficients"],
        [0.101439582630924, 1.231747222614938, -0.29327310032467657],
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        gls["standard_errors"],
        [0.23497792391181094, 0.11819165418351704, 0.11897045057731545],
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        [gls["r_squared"], gls["log_likelihood"]],
        [0.48737952129426376, -199.52162470129798],
        rtol=RTOL,
    )


def test_has_constant_detects_explicit_and_implicit_constants():
    dummies = np.repeat(np.eye(2), 3, axis=0)

    assert least_squares_model.has_constant(
        np.column_stack([np.full(6, 2.0), np.arange(6)])
    )
    assert least_squares_model.has_constant(dummies)
    assert not least_squares_model.has_constant(np.arange(6.0).reshape(-1, 1))


def test_nested_and_restriction_tests_match_the_reference():
    y, x = _data()
    full = regression_model.get_ols(y, x)
    restricted = regression_model.get_ols(y, x[:, :1])

    f_test = hypothesis_testing_model.get_f_test(restricted, full)
    np.testing.assert_allclose(
        [f_test["F-Statistic"], f_test["P-Value"]],
        [1.3249288235368488, 0.25205632231581176],
        rtol=RTOL,
    )

    lr_test = hypothesis_testing_model.get_likelihood_ratio_test(restricted, full)
    np.testing.assert_allclose(
        [lr_test["LR Statistic"], lr_test["P-Value"]],
        [1.3512647352930003, 0.24505713574051285],
        rtol=RTOL,
    )

    robust = regression_model.get_ols(y, x, cov_type="HC1")
    wald = hypothesis_testing_model.get_wald_test(
        robust, np.eye(3)[1:], np.array([0.5, 0.0])
    )
    np.testing.assert_allclose(
        [wald["Wald Statistic (Chi2)"], wald["Chi2 P-Value"]],
        [2.6104808183200515e01, 2.1449290316697930e-06],
        rtol=RTOL,
    )


def test_specification_tests_match_the_reference():
    y, x = _data()
    result = regression_model.get_ols(y, x)

    breusch_pagan = specification_tests_model.get_breusch_pagan_test(result)
    np.testing.assert_allclose(
        [breusch_pagan["Breusch-Pagan Statistic"], breusch_pagan["P-Value"]],
        [3.9483529231546655, 0.13887563458312563],
        rtol=RTOL,
    )

    white = specification_tests_model.get_white_test(result)
    np.testing.assert_allclose(
        [white["White Statistic"], white["P-Value"]],
        [1.4610882610619885e01, 1.2161032403124009e-02],
        rtol=RTOL,
    )

    durbin_watson = specification_tests_model.get_durbin_watson_test(result)
    np.testing.assert_allclose(
        durbin_watson["Durbin-Watson Statistic"], 0.9419398086752445, rtol=RTOL
    )

    reset = specification_tests_model.get_ramsey_reset_test(result, power=3)
    np.testing.assert_allclose(
        [reset["RESET F-Statistic"], reset["P-Value"]],
        [2.3969550515389515, 0.09552746088374935],
        rtol=RTOL,
    )
    reset_robust = specification_tests_model.get_ramsey_reset_test(
        regression_model.get_ols(y, x, cov_type="HC3"), power=3
    )
    np.testing.assert_allclose(
        [reset_robust["RESET F-Statistic"], reset_robust["P-Value"]],
        [0.5751626984463508, 0.564223340879405],
        rtol=RTOL,
    )

    vif = specification_tests_model.get_vif(pd.DataFrame(x, columns=["A", "B"]))
    np.testing.assert_allclose(vif.to_numpy(), [1.0033227598062096] * 2, rtol=RTOL)


def test_series_diagnostics_match_the_reference():
    y, x = _data()
    residuals = pd.Series(regression_model.get_ols(y, x)["residuals"])

    arch = diagnostics_model.get_arch_lm_test(residuals, lags=5)
    np.testing.assert_allclose(
        [arch["ARCH-LM Statistic"], arch["P-Value"]],
        [2.1417077648487275e01, 6.7548425633156975e-04],
        rtol=RTOL,
    )

    jarque_bera = diagnostics_model.get_jarque_bera_test(residuals)
    np.testing.assert_allclose(
        [jarque_bera["Jarque-Bera Statistic"], jarque_bera["P-Value"]],
        [5.677826428038009e02, 5.099946907394186e-124],
        rtol=RTOL,
    )

    ljung_box = diagnostics_model.get_ljung_box_test(residuals, lags=10, model_df=2)
    np.testing.assert_allclose(
        [ljung_box["Ljung-Box Statistic"], ljung_box["P-Value"]],
        [4.7164986858474833e01, 1.4262269782882796e-07],
        rtol=RTOL,
    )

    cusum = diagnostics_model.get_cusum_test(residuals)
    np.testing.assert_allclose(
        [cusum["CUSUM Statistic"], cusum["P-Value"]],
        [1.2189254512251795, 0.10243306258576322],
        rtol=RTOL,
    )

    granger = causality_model.get_granger_causality(
        pd.Series(y), pd.Series(x[:, 0]), max_lag=3
    )
    np.testing.assert_allclose(
        [granger["F-Statistic"], granger["P-Value"]],
        [1.016678254903437, 0.38822913400544024],
        rtol=RTOL,
    )


def test_two_sample_t_tests_match_the_reference():
    y, x = _data()

    welch = hypothesis_testing_model.get_two_sample_t_test(y[:50], x[:70, 0])
    np.testing.assert_allclose(
        welch[["t-Statistic", "P-Value", "Degrees of Freedom"]].to_numpy(dtype=float),
        [-0.6331740956722862, 0.5291498413378317, 57.055333576625614],
        rtol=RTOL,
    )

    pooled = hypothesis_testing_model.get_two_sample_t_test(
        y[:50], x[:70, 0], equal_variance=True
    )
    np.testing.assert_allclose(
        pooled[["t-Statistic", "P-Value", "Degrees of Freedom"]].to_numpy(dtype=float),
        [-0.7243739804330513, 0.47026997412953503, 118.0],
        rtol=RTOL,
    )


def test_kolmogorov_survival_matches_the_reference():
    values = [0.3, 0.8, 1.0, 1.36, 2.5]
    reference = [
        9.9999069419866549e-01,
        5.4414241157419807e-01,
        2.6999967167735456e-01,
        4.9485876755377876e-02,
        7.4533063441573419e-06,
    ]

    np.testing.assert_allclose(
        [least_squares_model.get_kolmogorov_survival(value) for value in values],
        reference,
        rtol=RTOL,
    )
    assert least_squares_model.get_kolmogorov_survival(0.0) == 1.0


def test_wald_statistic_counts_dependent_restrictions_once():
    coefficients = np.array([1.0, 2.0])
    covariance = np.diag([0.5, 0.25])
    # The second row repeats the first, so there is one independent restriction.
    restriction_matrix = np.array([[1.0, 0.0], [2.0, 0.0]])

    statistic, restrictions = least_squares_model.get_wald_statistic(
        coefficients, covariance, restriction_matrix
    )

    assert restrictions == 1
    np.testing.assert_allclose(statistic, 1.0**2 / 0.5, rtol=RTOL)
