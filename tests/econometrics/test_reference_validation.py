"""Econometrics Reference Validation Tests

Every estimator in `financetoolkit.econometrics` checked against an independent
reference on a canonical dataset: NIST's certified Longley regression, published
textbook results (Wooldridge, Baltagi), statsmodels/linearmodels/arch/scipy, or a
hand-rolled implementation of the estimator's own definition. These are absolute
correctness checks rather than snapshots -- they fail when a formula changes, not when
a value drifts.

Every dataset used here ships inside statsmodels or linearmodels, so nothing here
touches the network.
"""

# pylint: disable=missing-function-docstring,redefined-outer-name

import numpy as np
import pandas as pd
import pytest
import scipy.stats as st
import statsmodels.api as sm
from linearmodels.datasets import card, french, mroz
from linearmodels.iv import IV2SLS
from linearmodels.panel import PanelOLS, RandomEffects
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    breaks_cusumolsresid,
    het_arch,
    het_breuschpagan,
    het_white,
    linear_reset,
)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import (
    adfuller,
    coint,
    grangercausalitytests,
    kpss,
    zivot_andrews,
)
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen

from financetoolkit.econometrics import (
    causal_inference_model,
    causality_model,
    cointegration_model,
    diagnostics_model,
    event_study_model,
    fama_macbeth_model,
    forecast_evaluation_model,
    hypothesis_testing_model,
    panel_data_model,
    regression_model,
    specification_tests_model,
    time_series_model,
    unitroot_model,
)
from financetoolkit.risk import risk_model


@pytest.fixture(scope="module")
def longley():
    data = sm.datasets.longley.load_pandas().data
    return data["TOTEMP"], data.drop(columns=["TOTEMP"])


@pytest.fixture(scope="module")
def statecrime():
    data = sm.datasets.statecrime.load_pandas().data
    return data["murder"], data[["hs_grad", "poverty", "single", "urban"]]


@pytest.fixture(scope="module")
def macro():
    data = sm.datasets.macrodata.load_pandas().data
    return {
        "log_gdp": np.log(data["realgdp"]),
        "log_consumption": np.log(data["realcons"]),
        "gdp_returns": data["realgdp"].pct_change().dropna(),
    }


@pytest.fixture(scope="module")
def grunfeld():
    data = sm.datasets.grunfeld.load_pandas().data
    data["year"] = data["year"].astype(int)
    panel = data.set_index(["firm", "year"])
    return panel["invest"], panel[["value", "capital"]]


@pytest.fixture(scope="module")
def var_result(macro):
    frame = pd.concat(
        [macro["log_gdp"].diff(), macro["log_consumption"].diff()], axis=1
    ).dropna()
    frame.columns = ["gdp", "consumption"]
    return frame, time_series_model.get_var_forecast(frame, lags=2, forecast_steps=3)


# ── Ordinary Least Squares ────────────────────────────────────────────────────
# NIST Statistical Reference Datasets, Longley, certified to 15 significant digits.
# Deliberately ill-conditioned, so it catches a numerically sloppy solver.
NIST_COEFFICIENTS = [
    -3482258.63459582,
    15.0618722713733,
    -0.358191792925910e-01,
    -2.02022980381683,
    -1.03322686717359,
    -0.511041056535807e-01,
    1829.15146461355,
]
NIST_STANDARD_ERRORS = [
    890420.383607373,
    84.9149257747669,
    0.334910077722432e-01,
    0.488399681651699,
    0.214274163161675,
    0.226073200069370,
    455.478499142212,
]


def test_ols_matches_the_nist_certified_values(longley):
    y, x = longley
    result = regression_model.get_ols(y, x)

    np.testing.assert_allclose(result["coefficients"], NIST_COEFFICIENTS, rtol=1e-7)
    np.testing.assert_allclose(
        result["standard_errors"], NIST_STANDARD_ERRORS, rtol=1e-7
    )
    np.testing.assert_allclose(result["r_squared"], 0.995479004577296, rtol=1e-12)
    np.testing.assert_allclose(
        np.sqrt(result["residual_variance"]), 304.854073561965, rtol=1e-10
    )


def test_ols_inference_and_robust_covariances(longley):
    y, x = longley
    design = sm.add_constant(x)

    result = regression_model.get_ols(y, x)
    reference = sm.OLS(y, design).fit()
    np.testing.assert_allclose(
        result["t_statistics"], reference.tvalues.to_numpy(), rtol=1e-9
    )
    np.testing.assert_allclose(
        result["p_values"], reference.pvalues.to_numpy(), rtol=1e-9
    )

    robust = regression_model.get_ols(y, x, cov_type="HC1")
    np.testing.assert_allclose(
        robust["standard_errors"],
        sm.OLS(y, design).fit(cov_type="HC1").bse.to_numpy(),
        rtol=1e-9,
    )

    newey_west = regression_model.get_ols(y, x, cov_type="HAC", maxlags=2)
    np.testing.assert_allclose(
        newey_west["standard_errors"],
        sm.OLS(y, design).fit(cov_type="HAC", cov_kwds={"maxlags": 2}).bse.to_numpy(),
        rtol=1e-9,
    )


def test_weighted_and_generalized_least_squares(longley):
    y, x = longley
    design = sm.add_constant(x)
    weights = 1.0 / np.linspace(1.0, 2.0, len(y))
    omega = np.diag(np.linspace(1.0, 2.0, len(y)))

    np.testing.assert_allclose(
        regression_model.get_wls(y, x, weights=weights)["coefficients"],
        sm.WLS(y, design, weights=weights).fit().params.to_numpy(),
        rtol=1e-7,
    )
    np.testing.assert_allclose(
        regression_model.get_gls(y, x, omega=omega)["coefficients"],
        sm.GLS(y, design, sigma=omega).fit().params.to_numpy(),
        rtol=1e-7,
    )


def test_binary_choice_models():
    # Spector & Mazzeo (1980), the textbook logit/probit example.
    data = sm.datasets.spector.load_pandas().data
    y, x = data["GRADE"], data[["GPA", "TUCE", "PSI"]]
    design = sm.add_constant(x)

    logit = regression_model.get_logistic_regression(y, x)
    reference_logit = sm.Logit(y, design).fit(disp=0)
    np.testing.assert_allclose(
        logit["coefficients"], reference_logit.params.to_numpy(), rtol=1e-6
    )
    np.testing.assert_allclose(
        logit["standard_errors"], reference_logit.bse.to_numpy(), rtol=1e-6
    )

    probit = regression_model.get_probit_regression(y, x)
    np.testing.assert_allclose(
        probit["coefficients"],
        sm.Probit(y, design).fit(disp=0).params.to_numpy(),
        rtol=1e-6,
    )


def test_quantile_regression():
    # Engel (1857) food expenditure, statsmodels' own quantile regression example.
    data = sm.datasets.engel.load_pandas().data
    result = regression_model.get_quantile_regression(
        data["foodexp"], data[["income"]], tau=0.5
    )
    reference = QuantReg(data["foodexp"], sm.add_constant(data[["income"]])).fit(q=0.5)

    np.testing.assert_allclose(
        result["coefficients"], reference.params.to_numpy(), rtol=1e-6
    )


# ── Panel data ────────────────────────────────────────────────────────────────
def test_panel_estimators(grunfeld):
    y, x = grunfeld

    fixed_effects = panel_data_model.get_fixed_effects(y, x)["regression"]
    reference_fe = PanelOLS(y, x, entity_effects=True).fit()
    np.testing.assert_allclose(
        fixed_effects["coefficients"], reference_fe.params.to_numpy(), rtol=1e-9
    )
    np.testing.assert_allclose(
        fixed_effects["standard_errors"], reference_fe.std_errors.to_numpy(), rtol=1e-9
    )
    # Baltagi reports the Grunfeld within estimates as 0.1101 and 0.3100.
    np.testing.assert_allclose(
        fixed_effects["coefficients"], [0.1101, 0.3100], atol=5e-5
    )

    # Unlike Fixed Effects, the Random Effects estimator keeps an intercept.
    random_effects = panel_data_model.get_random_effects(y, x)
    reference_re = RandomEffects(y, sm.add_constant(x)).fit()
    np.testing.assert_allclose(
        random_effects["coefficients"],
        np.r_[
            reference_re.params["const"],
            reference_re.params[["value", "capital"]].to_numpy(),
        ],
        rtol=1e-6,
    )


def test_panel_hausman_test(grunfeld):
    y, x = grunfeld
    hausman = panel_data_model.get_hausman_test(y, x)

    fixed_effects = panel_data_model.get_fixed_effects(y, x)["regression"]
    random_effects = panel_data_model.get_random_effects(y, x)
    slopes = [
        list(random_effects["feature_names"]).index(name)
        for name in ["value", "capital"]
    ]
    difference = fixed_effects["coefficients"] - random_effects["coefficients"][slopes]
    covariance = (
        fixed_effects["covariance_matrix"]
        - random_effects["covariance_matrix"][np.ix_(slopes, slopes)]
    )

    np.testing.assert_allclose(
        hausman["Hausman Statistic"],
        difference @ np.linalg.pinv(covariance) @ difference,
        rtol=1e-9,
    )


# ── Instrumental variables ────────────────────────────────────────────────────
def test_instrumental_variables_on_mroz():
    data = mroz.load().dropna(subset=["lwage"])
    result = causal_inference_model.get_iv_2sls(
        y=data["lwage"], x_endogenous=data[["educ"]], instruments=data[["fatheduc"]]
    )
    reference = IV2SLS(
        data["lwage"],
        pd.DataFrame({"const": 1.0}, index=data.index),
        data[["educ"]],
        data[["fatheduc"]],
    ).fit(cov_type="unadjusted")

    np.testing.assert_allclose(
        result["coefficients"], reference.params.to_numpy(), rtol=1e-7
    )
    # Wooldridge, Introductory Econometrics, Example 15.5.
    np.testing.assert_allclose(result["coefficients"], [0.4411, 0.0592], atol=5e-5)
    np.testing.assert_allclose(result["standard_errors"], [0.4461, 0.0351], atol=5e-5)


def test_instrumental_variables_with_exogenous_controls():
    data = card.load()
    controls = ["exper", "black", "south", "smsa"]
    result = causal_inference_model.get_iv_2sls(
        y=data["lwage"],
        x_endogenous=data[["educ"]],
        instruments=data[["nearc4"]],
        x_exogenous=data[controls],
    )
    reference = IV2SLS(
        data["lwage"], sm.add_constant(data[controls]), data[["educ"]], data[["nearc4"]]
    ).fit(cov_type="unadjusted")

    np.testing.assert_allclose(
        result["coefficients"], reference.params.to_numpy(), rtol=1e-6
    )


def test_durbin_wu_hausman_endogeneity_test():
    data = mroz.load().dropna(subset=["lwage"])
    result = hypothesis_testing_model.get_hausman_wu_test(
        y=data["lwage"], x_suspect=data["educ"], instruments=data[["fatheduc"]]
    )

    first_stage = sm.OLS(data["educ"], sm.add_constant(data[["fatheduc"]])).fit()
    augmented = sm.OLS(
        data["lwage"],
        sm.add_constant(
            pd.DataFrame({"educ": data["educ"], "v_hat": first_stage.resid})
        ),
    ).fit()

    np.testing.assert_allclose(
        result["T-Statistic"], augmented.tvalues["v_hat"], rtol=1e-9
    )
    np.testing.assert_allclose(result["P-Value"], augmented.pvalues["v_hat"], rtol=1e-9)


# ── Unit roots and cointegration ──────────────────────────────────────────────
def test_augmented_dickey_fuller_and_kpss(macro):
    series = macro["log_gdp"]

    adf = unitroot_model.get_augmented_dickey_fuller(series)
    reference_adf = adfuller(series, regression="c", autolag="AIC")
    np.testing.assert_allclose(adf["ADF Statistic"], reference_adf[0], rtol=1e-9)
    np.testing.assert_allclose(adf["P-Value"], reference_adf[1], rtol=1e-9)

    kpss_result = unitroot_model.get_kpss_test(series)
    np.testing.assert_allclose(
        kpss_result["KPSS Statistic"],
        kpss(series, regression="c", nlags="auto")[0],
        rtol=1e-9,
    )


def test_zivot_andrews(macro):
    result = unitroot_model.get_zivot_andrews_test(macro["log_gdp"])
    np.testing.assert_allclose(
        result["Zivot-Andrews Statistic"],
        zivot_andrews(macro["log_gdp"], regression="c", trim=0.15)[0],
        rtol=1e-9,
    )


def test_phillips_perron_matches_arch(macro):
    # Phillips-Perron is one of the few tests here built by hand rather than delegated,
    # so it is checked against a genuinely independent implementation.
    arch_unitroot = pytest.importorskip("arch.unitroot")

    for regression, trend in [("c", "c"), ("ct", "ct")]:
        result = unitroot_model.get_phillips_perron_test(
            macro["log_gdp"], regression=regression
        )
        reference = arch_unitroot.PhillipsPerron(
            macro["log_gdp"].to_numpy(),
            lags=int(result["Lags Used"]),
            trend=trend,
            test_type="tau",
        )
        np.testing.assert_allclose(
            result["Phillips-Perron Statistic"], reference.stat, rtol=1e-8
        )


def test_cointegration_tests(macro):
    first, second = macro["log_gdp"], macro["log_consumption"]

    engle_granger = cointegration_model.get_engle_granger_cointegration(first, second)
    reference_eg = coint(first, second)
    np.testing.assert_allclose(
        engle_granger["Engle-Granger Statistic"], reference_eg[0], rtol=1e-9
    )
    np.testing.assert_allclose(engle_granger["P-Value"], reference_eg[1], rtol=1e-9)

    frame = pd.concat([first, second], axis=1)
    johansen = cointegration_model.get_johansen_cointegration(
        frame, det_order=0, k_ar_diff=1
    )
    reference_johansen = coint_johansen(frame, det_order=0, k_ar_diff=1)
    np.testing.assert_allclose(
        johansen["Trace Statistic"].to_numpy(), reference_johansen.lr1, rtol=1e-9
    )
    np.testing.assert_allclose(
        johansen["Eigenvalue"].to_numpy(), reference_johansen.eig, rtol=1e-9
    )


def test_granger_causality(macro):
    first = macro["log_gdp"].diff().dropna()
    second = macro["log_consumption"].diff().dropna()

    result = causality_model.get_granger_causality(first, second, max_lag=4)
    reference = grangercausalitytests(pd.concat([first, second], axis=1), maxlag=[4])[
        4
    ][0]["ssr_ftest"]

    np.testing.assert_allclose(result["F-Statistic"], reference[0], rtol=1e-9)
    np.testing.assert_allclose(result["P-Value"], reference[1], rtol=1e-9)


# ── Residual diagnostics ──────────────────────────────────────────────────────
def test_residual_diagnostics(macro):
    returns = macro["gdp_returns"]

    jarque_bera_result = diagnostics_model.get_jarque_bera_test(returns)
    reference_jb = jarque_bera(returns)
    np.testing.assert_allclose(
        jarque_bera_result["Jarque-Bera Statistic"], reference_jb[0], rtol=1e-9
    )
    np.testing.assert_allclose(
        jarque_bera_result["P-Value"], reference_jb[1], rtol=1e-9
    )

    ljung_box_result = diagnostics_model.get_ljung_box_test(returns, lags=10)
    reference_lb = acorr_ljungbox(returns, lags=[10])
    np.testing.assert_allclose(
        ljung_box_result["Ljung-Box Statistic"],
        reference_lb["lb_stat"].iloc[0],
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        ljung_box_result["P-Value"], reference_lb["lb_pvalue"].iloc[0], rtol=1e-9
    )

    # The test demeans first, i.e. it treats the input as residuals of a constant mean
    # equation, which is what Engle's test is defined on -- so the reference gets the same.
    arch_result = diagnostics_model.get_arch_lm_test(returns, lags=5)
    reference_arch = het_arch(returns - returns.mean(), nlags=5)
    np.testing.assert_allclose(
        arch_result["ARCH-LM Statistic"], reference_arch[0], rtol=1e-9
    )
    np.testing.assert_allclose(arch_result["P-Value"], reference_arch[1], rtol=1e-9)


def test_cusum_stability_test(macro):
    returns = macro["gdp_returns"]
    result = diagnostics_model.get_cusum_test(returns)
    reference = breaks_cusumolsresid(
        sm.OLS(returns.to_numpy(), np.ones((len(returns), 1))).fit().resid, ddof=1
    )

    np.testing.assert_allclose(result["CUSUM Statistic"], reference[0], rtol=1e-8)
    np.testing.assert_allclose(result["P-Value"], reference[1], rtol=1e-8)


def test_variance_ratio_matches_arch(macro):
    # Also hand-built. The toolkit implements the asymptotic Lo-MacKinlay form without
    # their finite-sample bias correction, so on ~200 observations it sits a few percent
    # closer to 1 than `arch` does -- same conclusion, slightly more conservative.
    arch_unitroot = pytest.importorskip("arch.unitroot")

    returns = macro["gdp_returns"]
    result = diagnostics_model.get_variance_ratio_test(returns, q=4)
    # `arch` takes the level whose increments are being tested, and demeans it.
    reference = arch_unitroot.VarianceRatio(
        np.cumsum(returns.to_numpy()), lags=4, robust=False, overlap=True, trend="c"
    )

    np.testing.assert_allclose(result["Variance Ratio"], reference.vr, rtol=0.05)
    np.testing.assert_allclose(
        result["Variance Ratio Statistic"], reference.stat, rtol=0.10
    )


def test_variance_ratio_is_correctly_sized_under_a_random_walk():
    generator = np.random.default_rng(0)
    statistics = np.array(
        [
            diagnostics_model.get_variance_ratio_test(
                pd.Series(generator.normal(size=800)), q=4
            )["Variance Ratio Statistic"]
            for _ in range(200)
        ]
    )

    # Increments are i.i.d., so the null holds: rejection should sit at or below 5%.
    assert np.mean(np.abs(statistics) > 1.96) <= 0.10
    assert abs(statistics.mean()) < 0.3


# ── Specification tests ───────────────────────────────────────────────────────
def test_specification_tests(statecrime):
    y, x = statecrime
    result = regression_model.get_ols(y, x)
    reference = sm.OLS(y, sm.add_constant(x)).fit()

    breusch_pagan = specification_tests_model.get_breusch_pagan_test(result)
    reference_bp = het_breuschpagan(reference.resid, reference.model.exog)
    np.testing.assert_allclose(
        breusch_pagan["Breusch-Pagan Statistic"], reference_bp[0], rtol=1e-9
    )
    np.testing.assert_allclose(breusch_pagan["P-Value"], reference_bp[1], rtol=1e-9)

    white = specification_tests_model.get_white_test(result)
    reference_white = het_white(reference.resid, reference.model.exog)
    np.testing.assert_allclose(white["White Statistic"], reference_white[0], rtol=1e-9)
    np.testing.assert_allclose(white["P-Value"], reference_white[1], rtol=1e-9)

    np.testing.assert_allclose(
        specification_tests_model.get_durbin_watson_test(result)[
            "Durbin-Watson Statistic"
        ],
        durbin_watson(reference.resid),
        rtol=1e-9,
    )

    reset = specification_tests_model.get_ramsey_reset_test(result, power=3)
    reference_reset = linear_reset(reference, power=3, test_type="fitted", use_f=True)
    np.testing.assert_allclose(
        reset["RESET F-Statistic"], reference_reset.fvalue, rtol=1e-9
    )
    np.testing.assert_allclose(reset["P-Value"], reference_reset.pvalue, rtol=1e-9)

    design = sm.add_constant(x).to_numpy()
    np.testing.assert_allclose(
        specification_tests_model.get_vif(x).to_numpy(),
        [
            variance_inflation_factor(design, position)
            for position in range(1, design.shape[1])
        ],
        rtol=1e-9,
    )


def test_chow_structural_break_test(statecrime):
    y, x = statecrime
    result = regression_model.get_ols(y, x)
    break_index = 25
    chow = specification_tests_model.get_chow_test(
        result, x, y, break_index=break_index
    )

    design = sm.add_constant(x).to_numpy()
    pooled = sm.OLS(y.to_numpy(), design).fit().ssr
    first = sm.OLS(y.to_numpy()[:break_index], design[:break_index]).fit().ssr
    second = sm.OLS(y.to_numpy()[break_index:], design[break_index:]).fit().ssr
    parameters = design.shape[1]
    reference_f = ((pooled - (first + second)) / parameters) / (
        (first + second) / (len(y) - 2 * parameters)
    )

    np.testing.assert_allclose(chow["Chow F-Statistic"], reference_f, rtol=1e-9)
    np.testing.assert_allclose(
        chow["P-Value"],
        st.f.sf(reference_f, parameters, len(y) - 2 * parameters),
        rtol=1e-9,
    )


def test_nested_model_tests(statecrime):
    y, x = statecrime
    unrestricted = regression_model.get_ols(y, x)
    restricted = regression_model.get_ols(y, x[["hs_grad", "poverty"]])

    reference_unrestricted = sm.OLS(y, sm.add_constant(x)).fit()
    reference_restricted = sm.OLS(y, sm.add_constant(x[["hs_grad", "poverty"]])).fit()

    f_test = hypothesis_testing_model.get_f_test(restricted, unrestricted)
    reference_f = reference_unrestricted.compare_f_test(reference_restricted)
    np.testing.assert_allclose(f_test["F-Statistic"], reference_f[0], rtol=1e-9)
    np.testing.assert_allclose(f_test["P-Value"], reference_f[1], rtol=1e-9)

    likelihood_ratio = hypothesis_testing_model.get_likelihood_ratio_test(
        restricted, unrestricted
    )
    reference_lr = len(y) * np.log(
        reference_restricted.ssr / reference_unrestricted.ssr
    )
    np.testing.assert_allclose(
        likelihood_ratio["LR Statistic"], reference_lr, rtol=1e-9
    )
    np.testing.assert_allclose(
        likelihood_ratio["P-Value"], st.chi2.sf(reference_lr, 2), rtol=1e-9
    )

    restrictions = np.zeros((2, 5))
    restrictions[0, 3] = 1.0
    restrictions[1, 4] = 1.0
    wald = hypothesis_testing_model.get_wald_test(unrestricted, restrictions)
    reference_wald = reference_unrestricted.wald_test(
        restrictions, scalar=True, use_f=False
    )
    np.testing.assert_allclose(
        wald["Wald Statistic (Chi2)"], float(reference_wald.statistic), rtol=1e-9
    )
    np.testing.assert_allclose(
        wald["Chi2 P-Value"], float(reference_wald.pvalue), rtol=1e-9
    )


def test_two_sample_t_test():
    sample_a = pd.Series(np.random.default_rng(0).normal(0, 1, 100))
    sample_b = pd.Series(np.random.default_rng(1).normal(0.5, 1, 120))

    result = hypothesis_testing_model.get_two_sample_t_test(
        sample_a, sample_b, equal_variance=False
    )
    reference = st.ttest_ind(sample_a, sample_b, equal_var=False)

    np.testing.assert_allclose(result["T-Statistic"], reference.statistic, rtol=1e-9)
    np.testing.assert_allclose(result["P-Value"], reference.pvalue, rtol=1e-9)


# ── Time series ───────────────────────────────────────────────────────────────
def test_arima_forecast(macro):
    result = time_series_model.get_arima_forecast(
        macro["log_gdp"], p=1, d=1, q=1, forecast_steps=5
    )
    reference = ARIMA(macro["log_gdp"], order=(1, 1, 1), trend="t").fit()

    np.testing.assert_allclose(
        result["forecast"], reference.forecast(5).to_numpy(), rtol=1e-6
    )


def test_var_forecast(var_result):
    frame, result = var_result
    reference = VAR(frame).fit(2)

    np.testing.assert_allclose(
        np.asarray(result["forecast"]),
        reference.forecast(frame.to_numpy()[-2:], 3),
        rtol=1e-9,
    )


def test_impulse_responses_and_variance_decomposition(var_result):
    frame, result = var_result
    reference = VAR(frame).fit(2)
    reference_irf = reference.irf(8)
    reference_fevd = reference.fevd(8)

    orthogonalized = time_series_model.get_impulse_response_function(
        result, periods=8, orthogonalized=True
    )
    simple = time_series_model.get_impulse_response_function(
        result, periods=8, orthogonalized=False
    )
    decomposition = time_series_model.get_variance_decomposition(result, periods=8)

    # statsmodels indexes impulse responses as [horizon, response, shock]; the toolkit
    # keys them by the shocked variable.
    for position, name in enumerate(frame.columns):
        np.testing.assert_allclose(
            orthogonalized["responses"][name].to_numpy(),
            reference_irf.orth_irfs[:, :, position],
            rtol=1e-9,
        )
        np.testing.assert_allclose(
            simple["responses"][name].to_numpy(),
            reference_irf.irfs[:, :, position],
            rtol=1e-9,
        )
        np.testing.assert_allclose(
            decomposition["decomposition"][name].to_numpy(),
            reference_fevd.decomp[position],
            rtol=1e-9,
        )


def test_vecm_forecast():
    # A genuinely cointegrated pair: a random walk plus a stationary spread.
    generator = np.random.default_rng(3)
    random_walk = np.cumsum(generator.normal(0, 1, 300))
    frame = pd.DataFrame(
        {
            "gdp": random_walk,
            "consumption": 2.0 * random_walk + generator.normal(0, 0.5, 300),
        }
    )

    result = time_series_model.get_vecm_forecast(frame, k_ar_diff=1, forecast_steps=4)
    reference = VECM(
        frame, k_ar_diff=1, coint_rank=int(result["rank"]), deterministic="co"
    ).fit()

    np.testing.assert_allclose(
        np.asarray(result["forecast"]), reference.predict(steps=4), rtol=1e-9
    )


def test_vecm_refuses_series_that_are_not_cointegrated():
    generator = np.random.default_rng(5)
    frame = pd.DataFrame(
        generator.normal(size=(200, 2)).cumsum(axis=0), columns=["a", "b"]
    )

    with pytest.raises(ValueError, match="cointegrating rank"):
        time_series_model.get_vecm_forecast(frame, k_ar_diff=1, forecast_steps=3)


# ── Asset pricing and event studies ───────────────────────────────────────────
def test_fama_macbeth_regression():
    data = french.load()
    names = ["NoDur", "Durbl", "Manuf", "Chems", "Enrgy", "Utils", "Shops", "Money"]
    portfolios = data[names].astype(float)
    factors = data[["MktRF", "SMB", "HML"]].astype(float)

    result = fama_macbeth_model.get_fama_macbeth_regression(portfolios, factors)

    # Two-pass reference: per-asset time series betas, then a cross sectional regression
    # each period, averaged over periods (Fama & MacBeth 1973).
    factor_design = sm.add_constant(factors.to_numpy())
    betas = np.column_stack(
        [
            np.linalg.lstsq(factor_design, portfolios[name].to_numpy(), rcond=None)[0][
                1:
            ]
            for name in names
        ]
    ).T
    cross_design = sm.add_constant(betas)
    lambdas = np.array(
        [
            np.linalg.lstsq(cross_design, portfolios.iloc[t].to_numpy(), rcond=None)[0]
            for t in range(len(portfolios))
        ]
    )

    np.testing.assert_allclose(np.asarray(result["betas"]), betas, rtol=1e-9)
    np.testing.assert_allclose(result["risk_premia"], lambdas.mean(axis=0), rtol=1e-9)
    np.testing.assert_allclose(
        result["standard_errors"],
        lambdas.std(axis=0, ddof=1) / np.sqrt(len(lambdas)),
        rtol=1e-9,
    )


def test_event_study():
    generator = np.random.default_rng(11)
    dates = pd.date_range("2020-01-01", periods=400, freq="B").strftime("%Y-%m-%d")
    market = pd.Series(generator.normal(0.0004, 0.010, 400), index=dates)
    asset = pd.Series(
        0.0002 + 1.3 * market.to_numpy() + generator.normal(0, 0.006, 400), index=dates
    )
    event_position = 330

    result = event_study_model.get_event_study(
        asset,
        market,
        event_date=dates[event_position],
        estimation_window=250,
        gap_days=30,
        pre_event_days=10,
        post_event_days=10,
    )

    estimation = slice(event_position - 30 - 250, event_position - 30)
    market_model = sm.OLS(
        asset.iloc[estimation].to_numpy(),
        sm.add_constant(market.iloc[estimation].to_numpy()),
    ).fit()
    window = slice(event_position - 10, event_position + 11)
    abnormal = asset.iloc[window].to_numpy() - (
        market_model.params[0] + market_model.params[1] * market.iloc[window].to_numpy()
    )

    np.testing.assert_allclose(
        [result["alpha"], result["beta"]], market_model.params, rtol=1e-9
    )
    np.testing.assert_allclose(
        result["cumulative_abnormal_return"], abnormal.sum(), rtol=1e-9
    )
    np.testing.assert_allclose(
        result["car_t_statistic"],
        abnormal.sum() / np.sqrt(len(abnormal) * market_model.mse_resid),
        rtol=1e-9,
    )


# ── Causal inference ──────────────────────────────────────────────────────────
def test_difference_in_differences():
    generator = np.random.default_rng(2)
    treated = pd.Series(generator.binomial(1, 0.5, 400).astype(float))
    post = pd.Series(np.tile([0.0, 1.0], 200))
    outcome = pd.Series(
        1.0
        + 0.5 * treated.to_numpy()
        + 0.3 * post.to_numpy()
        + 2.0 * treated.to_numpy() * post.to_numpy()
        + generator.normal(0, 0.4, 400)
    )

    result = causal_inference_model.get_difference_in_differences(
        outcome, treated, post
    )
    design = pd.DataFrame(
        {"treated": treated, "post": post, "interaction": treated * post}
    )
    reference = sm.OLS(outcome, sm.add_constant(design)).fit()

    np.testing.assert_allclose(
        result["coefficients"], reference.params.to_numpy(), rtol=1e-9
    )
    # The simulated interaction effect is 2.0.
    np.testing.assert_allclose(result["coefficients"][3], 2.0, atol=0.15)


def test_regression_discontinuity():
    generator = np.random.default_rng(11)
    running = pd.Series(generator.uniform(-1, 1, 800))
    outcome = pd.Series(
        1.0
        + 2.0 * running.to_numpy()
        + 3.0 * (running.to_numpy() >= 0)
        + generator.normal(0, 0.3, 800)
    )

    result = causal_inference_model.get_regression_discontinuity(
        outcome, running, cutoff=0.0, bandwidth=0.5
    )

    within = np.abs(running.to_numpy()) <= 0.5
    left = sm.OLS(
        outcome.to_numpy()[within & (running.to_numpy() < 0)],
        sm.add_constant(running.to_numpy()[within & (running.to_numpy() < 0)]),
    ).fit()
    right = sm.OLS(
        outcome.to_numpy()[within & (running.to_numpy() >= 0)],
        sm.add_constant(running.to_numpy()[within & (running.to_numpy() >= 0)]),
    ).fit()

    np.testing.assert_allclose(
        result["discontinuity"], right.params[0] - left.params[0], rtol=1e-9
    )
    np.testing.assert_allclose(
        result["standard_error"],
        np.sqrt(left.bse[0] ** 2 + right.bse[0] ** 2),
        rtol=1e-9,
    )
    # The simulated jump at the cutoff is 3.0.
    np.testing.assert_allclose(result["discontinuity"], 3.0, atol=0.05)


def test_propensity_score_matching():
    generator = np.random.default_rng(11)
    covariates = pd.DataFrame(
        {"x1": generator.normal(size=1200), "x2": generator.normal(size=1200)}
    )
    propensity = 1 / (1 + np.exp(-(0.8 * covariates["x1"] - 0.5 * covariates["x2"])))
    treatment = pd.Series(generator.binomial(1, propensity))
    outcome = pd.Series(
        2.0 * treatment.to_numpy()
        + 1.5 * covariates["x1"].to_numpy()
        - covariates["x2"].to_numpy()
        + generator.normal(0, 0.5, 1200)
    )

    result = causal_inference_model.get_propensity_score_matching(
        treatment, outcome, covariates
    )

    np.testing.assert_allclose(
        np.asarray(result["propensity_scores"]),
        sm.Logit(treatment, sm.add_constant(covariates)).fit(disp=0).predict(),
        rtol=1e-6,
    )
    matched = (
        outcome.to_numpy()[result["matched_treated_indices"]]
        - outcome.to_numpy()[result["matched_control_indices"]]
    )
    np.testing.assert_allclose(result["att"], matched.mean(), rtol=1e-9)
    # The simulated treatment effect is 2.0.
    np.testing.assert_allclose(result["att"], 2.0, atol=0.25)


def test_synthetic_control():
    generator = np.random.default_rng(11)
    periods = pd.date_range("2015-01-01", periods=80, freq="ME").strftime("%Y-%m-%d")
    donors = pd.DataFrame(
        generator.normal(0, 1, (80, 6)).cumsum(axis=0),
        index=periods,
        columns=list("ABCDEF"),
    )
    donor_mix = np.array([0.5, 0.3, 0.2, 0.0, 0.0, 0.0])
    treated = pd.Series(
        donors.to_numpy() @ donor_mix + generator.normal(0, 0.05, 80), index=periods
    )
    treated.iloc[60:] += 5.0  # a known intervention

    result = causal_inference_model.get_synthetic_control(
        treated, donors, treatment_period=periods[60]
    )
    weights = np.asarray(result["weights"])

    assert weights.min() >= -1e-9
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)
    np.testing.assert_allclose(weights, donor_mix, atol=0.05)
    np.testing.assert_allclose(result["average_treatment_effect"], 5.0, atol=0.1)


# ── Forecast evaluation ───────────────────────────────────────────────────────
def test_forecast_error_measures():
    actual = pd.Series(np.linspace(1, 10, 50))
    forecast = actual + np.sin(np.arange(50))

    np.testing.assert_allclose(
        forecast_evaluation_model.get_rmse(actual, forecast),
        np.sqrt(np.mean((actual - forecast) ** 2)),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        forecast_evaluation_model.get_mae(actual, forecast),
        np.mean(np.abs(actual - forecast)),
        rtol=1e-12,
    )


def test_diebold_mariano_test():
    generator = np.random.default_rng(7)
    truth = pd.Series(generator.normal(size=200))
    forecast_a = truth + generator.normal(scale=0.5, size=200)
    forecast_b = truth + generator.normal(scale=0.8, size=200)

    result = forecast_evaluation_model.get_diebold_mariano_test(
        truth, forecast_a, forecast_b, small_sample_correction=False
    )
    differential = (truth - forecast_a) ** 2 - (truth - forecast_b) ** 2

    np.testing.assert_allclose(
        result["Diebold-Mariano Statistic"],
        differential.mean() / np.sqrt(differential.var(ddof=0) / len(differential)),
        rtol=1e-9,
    )


def test_volatility_forecast():
    returns = pd.Series(np.random.default_rng(11).normal(0, 0.012, 500))

    np.testing.assert_allclose(
        forecast_evaluation_model.get_volatility_forecast(
            returns, method="ewma", window_size=20, lambda_=0.94
        )
        .dropna()
        .to_numpy(),
        (risk_model.get_ewma_volatility(returns, 0.94) ** 2)
        .shift(1)
        .dropna()
        .to_numpy(),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        forecast_evaluation_model.get_volatility_forecast(
            returns, method="rolling", window_size=20, lambda_=0.94
        )
        .dropna()
        .to_numpy(),
        (returns.rolling(20).std() ** 2).shift(1).dropna().to_numpy(),
        rtol=1e-12,
    )


def test_out_of_sample_validation():
    generator = np.random.default_rng(11)
    series = pd.Series(np.log(np.arange(1, 300) + 5.0) + generator.normal(0, 0.02, 299))

    result = forecast_evaluation_model.get_out_of_sample_validation(
        series, time_series_model.get_arima_forecast, train_fraction=0.8, p=1, d=1, q=0
    )

    split = int(len(series) * 0.8)
    holdout = series.iloc[split:]
    forecast = np.asarray(
        time_series_model.get_arima_forecast(
            series.iloc[:split], p=1, d=1, q=0, forecast_steps=len(holdout)
        )["forecast"]
    )

    np.testing.assert_allclose(
        result["RMSE"],
        np.sqrt(np.mean((holdout.to_numpy() - forecast) ** 2)),
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        result["MAE"], np.mean(np.abs(holdout.to_numpy() - forecast)), rtol=1e-9
    )
