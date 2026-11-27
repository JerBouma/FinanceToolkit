"""Dispersion Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.risk import risk_model
from financetoolkit.risk.distribution import dispersion_model

# Twenty returns; the reference values below come from riskfolio-lib 7.4.1, skfolio and the
# formulas of PerformanceAnalytics (R) on these returns.
RETURNS = pd.Series(
    [0.012, -0.023, 0.004, 0.031, -0.008, -0.041, 0.017, 0.009, -0.002, 0.026]
    + [-0.015, 0.006, -0.033, 0.021, 0.001, -0.012, 0.038, -0.005, 0.014, -0.019]
)
# 250 returns, enough for the L-moment risk measure's constraints to be met.
LONG_RETURNS = pd.Series(np.random.default_rng(42).normal(0.0005, 0.01, 250))


def test_gini_mean_difference_matches_references():
    # riskfolio-lib GMD and skfolio gini_mean_difference.
    assert dispersion_model.get_gini_mean_difference(RETURNS) == pytest.approx(
        0.024563157894736846, rel=1e-12
    )
    pairwise = np.abs(RETURNS.to_numpy()[:, None] - RETURNS.to_numpy()[None, :])
    assert dispersion_model.get_gini_mean_difference(RETURNS) == pytest.approx(
        pairwise.sum() / (20 * 19), rel=1e-12
    )


def test_tail_gini_matches_riskfolio():
    # riskfolio-lib TG reports the loss as a positive number.
    assert dispersion_model.get_tail_gini(RETURNS) == pytest.approx(-0.041, rel=1e-12)
    assert dispersion_model.get_tail_gini(RETURNS, alpha=0.2) == pytest.approx(
        -0.03374950503264968, rel=1e-12
    )


def test_risk_range_matches_riskfolio_and_the_toolkit_var():
    assert dispersion_model.get_risk_range(RETURNS) == pytest.approx(0.079, rel=1e-12)
    # riskfolio-lib TGRG(alpha=0.1).
    assert dispersion_model.get_risk_range(
        RETURNS, "tail_gini", alpha=0.1
    ) == pytest.approx(0.0752496219008265, rel=1e-12)
    # The VaR range is the distance between the interpolated quantiles (the historic VaR).
    assert dispersion_model.get_risk_range(RETURNS, "var", alpha=0.25) == pytest.approx(
        RETURNS.quantile(0.75) - RETURNS.quantile(0.25), rel=1e-12
    )
    returns = RETURNS.copy()
    lower, upper = returns.quantile(0.1), returns.quantile(0.8)
    assert dispersion_model.get_risk_range(
        returns, "cvar", alpha=0.1, beta=0.2
    ) == pytest.approx(
        returns[returns >= upper].mean() - returns[returns <= lower].mean(), rel=1e-12
    )


def test_l_moments_match_riskfolio_and_scipy():
    expected = [
        0.0010500000000000004,
        0.012281578947368421,
        -0.0005921052631578945,
        0.0014270897832817336,
    ]
    for order, value in enumerate(expected, start=1):
        assert dispersion_model.get_l_moment(RETURNS, order) == pytest.approx(
            value, rel=1e-10
        )
    # scipy stats.lmoment with standardize=True.
    assert dispersion_model.get_l_moment(RETURNS, 3, ratio=True) == pytest.approx(
        -0.04821084208270773, rel=1e-10
    )
    assert dispersion_model.get_l_moment(RETURNS, 4, ratio=True) == pytest.approx(
        0.1161975825256171, rel=1e-10
    )
    assert dispersion_model.get_l_moment(RETURNS, 2, ratio=True) == pytest.approx(
        11.69674185463659, rel=1e-10
    )
    assert np.isnan(dispersion_model.get_l_moment(RETURNS.iloc[:3], 4))


def test_l_moment_risk_measure():
    # riskfolio-lib L_Moment_CRM(method="CRRA").
    assert dispersion_model.get_l_moment_risk_measure(
        RETURNS, method="crra"
    ) == pytest.approx(0.007185432902743674, rel=1e-10)
    assert dispersion_model.get_l_moment_risk_measure(
        LONG_RETURNS, method="crra"
    ) == pytest.approx(0.0029473880475344868, rel=1e-10)
    # The convex programs solved by cvxpy with tight tolerances (riskfolio-lib's default
    # Clarabel tolerances give 0.00262039 and 0.00253052).
    assert dispersion_model.get_l_moment_risk_measure(LONG_RETURNS) == pytest.approx(
        0.0026205686892689147, rel=1e-7
    )
    assert dispersion_model.get_l_moment_risk_measure(
        LONG_RETURNS, method="mss"
    ) == pytest.approx(0.002563777887288079, rel=1e-6)
    # Twenty returns leave no weights that meet the constraints.
    with pytest.raises(ValueError, match="constraints"):
        dispersion_model.get_l_moment_risk_measure(RETURNS)
    with pytest.raises(ValueError, match="'msd', 'mss' or 'crra'"):
        dispersion_model.get_l_moment_risk_measure(RETURNS, method="me")


def test_central_moments_and_worst_realization():
    # skfolio third_central_moment and fourth_central_moment, riskfolio-lib Kurtosis.
    assert dispersion_model.get_central_moment(RETURNS, 3) == pytest.approx(
        -1.7086372500000022e-06, rel=1e-10
    )
    assert dispersion_model.get_central_moment(RETURNS, 4) == pytest.approx(
        4.1984388873125006e-07, rel=1e-10
    )
    assert np.sqrt(dispersion_model.get_central_moment(RETURNS, 4)) == pytest.approx(
        0.0006479536161881111, rel=1e-10
    )
    # riskfolio-lib WR and skfolio worst_realization report the loss as positive.
    assert dispersion_model.get_worst_realization(RETURNS) == -0.041


def test_skewness_kurtosis_ratio():
    assert dispersion_model.get_skewness_kurtosis_ratio(RETURNS) == pytest.approx(
        RETURNS.skew() / (RETURNS.kurt() + 3), rel=1e-12
    )


def test_partial_moments_match_references():
    # skfolio first_lower_partial_moment and fourth_lower_partial_moment (around the mean).
    assert risk_model.get_partial_moment(RETURNS, 1, "mean") == pytest.approx(
        0.008374999999999999, rel=1e-12
    )
    assert risk_model.get_partial_moment(RETURNS, 4, "mean") == pytest.approx(
        2.535202994281251e-07, rel=1e-12
    )
    # riskfolio-lib LPM(MAR=0, p=1) and SemiKurtosis.
    assert risk_model.get_partial_moment(RETURNS) == pytest.approx(0.0079, rel=1e-12)
    assert risk_model.get_partial_moment(
        RETURNS, 4, "mean", root=True
    ) ** 2 == pytest.approx(0.000503507993410358, rel=1e-10)
    # PerformanceAnalytics DownsideFrequency, UpsideFrequency and UpsideRisk at MAR = 0.002.
    assert risk_model.get_partial_moment(RETURNS, 0, 0.002) == 0.5
    assert risk_model.get_partial_moment(RETURNS, 0, 0.002, "upper") == 0.5
    assert risk_model.get_partial_moment(
        RETURNS, 2, 0.002, "upper", root=True
    ) == pytest.approx(0.013438749941865873, rel=1e-12)
    assert risk_model.get_partial_moment(RETURNS, 2, 0.002, "upper") == pytest.approx(
        0.0001806, rel=1e-12
    )
    assert risk_model.get_partial_moment(RETURNS, 1, 0.002, "upper") == pytest.approx(
        0.0079, rel=1e-12
    )
    with pytest.raises(ValueError):
        risk_model.get_partial_moment(RETURNS, 0, root=True)


def test_volatility_skewness_and_semi_deviation_match_references():
    # PerformanceAnalytics VolatilitySkewness at MAR = 0.002.
    assert risk_model.get_volatility_skewness(RETURNS, 0.002) == pytest.approx(
        0.7539135879774574, rel=1e-12
    )
    assert risk_model.get_volatility_skewness(
        RETURNS, 0.002, "variability"
    ) == pytest.approx(0.8682819749237327, rel=1e-12)
    assert np.isnan(risk_model.get_volatility_skewness(RETURNS.abs()))
    # riskfolio-lib SemiDeviation and skfolio semi_deviation.
    assert risk_model.get_downside_deviation(RETURNS, "mean") == pytest.approx(
        0.015327520828132712, rel=1e-12
    )


def test_measures_per_period_and_rolling():
    index = pd.MultiIndex.from_product(
        [pd.period_range("2020", periods=2, freq="Y"), range(10)]
    )
    frame = pd.DataFrame(
        {"A": RETURNS.to_numpy(), "B": RETURNS.to_numpy()[::-1]}, index=index
    )
    frame.iloc[0, 1] = np.nan

    per_period = dispersion_model.get_gini_mean_difference(frame)
    assert list(per_period.index) == list(index.levels[0])
    assert per_period.loc[index.levels[0][0], "B"] == pytest.approx(
        dispersion_model.get_gini_mean_difference(frame["B"].iloc[1:10])
    )

    rolling = dispersion_model.get_gini_mean_difference(RETURNS, window_size=5)
    assert rolling.iloc[:4].isna().all()
    assert rolling.iloc[10] == pytest.approx(
        dispersion_model.get_gini_mean_difference(RETURNS.iloc[6:11])
    )
    semi = risk_model.get_downside_deviation(RETURNS, "mean", window=8)
    assert semi.iloc[12] == pytest.approx(
        risk_model.get_downside_deviation(RETURNS.iloc[5:13], "mean")
    )


def test_get_risk_range_evar():
    # riskfolio-lib's EVRG(RETURNS, 0.05, 0.1), solved with its exponential cone; with beta = alpha
    # the worst loss and the best gain are each one twentieth of the returns, so both EVaRs are those
    # returns and the range is max - min.
    assert np.isclose(
        dispersion_model.get_risk_range(RETURNS, "evar", 0.05, 0.1),
        0.07711499516277938,
        atol=1e-9,
    )
    assert np.isclose(
        dispersion_model.get_risk_range(RETURNS, "evar", 0.05),
        RETURNS.max() - RETURNS.min(),
    )
    assert dispersion_model.get_risk_range(
        RETURNS, "evar", 0.05, 0.1
    ) > dispersion_model.get_risk_range(RETURNS, "cvar", 0.05, 0.1)
