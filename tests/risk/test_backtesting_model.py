"""VaR Backtesting Model Tests"""

import numpy as np
import pandas as pd
from scipy import stats as spstats

from financetoolkit.risk import backtesting_model

# pylint: disable=missing-function-docstring

SIGNIFICANCE_LEVEL = 0.05


def test_get_kupiec_test(recorder):
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.standard_normal(500) * 0.02)
    var_estimates = pd.Series(np.percentile(returns, 5) * np.ones(500))
    recorder.capture(
        backtesting_model.get_kupiec_test(returns, var_estimates, alpha=0.05).round(4)
    )


def test_get_kupiec_test_dataframe(recorder):
    rng = np.random.default_rng(1)
    returns_df = pd.DataFrame(
        {
            "AAPL": rng.standard_normal(500) * 0.02,
            "MSFT": rng.standard_normal(500) * 0.02,
        }
    )
    var_estimates_df = pd.DataFrame(
        {
            "AAPL": np.percentile(returns_df["AAPL"], 5) * np.ones(500),
            "MSFT": np.percentile(returns_df["MSFT"], 5) * np.ones(500),
        }
    )
    recorder.capture(
        backtesting_model.get_kupiec_test(
            returns_df, var_estimates_df, alpha=0.05
        ).round(4)
    )


def test_get_kupiec_test_never_breached(recorder):
    returns = pd.Series(np.random.default_rng(2).standard_normal(300) * 0.01)
    var_estimates = pd.Series(-1.0 * np.ones(300))
    recorder.capture(
        backtesting_model.get_kupiec_test(returns, var_estimates, alpha=0.05).round(4)
    )


def test_get_christoffersen_test(recorder):
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.standard_normal(500) * 0.02)
    var_estimates = pd.Series(np.percentile(returns, 5) * np.ones(500))
    recorder.capture(
        backtesting_model.get_christoffersen_test(returns, var_estimates).round(4)
    )


def test_get_christoffersen_test_dataframe(recorder):
    rng = np.random.default_rng(1)
    returns_df = pd.DataFrame(
        {
            "AAPL": rng.standard_normal(500) * 0.02,
            "MSFT": rng.standard_normal(500) * 0.02,
        }
    )
    var_estimates_df = pd.DataFrame(
        {
            "AAPL": np.percentile(returns_df["AAPL"], 5) * np.ones(500),
            "MSFT": np.percentile(returns_df["MSFT"], 5) * np.ones(500),
        }
    )
    recorder.capture(
        backtesting_model.get_christoffersen_test(returns_df, var_estimates_df).round(4)
    )


def test_get_acerbi_szekely_test_well_calibrated(recorder):
    rng = np.random.default_rng(9)
    sigma = 0.02
    alpha = 0.05
    returns = pd.Series(rng.standard_normal(1000) * sigma)
    za = spstats.norm.ppf(alpha)
    true_var = za * sigma
    true_cvar = -sigma * spstats.norm.pdf(za) / alpha
    result = backtesting_model.get_acerbi_szekely_test(
        returns,
        pd.Series(np.full(1000, true_var)),
        pd.Series(np.full(1000, true_cvar)),
        alpha=alpha,
        n_bootstrap=200,
        random_state=1,
    )
    assert abs(result["Acerbi-Szekely Statistic"]) < 1
    assert result["P-Value"] > SIGNIFICANCE_LEVEL
    recorder.capture(result.round(4))


def test_get_acerbi_szekely_test_underestimated_risk(recorder):
    rng = np.random.default_rng(9)
    sigma = 0.02
    alpha = 0.05
    returns = pd.Series(rng.standard_normal(1000) * sigma)
    za = spstats.norm.ppf(alpha)
    true_var = za * sigma
    true_cvar = -sigma * spstats.norm.pdf(za) / alpha
    understated_cvar = true_cvar * 0.3
    result = backtesting_model.get_acerbi_szekely_test(
        returns,
        pd.Series(np.full(1000, true_var)),
        pd.Series(np.full(1000, understated_cvar)),
        alpha=alpha,
        n_bootstrap=200,
        random_state=1,
    )
    assert result["Acerbi-Szekely Statistic"] > 0
    assert result["P-Value"] < SIGNIFICANCE_LEVEL
    recorder.capture(result.round(4))


def test_get_acerbi_szekely_test_dataframe(recorder):
    rng = np.random.default_rng(1)
    returns_df = pd.DataFrame(
        {
            "AAPL": rng.standard_normal(500) * 0.02,
            "MSFT": rng.standard_normal(500) * 0.02,
        }
    )
    var_df = pd.DataFrame(
        {
            "AAPL": np.percentile(returns_df["AAPL"], 5) * np.ones(500),
            "MSFT": np.percentile(returns_df["MSFT"], 5) * np.ones(500),
        }
    )
    cvar_df = pd.DataFrame(
        {
            "AAPL": returns_df["AAPL"][returns_df["AAPL"] <= var_df["AAPL"]].mean()
            * np.ones(500),
            "MSFT": returns_df["MSFT"][returns_df["MSFT"] <= var_df["MSFT"]].mean()
            * np.ones(500),
        }
    )
    recorder.capture(
        backtesting_model.get_acerbi_szekely_test(
            returns_df, var_df, cvar_df, alpha=0.05, n_bootstrap=100, random_state=1
        ).round(4)
    )


def test_get_acerbi_szekely_test_too_few_observations(recorder):
    returns = pd.Series([0.01, -0.02, 0.03])
    var_estimates = pd.Series([-0.01, -0.01, -0.01])
    cvar_estimates = pd.Series([-0.02, -0.02, -0.02])
    recorder.capture(
        backtesting_model.get_acerbi_szekely_test(
            returns, var_estimates, cvar_estimates, alpha=0.05
        )
    )
