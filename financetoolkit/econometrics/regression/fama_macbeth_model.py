"""Fama-MacBeth Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd
from scipy import stats

from financetoolkit.econometrics.least_squares_model import get_pseudo_inverse
from financetoolkit.econometrics.regression import regression_model

# Second-pass identification needs strictly more assets than coefficients.
MINIMUM_ASSET_BUFFER = 1

# The standard error is a time-series std of the per-period coefficients.
MINIMUM_PERIODS = 2


def fama_macbeth_summary_table(result: dict) -> pd.DataFrame:
    """
    Build the risk premium table of a Fama-MacBeth regression: the Risk Premium, Standard
    Error, t-Statistic and P-Value of every factor.

    The table takes a `get_fama_macbeth_regression` result dict and is indexed by factor
    name (plus "Intercept" if present).

    Args:
        result (dict): A fitted Fama-MacBeth result dict, as returned by
        `get_fama_macbeth_regression`.

    Returns:
        pd.DataFrame: The coefficient table.
    """
    return pd.DataFrame(
        {
            "Risk Premium": result["risk_premia"],
            "Standard Error": result["standard_errors"],
            "t-Statistic": result["t_statistics"],
            "P-Value": result["p_values"],
        }
    )


def get_fama_macbeth_regression(
    returns: pd.DataFrame,
    factors: pd.Series | pd.DataFrame,
    add_constant: bool = True,
) -> dict:
    """
    Fit a Fama-MacBeth (1973) two-pass cross-sectional regression: the risk premium of every
    factor, and whether it is priced in the cross-section of asset returns.

    Also known as: two-pass regression, Fama-MacBeth procedure.

    The method has two passes:

    1. **Time-series pass (per asset):** for each asset `i`, regress its return series
       on the factor(s) over the full sample to estimate that asset's factor
       loadings/betas: `R_i,t = alpha_i + beta_i' * F_t + e_i,t`.
    2. **Cross-sectional pass (per period):** for each period `t`, regress that
       period's cross-section of asset returns on the (time-invariant, from step 1)
       betas: `R_i,t = gamma_0,t + gamma_t' * beta_i + u_i,t`. This produces one
       estimate of the factor risk premium `gamma_t` per period.

    The final risk premium estimate is the time-series average of the `gamma_t`
    series, and -- the technique's key contribution -- its standard error is computed
    from the time-series VARIATION of `gamma_t` itself (`std(gamma_t) / sqrt(T)`)
    rather than from either pass's own regression standard errors. This automatically
    accounts for cross-sectional correlation in the residuals (which contaminates
    ordinary panel/pooled standard errors) without requiring it to be modeled
    explicitly.

    - First pass: `R_i,t = alpha_i + beta_i' * F_t + e_i,t`, for each asset `i`.
    - Second pass: `R_i,t = gamma_0,t + gamma_t' * beta_i + u_i,t`, for each period `t`.
    - Risk premium: `gamma_hat = mean_t(gamma_t)`.
    - Standard error: `SE(gamma_hat) = std_t(gamma_t) / sqrt(T)`.

    For more information about the method, see the following papers:

    - Fama, E.F. & MacBeth, J.D. (1973). "Risk, Return, and Equilibrium: Empirical
      Tests." Journal of Political Economy, 81(3), 607-636.
    - Cochrane, J.H. (2005). "Asset Pricing." Princeton University Press, Chapter 12.

    Shares similar logic:
    - linearmodels, FamaMacBeth: https://bashtage.github.io/linearmodels/panel/panel/linearmodels.panel.model.FamaMacBeth.html

    Args:
        returns (pd.DataFrame): One column per asset, one row per period -- the
            cross-section of returns the risk premia are estimated from.
        factors (pd.Series | pd.DataFrame): One column per factor (a single `pd.Series`
            for a one-factor model, e.g. the market factor for CAPM), aligned to
            `returns`'s index -- typically the output of a `performance` module factor
            model (e.g. Fama-French factors) or a benchmark excess return series.
        add_constant (bool, optional): Whether to include an intercept in the
            second-pass cross-sectional regression. Under a well-specified factor
            model this intercept should be economically close to zero (no
            unexplained/"pricing error" excess return) -- a large, significant
            intercept is evidence against the factor model. Defaults to True.

    Returns:
        dict: The estimated risk premia, their Fama-MacBeth standard errors/
        t-statistics/p-values, the underlying first-pass betas and the full
        second-pass coefficient history -- keys `risk_premia`, `standard_errors`,
        `t_statistics`, `p_values`, `betas`, `cross_sectional_coefficients`,
        `n_assets`, `n_periods`, `factor_names`. Call `fama_macbeth_summary_table`
        for a coefficient table.

    Raises:
        TypeError: If `returns` is not a `pd.DataFrame` or `factors` is not a
        `pd.Series`/`pd.DataFrame`.
        ValueError: If `returns` and `factors` share no overlapping periods, there are
        fewer than 2 overlapping periods, or there are not enough assets (columns of
        `returns`) to identify the cross-sectional regression (strictly more assets
        than factors, plus an intercept when a constant is added).

    Notes:
    - This implements the standard, simplified textbook variant using a SINGLE,
      full-sample set of first-pass betas held fixed across every second-pass period
      (see Cochrane, 2005, Ch. 12) -- not Fama & MacBeth's (1973) original
      rolling-portfolio-beta variant, which re-estimates betas from a trailing window
      before each cross-sectional regression. The fixed-beta version is standard
      practice and materially simpler, at the cost of not letting betas drift over
      time within the sample.
    - Unlike linearmodels' FamaMacBeth, which runs the cross-sectional regressions on given
      regressors, the first pass here estimates the betas from the returns.
    - Verified by simulating a synthetic single-factor model with a known risk
      premium and confirming the estimated `risk_premia` recovers it closely, and by
      confirming a zero-premium (pure noise) factor's estimated risk premium is not
      statistically distinguishable from zero.

    As an example:

    ```python
    import numpy as np
    import pandas as pd
    from financetoolkit.econometrics.regression import fama_macbeth_model

    rng = np.random.default_rng(1)
    n_periods, n_assets = 200, 25
    true_betas = rng.uniform(0.5, 1.5, n_assets)
    true_premium = 0.005

    factor = pd.Series(rng.standard_normal(n_periods) * 0.02, name="Market")
    returns = pd.DataFrame(
        {
            f"Asset_{i}": true_betas[i] * true_premium
            + true_betas[i] * factor.to_numpy()
            + rng.standard_normal(n_periods) * 0.01
            for i in range(n_assets)
        }
    )

    result = fama_macbeth_model.get_fama_macbeth_regression(returns, factor)
    print(fama_macbeth_model.fama_macbeth_summary_table(result).round(4))
    ```
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError(
            f"returns must be a pd.DataFrame, received {type(returns).__name__}."
        )
    if not isinstance(factors, (pd.Series, pd.DataFrame)):
        raise TypeError(
            "factors must be a pd.Series or pd.DataFrame, received "
            f"{type(factors).__name__}."
        )

    if isinstance(factors, pd.Series):
        factors = factors.to_frame(
            factors.name if factors.name is not None else "Factor"
        )

    factor_names = list(factors.columns)
    asset_names = list(returns.columns)

    aligned = pd.concat([returns, factors], axis=1, join="inner").dropna()
    n_periods = len(aligned)

    if n_periods < MINIMUM_PERIODS:
        raise ValueError(
            f"Not enough overlapping periods ({n_periods}) between returns and "
            f"factors -- need at least {MINIMUM_PERIODS}."
        )

    aligned_returns = aligned[asset_names]
    aligned_factors = aligned[factor_names]

    n_assets = len(asset_names)
    n_factors = len(factor_names)
    minimum_assets = n_factors + (1 if add_constant else 0) + MINIMUM_ASSET_BUFFER
    if n_assets < minimum_assets:
        raise ValueError(
            f"Not enough assets ({n_assets}) to identify the cross-sectional "
            f"regression -- need at least {minimum_assets}."
        )

    # First pass: one time-series regression per asset for its factor loadings. Every asset
    # is regressed on the same factors, so one pseudo-inverse of the design solves them all.
    return_values = aligned_returns.to_numpy(dtype=float)
    factor_design, _ = regression_model._to_design_matrix(
        aligned_factors, add_constant=True
    )
    regression_model._validate_design(factor_design, return_values[:, 0])
    betas = pd.DataFrame(
        (get_pseudo_inverse(factor_design)[0] @ return_values)[1:].T,
        index=asset_names,
        columns=factor_names,
    )

    # Second pass: one cross-sectional regression of returns on first-pass betas per period,
    # all on the same betas, so again one pseudo-inverse solves every period at once.
    coefficient_names = (["Intercept"] if add_constant else []) + factor_names
    beta_design, _ = regression_model._to_design_matrix(betas, add_constant)
    regression_model._validate_design(beta_design, return_values[0])
    cross_sectional_coefficients = pd.DataFrame(
        (get_pseudo_inverse(beta_design)[0] @ return_values.T).T,
        index=aligned.index,
        columns=coefficient_names,
    )

    risk_premia = cross_sectional_coefficients.mean()
    standard_errors = cross_sectional_coefficients.std(ddof=1) / np.sqrt(n_periods)
    t_statistics = risk_premia / standard_errors
    p_values = pd.Series(
        2 * (1 - stats.t.cdf(np.abs(t_statistics), n_periods - 1)),
        index=coefficient_names,
    )

    return {
        "risk_premia": risk_premia,
        "standard_errors": standard_errors,
        "t_statistics": t_statistics,
        "p_values": p_values,
        "betas": betas,
        "cross_sectional_coefficients": cross_sectional_coefficients,
        "n_assets": n_assets,
        "n_periods": n_periods,
        "factor_names": factor_names,
    }
