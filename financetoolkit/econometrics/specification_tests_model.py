"""Specification Tests Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

from financetoolkit.econometrics import regression_model

# The conventional 5% level for the boolean 'reject' flags in this module.
SIGNIFICANCE_LEVEL = 0.05

# Rule-of-thumb bands around 2; Durbin-Watson has no closed-form p-value.
DURBIN_WATSON_LOWER_BAND = 1.5
DURBIN_WATSON_UPPER_BAND = 2.5

# The nested F-test needs at least one added power term beyond the linear fit.
MINIMUM_RESET_POWER = 2


def get_breusch_pagan_test(result: dict) -> pd.Series:
    """
    Calculate the Breusch-Pagan test for heteroskedasticity of a fitted OLS
    regression's residuals, via `statsmodels.stats.diagnostic.het_breuschpagan`.

    Also known as: BP test, Breusch-Pagan-Godfrey test.

    The test regresses the *squared* residuals of the original model on the
    original regressors (`result["design_matrix"]`) and tests whether the
    resulting R-squared of that auxiliary regression is significantly
    different from zero:

    - LM = n * R^2_auxiliary

    Which is asymptotically chi-squared distributed with `k - 1` degrees of
    freedom under the null hypothesis of homoskedasticity (constant residual
    variance), where `k` is the number of parameters in the auxiliary
    regression.

    A significant result (low p-value) indicates that the residual variance is
    related to the level of the regressors -- e.g. errors grow (or shrink)
    systematically as a regressor increases -- which means the OLS standard
    errors, t-statistics and p-values (which assume constant residual
    variance) are no longer reliable, even though the coefficient estimates
    themselves remain unbiased. `get_white_test` is a more general (but less
    powerful, given many regressors) alternative that also catches non-linear
    forms of heteroskedasticity that Breusch-Pagan can miss; `regression_model.get_wls`
    is a remedy once heteroskedasticity is detected.

    For more information about the method, see the following paper:

    - Breusch, T.S., & Pagan, A.R. (1979). "A Simple Test for Heteroscedasticity
    and Random Coefficient Variation." Econometrica, 47(5), 1287-1294.

    Args:
        result (dict): A fitted OLS regression, from `regression_model.get_ols`.

    Returns:
        pd.Series: The Breusch-Pagan LM statistic, its p-value, and whether
        homoskedasticity is rejected at the 5% level.

    Notes:
    - Uses `result["design_matrix"]` (the design matrix actually used to fit `result`) as
    the regressors of the auxiliary regression, exactly as specified in the original
    Breusch-Pagan (1979) formulation.
    """
    lm_statistic, p_value, _, _ = het_breuschpagan(
        result["residuals"], result["design_matrix"]
    )

    return pd.Series(
        {
            "Breusch-Pagan Statistic": lm_statistic,
            "P-Value": p_value,
            "Reject Homoskedasticity (5%)": bool(p_value < SIGNIFICANCE_LEVEL),
        }
    )


def get_white_test(result: dict) -> pd.Series:
    """
    Calculate White's test for heteroskedasticity of a fitted OLS regression's
    residuals, via `statsmodels.stats.diagnostic.het_white`.

    Also known as: White's general test.

    A more general version of `get_breusch_pagan_test`: rather than regressing
    the squared residuals on only the original regressors, the auxiliary
    regression uses the full second-order polynomial expansion of the original
    (non-constant) regressors -- each regressor, its square, and every pairwise
    cross-product:

    - e_t^2 = a_0 + SUM_i(a_i * x_it) + SUM_i(b_i * x_it^2) + SUM_{i<j}(c_ij * x_it * x_jt) + v_t

    Same `LM = n * R^2_auxiliary` construction as Breusch-Pagan, chi-squared
    distributed with degrees of freedom equal to the number of terms in the
    polynomial expansion.

    Because it also picks up non-linear and cross-regressor heteroskedasticity
    patterns that Breusch-Pagan cannot (which only allows a linear relationship
    between the regressors and the residual variance), White's test is more
    robust to the *form* of heteroskedasticity. That generality comes at the
    cost of power with many regressors, since the auxiliary regression's size
    grows quadratically in the number of regressors while the sample size does
    not.

    For more information about the method, see the following paper:

    - White, H. (1980). "A Heteroskedasticity-Consistent Covariance Matrix
    Estimator and a Direct Test for Heteroskedasticity." Econometrica, 48(4),
    817-838.

    Args:
        result (dict): A fitted OLS regression, from `regression_model.get_ols`.

    Returns:
        pd.Series: White's LM statistic, its p-value, and whether homoskedasticity is
        rejected at the 5% level.

    Raises:
        ValueError: If the polynomial-expanded auxiliary design is rank-deficient
        (e.g. too few observations relative to the number of original regressors).
    """
    lm_statistic, p_value, _, _ = het_white(
        result["residuals"], result["design_matrix"]
    )

    return pd.Series(
        {
            "White Statistic": lm_statistic,
            "P-Value": p_value,
            "Reject Homoskedasticity (5%)": bool(p_value < SIGNIFICANCE_LEVEL),
        }
    )


def get_durbin_watson_test(result: dict) -> pd.Series:
    """
    Calculate the Durbin-Watson statistic for first-order autocorrelation of a
    fitted OLS regression's residuals, via `statsmodels.stats.stattools.durbin_watson`.

    Also known as: DW statistic.

    - DW = SUM_{t=2}^{n}((e_t - e_(t-1))^2) / SUM_{t=1}^{n}(e_t^2)

    The statistic ranges from 0 to 4. A value near 2 indicates no first-order
    autocorrelation; values below 2 indicate positive autocorrelation
    (consecutive residuals tend to have the same sign -- common in time series
    with an omitted trend or lagged effect); values above 2 indicate negative
    autocorrelation (consecutive residuals tend to alternate in sign). Detecting
    autocorrelated residuals matters because, like heteroskedasticity, it leaves
    OLS coefficients unbiased but invalidates the reported standard errors --
    `regression_model.get_gls` with an AR(1) `omega` is a standard remedy.

    For more information about the method, see the following paper:

    - Durbin, J., & Watson, G.S. (1950). "Testing for Serial Correlation in
    Least Squares Regression I." Biometrika, 37(3/4), 409-428.

    Args:
        result (dict): A fitted OLS regression, from `regression_model.get_ols`.

    Returns:
        pd.Series: The Durbin-Watson statistic and an approximate interpretation flag.

    Notes:
    - Unlike the other tests in this module, Durbin-Watson does not have a simple
    closed-form p-value at a stated significance level -- the classical Durbin-Watson
    tables give upper (`d_U`) and lower (`d_L`) critical bounds (which depend on `n`,
    `k`, and a chosen significance level, and leave an inconclusive region between
    them) rather than an exact distribution. The "Interpretation" returned here is
    therefore only an approximate rule of thumb (`DW < 1.5` -> "Positive Autocorrelation
    Likely", `DW > 2.5` -> "Negative Autocorrelation Likely", otherwise -> "No Strong
    Evidence"), not a formal hypothesis test at a stated significance level.
    """
    durbin_watson_statistic = float(durbin_watson(result["residuals"]))

    if durbin_watson_statistic < DURBIN_WATSON_LOWER_BAND:
        interpretation = "Positive Autocorrelation Likely"
    elif durbin_watson_statistic > DURBIN_WATSON_UPPER_BAND:
        interpretation = "Negative Autocorrelation Likely"
    else:
        interpretation = "No Strong Evidence"

    return pd.Series(
        {
            "Durbin-Watson Statistic": durbin_watson_statistic,
            "Interpretation": interpretation,
        }
    )


def get_vif(x: pd.DataFrame) -> pd.Series:
    """
    Calculate the Variance Inflation Factor (VIF) of each regressor in `x`, via
    `statsmodels.stats.outliers_influence.variance_inflation_factor`.

    Also known as: VIF.

    Multicollinearity -- one regressor being (near) linearly predictable from
    the others -- does not bias OLS coefficients but inflates their standard
    errors, making individual coefficients look statistically insignificant
    even when the regressors jointly explain the dependent variable well. VIF
    quantifies this per regressor by regressing it on all *other* regressors:

    - VIF_i = 1 / (1 - R^2_i)

    Where `R^2_i` is the R-squared of regressing regressor `i` on all other
    regressors in `x`. A VIF of 1 means no correlation with the other
    regressors; a VIF of `V` means that regressor's coefficient variance is `V`
    times larger than it would be under no multicollinearity.

    Unlike the other functions in this module, VIF takes the raw regressor
    matrix directly rather than a fitted regression result dict -- multicollinearity
    is a property of the regressors alone, independent of any particular
    dependent variable or fitted model.

    Args:
        x (pd.DataFrame): The regressor matrix, one column per regressor. A column
        named "Intercept" (e.g. from a regression result dict's `design_matrix` key,
        reassembled into a DataFrame), if present, is excluded automatically -- a
        constant is by definition uncorrelated with everything else and its VIF is
        not meaningful.

    Returns:
        pd.Series: The VIF of each regressor, indexed by column name.

    Raises:
        TypeError: If `x` is not a `pd.DataFrame`.

    Notes:
    - `VIF > 10` is the conventional rule-of-thumb threshold for concerning
    multicollinearity (`VIF > 5` is sometimes used as a stricter cutoff); `VIF < 1`
    is not possible by construction.

    For more information about the method, see the following paper:

    - Marquardt, D.W. (1970). "Generalized Inverses, Ridge Regression, Biased Linear
    Estimation, and Nonlinear Estimation." Technometrics, 12(3), 591-612.
    """
    if not isinstance(x, pd.DataFrame):
        raise TypeError(f"x must be a pd.DataFrame, received {type(x).__name__}.")

    regressor_names = [column for column in x.columns if column != "Intercept"]
    design = sm.add_constant(x[regressor_names].to_numpy())

    vif_values = {
        regressor: variance_inflation_factor(design, index + 1)
        for index, regressor in enumerate(regressor_names)
    }

    return pd.Series(vif_values, name="VIF")


def get_ramsey_reset_test(result: dict, power: int = 3) -> pd.Series:
    """
    Calculate Ramsey's Regression Equation Specification Error Test (RESET) for
    functional form misspecification of a fitted OLS regression, via
    `statsmodels.stats.diagnostic.linear_reset`.

    Also known as: RESET test, Ramsey RESET.

    If the true relationship between `y` and `x` is non-linear (or a relevant
    non-linear transformation/omitted variable is missing), the fitted values
    of a linear model still carry information about that missing non-linearity.
    RESET tests for this indirectly: it augments the original regressors with
    powers of the fitted values (`fitted^2`, ..., `fitted^power`) and tests
    whether those added terms are jointly significant via a standard nested F-test
    comparing the restricted (original) and unrestricted (augmented) model:

    - F = ((RSS_r - RSS_u) / q) / (RSS_u / (n - k_u))

    Where `RSS_r`/`RSS_u` are the restricted/unrestricted residual sums of
    squares, `q` is the number of added power terms (`power - 1`), `n` is the
    number of observations and `k_u` is the number of parameters in the
    unrestricted model. Under the null hypothesis that the linear form is
    correctly specified, `F` is F(q, n - k_u) distributed.

    A significant result (low p-value) suggests the linear functional form is
    misspecified -- e.g. a squared or interaction term, or a transformation
    (log, etc.) of a regressor, is missing from the model. RESET flags *that*
    something is likely missing without saying *what*; it is not, by itself, a
    prescription for which term to add.

    For more information about the method, see the following paper:

    - Ramsey, J.B. (1969). "Tests for Specification Errors in Classical Linear
    Least-Squares Regression Analysis." Journal of the Royal Statistical
    Society, Series B, 31(2), 350-371.

    Args:
        result (dict): A fitted OLS regression, from `regression_model.get_ols`.
        power (int, optional): The highest power of the fitted values to add,
        i.e. `fitted^2` through `fitted^power` are added. Defaults to 3.

    Returns:
        pd.Series: The RESET F-statistic, its p-value, and whether correct
        specification is rejected at the 5% level.

    Raises:
        ValueError: If `power` is less than 2, or the augmented design is
        rank-deficient or has too few observations relative to its parameters.
    """
    if power < MINIMUM_RESET_POWER:
        raise ValueError(
            f"power must be at least {MINIMUM_RESET_POWER}, received {power}."
        )

    # y is recoverable as fitted_values + residuals, by construction.
    target = result["fitted_values"] + result["residuals"]

    try:
        sm_result = sm.OLS(target, result["design_matrix"]).fit()
        reset_result = linear_reset(sm_result, power=power, use_f=True)
    except (ValueError, np.linalg.LinAlgError) as error:
        raise ValueError(
            "Could not fit the RESET-augmented regression -- the augmented design "
            "may be rank-deficient or have too few observations relative to its "
            "parameters."
        ) from error

    f_statistic = float(reset_result.fvalue)
    p_value = float(reset_result.pvalue)

    return pd.Series(
        {
            "RESET F-Statistic": f_statistic,
            "P-Value": p_value,
            "Reject Correct Specification (5%)": bool(p_value < SIGNIFICANCE_LEVEL),
        }
    )


def get_chow_test(
    result_full: dict,
    x: pd.DataFrame | pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    break_index: int,
    add_constant: bool = True,
) -> pd.Series:
    """
    Calculate the Chow test for a structural break at a known point `break_index`.

    Also known as: Chow breakpoint test.

    Splits the sample into a "before" and "after" sub-sample at `break_index`,
    fits OLS separately on each (via `regression_model.get_ols`, i.e.
    `statsmodels.api.OLS`), and F-tests whether allowing the coefficients to
    differ across the two sub-samples significantly reduces the combined
    residual sum of squares relative to the single pooled regression
    (`result_full`, fit on the entire, unsplit sample):

    - F = ((RSS_pooled - (RSS_1 + RSS_2)) / k) / ((RSS_1 + RSS_2) / (n - 2k))

    Where `RSS_pooled` is `result_full`'s residual sum of squares, `RSS_1`/`RSS_2`
    are the residual sums of squares of the "before"/"after" sub-sample
    regressions, `k` is the number of parameters (assumed identical across the
    pooled and both sub-sample models) and `n` is the total number of
    observations. Under the null hypothesis of no structural break (the same
    coefficients apply throughout), `F` is F(k, n - 2k) distributed.

    Neither `statsmodels` nor `linearmodels` ships a dedicated Chow test function
    (unlike `get_ramsey_reset_test`'s `linear_reset` or the heteroskedasticity
    tests above) -- this F-test combination remains hand-built, though all three
    underlying OLS fits it combines are themselves `statsmodels`-backed.

    A significant result (low p-value) indicates that the regression
    relationship differs meaningfully before versus after `break_index` --
    e.g. a regime change, a policy shift, or a structural event -- meaning a
    single pooled model is misspecified across the full sample.

    For more information about the method, see the following paper:

    - Chow, G.C. (1960). "Tests of Equality Between Sets of Coefficients in Two
    Linear Regressions." Econometrica, 28(3), 591-605.

    Args:
        result_full (dict): The OLS regression fit on the entire (unsplit)
        sample, from `regression_model.get_ols`.
        x (pd.DataFrame | pd.Series | np.ndarray): The independent variable(s)/regressor(s),
        matching the same sample (and row order) that produced `result_full`, excluding any
        constant column -- pass the same `x` originally given to `get_ols`, not
        `result_full["design_matrix"]`.
        y (pd.Series | np.ndarray): The dependent variable, matching `x`.
        break_index (int): The positional index (0-based) at which to split the sample --
        rows `[0, break_index)` form the "before" sub-sample, rows `[break_index, n)` form
        the "after" sub-sample.
        add_constant (bool, optional): Whether to include an intercept in both sub-sample
        regressions (should match how `result_full` itself was fit). Defaults to True.

    Returns:
        pd.Series: The Chow F-statistic, its p-value, and whether no structural break is
        rejected at the 5% level.

    Raises:
        TypeError: If `x` or `y` is not one of the accepted types.
        ValueError: If `break_index` does not leave at least one observation on each side,
        or either sub-sample has too few observations to estimate its `k` parameters.
    """
    x_values, _ = (
        regression_model._to_design_matrix(  # pylint: disable=protected-access
            x, add_constant=False
        )
    )
    y_values = regression_model._to_target_vector(y)  # pylint: disable=protected-access

    n_observations = x_values.shape[0]

    if not 0 < break_index < n_observations:
        raise ValueError(
            f"break_index must leave at least one observation on each side of the "
            f"split -- received break_index={break_index} for {n_observations} "
            f"observations."
        )

    k_parameters = x_values.shape[1] + (1 if add_constant else 0)
    n_before = break_index
    n_after = n_observations - break_index

    if n_before <= k_parameters:
        raise ValueError(
            f"Not enough observations before break_index ({n_before}) to estimate "
            f"{k_parameters} parameters -- need strictly more observations than "
            f"parameters in the 'before' sub-sample."
        )
    if n_after <= k_parameters:
        raise ValueError(
            f"Not enough observations after break_index ({n_after}) to estimate "
            f"{k_parameters} parameters -- need strictly more observations than "
            f"parameters in the 'after' sub-sample."
        )

    result_before = regression_model.get_ols(
        y_values[:break_index], x_values[:break_index], add_constant=add_constant
    )
    result_after = regression_model.get_ols(
        y_values[break_index:], x_values[break_index:], add_constant=add_constant
    )

    residual_sum_of_squares_pooled = float(np.sum(result_full["residuals"] ** 2))
    residual_sum_of_squares_before = float(np.sum(result_before["residuals"] ** 2))
    residual_sum_of_squares_after = float(np.sum(result_after["residuals"] ** 2))
    residual_sum_of_squares_split = (
        residual_sum_of_squares_before + residual_sum_of_squares_after
    )

    denominator_degrees_of_freedom = n_observations - 2 * k_parameters

    f_statistic = (
        (residual_sum_of_squares_pooled - residual_sum_of_squares_split) / k_parameters
    ) / (residual_sum_of_squares_split / denominator_degrees_of_freedom)
    p_value = stats.f.sf(f_statistic, k_parameters, denominator_degrees_of_freedom)

    return pd.Series(
        {
            "Chow F-Statistic": f_statistic,
            "P-Value": p_value,
            "Reject No Structural Break (5%)": bool(p_value < SIGNIFICANCE_LEVEL),
        }
    )
