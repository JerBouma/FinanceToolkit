"""Specification Tests Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd
from scipy import stats

from financetoolkit.econometrics.least_squares_model import (
    fit_least_squares,
    get_wald_statistic,
)
from financetoolkit.econometrics.regression import regression_model

# The conventional 5% level for the boolean 'reject' flags in this module.
SIGNIFICANCE_LEVEL = 0.05

# Rule-of-thumb bands around 2; Durbin-Watson has no closed-form p-value.
DURBIN_WATSON_LOWER_BAND = 1.5
DURBIN_WATSON_UPPER_BAND = 2.5

# The nested F-test needs at least one added power term beyond the linear fit.
MINIMUM_RESET_POWER = 2

# The cluster labels and the HAC lag are not part of a regression result, so a RESET refit cannot reproduce those two estimators -- see `get_ramsey_reset_test`'s Notes.  # noqa: E501
UNSUPPORTED_RESET_COV_TYPES = ("cluster", "HAC")

# A heteroskedasticity test needs a constant and at least one other regressor.
MINIMUM_HETEROSKEDASTICITY_REGRESSORS = 2

# A sum of squared residuals at most this share of the dependent variable's is rounding noise.
PERFECT_FIT_TOLERANCE = 1e-20


def _with_constant(design: np.ndarray) -> np.ndarray:
    """
    Prepends a column of ones to `design` unless one of its columns already is a nonzero
    constant, so an auxiliary regression always has an intercept. The check is on a
    nonzero constant column only, as statsmodels' `add_constant(has_constant="skip")`.
    """
    column_maximum = design.max(axis=0)
    if np.any((column_maximum == design.min(axis=0)) & (column_maximum != 0)):
        return design
    return np.column_stack([np.ones(design.shape[0]), design])


def _is_perfect_fit(result: dict) -> bool:
    """
    Whether the regression fits its dependent variable exactly: residuals that are only
    rounding noise (relative to the dependent variable) carry no information about their
    variance or autocorrelation, so the tests on them are not defined.
    """
    residuals = np.asarray(result["residuals"], dtype=float)
    dependent = residuals + np.asarray(result["fitted_values"], dtype=float)
    return bool(np.sum(residuals**2) <= PERFECT_FIT_TOLERANCE * np.sum(dependent**2))


def _get_heteroskedasticity_lm(
    residuals: np.ndarray,
    auxiliary_design: np.ndarray,
    test_name: str,
    perfect_fit: bool = False,
) -> tuple[float, float]:
    """
    The LM statistic n * R² of the regression of the squared residuals on
    `auxiliary_design` (which holds a constant) and its chi-squared p-value, with the
    degrees of freedom the rank of the design minus the constant, so a duplicated
    column (the square of a 0/1 dummy) is not counted twice.
    """
    if auxiliary_design.shape[1] < MINIMUM_HETEROSKEDASTICITY_REGRESSORS:
        raise ValueError(
            f"The {test_name} test requires exog to have at least two columns where "
            "one is a constant."
        )

    if perfect_fit:
        return np.nan, np.nan

    auxiliary = fit_least_squares(residuals**2, auxiliary_design)
    lm_statistic = len(residuals) * auxiliary["r_squared"]
    p_value = stats.chi2.sf(lm_statistic, auxiliary["rank"] - 1)

    return float(lm_statistic), float(p_value)


def get_breusch_pagan_test(result: dict) -> pd.Series:
    """
    Calculate the Breusch-Pagan test: whether the residual variance of a fitted OLS
    regression changes with its regressors (heteroskedasticity).

    Also known as: BP test, Breusch-Pagan-Godfrey test.

    The test regresses the *squared* residuals of the original model on the
    original regressors (`result["design_matrix"]`) and tests whether the
    resulting R-squared of that auxiliary regression is significantly
    different from zero:

    - e_t^2 = a_0 + SUM_i(a_i * x_it) + v_t
    - LM = n * R^2_auxiliary

    This is the studentized form of Koenker (1981), which is also statsmodels' default:
    unlike the original Breusch-Pagan statistic (half the explained
    sum of squares of the auxiliary regression on the scaled squared residuals), it
    does not assume normally distributed errors, which matters for fat-tailed returns.
    It is asymptotically chi-squared distributed with `k - 1` degrees of
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
    - Koenker, R. (1981). "A Note on Studentizing a Test for Heteroscedasticity."
    Journal of Econometrics, 17(1), 107-112.

    Referenced against:
    - statsmodels, het_breuschpagan: https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.het_breuschpagan.html

    Args:
        result (dict): A fitted OLS regression, from `regression_model.get_ols`.

    Returns:
        pd.Series: The (studentized) Breusch-Pagan LM statistic, its p-value, and whether
        homoskedasticity is rejected at the 5% level.

    Raises:
        ValueError: If the design has no regressor besides the constant.

    Notes:
    - Uses `result["design_matrix"]` (the design matrix actually used to fit `result`) as
    the regressors of the auxiliary regression, exactly as specified in the original
    Breusch-Pagan (1979) formulation.
    - The auxiliary regression always includes a constant, also when `result` was fitted
    without one: the LM statistic is built on the centered R-squared and its `k - 1`
    degrees of freedom count the non-constant regressors.
    """
    lm_statistic, p_value = _get_heteroskedasticity_lm(
        result["residuals"],
        _with_constant(result["design_matrix"]),
        "Breusch-Pagan",
        _is_perfect_fit(result),
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
    Calculate White's test: whether the residual variance of a fitted OLS regression changes
    with its regressors, their squares or their cross-products (heteroskedasticity of any form).

    Also known as: White's general test.

    A more general version of `get_breusch_pagan_test`: rather than regressing
    the squared residuals on only the original regressors, the auxiliary
    regression uses the full second-order polynomial expansion of the original
    (non-constant) regressors -- each regressor, its square, and every pairwise
    cross-product:

    - e_t^2 = a_0 + SUM_i(a_i * x_it) + SUM_i(b_i * x_it^2) + SUM_{i<j}(c_ij * x_it * x_jt) + v_t

    - LM = n * R^2_auxiliary

    Same construction as Breusch-Pagan, chi-squared distributed with degrees of
    freedom equal to the number of terms in the polynomial expansion (the rank of the
    auxiliary design minus the constant, so a term that duplicates another, such as
    the square of a 0/1 dummy, is counted once).

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

    Referenced against:
    - statsmodels, het_white: https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.het_white.html

    Args:
        result (dict): A fitted OLS regression, from `regression_model.get_ols`.

    Returns:
        pd.Series: White's LM statistic, its p-value, and whether homoskedasticity is
        rejected at the 5% level.

    Raises:
        ValueError: If the design has no regressor besides the constant.

    Notes:
    - The auxiliary regression always includes a constant and the regressors' levels,
    also when `result` was fitted without a constant: the products of every pair of columns
    of the design with a constant include the constant itself and each level.
    """
    design = _with_constant(result["design_matrix"])
    first, second = np.triu_indices(design.shape[1])
    lm_statistic, p_value = _get_heteroskedasticity_lm(
        result["residuals"],
        design[:, first] * design[:, second],
        "White's",
        _is_perfect_fit(result),
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
    Calculate the Durbin-Watson statistic: whether consecutive residuals of a fitted OLS
    regression are correlated (first-order autocorrelation).

    Also known as: DW statistic.

    - DW = SUM_{t=2}^{n}((e_t - e_(t-1))^2) / SUM_{t=1}^{n}(e_t^2)

    The statistic ranges from 0 to 4. A value near 2 indicates no first-order
    autocorrelation; values below 2 indicate positive autocorrelation
    (consecutive residuals tend to have the same sign -- common in time series
    with an omitted trend or lagged effect); values above 2 indicate negative
    autocorrelation (consecutive residuals tend to alternate in sign). Detecting
    autocorrelated residuals matters because, like heteroskedasticity, it leaves
    OLS coefficients unbiased but invalidates the reported standard errors --
    GLS with an AR(1) error covariance (`regression_model.get_gls`) is a standard remedy.

    For more information about the method, see the following paper:

    - Durbin, J., & Watson, G.S. (1950). "Testing for Serial Correlation in
    Least Squares Regression I." Biometrika, 37(3/4), 409-428.

    Referenced against:
    - statsmodels, durbin_watson: https://www.statsmodels.org/stable/generated/statsmodels.stats.stattools.durbin_watson.html

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
    residuals = result["residuals"]
    durbin_watson_statistic = (
        np.nan
        if _is_perfect_fit(result)
        else float(np.sum(np.diff(residuals) ** 2) / np.sum(residuals**2))
    )

    if np.isnan(durbin_watson_statistic):
        interpretation = "Not Defined (Perfect Fit)"
    elif durbin_watson_statistic < DURBIN_WATSON_LOWER_BAND:
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
    Calculate the Variance Inflation Factor (VIF) of each regressor: how much its correlation
    with the other regressors (multicollinearity) inflates the variance of its coefficient.

    Also known as: VIF.

    Multicollinearity -- one regressor being (near) linearly predictable from
    the others -- does not bias OLS coefficients but inflates their standard
    errors, making individual coefficients look statistically insignificant
    even when the regressors jointly explain the dependent variable well. VIF
    quantifies this per regressor by regressing it on all *other* regressors:

    - VIF_i = 1 / (1 - R^2_i)

    Where `R^2_i` is the (centered) R-squared of regressing regressor `i` on a
    constant and all other regressors in `x`. A VIF of 1 means no correlation with the other
    regressors; a VIF of `V` means that regressor's coefficient variance is `V`
    times larger than it would be under no multicollinearity.

    Unlike the other functions in this module, VIF takes the raw regressor
    matrix directly rather than a fitted regression result dict -- multicollinearity
    is a property of the regressors alone, independent of any particular
    dependent variable or fitted model.

    For more information about the method, see the following paper:

    - Marquardt, D.W. (1970). "Generalized Inverses, Ridge Regression, Biased Linear
    Estimation, and Nonlinear Estimation." Technometrics, 12(3), 591-612.

    Referenced against:
    - statsmodels, variance_inflation_factor: https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html

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
    - The auxiliary constant is always added, also when `x` already contains a
    constant column under some other name (an always-1 dummy, or a column with no
    variation over the sample), so every regressor is regressed on the same constant
    and the others. Such a constant column itself reports NaN: a regressor with no
    variance has no variance to inflate, so its VIF is undefined -- drop such a column
    before reading the result. A regressor that is an exact linear combination of the
    others has `R^2_i = 1` and a VIF of infinity.
    - Equals statsmodels' variance_inflation_factor on the design with a constant
    prepended.
    """
    if not isinstance(x, pd.DataFrame):
        raise TypeError(f"x must be a pd.DataFrame, received {type(x).__name__}.")

    regressor_names = [column for column in x.columns if column != "Intercept"]
    values = x[regressor_names].to_numpy(dtype=float)
    ones = np.ones((len(values), 1))

    vif_values = {}
    for index, regressor in enumerate(regressor_names):
        others = np.column_stack([ones, np.delete(values, index, axis=1)])
        r_squared = np.float64(fit_least_squares(values[:, index], others)["r_squared"])
        # A NumPy division, so an exact collinearity (R² = 1) gives inf rather than an error.  # noqa: E501
        with np.errstate(divide="ignore"):
            vif_values[regressor] = float(np.float64(1.0) / (1.0 - r_squared))

    return pd.Series(vif_values, name="VIF")


def get_ramsey_reset_test(result: dict, power: int = 3) -> pd.Series:
    """
    Calculate Ramsey's Regression Equation Specification Error Test (RESET): whether the
    linear functional form of a fitted OLS regression misses a non-linear term.

    Also known as: RESET test, Ramsey RESET.

    If the true relationship between `y` and `x` is non-linear (or a relevant
    non-linear transformation/omitted variable is missing), the fitted values
    of a linear model still carry information about that missing non-linearity.
    RESET tests for this indirectly: it augments the original regressors with
    powers of the fitted values (`fitted^2`, ..., `fitted^power`) and tests
    whether those added terms are jointly significant via a standard nested F-test
    comparing the restricted (original) and unrestricted (augmented) model:

    - y_t = x_t'beta + SUM_{j=2}^{power}(g_j * yhat_t^j) + u_t
    - F = W / q, with W the Wald statistic of g_2 = ... = g_power = 0
    - F = ((RSS_r - RSS_u) / q) / (RSS_u / (n - k_u)) for the classical covariance

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

    Referenced against:
    - statsmodels, linear_reset: https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.linear_reset.html

    Args:
        result (dict): A fitted OLS regression, from `regression_model.get_ols`.
        power (int, optional): The highest power of the fitted values to add,
        i.e. `fitted^2` through `fitted^power` are added. Defaults to 3.

    Returns:
        pd.Series: The RESET F-statistic, its p-value, and whether correct
        specification is rejected at the 5% level.

    Raises:
        ValueError: If `power` is less than 2, `result` was fit with a "cluster" or
        "HAC" covariance estimator (see Notes), or the augmented design is
        rank-deficient or has too few observations relative to its parameters.

    Notes:
    - The Wald test on the added power terms uses the SAME covariance estimator
    `result` itself was fit with, so a RESET on a heteroskedasticity-robust fit is
    itself heteroskedasticity-robust. Reporting the classical statistic for a robust
    fit would understate it (or overstate it) by whatever the robust correction is
    worth -- exactly the reason the fit requested a robust covariance in the first
    place.
    - Cluster-robust and HAC fits are rejected rather than silently downgraded to the
    classical statistic: a regression result does not keep the cluster labels or the
    HAC lag, so the augmented refit cannot reproduce those two estimators. Refit with an
    HC0 to HC3 covariance to run RESET on it.
    - The augmented model is always refit by ordinary least squares on the design
    matrix and the powers of the OLS fitted values, so for a `get_wls`/`get_gls` fit
    this is the RESET test of the OLS fit of the same `y` on the same `X`, not of a
    weighted one.
    - Equals statsmodels' linear_reset on the powers of the fitted values, in its F form.
    """
    if power < MINIMUM_RESET_POWER:
        raise ValueError(
            f"power must be at least {MINIMUM_RESET_POWER}, received {power}."
        )

    cov_type = result.get("cov_type", "nonrobust")
    if cov_type in UNSUPPORTED_RESET_COV_TYPES:
        raise ValueError(
            f"get_ramsey_reset_test cannot honour a {cov_type!r} covariance "
            "estimator -- the regression result does not keep the "
            "clusters/lags it needs. Refit with an 'HC0'..'HC3' covariance (or "
            "'nonrobust') to run RESET on it."
        )

    # y is recoverable as fitted_values + residuals, by construction.
    target = result["fitted_values"] + result["residuals"]

    design = result["design_matrix"]
    fitted = fit_least_squares(target, design)["fitted_values"]
    augmented = np.column_stack(
        [design, *[fitted**exponent for exponent in range(2, power + 1)]]
    )

    try:
        regression_model._validate_design(augmented, target)
    except ValueError as error:
        raise ValueError(
            "Could not fit the RESET-augmented regression -- the augmented design "
            "may be rank-deficient or have too few observations relative to its "
            "parameters."
        ) from error

    augmented_fit = fit_least_squares(target, augmented, cov_type=cov_type)
    added_terms = power - 1
    restriction_matrix = np.eye(added_terms, augmented.shape[1], k=design.shape[1])
    wald_statistic, restrictions = get_wald_statistic(
        augmented_fit["coefficients"],
        augmented_fit["covariance_matrix"],
        restriction_matrix,
    )

    f_statistic = wald_statistic / restrictions
    p_value = float(
        stats.f.sf(f_statistic, restrictions, augmented_fit["degrees_of_freedom"])
    )

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
    Calculate the Chow test: whether the coefficients of a regression differ before and
    after a known break point (a structural break).

    Also known as: Chow breakpoint test.

    Splits the sample into a "before" and "after" sub-sample at `break_index`,
    fits OLS separately on each (via `regression_model.get_ols`), and F-tests whether allowing the coefficients to
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
    x_values, _ = regression_model._to_design_matrix(x, add_constant=False)
    y_values = regression_model._to_target_vector(y)

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
