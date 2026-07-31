"""Hypothesis Testing Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.weightstats import ttest_ind

from financetoolkit.econometrics.regression_model import (
    RegressionResult,
    _to_design_matrix,
    _to_target_vector,
    get_ols,
)

# pylint: disable=too-many-locals

SIGNIFICANCE_LEVEL = 0.05


def get_two_sample_t_test(
    sample_a: pd.Series | np.ndarray,
    sample_b: pd.Series | np.ndarray,
    equal_variance: bool = False,
) -> pd.Series:
    """
    Calculate a two-sample t-test for a difference in means between `sample_a` and
    `sample_b`, via `statsmodels.stats.weightstats.ttest_ind`.

    Also known as: independent samples t-test, Welch's t-test (when `equal_variance`
    is False), Student's t-test (when `equal_variance` is True).

    Tests the null hypothesis that both samples are drawn from distributions with the
    same mean, `H0: mean(sample_a) = mean(sample_b)`, against the two-sided
    alternative that they differ. By default (`equal_variance=False`) this is Welch's
    t-test, which does not assume the two samples share a common variance:

    - t = (mean_a - mean_b) / sqrt(s_a^2 / n_a + s_b^2 / n_b)

    With degrees of freedom given by the Welch-Satterthwaite equation. Setting
    `equal_variance=True` instead pools the two sample variances into a single
    estimate (Student's original two-sample t-test), which is more powerful but only
    valid when the two samples genuinely share the same variance -- Welch's version is
    the safer default since it reduces to Student's test when variances happen to be
    equal, but remains valid when they are not.

    For more information about the method, see the following papers:

    - Student (1908). "The Probable Error of a Mean." Biometrika, 6(1), 1-25.
    - Welch, B.L. (1947). "The Generalization of 'Student's' Problem when Several
    Different Population Variances are Involved." Biometrika, 34(1/2), 28-35.

    Args:
        sample_a (pd.Series | np.ndarray): The first sample.
        sample_b (pd.Series | np.ndarray): The second sample.
        equal_variance (bool, optional): Whether to assume the two samples share a
        common variance (Student's pooled t-test) rather than allowing for unequal
        variances (Welch's t-test). Defaults to False.

    Returns:
        pd.Series: The t-statistic, degrees of freedom, its p-value, and the two
        sample means.

    Raises:
        TypeError: If `sample_a` or `sample_b` is not one of the accepted types.
    """
    sample_a_values = _to_target_vector(sample_a)
    sample_b_values = _to_target_vector(sample_b)
    sample_a_values = sample_a_values[~np.isnan(sample_a_values)]
    sample_b_values = sample_b_values[~np.isnan(sample_b_values)]

    t_statistic, p_value, degrees_of_freedom = ttest_ind(
        sample_a_values,
        sample_b_values,
        usevar="pooled" if equal_variance else "unequal",
    )

    return pd.Series(
        {
            "T-Statistic": float(t_statistic),
            "Degrees of Freedom": float(degrees_of_freedom),
            "P-Value": float(p_value),
            "Mean A": float(np.mean(sample_a_values)),
            "Mean B": float(np.mean(sample_b_values)),
        }
    )


def _validate_nested_models(
    result_restricted: RegressionResult, result_unrestricted: RegressionResult
) -> int:
    """
    Shared nesting validation for `get_f_test` and `get_likelihood_ratio_test` --
    both compare a "restricted" (fewer regressors) against an "unrestricted" (more
    regressors) model fit on the same dependent variable, and share the exact same
    requirements for that comparison to be meaningful. Returns `q`, the number of
    restrictions (the difference in parameter counts).
    """
    if result_unrestricted.n_parameters <= result_restricted.n_parameters:
        raise ValueError(
            "result_unrestricted must have strictly more parameters than "
            f"result_restricted, received {result_unrestricted.n_parameters} and "
            f"{result_restricted.n_parameters} respectively -- are these the right "
            "way around?"
        )
    if result_unrestricted.n_observations != result_restricted.n_observations:
        raise ValueError(
            "result_restricted and result_unrestricted must be fit on the same "
            f"number of observations, received {result_restricted.n_observations} "
            f"and {result_unrestricted.n_observations} respectively -- are both "
            "models fit on the same y?"
        )

    return result_unrestricted.n_parameters - result_restricted.n_parameters


def _refit(result: RegressionResult):
    """
    Refits `result` (a `RegressionResult` produced by `regression_model.get_ols`/
    `get_wls`/`get_gls`) as a `statsmodels.api.OLS` results object on the exact same
    (post-transform) design matrix and dependent variable, so the nested-model
    comparison and restriction-testing functions below can delegate to `statsmodels`'
    own `compare_f_test`/`compare_lr_test`/`wald_test` machinery rather than
    reimplementing the RSS-ratio/quadratic-form formulas by hand. `y` is not stored on
    `RegressionResult` directly, but is exactly recoverable from
    `fitted_values + residuals`.
    """
    target = result.fitted_values + result.residuals
    return sm.OLS(target, result.design_matrix).fit()


def get_f_test(
    result_restricted: RegressionResult, result_unrestricted: RegressionResult
) -> pd.Series:
    """
    Calculate a nested-model F-test for the joint significance of the extra
    regressors in `result_unrestricted` relative to `result_restricted`, via
    `statsmodels`' `RegressionResults.compare_f_test`.

    Also known as: nested F-test, restricted vs. unrestricted F-test, partial F-test.

    Given a "restricted" model (fewer regressors) and an "unrestricted" model (the
    same regressors plus `q` more, nesting the restricted one), both fit via
    `regression_model.get_ols`/`get_wls`/`get_gls` on the SAME dependent variable,
    tests the null hypothesis that the `q` extra regressors are all jointly zero --
    i.e. that the restricted model is not missing anything -- by comparing how much
    the Residual Sum of Squares (RSS) drops when they are added:

    - F = ((RSS_restricted - RSS_unrestricted) / q) / (RSS_unrestricted / (n - k_unrestricted))

    Where `q = k_unrestricted - k_restricted` is the number of restrictions. Under the
    null, `F` follows an F(q, n - k_unrestricted) distribution. A significant (low
    p-value) result means the extra regressors meaningfully improve the fit and
    should not be dropped. This is the same nested-F construction used internally by
    `causality_model.get_granger_causality`, exposed here as a general-purpose
    primitive for any two nested `RegressionResult`s.

    Args:
        result_restricted (RegressionResult): The fitted restricted model (fewer
        regressors).
        result_unrestricted (RegressionResult): The fitted unrestricted model (more
        regressors, nesting `result_restricted`).

    Returns:
        pd.Series: The F-statistic, the numerator (`q`) and denominator
        (`n - k_unrestricted`) degrees of freedom, its p-value, and whether the
        restrictions are rejected (i.e. the extra regressors are jointly
        significant) at the 5% level.

    Raises:
        ValueError: If `result_unrestricted` does not have strictly more parameters
        than `result_restricted`, or if the two were not fit on the same number of
        observations.
    """
    q = _validate_nested_models(result_restricted, result_unrestricted)

    sm_restricted = _refit(result_restricted)
    sm_unrestricted = _refit(result_unrestricted)

    f_statistic, p_value, _ = sm_unrestricted.compare_f_test(sm_restricted)
    degrees_of_freedom_denominator = result_unrestricted.degrees_of_freedom

    return pd.Series(
        {
            "F-Statistic": float(max(f_statistic, 0.0)),
            "Df Numerator": q,
            "Df Denominator": degrees_of_freedom_denominator,
            "P-Value": float(p_value),
            "Reject Restrictions (5%)": bool(p_value < SIGNIFICANCE_LEVEL),
        }
    )


def get_wald_test(
    result: RegressionResult,
    restriction_matrix: np.ndarray,
    restriction_values: np.ndarray | None = None,
) -> pd.Series:
    """
    Calculate a Wald test of `q` general linear restriction(s) on the coefficients of
    a single fitted `RegressionResult`, via `statsmodels`' `RegressionResults.wald_test`.

    Also known as: Wald chi-squared test.

    Tests the null hypothesis `H0: R @ beta = r`, where `R` (`restriction_matrix`) is
    a `(q, k)` matrix encoding `q` linear restrictions on the `k` coefficients and `r`
    (`restriction_values`) is a length-`q` vector of hypothesized values (zero by
    default). This generalizes the single-coefficient t-test reported on every
    `RegressionResult` to arbitrary linear combinations of coefficients -- e.g.
    testing that two coefficients are jointly zero (`R` with two one-hot rows),
    that one coefficient equals a specific non-zero value (`R` a single one-hot row,
    `r` that value), or that two coefficients are equal to each other
    (`R = [[0, 1, -1, 0, ...]]`, `r = 0`). The test statistic is:

    - W = (R @ beta_hat - r)' @ (R @ Cov(beta_hat) @ R')^-1 @ (R @ beta_hat - r)

    Which is chi-squared(q) distributed asymptotically. `W / q` also has a natural
    small-sample F(q, n - k) interpretation and is reported alongside the chi-squared
    version -- for a single restriction (`q = 1`) this F-statistic is an exact
    algebraic identity with the corresponding coefficient's own squared t-statistic
    (`F = t^2`), since both use the same `Cov(beta_hat)`.

    For more information about the method, see the following paper:

    - Wald, A. (1943). "Tests of Statistical Hypotheses Concerning Several Parameters
    When the Number of Observations is Large." Transactions of the American
    Mathematical Society, 54(3), 426-482.

    Args:
        result (RegressionResult): The fitted model whose coefficients are being
        tested.
        restriction_matrix (np.ndarray): The `(q, k)` restriction matrix `R`, `k`
        matching `result.n_parameters`.
        restriction_values (np.ndarray | None, optional): The length-`q` vector `r`
        of hypothesized values. Defaults to None, i.e. `r = 0` (the restrictions are
        all "equals zero").

    Returns:
        pd.Series: The Wald (chi-squared) statistic and its p-value, the small-sample
        F-statistic and its p-value, the number of restrictions `q`, and whether the
        restrictions are rejected at the 5% level (based on the chi-squared p-value).

    Raises:
        ValueError: If `restriction_matrix`'s number of columns does not match
        `result.n_parameters`, or `restriction_values`'s length does not match
        `restriction_matrix`'s number of rows.
    """
    restriction_matrix = np.atleast_2d(np.asarray(restriction_matrix, dtype=float))
    n_restrictions, n_coefficients = restriction_matrix.shape

    if n_coefficients != result.n_parameters:
        raise ValueError(
            f"restriction_matrix has {n_coefficients} columns but the fitted model "
            f"has {result.n_parameters} coefficients -- these must match."
        )

    if restriction_values is None:
        restriction_values = np.zeros(n_restrictions)
    else:
        restriction_values = np.asarray(restriction_values, dtype=float).ravel()
        if restriction_values.shape[0] != n_restrictions:
            raise ValueError(
                f"restriction_values has {restriction_values.shape[0]} entries but "
                f"restriction_matrix has {n_restrictions} rows -- these must match."
            )

    sm_result = _refit(result)
    wald = sm_result.wald_test(
        (restriction_matrix, restriction_values), scalar=True, use_f=False
    )
    wald_f = sm_result.wald_test(
        (restriction_matrix, restriction_values), scalar=True, use_f=True
    )

    wald_statistic = float(wald.statistic)
    chi2_p_value = float(wald.pvalue)
    f_statistic = float(wald_f.statistic)
    f_p_value = float(wald_f.pvalue)

    return pd.Series(
        {
            "Wald Statistic (Chi2)": wald_statistic,
            "Chi2 P-Value": chi2_p_value,
            "F-Statistic": f_statistic,
            "F P-Value": f_p_value,
            "Restrictions (q)": n_restrictions,
            "Reject Restrictions (5%)": bool(chi2_p_value < SIGNIFICANCE_LEVEL),
        }
    )


def get_likelihood_ratio_test(
    result_restricted: RegressionResult, result_unrestricted: RegressionResult
) -> pd.Series:
    """
    Calculate a nested-model Likelihood Ratio (LR) test, the Maximum Likelihood
    analogue of `get_f_test`, via `statsmodels`' `RegressionResults.compare_lr_test`.

    Also known as: LR test, Wilks' likelihood ratio test.

    Given the same "restricted vs. unrestricted" pair of nested OLS-family models as
    `get_f_test`, this instead compares them via the ratio of their (Gaussian)
    likelihoods rather than an exact F-distribution. For linear-Gaussian models the LR
    statistic reduces to a simple function of each model's Residual Sum of Squares
    (RSS):

    - LR = n * ln(RSS_restricted / RSS_unrestricted)

    Which is asymptotically chi-squared(q) distributed under the null that the `q`
    extra regressors in the unrestricted model are all jointly zero, where
    `q = k_unrestricted - k_restricted`. Unlike the F-test (which is exact in finite
    samples under the classical assumptions), the LR test's chi-squared distribution
    is only an asymptotic (large-`n`) approximation -- the two tests should agree
    for large samples and can diverge somewhat for small ones.

    For more information about the method, see the following paper:

    - Wilks, S.S. (1938). "The Large-Sample Distribution of the Likelihood Ratio for
    Testing Composite Hypotheses." Annals of Mathematical Statistics, 9(1), 60-62.

    Args:
        result_restricted (RegressionResult): The fitted restricted model (fewer
        regressors).
        result_unrestricted (RegressionResult): The fitted unrestricted model (more
        regressors, nesting `result_restricted`).

    Returns:
        pd.Series: The LR statistic, its degrees of freedom (`q`), its p-value, and
        whether the restrictions are rejected at the 5% level.

    Raises:
        ValueError: If `result_unrestricted` does not have strictly more parameters
        than `result_restricted`, or if the two were not fit on the same number of
        observations.
    """
    q = _validate_nested_models(result_restricted, result_unrestricted)

    sm_restricted = _refit(result_restricted)
    sm_unrestricted = _refit(result_unrestricted)

    lr_statistic, p_value, _ = sm_unrestricted.compare_lr_test(sm_restricted)

    return pd.Series(
        {
            "LR Statistic": float(max(lr_statistic, 0.0)),
            "Degrees of Freedom": q,
            "P-Value": float(p_value),
            "Reject Restrictions (5%)": bool(p_value < SIGNIFICANCE_LEVEL),
        }
    )


def _combine_regressors(
    *parts: pd.DataFrame | pd.Series | np.ndarray | None,
) -> tuple[np.ndarray, list[str]]:
    """
    Column-stacks any number of regressor blocks (each a `pd.DataFrame`, `pd.Series`,
    `np.ndarray`, or `None`, which is skipped) into a single design matrix plus a
    combined list of feature names, reusing `regression_model._to_design_matrix` for
    each individual block so every accepted input type is handled consistently with
    the rest of this module.
    """
    values_list = []
    names_list: list[str] = []

    for part in parts:
        if part is None:
            continue
        values, names = _to_design_matrix(part, add_constant=False)
        values_list.append(values)
        names_list.extend(names)

    if not values_list:
        raise ValueError("At least one regressor block must be provided.")

    return np.column_stack(values_list), names_list


def get_hausman_wu_test(
    y: pd.Series | np.ndarray,
    x_suspect: pd.Series | np.ndarray,
    instruments: pd.DataFrame | pd.Series | np.ndarray,
    x_other: pd.DataFrame | pd.Series | np.ndarray | None = None,
) -> pd.Series:
    """
    Calculate a regression-based Hausman-Wu test for the endogeneity of `x_suspect`
    in a regression of `y` on `x_suspect` (and, optionally, `x_other`).

    Also known as: Hausman test, Durbin-Wu-Hausman test, regression test for
    endogeneity.

    OLS requires regressors to be uncorrelated with the error term
    (`Cov(x, e) = 0`); when that fails (e.g. due to an omitted confounder, reverse
    causality, or measurement error) `x_suspect` is "endogenous" and OLS is biased,
    calling for an instrumental-variables approach instead (see
    `causal_inference_model.get_iv_2sls`, whose `linearmodels.iv.IV2SLS` results
    object exposes an equivalent `.wu_hausman()` test directly on the fitted IV model
    -- this function instead builds the classic two-regression version by hand from
    `regression_model.get_ols`, useful when you want to test endogeneity before
    committing to a full IV fit). This test operationalizes the Durbin-Wu-Hausman
    idea as two auxiliary OLS regressions:

    1. Regress `x_suspect` on `instruments` (and `x_other`, if given) and keep the
    residuals `v_hat` -- the part of `x_suspect` NOT explained by the instruments.
    2. Regress `y` on `x_suspect`, `v_hat` (and `x_other`), and test whether `v_hat`'s
    coefficient is significantly different from zero.

    If `x_suspect` were truly exogenous, `v_hat` (measurement noise/instrument-driven
    variation) would carry no additional information about `y` once `x_suspect`
    itself is already included, so its coefficient should be statistically zero. A
    significant coefficient on `v_hat` means the part of `x_suspect` that is
    correlated with the instruments' "exogenous variation" behaves differently from
    the part that isn't -- the signature of `x_suspect` being correlated with the
    structural error term, i.e. genuinely endogenous.

    For more information about the method, see the following papers:

    - Wu, D-M. (1973). "Alternative Tests of Independence Between Stochastic
    Regressors and Disturbances." Econometrica, 41(4), 733-750.
    - Hausman, J.A. (1978). "Specification Tests in Econometrics." Econometrica,
    46(6), 1251-1271.

    Args:
        y (pd.Series | np.ndarray): The dependent variable, shape `(n,)`.
        x_suspect (pd.Series | np.ndarray): The (possibly endogenous) regressor being
        tested, shape `(n,)`.
        instruments (pd.DataFrame | pd.Series | np.ndarray): One or more instruments
        for `x_suspect` -- variables correlated with `x_suspect` but not, under the
        maintained identifying assumption, directly with `y`'s error term.
        x_other (pd.DataFrame | pd.Series | np.ndarray | None, optional): Any other
        (assumed exogenous) regressors to include in both auxiliary regressions.
        Defaults to None.

    Returns:
        pd.Series: The coefficient on `v_hat`, its t-statistic and p-value (from the
        second-stage regression), and whether `x_suspect` is flagged as endogenous at
        the 5% level.

    Raises:
        TypeError: If any input is not one of the accepted types.
        ValueError: If there are not more observations than parameters in either
        auxiliary regression, or `instruments`/`x_other` is empty.
    """
    y_values = _to_target_vector(y)
    x_suspect_matrix, x_suspect_names = _to_design_matrix(x_suspect, add_constant=False)
    x_suspect_values = x_suspect_matrix.ravel()

    stage_one_design, _ = _combine_regressors(instruments, x_other)
    stage_one = get_ols(x_suspect_values, stage_one_design, add_constant=True)
    v_hat = stage_one.residuals

    stage_two_parts = [x_suspect_matrix, v_hat.reshape(-1, 1)]
    stage_two_names = [x_suspect_names[0], "Residual"]

    if x_other is not None:
        other_values, other_names = _to_design_matrix(x_other, add_constant=False)
        stage_two_parts.append(other_values)
        stage_two_names.extend(other_names)

    stage_two_design = pd.DataFrame(
        np.column_stack(stage_two_parts), columns=stage_two_names
    )
    stage_two = get_ols(y_values, stage_two_design, add_constant=True)

    residual_index = stage_two.feature_names.index("Residual")
    v_hat_coefficient = float(stage_two.coefficients[residual_index])
    t_statistic = float(stage_two.t_statistics[residual_index])
    p_value = float(stage_two.p_values[residual_index])

    return pd.Series(
        {
            "V-Hat Coefficient": v_hat_coefficient,
            "T-Statistic": t_statistic,
            "Degrees of Freedom": stage_two.degrees_of_freedom,
            "P-Value": p_value,
            "Endogenous (5%)": bool(p_value < SIGNIFICANCE_LEVEL),
        }
    )
