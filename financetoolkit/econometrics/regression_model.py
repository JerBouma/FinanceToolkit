"""Regression Model"""

__docformat__ = "google"

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import IterationLimitWarning

# pylint: disable=too-many-instance-attributes,too-many-locals,too-many-arguments

TWO_DIMENSIONAL = 2
COV_TYPES = ("nonrobust", "HC0", "HC1", "HC2", "HC3", "cluster", "HAC")

# Cluster-robust standard errors require at least 2 distinct clusters.
MINIMUM_CLUSTERS = 2

# Duplicated rows in a resample need more iterations than statsmodels' default 1000.
BOOTSTRAP_MAX_ITERATIONS = 10_000

# A standard deviation across bootstrap replicates needs more than one of them.
MINIMUM_CONVERGED_REPLICATES = 2


def _to_design_matrix(
    x: pd.DataFrame | pd.Series | np.ndarray, add_constant: bool
) -> tuple[np.ndarray, list[str]]:
    """
    Normalizes `x` into a 2D numpy design matrix plus a list of human-readable
    column names, optionally prepending a constant ("Intercept") column of ones.
    Shared by every regression function in this module so all of them accept the
    same flexible input types (a single `pd.Series`, a `pd.DataFrame` of multiple
    regressors, or a raw `np.ndarray`) and label their output consistently before
    handing it to `statsmodels`.
    """
    if isinstance(x, pd.DataFrame):
        feature_names = list(x.columns)
        values = x.to_numpy(dtype=float)
    elif isinstance(x, pd.Series):
        feature_names = [x.name if x.name is not None else "X1"]
        values = x.to_numpy(dtype=float).reshape(-1, 1)
    elif isinstance(x, np.ndarray):
        values = x if x.ndim == TWO_DIMENSIONAL else x.reshape(-1, 1)
        feature_names = [f"X{i + 1}" for i in range(values.shape[1])]
    else:
        raise TypeError(
            f"x must be a pd.DataFrame, pd.Series or np.ndarray, received {type(x).__name__}."
        )

    if add_constant:
        values = np.column_stack([np.ones(values.shape[0]), values])
        feature_names = ["Intercept", *feature_names]

    return values, feature_names


def _to_target_vector(y: pd.Series | np.ndarray) -> np.ndarray:
    if isinstance(y, pd.Series):
        return y.to_numpy(dtype=float)
    if isinstance(y, np.ndarray):
        return y.astype(float)
    raise TypeError(
        f"y must be a pd.Series or np.ndarray, received {type(y).__name__}."
    )


def _validate_design(x: np.ndarray, y: np.ndarray) -> None:
    n, k = x.shape
    if n <= k:
        raise ValueError(
            f"Not enough observations ({n}) to estimate {k} parameters -- need "
            f"strictly more observations than parameters."
        )
    if np.linalg.matrix_rank(x) < k:
        raise ValueError(
            "The design matrix is rank-deficient (perfectly collinear regressors) -- "
            "OLS coefficients are not uniquely identified."
        )
    if len(y) != n:
        raise ValueError(
            f"y and x must have the same number of observations, received {len(y)} "
            f"and {n} respectively."
        )


def _cov_type_and_kwds(
    cov_type: str, clusters: np.ndarray | None, maxlags: int | None = None
) -> tuple[str, dict]:
    if cov_type not in COV_TYPES:
        raise ValueError(f"cov_type must be one of {COV_TYPES}, received {cov_type!r}.")

    if cov_type == "HAC":
        if maxlags is None:
            raise ValueError("maxlags must be provided when cov_type='HAC'.")
        if maxlags < 0:
            raise ValueError(f"maxlags must be non-negative, received {maxlags}.")
        return "HAC", {"maxlags": maxlags}

    if cov_type != "cluster":
        return cov_type, {}

    if clusters is None:
        raise ValueError("clusters must be provided when cov_type='cluster'.")

    unique_clusters = np.unique(clusters)
    if len(unique_clusters) < MINIMUM_CLUSTERS:
        raise ValueError(
            "cluster-robust standard errors require at least 2 distinct clusters."
        )

    return "cluster", {"groups": clusters}


def regression_summary_table(result: dict) -> pd.DataFrame:
    """
    Builds the standard coefficient table (Coefficient, Std. Error, t-Statistic,
    P-Value) from a `get_ols`/`get_wls`/`get_gls` result dict, indexed by
    `result["feature_names"]`.

    Args:
        result (dict): A fitted regression result dict, as returned by `get_ols`,
        `get_wls` or `get_gls`.

    Returns:
        pd.DataFrame: The coefficient table.
    """
    return pd.DataFrame(
        {
            "Coefficient": result["coefficients"],
            "Std. Error": result["standard_errors"],
            "t-Statistic": result["t_statistics"],
            "P-Value": result["p_values"],
        },
        index=result["feature_names"],
    )


def _coefficient_of_determination(sm_result) -> tuple[float, float]:
    """
    Returns R-squared and adjusted R-squared, or NaN where they are not defined.

    R-squared is the share of the total sum of squares that the model explains, so
    it has no meaning when that total is zero -- an outcome with no variation left
    to explain. `statsmodels` computes it as a bare division and lets NumPy produce
    a NaN through a `0/0`, which is the right value but arrives with a runtime
    warning attached. The case is real rather than pathological: a cross-sectional
    Fama-MacBeth regression through the origin hits it whenever every asset in one
    period returned exactly zero.

    Which total applies depends on the model. With an intercept it is the total sum
    of squares about the mean; a regression through the origin instead compares
    against the uncentered total, since there is no fitted mean to deviate from.

    Args:
        sm_result: A fitted `statsmodels` OLS/WLS/GLS results object.

    Returns:
        tuple[float, float]: `(r_squared, adjusted_r_squared)`, both NaN when the
        total sum of squares is zero or the residual degrees of freedom are
        exhausted.
    """
    total_sum_of_squares = (
        sm_result.centered_tss if sm_result.k_constant else sm_result.uncentered_tss
    )

    if not np.isfinite(total_sum_of_squares) or total_sum_of_squares == 0:
        return float("nan"), float("nan")

    r_squared = float(sm_result.rsquared)

    if sm_result.df_resid <= 0:
        return r_squared, float("nan")

    return r_squared, float(sm_result.rsquared_adj)


def _from_statsmodels_ols(
    sm_result, feature_names: list[str], design_matrix: np.ndarray, cov_type: str
) -> dict:
    """
    Translates a fitted `statsmodels` OLS/WLS/GLS results object into the shared
    regression result dict returned by `get_ols`/`get_wls`/`get_gls` -- one shared,
    reusable shape since every OLS-family estimator produces the same set of
    quantities (coefficients, their standard errors, residuals, fit statistics), just
    via a different weighting of the observations before `statsmodels` solves the
    normal equations. This is deliberately NOT the final, user-facing return value in
    most cases -- controller methods extract the specific piece(s) a caller needs
    (e.g. just the coefficient table, via `regression_summary_table`) into a plain
    `pd.DataFrame`/`pd.Series`, matching the rest of this codebase's convention -- but
    is the shared internal contract that `panel_data_model`, `specification_tests_model`,
    `hypothesis_testing_model` and `causal_inference_model` build their diagnostics and
    tests on top of, so that a Breusch-Pagan test, a VIF, or a Wald test all operate on
    an identical notion of "a fitted linear regression" regardless of which of the
    three estimators produced it.

    Keys:
        coefficients (np.ndarray): The estimated coefficients, shape `(k,)`.
        standard_errors (np.ndarray): The standard errors of the coefficients, shape `(k,)`.
        t_statistics (np.ndarray): The coefficients divided by their standard errors.
        p_values (np.ndarray): Two-sided p-values for those ratios. `cov_type="nonrobust"`
        reads them off the Student-T(n - k) distribution; every robust estimator
        ("HC0".."HC3", "cluster", "HAC") is an asymptotic result, so `statsmodels`
        switches to the standard normal for those and `t_statistics` are then Wald
        z-statistics rather than exact finite-sample t-statistics.
        residuals (np.ndarray): The residuals `y - X @ coefficients`, in `y`'s
        original (unweighted) scale for every estimator including WLS/GLS --
        `statsmodels` reserves its (internal, whitened-scale) `wresid` attribute for
        the weighted residuals actually used to solve the normal equations, which is
        NOT what is stored here. Shape `(n,)`.
        fitted_values (np.ndarray): The fitted values `X @ coefficients`, shape `(n,)`.
        covariance_matrix (np.ndarray): The estimated covariance matrix of the
        coefficients, shape `(k, k)`.
        r_squared (float): The coefficient of determination. NaN when there is no
        variation in the outcome for the model to explain, which leaves it undefined.
        adjusted_r_squared (float): R-squared adjusted for the number of regressors.
        NaN under the same condition, and also once the residual degrees of freedom
        are exhausted (`n == k`), which leaves the adjustment itself undefined.
        residual_variance (float): The estimated residual variance,
        `sigma^2 = SSR / (n - k)` for OLS -- for WLS/GLS this is the WEIGHTED
        residual sum of squares divided by `(n - k)` (`statsmodels`' `mse_resid`),
        consistent with `standard_errors`/`covariance_matrix` also being computed
        on the weighted scale.
        degrees_of_freedom (int): `n - k`, the residual degrees of freedom.
        n_observations (int): The number of observations used, `n`.
        n_parameters (int): The number of estimated parameters (including the
        intercept, if any), `k`.
        feature_names (list[str]): The name of each coefficient, in order.
        design_matrix (np.ndarray): The design matrix actually used in estimation
        (post constant-augmentation, pre any WLS/GLS whitening), shape `(n, k)`.
        cov_type (str): Which covariance estimator was used to compute
        `standard_errors`/`t_statistics`/`p_values`/`covariance_matrix` -- one of
        "nonrobust" (classical, assumes homoskedastic errors), "HC0"/"HC1"/"HC2"/"HC3"
        (heteroskedasticity-robust) or "cluster" (cluster-robust). See `get_ols`'s
        `cov_type` argument.
    """
    r_squared, adjusted_r_squared = _coefficient_of_determination(sm_result)

    return {
        "coefficients": np.asarray(sm_result.params),
        "standard_errors": np.asarray(sm_result.bse),
        "t_statistics": np.asarray(sm_result.tvalues),
        "p_values": np.asarray(sm_result.pvalues),
        "residuals": np.asarray(sm_result.resid),
        "fitted_values": np.asarray(sm_result.fittedvalues),
        "covariance_matrix": np.asarray(sm_result.cov_params()),
        "r_squared": r_squared,
        "adjusted_r_squared": adjusted_r_squared,
        "residual_variance": float(sm_result.mse_resid),
        "degrees_of_freedom": int(sm_result.df_resid),
        "n_observations": int(sm_result.nobs),
        "n_parameters": len(feature_names),
        "feature_names": feature_names,
        "design_matrix": design_matrix,
        "cov_type": cov_type,
        # The fitted statsmodels object is carried along so that the nested-model tests
        # can delegate to its own machinery without refitting. Refitting from the design
        # matrix alone would silently drop the covariance type and the WLS/GLS weights.
        "statsmodels_result": sm_result,
    }


def get_ols(
    y: pd.Series | np.ndarray,
    x: pd.DataFrame | pd.Series | np.ndarray,
    add_constant: bool = True,
    cov_type: str = "nonrobust",
    clusters: pd.Series | np.ndarray | None = None,
    maxlags: int | None = None,
) -> dict:
    """
    Fit an Ordinary Least Squares (OLS) linear regression of `y` on `x`, via
    `statsmodels.api.OLS`.

    Also known as: linear regression, least squares regression.

    OLS estimates the coefficients `beta` that minimize the sum of squared
    residuals:

    - beta_hat = (X'X)^-1 X'y

    Under the classical assumptions (linearity, no perfect multicollinearity,
    homoskedastic and uncorrelated errors), this is the Best Linear Unbiased
    Estimator (BLUE, the Gauss-Markov theorem). The coefficients themselves stay
    unbiased even when errors are heteroskedastic -- what breaks is the classical
    ("nonrobust") standard errors, which is why `cov_type` exists: use
    `specification_tests_model.get_breusch_pagan_test`/`get_white_test` to check for
    heteroskedasticity first, then refit with `cov_type="HC1"` (or "HC3" for small/
    moderate samples) if it's present, rather than switching estimator entirely.

    Args:
        y (pd.Series | np.ndarray): The dependent variable, shape `(n,)`.
        x (pd.DataFrame | pd.Series | np.ndarray): The independent variable(s)/regressor(s).
        A `pd.DataFrame` with one column per regressor, a single `pd.Series`, or a raw
        `(n,)`/`(n, k)` array are all accepted.
        add_constant (bool, optional): Whether to prepend an intercept column of ones
        to `x`. Defaults to True.
        cov_type (str, optional): Which covariance estimator to use for the standard
        errors/t-statistics/p-values, one of:

        - "nonrobust": the classical `sigma^2 * (X'X)^-1` estimator, valid under
          homoskedastic errors. Defaults to this.
        - "HC0"/"HC1"/"HC2"/"HC3": heteroskedasticity-consistent ("robust"/"sandwich")
          estimators. HC1 matches the common `robust` option in other statistical
          software; HC3 is the more conservative, recommended choice for small/
          moderate samples.
        - "cluster": cluster-robust standard errors, valid under arbitrary
          within-cluster correlation (but requires independence ACROSS clusters).
          Requires `clusters`.
        - "HAC": Newey-West heteroskedasticity-and-autocorrelation-consistent
          standard errors -- valid under both heteroskedastic AND
          serially-correlated errors, the standard choice for time-series
          regressions in finance (HC-robust alone corrects for heteroskedasticity
          but assumes no autocorrelation). Requires `maxlags`.

        clusters (pd.Series | np.ndarray | None, optional): The cluster label for each
        observation, required (and only used) when `cov_type="cluster"`. Defaults to
        None.
        maxlags (int | None, optional): The maximum lag to include when estimating the
        HAC (Newey-West) covariance matrix, required (and only used) when
        `cov_type="HAC"`. A common rule of thumb is `floor(4 * (n / 100)^(2/9))`
        (Newey & West, 1994). Defaults to None.

    Returns:
        dict: The fitted coefficients, their standard errors/t-statistics/
        p-values, residuals, fitted values, R-squared and related fit statistics -- see
        `_from_statsmodels_ols` for the full key list. Call `regression_summary_table`
        for a coefficient table.

    Raises:
        TypeError: If `y` or `x` is not one of the accepted types.
        ValueError: If there are not more observations than parameters, `x` is
        rank-deficient (perfectly collinear regressors), `cov_type` is not a
        recognized value, `cov_type="cluster"` without `clusters` (or with fewer
        than 2 distinct clusters), or `cov_type="HAC"` without a non-negative
        `maxlags`.

    Notes:
        Reference: Newey, W.K. & West, K.D. (1987). "A Simple, Positive
        Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance
        Matrix." Econometrica, 55(3), 703-708.

    As an example:

    ```python
    import numpy as np
    import pandas as pd
    from financetoolkit.econometrics import regression_model

    rng = np.random.default_rng(42)
    x = pd.Series(rng.standard_normal(100), name="Market Return")
    y = 0.02 + 1.2 * x + rng.standard_normal(100) * 0.05

    result = regression_model.get_ols(y, x)
    print(regression_model.regression_summary_table(result).round(4))
    ```

    Which returns:

    |               |   Coefficient |   Std. Error |   t-Statistic |   P-Value |
    |:--------------|---------------:|--------------:|---------------:|-----------:|
    | Intercept     |         0.0198 |        0.0049 |         4.0254 |     0.0001 |
    | Market Return |         1.2060 |        0.0063 |       190.1749 |     0.0000 |
    """
    y_values = _to_target_vector(y)
    x_values, feature_names = _to_design_matrix(x, add_constant)
    cluster_values = (
        clusters.to_numpy() if isinstance(clusters, pd.Series) else clusters
    )

    _validate_design(x_values, y_values)
    sm_cov_type, cov_kwds = _cov_type_and_kwds(cov_type, cluster_values, maxlags)

    sm_result = sm.OLS(y_values, x_values).fit(cov_type=sm_cov_type, cov_kwds=cov_kwds)

    return _from_statsmodels_ols(sm_result, feature_names, x_values, cov_type)


def get_wls(
    y: pd.Series | np.ndarray,
    x: pd.DataFrame | pd.Series | np.ndarray,
    weights: pd.Series | np.ndarray,
    add_constant: bool = True,
    cov_type: str = "nonrobust",
    clusters: pd.Series | np.ndarray | None = None,
    maxlags: int | None = None,
) -> dict:
    """
    Fit a Weighted Least Squares (WLS) regression of `y` on `x`, via
    `statsmodels.api.WLS`.

    Also known as: weighted regression.

    When errors are heteroskedastic with a known (or estimable) variance structure
    `Var(e_i) = sigma^2 / w_i`, OLS remains unbiased but is no longer efficient (BLUE) --
    WLS restores efficiency by minimizing the weighted sum of squared residuals, giving
    less weight to noisier observations:

    - beta_hat = (X'WX)^-1 X'Wy,  W = diag(weights)

    Args:
        y (pd.Series | np.ndarray): The dependent variable, shape `(n,)`.
        x (pd.DataFrame | pd.Series | np.ndarray): The independent variable(s)/regressor(s).
        weights (pd.Series | np.ndarray): The (positive) weight of each observation,
        shape `(n,)`. Larger weights pull the fit more towards that observation --
        e.g. `1 / variance` if each observation's error variance is known or estimated.
        add_constant (bool, optional): Whether to prepend an intercept column of ones
        to `x`. Defaults to True.
        cov_type (str, optional): Which covariance estimator to use -- see `get_ols`'s
        `cov_type` for the full list of options. Defaults to "nonrobust".
        clusters (pd.Series | np.ndarray | None, optional): The cluster label for each
        observation, required when `cov_type="cluster"`. Defaults to None.
        maxlags (int | None, optional): The maximum lag to include when estimating the
        HAC (Newey-West) covariance matrix, required when `cov_type="HAC"`. See
        `get_ols`'s `maxlags` for the rule-of-thumb formula. Defaults to None.

    Returns:
        dict: The fitted coefficients, their standard errors/t-statistics/
        p-values, residuals, fitted values, R-squared and related fit statistics -- see
        `_from_statsmodels_ols` for the full key list. Call `regression_summary_table`
        for a coefficient table.

    Raises:
        TypeError: If `y`, `x` or `weights` is not one of the accepted types.
        ValueError: If any weight is not strictly positive, there are not more
        observations than parameters, or `cov_type` is invalid.
    """
    y_values = _to_target_vector(y)
    x_values, feature_names = _to_design_matrix(x, add_constant)
    weight_values = (
        weights.to_numpy(dtype=float)
        if isinstance(weights, pd.Series)
        else np.asarray(weights, dtype=float)
    )
    cluster_values = (
        clusters.to_numpy() if isinstance(clusters, pd.Series) else clusters
    )

    if np.any(weight_values <= 0):
        raise ValueError("All weights must be strictly positive.")

    _validate_design(x_values, y_values)
    sm_cov_type, cov_kwds = _cov_type_and_kwds(cov_type, cluster_values, maxlags)

    sm_result = sm.WLS(y_values, x_values, weights=weight_values).fit(
        cov_type=sm_cov_type, cov_kwds=cov_kwds
    )

    return _from_statsmodels_ols(sm_result, feature_names, x_values, cov_type)


def get_gls(
    y: pd.Series | np.ndarray,
    x: pd.DataFrame | pd.Series | np.ndarray,
    omega: np.ndarray,
    add_constant: bool = True,
) -> dict:
    """
    Fit a Generalized Least Squares (GLS) regression of `y` on `x`, given a known
    error covariance structure `omega`, via `statsmodels.api.GLS`.

    Also known as: GLS.

    GLS generalizes WLS to an arbitrary (not necessarily diagonal) error covariance
    matrix `Var(e) = sigma^2 * Omega`, which also corrects for autocorrelated errors
    (a non-diagonal `Omega`), not just heteroskedastic ones:

    - beta_hat = (X' Omega^-1 X)^-1 X' Omega^-1 y

    Note that `Omega` must be supplied (e.g. from a fitted AR(1) error structure or
    a known covariance model) -- this implements the GLS estimation step itself, not
    the (separate, model-specific) problem of estimating `Omega` in the first place
    (Feasible GLS).

    Args:
        y (pd.Series | np.ndarray): The dependent variable, shape `(n,)`.
        x (pd.DataFrame | pd.Series | np.ndarray): The independent variable(s)/regressor(s).
        omega (np.ndarray): The (symmetric, positive-definite) error covariance
        structure, up to the scalar `sigma^2`, shape `(n, n)`.
        add_constant (bool, optional): Whether to prepend an intercept column of ones
        to `x`. Defaults to True.

    Returns:
        dict: The fitted coefficients, their standard errors/t-statistics/
        p-values, residuals, fitted values, R-squared and related fit statistics -- see
        `_from_statsmodels_ols` for the full key list. Call `regression_summary_table`
        for a coefficient table.

    Raises:
        TypeError: If `y` or `x` is not one of the accepted types.
        ValueError: If `omega` is not a square `(n, n)` matrix, is not positive
        definite, or there are not more observations than parameters.

    Notes:
        Reference: Aitken, A.C. (1935). "On Least Squares and Linear Combinations of
        Observations." Proceedings of the Royal Society of Edinburgh, 55, 42-48.
    """
    y_values = _to_target_vector(y)
    x_values, feature_names = _to_design_matrix(x, add_constant)

    n = len(y_values)
    if omega.shape != (n, n):
        raise ValueError(
            f"omega must be a square ({n}, {n}) matrix matching the number of "
            f"observations, received shape {omega.shape}."
        )

    try:
        np.linalg.cholesky(omega)
    except np.linalg.LinAlgError as error:
        raise ValueError("omega must be positive definite.") from error

    _validate_design(x_values, y_values)

    sm_result = sm.GLS(y_values, x_values, sigma=omega).fit()

    return _from_statsmodels_ols(sm_result, feature_names, x_values, "nonrobust")


def binary_regression_summary_table(result: dict) -> pd.DataFrame:
    """
    Builds the standard coefficient table (Coefficient, Std. Error, z-Statistic,
    P-Value) from a `get_logistic_regression`/`get_probit_regression` result dict,
    indexed by `result["feature_names"]`.

    Args:
        result (dict): A fitted binary regression result dict, as returned by
        `get_logistic_regression` or `get_probit_regression`.

    Returns:
        pd.DataFrame: The coefficient table.
    """
    return pd.DataFrame(
        {
            "Coefficient": result["coefficients"],
            "Std. Error": result["standard_errors"],
            "z-Statistic": result["z_statistics"],
            "P-Value": result["p_values"],
        },
        index=result["feature_names"],
    )


def _fit_binary_glm(
    y: np.ndarray,
    x: np.ndarray,
    feature_names: list[str],
    link: str,
    max_iterations: int,
    tolerance: float,
) -> dict:
    """
    Fits a binary-outcome regression via `statsmodels.api.Logit`/`Probit`, used by
    both `get_logistic_regression` (`link="logit"`) and `get_probit_regression`
    (`link="probit"`). Returns the fitted output as a dict:

    Keys:
        coefficients (np.ndarray): The estimated coefficients, shape `(k,)`.
        standard_errors (np.ndarray): The standard errors, from the inverse Fisher
        information matrix at convergence, shape `(k,)`.
        z_statistics (np.ndarray): The coefficients divided by their standard errors
        (a Wald z-statistic, not a t-statistic -- these are asymptotic maximum
        likelihood estimates, not exact finite-sample OLS ones).
        p_values (np.ndarray): Two-sided p-values from the standard normal distribution.
        fitted_probabilities (np.ndarray): The fitted P(y=1 | x) for each observation,
        shape `(n,)`.
        log_likelihood (float): The maximized log-likelihood.
        null_log_likelihood (float): The log-likelihood of an intercept-only model,
        used as the baseline for `pseudo_r_squared`.
        pseudo_r_squared (float): McFadden's pseudo-R-squared,
        `1 - log_likelihood / null_log_likelihood`.
        n_observations (int): The number of observations used, `n`.
        n_parameters (int): The number of estimated parameters, `k`.
        n_iterations (int): The number of solver iterations used.
        converged (bool): Whether the solver converged within `max_iterations`.
        feature_names (list[str]): The name of each coefficient, in order.
    """
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y must contain only 0/1 values for a binary regression.")

    n, k = x.shape
    if n <= k:
        raise ValueError(
            f"Not enough observations ({n}) to estimate {k} parameters -- need "
            f"strictly more observations than parameters."
        )

    model_cls = sm.Logit if link == "logit" else sm.Probit
    sm_result = model_cls(y, x).fit(
        disp=0, maxiter=max_iterations, tol=tolerance, warn_convergence=False
    )

    return {
        "coefficients": np.asarray(sm_result.params),
        "standard_errors": np.asarray(sm_result.bse),
        "z_statistics": np.asarray(sm_result.tvalues),
        "p_values": np.asarray(sm_result.pvalues),
        "fitted_probabilities": np.asarray(sm_result.predict()),
        "log_likelihood": float(sm_result.llf),
        "null_log_likelihood": float(sm_result.llnull),
        "pseudo_r_squared": float(sm_result.prsquared),
        "n_observations": int(sm_result.nobs),
        "n_parameters": k,
        "n_iterations": int(sm_result.mle_retvals.get("iterations", 0)),
        "converged": bool(sm_result.mle_retvals.get("converged", False)),
        "feature_names": feature_names,
    }


def get_logistic_regression(
    y: pd.Series | np.ndarray,
    x: pd.DataFrame | pd.Series | np.ndarray,
    add_constant: bool = True,
    max_iterations: int = 50,
    tolerance: float = 1e-8,
) -> dict:
    """
    Fit a Logistic Regression (Logit model) of a binary outcome `y` on `x`, via
    `statsmodels.api.Logit`.

    Also known as: logit model, logit regression.

    Models the probability of a binary outcome via the logistic (sigmoid) link
    function:

    - P(y = 1 | x) = 1 / (1 + e^(-x'beta))

    Fit by Maximum Likelihood. Coefficients are log-odds ratios: a one-unit increase
    in a regressor changes the log-odds of `y = 1` by that regressor's coefficient,
    holding the others fixed.

    Args:
        y (pd.Series | np.ndarray): The binary (0/1) dependent variable, shape `(n,)`.
        x (pd.DataFrame | pd.Series | np.ndarray): The independent variable(s)/regressor(s).
        add_constant (bool, optional): Whether to prepend an intercept column of ones
        to `x`. Defaults to True.
        max_iterations (int, optional): The maximum number of solver iterations.
        Defaults to 50.
        tolerance (float, optional): The convergence tolerance. Defaults to 1e-8.

    Returns:
        dict: The fitted coefficients, their standard errors/z-statistics/p-values,
        fitted probabilities, log-likelihood and McFadden's pseudo-R-squared -- see
        `_fit_binary_glm` for the full key list. Call `binary_regression_summary_table`
        for a coefficient table.

    Raises:
        TypeError: If `y` or `x` is not one of the accepted types.
        ValueError: If `y` contains values other than 0/1, or there are not more
        observations than parameters.

    Notes:
        Reference: Berkson, J. (1944). "Application of the Logistic Function to
        Bio-Assay." Journal of the American Statistical Association, 39(227), 357-365.
    """
    y_values = _to_target_vector(y)
    x_values, feature_names = _to_design_matrix(x, add_constant)

    return _fit_binary_glm(
        y_values, x_values, feature_names, "logit", max_iterations, tolerance
    )


def get_probit_regression(
    y: pd.Series | np.ndarray,
    x: pd.DataFrame | pd.Series | np.ndarray,
    add_constant: bool = True,
    max_iterations: int = 50,
    tolerance: float = 1e-8,
) -> dict:
    """
    Fit a Probit Regression of a binary outcome `y` on `x`, via
    `statsmodels.api.Probit`.

    Also known as: probit model.

    Models the probability of a binary outcome via the standard normal CDF link
    function, rather than Logit's logistic CDF:

    - P(y = 1 | x) = Phi(x'beta)

    Where `Phi` is the standard normal cumulative distribution function. In practice
    Probit and Logit give very similar fitted probabilities for most data (their link
    functions are both roughly S-shaped and symmetric); Probit coefficients are not
    directly comparable to Logit's log-odds interpretation, but the model is
    preferred in some econometric traditions and is required for one specific
    two-step causal inference building block: it underlies the Heckman selection
    correction and closely related two-stage estimators.

    Args:
        y (pd.Series | np.ndarray): The binary (0/1) dependent variable, shape `(n,)`.
        x (pd.DataFrame | pd.Series | np.ndarray): The independent variable(s)/regressor(s).
        add_constant (bool, optional): Whether to prepend an intercept column of ones
        to `x`. Defaults to True.
        max_iterations (int, optional): The maximum number of solver iterations.
        Defaults to 50.
        tolerance (float, optional): The convergence tolerance. Defaults to 1e-8.

    Returns:
        dict: The fitted coefficients, their standard errors/z-statistics/p-values,
        fitted probabilities, log-likelihood and McFadden's pseudo-R-squared -- see
        `_fit_binary_glm` for the full key list. Call `binary_regression_summary_table`
        for a coefficient table.

    Raises:
        TypeError: If `y` or `x` is not one of the accepted types.
        ValueError: If `y` contains values other than 0/1, or there are not more
        observations than parameters.

    Notes:
        Reference: Bliss, C.I. (1934). "The Method of Probits." Science, 79(2037),
        38-39.
    """
    y_values = _to_target_vector(y)
    x_values, feature_names = _to_design_matrix(x, add_constant)

    return _fit_binary_glm(
        y_values, x_values, feature_names, "probit", max_iterations, tolerance
    )


def quantile_regression_summary_table(result: dict) -> pd.DataFrame:
    """
    Builds the coefficient table (Coefficient, Std. Error) from a
    `get_quantile_regression` result dict, indexed by `result["feature_names"]`. The
    standard errors are `statsmodels`' analytic (kernel density-based) ones unless
    the fit requested bootstrap replicates, in which case they are the bootstrap
    ones -- either way they are always present.

    Args:
        result (dict): A fitted quantile regression result dict, as returned by
        `get_quantile_regression`.

    Returns:
        pd.DataFrame: The coefficient table.
    """
    data = {"Coefficient": result["coefficients"]}
    if result["standard_errors"] is not None:
        data["Std. Error"] = result["standard_errors"]
    return pd.DataFrame(data, index=result["feature_names"])


def get_quantile_regression(
    y: pd.Series | np.ndarray,
    x: pd.DataFrame | pd.Series | np.ndarray,
    tau: float = 0.5,
    add_constant: bool = True,
    n_bootstrap: int = 0,
    seed: int | None = None,
) -> dict:
    """
    Fit a (multi-predictor) linear Quantile Regression of `y` on `x` at quantile
    `tau`, via `statsmodels.api.QuantReg`.

    Also known as: QR.

    Unlike OLS (which minimizes squared residuals and estimates the conditional
    mean), Quantile Regression minimizes an asymmetric ("pinball") loss that is
    minimized at the conditional `tau`-quantile of `y` given `x`:

    - minimize: SUM(tau * max(r_t, 0) + (1 - tau) * max(-r_t, 0)),  r_t = y_t - x_t'beta

    Solved exactly as a Linear Program (Koenker & Bassett, 1978). Fitting at multiple
    values of `tau` (e.g. 0.1, 0.5, 0.9) characterizes how the *entire conditional
    distribution* of `y`, not just its mean, depends on `x` -- e.g. whether a
    regressor's effect is larger in the tails than at the median.

    Args:
        y (pd.Series | np.ndarray): The dependent variable, shape `(n,)`.
        x (pd.DataFrame | pd.Series | np.ndarray): The independent variable(s)/regressor(s).
        tau (float, optional): The quantile to fit, in (0, 1). Defaults to 0.5 (the
        median, i.e. Least Absolute Deviations regression).
        add_constant (bool, optional): Whether to prepend an intercept column of ones
        to `x`. Defaults to True.
        n_bootstrap (int, optional): The number of paired (row-)bootstrap resamples
        used to estimate coefficient standard errors, overriding `statsmodels`'
        default analytic (kernel density-based) standard errors. Defaults to 0 (use
        the analytic standard errors).
        seed (int | None, optional): The seed for the bootstrap random number
        generator, only used when `n_bootstrap > 0`. Defaults to None.

    Returns:
        dict: The fitted coefficients and standard errors, residuals, fitted values
        and Koenker-Machado pseudo-R-squared -- keys `coefficients`,
        `standard_errors` (the bootstrap standard deviation across replicates when
        `n_bootstrap > 0`, otherwise `statsmodels`' analytic kernel density-based
        ones -- always populated either way), `residuals`, `fitted_values`,
        `pseudo_r_squared`, `tau`, `n_observations`, `n_parameters`,
        `feature_names`. Call `quantile_regression_summary_table` for a coefficient
        table.

    Raises:
        TypeError: If `y` or `x` is not one of the accepted types.
        ValueError: If `tau` is not in (0, 1), or there are not more observations
        than parameters.

    Notes:
        Reference: Koenker, R., & Bassett, G. (1978). "Regression Quantiles."
        Econometrica, 46(1), 33-50.
    """
    if not 0 < tau < 1:
        raise ValueError(f"tau must be in (0, 1), received {tau}.")

    y_values = _to_target_vector(y)
    x_values, feature_names = _to_design_matrix(x, add_constant)
    n, k = x_values.shape

    _validate_design(x_values, y_values)

    sm_result = sm.QuantReg(y_values, x_values).fit(q=tau)

    standard_errors = np.asarray(sm_result.bse)
    if n_bootstrap > 0:
        rng = np.random.default_rng(seed)
        bootstrap_coefficients = np.empty((n_bootstrap, k))

        for i in range(n_bootstrap):
            sample_indices = rng.integers(0, n, size=n)

            with warnings.catch_warnings(record=True) as raised:
                warnings.simplefilter("always", IterationLimitWarning)
                replicate = sm.QuantReg(
                    y_values[sample_indices], x_values[sample_indices]
                ).fit(q=tau, max_iter=BOOTSTRAP_MAX_ITERATIONS)

            hit_iteration_limit = False

            for entry in raised:
                if entry.category is IterationLimitWarning:
                    hit_iteration_limit = True
                else:
                    # Re-emitted: recording captures every category, not just this one.
                    warnings.warn_explicit(
                        entry.message,
                        entry.category,
                        entry.filename,
                        entry.lineno,
                    )

            # Stopped short of the requested quantile, so it describes no quantile.
            if hit_iteration_limit:
                bootstrap_coefficients[i] = np.nan
                continue

            bootstrap_coefficients[i] = np.asarray(replicate.params)

        n_converged = int(np.sum(np.isfinite(bootstrap_coefficients[:, 0])))

        if n_converged < MINIMUM_CONVERGED_REPLICATES:
            raise ValueError(
                f"Only {n_converged} of {n_bootstrap} bootstrap replicates converged, "
                "which is too few to estimate a standard error from. Raise "
                "n_bootstrap, or drop it to fall back on the analytic standard errors."
            )

        standard_errors = np.nanstd(bootstrap_coefficients, axis=0, ddof=1)

    return {
        "coefficients": np.asarray(sm_result.params),
        "standard_errors": standard_errors,
        "residuals": np.asarray(sm_result.resid),
        "fitted_values": np.asarray(sm_result.fittedvalues),
        "pseudo_r_squared": float(sm_result.prsquared),
        "tau": tau,
        "n_observations": n,
        "n_parameters": k,
        "feature_names": feature_names,
    }
