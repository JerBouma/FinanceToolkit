"""Causal Inference Model"""

__docformat__ = "google"

from dataclasses import dataclass

import numpy as np
import pandas as pd
from linearmodels.iv import IV2SLS
from scipy import stats

from financetoolkit.econometrics.regression_model import (
    RegressionResult,
    _to_design_matrix,
    _to_target_vector,
    get_logistic_regression,
    get_ols,
    get_wls,
)

# pylint: disable=too-many-locals,too-many-arguments,too-many-instance-attributes

MINIMUM_LOCAL_OBSERVATIONS = 3


def get_iv_2sls(
    y: pd.Series | np.ndarray,
    x_endogenous: pd.DataFrame | pd.Series | np.ndarray,
    instruments: pd.DataFrame | pd.Series | np.ndarray,
    x_exogenous: pd.DataFrame | pd.Series | np.ndarray | None = None,
    add_constant: bool = True,
) -> RegressionResult:
    """
    Fit an Instrumental Variables regression via Two-Stage Least Squares (2SLS) of `y`
    on `x_endogenous` (+ `x_exogenous`), using `instruments` to isolate the variation in
    `x_endogenous` that is uncorrelated with the error term, via `linearmodels.iv.IV2SLS`.

    Also known as: IV, 2SLS, IV-2SLS.

    Ordinary Least Squares assumes every regressor is uncorrelated with the error term
    (exogeneity). When a regressor is instead correlated with the error -- e.g. because of
    reverse causality (`y` also affects `x_endogenous`), an omitted confounder that drives
    both, or measurement error -- OLS is biased and inconsistent, no matter how large the
    sample. 2SLS corrects for this given a valid instrument `z`: a variable that (a) is
    correlated with the endogenous regressor ("relevance") and (b) affects `y` *only*
    through its effect on `x_endogenous`, i.e. is itself uncorrelated with the error
    ("exclusion restriction"). Conceptually it proceeds in two stages:

    - Stage 1: regress `x_endogenous` on `instruments` (+ `x_exogenous`), and keep only the
    *fitted* (instrument-explained) part of `x_endogenous`: `x_hat`.
    - Stage 2: regress `y` on `x_hat` (+ `x_exogenous`).

    Since `x_hat` is, by construction, uncorrelated with the structural error (it is a
    linear combination of the instruments and exogenous controls, both assumed exogenous),
    Stage 2 recovers a consistent estimate of the causal effect of `x_endogenous` on `y`,
    which naive OLS of `y` on the actual (untransformed) `x_endogenous` would not.

    IMPORTANT: simply running the two OLS regressions above back-to-back (as literally
    described) gives the *correct point estimates* for the coefficients, but running plain
    OLS diagnostics on the Stage 2 regression gives *wrong* standard errors -- it both
    computes residuals from `y - fitted(x_hat)` instead of the correct 2SLS residuals
    `y - fitted(x_actual)`, and altogether ignores the extra estimation uncertainty
    introduced by Stage 1. `linearmodels.iv.IV2SLS` handles this correction internally
    (fit here with `cov_type="unadjusted"` -- the classical, homoskedastic 2SLS covariance
    estimator -- and `debiased=True` for the small-sample `n - k` degrees-of-freedom
    correction), rather than the two-stage-by-hand approach this function used before.

    Args:
        y (pd.Series | np.ndarray): The dependent variable, shape `(n,)`.
        x_endogenous (pd.DataFrame | pd.Series | np.ndarray): The endogenous regressor(s) --
        suspected correlated with the error term.
        instruments (pd.DataFrame | pd.Series | np.ndarray): The excluded instrument(s) --
        correlated with `x_endogenous` but assumed uncorrelated with the error term. Must
        supply at least as many instruments as endogenous regressors (the order condition
        for identification).
        x_exogenous (pd.DataFrame | pd.Series | np.ndarray | None, optional): Other,
        non-instrumented control regressors included as-is (they act as their own
        instruments) in both stages. Defaults to None.
        add_constant (bool, optional): Whether to include an intercept in both stages.
        Defaults to True.

    Returns:
        RegressionResult: The 2SLS coefficients (on `x_exogenous` then `x_endogenous`, in
        that order), their standard errors/t-statistics/p-values, the actual-X residuals/
        fitted values, and related fit statistics. Call `.summary()` for a coefficient
        table. `design_matrix` holds the actual (not fitted/instrumented) `X`.

    Raises:
        TypeError: If any input is not one of the accepted types.
        ValueError: If the inputs have mismatched lengths, the model is underidentified
        (fewer instruments than endogenous regressors), there are not more observations
        than parameters, or the design is rank-deficient.

    Notes:
        Reference: Theil, H. (1953). "Repeated Least-Squares Applied to Complete
        Equation Systems." Central Planning Bureau (mimeo). See also Wooldridge, J.M.
        (2010). "Econometric Analysis of Cross Section and Panel Data," 2nd ed., MIT
        Press, Ch. 5, for the standard 2SLS variance formula.

    As an example:

    ```python
    import numpy as np
    import pandas as pd
    from financetoolkit.econometrics import causal_inference_model

    rng = np.random.default_rng(42)
    n = 2000
    confounder = rng.standard_normal(n)
    instrument = pd.Series(rng.standard_normal(n), name="Instrument")

    # x is driven by both the instrument and the confounder -- the confounder also
    # drives y directly, so x is endogenous (correlated with y's error term).
    x = pd.Series(0.8 * instrument + 0.6 * confounder + rng.standard_normal(n) * 0.3, name="X")
    y = 1.0 + 2.0 * x + 0.9 * confounder + rng.standard_normal(n) * 0.3

    result = causal_inference_model.get_iv_2sls(y, x, instrument)
    print(result.summary().round(4))
    ```

    Which returns (recovering the true slope of 2.0, unlike naive OLS which would be
    biased upward by the shared confounder):

    |           |   Coefficient |   Std. Error |   t-Statistic |   P-Value |
    |:----------|---------------:|--------------:|---------------:|-----------:|
    | Intercept |         0.9465 |        0.0211 |        44.9292 |     0.0000 |
    | X         |         2.0243 |        0.0259 |        78.2066 |     0.0000 |
    """
    y_values = _to_target_vector(y)
    endogenous_values, endogenous_names = _to_design_matrix(
        x_endogenous, add_constant=False
    )
    instrument_values, instrument_names = _to_design_matrix(
        instruments, add_constant=False
    )

    n = len(y_values)
    if x_exogenous is not None:
        exogenous_values, exogenous_names = _to_design_matrix(
            x_exogenous, add_constant=False
        )
    else:
        exogenous_values, exogenous_names = np.empty((n, 0)), []

    if not (
        endogenous_values.shape[0] == n
        and instrument_values.shape[0] == n
        and exogenous_values.shape[0] == n
    ):
        raise ValueError(
            "y, x_endogenous, instruments and x_exogenous must all have the same "
            "number of observations."
        )

    n_endogenous = endogenous_values.shape[1]
    n_instruments = instrument_values.shape[1]
    if n_instruments < n_endogenous:
        raise ValueError(
            f"The model is underidentified: {n_endogenous} endogenous regressor(s) "
            f"require at least as many excluded instruments, received {n_instruments}."
        )

    if add_constant:
        exogenous_values = np.column_stack([np.ones(n), exogenous_values])
        exogenous_names = ["Intercept", *exogenous_names]

    exog_frame = pd.DataFrame(exogenous_values, columns=exogenous_names)
    endog_frame = pd.DataFrame(endogenous_values, columns=endogenous_names)
    instrument_frame = pd.DataFrame(instrument_values, columns=instrument_names)

    try:
        sm_result = IV2SLS(
            y_values,
            exog_frame if exogenous_names else None,
            endog_frame,
            instrument_frame,
        ).fit(cov_type="unadjusted", debiased=True)
    except (ValueError, np.linalg.LinAlgError) as error:
        raise ValueError(
            "Could not fit the 2SLS model -- check that there are more observations "
            "than parameters and the design is not rank-deficient."
        ) from error

    feature_names = [*exogenous_names, *endogenous_names]
    design_matrix = np.column_stack([exogenous_values, endogenous_values])
    k = len(feature_names)
    degrees_of_freedom = int(sm_result.df_resid)

    return RegressionResult(
        coefficients=sm_result.params.to_numpy(),
        standard_errors=sm_result.std_errors.to_numpy(),
        t_statistics=sm_result.tstats.to_numpy(),
        p_values=sm_result.pvalues.to_numpy(),
        residuals=sm_result.resids.to_numpy(),
        fitted_values=sm_result.fitted_values.to_numpy().ravel(),
        covariance_matrix=sm_result.cov.to_numpy(),
        r_squared=float(sm_result.rsquared),
        adjusted_r_squared=float(sm_result.rsquared_adj),
        residual_variance=float(sm_result.s2),
        degrees_of_freedom=degrees_of_freedom,
        n_observations=n,
        n_parameters=k,
        feature_names=feature_names,
        design_matrix=design_matrix,
    )


def get_difference_in_differences(
    y: pd.Series | np.ndarray,
    treated: pd.Series | np.ndarray,
    post: pd.Series | np.ndarray,
    x_controls: pd.DataFrame | pd.Series | np.ndarray | None = None,
    add_constant: bool = True,
) -> RegressionResult:
    """
    Fit a Difference-in-Differences (DiD) regression of `y` on a treatment-group dummy
    (`treated`), a post-period dummy (`post`), and their interaction.

    Also known as: DiD, DD, difference-in-differences estimator.

    DiD estimates a causal treatment effect from panel/repeated-cross-section data by
    comparing the *change* in the outcome for a treated group before vs. after treatment
    to the *change* over the same two periods for an untreated (control) group, which
    nets out any time trend common to both groups (something a simple before/after
    comparison on the treated group alone could not distinguish from the treatment
    effect itself). It is estimated as a single OLS regression with a treatment-group
    dummy, a post-period dummy, and their interaction:

    - y = a + b*treated + c*post + d*(treated*post) + e

    Where `treated` is 1 for units that are EVER treated (regardless of period) and 0
    for control units, and `post` is 1 for observations in the post-treatment period
    (regardless of group). The DiD estimate is `d`, the coefficient on the interaction
    term `treated*post` -- it isolates the extra change experienced by the treated group,
    over and above both the average treated/control gap (`b`) and the common time trend
    (`c`). This identifies a genuine causal effect under the "parallel trends" assumption:
    absent treatment, the treated and control groups would have moved in parallel.

    Args:
        y (pd.Series | np.ndarray): The outcome variable, shape `(n,)`, one observation
        per unit-period.
        treated (pd.Series | np.ndarray): 1 for units that are ever treated, 0 for
        control units, shape `(n,)`. Constant within a unit across periods.
        post (pd.Series | np.ndarray): 1 for post-treatment-period observations, 0 for
        pre-treatment-period observations, shape `(n,)`. Constant within a period
        across units.
        x_controls (pd.DataFrame | pd.Series | np.ndarray | None, optional): Additional
        control regressors included alongside the DiD terms. Defaults to None.
        add_constant (bool, optional): Whether to include an intercept. Defaults to True.

    Returns:
        RegressionResult: The fitted coefficients (`Treated`, `Post`, `Treated x Post`,
        then any controls), their standard errors/t-statistics/p-values, residuals,
        fitted values and related fit statistics. Call `.summary()` for a coefficient
        table -- the row labeled `"Treated x Post"` is the DiD treatment-effect estimate.

    Raises:
        TypeError: If any input is not one of the accepted types.
        ValueError: If the inputs have mismatched lengths, `treated`/`post` contain
        values other than 0/1, or there are not more observations than parameters.

    Notes:
        Reference: Card, D., & Krueger, A.B. (1994). "Minimum Wages and Employment: A
        Case Study of the Fast-Food Industry in New Jersey and Pennsylvania." American
        Economic Review, 84(4), 772-793.

    As an example:

    ```python
    import numpy as np
    import pandas as pd
    from financetoolkit.econometrics import causal_inference_model

    rng = np.random.default_rng(1)
    n_units, n_periods = 100, 2
    unit = np.repeat(np.arange(n_units), n_periods)
    period = np.tile(np.arange(n_periods), n_units)
    treated = (unit % 2 == 0).astype(float)  # half the units are treated
    post = (period == 1).astype(float)

    true_effect = 3.0
    y = (
        1.0
        + 0.5 * treated
        + 0.2 * post
        + true_effect * treated * post
        + rng.standard_normal(n_units * n_periods) * 0.2
    )

    result = causal_inference_model.get_difference_in_differences(
        pd.Series(y), pd.Series(treated), pd.Series(post)
    )
    print(result.summary().round(4))
    ```

    Which returns (recovering the true treatment effect of 3.0 in the `Treated x Post` row):

    |                 |   Coefficient |   Std. Error |   t-Statistic |   P-Value |
    |:----------------|---------------:|--------------:|---------------:|-----------:|
    | Intercept       |         0.9645 |        0.0263 |        36.6630 |     0.0000 |
    | Treated         |         0.5413 |        0.0372 |        14.5482 |     0.0000 |
    | Post            |         0.2317 |        0.0372 |         6.2270 |     0.0000 |
    | Treated x Post  |         2.9370 |        0.0526 |        55.8183 |     0.0000 |
    """
    y_values = _to_target_vector(y)
    treated_values = _to_target_vector(treated)
    post_values = _to_target_vector(post)

    n = len(y_values)
    if len(treated_values) != n or len(post_values) != n:
        raise ValueError(
            "y, treated and post must all have the same number of observations."
        )
    if not np.all(np.isin(treated_values, [0, 1])):
        raise ValueError("treated must contain only 0/1 values.")
    if not np.all(np.isin(post_values, [0, 1])):
        raise ValueError("post must contain only 0/1 values.")

    interaction_values = treated_values * post_values

    design = pd.DataFrame(
        {
            "Treated": treated_values,
            "Post": post_values,
            "Treated x Post": interaction_values,
        }
    )

    if x_controls is not None:
        control_values, control_names = _to_design_matrix(
            x_controls, add_constant=False
        )
        if control_values.shape[0] != n:
            raise ValueError(
                "x_controls must have the same number of observations as y."
            )
        for name, column in zip(control_names, control_values.T):
            design[name] = column

    return get_ols(y_values, design, add_constant=add_constant)


@dataclass
class RegressionDiscontinuityResult:
    """
    The fitted output of `get_regression_discontinuity`.

    Attributes:
        discontinuity (float): The estimated jump in the outcome at the cutoff (the
        right-side local intercept minus the left-side local intercept).
        standard_error (float): The standard error of `discontinuity`, combining the
        two local regressions' intercept standard errors under independence:
        `sqrt(se_right^2 + se_left^2)`.
        t_statistic (float): `discontinuity / standard_error`.
        p_value (float): The two-sided p-value from the Student-T distribution with
        `left_result.degrees_of_freedom + right_result.degrees_of_freedom` degrees of
        freedom.
        cutoff (float): The running-variable cutoff used.
        bandwidth (float): The bandwidth used (observations further than this from
        `cutoff` are excluded from both local regressions).
        kernel (str): The kernel used to weight observations by distance to the cutoff,
        `"uniform"` or `"triangular"`.
        n_left (int): The number of observations used in the left-side local regression.
        n_right (int): The number of observations used in the right-side local regression.
        left_result (RegressionResult): The left-side (running variable < cutoff) local
        linear regression.
        right_result (RegressionResult): The right-side (running variable >= cutoff)
        local linear regression.
    """

    discontinuity: float
    standard_error: float
    t_statistic: float
    p_value: float
    cutoff: float
    bandwidth: float
    kernel: str
    n_left: int
    n_right: int
    left_result: RegressionResult
    right_result: RegressionResult

    def summary(self) -> pd.DataFrame:
        """
        Returns:
            pd.DataFrame: A one-column table of the discontinuity estimate, its standard
            error/t-statistic/p-value, and the cutoff/bandwidth/sample sizes used.
        """
        return pd.DataFrame(
            {
                "Value": [
                    self.discontinuity,
                    self.standard_error,
                    self.t_statistic,
                    self.p_value,
                    self.cutoff,
                    self.bandwidth,
                    self.n_left,
                    self.n_right,
                ]
            },
            index=[
                "Discontinuity",
                "Std. Error",
                "t-Statistic",
                "P-Value",
                "Cutoff",
                "Bandwidth",
                "N Left",
                "N Right",
            ],
        )


def get_regression_discontinuity(
    y: pd.Series | np.ndarray,
    running_variable: pd.Series | np.ndarray,
    cutoff: float,
    bandwidth: float | None = None,
    kernel: str = "uniform",
) -> RegressionDiscontinuityResult:
    """
    Estimate a Sharp Regression Discontinuity Design (RDD): the jump in `y` exactly at
    `cutoff`, where treatment assignment switches on/off deterministically based on
    whether `running_variable` is above or below `cutoff`.

    Also known as: RDD, sharp RD.

    When a treatment is assigned purely by whether some observable "running variable"
    crosses a threshold (e.g. a test score cutoff for a scholarship, an index-inclusion
    market-cap threshold, an age-based eligibility rule), units just above and just below
    the cutoff are, in expectation, comparable in everything except treatment status --
    the discontinuity in the outcome AT the cutoff can therefore be interpreted causally,
    as a local average treatment effect for units near the threshold. This is estimated
    by fitting separate LOCAL linear regressions of `y` on the (cutoff-centered) running
    variable, one on each side of `cutoff`, restricted to observations within `bandwidth`:

    - Left:  y = a_left  + b_left  * (running_variable - cutoff),  running_variable <  cutoff
    - Right: y = a_right + b_right * (running_variable - cutoff),  running_variable >= cutoff

    The discontinuity estimate is `a_right - a_left` -- the gap between the two local
    regression lines evaluated exactly at the cutoff (where the centered running variable
    is zero, so each side's intercept IS its predicted value at the cutoff).

    Args:
        y (pd.Series | np.ndarray): The outcome variable, shape `(n,)`.
        running_variable (pd.Series | np.ndarray): The (continuous) variable that
        determines treatment assignment via `cutoff`, shape `(n,)`.
        cutoff (float): The threshold at which treatment status switches. Observations
        with `running_variable >= cutoff` are treated as "right"/treated,
        `running_variable < cutoff` as "left"/untreated.
        bandwidth (float | None, optional): The maximum distance from `cutoff` (in
        `running_variable` units) an observation may be to be included in either local
        regression. Defaults to None, which uses a simple, conservative rule of thumb:
        half of `running_variable`'s observed range on each side. This is deliberately
        naive (not an MSE-optimal bandwidth such as Imbens & Kalyanaraman, 2012) -- for
        production use, consider tuning `bandwidth` via cross-validation or supplying a
        bandwidth from a dedicated bandwidth-selection procedure.
        kernel (str, optional): How to weight observations by distance to the cutoff
        within the bandwidth, one of "uniform" (equal weight, local linear OLS) or
        "triangular" (linearly decaying weight, local linear WLS, standard in the RDD
        literature since it prioritizes cutoff-adjacent observations). Defaults to
        "uniform".

    Returns:
        RegressionDiscontinuityResult: The estimated discontinuity, its standard error/
        t-statistic/p-value, the cutoff/bandwidth/kernel used, and the two underlying
        local `RegressionResult`s. Call `.summary()` for a one-column results table.

    Raises:
        TypeError: If `y` or `running_variable` is not one of the accepted types.
        ValueError: If the inputs have mismatched lengths, `kernel` is invalid,
        `bandwidth` is not strictly positive, or there are fewer than 3 observations
        within the bandwidth on either side of the cutoff.

    Notes:
        Reference: Thistlethwaite, D.L., & Campbell, D.T. (1960). "Regression-
        Discontinuity Analysis: An Alternative to the Ex Post Facto Experiment." Journal
        of Educational Psychology, 51(6), 309-317. See also Imbens, G.W., & Lemieux, T.
        (2008). "Regression Discontinuity Designs: A Guide to Practice." Journal of
        Econometrics, 142(2), 615-635.

    As an example:

    ```python
    import numpy as np
    import pandas as pd
    from financetoolkit.econometrics import causal_inference_model

    rng = np.random.default_rng(7)
    n = 2000
    running = pd.Series(rng.uniform(-10, 10, n), name="Running")
    true_jump = 4.0
    y = 1.0 + 0.3 * running + true_jump * (running >= 0) + rng.standard_normal(n) * 0.5

    result = causal_inference_model.get_regression_discontinuity(y, running, cutoff=0.0)
    print(result.summary().round(4))
    ```

    Which returns (recovering the true jump of 4.0 at the cutoff):

    |               |    Value |
    |:--------------|---------:|
    | Discontinuity |   4.0129 |
    | Std. Error    |   0.0449 |
    | t-Statistic   |  89.2815 |
    | P-Value       |   0.0000 |
    | Cutoff        |   0.0000 |
    | Bandwidth     |   9.9877 |
    | N Left        | 1013     |
    | N Right       |  986     |
    """
    if kernel not in ("uniform", "triangular"):
        raise ValueError(
            f"kernel must be 'uniform' or 'triangular', received {kernel!r}."
        )

    y_values = _to_target_vector(y)
    running_values = _to_target_vector(running_variable)

    if len(running_values) != len(y_values):
        raise ValueError(
            "y and running_variable must have the same number of observations."
        )

    if bandwidth is None:
        bandwidth = 0.5 * (running_values.max() - running_values.min())

    if bandwidth <= 0:
        raise ValueError("bandwidth must be strictly positive.")

    centered = running_values - cutoff
    within_bandwidth = np.abs(centered) < bandwidth
    left_mask = within_bandwidth & (centered < 0)
    right_mask = within_bandwidth & (centered >= 0)

    n_left = int(left_mask.sum())
    n_right = int(right_mask.sum())
    if n_left < MINIMUM_LOCAL_OBSERVATIONS or n_right < MINIMUM_LOCAL_OBSERVATIONS:
        raise ValueError(
            f"Not enough observations within the bandwidth on each side of the cutoff "
            f"(left={n_left}, right={n_right}) -- need at least "
            f"{MINIMUM_LOCAL_OBSERVATIONS} per side to fit a local linear regression. "
            f"Consider increasing `bandwidth`."
        )

    def _fit_local(mask: np.ndarray) -> RegressionResult:
        x_side = centered[mask]
        y_side = y_values[mask]
        if kernel == "triangular":
            weights = 1 - np.abs(x_side) / bandwidth
            return get_wls(y_side, x_side, weights, add_constant=True)
        return get_ols(y_side, x_side, add_constant=True)

    left_result = _fit_local(left_mask)
    right_result = _fit_local(right_mask)

    intercept_index = left_result.feature_names.index("Intercept")
    left_intercept = left_result.coefficients[intercept_index]
    right_intercept = right_result.coefficients[intercept_index]
    left_se = left_result.standard_errors[intercept_index]
    right_se = right_result.standard_errors[intercept_index]

    discontinuity = float(right_intercept - left_intercept)
    standard_error = float(np.sqrt(left_se**2 + right_se**2))
    degrees_of_freedom = (
        left_result.degrees_of_freedom + right_result.degrees_of_freedom
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        t_statistic = discontinuity / standard_error
    p_value = float(2 * stats.t.sf(np.abs(t_statistic), degrees_of_freedom))

    return RegressionDiscontinuityResult(
        discontinuity=discontinuity,
        standard_error=standard_error,
        t_statistic=t_statistic,
        p_value=p_value,
        cutoff=float(cutoff),
        bandwidth=float(bandwidth),
        kernel=kernel,
        n_left=n_left,
        n_right=n_right,
        left_result=left_result,
        right_result=right_result,
    )


@dataclass
class PropensityScoreMatchingResult:
    """
    The fitted output of `get_propensity_score_matching`.

    Attributes:
        att (float): The estimated Average Treatment effect on the Treated -- the mean
        outcome difference (treated minus matched control) across matched pairs.
        standard_error (float): The standard error of `att`, the standard error of the
        mean of the (treated - matched control) paired differences.
        t_statistic (float): `att / standard_error`.
        p_value (float): The two-sided p-value from the Student-T distribution with
        `n_matched_pairs - 1` degrees of freedom.
        n_matched_pairs (int): The number of treated units successfully matched to a
        (not-yet-used) control unit.
        n_treated (int): The total number of treated units.
        n_control (int): The total number of control units.
        propensity_scores (np.ndarray): The fitted P(treated=1 | covariates) for every
        input observation, shape `(n,)`.
        matched_treated_indices (np.ndarray): The positional index (into the original
        input arrays) of each matched treated unit.
        matched_control_indices (np.ndarray): The positional index of each matched
        treated unit's matched control unit, in the same order as
        `matched_treated_indices`.
    """

    att: float
    standard_error: float
    t_statistic: float
    p_value: float
    n_matched_pairs: int
    n_treated: int
    n_control: int
    propensity_scores: np.ndarray
    matched_treated_indices: np.ndarray
    matched_control_indices: np.ndarray

    def summary(self) -> pd.Series:
        """
        Returns:
            pd.Series: The ATT estimate, its standard error/t-statistic/p-value, and the
            number of matched pairs / treated / control units.
        """
        return pd.Series(
            {
                "ATT": self.att,
                "Std. Error": self.standard_error,
                "t-Statistic": self.t_statistic,
                "P-Value": self.p_value,
                "Matched Pairs": self.n_matched_pairs,
                "N Treated": self.n_treated,
                "N Control": self.n_control,
            }
        )


def get_propensity_score_matching(
    treatment: pd.Series | np.ndarray,
    outcome: pd.Series | np.ndarray,
    covariates: pd.DataFrame | pd.Series | np.ndarray,
    caliper: float | None = None,
    add_constant: bool = True,
) -> PropensityScoreMatchingResult:
    """
    Estimate the Average Treatment effect on the Treated (ATT) via Propensity Score
    Matching (PSM): 1-to-1 nearest-neighbor matching on the estimated probability of
    treatment, without replacement.

    Also known as: PSM, nearest-neighbor propensity matching.

    When treatment is not randomly assigned but instead correlated with observable
    characteristics ("selection on observables" -- e.g. larger/more liquid assets are
    more likely to be included in an index, or riskier borrowers are more likely to
    take a given loan product), a naive comparison of mean outcomes between treated and
    untreated units is confounded by those same characteristics, not just the treatment
    itself. PSM addresses this in two steps:

    - (a) Fit a Logistic Regression of `treatment` on `covariates` (via
    `regression_model.get_logistic_regression`) to estimate each unit's propensity score,
    `P(treatment=1 | covariates)` -- a single number summarizing how "similar to a
    treated unit" that unit's covariates make it look, regardless of treatment status.
    - (b) For each treated unit, find the untreated (control) unit with the closest
    propensity score that has not already been matched to another treated unit
    (nearest-neighbor matching, 1-to-1, without replacement), in the order the treated
    units are given.

    Given (approximately) matched propensity scores, matched pairs differ (in
    expectation) only in treatment status, not in the covariates that drove selection
    into treatment -- so the mean outcome difference across matched pairs is a
    consistent estimate of the ATT, unlike the naive (unmatched) mean difference.

    Two implementation details, both standard recommended practice rather than
    arbitrary choices:

    - Matching distance is computed on the LOGIT of the propensity score (the model's
    linear predictor), not the raw 0-1 probability. Raw probabilities are compressed
    near 0 and 1 (two units can have very different covariates but near-identical,
    saturated probabilities), which the logit transform undoes -- see Austin, P.C.
    (2011), cited below.
    - A caliper defaults to `0.2 * (sample standard deviation of the logit propensity
    score)` when `caliper=None` -- Austin's (2011) widely cited rule-of-thumb width.
    Without SOME caliper, treated units in a region with no comparable control (poor
    "common support"/overlap between the treated and control covariate distributions)
    get matched to a distant control anyway, which reintroduces exactly the selection
    bias PSM is meant to remove.

    Args:
        treatment (pd.Series | np.ndarray): The binary (0/1) treatment indicator, shape
        `(n,)`.
        outcome (pd.Series | np.ndarray): The outcome variable, shape `(n,)`.
        covariates (pd.DataFrame | pd.Series | np.ndarray): The covariate(s) used to
        estimate the propensity score -- should include the observable(s) believed to
        drive selection into treatment.
        caliper (float | None, optional): The maximum allowed distance, in LOGIT-of-
        propensity-score units, between a treated unit and its matched control; treated
        units with no control within this distance are left unmatched. Defaults to
        None, which uses `0.2 * std(logit(propensity_score))` (Austin, 2011). Pass
        `np.inf` to disable the caliper entirely (match every treated unit to its
        nearest available control, however far).
        add_constant (bool, optional): Whether to include an intercept in the propensity
        score model. Defaults to True.

    Returns:
        PropensityScoreMatchingResult: The ATT estimate, its standard error/t-statistic/
        p-value, the number of matched pairs/treated/control units, the fitted
        propensity scores, and the matched index pairs. Call `.summary()` for a results
        table.

    Raises:
        TypeError: If any input is not one of the accepted types.
        ValueError: If the inputs have mismatched lengths, `treatment` contains values
        other than 0/1, there are no treated or no control units, or no treated unit can
        be matched within `caliper`.

    Notes:
        - Matching is greedy (treated units are matched in the order supplied, without
        revisiting earlier matches once later, closer matches are found) and without
        replacement (each control unit is used at most once) -- both standard,
        simplifying choices; matching with replacement is not implemented.
        - References: Rosenbaum, P.R., & Rubin, D.B. (1983). "The Central Role of the
        Propensity Score in Observational Studies for Causal Effects." Biometrika,
        70(1), 41-55. Austin, P.C. (2011). "Optimal Caliper Widths for Propensity-Score
        Matching when Estimating Differences in Means and Differences in Proportions in
        Observational Studies." Pharmaceutical Statistics, 10(2), 150-161.

    As an example:

    ```python
    import numpy as np
    import pandas as pd
    from financetoolkit.econometrics import causal_inference_model

    rng = np.random.default_rng(3)
    n = 2000
    covariate = pd.Series(rng.standard_normal(n), name="Size")

    # Selection bias: units with a larger covariate are more likely to be treated...
    propensity = 1 / (1 + np.exp(-1.5 * covariate))
    treatment = pd.Series((rng.uniform(size=n) < propensity).astype(float))

    # ...and the covariate ALSO drives the outcome directly, so a naive mean difference
    # is confounded -- the true, constant treatment effect is 2.0.
    true_effect = 2.0
    outcome = pd.Series(
        1.0 + 3.0 * covariate + true_effect * treatment + rng.standard_normal(n) * 0.5
    )

    result = causal_inference_model.get_propensity_score_matching(
        treatment, outcome, covariate
    )
    print(result.summary().round(4))
    ```

    Which returns (recovering the true effect of 2.0 in `att`, unlike the naive,
    upward-biased mean difference `outcome[treatment == 1].mean() - outcome[treatment == 0].mean()`):

    | Metric        |    Value |
    |:--------------|---------:|
    | ATT           |   2.0647 |
    | Std. Error    |   0.0302 |
    | t-Statistic   |  68.2821 |
    | P-Value       |   0.0000 |
    | Matched Pairs |  562     |
    | N Treated     | 1024     |
    | N Control     |  976     |
    """
    treatment_values = _to_target_vector(treatment)
    outcome_values = _to_target_vector(outcome)
    covariate_values, _ = _to_design_matrix(covariates, add_constant=False)

    n = len(treatment_values)
    if len(outcome_values) != n or covariate_values.shape[0] != n:
        raise ValueError(
            "treatment, outcome and covariates must all have the same number of observations."
        )
    if not np.all(np.isin(treatment_values, [0, 1])):
        raise ValueError("treatment must contain only 0/1 values.")

    propensity_model = get_logistic_regression(
        treatment_values, covariate_values, add_constant=add_constant
    )
    propensity_scores = propensity_model.fitted_probabilities

    # Match on the LOGIT of the propensity score, not the raw probability -- see
    # docstring ("Austin, 2011") for why this materially improves match quality.
    clipped_scores = np.clip(propensity_scores, 1e-6, 1 - 1e-6)
    logit_scores = np.log(clipped_scores / (1 - clipped_scores))

    treated_indices = np.flatnonzero(treatment_values == 1)
    control_indices = np.flatnonzero(treatment_values == 0)

    if len(treated_indices) == 0 or len(control_indices) == 0:
        raise ValueError(
            "Need at least one treated and one control observation to match."
        )

    if caliper is None:
        logit_std = np.std(logit_scores, ddof=1) if len(logit_scores) > 1 else 0.0
        caliper = 0.2 * logit_std

    available_control = list(control_indices)
    matched_treated: list[int] = []
    matched_control: list[int] = []

    for treated_index in treated_indices:
        if not available_control:
            break

        distances = np.abs(
            logit_scores[available_control] - logit_scores[treated_index]
        )
        closest_position = int(np.argmin(distances))
        closest_distance = distances[closest_position]

        if closest_distance > caliper:
            continue

        matched_treated.append(int(treated_index))
        matched_control.append(available_control.pop(closest_position))

    n_matched_pairs = len(matched_treated)
    if n_matched_pairs == 0:
        raise ValueError("No treated unit could be matched within the given caliper.")

    paired_differences = (
        outcome_values[matched_treated] - outcome_values[matched_control]
    )
    att = float(paired_differences.mean())

    if n_matched_pairs > 1:
        standard_error = float(
            paired_differences.std(ddof=1) / np.sqrt(n_matched_pairs)
        )
        degrees_of_freedom = n_matched_pairs - 1
        with np.errstate(divide="ignore", invalid="ignore"):
            t_statistic = att / standard_error
        p_value = float(2 * stats.t.sf(np.abs(t_statistic), degrees_of_freedom))
    else:
        standard_error = np.nan
        t_statistic = np.nan
        p_value = np.nan

    return PropensityScoreMatchingResult(
        att=att,
        standard_error=standard_error,
        t_statistic=t_statistic,
        p_value=p_value,
        n_matched_pairs=n_matched_pairs,
        n_treated=len(treated_indices),
        n_control=len(control_indices),
        propensity_scores=propensity_scores,
        matched_treated_indices=np.array(matched_treated),
        matched_control_indices=np.array(matched_control),
    )
