"""Panel Data Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS, RandomEffects
from scipy import stats

# pylint: disable=too-many-locals

# The number of index levels a "panel" (entity, time) Series/DataFrame is expected
# to have -- this is a genuinely different multi-index convention from the
# (period, sub-period) "within period" one used throughout `risk_model.py`/
# `diagnostics_model.py` (see `MULTI_PERIOD_INDEX_LEVELS` there): here the two
# levels are simultaneously meaningful (which entity, which point in time) rather
# than one level being a nested nesting of the other, so it is kept as its own
# constant rather than reused from those modules.
ENTITY_TIME_INDEX_LEVELS = 2

# The conventional 5% significance level used to flag whether the Hausman test
# rejects Random Effects' exogeneity assumption in favor of Fixed Effects.
SIGNIFICANCE_LEVEL = 0.05


def _normalize_time_level(index: pd.MultiIndex) -> pd.MultiIndex:
    """
    `linearmodels.panel.PanelData` requires the "time" index level to be numeric or
    `datetime64`-like -- this codebase's weekly/monthly/quarterly/yearly historical
    data is indexed by `pd.PeriodIndex` (e.g. "2020-01-06/2020-01-12" for a week),
    which is neither, so the time level is converted to its `Timestamp` (period end)
    representation here before handing the panel off to `linearmodels`.
    """
    time_codebook = index.levels[index.names.index("time")]
    if isinstance(time_codebook, pd.PeriodIndex):
        return index.set_levels(time_codebook.to_timestamp(), level="time")
    return index


def _to_panel_series(data: pd.Series | pd.DataFrame, label: str) -> pd.Series:
    """
    Normalizes `data` into a `pd.Series` with a 2-level `(entity, time)` MultiIndex,
    accepting either that shape already, or a wide entity-by-time `pd.DataFrame`
    (one row per entity, one column per time period -- e.g. a `tickers x dates`
    frame) which is stacked into the same long shape. Shared by `y` in
    `get_fixed_effects`/`get_random_effects` and, via `_to_panel_frame`, by
    single-regressor `x` inputs too. This is the shape `linearmodels.panel`
    expects natively, so no further reshaping is needed once normalized.
    """
    if isinstance(data, pd.Series):
        if (
            not isinstance(data.index, pd.MultiIndex)
            or data.index.nlevels != ENTITY_TIME_INDEX_LEVELS
        ):
            raise TypeError(
                f"{label} must be a pd.Series with a 2-level (entity, time) "
                f"MultiIndex, or a wide entity x time pd.DataFrame, received a "
                f"pd.Series with a {getattr(data.index, 'nlevels', 1)}-level index."
            )
        series = data.copy()
    elif isinstance(data, pd.DataFrame):
        series = data.stack()
    else:
        raise TypeError(
            f"{label} must be a pd.Series or pd.DataFrame, received "
            f"{type(data).__name__}."
        )

    series.index = _normalize_time_level(series.index.set_names(["entity", "time"]))
    return series.astype(float)


def _to_panel_frame(data: pd.Series | pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Normalizes `x` into a `pd.DataFrame` with a 2-level `(entity, time)` MultiIndex
    and one column per regressor -- accepting a multi-regressor `pd.DataFrame`
    that is already `(entity, time)`-indexed (one column per regressor), or
    anything `_to_panel_series` accepts for a single regressor (a `(entity,
    time)`-indexed `pd.Series`, or a wide entity x time `pd.DataFrame`).
    """
    if (
        isinstance(data, pd.DataFrame)
        and isinstance(data.index, pd.MultiIndex)
        and data.index.nlevels == ENTITY_TIME_INDEX_LEVELS
    ):
        frame = data.copy()
        frame.index = _normalize_time_level(frame.index.set_names(["entity", "time"]))
        return frame.astype(float)

    series = _to_panel_series(data, label)
    name = series.name if series.name is not None else "X1"
    return series.to_frame(name)


def _align_panel(y: pd.Series, x: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """
    Inner-joins `y` and `x` on their shared `(entity, time)` index and drops any
    row with a missing value in either -- panels built from real return data are
    frequently unbalanced (assets with different listing histories), so this is
    not optional the way it might be for a single already-aligned series.
    """
    combined = pd.concat([y.rename("__y__"), x], axis=1, join="inner").dropna()

    if combined.empty:
        raise ValueError(
            "y and x have no overlapping, non-missing (entity, time) observations."
        )

    return combined["__y__"], combined.drop(columns="__y__")


def _effect_intercepts(
    y_panel: pd.Series, x_panel: pd.DataFrame, coefficients: np.ndarray, level: str
) -> pd.Series:
    """
    Recovers the fixed-effect intercepts implied a fitted within-regression:
    `alpha_g = mean_{within g}(y - X @ beta)`, which (since the mean is linear) is
    algebraically identical to `mean(y) - mean(X) @ beta` for each group `g` of the
    given `level` ("entity" or "time"). `linearmodels.panel.PanelOLS`'s own
    `estimated_effects` combines entity and time effects into a single column for
    the two-way case, so this recomputes them from the fitted slope coefficients
    directly instead, keeping them separable regardless of which effects are active.
    """
    residual_from_slope = y_panel.to_numpy() - x_panel.to_numpy() @ coefficients
    return (
        pd.Series(residual_from_slope, index=y_panel.index).groupby(level=level).mean()
    )


def _from_panel_result(
    sm_result, feature_names: list[str], design_matrix: np.ndarray
) -> dict:
    """
    Translates a fitted `linearmodels` panel results object into the same
    regression result dict shape `regression_model._from_statsmodels_ols` returns
    (see its docstring for the full key list) -- keeps `PanelOLS`/`RandomEffects`
    fits interchangeable with OLS/WLS/GLS fits everywhere downstream.
    """
    n = int(sm_result.nobs)
    degrees_of_freedom = int(sm_result.df_resid)
    r_squared = float(sm_result.rsquared)
    n_predictors = len(feature_names) - (1 if "Intercept" in feature_names else 0)
    adjusted_r_squared = (
        1 - (1 - r_squared) * (n - 1) / degrees_of_freedom
        if n_predictors > 0 and degrees_of_freedom > 0
        else r_squared
    )

    return {
        "coefficients": sm_result.params.to_numpy(),
        "standard_errors": sm_result.std_errors.to_numpy(),
        "t_statistics": sm_result.tstats.to_numpy(),
        "p_values": sm_result.pvalues.to_numpy(),
        "residuals": sm_result.resids.to_numpy(),
        "fitted_values": sm_result.fitted_values.to_numpy().ravel(),
        "covariance_matrix": sm_result.cov.to_numpy(),
        "r_squared": r_squared,
        "adjusted_r_squared": adjusted_r_squared,
        "residual_variance": float(sm_result.s2),
        "degrees_of_freedom": degrees_of_freedom,
        "n_observations": n,
        "n_parameters": len(feature_names),
        "feature_names": feature_names,
        "design_matrix": design_matrix,
        "cov_type": "nonrobust",
    }


def get_fixed_effects(
    y: pd.Series | pd.DataFrame,
    x: pd.DataFrame | pd.Series,
    entity_effects: bool = True,
    time_effects: bool = False,
) -> dict:
    """
    Fit a Fixed Effects ("within") estimator of `y` on `x` for panel data, via
    `linearmodels.panel.PanelOLS`.

    Also known as: within estimator, FE, least squares dummy variable (LSDV)
    estimator (numerically identical to, but far cheaper than, running OLS with a
    dummy regressor per entity).

    Panel data observes multiple entities (e.g. stocks) over multiple time periods
    (e.g. months), and often each entity has its own, time-invariant unobserved
    characteristic (e.g. a stock's typical risk premium) that is correlated with
    the regressors -- ignoring this (pooled OLS) yields biased coefficients via
    omitted-variable bias. Fixed Effects sidesteps the need to observe or even name
    that characteristic by demeaning it away: subtracting each entity's mean from
    every variable removes any purely entity-specific (time-invariant) component,
    including the unobserved one, before running OLS on what remains:

    - y~_it = y_it - ybar_i.,  x~_it = x_it - xbar_i.  (entity_effects)
    - beta_hat = (X~'X~)^-1 X~'y~

    `time_effects=True` additionally (or instead) demeans by the time-period mean,
    controlling for anything common to all entities at a given time (e.g. a
    market-wide shock) the same way. Because the entity/time intercepts are
    absorbed rather than estimated as coefficients, they are recovered separately
    (see the returned dict's `entity_effects`/`time_effects` keys) as
    `alpha_i = mean_t(y_it) - mean_t(x_it) @ beta`.

    Args:
        y (pd.Series | pd.DataFrame): The dependent variable. Either a `pd.Series`
        with a 2-level `(entity, time)` MultiIndex, or a wide `pd.DataFrame` shaped
        entity x time (one row per entity, one column per time period).
        x (pd.DataFrame | pd.Series): The independent variable(s)/regressor(s), in
        the same `(entity, time)` shape as `y` -- a `pd.DataFrame` with a 2-level
        `(entity, time)` MultiIndex and one column per regressor, or (for a single
        regressor) anything `y` accepts.
        entity_effects (bool, optional): Whether to demean by the entity mean, i.e.
        control for time-invariant entity-specific characteristics. Defaults to True.
        time_effects (bool, optional): Whether to demean by the time-period mean,
        i.e. control for entity-invariant time-specific shocks. Defaults to False.

    Returns:
        dict: `{"regression": ..., "entity_effects": ..., "time_effects": ...}` --
        `regression` is the fitted within-regression result dict (degrees-of-freedom
        already corrected for the absorbed fixed effects, no intercept: it is
        absorbed by the fixed effects; see `_from_panel_result` for the full key
        list), `entity_effects`/`time_effects` are the recovered entity-/time-
        specific intercepts (`pd.Series`, or `None` if that effect was not fit).

    Raises:
        TypeError: If `y` or `x` is not one of the accepted types/shapes.
        ValueError: If both `entity_effects` and `time_effects` are False, `y` and
        `x` share no overlapping non-missing observations, or there are not enough
        observations to estimate the regressors plus the absorbed fixed-effect
        parameters.

    Notes:
        Reference: Wooldridge, J.M. (2010). "Econometric Analysis of Cross Section
        and Panel Data," 2nd ed., MIT Press, Chapter 10.

    As an example:

    ```python
    import numpy as np
    import pandas as pd
    from financetoolkit.econometrics import panel_data_model, regression_model

    rng = np.random.default_rng(42)
    entities = [f"E{i}" for i in range(6)]
    times = range(40)
    alpha = {entity: rng.uniform(-5, 5) for entity in entities}
    index = pd.MultiIndex.from_product([entities, times], names=["entity", "time"])

    x = pd.Series(rng.standard_normal(len(index)), index=index, name="X")
    y = pd.Series(
        [alpha[entity] + 2.0 * x.loc[entity, time] for entity, time in index],
        index=index,
    ) + rng.standard_normal(len(index)) * 0.1

    result = panel_data_model.get_fixed_effects(y, x)
    print(regression_model.regression_summary_table(result["regression"]).round(4))
    ```

    Which returns:

    |    |   Coefficient |   Std. Error |   t-Statistic |   P-Value |
    |:---|---------------:|--------------:|---------------:|-----------:|
    | X  |         1.9916 |        0.0072 |       278.1999 |     0.0000 |
    """
    if not entity_effects and not time_effects:
        raise ValueError("At least one of entity_effects or time_effects must be True.")

    y_panel = _to_panel_series(y, "y")
    x_panel = _to_panel_frame(x, "x")
    y_panel, x_panel = _align_panel(y_panel, x_panel)

    n = len(y_panel)
    k = x_panel.shape[1]
    n_entities = y_panel.index.get_level_values("entity").nunique()
    n_times = y_panel.index.get_level_values("time").nunique()

    if entity_effects and time_effects:
        absorbed = n_entities + n_times - 1
    elif entity_effects:
        absorbed = n_entities
    else:
        absorbed = n_times

    if n - k - absorbed <= 0:
        raise ValueError(
            f"Not enough observations ({n}) to estimate {k} regressor(s) plus "
            f"{absorbed} absorbed fixed-effect parameter(s)."
        )

    try:
        sm_result = PanelOLS(
            y_panel, x_panel, entity_effects=entity_effects, time_effects=time_effects
        ).fit()
    except (ValueError, ZeroDivisionError, np.linalg.LinAlgError) as error:
        raise ValueError(
            "Could not fit the Fixed Effects model -- check that there are enough "
            "observations per entity/time period."
        ) from error

    fit = _from_panel_result(sm_result, list(x_panel.columns), x_panel.to_numpy())

    entity_alpha = (
        _effect_intercepts(y_panel, x_panel, fit["coefficients"], "entity")
        if entity_effects
        else None
    )
    time_alpha = (
        _effect_intercepts(y_panel, x_panel, fit["coefficients"], "time")
        if time_effects
        else None
    )

    return {
        "regression": fit,
        "entity_effects": entity_alpha,
        "time_effects": time_alpha,
    }


def get_random_effects(
    y: pd.Series | pd.DataFrame,
    x: pd.DataFrame | pd.Series,
) -> dict:
    """
    Fit a Random Effects estimator of `y` on `x` for panel data, via
    `linearmodels.panel.RandomEffects` (Swamy-Arora feasible Generalized Least
    Squares).

    Also known as: RE, GLS panel estimator, Swamy-Arora estimator.

    Where Fixed Effects (`get_fixed_effects`) treats each entity's time-invariant
    characteristic as a nuisance parameter to demean away entirely, Random Effects
    treats it as a random draw `u_i ~ (0, sigma_u^2)`, uncorrelated with the
    regressors, that becomes part of a composite error term
    `e_it = u_i + v_it` (`v_it` the ordinary idiosyncratic error). Because part of
    the entity-specific variation is now "signal" rather than pure nuisance, RE is
    more efficient than FE -- but only if that uncorrelatedness assumption actually
    holds (see `get_hausman_test`). It is estimated by quasi-demeaning -- removing
    only a fraction `theta_i` of each entity's mean, rather than all of it:

    - y*_it = y_it - theta_i * ybar_i.,  x*_it = x_it - theta_i * xbar_i.
    - theta_i = 1 - sqrt(sigma_v^2 / (sigma_v^2 + T_i * sigma_u^2))

    `sigma_v^2` (the idiosyncratic variance) is estimated from the Fixed Effects
    residual variance; `sigma_u^2` (the between-entity variance) is estimated from
    the "between" regression of entity-mean `y` on entity-mean `x`
    (Swamy & Arora, 1972). `theta_i -> 1` recovers Fixed Effects (full demeaning)
    as `T_i -> infinity` or `sigma_u^2 >> sigma_v^2`; `theta_i -> 0` recovers pooled
    OLS (no demeaning at all) as `sigma_u^2 -> 0`. Unlike Fixed Effects, an
    intercept is retained (its own value is quasi-demeaned to `1 - theta_i` per
    observation, rather than dropped).

    Args:
        y (pd.Series | pd.DataFrame): The dependent variable. Either a `pd.Series`
        with a 2-level `(entity, time)` MultiIndex, or a wide `pd.DataFrame` shaped
        entity x time (one row per entity, one column per time period).
        x (pd.DataFrame | pd.Series): The independent variable(s)/regressor(s), in
        the same `(entity, time)` shape as `y`.

    Returns:
        dict: The fitted GLS regression result dict on the quasi-demeaned data,
        including an "Intercept" coefficient (the pooled/population-average
        intercept, distinct from Fixed Effects' per-entity intercepts) -- see
        `_from_panel_result` for the full key list.

    Raises:
        TypeError: If `y` or `x` is not one of the accepted types/shapes.
        ValueError: If `y` and `x` share no overlapping non-missing observations,
        or there are not enough observations to estimate the model.

    Notes:
        Reference: Swamy, P.A.V.B., & Arora, S.S. (1972). "The Exact Finite Sample
        Properties of the Estimators of Coefficients in the Error Components
        Regression Models." Econometrica, 40(2), 261-275.

    As an example:

    ```python
    import numpy as np
    import pandas as pd
    from financetoolkit.econometrics import panel_data_model, regression_model

    rng = np.random.default_rng(42)
    entities = [f"E{i}" for i in range(6)]
    times = range(40)
    # Entity effects here are independent of X -- the assumption Random Effects
    # needs to be a good (efficient, not just consistent) fit -- see get_hausman_test.
    alpha = {entity: rng.uniform(-5, 5) for entity in entities}
    index = pd.MultiIndex.from_product([entities, times], names=["entity", "time"])

    x = pd.Series(rng.standard_normal(len(index)), index=index, name="X")
    y = pd.Series(
        [alpha[entity] + 2.0 * x.loc[entity, time] for entity, time in index],
        index=index,
    ) + rng.standard_normal(len(index)) * 0.1

    result = panel_data_model.get_random_effects(y, x)
    print(regression_model.regression_summary_table(result).round(4))
    ```

    Which returns:

    |           |   Coefficient |   Std. Error |   t-Statistic |   P-Value |
    |:----------|---------------:|--------------:|---------------:|-----------:|
    | Intercept |         1.4022 |        1.1402 |         1.2298 |     0.2200 |
    | X         |         1.9916 |        0.0072 |       277.1783 |     0.0000 |
    """
    y_panel = _to_panel_series(y, "y")
    x_panel = _to_panel_frame(x, "x")
    y_panel, x_panel = _align_panel(y_panel, x_panel)

    n = len(y_panel)
    k = x_panel.shape[1]

    if n - k - 1 <= 0:
        raise ValueError(
            f"Not enough observations ({n}) to estimate {k} parameter(s) plus an "
            "intercept."
        )

    x_with_constant = x_panel.copy()
    x_with_constant.insert(0, "Intercept", 1.0)

    try:
        sm_result = RandomEffects(y_panel, x_with_constant).fit()
    except (ValueError, ZeroDivisionError, np.linalg.LinAlgError) as error:
        raise ValueError(
            "Could not fit the Random Effects model -- check that there are enough "
            "entities and observations."
        ) from error

    return _from_panel_result(
        sm_result, list(x_with_constant.columns), x_with_constant.to_numpy()
    )


def get_hausman_test(
    y: pd.Series | pd.DataFrame,
    x: pd.DataFrame | pd.Series,
) -> pd.Series:
    """
    Calculate the Hausman specification test comparing a Fixed Effects and a
    Random Effects fit of the same panel data model.

    Also known as: Hausman specification test, Hausman-Wu test.

    Random Effects (`get_random_effects`) is more efficient than Fixed Effects
    (`get_fixed_effects`) but relies on entity effects being uncorrelated with the
    regressors -- if that assumption is violated, Random Effects is inconsistent
    (systematically biased) while Fixed Effects remains consistent regardless
    (since it removes entity effects entirely rather than modeling them). The
    Hausman test formalizes "should I trust the efficiency gain of Random Effects,
    or is Fixed Effects the safer choice here?" by testing whether the two
    estimators' coefficients differ by more than sampling variation would explain:

    - H = (b_FE - b_RE)' * [Var(b_FE) - Var(b_RE)]^-1 * (b_FE - b_RE)

    Under the null hypothesis that Random Effects' exogeneity assumption holds
    (and is therefore consistent AND efficient), `H` is asymptotically
    chi-squared distributed with degrees of freedom equal to the number of
    coefficients compared. A significant result rejects that null -- i.e.
    indicates Random Effects is inconsistent and Fixed Effects should be
    preferred, even though it discards the between-entity variation Random
    Effects would otherwise use.

    Neither `linearmodels` nor `statsmodels` ships a single Hausman-test function
    for comparing separately fitted `PanelOLS`/`RandomEffects` results -- this
    combines their `.params`/`.cov` directly via the standard formula above, the
    usual practice recommended in the `linearmodels` documentation itself.

    Args:
        y (pd.Series | pd.DataFrame): The dependent variable. Either a `pd.Series`
        with a 2-level `(entity, time)` MultiIndex, or a wide `pd.DataFrame` shaped
        entity x time (one row per entity, one column per time period).
        x (pd.DataFrame | pd.Series): The independent variable(s)/regressor(s), in
        the same `(entity, time)` shape as `y`.

    Returns:
        pd.Series: The Hausman statistic, its degrees of freedom, its p-value, and
        whether Fixed Effects is preferred over Random Effects at the 5% level.

    Raises:
        TypeError: If `y` or `x` is not one of the accepted types/shapes.
        ValueError: If `y` and `x` share no overlapping non-missing observations,
        there are not enough observations to fit either model, or the two
        estimators share no common regressors to compare.

    Notes:
        Reference: Hausman, J.A. (1978). "Specification Tests in Econometrics."
        Econometrica, 46(6), 1251-1271.

    As an example:

    ```python
    import numpy as np
    import pandas as pd
    from financetoolkit.econometrics import panel_data_model

    rng = np.random.default_rng(1)
    entities = [f"E{i}" for i in range(300)]
    times = range(8)
    index = pd.MultiIndex.from_product([entities, times], names=["entity", "time"])
    x = pd.Series(rng.standard_normal(len(index)), index=index, name="X")

    # Entity effects are a (noisy) function of X's own entity-mean -- this
    # violates Random Effects' exogeneity assumption, so the Hausman test below
    # should reject.
    entity_mean_x = x.groupby(level="entity").mean()
    eta = pd.Series(rng.normal(0, 0.3, len(entities)), index=entities)
    alpha = 0.5 * entity_mean_x + eta

    y = pd.Series(
        [alpha[entity] + 2.0 * x.loc[entity, time] for entity, time in index],
        index=index,
    ) + rng.standard_normal(len(index)) * 0.6

    result = panel_data_model.get_hausman_test(y, x)
    print(result.round(4))
    ```

    Which returns:

    | Metric                     |     Value |
    |:----------------------------|----------:|
    | Hausman Statistic           |  134.3958 |
    | Degrees of Freedom          |    1      |
    | P-Value                     |    0      |
    | Prefer Fixed Effects (5%)   |    1      |

    Whereas if `alpha` is instead independent of `X` (`alpha = eta` alone, no
    `entity_mean_x` term -- satisfying Random Effects' exogeneity assumption), the
    same test on that data gives a Hausman Statistic of 0.483 and a P-Value of
    0.4871: comfortably fails to reject, correctly preferring the more efficient
    Random Effects estimator in that case.
    """
    fe_result = get_fixed_effects(y, x, entity_effects=True, time_effects=False)
    re_result = get_random_effects(y, x)

    fe = fe_result["regression"]
    re = re_result

    common_names = [name for name in fe["feature_names"] if name in re["feature_names"]]
    if not common_names:
        raise ValueError(
            "The Fixed Effects and Random Effects fits share no common "
            "regressors to compare -- Random Effects always adds an 'Intercept' "
            "that Fixed Effects does not estimate, but there should still be at "
            "least one shared slope regressor."
        )

    fe_index = [fe["feature_names"].index(name) for name in common_names]
    re_index = [re["feature_names"].index(name) for name in common_names]

    b_fe = fe["coefficients"][fe_index]
    b_re = re["coefficients"][re_index]
    var_fe = fe["covariance_matrix"][np.ix_(fe_index, fe_index)]
    var_re = re["covariance_matrix"][np.ix_(re_index, re_index)]

    difference = b_fe - b_re
    variance_difference = var_fe - var_re

    try:
        inverse_variance_difference = np.linalg.inv(variance_difference)
    except np.linalg.LinAlgError:
        # A non-invertible (e.g. singular, or not positive definite due to small-
        # sample noise) Var(b_FE) - Var(b_RE) is a well known practical issue with
        # the Hausman test -- fall back to the Moore-Penrose pseudo-inverse rather
        # than raising, matching how e.g. Stata's `hausman` degrades in this case.
        inverse_variance_difference = np.linalg.pinv(variance_difference)

    statistic = float(difference @ inverse_variance_difference @ difference)
    statistic = max(statistic, 0.0)
    degrees_of_freedom = len(common_names)
    p_value = float(stats.chi2.sf(statistic, degrees_of_freedom))

    return pd.Series(
        {
            "Hausman Statistic": statistic,
            "Degrees of Freedom": degrees_of_freedom,
            "P-Value": p_value,
            "Prefer Fixed Effects (5%)": bool(p_value < SIGNIFICANCE_LEVEL),
        }
    )
