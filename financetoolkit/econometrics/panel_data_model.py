"""Panel Data Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS, RandomEffects
from linearmodels.panel.utility import AbsorbingEffectError
from scipy import sparse, stats
from scipy.sparse.linalg import lsmr

from financetoolkit.utilities.statistics_model import to_multi_index

# pylint: disable=too-many-locals

# (entity, time) levels are simultaneously meaningful, unlike the nested case.
ENTITY_TIME_INDEX_LEVELS = 2

# The 5% level for flagging whether Hausman rejects Random Effects.
SIGNIFICANCE_LEVEL = 0.05

# Stopping tolerances of the sparse least squares fit on the entity and time dummies:
# near machine precision, so the fit is exact on weakly connected panels too.
TWO_WAY_SOLVER_TOLERANCE = 1e-15

# A regressor is absorbed when the variation the fixed effects leave of it is below this
# share of its scale. The within-group range of a one-way absorbed regressor is exactly 0;
# the two-way least squares fit leaves rounding noise of about 1e-14, also on weakly
# connected panels, while a genuine regressor such as 1e6 plus noise of 1e-4 keeps about 3e-10.
ONE_WAY_ABSORBED_SHARE = 1e-12
TWO_WAY_ABSORBED_SHARE = 1e-11

# Why a regressor cannot be estimated next to the fixed effects that absorb it.
ABSORBED_REGRESSOR_EXPLANATION = (
    "a regressor that is the same for every entity at each date cannot be combined with "
    "time_effects, one that is constant per entity cannot be combined with "
    "entity_effects, and one that is the sum of an entity part and a date part cannot be "
    "combined with both."
)


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

    series.index = _normalize_time_level(
        to_multi_index(series.index).set_names(["entity", "time"])
    )
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
        frame.index = _normalize_time_level(
            to_multi_index(frame.index).set_names(["entity", "time"])
        )
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


def _two_way_demean(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Removes the entity and time effects of every column by regressing it on the entity
    and time dummies with sparse least squares (LSMR). This is the two-way within
    transformation `x_it - a_i - b_t` of a balanced or unbalanced panel, exact also on a
    weakly connected one (short overlapping histories, groups of entities sharing only
    a few dates) where alternating entity and time demeaning stalls.

    Returns:
        tuple: The demeaned data and the entity part `a_i` and time part `b_t` that were
        removed, all per observation. The split between `a_i` and `b_t` is only unique
        up to a level shift.

    Raises:
        ValueError: If the least squares fit does not converge.
    """
    entity_codes, entities = pd.factorize(data.index.get_level_values("entity"))
    time_codes, _ = pd.factorize(data.index.get_level_values("time"))
    rows = np.arange(len(data))
    dummies = sparse.csr_matrix(
        (
            np.ones(2 * len(data)),
            (np.r_[rows, rows], np.r_[entity_codes, len(entities) + time_codes]),
        )
    )
    # Scaling each dummy by one over the square root of its number of observations
    # (Jacobi preconditioning) cuts the iterations on a large staggered panel from about
    # 300 to 15; a weakly connected staircase of short histories still takes over 100.
    column_scale = 1 / np.sqrt(np.asarray(dummies.sum(axis=0)).ravel())
    scaled_dummies = dummies @ sparse.diags(column_scale)

    # The column mean goes to the entity part first: it lies in the span of the dummies,
    # and removing it keeps a large level (prices, market caps) out of the solver's
    # relative tolerance. conlim=0 drops the condition number stop, so only a converged
    # fit or the iteration limit ends the solve.
    values = data.to_numpy(dtype=float)
    entity_part = np.tile(values.mean(axis=0), (len(data), 1))
    time_part = np.zeros_like(values)
    for column in range(values.shape[1]):
        scaled_solution, istop, iterations = lsmr(
            scaled_dummies,
            values[:, column] - entity_part[:, column],
            atol=TWO_WAY_SOLVER_TOLERANCE,
            btol=TWO_WAY_SOLVER_TOLERANCE,
            conlim=0,
            maxiter=10 * dummies.shape[1],
        )[:3]
        # LSMR stops without a least squares solution on a condition number beyond
        # 1 / machine precision (6) or at the iteration limit (7).
        if istop in (6, 7):
            raise ValueError(
                f"The two-way fixed effects fit of {data.columns[column]} did not "
                f"converge (LSMR stop code {istop} after {iterations} iterations)."
            )
        solution = scaled_solution * column_scale
        entity_part[:, column] += solution[: len(entities)][entity_codes]
        time_part[:, column] = solution[len(entities) :][time_codes]

    index, columns = data.index, data.columns
    return (
        pd.DataFrame(values - entity_part - time_part, index=index, columns=columns),
        pd.DataFrame(entity_part, index=index, columns=columns),
        pd.DataFrame(time_part, index=index, columns=columns),
    )


def _effect_intercepts(
    y_panel: pd.Series,
    x_panel: pd.DataFrame,
    coefficients: np.ndarray,
    entity_effects: bool,
    time_effects: bool,
) -> tuple[pd.Series | None, pd.Series | None]:
    """
    Recovers the fixed-effect intercepts implied by a fitted within-regression from the
    slope-adjusted outcome `r_it = y_it - X_it @ beta`.

    - One-way: `alpha_g = mean_{within g}(r)` for each entity (or time period) `g`.
    - Two-way: `r_it = alpha_i + lambda_t + e_it`, normalized so that the time effects
    average zero over the observations. The entity effects then carry the overall level
    and `alpha_i + lambda_t + X_it @ beta` is the fitted value. Taking both one-way group
    means instead would count the grand mean of `r` twice.

    `linearmodels.panel.PanelOLS`'s own `estimated_effects` combines entity and time
    effects into a single column for the two-way case, so this separates them from the
    fitted slope coefficients directly. For a balanced panel the two-way solution is the
    closed form `alpha_i = rbar_i.` and `lambda_t = rbar_.t - rbar_..`; an unbalanced panel
    is solved by sparse least squares on the entity and time dummies.
    """
    residual_from_slope = pd.Series(
        y_panel.to_numpy() - x_panel.to_numpy() @ coefficients, index=y_panel.index
    )

    if not (entity_effects and time_effects):
        level = "entity" if entity_effects else "time"
        alpha = residual_from_slope.groupby(level=level).mean()
        return (alpha, None) if entity_effects else (None, alpha)

    _, entity_part, time_part = _two_way_demean(residual_from_slope.to_frame())
    # Pins the free level shift between the two sets of effects on the entity side.
    level_shift = time_part.iloc[:, 0].mean()
    entity_alpha = entity_part.iloc[:, 0].groupby(level="entity").mean() + level_shift
    time_alpha = time_part.iloc[:, 0].groupby(level="time").mean() - level_shift

    return entity_alpha, time_alpha


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
    `alpha_i = mean_t(y_it) - mean_t(x_it) @ beta`. With both effects, the time effects
    are normalized to average zero over the observations, so the entity effects carry
    the overall level and `alpha_i + lambda_t + x_it @ beta` is the fitted value.

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
        specific intercepts (`pd.Series`, or `None` if that effect was not fit). With
        both effects, the time effects are deviations that average zero over the
        observations.

    Raises:
        TypeError: If `y` or `x` is not one of the accepted types/shapes.
        ValueError: If both `entity_effects` and `time_effects` are False, `y` and
        `x` share no overlapping non-missing observations, there are not enough
        observations to estimate the regressors plus the absorbed fixed-effect
        parameters, or the fixed effects fully absorb a regressor (e.g. a common
        factor, identical for every entity at each date, with `time_effects=True`).

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
    regression_model.regression_summary_table(result["regression"]).round(4)
    ```

    Which returns:

    |    |   Coefficient |   Std. Error |   t-Statistic |   P-Value |
    |:---|--------------:|-------------:|--------------:|----------:|
    | X  |        1.9916 |       0.0072 |         278.2 |         0 |
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

    # PanelOLS only raises AbsorbingEffectError when the absorbed regressors leave the
    # demeaned design rank deficient as a whole; a single absorbed regressor is demeaned to
    # floating-point noise instead and fitted with a coefficient of 0 and a standard error
    # of order 1e15, or, when it is the sum a_i + b_t of an entity and a date part under
    # two-way effects, of order 1e9. Checking the variation that is left after the fixed
    # effects first catches all of these: the within-group range for one-way effects and
    # the two-way demeaned values (which also vanish for a regressor that is constant per
    # entity or per date) for two-way effects.
    scale = x_panel.abs().max().clip(lower=1.0)
    if entity_effects and time_effects:
        remaining = _two_way_demean(x_panel)[0].abs().max() / scale
        is_absorbed = remaining <= TWO_WAY_ABSORBED_SHARE
    else:
        grouped = x_panel.groupby(level="entity" if entity_effects else "time")
        remaining = (grouped.max() - grouped.min()).abs().max() / scale
        is_absorbed = remaining <= ONE_WAY_ABSORBED_SHARE
    absorbed_columns = [
        str(column) for column in x_panel.columns[is_absorbed.to_numpy()]
    ]
    if absorbed_columns:
        raise ValueError(
            "The fixed effects fully absorb one or more regressors "
            f"({', '.join(absorbed_columns)}): {ABSORBED_REGRESSOR_EXPLANATION}"
        )

    try:
        sm_result = PanelOLS(
            y_panel, x_panel, entity_effects=entity_effects, time_effects=time_effects
        ).fit()
    except AbsorbingEffectError as error:
        # A regressor that is identical for every entity at each date (a common factor
        # such as a market benchmark) is fully explained by the time effects, and one
        # that is constant per entity by the entity effects, so its slope is not identified.  # noqa: E501
        raise ValueError(
            "The fixed effects fully absorb one or more regressors: "
            f"{ABSORBED_REGRESSOR_EXPLANATION}"
        ) from error
    except (ValueError, ZeroDivisionError, np.linalg.LinAlgError) as error:
        raise ValueError(
            "Could not fit the Fixed Effects model -- check that there are enough "
            "observations per entity/time period."
        ) from error

    fit = _from_panel_result(sm_result, list(x_panel.columns), x_panel.to_numpy())

    entity_alpha, time_alpha = _effect_intercepts(
        y_panel, x_panel, fit["coefficients"], entity_effects, time_effects
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
        there are not enough observations to estimate the model, or there are no
        more entities than parameters (the Swamy-Arora between-entity variance
        divides by `N - k`, with `k` the parameters including the intercept).

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
    # Entity effects are independent of X here -- what Random Effects needs to be efficient -- see get_hausman_test.
    alpha = {entity: rng.uniform(-5, 5) for entity in entities}
    index = pd.MultiIndex.from_product([entities, times], names=["entity", "time"])

    x = pd.Series(rng.standard_normal(len(index)), index=index, name="X")
    y = pd.Series(
        [alpha[entity] + 2.0 * x.loc[entity, time] for entity, time in index],
        index=index,
    ) + rng.standard_normal(len(index)) * 0.1

    result = panel_data_model.get_random_effects(y, x)
    regression_model.regression_summary_table(result).round(4)
    ```

    Which returns:

    |           |   Coefficient |   Std. Error |   t-Statistic |   P-Value |
    |:----------|--------------:|-------------:|--------------:|----------:|
    | Intercept |        1.4022 |       1.1402 |        1.2298 |      0.22 |
    | X         |        1.9916 |       0.0072 |      277.178  |      0    |
    """
    y_panel = _to_panel_series(y, "y")
    x_panel = _to_panel_frame(x, "x")
    y_panel, x_panel = _align_panel(y_panel, x_panel)

    n = len(y_panel)
    k = x_panel.shape[1]
    n_entities = y_panel.index.get_level_values("entity").nunique()

    if n - k - 1 <= 0:
        raise ValueError(
            f"Not enough observations ({n}) to estimate {k} parameter(s) plus an "
            "intercept."
        )
    # The Swamy-Arora between-entity variance divides the between regression's residual sum
    # of squares by N - (k + 1), so it is undefined unless there are more entities than parameters.  # noqa: E501
    if n_entities <= k + 1:
        raise ValueError(
            f"Random Effects needs more entities ({n_entities}) than parameters "
            f"({k + 1}, including the intercept) to estimate the between-entity variance."
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
    chi-squared distributed with degrees of freedom equal to the rank of
    `Var(b_FE) - Var(b_RE)`: the number of coefficients compared, unless that
    difference is singular, in which case its pseudo-inverse is used. A significant result rejects that null -- i.e.
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
        there are not enough observations (or entities, see `get_random_effects`) to
        fit either model, the two estimators share no common regressors to compare,
        or `Var(b_FE) - Var(b_RE)` has rank 0.

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

    # Entity effects are a noisy function of X's mean, violating RE exogeneity, so the Hausman test below should reject.
    entity_mean_x = x.groupby(level="entity").mean()
    eta = pd.Series(rng.normal(0, 0.3, len(entities)), index=entities)
    alpha = 0.5 * entity_mean_x + eta

    y = pd.Series(
        [alpha[entity] + 2.0 * x.loc[entity, time] for entity, time in index],
        index=index,
    ) + rng.standard_normal(len(index)) * 0.6

    result = panel_data_model.get_hausman_test(y, x)
    result.round(4)
    ```

    Which returns:

    |                           |       0 |
    |:--------------------------|--------:|
    | Hausman Statistic         | 134.396 |
    | Degrees of Freedom        |   1     |
    | P-Value                   |   0     |
    | Prefer Fixed Effects (5%) |   1     |

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

    # The chi-squared degrees of freedom are the rank of Var(b_FE) - Var(b_RE), which equals
    # the number of compared coefficients unless the difference is singular (Hausman, 1978).  # noqa: E501
    degrees_of_freedom = int(np.linalg.matrix_rank(variance_difference))
    if degrees_of_freedom == 0:
        raise ValueError(
            "Var(b_FE) - Var(b_RE) has rank 0, so the Hausman statistic is undefined."
        )

    # A rank-deficient difference uses the pseudo-inverse with the same rank as the degrees
    # of freedom, as Stata does: np.linalg.inv often succeeds on a numerically singular
    # matrix and returns an inverse dominated by rounding noise.  # noqa: E501
    if degrees_of_freedom < len(common_names):
        # The cut-off matches np.linalg.matrix_rank, so pinv keeps exactly
        # `degrees_of_freedom` singular values.
        tolerance = max(variance_difference.shape) * np.finfo(float).eps
        inverse_variance_difference = np.linalg.pinv(
            variance_difference, rcond=tolerance
        )
    else:
        inverse_variance_difference = np.linalg.inv(variance_difference)

    statistic = float(difference @ inverse_variance_difference @ difference)
    statistic = max(statistic, 0.0)
    p_value = float(stats.chi2.sf(statistic, degrees_of_freedom))

    return pd.Series(
        {
            "Hausman Statistic": statistic,
            "Degrees of Freedom": degrees_of_freedom,
            "P-Value": p_value,
            "Prefer Fixed Effects (5%)": bool(p_value < SIGNIFICANCE_LEVEL),
        }
    )
