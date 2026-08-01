"""Panel Data Model Tests"""

import numpy as np
import pandas as pd

from financetoolkit.econometrics import panel_data_model, regression_model

# pylint: disable=missing-function-docstring


def _make_panel(
    n_entities: int,
    n_times: int,
    beta: float,
    alpha: dict,
    seed: int,
    noise_std: float = 0.1,
) -> tuple[pd.Series, pd.Series]:
    """Builds a balanced (entity, time)-indexed panel y = alpha_i + beta * x_it + e_it."""
    rng = np.random.default_rng(seed)
    entities = [f"E{i}" for i in range(n_entities)]
    times = range(n_times)
    index = pd.MultiIndex.from_product([entities, times], names=["entity", "time"])

    x = pd.Series(rng.standard_normal(len(index)), index=index, name="X")
    y = (
        pd.Series([alpha[entity] for entity, _ in index], index=index)
        + beta * x
        + rng.standard_normal(len(index)) * noise_std
    )
    return y, x


def test_get_fixed_effects_recovers_known_beta(recorder):
    alpha = {
        f"E{i}": rng_value
        for i, rng_value in enumerate([2.7, -0.6, 3.6, 2.0, -4.1, 4.8])
    }
    y, x = _make_panel(6, 40, beta=2.0, alpha=alpha, seed=42)

    result = panel_data_model.get_fixed_effects(y, x)

    assert abs(result["regression"]["coefficients"][0] - 2.0) < 0.05
    recorder.capture(
        regression_model.regression_summary_table(result["regression"]).round(4)
    )


def test_get_fixed_effects_recovers_entity_intercepts():
    alpha = {"E0": 2.7, "E1": -0.6, "E2": 3.6, "E3": 2.0, "E4": -4.1, "E5": 4.8}
    y, x = _make_panel(6, 40, beta=2.0, alpha=alpha, seed=42)

    result = panel_data_model.get_fixed_effects(y, x)

    assert result["entity_effects"] is not None
    for entity, true_alpha in alpha.items():
        assert abs(result["entity_effects"][entity] - true_alpha) < 0.2


def test_get_fixed_effects_is_invariant_to_true_alpha_values():
    # The whole point of Fixed Effects: however wildly the true entity intercepts
    # differ, the recovered slope beta should barely move.
    y, x = _make_panel(6, 40, beta=2.0, alpha={f"E{i}": 0.0 for i in range(6)}, seed=1)
    y_shifted, x_shifted = _make_panel(
        6, 40, beta=2.0, alpha={f"E{i}": (i - 2) * 1000.0 for i in range(6)}, seed=1
    )

    result = panel_data_model.get_fixed_effects(y, x)
    result_shifted = panel_data_model.get_fixed_effects(y_shifted, x_shifted)

    assert (
        abs(
            result["regression"]["coefficients"][0]
            - result_shifted["regression"]["coefficients"][0]
        )
        < 1e-8
    )


def test_get_fixed_effects_matches_dummy_variable_ols():
    # Fixed Effects (the within/demeaned estimator) is algebraically identical to
    # running OLS with one dummy regressor per entity (LSDV) -- verify that exact
    # identity by hand on a small dataset, without any panel-data library.
    rng = np.random.default_rng(3)
    entities = ["A", "B", "C"]
    times = range(8)
    index = pd.MultiIndex.from_product([entities, times], names=["entity", "time"])
    alpha = {"A": 1.0, "B": -2.0, "C": 5.0}
    x = pd.Series(rng.standard_normal(len(index)), index=index, name="X")
    y = (
        pd.Series([alpha[entity] for entity, _ in index], index=index)
        + 1.5 * x
        + rng.standard_normal(len(index)) * 0.05
    )

    fe_result = panel_data_model.get_fixed_effects(y, x)

    # LSDV: OLS of y on [dummy_A, dummy_B, dummy_C, x] (no separate intercept).
    dummies = pd.get_dummies(
        pd.Series([entity for entity, _ in index], index=index), dtype=float
    )
    design = np.column_stack([dummies.to_numpy(), x.to_numpy()])
    lsdv_coefficients, _, _, _ = np.linalg.lstsq(design, y.to_numpy(), rcond=None)

    assert np.allclose(
        fe_result["regression"]["coefficients"][0], lsdv_coefficients[-1], atol=1e-8
    )

    # The recovered entity intercepts should also match the LSDV dummy coefficients.
    for i, entity in enumerate(entities):
        assert abs(fe_result["entity_effects"][entity] - lsdv_coefficients[i]) < 1e-8


def test_get_fixed_effects_degrees_of_freedom_correction():
    # A naive OLS on the demeaned data (without correcting for the absorbed entity
    # dummies) would report n - k residual degrees of freedom; Fixed Effects must
    # additionally subtract the number of entities.
    y, x = _make_panel(
        6, 40, beta=2.0, alpha={f"E{i}": float(i) for i in range(6)}, seed=7
    )

    result = panel_data_model.get_fixed_effects(y, x)

    n = 6 * 40
    assert result["regression"]["degrees_of_freedom"] == n - 1 - 6


def test_get_fixed_effects_two_way():
    rng = np.random.default_rng(11)
    entities = [f"E{i}" for i in range(5)]
    times = range(20)
    index = pd.MultiIndex.from_product([entities, times], names=["entity", "time"])
    entity_alpha = {entity: rng.uniform(-3, 3) for entity in entities}
    time_gamma = {t: rng.uniform(-1, 1) for t in times}

    x = pd.Series(rng.standard_normal(len(index)), index=index, name="X")
    y = (
        pd.Series([entity_alpha[e] + time_gamma[t] for e, t in index], index=index)
        + 1.5 * x
        + rng.standard_normal(len(index)) * 0.05
    )

    result = panel_data_model.get_fixed_effects(
        y, x, entity_effects=True, time_effects=True
    )

    assert abs(result["regression"]["coefficients"][0] - 1.5) < 0.05
    assert result["entity_effects"] is not None
    assert result["time_effects"] is not None
    assert len(result["time_effects"]) == 20


def test_get_fixed_effects_accepts_wide_dataframe():
    y, x = _make_panel(
        5, 15, beta=1.0, alpha={f"E{i}": float(i) for i in range(5)}, seed=5
    )
    y_wide = y.unstack("time")
    x_wide = x.unstack("time")

    result_multiindex = panel_data_model.get_fixed_effects(y, x)
    result_wide = panel_data_model.get_fixed_effects(y_wide, x_wide)

    assert np.allclose(
        result_multiindex["regression"]["coefficients"],
        result_wide["regression"]["coefficients"],
    )


def test_get_fixed_effects_requires_at_least_one_effect():
    y, x = _make_panel(4, 10, beta=1.0, alpha={f"E{i}": 0.0 for i in range(4)}, seed=1)
    try:
        panel_data_model.get_fixed_effects(
            y, x, entity_effects=False, time_effects=False
        )
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_fixed_effects_invalid_type():
    try:
        panel_data_model.get_fixed_effects([1, 2, 3], [1, 2, 3])
        raise AssertionError("Expected TypeError")
    except TypeError:
        pass


def test_get_fixed_effects_too_few_observations():
    # 2 entities, 1 observation each: 1 regressor + 2 absorbed entity intercepts
    # cannot be estimated from only 2 observations.
    index = pd.MultiIndex.from_tuples([("A", 0), ("B", 0)], names=["entity", "time"])
    y = pd.Series([1.0, 2.0], index=index)
    x = pd.Series([1.0, 2.0], index=index)
    try:
        panel_data_model.get_fixed_effects(y, x)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_fixed_effects_perfect_within_fit_raises():
    # Exactly 2 observations per entity: the within-transformed data has exactly
    # one usable data point per entity, so the fit is a perfect fit with zero
    # residual variance. `linearmodels.panel.PanelOLS` itself divides by zero
    # computing its internal pooled F-statistic post-estimation diagnostic for
    # this degenerate case, which this wraps into a ValueError.
    index = pd.MultiIndex.from_tuples(
        [("A", 0), ("A", 1), ("B", 0), ("B", 1)], names=["entity", "time"]
    )
    y = pd.Series([1.0, 2.0, 3.0, 4.0], index=index)
    x = pd.Series([1.0, 2.0, 3.0, 4.0], index=index)

    try:
        panel_data_model.get_fixed_effects(y, x)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_random_effects_recovers_known_beta(recorder):
    alpha = {
        f"E{i}": rng_value
        for i, rng_value in enumerate([2.7, -0.6, 3.6, 2.0, -4.1, 4.8])
    }
    y, x = _make_panel(6, 40, beta=2.0, alpha=alpha, seed=42)

    result = panel_data_model.get_random_effects(y, x)

    assert result["feature_names"] == ["Intercept", "X"]
    assert abs(result["coefficients"][1] - 2.0) < 0.05
    recorder.capture(regression_model.regression_summary_table(result).round(4))


def test_get_random_effects_matches_pooled_ols_when_no_entity_variance():
    # If entity effects are identical across entities (zero between-entity
    # variance), Random Effects' theta should collapse to 0 (no demeaning at all),
    # reducing exactly to pooled OLS on the raw (undemeaned) panel.
    y, x = _make_panel(
        10, 20, beta=1.5, alpha={f"E{i}": 0.0 for i in range(10)}, seed=9
    )

    re_result = panel_data_model.get_random_effects(y, x)

    from financetoolkit.econometrics.regression_model import get_ols

    pooled = get_ols(y.to_numpy(), x.to_numpy(), add_constant=True)

    assert np.allclose(re_result["coefficients"], pooled["coefficients"], atol=0.05)


def test_get_random_effects_invalid_type():
    try:
        panel_data_model.get_random_effects([1, 2, 3], [1, 2, 3])
        raise AssertionError("Expected TypeError")
    except TypeError:
        pass


def test_get_random_effects_not_enough_entities():
    index = pd.MultiIndex.from_product(
        [["A", "B"], range(20)], names=["entity", "time"]
    )
    y = pd.Series(np.random.default_rng(1).standard_normal(40), index=index)
    x = pd.Series(np.random.default_rng(2).standard_normal(40), index=index)
    try:
        panel_data_model.get_random_effects(y, x)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_hausman_test_fails_to_reject_when_exogenous(recorder):
    # Entity effects are drawn independently of X -- Random Effects' exogeneity
    # assumption holds, so the Hausman test should NOT reject at the 5% level.
    rng = np.random.default_rng(1)
    entities = [f"E{i}" for i in range(300)]
    times = range(8)
    index = pd.MultiIndex.from_product([entities, times], names=["entity", "time"])
    x = pd.Series(rng.standard_normal(len(index)), index=index, name="X")

    eta = pd.Series(rng.normal(0, 0.3, len(entities)), index=entities)
    alpha = {entity: eta[entity] for entity in entities}

    y = (
        pd.Series(
            [alpha[entity] + 2.0 * x.loc[entity, time] for entity, time in index],
            index=index,
        )
        + rng.standard_normal(len(index)) * 0.6
    )

    result = panel_data_model.get_hausman_test(y, x)

    assert result["P-Value"] > 0.05
    assert result["Prefer Fixed Effects (5%)"] is False
    recorder.capture(result.round(4))


def test_get_hausman_test_rejects_when_endogenous(recorder):
    # Entity effects are a function of X's own entity-mean (plus independent
    # noise) -- this violates Random Effects' exogeneity assumption, so the
    # Hausman test should reject at the 5% level, correctly flagging that Fixed
    # Effects should be preferred.
    rng = np.random.default_rng(1)
    entities = [f"E{i}" for i in range(300)]
    times = range(8)
    index = pd.MultiIndex.from_product([entities, times], names=["entity", "time"])
    x = pd.Series(rng.standard_normal(len(index)), index=index, name="X")

    entity_mean_x = x.groupby(level="entity").mean()
    eta = pd.Series(rng.normal(0, 0.3, len(entities)), index=entities)
    alpha = 0.5 * entity_mean_x + eta

    y = (
        pd.Series(
            [alpha[entity] + 2.0 * x.loc[entity, time] for entity, time in index],
            index=index,
        )
        + rng.standard_normal(len(index)) * 0.6
    )

    result = panel_data_model.get_hausman_test(y, x)

    assert result["P-Value"] < 0.05
    assert result["Prefer Fixed Effects (5%)"] is True
    assert result["Hausman Statistic"] > 0
    recorder.capture(result.round(4))


def test_get_hausman_test_degrees_of_freedom_matches_common_regressors():
    y, x = _make_panel(
        6, 30, beta=1.0, alpha={f"E{i}": float(i) for i in range(6)}, seed=4
    )
    result = panel_data_model.get_hausman_test(y, x)

    assert result["Degrees of Freedom"] == 1
