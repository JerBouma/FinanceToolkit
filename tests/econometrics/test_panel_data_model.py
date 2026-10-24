"""Panel Data Model Tests"""

import numpy as np
import pandas as pd
import pytest

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
    # However wildly the true entity intercepts differ, the slope should barely move.
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
    # Fixed Effects is algebraically identical to OLS with one dummy per entity.
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
    # Fixed Effects must subtract the entities on top of the n - k residual df.
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
    # 1 regressor plus 2 absorbed intercepts cannot come from only 2 observations.
    index = pd.MultiIndex.from_tuples([("A", 0), ("B", 0)], names=["entity", "time"])
    y = pd.Series([1.0, 2.0], index=index)
    x = pd.Series([1.0, 2.0], index=index)
    try:
        panel_data_model.get_fixed_effects(y, x)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_fixed_effects_perfect_within_fit_raises():
    # Two per entity is a perfect fit, and PanelOLS then divides by zero internally.
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
    # Zero between-entity variance collapses theta to 0, reducing to pooled OLS.
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
    # Entity effects drawn independently of X, so Hausman should not reject at 5%.
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
    # Entity effects depend on X's entity-mean, so Hausman should reject at 5%.
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


def test_get_fixed_effects_two_way_effects_count_the_level_once():
    # The entity plus time effects must equal linearmodels' combined estimated effects,
    # balanced or not, so the grand mean of y - X @ beta is not counted twice.
    from linearmodels.panel import PanelOLS

    rng = np.random.default_rng(3)
    index = pd.MultiIndex.from_product(
        [[f"E{i}" for i in range(5)], range(30)], names=["entity", "time"]
    )
    x = pd.Series(rng.standard_normal(len(index)), index=index, name="X")
    y = pd.Series(rng.standard_normal(len(index)), index=index) + 2 * x + 3

    for keep in (np.ones(len(index), dtype=bool), rng.random(len(index)) > 0.2):
        y_kept, x_kept = y[keep], x[keep]
        result = panel_data_model.get_fixed_effects(
            y_kept, x_kept, entity_effects=True, time_effects=True
        )
        reference = PanelOLS(
            y_kept, x_kept.to_frame(), entity_effects=True, time_effects=True
        ).fit()

        combined = (
            result["entity_effects"]
            .reindex(y_kept.index.get_level_values("entity"))
            .to_numpy()
            + result["time_effects"]
            .reindex(y_kept.index.get_level_values("time"))
            .to_numpy()
        )
        np.testing.assert_allclose(
            combined, reference.estimated_effects.to_numpy().ravel(), atol=1e-9
        )
        time_effect_per_observation = result["time_effects"].reindex(
            y_kept.index.get_level_values("time")
        )
        assert abs(time_effect_per_observation.mean()) < 1e-9


def test_get_fixed_effects_time_effects_absorbing_a_common_factor_raises():
    rng = np.random.default_rng(3)
    index = pd.MultiIndex.from_product(
        [[f"E{i}" for i in range(4)], range(30)], names=["entity", "time"]
    )
    common_factor = pd.Series(
        np.tile(rng.standard_normal(30), 4), index=index, name="X"
    )
    y = pd.Series(rng.standard_normal(len(index)), index=index) + common_factor

    with pytest.raises(ValueError, match="absorb"):
        panel_data_model.get_fixed_effects(
            y, common_factor, entity_effects=True, time_effects=True
        )


def test_get_random_effects_and_hausman_need_more_entities_than_parameters():
    rng = np.random.default_rng(3)
    index = pd.MultiIndex.from_product(
        [["A", "B"], range(50)], names=["entity", "time"]
    )
    x = pd.DataFrame(rng.standard_normal((100, 2)), index=index, columns=["a", "b"])
    y = pd.Series(rng.standard_normal(100), index=index)

    with pytest.raises(ValueError, match="more entities"):
        panel_data_model.get_random_effects(y, x)
    with pytest.raises(ValueError, match="more entities"):
        panel_data_model.get_hausman_test(y, x)


def test_get_hausman_test_degrees_of_freedom_is_the_rank_of_the_variance_difference(
    monkeypatch,
):
    # Two compared slopes whose variance difference has rank 1, so the statistic is
    # chi-squared with 1 degree of freedom rather than 2.
    fe = {
        "feature_names": ["a", "b"],
        "coefficients": np.array([1.2, 0.5]),
        "covariance_matrix": np.array([[0.02, 0.01], [0.01, 0.02]]),
    }
    re = {
        "feature_names": ["Intercept", "a", "b"],
        "coefficients": np.array([0.0, 1.0, 0.5]),
        "covariance_matrix": np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]]
        ),
    }
    monkeypatch.setattr(
        panel_data_model, "get_fixed_effects", lambda *_, **__: {"regression": fe}
    )
    monkeypatch.setattr(panel_data_model, "get_random_effects", lambda *_, **__: re)

    result = panel_data_model.get_hausman_test(None, None)

    assert result["Degrees of Freedom"] == 1
    # pinv([[0.01, 0.01], [0.01, 0.01]]) = 25 * ones, and the difference is (0.2, 0).
    np.testing.assert_allclose(result["Hausman Statistic"], 25 * 0.2**2)


@pytest.mark.parametrize(
    ("absorbed_by", "entity_effects", "time_effects"),
    [("time", True, True), ("time", False, True), ("entity", True, False)],
)
def test_get_fixed_effects_single_absorbed_regressor_raises(
    absorbed_by, entity_effects, time_effects
):
    # A single absorbed regressor does not make PanelOLS raise: it is demeaned to rounding
    # noise and fitted with a coefficient of 0 and a standard error of order 1e15.
    rng = np.random.default_rng(3)
    n_entities, n_times = 6, 40
    index = pd.MultiIndex.from_product(
        [[f"E{i}" for i in range(n_entities)], range(n_times)],
        names=["entity", "time"],
    )
    if absorbed_by == "time":
        values = np.tile(rng.standard_normal(n_times), n_entities)
    else:
        values = np.repeat(rng.standard_normal(n_entities), n_times)
    x = pd.DataFrame({"m": values}, index=index)
    y = pd.Series(rng.standard_normal(len(index)), index=index)

    with pytest.raises(ValueError, match=r"absorb one or more regressors \(m\)"):
        panel_data_model.get_fixed_effects(
            y, x, entity_effects=entity_effects, time_effects=time_effects
        )

    # Next to a regressor that varies, only the absorbed one is named.
    x["z"] = rng.standard_normal(len(index))
    with pytest.raises(ValueError, match=r"\(m\)"):
        panel_data_model.get_fixed_effects(
            y, x, entity_effects=entity_effects, time_effects=time_effects
        )


def test_get_hausman_test_numerically_singular_difference_uses_the_pseudo_inverse(
    monkeypatch,
):
    # A variance difference that is singular up to rounding: np.linalg.inv succeeds on it
    # and returns an inverse of order 1e16, whereas the pseudo-inverse of the same rank as
    # the degrees of freedom gives the rank-1 statistic.
    variance_difference = np.array([[0.01, 0.01], [0.01, 0.01 + 1e-18]])
    fe = {
        "feature_names": ["a", "b"],
        "coefficients": np.array([1.2, 0.5]),
        "covariance_matrix": variance_difference + 0.01 * np.eye(2),
    }
    re = {
        "feature_names": ["Intercept", "a", "b"],
        "coefficients": np.array([0.0, 1.0, 0.5]),
        "covariance_matrix": np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]]
        ),
    }
    monkeypatch.setattr(
        panel_data_model, "get_fixed_effects", lambda *_, **__: {"regression": fe}
    )
    monkeypatch.setattr(panel_data_model, "get_random_effects", lambda *_, **__: re)

    result = panel_data_model.get_hausman_test(None, None)

    assert result["Degrees of Freedom"] == 1
    np.testing.assert_allclose(result["Hausman Statistic"], 25 * 0.2**2, rtol=1e-6)
