"""Copula (Tail Dependence) Model Tests"""

import numpy as np
import pandas as pd

from financetoolkit.risk import copula_model

# pylint: disable=missing-function-docstring

INDEPENDENCE_TOLERANCE = 0.02
STUDENT_T_TAIL_DEPENDENCE_MINIMUM = 0.3
STUDENT_T_GAUSSIAN_AGREEMENT_TOLERANCE = 0.1


def test_get_tail_dependence_coefficient_independent(recorder):
    rng = np.random.default_rng(3)
    n = 20000
    a = pd.Series(rng.standard_normal(n))
    b = pd.Series(rng.standard_normal(n))
    result = copula_model.get_tail_dependence_coefficient(a, b, q=0.95)
    # Under independence, P(both extreme | one extreme) should equal roughly (1 - q).
    assert abs(result["Lower Tail Dependence"] - 0.05) < INDEPENDENCE_TOLERANCE
    assert abs(result["Upper Tail Dependence"] - 0.05) < INDEPENDENCE_TOLERANCE
    recorder.capture(result.round(4))


def test_get_tail_dependence_coefficient_comonotonic(recorder):
    rng = np.random.default_rng(3)
    a = pd.Series(rng.standard_normal(5000))
    result = copula_model.get_tail_dependence_coefficient(a, a, q=0.95)
    assert result["Lower Tail Dependence"] == 1.0
    assert result["Upper Tail Dependence"] == 1.0
    recorder.capture(result.round(4))


def test_get_tail_dependence_coefficient_gaussian_method(recorder):
    rng = np.random.default_rng(3)
    n = 5000
    cov = [[1, 0.9], [0.9, 1]]
    mvn = rng.multivariate_normal([0, 0], cov, size=n)
    a, b = pd.Series(mvn[:, 0]), pd.Series(mvn[:, 1])
    result = copula_model.get_tail_dependence_coefficient(a, b, method="gaussian")
    # A gaussian copula has zero asymptotic tail dependence regardless of correlation.
    assert result["Lower Tail Dependence"] == 0.0
    assert result["Upper Tail Dependence"] == 0.0
    recorder.capture(result.round(4))


def test_get_tail_dependence_coefficient_student_t_method(recorder):
    rng = np.random.default_rng(3)
    n = 5000
    cov = [[1, 0.9], [0.9, 1]]
    mvn = rng.multivariate_normal([0, 0], cov, size=n)
    v = 4
    w = rng.chisquare(v, size=n)
    ta = pd.Series(mvn[:, 0] / np.sqrt(w / v))
    tb = pd.Series(mvn[:, 1] / np.sqrt(w / v))

    result_empirical = copula_model.get_tail_dependence_coefficient(
        ta, tb, q=0.99, method="empirical"
    )
    result_formula = copula_model.get_tail_dependence_coefficient(
        ta, tb, q=0.99, method="student-t", dof=v
    )
    # A Student-T copula has nonzero tail dependence, unlike the gaussian one, and
    # the closed-form value should roughly agree with the empirical estimate.
    assert result_formula["Upper Tail Dependence"] > STUDENT_T_TAIL_DEPENDENCE_MINIMUM
    assert (
        abs(
            result_formula["Upper Tail Dependence"]
            - result_empirical["Upper Tail Dependence"]
        )
        < STUDENT_T_GAUSSIAN_AGREEMENT_TOLERANCE
    )
    recorder.capture(result_formula.round(4))


def test_get_tail_dependence_coefficient_invalid_type():
    try:
        copula_model.get_tail_dependence_coefficient(1, pd.Series([1.0, 2.0]))  # type: ignore
        raise AssertionError("Expected TypeError")
    except TypeError:
        pass


def test_get_tail_dependence_coefficient_invalid_method():
    a = pd.Series([1.0, 2.0, 3.0])
    try:
        copula_model.get_tail_dependence_coefficient(a, a, method="bad")
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_tail_dependence_coefficient_invalid_q():
    a = pd.Series([1.0, 2.0, 3.0])
    try:
        copula_model.get_tail_dependence_coefficient(a, a, q=0.4)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_tail_dependence_coefficient_too_few_observations(recorder):
    a = pd.Series([1.0, 2.0])
    b = pd.Series([1.0, 2.0])
    recorder.capture(copula_model.get_tail_dependence_coefficient(a, b))
