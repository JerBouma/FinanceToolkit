"""Unit Root Model Tests"""

import numpy as np
import pandas as pd

from financetoolkit.timeseries import unitroot_model

# pylint: disable=missing-function-docstring


def test_get_augmented_dickey_fuller_random_walk(recorder):
    rng = np.random.default_rng(1)
    random_walk = pd.Series(np.cumsum(rng.standard_normal(500)))
    recorder.capture(unitroot_model.get_augmented_dickey_fuller(random_walk).round(4))


def test_get_augmented_dickey_fuller_stationary(recorder):
    rng = np.random.default_rng(1)
    n = 500
    values = np.zeros(n)
    for i in range(1, n):
        values[i] = 0.5 * values[i - 1] + rng.standard_normal()
    stationary = pd.Series(values)
    recorder.capture(unitroot_model.get_augmented_dickey_fuller(stationary).round(4))


def test_get_augmented_dickey_fuller_regression_types(recorder):
    rng = np.random.default_rng(1)
    random_walk = pd.Series(np.cumsum(rng.standard_normal(500)))
    recorder.capture(
        unitroot_model.get_augmented_dickey_fuller(random_walk, regression="n").round(4)
    )
    recorder.capture(
        unitroot_model.get_augmented_dickey_fuller(random_walk, regression="ct").round(
            4
        )
    )


def test_get_augmented_dickey_fuller_too_few_observations(recorder):
    series = pd.Series([1.0, 2.0, 3.0])
    recorder.capture(unitroot_model.get_augmented_dickey_fuller(series))


def test_get_augmented_dickey_fuller_invalid_regression():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    try:
        unitroot_model.get_augmented_dickey_fuller(series, regression="invalid")
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass
