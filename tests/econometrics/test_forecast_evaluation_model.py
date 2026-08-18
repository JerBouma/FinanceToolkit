"""Forecast Evaluation Model Tests"""

import numpy as np
import pandas as pd

from financetoolkit.econometrics import forecast_evaluation_model

# pylint: disable=missing-function-docstring

SIGNIFICANCE_LEVEL = 0.05


def test_get_diebold_mariano_test_better_forecast(recorder):
    rng = np.random.default_rng(9)
    n = 500
    actual = rng.standard_normal(n)
    forecast_good = actual + rng.standard_normal(n) * 0.1
    forecast_bad = actual + rng.standard_normal(n) * 1.0
    result = forecast_evaluation_model.get_diebold_mariano_test(
        pd.Series(actual), pd.Series(forecast_good), pd.Series(forecast_bad)
    )
    assert result["Diebold-Mariano Statistic"] < 0
    assert result["P-Value"] < SIGNIFICANCE_LEVEL
    recorder.capture(result.round(4))


def test_get_diebold_mariano_test_equal_forecasts(recorder):
    rng = np.random.default_rng(9)
    n = 500
    actual = rng.standard_normal(n)
    forecast_a = actual + rng.standard_normal(n) * 0.5
    forecast_b = actual + rng.standard_normal(n) * 0.5
    recorder.capture(
        forecast_evaluation_model.get_diebold_mariano_test(
            pd.Series(actual), pd.Series(forecast_a), pd.Series(forecast_b)
        ).round(4)
    )


def test_get_diebold_mariano_test_dataframe(recorder):
    rng = np.random.default_rng(9)
    n = 500
    actual = rng.standard_normal(n)
    forecast_good = actual + rng.standard_normal(n) * 0.1
    forecast_bad = actual + rng.standard_normal(n) * 1.0
    actual_df = pd.DataFrame({"AAPL": actual, "MSFT": actual})
    forecast_a_df = pd.DataFrame({"AAPL": forecast_good, "MSFT": forecast_good})
    forecast_b_df = pd.DataFrame({"AAPL": forecast_bad, "MSFT": forecast_bad})
    recorder.capture(
        forecast_evaluation_model.get_diebold_mariano_test(
            actual_df, forecast_a_df, forecast_b_df
        ).round(4)
    )


def test_get_diebold_mariano_test_absolute_loss(recorder):
    rng = np.random.default_rng(9)
    n = 500
    actual = rng.standard_normal(n)
    forecast_a = actual + rng.standard_normal(n) * 0.2
    forecast_b = actual + rng.standard_normal(n) * 0.8
    recorder.capture(
        forecast_evaluation_model.get_diebold_mariano_test(
            pd.Series(actual),
            pd.Series(forecast_a),
            pd.Series(forecast_b),
            loss="absolute",
        ).round(4)
    )


def test_get_diebold_mariano_test_invalid_loss():
    try:
        forecast_evaluation_model.get_diebold_mariano_test(
            pd.Series([1.0, 2.0]),
            pd.Series([1.0, 2.0]),
            pd.Series([1.0, 2.0]),
            loss="bad",
        )
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_diebold_mariano_test_too_few_observations(recorder):
    actual = pd.Series([1.0, 2.0, 3.0])
    forecast_a = pd.Series([1.1, 2.1, 3.1])
    forecast_b = pd.Series([0.9, 1.9, 2.9])
    recorder.capture(
        forecast_evaluation_model.get_diebold_mariano_test(
            actual, forecast_a, forecast_b
        )
    )


def test_get_rmse_hand_computed():
    # actual - forecast = [-1, 1, -2, 1] -> squared = [1, 1, 4, 1] -> mean 1.75
    actual = pd.Series([10.0, 12.0, 11.0, 13.0])
    forecast = pd.Series([11.0, 11.0, 13.0, 12.0])
    rmse = forecast_evaluation_model.get_rmse(actual, forecast)
    assert abs(rmse - np.sqrt(1.75)) < 1e-9


def test_get_rmse_constant_error():
    # A constant error of 2 everywhere gives RMSE = 2 exactly.
    actual = pd.Series([1.0, 2.0, 3.0, 4.0])
    forecast = actual - 2.0
    assert abs(forecast_evaluation_model.get_rmse(actual, forecast) - 2.0) < 1e-9


def test_get_rmse_zero_for_perfect_forecast():
    actual = pd.Series([1.0, 2.0, 3.0])
    assert forecast_evaluation_model.get_rmse(actual, actual) == 0.0


def test_get_rmse_dataframe():
    actual = pd.DataFrame({"A": [10.0, 12.0], "B": [1.0, 2.0]})
    forecast = pd.DataFrame({"A": [11.0, 11.0], "B": [1.0, 2.0]})
    result = forecast_evaluation_model.get_rmse(actual, forecast)
    assert isinstance(result, pd.Series)
    assert abs(result["A"] - np.sqrt((1**2 + 1**2) / 2)) < 1e-9
    assert result["B"] == 0.0


def test_get_rmse_invalid_type():
    try:
        forecast_evaluation_model.get_rmse([1, 2, 3], [1, 2, 3])
        raise AssertionError("Expected TypeError")
    except TypeError:
        pass


def test_get_mae_hand_computed():
    # |actual - forecast| = [1, 1, 2, 1] -> mean 1.25
    actual = pd.Series([10.0, 12.0, 11.0, 13.0])
    forecast = pd.Series([11.0, 11.0, 13.0, 12.0])
    mae = forecast_evaluation_model.get_mae(actual, forecast)
    assert abs(mae - 1.25) < 1e-9


def test_get_mae_constant_error():
    actual = pd.Series([1.0, 2.0, 3.0, 4.0])
    forecast = actual - 3.0
    assert abs(forecast_evaluation_model.get_mae(actual, forecast) - 3.0) < 1e-9


def test_get_mae_zero_for_perfect_forecast():
    actual = pd.Series([1.0, 2.0, 3.0])
    assert forecast_evaluation_model.get_mae(actual, actual) == 0.0


def test_get_mae_dataframe():
    actual = pd.DataFrame({"A": [10.0, 12.0], "B": [1.0, 2.0]})
    forecast = pd.DataFrame({"A": [11.0, 11.0], "B": [1.0, 2.0]})
    result = forecast_evaluation_model.get_mae(actual, forecast)
    assert isinstance(result, pd.Series)
    assert abs(result["A"] - 1.0) < 1e-9
    assert result["B"] == 0.0


def test_get_mae_invalid_type():
    try:
        forecast_evaluation_model.get_mae([1, 2, 3], [1, 2, 3])
        raise AssertionError("Expected TypeError")
    except TypeError:
        pass


def _naive_last_value_forecast(train: pd.Series, forecast_steps: int) -> pd.Series:
    """A trivial dummy forecaster used to hand-verify the out-of-sample harness: it
    just repeats the last training observation `forecast_steps` times."""
    return pd.Series([train.iloc[-1]] * forecast_steps)


def test_get_out_of_sample_validation_no_leakage():
    # Confirm the split is chronological and training excludes every holdout index.
    series = pd.Series(np.arange(20.0), index=pd.RangeIndex(20))
    result = forecast_evaluation_model.get_out_of_sample_validation(
        series, _naive_last_value_forecast, train_fraction=0.8
    )
    assert result["Holdout Observations"] == 4

    split = int(len(series) * 0.8)
    train_index = set(series.index[:split])
    holdout_index = set(series.index[split:])
    assert train_index.isdisjoint(holdout_index)
    assert len(train_index) + len(holdout_index) == len(series)


def test_get_out_of_sample_validation_naive_forecaster_hand_computed():
    # train=[0..15], holdout=[16..19]; repeating 15 gives errors [1, 2, 3, 4].
    series = pd.Series(np.arange(20.0))
    result = forecast_evaluation_model.get_out_of_sample_validation(
        series, _naive_last_value_forecast, train_fraction=0.8
    )

    expected_errors = np.array([1.0, 2.0, 3.0, 4.0])
    expected_rmse = np.sqrt(np.mean(expected_errors**2))
    expected_mae = np.mean(expected_errors)

    assert abs(result["RMSE"] - expected_rmse) < 1e-9
    assert abs(result["MAE"] - expected_mae) < 1e-9
    assert result["Holdout Observations"] == 4


def test_get_out_of_sample_validation_plugged_into_arima(recorder):
    # End-to-end wiring check against a real Part-1 forecasting function.
    from financetoolkit.econometrics import time_series_model

    rng = np.random.default_rng(1)
    n = 200
    e = rng.standard_normal(n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.6 * y[t - 1] + e[t]

    result = forecast_evaluation_model.get_out_of_sample_validation(
        pd.Series(y),
        time_series_model.get_arima_forecast,
        train_fraction=0.9,
        p=1,
        d=0,
        q=0,
    )
    assert result["Holdout Observations"] == 20
    assert result["RMSE"] > 0
    assert result["MAE"] > 0
    recorder.capture(result.round(4))


def test_get_out_of_sample_validation_invalid_type():
    try:
        forecast_evaluation_model.get_out_of_sample_validation(
            [1, 2, 3], _naive_last_value_forecast
        )
        raise AssertionError("Expected TypeError")
    except TypeError:
        pass


def test_get_out_of_sample_validation_invalid_train_fraction():
    series = pd.Series(np.arange(20.0))
    try:
        forecast_evaluation_model.get_out_of_sample_validation(
            series, _naive_last_value_forecast, train_fraction=1.5
        )
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_out_of_sample_validation_mismatched_forecast_length():
    def _wrong_length_forecaster(train: pd.Series, forecast_steps: int) -> pd.Series:
        return pd.Series([0.0] * (forecast_steps - 1))

    series = pd.Series(np.arange(20.0))
    try:
        forecast_evaluation_model.get_out_of_sample_validation(
            series, _wrong_length_forecaster, train_fraction=0.8
        )
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass
