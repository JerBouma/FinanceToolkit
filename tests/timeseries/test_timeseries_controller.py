"""Time Series Controller Tests"""

# pylint: disable=missing-function-docstring


def test_get_augmented_dickey_fuller(recorder, timeseries_module):
    recorder.capture(timeseries_module.get_augmented_dickey_fuller(period="quarterly"))
    recorder.capture(
        timeseries_module.get_augmented_dickey_fuller(period="monthly", regression="n")
    )


def test_get_engle_granger_cointegration(recorder, timeseries_module):
    recorder.capture(
        timeseries_module.get_engle_granger_cointegration(
            "AAPL", "MSFT", period="quarterly"
        )
    )


def test_get_granger_causality(recorder, timeseries_module):
    recorder.capture(
        timeseries_module.get_granger_causality(
            "AAPL", "MSFT", period="weekly", max_lag=3
        )
    )
