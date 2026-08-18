"""Recorder Comparison Tests"""

# pylint: disable=missing-function-docstring

from tests.conftest import Record

HEADER = ",AAPL\n"


def compare(recorded: str, captured: str) -> bool:
    return Record._csv_strings_approx_equal(  # noqa: SLF001
        HEADER + recorded, HEADER + captured
    )


def test_last_recorded_digit_may_drift():
    # A value stored with four decimals tolerates a different fourth decimal, which is
    # what a solver landing on the other side of a rounding boundary produces.
    assert compare("Beta,0.3754\n", "Beta,0.3753\n")


def test_full_precision_values_tolerate_float_noise():
    assert compare(
        "ADF Statistic,-1.709327393005512\n", "ADF Statistic,-1.7093273930055113\n"
    )
    assert compare(
        "Entropic Value at Risk,2.891388267928218e+199\n",
        "Entropic Value at Risk,2.8913882767018233e+199\n",
    )
    # A GARCH-family solver reaching a slightly different optimum.
    assert compare("Forecast,224738942717.6637\n", "Forecast,224739396143.0518\n")


def test_an_iterative_forecast_may_drift_a_few_last_digits():
    # ARIMA compounds its own noise the further ahead it forecasts.
    assert compare("4,162.3228\n", "4,162.3231\n")


def test_signed_zero_is_equal():
    assert compare("Put Call Parity,-0.0\n", "Put Call Parity,0.0\n")


def test_infinities_only_match_themselves():
    assert compare("Value at Risk,-inf\n", "Value at Risk,-inf\n")
    assert not compare("Value at Risk,-inf\n", "Value at Risk,inf\n")


def test_a_real_change_still_fails():
    assert not compare("Beta,0.3754\n", "Beta,0.3812\n")
    assert not compare(
        "ADF Statistic,-1.709327393005512\n", "ADF Statistic,-1.719327393005512\n"
    )


def test_text_must_match_exactly():
    assert not compare(
        "Interpretation,Negative Autocorrelation Likely\n",
        "Interpretation,Positive Autocorrelation Likely\n",
    )


def test_shape_and_headers_must_match_exactly():
    assert not Record._csv_strings_approx_equal(  # noqa: SLF001
        ",AAPL\nBeta,0.3754\n", ",MSFT\nBeta,0.3754\n"
    )
    assert not Record._csv_strings_approx_equal(  # noqa: SLF001
        ",AAPL\nBeta,0.3754\n", ",AAPL\nBeta,0.3754\nAlpha,0.1\n"
    )
    assert not Record._csv_strings_approx_equal(  # noqa: SLF001
        ",AAPL\nBeta,0.3754\n", ",AAPL,MSFT\nBeta,0.3754,0.2\n"
    )
