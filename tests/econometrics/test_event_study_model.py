"""Event Study Model Tests"""

import numpy as np
import pandas as pd

from financetoolkit.econometrics.causal import event_study_model


def _synthetic_returns(n=300, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    market = pd.Series(rng.normal(0.0005, 0.01, n), index=dates)
    idiosyncratic = rng.normal(0, 0.005, n)
    returns = 0.0002 + 1.2 * market.to_numpy() + idiosyncratic
    returns = pd.Series(returns, index=dates)
    return returns, market


def test_get_event_study(recorder):
    returns, market = _synthetic_returns()
    event_date = returns.index[280]

    result = event_study_model.get_event_study(
        returns=returns,
        market_returns=market,
        event_date=event_date,
        estimation_window=200,
        gap_days=20,
        pre_event_days=5,
        post_event_days=5,
    )

    recorder.capture(round(result["cumulative_abnormal_return"], 6))
    recorder.capture(round(result["alpha"], 6))
    recorder.capture(round(result["beta"], 4))
    recorder.capture(result["estimation_window_observations"])
    recorder.capture(len(result["abnormal_returns"]))


def test_get_event_study_event_date_not_found():
    returns, market = _synthetic_returns()

    try:
        event_study_model.get_event_study(
            returns=returns, market_returns=market, event_date="1999-01-01"
        )
        raise AssertionError("Expected a ValueError to be raised.")
    except ValueError:
        pass


def test_get_event_study_insufficient_history():
    returns, market = _synthetic_returns(n=50)

    try:
        event_study_model.get_event_study(
            returns=returns, market_returns=market, event_date=returns.index[10]
        )
        raise AssertionError("Expected a ValueError to be raised.")
    except ValueError:
        pass


def test_get_event_study_invalid_type():
    returns, market = _synthetic_returns()

    try:
        event_study_model.get_event_study(
            returns=returns.to_numpy(), market_returns=market, event_date=None
        )
        raise AssertionError("Expected a TypeError to be raised.")
    except TypeError:
        pass


def test_get_event_study_moves_a_date_without_a_return_to_the_next_one():
    returns, market = _synthetic_returns()
    # 2020-12-26 is a Saturday: the event is first priced on Monday 2020-12-28.
    saturday, monday = "2020-12-26", "2020-12-28"
    assert pd.Timestamp(saturday) not in returns.index

    keywords = {"estimation_window": 200, "gap_days": 20}
    on_saturday = event_study_model.get_event_study(
        returns=returns, market_returns=market, event_date=saturday, **keywords
    )
    on_monday = event_study_model.get_event_study(
        returns=returns, market_returns=market, event_date=monday, **keywords
    )

    assert (
        on_saturday["cumulative_abnormal_return"]
        == on_monday["cumulative_abnormal_return"]
    )
    assert on_saturday["abnormal_returns"].index.equals(
        on_monday["abnormal_returns"].index
    )


def test_get_event_study_event_date_after_the_returns():
    returns, market = _synthetic_returns()

    try:
        event_study_model.get_event_study(
            returns=returns, market_returns=market, event_date="2030-01-01"
        )
        raise AssertionError("Expected a ValueError to be raised.")
    except ValueError as error:
        assert "outside the returns" in str(error)


def test_get_event_study_reads_a_plain_date_in_the_time_zone_of_the_returns():
    returns, market = _synthetic_returns()
    keywords = {"estimation_window": 200, "gap_days": 20}
    plain = event_study_model.get_event_study(
        returns=returns, market_returns=market, event_date="2020-12-28", **keywords
    )

    aware = event_study_model.get_event_study(
        returns=returns.tz_localize("America/New_York"),
        market_returns=market.tz_localize("America/New_York"),
        event_date="2020-12-28",
        **keywords,
    )

    assert aware["cumulative_abnormal_return"] == plain["cumulative_abnormal_return"]
