"""Serialization Model Tests"""

import datetime
import pickle
import sqlite3
import zlib

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal

from financetoolkit.cache import cache_controller, serialization_model

UNPICKLED = []


class Recorder:
    """Records that it was unpickled, which a cache read must never do."""

    def __reduce__(self):
        return (UNPICKLED.append, ("unpickled",))


def round_trip(value):
    return serialization_model.decode_object(serialization_model.encode_object(value))


def test_a_frame_with_multiindex_columns_and_a_period_index_round_trips():
    columns = pd.MultiIndex.from_product(
        [["Close", "Volume"], ["AAPL", "MSFT"]], names=[None, "Ticker"]
    )
    frame = pd.DataFrame(
        np.arange(12, dtype=float).reshape(3, 4),
        index=pd.period_range("2020Q1", periods=3, freq="Q", name="Date"),
        columns=columns,
    )
    frame.iloc[0, 0] = np.nan

    assert_frame_equal(round_trip(frame), frame)


def test_every_column_dtype_round_trips():
    frame = pd.DataFrame(
        {
            "float": [1.5, np.nan, np.inf],
            "int": [1, 2, 3],
            "uint": np.array([1, 2, 3], dtype="uint8"),
            "bool": [True, False, True],
            "object": ["a", None, pd.Timestamp("2020-01-01")],
            "str": pd.array(["x", None, "z"], dtype="str"),
            "datetime": pd.to_datetime(["2020-01-01", None, "2021-05-03"]),
            "datetime_seconds": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03"]
            ).as_unit("s"),
            "datetime_tz": pd.to_datetime(
                ["2020-01-01 00:00", "2020-10-25 02:30", "2021-05-03 00:00"]
            ).tz_localize("Europe/Amsterdam", ambiguous=[False, True, False]),
            "period": pd.PeriodIndex(["2020-01", None, "2020-03"], freq="M"),
            "timedelta": pd.to_timedelta([1, 2, None], unit="D"),
            "Int64": pd.array([1, None, 3], dtype="Int64"),
            "boolean": pd.array([True, None, False], dtype="boolean"),
            "category": pd.Categorical(
                ["a", "b", "a"], categories=["b", "a"], ordered=True
            ),
        },
        index=pd.Index(["r1", "r2", "r3"], name="Row"),
    )

    assert_frame_equal(round_trip(frame), frame)


def test_frames_without_rows_columns_or_unique_names_round_trip():
    for frame in (
        pd.DataFrame(),
        pd.DataFrame(index=pd.period_range("2020", periods=2, freq="Y")),
        pd.DataFrame([[1, 2]], columns=["a", "a"]),
        pd.DataFrame({"a": ["x"]}, dtype=object),
    ):
        assert_frame_equal(round_trip(frame), frame)


def test_series_and_indexes_round_trip():
    prices = pd.Series(
        [1.0, 2.0],
        index=pd.DatetimeIndex(["2020-01-01", "2020-01-02"], name="Date"),
        name="Close",
    )
    names = pd.Series(["a.zip", "b.zip"])
    objects = pd.Series(["a", "b"], dtype=object)

    for series in (prices, names, objects):
        assert_series_equal(round_trip(series), series)

    index = pd.period_range("2020-01", periods=3, freq="M")
    assert_index_equal(round_trip(index), index)


def test_dictionaries_lists_and_scalars_keep_their_types():
    value = {
        "plan": "Premium",
        "invalid_key": False,
        1: (1, 2.5),
        "expiries": ["2025-01-17", "2025-02-21"],
        "numbers": [np.float64(0.05), np.int64(3), np.bool_(True)],
        "timestamp": pd.Timestamp("2020-01-01 10:00", tz="America/New_York"),
        "period": pd.Period("2020Q1"),
        "date": datetime.date(2020, 1, 2),
        "datetime": datetime.datetime(2020, 1, 2, 3, 4),
        "timedelta": pd.Timedelta("1D"),
        "missing": [None, pd.NaT, pd.NA],
        "array": np.array([1.0, 2.0]),
        "adjustments": [{"fiscal_year": 2020, "calendar_year": 2021}],
    }

    restored = round_trip(value)

    assert restored.keys() == value.keys()
    for key in ("plan", "invalid_key", 1, "expiries", "period", "date", "datetime"):
        assert restored[key] == value[key]
        assert type(restored[key]) is type(value[key])
    for original, copy in zip(value["numbers"], restored["numbers"], strict=True):
        assert copy == original and type(copy) is type(original)
    assert restored["timestamp"] == value["timestamp"]
    assert str(restored["timestamp"].tz) == "America/New_York"
    assert restored["timedelta"] == value["timedelta"]
    assert restored["missing"][0] is None
    assert restored["missing"][1] is pd.NaT
    assert restored["missing"][2] is pd.NA
    np.testing.assert_array_equal(restored["array"], value["array"])
    assert restored["adjustments"] == value["adjustments"]
    assert np.isnan(round_trip(float("nan")))
    assert round_trip(True) is True


def test_a_value_that_cannot_be_stored_raises_a_type_error():
    with pytest.raises(TypeError):
        serialization_model.encode_object({1, 2})


def test_a_pickled_payload_is_refused_and_never_unpickled():
    payload = zlib.compress(pickle.dumps(Recorder()))

    with pytest.raises(ValueError):
        serialization_model.decode_object(payload)

    assert not UNPICKLED


def test_pickled_cache_entries_are_ignored_and_replaced(tmp_path):
    cache = cache_controller.Cache(location=tmp_path / "cache.db")
    frame = pd.DataFrame(
        {"Close": [1.0, 2.0]},
        index=pd.period_range("2020-01-01", periods=2, freq="D"),
    )
    cache.store(
        source="FinancialModelingPrep",
        dataset="historical",
        entity="AAPL",
        data=frame,
        start="2020-01-01",
        end="2020-01-02",
    )
    cache.set(source="FinancialModelingPrep", dataset="profile", entity="AAPL", data={})

    # A database written with pickled payloads, or one planted by someone else.
    planted = zlib.compress(pickle.dumps(Recorder()))
    with sqlite3.connect(cache.location) as connection:
        connection.execute("UPDATE series SET payload = ?", (planted,))
        connection.execute("UPDATE blobs SET payload = ?", (planted,))

    plan = cache.plan(
        source="FinancialModelingPrep",
        dataset="historical",
        entities=["AAPL"],
        start="2020-01-01",
        end="2020-01-02",
    )
    blob = cache.get(source="FinancialModelingPrep", dataset="profile", entity="AAPL")

    assert not UNPICKLED
    assert plan.cached_frame("AAPL") is None
    assert "AAPL" in plan.missing
    assert blob is None

    # The unreadable entry went with its coverage, so the data fetched again is stored.
    cache.store(
        source="FinancialModelingPrep",
        dataset="historical",
        entity="AAPL",
        data=frame,
        start="2020-01-01",
        end="2020-01-02",
    )
    plan = cache.plan(
        source="FinancialModelingPrep",
        dataset="historical",
        entities=["AAPL"],
        start="2020-01-01",
        end="2020-01-02",
    )

    assert_frame_equal(plan.cached_frame("AAPL"), frame)
    assert not plan.missing
