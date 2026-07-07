"""Dataframe Model Tests"""

# ruff: noqa: PLR2004, PD901

from unittest.mock import MagicMock

import pandas as pd

from financetoolkit.utilities import dataframe_model


def test_combine_dataframes():
    """Test combining dataframes from different companies."""
    df1 = pd.DataFrame(
        {"2020": [100, 50], "2021": [110, 55]}, index=["Revenue", "Profit"]
    )

    df2 = pd.DataFrame(
        {"2020": [200, 80], "2021": [220, 88]}, index=["Revenue", "Profit"]
    )

    dataset_dict = {"AAPL": df1, "MSFT": df2}

    result = dataframe_model.combine_dataframes(dataset_dict)

    # Check structure
    assert isinstance(result.index, pd.MultiIndex)
    assert result.index.levels[0].tolist() == ["AAPL", "MSFT"]
    assert result.index.levels[1].tolist() == ["Profit", "Revenue"]

    # Check values
    assert result.loc[("AAPL", "Revenue"), "2020"] == 100
    assert result.loc[("MSFT", "Revenue"), "2020"] == 200


def test_combine_dataframes_single_ticker():
    """Test combining single ticker."""
    df1 = pd.DataFrame({"2020": [100], "2021": [110]}, index=["Revenue"])

    dataset_dict = {"AAPL": df1}

    result = dataframe_model.combine_dataframes(dataset_dict)

    assert isinstance(result.index, pd.MultiIndex)
    assert result.loc[("AAPL", "Revenue"), "2020"] == 100


def test_equal_length_first_dataset_starts_later():
    """Test equal_length when first dataset starts later."""
    # Create mock dataframes with different starting years
    df1 = pd.DataFrame([[100]], columns=["2022"])
    df2 = pd.DataFrame([[200]], columns=["2020"])

    # Mock the insert method
    df1.insert = MagicMock()
    df1.sort_index = MagicMock(return_value=df1)

    dataframe_model.equal_length(df1, df2)

    # Should have called insert to add earlier years
    assert df1.insert.call_count == 2  # 2021 and 2020
    df1.sort_index.assert_called_once()


def test_equal_length_second_dataset_starts_later():
    """Test equal_length when second dataset starts later."""
    df1 = pd.DataFrame([[100]], columns=["2020"])
    df2 = pd.DataFrame([[200]], columns=["2022"])

    # Mock the insert method
    df2.insert = MagicMock()
    df2.sort_index = MagicMock(return_value=df2)

    dataframe_model.equal_length(df1, df2)

    # Should have called insert to add earlier years
    assert df2.insert.call_count == 2  # 2021 and 2020
    df2.sort_index.assert_called_once()


def test_equal_length_same_start():
    """Test equal_length when both datasets start at same time."""
    df1 = pd.DataFrame([[100]], columns=["2020"])
    df2 = pd.DataFrame([[200]], columns=["2020"])

    result1, result2 = dataframe_model.equal_length(df1, df2)

    # Should return original dataframes
    assert result1.equals(df1)
    assert result2.equals(df2)


def test_filter_columns_none_returns_unchanged():
    """Test filter_columns returns the result unchanged when show_columns is None."""
    df = pd.DataFrame({"A": [1], "B": [2]})

    result = dataframe_model.filter_columns(df, None)

    pd.testing.assert_frame_equal(result, df)


def test_filter_columns_flat_columns():
    """Test filter_columns filters a flat-column DataFrame."""
    df = pd.DataFrame({"Revenue": [1, 2], "Expenses": [3, 4], "Profit": [5, 6]})

    result = dataframe_model.filter_columns(df, ["Revenue", "Profit"])

    assert list(result.columns) == ["Revenue", "Profit"]


def test_filter_columns_multiindex_columns():
    """Test filter_columns filters a MultiIndex-column DataFrame by first level."""
    columns = pd.MultiIndex.from_tuples(
        [("Open", "AAPL"), ("Close", "AAPL"), ("Open", "MSFT")]
    )
    df = pd.DataFrame([[1, 2, 3]], columns=columns)

    result = dataframe_model.filter_columns(df, ["Open"])

    assert set(result.columns.get_level_values(0)) == {"Open"}


def test_filter_columns_row_index_fallback():
    """Test filter_columns falls back to filtering by flat row index."""
    df = pd.DataFrame({"2020": [1, 2, 3]}, index=["Revenue", "Expenses", "Profit"])

    result = dataframe_model.filter_columns(df, ["Revenue", "Profit"])

    assert set(result.index) == {"Revenue", "Profit"}


def test_filter_columns_multiindex_row_fallback_drops_redundant_level():
    """Test filter_columns drops the metric level when only one metric remains."""
    index = pd.MultiIndex.from_tuples(
        [("AAPL", "Revenue"), ("MSFT", "Revenue")],
    )
    df = pd.DataFrame({"2020": [100, 200]}, index=index)

    result = dataframe_model.filter_columns(df, ["Revenue"])

    assert not isinstance(result.index, pd.MultiIndex)
    assert result.loc["AAPL", "2020"] == 100


def test_filter_columns_dict_of_dataframes():
    """Test filter_columns applies the filter to each DataFrame in a dict."""
    data = {
        "AAPL": pd.DataFrame({"Revenue": [1], "Expenses": [2]}),
        "MSFT": pd.DataFrame({"Revenue": [3], "Expenses": [4]}),
    }

    result = dataframe_model.filter_columns(data, ["Revenue"])

    assert list(result["AAPL"].columns) == ["Revenue"]
    assert list(result["MSFT"].columns) == ["Revenue"]


def test_filter_columns_no_match_returns_unchanged():
    """Test filter_columns returns the original DataFrame when nothing matches."""
    df = pd.DataFrame({"Revenue": [1], "Expenses": [2]})

    result = dataframe_model.filter_columns(df, ["Nonexistent"])

    pd.testing.assert_frame_equal(result, df)


def test_filter_columns_empty_dataframe():
    """Test filter_columns passes through an empty DataFrame unchanged."""
    df = pd.DataFrame()

    result = dataframe_model.filter_columns(df, ["Revenue"])

    assert result.empty


def test_filter_columns_passthrough_series():
    """Test filter_columns passes through a Series unchanged."""
    series = pd.Series([1, 2, 3])

    result = dataframe_model.filter_columns(series, ["Revenue"])

    pd.testing.assert_series_equal(result, series)
