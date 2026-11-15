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
        [("Open", "AAPL"), ("Close", "AAPL"), ("Open", "MSFT"), ("Close", "MSFT")]
    )
    df = pd.DataFrame([[1, 2, 3, 4]], columns=columns)

    result = dataframe_model.filter_columns(df, ["Open", "Close"])

    assert list(result.columns) == list(columns)

    # One name left on a level is dropped, so the Open prices are indexed by ticker.
    result = dataframe_model.filter_columns(df, ["Open"])

    assert list(result.columns) == ["AAPL", "MSFT"]
    assert result.loc[0].tolist() == [1, 3]


def test_filter_columns_multiindex_numeric_year():
    """A displayed year should match an integer MultiIndex level value."""
    columns = pd.MultiIndex.from_tuples(
        [("Revenue", 2024), ("Revenue", 2025), ("Profit", 2024)]
    )
    df = pd.DataFrame([[100, 110, 20]], columns=columns)

    result = dataframe_model.filter_columns(df, ["2024"])

    # The year is the only value left on its level, so the columns are the line items.
    pd.testing.assert_frame_equal(
        result, df.iloc[:, [0, 2]].droplevel(1, axis="columns")
    )


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


def test_filtering_one_row_of_one_ticker_keeps_the_ticker():
    from financetoolkit.utilities.dataframe_model import filter_columns

    quote = pd.DataFrame({"AAPL": [336.64, "Apple Inc."]}, index=["Price", "Name"])

    assert filter_columns(quote, ["Price"]).to_dict() == {"AAPL": 336.64}


def test_show_columns_parameter_filters_every_method():
    """A class decorated with filter_shown_columns filters on its show_columns parameter."""

    @dataframe_model.filter_shown_columns
    class Data:
        def get_data(self, rounding: int = 2, show_columns=None) -> pd.DataFrame:
            return pd.DataFrame({"Price": [1.0], "CAPE": [2.0], "Dividend": [3.0]})

        def get_other(self) -> pd.DataFrame:
            return pd.DataFrame({"Price": [1.0]})

    data = Data()

    assert list(data.get_data().columns) == ["Price", "CAPE", "Dividend"]
    assert list(data.get_data(show_columns=["CAPE", "Price"]).columns) == [
        "CAPE",
        "Price",
    ]
    assert list(data.get_data(show_columns="CAPE").columns) == ["CAPE"]
    assert list(data.get_data(2, ["Dividend"]).columns) == ["Dividend"]
    assert data.get_other.__name__ == "get_other"


def test_every_public_method_takes_show_columns():
    """Every get_, collect_ and search_ method of the controllers takes show_columns."""
    import inspect

    from financetoolkit import Economics, Toolkit
    from financetoolkit.discovery.discovery_controller import Discovery
    from financetoolkit.econometrics.econometrics_controller import Econometrics
    from financetoolkit.etf.etf_controller import ETF
    from financetoolkit.fixedincome.fixedincome_controller import FixedIncome
    from financetoolkit.models.models_controller import Models
    from financetoolkit.options.options_controller import Options
    from financetoolkit.performance.performance_controller import Performance
    from financetoolkit.portfolio.construction.construction_controller import (
        PortfolioConstruction,
    )
    from financetoolkit.portfolio.portfolio_controller import Portfolio
    from financetoolkit.ratios.ratios_controller import Ratios
    from financetoolkit.risk.risk_controller import Risk
    from financetoolkit.technicals.technicals_controller import Technicals

    missing = [
        f"{cls.__name__}.{name}"
        for cls in (
            Toolkit,
            Economics,
            Discovery,
            Econometrics,
            ETF,
            FixedIncome,
            Models,
            Options,
            Performance,
            PortfolioConstruction,
            Portfolio,
            Ratios,
            Risk,
            Technicals,
        )
        for name, member in vars(cls).items()
        if name.startswith(("get_", "collect_", "search_"))
        and inspect.isfunction(member)
        and "show_columns" not in inspect.signature(member).parameters
    ]

    assert not missing
