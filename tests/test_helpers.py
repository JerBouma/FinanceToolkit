# ruff: noqa

"""Helpers Tests"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from financetoolkit import helpers


def test_enrich_historical_data_basic():
    """Test basic enrichment of historical data."""
    data = pd.DataFrame(
        {"Adj Close": [100, 110, 121, 133]},
        index=pd.date_range("2020-01-01", periods=4, freq="D"),
    )

    result = helpers.enrich_historical_data(data)

    # Check new columns
    assert "Return" in result.columns
    assert "Cumulative Return" in result.columns

    # Volatility lives in the Risk module now (risk_model.get_volatility).
    assert "Volatility" not in result.columns

    # Check return calculation
    assert pd.isna(result["Return"].iloc[0])
    assert abs(result["Return"].iloc[1] - 0.1) < 0.01


def test_enrich_historical_data_with_date_range():
    """Test enrichment with start and end dates."""
    data = pd.DataFrame(
        {"Adj Close": [100, 110, 121, 133, 146]},
        index=pd.date_range("2020-01-01", periods=5, freq="D"),
    )

    result = helpers.enrich_historical_data(data, start="2020-01-02", end="2020-01-04")

    # Should still have all rows but calculations based on subset
    assert len(result) == 5
    assert "Return" in result.columns


def test_enrich_historical_data_custom_return_column():
    """Test enrichment with custom return column."""
    data = pd.DataFrame(
        {"Close": [100, 110, 121, 133], "Adj Close": [90, 99, 109, 120]},
        index=pd.date_range("2020-01-01", periods=4, freq="D"),
    )

    result = helpers.enrich_historical_data(data, return_column="Close")

    # Should use Close for calculations
    assert abs(result["Return"].iloc[1] - 0.1) < 0.01


def test_handle_portfolio_decorator_basic():
    """Test handle_portfolio decorator basic functionality."""

    class MockSelf:
        def __init__(self):
            self._tickers = ["AAPL", "MSFT", "Portfolio"]
            self._rounding = 4
            self._quarterly = False
            self._portfolio_weights = {
                "yearly": pd.DataFrame(
                    {"AAPL": [0.6, 0.6, 0.6], "MSFT": [0.4, 0.4, 0.4]},
                    index=pd.PeriodIndex(["2020", "2021", "2022"], freq="Y"),
                )
            }

    @helpers.handle_portfolio
    def test_function(self):
        return pd.DataFrame(
            {"AAPL": [10, 20, 30], "MSFT": [15, 25, 35]},
            index=pd.PeriodIndex(["2020", "2021", "2022"], freq="Y"),
        )

    mock_self = MockSelf()
    result = test_function(mock_self)

    # Should have Portfolio column
    assert "Portfolio" in result.columns

    # Check weighted average calculation
    expected_portfolio = 0.6 * 10 + 0.4 * 15  # First row: 0.6*10 + 0.4*15 = 12
    assert abs(result["Portfolio"].iloc[0] - expected_portfolio) < 0.01


def test_handle_portfolio_decorator_with_benchmark():
    """Test handle_portfolio decorator excludes benchmark."""

    class MockSelf:
        def __init__(self):
            self._tickers = ["AAPL", "MSFT", "Portfolio"]
            self._rounding = 4
            self._quarterly = False
            self._portfolio_weights = {
                "yearly": pd.DataFrame(
                    {"AAPL": [0.6, 0.6], "MSFT": [0.4, 0.4]},
                    index=pd.PeriodIndex(["2020", "2021"], freq="Y"),
                )
            }

    @helpers.handle_portfolio
    def test_function(self):
        return pd.DataFrame(
            {"AAPL": [10, 20], "MSFT": [15, 25], "Benchmark": [12, 22]},
            index=pd.PeriodIndex(["2020", "2021"], freq="Y"),
        )

    mock_self = MockSelf()
    result = test_function(mock_self)

    # Should have Portfolio column
    assert "Portfolio" in result.columns
    # Should still have Benchmark column
    assert "Benchmark" in result.columns

    # Portfolio calculation should exclude benchmark
    expected_portfolio = 0.6 * 10 + 0.4 * 15  # Should be 12
    assert abs(result["Portfolio"].iloc[0] - expected_portfolio) < 0.01


def test_handle_portfolio_decorator_no_portfolio():
    """Test handle_portfolio decorator when Portfolio not in tickers."""

    class MockSelf:
        def __init__(self):
            self._tickers = ["AAPL", "MSFT"]
            self._rounding = 4

    @helpers.handle_portfolio
    def test_function(self):
        return pd.DataFrame({"AAPL": [10, 20], "MSFT": [15, 25]})

    mock_self = MockSelf()
    result = test_function(mock_self)

    # Should not have Portfolio column
    assert "Portfolio" not in result.columns
    assert len(result.columns) == 2


def test_handle_portfolio_decorator_quarterly():
    """Test handle_portfolio decorator with quarterly data."""

    class MockSelf:
        def __init__(self):
            self._tickers = ["AAPL", "MSFT", "Portfolio"]
            self._rounding = 4
            self._quarterly = True
            self._portfolio_weights = {
                "quarterly": pd.DataFrame(
                    {"AAPL": [0.6, 0.6], "MSFT": [0.4, 0.4]},
                    index=pd.PeriodIndex(["2020Q1", "2020Q2"], freq="Q"),
                )
            }

    @helpers.handle_portfolio
    def test_function(self):
        return pd.DataFrame(
            {"AAPL": [10, 20], "MSFT": [15, 25]},
            index=pd.PeriodIndex(["2020Q1", "2020Q2"], freq="Q"),
        )

    mock_self = MockSelf()
    result = test_function(mock_self)

    # Should have Portfolio column
    assert "Portfolio" in result.columns


def test_handle_portfolio_decorator_with_growth_warning():
    """Test handle_portfolio decorator shows warning for growth with multiple lags."""

    class MockSelf:
        def __init__(self):
            self._tickers = ["AAPL", "MSFT", "Portfolio"]
            self._rounding = 4
            self._quarterly = False
            self._portfolio_weights = {
                "yearly": pd.DataFrame(
                    {"AAPL": [0.6], "MSFT": [0.4]},
                    index=pd.PeriodIndex(["2020"], freq="Y"),
                )
            }

    @helpers.handle_portfolio
    def test_function(self, growth=False, lag=1):
        return pd.DataFrame(
            {"AAPL": [10], "MSFT": [15]}, index=pd.PeriodIndex(["2020"], freq="Y")
        )

    mock_self = MockSelf()

    with patch("financetoolkit.helpers.logger") as mock_logger:
        result = test_function(mock_self, growth=True, lag=[1, 2])

        mock_logger.warning.assert_called_once()
        assert "multiple lags" in mock_logger.warning.call_args[0][0]


def test_handle_portfolio_decorator_non_dataframe_result():
    """Test handle_portfolio decorator with non-DataFrame result."""

    class MockSelf:
        def __init__(self):
            self._tickers = ["AAPL", "MSFT", "Portfolio"]
            self._rounding = 4

    @helpers.handle_portfolio
    def test_function(self):
        return "Not a DataFrame"

    mock_self = MockSelf()
    result = test_function(mock_self)

    # Should return original result unchanged
    assert result == "Not a DataFrame"


def test_handle_portfolio_decorator_multiindex_columns():
    """Test handle_portfolio decorator with MultiIndex columns."""

    class MockSelf:
        def __init__(self):
            self._tickers = ["AAPL", "MSFT", "Portfolio"]
            self._rounding = 4
            self._quarterly = False
            self._portfolio_weights = {
                "yearly": pd.DataFrame(
                    {"AAPL": [0.6], "MSFT": [0.4]},
                    index=pd.PeriodIndex(["2020"], freq="Y"),
                )
            }

    @helpers.handle_portfolio
    def test_function(self):
        columns = pd.MultiIndex.from_tuples([("AAPL", "Value"), ("MSFT", "Value")])
        return pd.DataFrame(
            {("AAPL", "Value"): [10], ("MSFT", "Value"): [15]},
            index=pd.PeriodIndex(["2020"], freq="Y"),
            columns=columns,
        )

    mock_self = MockSelf()
    result = test_function(mock_self)

    # Should not add Portfolio column for MultiIndex columns
    assert len(result.columns) == 2


def test_run_in_parallel_preserves_order():
    """Test that results come back in the same order as the argument tuples."""
    result = helpers.run_in_parallel(
        lambda base, exponent: base**exponent,
        [(2, 3), (3, 2), (10, 0)],
    )

    assert result == [8, 9, 1]


def test_run_in_parallel_empty_items():
    """Test that an empty collection returns an empty list without spawning a pool."""
    assert helpers.run_in_parallel(lambda: None, []) == []


def test_run_in_parallel_collects_into_shared_dict():
    """Test the pattern the data collection functions use: workers mutate a shared dict."""
    collected: dict[str, int] = {}

    def worker(ticker, dataset_dictionary):
        dataset_dictionary[ticker] = len(ticker)

    helpers.run_in_parallel(
        worker, [(ticker, collected) for ticker in ["AAPL", "MSFT", "GOOGL"]]
    )

    assert collected == {"AAPL": 4, "MSFT": 4, "GOOGL": 5}


def test_run_in_parallel_propagates_worker_defects():
    """Test that a defect inside a worker surfaces instead of dying in its thread."""

    def worker(ticker):
        raise TypeError(f"unsupported type for {ticker}")

    with pytest.raises(TypeError):
        helpers.run_in_parallel(worker, [("AAPL",), ("MSFT",)])


@pytest.mark.parametrize("error", [KeyError, ValueError, IndexError, ZeroDivisionError])
def test_run_in_parallel_keeps_the_others_when_one_worker_fails(error, caplog):
    """Test that a worker failing on its data is logged and the other results are kept."""

    def worker(ticker, api_key):  # noqa: ARG001
        if ticker == "MSFT":
            raise error(f"https://example.com/quote?symbol={ticker}&apikey={api_key}")

        return ticker, len(ticker)

    with caplog.at_level("ERROR"):
        result = helpers.run_in_parallel(
            worker, [(ticker, "SECRET") for ticker in ["AAPL", "MSFT", "GOOGL"]]
        )

    assert result == [("AAPL", 4), ("GOOGL", 5)]
    assert "No data could be retrieved for MSFT" in caplog.text
    assert "SECRET" not in caplog.text


def test_run_in_parallel_raises_every_error_in_strict_mode(monkeypatch):
    """Test that strict error handling raises a worker's data error as well."""
    monkeypatch.setenv("FINANCETOOLKIT_STRICT_ERRORS", "1")

    def worker(ticker):
        raise KeyError(f"no data for {ticker}")

    with pytest.raises(KeyError):
        helpers.run_in_parallel(worker, [("AAPL",)])


def test_determine_max_workers_default():
    """Test the default when the environment variable is not set."""
    with patch.dict("os.environ", {}, clear=True):
        assert helpers.determine_max_workers() == helpers.DEFAULT_MAX_WORKERS


def test_determine_max_workers_environment_variable():
    """Test that a positive integer in the environment variable is used."""
    with patch.dict(
        "os.environ", {helpers.MAX_WORKERS_ENVIRONMENT_VARIABLE: "3"}, clear=True
    ):
        assert helpers.determine_max_workers() == 3


def test_determine_max_workers_invalid_environment_variable():
    """Test that a non-positive or non-integer value falls back to the default."""
    for invalid in ["0", "-4", "many", "2.5"]:
        with patch.dict(
            "os.environ",
            {helpers.MAX_WORKERS_ENVIRONMENT_VARIABLE: invalid},
            clear=True,
        ):
            assert helpers.determine_max_workers() == helpers.DEFAULT_MAX_WORKERS


def test_handle_portfolio_weighs_the_tickers_with_a_value_by_the_result_period():
    """The portfolio row uses the weights of the result's period and skips missing values."""
    import numpy as np

    daily = pd.period_range("2024-01-01", periods=3, freq="D")
    yearly = pd.period_range("2024", periods=1, freq="Y")

    class Module:
        _tickers = ["A", "B", "Portfolio"]
        _rounding = 4
        _portfolio_weights = {
            "daily": pd.DataFrame({"A": 0.25, "B": 0.75}, index=daily),
            "yearly": pd.DataFrame({"A": 0.5, "B": 0.5}, index=yearly),
        }

        @helpers.handle_portfolio
        def get_metric(self, period: str | None = None):  # noqa: ARG002
            # Most risk metrics default to daily results.
            return pd.DataFrame(
                {"A": [1.0, 1.0, 1.0], "B": [3.0, np.nan, 3.0]}, index=daily
            )

    result = Module().get_metric()

    assert result["Portfolio"].tolist() == [2.5, 1.0, 2.5]


def test_enriching_a_response_without_prices_adds_no_row():
    """
    An empty response (a range before the listing) stays empty: it once gained a row
    labelled 0, which was cached and merged into the dates of the real prices.
    """
    columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends"]
    empty = pd.DataFrame(columns=columns, index=pd.PeriodIndex([], freq="D"))

    enriched = helpers.enrich_historical_data(empty, "1980-01-01", "1984-12-31")

    assert enriched.empty


def test_availability_dates_use_real_filings_and_otherwise_the_deadline():
    periods = pd.PeriodIndex(["2023Q1", "2023Q2", "2023Q3"], freq="Q")
    period_end_dates = pd.DataFrame(
        [pd.to_datetime(["2023-03-31", "2023-06-30", "2023-09-30"])] * 2,
        index=["SEC", "NONSEC"],
        columns=periods,
    )
    statistics = pd.DataFrame(
        [
            ["2023-04-28 06:01:00", "2023-08-04 16:30:00", None],
            [None, None, "2023-10-25"],
            ["2023-03-30 19:00:00", "2023-07-02 19:00:00", "2023-09-29 20:00:00"],
            ["2023-03-31", "2023-07-02", "2023-09-30"],
        ],
        index=pd.MultiIndex.from_tuples(
            [
                ("SEC", "Accepted Date"),
                ("SEC", "Filing Date"),
                ("NONSEC", "Accepted Date"),
                ("NONSEC", "Filing Date"),
            ]
        ),
        columns=periods.astype(str),
    )

    availability = helpers.determine_availability_dates(
        period_end_dates, statistics, quarter=True
    )

    # Accepted before the close counts that day, after the close the next day, and a
    # filing date without a time from the day after it.
    assert availability.loc["SEC"].tolist() == list(
        pd.to_datetime(["2023-04-28", "2023-08-05", "2023-10-26"])
    )
    # Dates around the period end are not filings, so the quarterly deadline is used.
    assert availability.loc["NONSEC"].tolist() == list(
        pd.to_datetime(["2023-06-29", "2023-09-28", "2023-12-29"])
    )


def test_availability_dates_without_statistics_use_the_annual_deadline():
    period_end_dates = pd.DataFrame(
        [pd.to_datetime(["2022-12-31"])],
        index=["AAPL"],
        columns=pd.PeriodIndex(["2022"], freq="Y"),
    )

    availability = helpers.determine_availability_dates(
        period_end_dates, None, quarter=False
    )

    assert availability.loc["AAPL"].tolist() == [pd.Timestamp("2023-04-30")]


def test_availability_of_the_quarter_closing_a_fiscal_year_uses_the_annual_deadline():
    periods = pd.PeriodIndex(["2023Q3", "2023Q4"], freq="Q")
    period_end_dates = pd.DataFrame(
        [
            pd.to_datetime(["2023-09-30", "2023-12-31"]),
            pd.to_datetime(["2023-09-30", "2023-12-31"]),
        ],
        index=["SEPTEMBER", "DECEMBER"],
        columns=periods,
    )
    statistics = pd.DataFrame(
        [["Q4", "Q1"]],
        index=pd.MultiIndex.from_tuples([("SEPTEMBER", "Period")]),
        columns=periods.astype(str),
    )

    availability = helpers.determine_availability_dates(
        period_end_dates, statistics, quarter=True
    )

    # The fiscal Q4 label decides where the source reports it, December otherwise.
    assert availability.loc["SEPTEMBER"].tolist() == list(
        pd.to_datetime(["2024-01-28", "2024-03-30"])
    )
    assert availability.loc["DECEMBER"].tolist() == list(
        pd.to_datetime(["2023-12-29", "2024-04-29"])
    )


def test_importing_the_package_keeps_runtime_warnings_visible():
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    # A fresh interpreter, as the test session itself may already have imported it.
    code = (
        "import warnings, financetoolkit.toolkit_controller, "
        "financetoolkit.ratios.ratios_controller, financetoolkit.risk.risk_controller, "
        "financetoolkit.performance.performance_controller\n"
        "assert not [f for f in warnings.filters if f[0] == 'ignore' and f[2] is RuntimeWarning]\n"
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.stdout.strip() == "ok", result.stderr


def test_round_half_up_rounds_a_half_away_from_zero():
    from decimal import Decimal

    # round() rounds these to even or misses them on the binary float.
    assert helpers.round_half_up(2.675, 2) == 2.68
    assert helpers.round_half_up(0.125, 2) == 0.13
    assert helpers.round_half_up(Decimal("17015150") / 1000, 1) == 17015.2
    assert pd.isna(helpers.round_half_up(float("nan"), 2))


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (None, []),
        ("", []),
        (" a ,, b ,", ["a", "b"]),
        ([" a", "b ", "", None], ["a", "b"]),
        (("a",), ["a"]),
        ([1, 5, 10], [1, 5, 10]),
        (5, [5]),
        (0.5, [0.5]),
        (pd.Index(["a", "b"]), ["a", "b"]),
    ],
)
def test_split_values_returns_a_list_of_the_given_values(values, expected):
    assert helpers.split_values(values) == expected


# Fiscal period ends as FinancialModelingPrep reports them (the actual last day) next to the
# month end Yahoo Finance reports for the same period, with the calendar label both get.
FISCAL_PERIOD_ENDS = [
    # ticker, quarter, FinancialModelingPrep, Yahoo Finance, label
    ("AAPL", False, "2025-09-27", "2025-09-30", "2025"),
    ("NVDA", False, "2026-01-25", "2026-01-31", "2025"),
    ("MSFT", False, "2025-06-30", "2025-06-30", "2025"),
    ("WMT", False, "2026-01-31", "2026-01-31", "2025"),
    ("COST", False, "2023-09-03", "2023-08-31", "2023"),
    ("WDC", False, "2022-07-01", "2022-06-30", "2022"),
    ("HD", False, "2026-02-01", "2026-01-31", "2025"),
    ("TGT", False, "2024-02-03", "2024-01-31", "2023"),
    ("AAPL", True, "2025-12-27", "2025-12-31", "2025Q4"),
    ("NVDA", True, "2025-10-26", "2025-10-31", "2025Q3"),
    ("COST", True, "2025-05-11", "2025-05-31", "2025Q2"),
    ("WDC", True, "2026-01-02", "2025-12-31", "2025Q4"),
    ("HD", True, "2025-08-03", "2025-07-31", "2025Q2"),
    ("HD", True, "2026-05-03", "2026-04-30", "2026Q1"),
    ("TGT", True, "2025-11-01", "2025-10-31", "2025Q3"),
    ("TGT", True, "2026-08-01", "2026-07-31", "2026Q2"),
    ("LOW", True, "2025-08-01", "2025-07-31", "2025Q2"),
    ("DG", True, "2026-05-01", "2026-04-30", "2026Q1"),
]


@pytest.mark.parametrize(
    "ticker, quarter, fmp_date, yahoo_date, label", FISCAL_PERIOD_ENDS
)
def test_both_sources_label_a_fiscal_period_alike(
    ticker, quarter, fmp_date, yahoo_date, label
):
    fmp_label, yahoo_label = helpers.convert_period_end_dates_to_calendar_periods(
        pd.to_datetime([fmp_date, yahoo_date]), quarter=quarter
    )

    assert str(fmp_label) == str(yahoo_label) == label, ticker


def test_consecutive_52_53_week_quarters_keep_their_own_calendar_quarter():
    # Home Depot's quarters end on a Sunday a few days into August, November, February and May.
    dates = pd.to_datetime(["2020-05-03", "2020-08-02", "2020-11-01", "2021-01-31"])
    adjustments = {}

    labels = helpers.convert_period_end_dates_to_calendar_periods(
        dates, quarter=True, ticker="HD", fiscal_year_adjustments=adjustments
    )

    assert [str(label) for label in labels] == ["2020Q1", "2020Q2", "2020Q3", "2020Q4"]
    # Each quarter ends in the month after a calendar quarter (April, July, October, January).
    assert adjustments["HD"] == [
        {"fiscal_year": fiscal, "calendar_year": calendar}
        for fiscal, calendar in [
            ("2020Q2", "2020Q1"),
            ("2020Q3", "2020Q2"),
            ("2020Q4", "2020Q3"),
            ("2021Q1", "2020Q4"),
        ]
    ]


def test_twelve_week_quarters_ending_later_in_a_month_stay_in_that_month():
    # Costco's 12-week quarters ending 12 February and 7 May 2023.
    labels = helpers.convert_period_end_dates_to_calendar_periods(
        pd.to_datetime(["2023-02-12", "2023-05-07"]), quarter=True
    )

    assert [str(label) for label in labels] == ["2023Q1", "2023Q2"]


def test_a_fiscal_year_ending_in_the_first_days_of_june_belongs_to_the_previous_year():
    labels = helpers.convert_period_end_dates_to_calendar_periods(
        pd.to_datetime(["2023-06-05", "2023-06-06"]), quarter=False
    )

    assert [str(label) for label in labels] == ["2022", "2023"]


def test_goodwill_and_intangible_assets_fall_back_on_the_separate_rows():
    """
    The combined row is used where it is reported; elsewhere goodwill and the other
    intangibles are added up with a missing one counting as zero, and only a period
    without any of the three is NaN.
    """
    nan = np.nan
    periods = ["2021", "2022", "2023", "2024"]
    rows = {
        ("A", "Goodwill and Intangible Assets"): [100, nan, nan, nan],
        ("A", "Goodwill"): [60, 70, nan, nan],
        ("A", "Intangible Assets"): [40, 20, 15, nan],
        ("B", "Goodwill and Intangible Assets"): [nan, nan, nan, 0],
        ("B", "Goodwill"): [5, nan, 3, nan],
        ("B", "Intangible Assets"): [nan, nan, 2, nan],
    }
    balance_sheet = pd.DataFrame.from_dict(rows, orient="index", columns=periods)
    balance_sheet.index = pd.MultiIndex.from_tuples(balance_sheet.index)

    result = helpers.get_goodwill_and_intangible_assets(balance_sheet)

    assert result.loc["A"].tolist()[:3] == [100, 90, 15]
    assert np.isnan(result.loc["A", "2024"])
    assert result.loc["B", "2021"] == 5
    assert np.isnan(result.loc["B", "2022"])
    assert result.loc["B", "2023"] == 5
    assert result.loc["B", "2024"] == 0


def test_deflate_with_price_index_uses_the_latest_month_as_base():
    months = pd.period_range("2024-01", "2024-04", freq="M")
    values = pd.DataFrame({"Gold": [100.0, 100.0, 100.0, 100.0]}, index=months)
    price_index = pd.Series([50.0, np.nan, 100.0], index=months[[0, 1, 2]])

    real = helpers.deflate_with_price_index(values, price_index)

    # January's prices were worth twice as much in March money; a month the index skipped
    # has no real value and a month after its latest keeps its nominal value.
    assert real["Gold"].iloc[0] == 200.0
    assert np.isnan(real["Gold"].iloc[1])
    assert real["Gold"].iloc[2:].tolist() == [100.0, 100.0]


def test_change_over_periods_compares_the_same_period_a_year_earlier():
    from financetoolkit.economics.helpers import change_over_periods

    months = pd.period_range("2024-01", "2025-12", freq="M").delete(21)
    index = pd.DataFrame({"Japan": range(100, 100 + len(months))}, index=months)

    change = change_over_periods(index, 12)

    # October 2025 is missing: it and the month a year later are not compared with
    # whichever row is twelve places back.
    assert pd.isna(change.loc[pd.Period("2025-10", "M"), "Japan"])
    assert change.loc[pd.Period("2025-11", "M"), "Japan"] == pytest.approx(
        121 / 110 - 1
    )
