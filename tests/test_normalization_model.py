# ruff: noqa

"""Normalization Model Tests"""

import pandas as pd

from financetoolkit import normalization_model

# pylint: disable=missing-function-docstring


def test_convert_financial_statements_empty_format(recorder):
    financial_statements = pd.DataFrame(
        {("AAPL", "2023"): [1000, 2000, 3000], ("AAPL", "2022"): [900, 1800, 2700]}
    )
    financial_statements.index = ["Revenue", "Cost of Revenue", "Gross Profit"]

    empty_format = pd.DataFrame()

    result = normalization_model.convert_financial_statements(
        financial_statements=financial_statements,
        statement_format=empty_format,
    )

    recorder.capture(result.shape)


def _partial_statement(statement_format, reported: int = 2):
    """A statement holding only the first `reported` line items of the format."""
    index = pd.MultiIndex.from_tuples(
        [("AAPL", name) for name in statement_format.index[:reported]]
    )

    return pd.DataFrame(
        [[100.0, 110.0], [40.0, 45.0]][:reported],
        index=index,
        columns=["2023", "2024"],
    )


def test_adjust_financial_statements_fills_missing_items_with_nan():
    """An item the provider never reported must not read as a genuine zero.

    Filling with zero makes an unreported line item indistinguishable from one
    reported as actually zero, and that fabricated zero then flows into any ratio
    using it as a term or a denominator.
    """
    statement_format = normalization_model.read_normalization_file("balance")
    financial_statements = _partial_statement(statement_format)

    result = normalization_model.convert_financial_statements(
        financial_statements=financial_statements,
        statement_format=statement_format,
        adjust_financial_statements=True,
    )

    unreported = result.loc[("AAPL", statement_format.iloc[2])]

    assert unreported.isna().all()
    assert not (unreported == 0).any()


def test_adjust_financial_statements_still_aligns_every_row():
    """Filling with NaN must not cost the row alignment the option exists for."""
    statement_format = normalization_model.read_normalization_file("balance")
    financial_statements = _partial_statement(statement_format)

    result = normalization_model.convert_financial_statements(
        financial_statements=financial_statements,
        statement_format=statement_format,
        adjust_financial_statements=True,
    )

    assert len(result.loc["AAPL"]) == len(statement_format)


def test_adjust_financial_statements_leaves_reported_values_untouched():
    statement_format = normalization_model.read_normalization_file("balance")
    financial_statements = _partial_statement(statement_format)

    result = normalization_model.convert_financial_statements(
        financial_statements=financial_statements,
        statement_format=statement_format,
        adjust_financial_statements=True,
    )

    reported = result.loc[("AAPL", statement_format.iloc[0])]

    assert list(reported) == [100.0, 110.0]


def test_get_normalization_files_saves_to_the_given_folder(tmp_path):
    from financetoolkit import get_normalization_files

    get_normalization_files(tmp_path)

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "balance.csv",
        "cash.csv",
        "income.csv",
        "statistics.csv",
    ]
    assert not pd.read_csv(tmp_path / "balance.csv").empty


def test_toolkit_get_normalization_files_saves_to_the_given_folder(tmp_path):
    # The path is where the files go, not where they are read from.
    from financetoolkit import Toolkit

    Toolkit(["AAPL"], api_key="", sleep_timer=False).get_normalization_files(
        str(tmp_path)
    )

    assert len(list(tmp_path.glob("*.csv"))) == 4


def test_normalization_maps_the_period_end_date_and_yahoo_forex_changes():
    from importlib.resources import files

    statistics = pd.read_csv(
        files("financetoolkit") / "normalization" / "statistics.csv", index_col=0
    )
    cash_yf = pd.read_csv(
        files("financetoolkit") / "normalization" / "cash_yf.csv", index_col=0
    )

    statistics_yf = pd.read_csv(
        files("financetoolkit") / "normalization" / "statistics_yf.csv", index_col=0
    )

    assert statistics.loc["periodEndDate"].iloc[0] == "Period End Date"
    assert statistics_yf.loc["periodEndDate"].iloc[0] == "Period End Date"
    # Yahoo Finance's Net Change in Cash excludes the exchange rate effect, which it reports
    # separately; the translation of the cash flow statement reconciles with it.
    assert cash_yf.loc["EffectOfExchangeRateChanges"].iloc[0] == "Forex Changes on Cash"


def test_normalization_files_fall_back_per_statement(tmp_path):
    folder = tmp_path / "missing" / "folder"
    normalization_model.get_normalization_files(folder)

    assert sorted(path.name for path in folder.iterdir()) == [
        "balance.csv",
        "cash.csv",
        "income.csv",
        "statistics.csv",
    ]
    # The Yahoo Finance format is not among the copied files, so the packaged one is read.
    assert not normalization_model.read_normalization_file(
        "balance_yf", str(folder)
    ).empty


def test_statement_exchange_rates_past_the_last_rate_are_missing():
    from financetoolkit.currencies_model import get_statement_exchange_rates

    dates = pd.bdate_range("2021-01-01", "2023-01-01")
    daily = pd.DataFrame({"EURUSD=X": range(len(dates))}, index=dates, dtype=float)
    period_ends = pd.DataFrame(
        [[pd.Timestamp("2022-12-31"), pd.Timestamp("2023-09-30")]],
        index=["X"],
        columns=pd.PeriodIndex(["2022", "2023"], freq="Y"),
    )
    closing = get_statement_exchange_rates(
        daily, period_ends, pd.Series({"X": "EURUSD=X"}), False, "closing"
    )

    assert closing.loc[pd.Period("2022", "Y"), "X"] == len(dates) - 1
    assert pd.isna(closing.loc[pd.Period("2023", "Y"), "X"])


def test_statement_exchange_rates_bridge_a_gap_of_a_few_weeks():
    from financetoolkit.currencies_model import get_statement_exchange_rates

    # A sparse pair quoted for the last time three weeks before the period end.
    daily = pd.DataFrame(
        {"EURUSD=X": [1.1, 1.2]},
        index=pd.to_datetime(["2022-11-01", "2022-12-10"]),
    )
    period_ends = pd.DataFrame(
        [[pd.Timestamp("2022-12-31")]],
        index=["X"],
        columns=pd.PeriodIndex(["2022"], freq="Y"),
    )
    closing = get_statement_exchange_rates(
        daily, period_ends, pd.Series({"X": "EURUSD=X"}), False, "closing"
    )

    assert closing.loc[pd.Period("2022", "Y"), "X"] == 1.2


def test_yahoo_balance_sheet_maps_every_working_capital_item():
    # The cash-based operating profitability needs each of these balances, so a Yahoo
    # Finance statement maps them as a FinancialModelingPrep statement does.
    balance_yf = normalization_model.read_normalization_file("balance_yf")

    assert balance_yf.loc["PrepaidAssets"] == "Prepaids"
    assert {
        "Total Assets",
        "Net Receivables",
        "Inventory",
        "Prepaids",
        "Deferred Revenue",
        "Deferred Revenue Non Current",
        "Accounts Payable",
        "Accrued Expenses",
    } <= set(balance_yf.to_numpy())
