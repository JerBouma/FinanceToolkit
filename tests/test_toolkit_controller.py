# ruff: noqa
"""Toolkit Controller Tests""" ""
import pandas as pd
import pytest

from financetoolkit import Toolkit

balance_dataset = pd.read_pickle("tests/datasets/balance_dataset.pickle")
income_dataset = pd.read_pickle("tests/datasets/income_dataset.pickle")
cash_dataset = pd.read_pickle("tests/datasets/cash_dataset.pickle")
historical_dataset = pd.read_pickle("tests/datasets/historical_dataset.pickle")
risk_free_rate = pd.read_pickle("tests/datasets/risk_free_rate.pickle")
treasury_data = pd.read_pickle("tests/datasets/treasury_data.pickle")


# pylint: disable=missing-function-docstring


def test_toolkit_balance(recorder):
    toolkit = Toolkit(
        tickers=["AAPL", "MSFT"],
        balance=balance_dataset,
        convert_currency=False,
        start_date="2019-12-31",
        end_date="2023-01-01",
        sleep_timer=False,
    )

    toolkit._daily_risk_free_rate = risk_free_rate
    toolkit._daily_treasury_data = treasury_data

    recorder.capture(toolkit.get_balance_sheet_statement())
    recorder.capture(toolkit.get_balance_sheet_statement(growth=True))
    recorder.capture(toolkit.get_balance_sheet_statement(growth=True, lag=[1, 2, 3]))


def test_toolkit_income(recorder):
    toolkit = Toolkit(
        tickers=["AAPL", "MSFT"],
        income=income_dataset,
        convert_currency=False,
        start_date="2019-12-31",
        end_date="2023-01-01",
        sleep_timer=False,
    )

    toolkit._daily_risk_free_rate = risk_free_rate
    toolkit._daily_treasury_data = treasury_data

    recorder.capture(toolkit.get_income_statement())
    recorder.capture(toolkit.get_income_statement(growth=True))
    recorder.capture(toolkit.get_income_statement(growth=True, lag=[1, 2, 3]))


def test_toolkit_cash(recorder):
    toolkit = Toolkit(
        tickers=["AAPL", "MSFT"],
        cash=cash_dataset,
        convert_currency=False,
        start_date="2019-12-31",
        end_date="2023-01-01",
        sleep_timer=False,
    )

    toolkit._daily_risk_free_rate = risk_free_rate
    toolkit._daily_treasury_data = treasury_data

    recorder.capture(toolkit.get_cash_flow_statement())
    recorder.capture(toolkit.get_cash_flow_statement(growth=True))
    recorder.capture(toolkit.get_cash_flow_statement(growth=True, lag=[1, 2, 3]))


def test_toolkit_historical(recorder):
    toolkit = Toolkit(
        tickers=["AAPL", "MSFT"],
        historical=historical_dataset,
        start_date="2019-12-31",
        end_date="2023-01-01",
        sleep_timer=False,
    )

    toolkit._daily_risk_free_rate = risk_free_rate
    toolkit._daily_treasury_data = treasury_data

    recorder.capture(toolkit.get_historical_data(period="yearly").round(0))


def test_toolkit_ratios(recorder):
    toolkit = Toolkit(
        tickers=["AAPL", "MSFT"],
        balance=balance_dataset,
        income=income_dataset,
        cash=cash_dataset,
        historical=historical_dataset,
        convert_currency=False,
        start_date="2019-12-31",
        end_date="2023-01-01",
        sleep_timer=False,
    )

    toolkit._daily_risk_free_rate = risk_free_rate
    toolkit._daily_treasury_data = treasury_data

    recorder.capture(toolkit.ratios.get_debt_to_assets_ratio())
    recorder.capture(toolkit.ratios.get_debt_to_assets_ratio(growth=True))
    recorder.capture(
        toolkit.ratios.get_debt_to_assets_ratio(growth=True, lag=[1, 2, 3])
    )


def test_toolkit_models(recorder):
    toolkit = Toolkit(
        tickers=["AAPL", "MSFT"],
        historical=historical_dataset,
        balance=balance_dataset,
        income=income_dataset,
        cash=cash_dataset,
        convert_currency=False,
        start_date="2019-12-31",
        end_date="2023-01-01",
        sleep_timer=False,
    )

    toolkit._daily_risk_free_rate = risk_free_rate
    toolkit._daily_treasury_data = treasury_data

    recorder.capture(toolkit.models.get_dupont_analysis())
    recorder.capture(toolkit.models.get_dupont_analysis(growth=True))
    recorder.capture(toolkit.models.get_dupont_analysis(growth=True, lag=[1, 2, 3]))


def test_toolkit_performance(recorder):
    toolkit = Toolkit(
        tickers=["AAPL", "MSFT"],
        balance=balance_dataset,
        income=income_dataset,
        cash=cash_dataset,
        historical=historical_dataset,
        convert_currency=False,
        start_date="2019-12-31",
        end_date="2023-01-01",
        sleep_timer=False,
    )

    toolkit._daily_risk_free_rate = risk_free_rate
    toolkit._daily_treasury_data = treasury_data

    recorder.capture(toolkit.performance.get_beta())
    recorder.capture(toolkit.performance.get_beta(growth=True))
    recorder.capture(toolkit.performance.get_beta(growth=True, lag=[1, 2, 3]))


def test_toolkit_risk(recorder):
    toolkit = Toolkit(
        tickers=["AAPL", "MSFT"],
        balance=balance_dataset,
        income=income_dataset,
        cash=cash_dataset,
        historical=historical_dataset,
        convert_currency=False,
        start_date="2019-12-31",
        end_date="2023-01-01",
        sleep_timer=False,
    )

    toolkit._daily_risk_free_rate = risk_free_rate
    toolkit._daily_treasury_data = treasury_data

    recorder.capture(toolkit.risk.get_conditional_value_at_risk())
    recorder.capture(
        toolkit.risk.get_conditional_value_at_risk(
            period="yearly", within_period=True, growth=True
        )
    )
    recorder.capture(
        toolkit.risk.get_conditional_value_at_risk(
            period="yearly", within_period=True, growth=True, lag=[1, 2, 3]
        )
    )


def test_toolkit_technicals(recorder):
    toolkit = Toolkit(
        tickers=["AAPL", "MSFT"],
        balance=balance_dataset,
        income=income_dataset,
        cash=cash_dataset,
        historical=historical_dataset,
        convert_currency=False,
        start_date="2019-12-31",
        end_date="2023-01-01",
        sleep_timer=False,
    )

    toolkit._daily_risk_free_rate = risk_free_rate
    toolkit._daily_treasury_data = treasury_data

    recorder.capture(toolkit.technicals.collect_all_indicators().round(0))
    recorder.capture(toolkit.technicals.collect_all_indicators(growth=True).round(0))
    recorder.capture(
        toolkit.technicals.collect_all_indicators(growth=True, lag=[1, 2, 3]).round(0)
    )


def test_toolkit_converts_each_period_once(monkeypatch):
    """Module access asks for every period, which is converted once per daily data."""
    from financetoolkit import toolkit_controller

    conversions = []
    convert = toolkit_controller._convert_daily_to_other_period

    def counted_convert(**kwargs):
        conversions.append(kwargs["period"])
        return convert(**kwargs)

    monkeypatch.setattr(
        toolkit_controller, "_convert_daily_to_other_period", counted_convert
    )

    toolkit = Toolkit(
        tickers=["AAPL", "MSFT"],
        historical=historical_dataset,
        start_date="2019-12-31",
        end_date="2023-01-01",
        sleep_timer=False,
    )
    toolkit._daily_risk_free_rate = risk_free_rate
    toolkit._daily_treasury_data = treasury_data

    # The first request also converts the treasury data to the period.
    first = toolkit.get_historical_data(period="monthly")
    conversions_after_first = len(conversions)
    toolkit.get_historical_data(period="monthly")
    assert len(conversions) == conversions_after_first

    # A different rounding is a different conversion.
    rounded = toolkit.get_historical_data(period="monthly", rounding=0)
    assert len(conversions) == conversions_after_first + 1
    assert not rounded.equals(first)


def test_toolkit_cash_flow_trailing_keeps_the_cash_balances():
    """A trailing window starts with the first period's cash and ends with the last's."""
    toolkit = Toolkit(
        tickers=["AAPL", "MSFT"],
        cash=cash_dataset,
        convert_currency=False,
        start_date="2017-12-31",
        end_date="2023-01-01",
        sleep_timer=False,
    )

    statement = toolkit.get_cash_flow_statement(rounding=None).loc["AAPL"]
    trailing = toolkit.get_cash_flow_statement(trailing=2, rounding=None).loc["AAPL"]

    periods = statement.columns
    for previous, current in zip(periods[:-1], periods[1:]):
        assert (
            trailing.loc["Cash End of Period", current]
            == statement.loc["Cash End of Period", current]
        )
        assert (
            trailing.loc["Cash Beginning of Period", current]
            == statement.loc["Cash Beginning of Period", previous]
        )
        assert trailing.loc["Net Change in Cash", current] == (
            statement.loc["Net Change in Cash", [previous, current]].sum()
        )

    assert pd.isna(trailing.loc["Cash End of Period", periods[0]])


def test_toolkit_income_trailing_does_not_sum_the_tax_rate():
    toolkit = Toolkit(
        tickers=["AAPL", "MSFT"],
        income=income_dataset,
        convert_currency=False,
        start_date="2017-12-31",
        end_date="2023-01-01",
        sleep_timer=False,
    )
    statement = toolkit.get_income_statement()

    # Yahoo Finance statements hold a Tax Rate for Calcs row, a ratio.
    income = toolkit._income_statement
    tax_rate = income.xs("Net Income", level=1, drop_level=False) * 0 + 0.21
    tax_rate.index = pd.MultiIndex.from_tuples(
        [(ticker, "Tax Rate for Calcs") for ticker, _ in tax_rate.index]
    )
    toolkit._income_statement = pd.concat([income, tax_rate])

    trailing = toolkit.get_income_statement(trailing=2).loc["AAPL"]

    assert trailing.loc["Tax Rate for Calcs"].dropna().eq(0.21).all()
    assert len(trailing.loc["Tax Rate for Calcs"].dropna()) > 0
    assert (
        trailing.loc["Weighted Average Shares"].iloc[-1]
        == statement.loc["AAPL"].loc["Weighted Average Shares"].iloc[-1]
    )
    assert trailing.loc["Revenue"].iloc[-1] == pytest.approx(
        statement.loc["AAPL"].loc["Revenue"].iloc[-2:].sum()
    )


def test_toolkit_period_risk_free_rate_is_the_average_daily_yield():
    """The yield of a period is the average of its daily yields, not its last one."""
    toolkit = Toolkit(
        tickers=["AAPL", "MSFT"],
        historical=historical_dataset,
        start_date="2019-12-31",
        end_date="2023-01-01",
        sleep_timer=False,
    )
    toolkit._daily_risk_free_rate = risk_free_rate
    toolkit._daily_treasury_data = treasury_data

    treasury = toolkit.get_treasury_data(period="quarterly", rounding=None)
    daily_yield = treasury_data.xs("10 Year", level=1, axis=1)["Adj Close"]
    # Rounded like every other Toolkit output (4 decimals by default).
    expected = daily_yield.groupby(daily_yield.index.asfreq("Q")).mean().round(4)

    period_rate = toolkit._quarterly_risk_free_rate["Adj Close"]
    pd.testing.assert_series_equal(
        period_rate, expected.reindex(period_rate.index), check_names=False
    )
    # The treasury data itself keeps the closing yield of each period.
    last_yield = daily_yield.groupby(daily_yield.index.asfreq("Q")).last()
    pd.testing.assert_series_equal(
        treasury["Adj Close", "10 Year"],
        last_yield.reindex(treasury.index),
        check_names=False,
    )


def test_toolkit_statement_exchange_rates_are_period_averages():
    toolkit = Toolkit(
        tickers=["AAPL"],
        income=income_dataset,
        convert_currency=False,
        start_date="2019-12-31",
        end_date="2023-01-01",
        sleep_timer=False,
    )
    toolkit._daily_exchange_rate_data = pd.DataFrame(
        {("Adj Close", "EURUSD=X"): [1.0, 1.2, 1.1, 1.3]},
        index=pd.PeriodIndex(
            ["2021-03-01", "2021-12-31", "2022-01-03", "2022-06-30"], freq="D"
        ),
    )

    average_rates = toolkit._get_average_statement_exchange_rates()

    assert average_rates["EURUSD=X"].tolist() == pytest.approx([1.1, 1.2])
    assert [str(period) for period in average_rates.index] == ["2021", "2022"]
