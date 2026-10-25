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


def _fiscal_exchange_rate_toolkit():
    toolkit = Toolkit(
        tickers=["AAPL"],
        income=income_dataset.loc[["AAPL"]],
        convert_currency=False,
        start_date="2019-12-31",
        end_date="2023-01-01",
        sleep_timer=False,
    )
    days = pd.period_range("2019-01-01", "2023-12-31", freq="D")
    rates = pd.Series(range(len(days)), index=days, dtype=float) / 1000 + 1
    toolkit._daily_exchange_rate_data = pd.DataFrame({("Adj Close", "EURUSD=X"): rates})
    toolkit._statement_currencies = pd.Series({"AAPL": "EURUSD=X"})
    # Apple's fiscal years end on the last Saturday of September.
    toolkit._statistics_statement = pd.DataFrame(
        [["2020-09-26", "2021-09-25", "2022-09-24", "2023-09-30"]],
        index=pd.MultiIndex.from_tuples([("AAPL", "Period End Date")]),
        columns=income_dataset.columns,
    )
    return toolkit, rates.set_axis(days.to_timestamp())


def test_toolkit_statement_exchange_rates_use_the_fiscal_period():
    toolkit, rates = _fiscal_exchange_rate_toolkit()
    statement = toolkit._income_statement

    closing = toolkit._get_statement_exchange_rates(statement, rate="closing")["AAPL"]
    average = toolkit._get_statement_exchange_rates(statement, rate="average")["AAPL"]
    opening = toolkit._get_statement_exchange_rates(statement, rate="opening")["AAPL"]

    # The closing rate is the rate on the fiscal year end, not on 31 December.
    assert closing[pd.Period("2021", "Y")] == pytest.approx(rates["2021-09-25"])
    # The average runs from the day after the previous fiscal year end to this one.
    assert average[pd.Period("2021", "Y")] == pytest.approx(
        rates["2020-09-27":"2021-09-25"].mean()
    )
    # The first period has no previous period, so its window is the year before its end.
    assert average[pd.Period("2020", "Y")] == pytest.approx(
        rates["2019-09-27":"2020-09-26"].mean()
    )
    assert opening[pd.Period("2022", "Y")] == pytest.approx(rates["2021-09-25"])


def test_toolkit_period_end_dates_fall_back_to_the_label():
    toolkit, _ = _fiscal_exchange_rate_toolkit()
    toolkit._statistics_statement = pd.DataFrame()
    toolkit._fiscal_year_adjustments = {
        "AAPL": [{"fiscal_year": 2023, "calendar_year": 2022}]
    }

    end_dates = toolkit._get_period_end_dates(toolkit._income_statement).loc["AAPL"]

    assert end_dates[pd.Period("2021", "Y")] == pd.Timestamp("2021-12-31")
    # A label moved back a year ends at the latest on 31 May of the next year.
    assert end_dates[pd.Period("2022", "Y")] == pd.Timestamp("2023-05-31")


def test_toolkit_statement_exchange_rates_are_not_rounded(monkeypatch):
    from financetoolkit import toolkit_controller

    captured = {}

    def fake_historical_data(**kwargs):
        captured["rounding"] = kwargs["rounding"]
        days = pd.period_range("2022-01-01", "2022-01-10", freq="D")
        frame = pd.DataFrame(
            {
                (column, "KRWUSD=X"): [0.000771234] * len(days)
                for column in ["Open", "High", "Low", "Close", "Adj Close"]
            },
            index=days,
        )
        return frame, []

    monkeypatch.setattr(
        toolkit_controller, "_get_historical_data", fake_historical_data
    )

    toolkit = Toolkit(
        tickers=["AAPL"],
        income=income_dataset.loc[["AAPL"]],
        convert_currency=False,
        start_date="2022-01-01",
        end_date="2022-01-10",
        sleep_timer=False,
    )
    toolkit._currencies = ["KRWUSD=X"]

    returned = toolkit.get_exchange_rates()

    # Fetched unrounded for the statement translation; only the returned rates are rounded.
    assert captured["rounding"] is None
    assert (
        toolkit._daily_exchange_rate_data[("Adj Close", "KRWUSD=X")].iloc[0]
        == 0.000771234
    )
    assert returned["Adj Close"].iloc[0] == 0.0008


def _counting_toolkit(test_toolkit):
    """
    A Toolkit without data whose retrieval methods record what they retrieve and take it
    from the test Toolkit, so a test can tell what a module loads.
    """
    toolkit = Toolkit(
        tickers=["AAPL", "MSFT"],
        api_key="",
        convert_currency=False,
        start_date="2019-12-31",
        end_date="2023-01-01",
        sleep_timer=False,
    )
    calls = []

    def statement_getter(name):
        def getter(*args, **kwargs):
            calls.append(name)
            setattr(toolkit, f"_{name}", getattr(test_toolkit, f"_{name}"))

        return getter

    for name in ("balance_sheet_statement", "income_statement", "cash_flow_statement"):
        setattr(toolkit, f"get_{name}", statement_getter(name))

    def get_historical_data(period="daily", **kwargs):
        calls.append(f"{period} prices")
        test_toolkit.get_historical_data(period=period)
        for attribute in (
            "_daily_historical_data",
            "_daily_risk_free_rate",
            f"_{period}_historical_data",
            f"_{period}_risk_free_rate",
        ):
            setattr(toolkit, attribute, getattr(test_toolkit, attribute))

    def get_intraday_data(*args, **kwargs):
        calls.append("intraday prices")

    toolkit.get_historical_data = get_historical_data
    toolkit.get_intraday_data = get_intraday_data

    return toolkit, calls


def test_module_access_loads_nothing(test_toolkit):
    toolkit, calls = _counting_toolkit(test_toolkit)
    toolkit._intraday_period = "1hour"

    for module in (
        "ratios",
        "models",
        "options",
        "technicals",
        "performance",
        "risk",
        "econometrics",
    ):
        getattr(toolkit, module)

    assert calls == []


def test_ratio_loads_only_the_statement_it_uses(test_toolkit):
    toolkit, calls = _counting_toolkit(test_toolkit)

    gross_margin = toolkit.ratios.get_gross_margin()

    assert calls == ["income_statement"]
    pd.testing.assert_frame_equal(gross_margin, test_toolkit.ratios.get_gross_margin())


def test_valuation_ratio_loads_prices(test_toolkit):
    toolkit, calls = _counting_toolkit(test_toolkit)

    price_to_book = toolkit.ratios.get_price_to_book_ratio()

    assert "yearly prices" in calls
    assert "cash_flow_statement" not in calls
    pd.testing.assert_frame_equal(
        price_to_book, test_toolkit.ratios.get_price_to_book_ratio()
    )


def test_weekly_metric_loads_only_the_weekly_period(test_toolkit):
    toolkit, calls = _counting_toolkit(test_toolkit)
    toolkit._intraday_period = "1hour"

    performance = toolkit.performance
    within_weeks = performance.get_sharpe_ratio(period="weekly")
    rolling_weeks = performance.get_sharpe_ratio(period="weekly", rolling=4)

    # The daily prices are the base the weekly prices are converted from; no other period
    # and no intraday data is retrieved.
    assert set(calls) == {"weekly prices", "daily prices"}
    assert set(performance._historical_data) == {"weekly", "daily"}
    assert set(performance._within_historical_data) == {"weekly"}
    pd.testing.assert_frame_equal(
        within_weeks, test_toolkit.performance.get_sharpe_ratio(period="weekly")
    )
    pd.testing.assert_frame_equal(
        rolling_weeks,
        test_toolkit.performance.get_sharpe_ratio(period="weekly", rolling=4),
    )


def test_repeated_module_access_does_not_retrieve_again(test_toolkit):
    toolkit, calls = _counting_toolkit(test_toolkit)
    # The Models module requires an API key to retrieve statements.
    toolkit._api_key = "API_KEY"

    toolkit.ratios.get_gross_margin()
    toolkit.ratios.get_gross_margin()
    altman_z_score = toolkit.models.get_altman_z_score()

    # The Toolkit keeps the statements it retrieved, so only the first use retrieves one.
    assert calls.count("income_statement") == 1
    pd.testing.assert_frame_equal(
        altman_z_score, test_toolkit.models.get_altman_z_score()
    )


def test_collection_without_statements_fails_once():
    toolkit = Toolkit(
        tickers=["QQQ"],
        api_key="",
        start_date="2019-12-31",
        end_date="2023-01-01",
        sleep_timer=False,
    )
    calls = []

    for name in ("balance_sheet_statement", "income_statement", "cash_flow_statement"):
        setattr(toolkit, f"get_{name}", lambda name=name: calls.append(name))

    assert toolkit.ratios.get_gross_margin().empty
    calls.clear()

    # The collection retrieves the statements once before its ratios rather than per ratio.
    assert toolkit.ratios.collect_profitability_ratios().empty
    assert sorted(calls) == [
        "balance_sheet_statement",
        "cash_flow_statement",
        "income_statement",
    ]
