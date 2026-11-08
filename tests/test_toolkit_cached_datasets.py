"""The datasets a Toolkit keeps between calls answer later calls with other settings."""

import pandas as pd

from financetoolkit import Toolkit, toolkit_controller


def _toolkit() -> Toolkit:
    toolkit = Toolkit(
        tickers=["AAPL"],
        start_date="2023-01-01",
        end_date="2024-12-31",
        convert_currency=False,
        sleep_timer=False,
    )
    toolkit._api_key = "KEY"
    toolkit._analyst_estimates = pd.DataFrame()
    toolkit._analyst_estimates_growth = pd.DataFrame()
    toolkit._earnings_calendar = pd.DataFrame()
    toolkit._earnings_calendar_actual_dates = None
    return toolkit


def test_the_earnings_calendar_follows_actual_dates(monkeypatch):
    toolkit = _toolkit()
    requested = []

    def collect(
        dataset, tickers, ticker_axis, collector, parameters=None
    ):  # noqa: ARG001
        requested.append(parameters["actual_dates"])
        date = (
            pd.Timestamp("2024-05-02")
            if parameters["actual_dates"]
            else pd.Period("2024Q2")
        )
        return (
            pd.DataFrame(
                {"EPS": [1.53]}, index=pd.MultiIndex.from_tuples([("AAPL", date)])
            ),
            [],
        )

    monkeypatch.setattr(toolkit, "_collect_per_ticker", collect)

    dates = toolkit.get_earnings_calendar(actual_dates=True).index
    quarters = toolkit.get_earnings_calendar(actual_dates=False).index
    toolkit.get_earnings_calendar(actual_dates=False)

    assert list(dates) == [pd.Timestamp("2024-05-02")]
    assert list(quarters) == [pd.Period("2024Q2")]
    # Retrieved again only when the choice changes.
    assert requested == [True, False]


def test_analyst_estimates_round_on_every_call(monkeypatch):
    toolkit = _toolkit()
    period = pd.Period("2025", freq="Y")

    def collect(
        dataset, tickers, ticker_axis, collector, parameters=None
    ):  # noqa: ARG001
        estimates = pd.DataFrame(
            {period: [6.987654]},
            index=pd.MultiIndex.from_tuples([("AAPL", "Estimated EPS Average")]),
        )
        return estimates, []

    monkeypatch.setattr(toolkit, "_collect_per_ticker", collect)

    assert (
        toolkit.get_analyst_estimates(rounding=0).loc["Estimated EPS Average", period]
        == 7.0
    )
    assert (
        toolkit.get_analyst_estimates(rounding=4).loc["Estimated EPS Average", period]
        == 6.9877
    )


def test_statements_are_kept_unrounded(monkeypatch):
    toolkit = _toolkit()
    period = pd.Period("2024", freq="Y")
    roundings = []

    def collect(*args, rounding=None, **kwargs):  # noqa: ARG001
        roundings.append(rounding)
        statement = pd.DataFrame(
            {period: [6.08734]},
            index=pd.MultiIndex.from_tuples([("AAPL", "EPS Diluted")]),
        )
        return statement, pd.DataFrame(), [], {}

    monkeypatch.setattr(toolkit_controller, "collect_financial_statements", collect)

    assert toolkit.get_income_statement(rounding=0).loc["EPS Diluted", period] == 6.0
    # What the ratios and models read is the figure as reported, whatever the rounding asked for.
    assert roundings == [None]
    assert toolkit._income_statement.loc[("AAPL", "EPS Diluted"), period] == 6.08734
    assert toolkit.get_income_statement(rounding=2).loc["EPS Diluted", period] == 6.09
