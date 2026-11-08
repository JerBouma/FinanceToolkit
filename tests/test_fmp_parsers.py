"""Offline tests of how the FMP responses of the company datasets are parsed."""

import numpy as np
import pandas as pd
import pytest

from financetoolkit import fmp_model


def _answer(monkeypatch, response):
    """Answer every FMP request with the given response, raw or as a DataFrame."""

    def fake(
        url, sleep_timer=False, raw=False, user_subscription="Free"
    ):  # noqa: ARG001
        return response if raw else pd.DataFrame(response)

    monkeypatch.setattr(fmp_model, "get_financial_data", fake)


def test_annual_esg_scores_take_the_latest_disclosure_of_a_year(monkeypatch):
    _answer(
        monkeypatch,
        [
            # Not in date order, as a response can be.
            {"date": "2023-11-01", "environmentalScore": 60, "socialScore": 50,
             "governanceScore": 70, "ESGScore": 60},
            {"date": "2023-02-01", "environmentalScore": 57, "socialScore": 48,
             "governanceScore": 66, "ESGScore": 57},
            {"date": "2022-11-01", "environmentalScore": 55, "socialScore": 47,
             "governanceScore": 65, "ESGScore": 55},
        ],
    )  # fmt: skip

    scores, no_data = fmp_model.get_esg_scores("AAPL", api_key="KEY")

    assert no_data == []
    environmental = scores[("Environmental Score", "AAPL")]
    assert environmental[pd.Period("2023", freq="Y")] == 60
    assert environmental[pd.Period("2022", freq="Y")] == 55


def test_revenue_segments_keep_their_names_and_missing_periods(monkeypatch):
    _answer(
        monkeypatch,
        [
            {"date": "2024-09-28", "data": {"iPhone": 201.0, "Services": 96.0}},
            {"date": "2023-09-30", "data": {"iPhone": 200.0, "services": 85.0}},
            {"date": "2020-09-26", "data": {"iPhone": 137.0, "iPod": 1.0}},
        ],
    )

    segments, no_data = fmp_model.get_revenue_segmentation(
        "AAPL", method="product", api_key="KEY"
    )

    assert no_data == []
    segments = segments.loc["AAPL"]
    # "iPhone" is not turned into "Iphone", and "services" is the same segment as "Services".
    assert set(segments.index) == {"iPhone", "Services", "iPod"}
    assert segments.loc["Services", pd.Period("2023", freq="Y")] == 85.0
    # A period without the segment is missing, not zero.
    assert np.isnan(segments.loc["iPod", pd.Period("2024", freq="Y")])
    assert segments.loc["iPod", pd.Period("2020", freq="Y")] == 1.0


def test_an_unknown_segmentation_method_names_the_valid_ones():
    with pytest.raises(ValueError, match="'geographic' or 'product'"):
        fmp_model.get_revenue_segmentation("AAPL", method="income", api_key="KEY")


@pytest.mark.parametrize("actual_dates", [True, False])
def test_the_earnings_calendar_returns_dates_or_quarters(monkeypatch, actual_dates):
    _answer(
        monkeypatch,
        [
            {"date": "2024-05-02", "epsActual": 1.53, "epsEstimated": 1.5,
             "revenueActual": 90.8e9, "revenueEstimated": 90.3e9, "lastUpdated": "2024-05-03"},
            {"date": "2024-02-01", "epsActual": 2.18, "epsEstimated": 2.1,
             "revenueActual": 119.6e9, "revenueEstimated": 117.9e9, "lastUpdated": "2024-02-02"},
        ],
    )  # fmt: skip

    calendar, no_data = fmp_model.get_earnings_calendar(
        "AAPL", api_key="KEY", actual_dates=actual_dates
    )

    assert no_data == []
    dates = calendar.loc["AAPL"].index
    if actual_dates:
        assert list(dates) == [pd.Timestamp("2024-02-01"), pd.Timestamp("2024-05-02")]
    else:
        assert list(dates) == [pd.Period("2024Q1"), pd.Period("2024Q2")]
    assert calendar.loc["AAPL", "EPS"].tolist() == [2.18, 1.53]


def test_the_dividend_yield_is_a_decimal(monkeypatch):
    _answer(
        monkeypatch,
        [
            {"date": "2024-08-12", "adjDividend": 0.25, "dividend": 0.25, "yield": 0.44,
             "recordDate": "2024-08-12", "paymentDate": "2024-08-15",
             "declarationDate": "2024-08-01"},
            {"date": "2024-05-10", "adjDividend": 0.25, "dividend": 0.25, "yield": 0.52,
             "recordDate": "2024-05-13", "paymentDate": "2024-05-16",
             "declarationDate": "2024-05-02"},
        ],
    )  # fmt: skip

    calendar, no_data = fmp_model.get_dividend_calendar("AAPL", api_key="KEY")

    assert no_data == []
    assert calendar.loc["AAPL", "Yield"].tolist() == pytest.approx([0.0052, 0.0044])
    assert list(calendar.loc["AAPL"].index) == [
        pd.Period("2024-05-10", freq="D"),
        pd.Period("2024-08-12", freq="D"),
    ]


def test_analyst_estimates_count_the_analysts_and_stay_unrounded(monkeypatch):
    estimate = {
        "symbol": "AAPL",
        "revenueLow": 1.0e11, "revenueHigh": 1.2e11, "revenueAvg": 1.1e11,
        "ebitdaLow": 3.0e10, "ebitdaHigh": 3.5e10, "ebitdaAvg": 3.2e10,
        "ebitLow": 2.0e10, "ebitHigh": 2.5e10, "ebitAvg": 2.2e10,
        "netIncomeLow": 1.5e10, "netIncomeHigh": 1.9e10, "netIncomeAvg": 1.7e10,
        "sgaExpenseLow": 1.0e10, "sgaExpenseHigh": 1.1e10, "sgaExpenseAvg": 1.05e10,
        "epsAvg": 6.987654, "epsHigh": 7.2, "epsLow": 6.7,
        "numAnalystsRevenue": 30, "numAnalystsEps": 26,
    }  # fmt: skip
    _answer(monkeypatch, [{**estimate, "date": "2025-09-27"}])

    estimates, no_data = fmp_model.get_analyst_estimates(
        "AAPL", api_key="KEY", rounding=None
    )

    assert no_data == []
    period = pd.Period("2025", freq="Y")
    assert estimates.loc[("AAPL", "Number of Analysts Revenue"), period] == 30
    assert estimates.loc[("AAPL", "Number of Analysts EPS"), period] == 26
    assert estimates.loc[("AAPL", "Estimated EPS Average"), period] == 6.987654


def test_analyst_estimates_are_labelled_like_the_statements(monkeypatch):
    estimate = {
        "symbol": "NVDA",
        "epsAvg": 9.29,
        "numAnalystsRevenue": 41,
        "numAnalystsEps": 33,
    }
    _answer(monkeypatch, [{**estimate, "date": "2027-01-31"}])

    estimates, _ = fmp_model.get_analyst_estimates("NVDA", api_key="KEY")

    # A fiscal year ending in January is mostly the previous calendar year, as in the statements.
    assert list(estimates.columns) == [pd.Period("2026", freq="Y")]
