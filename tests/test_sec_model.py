"""Tests for the first reported figures from the SEC's XBRL company facts."""

import pandas as pd
import pytest

from financetoolkit import sec_model


def fact(val, end, filed, form, start=None, fy=None, fp=None):
    entry = {
        "end": end,
        "val": val,
        "accn": f"acc-{filed}",
        "form": form,
        "filed": filed,
    }
    if start is not None:
        entry["start"] = start
    if fy is not None:
        entry["fy"], entry["fp"] = fy, fp
    return entry


def company_facts():
    """A trimmed companyfacts document with a fiscal year that ends in December."""
    revenue = [
        # Fiscal 2021 under the old concept, below; fiscal 2022 first reported at 100 and
        # restated to 110 in the 10-K of fiscal 2023, which repeats it as a comparative.
        fact(100, "2022-12-31", "2023-02-15", "10-K", start="2022-01-01"),
        fact(110, "2022-12-31", "2024-02-15", "10-K", start="2022-01-01"),
        fact(120, "2023-12-31", "2024-02-15", "10-K", start="2023-01-01"),
        # The quarters of fiscal 2023: Q1 to Q3 on their own, Q4 only within the year.
        fact(25, "2023-03-31", "2023-05-01", "10-Q", start="2023-01-01"),
        fact(28, "2023-06-30", "2023-08-01", "10-Q", start="2023-04-01"),
        fact(30, "2023-09-30", "2023-11-01", "10-Q", start="2023-07-01"),
        fact(83, "2023-09-30", "2023-11-01", "10-Q", start="2023-01-01"),
    ]
    cash_flow = [
        # Cash flows are only reported year to date.
        fact(10, "2023-03-31", "2023-05-01", "10-Q", start="2023-01-01"),
        fact(22, "2023-06-30", "2023-08-01", "10-Q", start="2023-01-01"),
        fact(35, "2023-09-30", "2023-11-01", "10-Q", start="2023-01-01"),
        # The 6M figure was restated in the 10-Q of the third quarter.
        fact(21, "2023-06-30", "2023-11-01", "10-Q", start="2023-01-01"),
        fact(50, "2023-12-31", "2024-02-15", "10-K", start="2023-01-01"),
    ]
    assets = [
        fact(500, "2022-12-31", "2023-02-15", "10-K"),
        fact(505, "2023-03-31", "2023-05-01", "10-Q"),
        fact(520, "2023-12-31", "2024-02-15", "10-K"),
        fact(500, "2022-12-31", "2024-02-15", "10-K"),
    ]
    eps = [
        fact(1.5, "2023-03-31", "2023-05-01", "10-Q", start="2023-01-01"),
        fact(6.0, "2023-12-31", "2024-02-15", "10-K", start="2023-01-01"),
        fact(4.0, "2023-09-30", "2023-11-01", "10-Q", start="2023-01-01"),
    ]
    net_income = [
        # A loss that was restated to a larger loss.
        fact(-40, "2022-12-31", "2023-02-15", "10-K", start="2022-01-01"),
        fact(-50, "2022-12-31", "2024-02-15", "10-K", start="2022-01-01"),
        # A registration statement repeats figures but is not a 10-K or 10-Q.
        fact(-45, "2022-12-31", "2023-06-01", "S-1", start="2022-01-01"),
    ]

    return {
        "cik": 1,
        "entityName": "Example Corp",
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {"USD": revenue}
                },
                "SalesRevenueNet": {
                    "units": {
                        "USD": [
                            fact(
                                90,
                                "2021-12-31",
                                "2022-02-15",
                                "10-K",
                                start="2021-01-01",
                            ),
                            # The old concept also reports 2022, which the new one covers.
                            fact(
                                999,
                                "2022-12-31",
                                "2023-02-15",
                                "10-K",
                                start="2022-01-01",
                            ),
                        ]
                    }
                },
                "NetCashProvidedByUsedInOperatingActivities": {
                    "units": {"USD": cash_flow}
                },
                "Assets": {"units": {"USD": assets}},
                "EarningsPerShareDiluted": {"units": {"USD/shares": eps}},
                "NetIncomeLoss": {"units": {"USD": net_income}},
            }
        },
    }


def figure(figures, item, period):
    return figures.loc[(item, pd.Period(period, freq="Q" if "Q" in period else "Y"))]


def test_a_restated_year_keeps_its_first_reported_value():
    figures = sec_model.get_first_reported_figures(company_facts())

    revenue = figure(figures, "Revenue", "2022")

    assert revenue["First Reported"] == 100
    assert revenue["First Filed"] == pd.Timestamp("2023-02-15")
    assert revenue["Latest"] == 110
    assert revenue["Latest Filed"] == pd.Timestamp("2024-02-15")
    assert revenue["Restatement"] == pytest.approx(0.1)


def test_an_unrestated_figure_repeated_as_comparative_shows_no_restatement():
    figures = sec_model.get_first_reported_figures(company_facts())

    assets = figure(figures, "Total Assets", "2022")

    assert assets["First Reported"] == assets["Latest"] == 500
    assert assets["First Filed"] == pd.Timestamp("2023-02-15")
    assert assets["Restatement"] == 0


def test_an_older_concept_only_fills_the_periods_the_newer_one_lacks():
    figures = sec_model.get_first_reported_figures(company_facts())

    assert figure(figures, "Revenue", "2021")["First Reported"] == 90
    # 2022 comes from the newer concept, not the 999 of the older one.
    assert figure(figures, "Revenue", "2022")["First Reported"] == 100


def test_a_restated_loss_is_measured_against_its_absolute_value():
    figures = sec_model.get_first_reported_figures(company_facts())

    net_income = figure(figures, "Net Income", "2022")

    # A registration statement does not count as a report of the figure.
    assert net_income["First Reported"] == -40
    assert net_income["Restatement"] == pytest.approx(-0.25)


def test_the_fourth_quarter_is_the_year_less_the_first_nine_months():
    figures = sec_model.get_first_reported_figures(company_facts(), quarterly=True)

    fourth_quarter = figure(figures, "Revenue", "2023Q4")

    assert fourth_quarter["First Reported"] == 120 - 83
    assert fourth_quarter["First Filed"] == pd.Timestamp("2024-02-15")
    assert figure(figures, "Revenue", "2023Q2")["First Reported"] == 28


def test_quarterly_cash_flows_are_taken_from_the_year_to_date_figures():
    figures = sec_model.get_first_reported_figures(company_facts(), quarterly=True)

    assert figure(figures, "Operating Cash Flow", "2023Q1")["First Reported"] == 10

    # The 6M figure was 22 when it was first filed, so the second quarter was first
    # reported as 12; its restatement to 21 makes the latest second quarter 11.
    second_quarter = figure(figures, "Operating Cash Flow", "2023Q2")
    assert second_quarter["First Reported"] == 12
    assert second_quarter["Latest"] == 11

    # The third quarter was first reported on the day the 6M figure was restated, so it
    # is measured against the restated 21.
    assert figure(figures, "Operating Cash Flow", "2023Q3")["First Reported"] == 14
    assert figure(figures, "Operating Cash Flow", "2023Q4")["First Reported"] == 15


def test_a_per_share_quarter_inside_a_year_to_date_figure_is_left_out():
    figures = sec_model.get_first_reported_figures(company_facts(), quarterly=True)

    assert figure(figures, "EPS Diluted", "2023Q1")["First Reported"] == 1.5
    assert ("EPS Diluted", pd.Period("2023Q4", freq="Q")) not in figures.index


def test_balances_are_taken_at_every_quarter_end():
    figures = sec_model.get_first_reported_figures(company_facts(), quarterly=True)

    assert figure(figures, "Total Assets", "2023Q1")["First Reported"] == 505
    assert figure(figures, "Total Assets", "2023Q4")["First Reported"] == 520


def test_as_of_a_date_only_uses_the_filings_made_by_then():
    figures = sec_model.get_first_reported_figures(company_facts(), as_of="2023-12-31")

    revenue = figure(figures, "Revenue", "2022")

    # The restatement was filed in 2024, so on 2023-12-31 the figure was still 100.
    assert revenue["Latest"] == 100
    assert revenue["Restatement"] == 0
    assert ("Revenue", pd.Period("2023", freq="Y")) not in figures.index


def test_a_company_without_us_gaap_dollar_figures_has_no_data():
    facts = {"facts": {"ifrs-full": {"Revenue": {"units": {"EUR": []}}}}}

    assert sec_model.get_first_reported_figures(facts).empty


def test_requests_need_a_user_agent_with_an_email_address(monkeypatch):
    monkeypatch.delenv(sec_model.USER_AGENT_ENVIRONMENT_VARIABLE, raising=False)

    with pytest.raises(ValueError, match="FINANCETOOLKIT_SEC_USER_AGENT"):
        sec_model.get_user_agent()

    monkeypatch.setenv(
        sec_model.USER_AGENT_ENVIRONMENT_VARIABLE, "Jane Doe jane@example.com"
    )
    assert sec_model.get_user_agent() == "Jane Doe jane@example.com"


def test_the_cik_numbers_are_read_from_the_ticker_list(monkeypatch):
    monkeypatch.setattr(sec_model, "get_active_cache", lambda: None)
    monkeypatch.setattr(
        sec_model,
        "get_sec_json",
        lambda url: {
            "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
            "1": {"cik_str": 1067983, "ticker": "BRK-B", "title": "Berkshire Hathaway"},
        },
    )

    assert sec_model.get_cik_numbers() == {"AAPL": 320193, "BRK-B": 1067983}


def test_the_company_facts_are_requested_by_ten_digit_cik(monkeypatch):
    requested = []
    monkeypatch.setattr(sec_model, "get_active_cache", lambda: None)
    monkeypatch.setattr(
        sec_model,
        "get_sec_json",
        lambda url: requested.append(url) or {"facts": {}},
    )

    sec_model.get_company_facts(320193)

    assert requested == [
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"
    ]


@pytest.mark.parametrize("point_in_time", [False, True])
def test_the_toolkit_returns_the_figures_per_ticker_within_its_dates(
    monkeypatch, point_in_time
):
    from financetoolkit import Toolkit

    monkeypatch.setattr(sec_model, "get_cik_numbers", lambda: {"EXMP": 1})
    monkeypatch.setattr(sec_model, "get_company_facts", lambda cik: company_facts())

    toolkit = Toolkit(
        tickers=["EXMP", "ASML.AS"],
        start_date="2022-01-01",
        end_date="2023-12-31",
        point_in_time=point_in_time,
        sleep_timer=False,
    )

    figures = toolkit.get_first_reported_figures()
    revenue = figures.sort_index().loc[("EXMP", "Revenue")]

    # 2021 is before the start date; 2023 was filed after the end date, which only
    # point-in-time mode leaves out, together with the 2024 restatement of 2022.
    assert pd.Period("2021", freq="Y") not in revenue.index
    assert (pd.Period("2023", freq="Y") in revenue.index) is not point_in_time
    assert revenue.loc[pd.Period("2022", freq="Y"), "Latest"] == (
        100 if point_in_time else 110
    )
    assert "ASML.AS" not in figures.index.get_level_values("Ticker")


def test_the_toolkit_reports_a_missing_user_agent_instead_of_failing(
    monkeypatch, caplog
):
    from financetoolkit import Toolkit

    monkeypatch.delenv(sec_model.USER_AGENT_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.setattr(sec_model, "get_active_cache", lambda: None)

    toolkit = Toolkit(
        tickers=["AAPL"],
        start_date="2022-01-01",
        end_date="2023-12-31",
        sleep_timer=False,
    )

    with caplog.at_level("ERROR"):
        figures = toolkit.get_first_reported_figures()

    assert figures.empty
    assert "FINANCETOOLKIT_SEC_USER_AGENT" in caplog.text
