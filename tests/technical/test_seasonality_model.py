"""Seasonality Model Tests: the calendar effects"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.technicals.seasonality import seasonality_model


@pytest.fixture(name="returns")
def fixture_returns():
    """Five years of daily returns with a lower return on Mondays."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2019-01-01", "2023-12-29")
    values = rng.normal(0.0004, 0.01, len(dates)) + np.where(
        dates.weekday == 0, -0.002, 0.0
    )
    return pd.Series(values, index=dates.to_period("D"))


def test_day_of_week_effect_against_statsmodels(returns):
    effect = seasonality_model.get_day_of_week_effect(returns, lags=6)

    # statsmodels 0.14.6: OLS(returns, weekday dummies).fit(cov_type="HAC",
    # cov_kwds={"maxlags": 6}), its params, bse and f_test of equal coefficients.
    np.testing.assert_allclose(
        effect["Average Return"].iloc[:5],
        [-0.0027415812, -0.0008500224, 0.0014667269, -0.0005193143, -0.0003328388],
        atol=1e-10,
    )
    np.testing.assert_allclose(
        effect["Standard Error"].iloc[:5],
        [0.0005917377, 0.0005713682, 0.000595872, 0.0005820714, 0.0005828176],
        atol=1e-10,
    )
    assert effect.loc["Equal Means", "Test Statistic"] == pytest.approx(
        6.43838756354554
    )
    assert effect.loc["Equal Means", "p-Value"] == pytest.approx(3.9516999729925653e-05)
    assert list(effect.index) == [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Equal Means",
    ]
    assert effect.loc["Equal Means", "Observations"] == len(returns)
    # The weekday averages are the plain means of their returns.
    assert effect.loc["Monday", "Average Return"] == pytest.approx(
        returns[returns.index.dayofweek == 0].mean()
    )


def test_turn_of_month_effect_against_statsmodels(returns):
    marked = seasonality_model.get_turn_of_month_days(returns.index)
    effect = seasonality_model.get_calendar_effect(
        returns, marked, "Turn of Month", lags=6
    )

    # statsmodels 0.14.6: OLS(returns, [1, turn of month]).fit(cov_type="HAC",
    # cov_kwds={"maxlags": 6}) and its t_test of the constant plus the slope.
    assert effect.loc["Other Days", "Average Return"] == pytest.approx(-0.0003989347)
    assert effect.loc["Difference", "Average Return"] == pytest.approx(-0.0010585516)
    assert effect.loc["Other Days", "Standard Error"] == pytest.approx(0.000314871)
    assert effect.loc["Difference", "Standard Error"] == pytest.approx(0.0006983095)
    assert effect.loc["Turn of Month", "Standard Error"] == pytest.approx(
        0.0006261977640254351
    )
    # January 2019 starts after its first business day (New Year's Day), so it is left out.
    assert effect["Observations"].tolist() == [240, 1064, 1304]


def test_turn_of_month_days_mark_the_last_and_first_days():
    dates = pd.bdate_range("2025-01-01", "2025-03-31")

    marked = seasonality_model.get_turn_of_month_days(dates)

    assert marked[marked == 1].index.strftime("%m-%d").tolist() == [
        "01-01",
        "01-02",
        "01-03",
        "01-31",
        "02-03",
        "02-04",
        "02-05",
        "02-28",
        "03-03",
        "03-04",
        "03-05",
        "03-31",
    ]
    # Ariel's window: the last trading day and the first nine.
    wide = seasonality_model.get_turn_of_month_days(dates, 1, 9)
    assert int(wide.loc["2025-02-01":"2025-02-28"].sum()) == 10  # noqa: PLR2004
    # A last month that is not over yet has no turn of the month.
    assert (
        seasonality_model.get_turn_of_month_days(dates[:-5]).loc["2025-03"].isna().all()
    )


def test_pre_holiday_days_read_the_holidays_from_the_dates():
    holidays = pd.to_datetime(["2025-11-27", "2025-12-25", "2026-01-01", "2026-01-19"])
    dates = pd.bdate_range("2025-11-03", "2026-01-30").drop(holidays)

    marked = seasonality_model.get_pre_holiday_days(dates)

    # The day before Thanksgiving, Christmas and New Year's Day, and the Friday before
    # Martin Luther King Jr. Day.
    assert marked[marked == 1].index.strftime("%Y-%m-%d").tolist() == [
        "2025-11-26",
        "2025-12-24",
        "2025-12-31",
        "2026-01-16",
    ]
    assert np.isnan(marked.iloc[-1])


def test_pre_holiday_days_follow_the_trading_week():
    # An exchange that trades from Sunday to Thursday has no holiday every Friday.
    days = pd.date_range("2025-01-05", "2025-06-26")
    sunday_to_thursday = days[days.dayofweek.isin([6, 0, 1, 2, 3])]
    assert seasonality_model.get_pre_holiday_days(sunday_to_thursday).sum() == 0

    # A cryptocurrency trades every day, so it has no holidays to read.
    assert seasonality_model.get_pre_holiday_days(days).isna().all()


def test_calendar_effect_without_marked_days(returns):
    effect = seasonality_model.get_calendar_effect(
        returns, pd.Series(0.0, index=returns.index), "Pre-Holiday"
    )

    assert effect["Average Return"].isna().all()
    assert effect["Observations"].tolist() == [0, len(returns), len(returns)]
