"""Growth Model Tests""" ""

import pandas as pd
import pytest

from financetoolkit.models import growth_model


@pytest.mark.parametrize(
    "source_dates, trading_dates",
    [
        (
            ["2024-03-31", "2024-06-30"],
            ["2024-03-28", "2024-04-01", "2024-06-28", "2024-07-01"],
        ),
        (
            ["2024-03-29", "2024-06-27"],
            ["2024-03-28", "2024-04-01", "2024-06-26", "2024-07-01"],
        ),
    ],
)
def test_daily_pvgo_retains_source_dates_outside_the_price_index(
    source_dates, trading_dates
):
    columns = pd.PeriodIndex(source_dates, freq="D")
    index = pd.PeriodIndex(trading_dates, freq="D", name="Date")
    earnings = pd.DataFrame([[2.0, 3.0], [4.0, 5.0]], ["A", "B"], columns)
    costs = pd.DataFrame([[0.1, 0.1], [0.2, 0.2]], ["A", "B"], columns)
    prices = pd.DataFrame(
        {"A": [50.0, 51.0, 52.0, 53.0], "B": [40.0, 41.0, 42.0, 43.0]},
        index=index,
    )

    result = growth_model.get_present_value_of_growth_opportunities(
        costs, earnings, prices, calculate_daily=True
    )

    expected = pd.DataFrame(
        {"A": [float("nan"), 31.0, 32.0, 23.0], "B": [float("nan"), 21.0, 22.0, 18.0]},
        index=index,
    )
    pd.testing.assert_frame_equal(result, expected)


def test_daily_pvgo_does_not_use_future_earnings():
    columns = pd.PeriodIndex(["2024-03-31", "2024-06-30"], freq="D")
    costs = pd.DataFrame([[0.1, 0.1]], ["A"], columns)
    earnings = pd.DataFrame([[2.0, 3.0]], ["A"], columns)
    prices = pd.DataFrame(
        {"A": [51.0, 52.0, 53.0]},
        index=pd.PeriodIndex(["2024-04-01", "2024-06-28", "2024-07-01"], freq="D"),
    )
    original = growth_model.get_present_value_of_growth_opportunities(
        costs, earnings, prices, calculate_daily=True
    )
    earnings.iloc[0, 1] = 100.0
    changed = growth_model.get_present_value_of_growth_opportunities(
        costs, earnings, prices, calculate_daily=True
    )

    pd.testing.assert_frame_equal(original.iloc[:2], changed.iloc[:2])
    assert original.iloc[0, 0] == pytest.approx(31.0)
    assert changed.iloc[-1, 0] == pytest.approx(-947.0)


def test_daily_pvgo_keeps_the_single_series_dataframe_layout():
    dates = pd.PeriodIndex(["2024-03-29", "2024-04-01"], freq="D")
    result = growth_model.get_present_value_of_growth_opportunities(
        pd.Series([0.1], index=dates[:1]),
        pd.Series([2.0], index=dates[:1]),
        pd.DataFrame([50.0, 51.0], index=dates),
        calculate_daily=True,
    )

    pd.testing.assert_frame_equal(result, pd.DataFrame([30.0, 31.0], index=dates))


def test_daily_pvgo_keeps_existing_forward_fill_of_missing_values():
    sources = pd.PeriodIndex(["2024-03-29", "2024-04-02"], freq="D")
    dates = pd.PeriodIndex(["2024-03-29", "2024-04-01", "2024-04-02"], freq="D")
    result = growth_model.get_present_value_of_growth_opportunities(
        pd.DataFrame([[0.1, 0.1]], ["A"], sources),
        pd.DataFrame([[2.0, float("nan")]], ["A"], sources),
        pd.DataFrame({"A": [50.0, 51.0, 52.0]}, index=dates),
        calculate_daily=True,
    )

    pd.testing.assert_frame_equal(
        result, pd.DataFrame({"A": [30.0, 31.0, 32.0]}, index=dates)
    )


def test_get_present_value_of_growth_opportunities(recorder):
    recorder.capture(
        growth_model.get_present_value_of_growth_opportunities(
            cost_of_equity=pd.Series([0.30, 0.05, 0.10]),
            earnings_per_share=pd.Series([0.10, 0.02, 0.03]),
            close_prices=pd.Series([1, 2, 3]),
        )
    )


def test_get_sustainable_growth_rate(recorder):
    recorder.capture(
        growth_model.get_sustainable_growth_rate(
            return_on_equity=pd.Series([0.15, 0.18, 0.12, 0.2]),
            retention_ratio=pd.Series([0.6, 0.55, 0.7, 0.5]),
        )
    )


def test_get_internal_growth_rate(recorder):
    recorder.capture(
        growth_model.get_internal_growth_rate(
            return_on_assets=pd.Series([0.08, 0.09, 0.07, 0.1]),
            retention_ratio=pd.Series([0.6, 0.55, 0.7, 0.5]),
        )
    )


def test_get_return_decomposition(recorder):
    periods = pd.period_range("2022", periods=3, freq="Y")
    recorder.capture(
        growth_model.get_return_decomposition(
            close_prices=pd.DataFrame(
                [[100.0, 150.0, 120.0], [10.0, 12.0, 9.0]], ["A", "B"], periods
            ),
            earnings_per_share=pd.DataFrame(
                [[5.0, 6.0, 8.0], [1.0, -0.5, 0.5]], ["A", "B"], periods
            ),
            dividends_per_share=pd.DataFrame(
                [[1.0, 2.0, 2.0], [0.0, 0.0, 0.0]], ["A", "B"], periods
            ),
        )
    )


def test_the_return_is_the_earnings_growth_times_the_multiple_change():
    periods = pd.period_range("2022", periods=2, freq="Y")
    decomposition = growth_model.get_return_decomposition(
        close_prices=pd.DataFrame([[100.0, 150.0]], ["A"], periods),
        earnings_per_share=pd.DataFrame([[5.0, 6.0]], ["A"], periods),
        dividends_per_share=pd.DataFrame([[1.0, 3.0]], ["A"], periods),
    )["2023"]

    # EPS 5 -> 6 (+20%) and P/E 20 -> 25 (+25%) give the price 100 -> 150 (+50%).
    assert decomposition["A", "EPS Growth"] == pytest.approx(0.2)
    assert decomposition["A", "P/E Change"] == pytest.approx(0.25)
    assert decomposition["A", "Price Return"] == pytest.approx(0.5)
    # The dividend paid in the window, 3, on the starting price of 100.
    assert decomposition["A", "Dividend Return"] == pytest.approx(0.03)
    assert decomposition["A", "Total Return"] == pytest.approx(0.53)

    logarithmic = growth_model.get_return_decomposition(
        close_prices=pd.DataFrame([[100.0, 150.0]], ["A"], periods),
        earnings_per_share=pd.DataFrame([[5.0, 6.0]], ["A"], periods),
        dividends_per_share=pd.DataFrame([[1.0, 3.0]], ["A"], periods),
        logarithmic=True,
    )["2023"]["A"]
    assert logarithmic["EPS Growth"] + logarithmic["P/E Change"] + logarithmic[
        "Dividend Return"
    ] == pytest.approx(logarithmic["Total Return"])


def test_a_loss_leaves_the_return_unsplit():
    periods = pd.period_range("2022", periods=2, freq="Y")
    decomposition = growth_model.get_return_decomposition(
        close_prices=pd.DataFrame([[10.0, 12.0]], ["B"], periods),
        earnings_per_share=pd.DataFrame([[-1.0, 0.5]], ["B"], periods),
        dividends_per_share=pd.DataFrame([[0.0, 0.0]], ["B"], periods),
    )["2023"]["B"]

    assert decomposition["Price Return"] == pytest.approx(0.2)
    assert pd.isna(decomposition["EPS Growth"])
    assert pd.isna(decomposition["P/E Change"])
