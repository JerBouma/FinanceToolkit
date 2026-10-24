"""Offline tests of the FixedIncome yield curve analytics, on a simulated curve history."""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.cache.cache_controller import clear_active_cache
from financetoolkit.fixedincome import curve_model, fiscaldata_model
from financetoolkit.fixedincome.fixedincome_controller import FixedIncome

# pylint: disable=missing-function-docstring,protected-access

LABELS = ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]


def simulated_curves(start="2015-01-01", end="2024-12-31", floor=0.001):
    """A Treasury-like par curve history: a mean-reverting short rate plus a slope."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range(start, end).to_period("D")
    short = np.empty(len(dates))
    short[0] = 0.02
    for day in range(1, len(dates)):
        short[day] = short[day - 1] + 0.3 * (0.03 - short[day - 1]) / 252
        short[day] += 0.008 * np.sqrt(1 / 252) * rng.standard_normal()
    short = np.maximum(short, floor)
    slope = 0.012 + 0.004 * np.sin(np.arange(len(dates)) / 400)
    maturities = np.array([curve_model.maturity_in_years(label) for label in LABELS])
    curves = short[:, None] + slope[:, None] * (1 - np.exp(-maturities / 3))[None, :]
    return pd.DataFrame(curves, index=dates, columns=LABELS)


@pytest.fixture
def fixedincome(monkeypatch):
    clear_active_cache()
    curves = simulated_curves()
    monkeypatch.setattr(
        FixedIncome,
        "_get_country_yield_curves",
        lambda self, start_date: {
            "United States": lambda: curves,
            "Germany": lambda: curves[
                ["1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
            ],
            "Australia": lambda: curves[["2Y", "3Y", "5Y", "10Y"]],
        },
    )
    yield FixedIncome(start_date="2015-01-01", end_date="2024-12-31", rounding=None)
    clear_active_cache()


def test_zero_coupon_curve(fixedincome):
    zero_curve = fixedincome.get_zero_coupon_curve(["United States", "Germany"])
    curves = simulated_curves()

    assert list(zero_curve.columns.get_level_values(0).unique()) == [
        "United States",
        "Germany",
    ]
    # Germany's curve is zero-coupon already, published continuously compounded; the US
    # bills (bond-equivalent yields, simple interest up to half a year) only change
    # compounding and the zero rates of an upward sloping par curve lie above it.
    assert zero_curve["Germany"].iloc[-1].to_numpy() == pytest.approx(
        np.expm1(curves.iloc[-1][zero_curve["Germany"].columns].to_numpy())
    )
    last = zero_curve["United States"].iloc[-1]
    assert last["3M"] == pytest.approx((1 + curves.iloc[-1]["3M"] / 4) ** 4 - 1)
    assert last["30Y"] > curves.iloc[-1]["30Y"]


def test_zero_coupon_curve_needs_a_short_maturity(fixedincome):
    assert fixedincome.get_zero_coupon_curve("Australia").empty


def test_nelson_siegel_svensson(fixedincome):
    parameters = fixedincome.get_nelson_siegel_svensson(svensson=False)
    us = parameters["United States"]

    assert list(us.columns) == [
        "Beta 0",
        "Beta 1",
        "Beta 2",
        "Tau 1",
        "Root Mean Squared Error",
    ]
    assert len(us) == len(simulated_curves())
    assert (us["Root Mean Squared Error"] < 0.0015).all()

    fitted = fixedincome.get_nelson_siegel_svensson(maturities=[1, "10Y"])
    # The MCP server passes a list as a comma-separated string.
    from_string = fixedincome.get_nelson_siegel_svensson(maturities="1, 10Y")
    pd.testing.assert_frame_equal(fitted, from_string)
    zero_curve = fixedincome.get_zero_coupon_curve()["United States"]
    assert list(fitted["United States"].columns) == ["1Y", "10Y"]
    assert fitted["United States"]["10Y"].iloc[-1] == pytest.approx(
        zero_curve["10Y"].iloc[-1], abs=0.002
    )


def test_principal_components(fixedincome):
    loadings = fixedincome.get_yield_curve_principal_components(components=2)
    scores = fixedincome.get_yield_curve_principal_components(components=2, scores=True)

    assert list(loadings.index) == [*LABELS, "Explained Variance"]
    assert (loadings.loc[LABELS, "PC1"] > 0).all()
    # PC2 is oriented as a steepening: the long end rises against the short end.
    assert loadings.loc["30Y", "PC2"] > loadings.loc["1M", "PC2"]
    assert loadings.loc["Explained Variance", "PC1"] > 0.9
    assert list(scores.columns) == ["PC1", "PC2"]


def test_short_rate_model(fixedincome, monkeypatch):
    result = fixedincome.get_short_rate_model()

    assert list(result.columns) == ["Vasicek", "CIR", "Market"]
    assert result.loc["Long-Run Mean", "Vasicek"] == pytest.approx(0.03, abs=0.02)
    # The models describe a continuously compounded rate; the 3M bill yield is simple
    # interest over the quarter.
    assert result.loc["Current Rate", "Market"] == pytest.approx(
        4 * np.log1p(simulated_curves()["3M"].iloc[-1] / 4)
    )
    assert result.loc["Zero Rate 30Y", "Market"] > result.loc["Zero Rate 1M", "Market"]

    monkeypatch.setenv("FINANCETOOLKIT_STRICT_ERRORS", "1")
    with pytest.raises(ValueError, match="no 4M maturity"):
        fixedincome.get_short_rate_model(short_rate="4M")


@pytest.mark.parametrize("model", ["hull_white", "vasicek", "cir"])
def test_short_rate_simulation(fixedincome, model):
    simulation = fixedincome.get_short_rate_simulation(
        model=model, years=2, steps_per_year=4, paths=2000
    )

    assert list(simulation.columns) == ["Mean", "5%", "25%", "50%", "75%", "95%"]
    assert list(simulation.index) == [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
    assert (simulation["5%"] <= simulation["95%"]).all()
    assert simulation.iloc[0].to_numpy() == pytest.approx(
        np.full(6, 4 * np.log1p(simulated_curves()["3M"].iloc[-1] / 4))
    )


def test_short_rate_simulation_is_bounded(fixedincome, monkeypatch):
    monkeypatch.setenv("FINANCETOOLKIT_STRICT_ERRORS", "1")
    with pytest.raises(ValueError, match="too large"):
        fixedincome.get_short_rate_simulation(paths=10_000_000, years=10)


def test_option_adjusted_spread(fixedincome):
    spot_rates = {1: 0.036, 2: 0.035, 5: 0.037, 10: 0.041, 20: 0.047, 30: 0.047}
    result = fixedincome.get_option_adjusted_spread(
        bond_price=[98, 102], spot_rates=spot_rates
    )

    assert list(result.columns) == [98, 102]
    assert (
        result.loc["Option-Adjusted Spread", 98]
        > result.loc["Option-Adjusted Spread", 102]
    )
    assert (result.loc["Option Value"] > 0).all()

    # Without a curve, the US zero-coupon curve of the end date is used.
    latest = fixedincome.get_option_adjusted_spread(call_price=None)
    assert latest.loc["Option Value", 100.0] == pytest.approx(0, abs=1e-8)


def test_implied_default_probability(fixedincome):
    result = fixedincome.get_implied_default_probability(
        {"BBB": 0.012, "B": 0.03}, horizons=[1, 5]
    )

    assert list(result.columns) == [
        "Spread",
        "Hazard Rate",
        "1Y Default Probability",
        "5Y Default Probability",
    ]
    assert result.loc["BBB", "Hazard Rate"] == pytest.approx(0.02)
    assert fixedincome.get_implied_default_probability(0.006).loc[
        "Spread", "Hazard Rate"
    ] == (pytest.approx(0.01))
    from_string = fixedincome.get_implied_default_probability(
        "BBB:0.012, B:0.03", horizons="1,5"
    )
    pd.testing.assert_frame_equal(result, from_string)


def test_treasury_auctions(fixedincome, monkeypatch):
    auctions = pd.DataFrame(
        {
            "Auction Date": ["2024-01-08", "2024-01-10", "2024-01-11"],
            "Security Type": ["Bill", "Note", "TIPS"],
            "Security Term": ["13-Week", "10-Year", "10-Year"],
            "CUSIP": ["A", "B", "C"],
            "Issue Date": ["2024-01-11", "2024-01-16", "2024-01-31"],
            "Maturity Date": ["2024-04-11", "2034-01-15", "2034-01-15"],
            "Reopening": ["No", "No", "No"],
            "Coupon Rate": [np.nan, 0.04, 0.0175],
            "High Yield": [np.nan, 0.0405, 0.0181],
            "High Investment Rate": [0.0531, np.nan, np.nan],
            "High Discount Rate": [0.052, np.nan, np.nan],
            "Bid-to-Cover Ratio": [2.8, 2.5, 2.4],
            "Offering Amount": [7e10, 3.7e10, 1.7e10],
            "Total Tendered": [2e11, 9e10, 4e10],
            "Total Accepted": [7e10, 3.7e10, 1.7e10],
            "Primary Dealers Accepted": [1e10, 5e9, 2e9],
            "Direct Bidders Accepted": [1e10, 6e9, 3e9],
            "Indirect Bidders Accepted": [4e10, 2.5e10, 1.1e10],
        }
    )
    monkeypatch.setattr(
        fiscaldata_model, "get_treasury_auctions", lambda start, end: auctions
    )

    result = fixedincome.get_treasury_auctions()
    assert result.index.name == "Auction Date"
    assert list(result["Yield"]) == pytest.approx([0.0531, 0.0405, 0.0181])
    assert result["Indirect Bidder Share"].iloc[0] == pytest.approx(4 / 6)

    notes = fixedincome.get_treasury_auctions(
        security_type="note", security_term="10-year"
    )
    assert list(notes["CUSIP"]) == ["B"]
    coupon_bonds = fixedincome.get_treasury_auctions(security_type="Note, TIPS")
    assert list(coupon_bonds["CUSIP"]) == ["B", "C"]

    monkeypatch.setenv("FINANCETOOLKIT_STRICT_ERRORS", "1")
    with pytest.raises(ValueError, match="does not exist"):
        fixedincome.get_treasury_auctions(security_type="Strip")
