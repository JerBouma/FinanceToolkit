"""Offline tests of the FixedIncome yield curve analytics, on a simulated curve history."""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.cache.cache_controller import clear_active_cache
from financetoolkit.economics.sources import ecb_model as economics_ecb_model
from financetoolkit.fixedincome.curves import curve_model, yieldcurve_model
from financetoolkit.fixedincome.fixedincome_controller import FixedIncome
from financetoolkit.fixedincome.sources import fiscaldata_model, fred_model
from financetoolkit.utilities.error_model import InvalidArgumentError

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
    # A monthly curve request adds the ECB's convergence yields of the other EU members,
    # which the simulated history leaves out.
    monkeypatch.setattr(
        economics_ecb_model,
        "get_long_term_convergence_yields",
        lambda start_date, end_date: pd.DataFrame(),
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
    parameters = fixedincome.get_nelson_siegel_svensson(method="nelson_siegel")
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

    # The Svensson fits run on the month-end curves, which keeps the test quick.
    fitted = fixedincome.get_nelson_siegel_svensson(
        maturities=[1, "10Y"], period="monthly"
    )
    # The MCP server passes a list as a comma-separated string.
    from_string = fixedincome.get_nelson_siegel_svensson(
        maturities="1, 10Y", period="monthly"
    )
    pd.testing.assert_frame_equal(fitted, from_string)
    zero_curve = fixedincome.get_zero_coupon_curve(period="monthly")["United States"]
    assert list(fitted["United States"].columns) == ["1Y", "10Y"]
    assert fitted["United States"]["10Y"].iloc[-1] == pytest.approx(
        zero_curve["10Y"].iloc[-1], abs=0.002
    )


def test_carry_and_roll_down(fixedincome):
    result = fixedincome.get_carry_and_roll_down(
        ["United States", "Germany", "Australia"], period="monthly"
    )

    # Australia's curve starts at two years, too long to bootstrap zero-coupon rates from.
    assert list(result.columns.get_level_values(0).unique()) == [
        "United States",
        "Germany",
    ]
    assert result.columns.names == ["Country", "Component", "Maturity"]
    us = result["United States"]
    assert list(us.columns.get_level_values(0).unique()) == [
        "Carry",
        "Roll-Down",
        "Total",
    ]
    # Every maturity longer than the three-month horizon.
    assert list(us["Total"].columns) == LABELS[2:]
    assert us["Total"].to_numpy() == pytest.approx(
        (us["Carry"] + us["Roll-Down"]).to_numpy()
    )

    # Each date is the model on that day's zero-coupon curve, with the coupon dates of the
    # bootstrap, financed at the three-month zero rate.
    curve = fixedincome.get_government_bond_yield_curve(
        "United States", period="monthly", rounding=10
    )["United States"]
    zero_curve = fixedincome._zero_curve(curve.iloc[-1], "United States")
    expected = yieldcurve_model.get_carry_and_roll_down(
        zero_curve, [0.5, 2.0, 10.0], 0.25
    )
    last = us.iloc[-1]
    for years, label in zip([0.5, 2.0, 10.0], ["6M", "2Y", "10Y"], strict=True):
        for component in ["Carry", "Roll-Down", "Total"]:
            assert last[(component, label)] == pytest.approx(
                expected.loc[years, component]
            )

    # The simulated par curve rises with maturity, so every bond earns carry and rolls down
    # it up to 20 years. Beyond 20 years the par curve flattens, and the zero-coupon rates
    # bootstrapped from it fall slightly, so the 30-year bond rolls up the curve.
    assert (us["Carry"].iloc[-1] > 0).all()
    assert (us["Roll-Down"].iloc[-1].drop("30Y") > 0).all()
    assert us["Roll-Down"]["30Y"].iloc[-1] < 0


def test_carry_and_roll_down_horizon_and_maturities(fixedincome):
    yearly = fixedincome.get_carry_and_roll_down(
        horizon="1Y", maturities=[0.5, "1Y", 2, "10Y"], period="monthly"
    )["United States"]
    from_string = fixedincome.get_carry_and_roll_down(
        horizon=1, maturities="6M, 1Y, 2Y, 10Y", period="monthly"
    )["United States"]

    pd.testing.assert_frame_equal(yearly, from_string)
    assert list(yearly["Carry"].columns) == ["6M", "1Y", "2Y", "10Y"]
    # A six-month bill is repaid before the year is out; the 1-year bill is the financing.
    assert yearly["Total"]["6M"].isna().all()
    assert yearly["Total"]["1Y"].to_numpy() == pytest.approx(0, abs=1e-12)
    with pytest.raises(InvalidArgumentError, match="horizon must be longer than zero"):
        fixedincome.get_carry_and_roll_down(horizon=0)


def test_carry_and_roll_down_horizon_beyond_the_curve(fixedincome, caplog):
    # No maturity of the curve outlasts a 40-year horizon, so no bond is held over it.
    with caplog.at_level("WARNING"):
        result = fixedincome.get_carry_and_roll_down(horizon="40Y", period="monthly")

    assert result.empty
    assert "no maturity longer than the horizon" in caplog.text


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
    assert np.isnan(result.loc["Feller Ratio", "Vasicek"])
    assert result.loc["Feller Ratio", "CIR"] == pytest.approx(
        2
        * result.loc["Speed of Mean Reversion", "CIR"]
        * result.loc["Long-Run Mean", "CIR"]
        / result.loc["Volatility", "CIR"] ** 2
    )

    monkeypatch.setenv("FINANCETOOLKIT_STRICT_ERRORS", "1")
    with pytest.raises(ValueError, match="no 4M maturity"):
        fixedincome.get_short_rate_model(short_rate="4M")


@pytest.mark.parametrize("scheme", ["euler", "exact"])
@pytest.mark.parametrize("model", ["hull_white", "vasicek", "cir"])
def test_short_rate_simulation(fixedincome, model, scheme):
    simulation = fixedincome.get_short_rate_simulation(
        model=model, years=2, steps_per_year=4, simulations=2000, scheme=scheme
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
        fixedincome.get_short_rate_simulation(simulations=10_000_000, years=10)
    with pytest.raises(ValueError, match="scheme must be one of"):
        fixedincome.get_short_rate_simulation(scheme="milstein")


def test_get_swaption_price(recorder, fixedincome):
    recorder.capture(fixedincome.get_swaption_price())
    recorder.capture(
        fixedincome.get_swaption_price(
            expiries="6M,1Y", tenors=[2, 10], model="cir++", swaption_type="receiver"
        )
    )
    recorder.capture(
        fixedincome.get_swaption_price(
            expiries=1, tenors=5, strike=0.03, fixed_frequency=2
        )
    )


def test_get_swaption_price_bermudan(recorder, fixedincome):
    recorder.capture(
        fixedincome.get_swaption_price(
            expiries=[1, 5], tenors=[2, 10], exercise="bermudan"
        )
    )
    recorder.capture(
        fixedincome.get_swaption_price(
            expiries=1,
            tenors=5,
            strike=0.03,
            swaption_type="receiver",
            fixed_frequency=2,
            exercise="bermudan",
        )
    )


def test_bermudan_swaption_is_worth_at_least_the_european(fixedincome, monkeypatch):
    european = fixedincome.get_swaption_price(expiries=[1, 5], tenors=[2, 10])
    bermudan = fixedincome.get_swaption_price(
        expiries=[1, 5], tenors=[2, 10], exercise="bermudan"
    )
    # The tree prices the European exercise within a basis point of notional of the closed
    # form, and the extra exercise dates only add value.
    assert (bermudan.to_numpy() >= european.to_numpy() - 1e-4).all()
    assert bermudan.loc[5, 10] > european.loc[5, 10]

    monkeypatch.setenv("FINANCETOOLKIT_STRICT_ERRORS", "1")
    with pytest.raises(ValueError, match="Hull-White tree"):
        fixedincome.get_swaption_price(model="cir++", exercise="bermudan")


def test_swaption_price_is_at_the_money(fixedincome, monkeypatch):
    # At the money a payer and a receiver swaption are worth the same.
    for model in ("hull_white", "cir++"):
        payer = fixedincome.get_swaption_price(
            expiries=[1, 5], tenors=[2, 10], model=model
        )
        receiver = fixedincome.get_swaption_price(
            expiries=[1, 5], tenors=[2, 10], model=model, swaption_type="receiver"
        )
        assert payer.to_numpy() == pytest.approx(receiver.to_numpy(), abs=1e-12)
        assert (payer.to_numpy() > 0).all()

    monkeypatch.setenv("FINANCETOOLKIT_STRICT_ERRORS", "1")
    with pytest.raises(ValueError, match="model must be one of"):
        fixedincome.get_swaption_price(model="vasicek")


def test_implied_default_probability(fixedincome, monkeypatch):
    spreads = pd.DataFrame(
        [[0.008, 0.012, 0.03], [0.007, 0.012, 0.03]],
        index=pd.period_range("2024-12-30", periods=2, freq="D"),
        columns=["A", "BBB", "B"],
    )
    monkeypatch.setattr(
        fred_model,
        "get_rating_option_adjusted_spread",
        lambda start_date, end_date, api_key: spreads,
    )
    fixedincome._fred_api_key = "test"

    result = fixedincome.get_implied_default_probability(horizons=[1, 5])

    assert result.index.tolist() == ["A", "BBB", "B"]
    assert list(result.columns) == [
        "Spread",
        "Hazard Rate",
        "1Y Default Probability",
        "5Y Default Probability",
    ]
    # The latest spread, divided by one minus the recovery rate.
    assert result.loc["A", "Hazard Rate"] == pytest.approx(0.007 / 0.6)
    assert result.loc["BBB", "Hazard Rate"] == pytest.approx(0.02)

    selected = fixedincome.get_implied_default_probability(
        ratings="bbb, B", horizons="1,5"
    )
    pd.testing.assert_frame_equal(selected, result.loc[["BBB", "B"]])

    # An unknown rating is logged and gives an empty result.
    assert fixedincome.get_implied_default_probability(ratings="ZZ").empty


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


def test_curve_arguments_that_cannot_give_an_answer_are_refused(fixedincome):
    # A far maturity nowhere above the near one has no forward rate at all.
    with pytest.raises(InvalidArgumentError, match="far_maturity"):
        fixedincome.get_forward_rate(near_maturity=10, far_maturity=2)
    # No maturity outlasts the horizon, so no bond is held over it.
    with pytest.raises(InvalidArgumentError, match="longer than it"):
        fixedincome.get_carry_and_roll_down(horizon="5Y", maturities="1Y,2Y")
    with pytest.raises(InvalidArgumentError, match="components"):
        fixedincome.get_yield_curve_principal_components(components=0)
    with pytest.raises(InvalidArgumentError, match="at most 11 components"):
        fixedincome.get_yield_curve_principal_components(components=12)
    with pytest.raises(InvalidArgumentError, match="period"):
        fixedincome.get_carry_and_roll_down(period="yearly")


def test_principal_components_need_two_maturities():
    curve = pd.DataFrame({"10Y": [0.04, 0.041, 0.042]})

    with pytest.raises(ValueError, match="at least two maturities"):
        curve_model.get_principal_components(curve, components=1)
    with pytest.raises(ValueError, match="components"):
        curve_model.get_principal_components(
            simulated_curves().iloc[:50], components=12
        )


def test_nelson_siegel_svensson_reports_a_curve_without_short_maturities(
    fixedincome, caplog
):
    # The Australian curve starts at two years, so there is no zero-coupon curve to fit.
    with caplog.at_level("WARNING"):
        result = fixedincome.get_nelson_siegel_svensson(countries="Australia")

    assert result.empty
    assert "no maturity of a year or less" in caplog.text


def test_short_rate_model_reports_a_flat_curve(monkeypatch, caplog):
    clear_active_cache()
    flat = simulated_curves() * 0 + 0.03
    monkeypatch.setattr(
        FixedIncome,
        "_get_country_yield_curves",
        lambda self, start_date: {"United States": lambda: flat},
    )
    monkeypatch.setattr(
        economics_ecb_model,
        "get_long_term_convergence_yields",
        lambda start_date, end_date: pd.DataFrame(),
    )
    fixedincome = FixedIncome(
        start_date="2015-01-01", end_date="2024-12-31", rounding=None
    )

    with caplog.at_level("WARNING"):
        result = fixedincome.get_short_rate_model()

    assert result["Market"].notna().any()
    assert result.drop(columns="Market").isna().all().all()
    assert "does not move" in caplog.text
    clear_active_cache()
