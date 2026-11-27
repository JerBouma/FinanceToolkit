"""Tests of the yield curve construction: interpolation and bootstrapping, Smith-Wilson extrapolation
and bond curves fitted to prices, against QuantLib, rateslib and EIOPA's published curve.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from financetoolkit.cache.cache_controller import clear_active_cache
from financetoolkit.fixedincome.curves import (
    curve_model,
    interpolation_model,
    smith_wilson_model,
)
from financetoolkit.fixedincome.fixedincome_controller import FixedIncome
from financetoolkit.fixedincome.markets import bond_data_model, fitted_curve_model
from financetoolkit.fixedincome.sources import treasurydirect_model
from financetoolkit.utilities.error_model import InvalidArgumentError

# A humped curve of continuously compounded zero rates on ACT/365 days, and the times the
# QuantLib values below are read at.
DAYS = np.array([91, 182, 365, 730, 1095, 1825, 2555, 3650, 7300, 10950])
ZERO_RATES = np.array(
    [0.043, 0.042, 0.04, 0.038, 0.0375, 0.039, 0.041, 0.044, 0.049, 0.047]
)
RISING_RATES = np.array(
    [0.03, 0.031, 0.033, 0.036, 0.038, 0.041, 0.043, 0.045, 0.048, 0.049]
)
YEARS = [0.1, 0.75, 1.5, 4.0, 8.5, 15.0, 25.0]

# QuantLib 1.44: ZeroCurve (Linear), NaturalCubicZeroCurve, DiscountCurve (log-linear),
# MonotonicCubicZeroCurve (on the rising rates) and PiecewiseConvexMonotoneForward with
# ConvexMonotone(0, 1, False) on deposits that reprice the zero rates, all on
# Actual365Fixed, with a node at the reference date at the first rate.
QUANTLIB = {
    "linear_zero": (
        [0.995709231763, 0.969719959397, 0.9431782404, 0.858129721811, 0.696804775496, 0.497828320174, 0.301194211912],
        [0.042999999999, 0.038005464481, 0.036, 0.041250000001, 0.051, 0.053999999999, 0.043000000001],
    ),
    "natural_cubic": (
        [0.995700574472, 0.969791335576, 0.943554401825, 0.858974593552, 0.696659840709, 0.489885134476, 0.297322087395],
        [0.043140553093, 0.037911854737, 0.03584710154, 0.041394075075, 0.051166850536, 0.055035140676, 0.042654953107],
    ),
    "log_linear": (
        [0.995709231763, 0.969963082236, 0.943649947437, 0.857486365809, 0.695238727216, 0.491644197461, 0.302703954182],
        [0.042999999999, 0.038010928963, 0.036, 0.041250000001, 0.051, 0.053999999999, 0.043000000001],
    ),
    "monotone_convex": (
        [0.995667456679, 0.969812647194, 0.943664445205, 0.858675165012, 0.695959098106, 0.492078031454, 0.299598360395],
        [0.043258679487, 0.037640684858, 0.035865511714, 0.041447916667, 0.051646723646, 0.055294345782, 0.042312499999],
    ),
    "monotone_cubic": (
        [0.997004495503, 0.976227787578, 0.949376465146, 0.853429008956, 0.687363047018, 0.49415395382, 0.296818524984],
        [0.030000000001, 0.035021916396, 0.039152936049, 0.045655754944, 0.049712206323, 0.051170748314, 0.050943083896],
    ),
}  # fmt: skip

# Bills and semi-annual par bonds, and QuantLib's piecewise curves bootstrapped from them
# (DepositRateHelper and FixedRateBondHelper on Thirty360, so the coupon dates fall on whole
# half years) at 0.75, 1.5, 4, 8.5, 15 and 25 years.
PAR_MATURITIES = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
PAR_YIELDS = np.array(
    [0.043, 0.042, 0.040, 0.038, 0.0375, 0.039, 0.041, 0.044, 0.049, 0.047]
)
BOOTSTRAP_YEARS = [0.75, 1.5, 4.0, 8.5, 15.0, 25.0]
QUANTLIB_BOOTSTRAP = {
    "linear_zero": [0.970019722847, 0.943746530721, 0.859353055502, 0.696425878665, 0.490522376923, 0.297293420369],
    "log_linear": [0.970257385421, 0.944217273498, 0.858676670169, 0.694784172492, 0.484166010262, 0.301693926883],
    "natural_cubic": [0.970087397537, 0.944112221615, 0.860202844766, 0.69635556485, 0.481935118623, 0.293663159385],
    "monotone_convex": [0.970106162883, 0.944232346828, 0.859913991149, 0.695663030263, 0.484679866678, 0.29695423017],
}  # fmt: skip

# EIOPA's euro risk-free curve of 31 October 2026 (no volatility adjustment): its par swap
# rates after the credit risk adjustment at the liquid maturities (annual coupons, read off
# EIOPA's Smith-Wilson coefficients), its UFR, alpha and last liquid point, and the rates
# it publishes.
EIOPA_SWAPS = {
    1: 0.03138, 2: 0.03271, 3: 0.03291, 4: 0.03291, 5: 0.03287, 6: 0.032897, 7: 0.032979,
    8: 0.033099, 9: 0.03325, 10: 0.033409, 11: 0.033554, 12: 0.033693, 13: 0.033791,
    14: 0.033851, 15: 0.033885, 20: 0.033562,
}  # fmt: skip
EIOPA_UFR = 0.033
EIOPA_ALPHA = 0.068369
EIOPA_RATES = {
    1: 0.03138, 5: 0.03288, 10: 0.03348, 15: 0.03402, 20: 0.03355, 25: 0.03317,
    30: 0.03298, 40: 0.03284, 60: 0.03282, 100: 0.03288, 150: 0.03292,
}  # fmt: skip


def eiopa_instruments():
    times = np.arange(1, 21, dtype=float)
    cash_flows = np.zeros((len(EIOPA_SWAPS), len(times)))
    for row, (maturity, rate) in enumerate(EIOPA_SWAPS.items()):
        cash_flows[row, :maturity] = rate
        cash_flows[row, maturity - 1] += 1
    return times, cash_flows, np.ones(len(EIOPA_SWAPS))


@pytest.mark.parametrize("interpolation", list(QUANTLIB))
def test_interpolation_against_quantlib(interpolation):
    rates = RISING_RATES if interpolation == "monotone_cubic" else ZERO_RATES
    discount, forwards = QUANTLIB[interpolation]

    assert interpolation_model.get_discount_factors(
        DAYS / 365, rates, YEARS, interpolation
    ) == pytest.approx(discount, abs=1e-11)
    assert interpolation_model.get_instantaneous_forward_rates(
        DAYS / 365, rates, YEARS, interpolation
    ) == pytest.approx(forwards, abs=1e-9)


@pytest.mark.parametrize("interpolation", interpolation_model.INTERPOLATIONS)
def test_interpolations_keep_the_curve_and_its_ends(interpolation):
    maturities = DAYS / 365
    discount = interpolation_model.get_discount_factors(
        maturities, ZERO_RATES, maturities, interpolation
    )

    assert -np.log(discount) / maturities == pytest.approx(
        np.log1p(np.expm1(ZERO_RATES)) if interpolation != "linear_par" else ZERO_RATES
    )
    assert interpolation_model.get_discount_factors(
        maturities, ZERO_RATES, 0.0, interpolation
    ) == pytest.approx(1.0)
    # Beyond the last maturity the zero rate is held at the last one.
    assert interpolation_model.get_discount_factors(
        maturities, ZERO_RATES, 50.0, interpolation
    ) == pytest.approx(np.exp(-ZERO_RATES[-1] * 50))


def test_monotone_convex_keeps_the_discrete_forwards_and_is_continuous():
    maturities = DAYS / 365
    times = np.concatenate([[0.0], maturities])
    integrals = np.concatenate([[0.0], ZERO_RATES * maturities])

    for start, end, low, high in zip(
        times[:-1], times[1:], integrals[:-1], integrals[1:], strict=True
    ):
        grid = np.linspace(start, end, 2001)
        forwards = interpolation_model.get_instantaneous_forward_rates(
            maturities, ZERO_RATES, grid, "monotone_convex"
        )
        # The forward integrates to the discrete forward of the interval.
        assert np.trapezoid(forwards, grid) == pytest.approx(high - low, abs=1e-7)

    # Continuous across the maturities, unlike the forward of linear zero rates.
    for maturity in maturities[:-1]:
        around = interpolation_model.get_instantaneous_forward_rates(
            maturities, ZERO_RATES, [maturity - 1e-9, maturity], "monotone_convex"
        )
        assert around[0] == pytest.approx(around[1], abs=1e-7)


def test_interpolation_that_does_not_exist():
    with pytest.raises(InvalidArgumentError, match="interpolation must be one of"):
        interpolation_model.get_discount_factors([1, 2], [0.03, 0.04], 1.5, "spline")


@pytest.mark.parametrize("interpolation", list(QUANTLIB_BOOTSTRAP))
def test_bootstrap_against_quantlib(interpolation):
    maturities, zero_rates = interpolation_model.bootstrap_zero_curve(
        PAR_MATURITIES, PAR_YIELDS, 2, None, interpolation
    )

    assert interpolation_model.get_discount_factors(
        maturities, zero_rates, BOOTSTRAP_YEARS, interpolation
    ) == pytest.approx(QUANTLIB_BOOTSTRAP[interpolation], abs=1e-11)


@pytest.mark.parametrize("interpolation", interpolation_model.INTERPOLATIONS)
def test_bootstrap_reprices_every_par_bond(interpolation):
    maturities, zero_rates = interpolation_model.bootstrap_zero_curve(
        PAR_MATURITIES, PAR_YIELDS, 2, "bond-equivalent", interpolation
    )

    for maturity, par_yield in zip(PAR_MATURITIES, PAR_YIELDS, strict=True):
        if maturity <= 1:
            continue
        times = np.arange(0.5, maturity + 1e-9, 0.5)
        discount = interpolation_model.get_discount_factors(
            maturities, zero_rates, times, interpolation
        )
        assert par_yield / 2 * discount.sum() + discount[-1] == pytest.approx(
            1.0, abs=1e-12
        )


def test_bootstrap_linear_par_is_the_zero_coupon_curve():
    maturities, zero_rates = interpolation_model.bootstrap_zero_curve(
        PAR_MATURITIES, PAR_YIELDS, 2, "bond-equivalent"
    )
    grid, annual = curve_model.bootstrap_zero_rates(
        PAR_MATURITIES, PAR_YIELDS, 2, "bond-equivalent", return_grid=True
    )

    assert maturities == pytest.approx(grid)
    assert np.expm1(zero_rates) == pytest.approx(annual)


def test_smith_wilson_reproduces_eiopa():
    times, cash_flows, prices = eiopa_instruments()

    alpha = smith_wilson_model.get_smith_wilson_alpha(
        times, cash_flows, prices, EIOPA_UFR, 60
    )
    assert alpha == pytest.approx(EIOPA_ALPHA, abs=1e-6)

    curve = smith_wilson_model.get_smith_wilson_curve(
        times, cash_flows, prices, list(EIOPA_RATES), EIOPA_UFR, EIOPA_ALPHA
    )
    # EIOPA publishes the rates to five decimals.
    assert curve["Zero Rate"].to_numpy() == pytest.approx(
        list(EIOPA_RATES.values()), abs=5.1e-6
    )
    # Every swap reprices at par and the forward converges to ln(1 + UFR).
    discount = smith_wilson_model.get_smith_wilson_curve(
        times, cash_flows, prices, times, EIOPA_UFR, EIOPA_ALPHA
    )["Discount Factor"].to_numpy()
    assert cash_flows @ discount == pytest.approx(prices, abs=1e-12)
    assert curve["Instantaneous Forward Rate"].iloc[-1] == pytest.approx(
        np.log1p(EIOPA_UFR), abs=1e-5
    )


def test_smith_wilson_forward_is_the_slope_of_the_discount_factors():
    times, cash_flows, prices = eiopa_instruments()
    years = np.array([0.5, 7.3, 20.0, 45.0])
    step = 1e-5

    def log_discount(shift):
        return np.log(
            smith_wilson_model.get_smith_wilson_curve(
                times, cash_flows, prices, years + shift, EIOPA_UFR, 0.1
            )["Discount Factor"].to_numpy()
        )

    curve = smith_wilson_model.get_smith_wilson_curve(
        times, cash_flows, prices, years, EIOPA_UFR, 0.1
    )
    assert curve["Instantaneous Forward Rate"].to_numpy() == pytest.approx(
        (log_discount(-step) - log_discount(step)) / (2 * step), abs=1e-8
    )


def test_smith_wilson_against_rateslib():
    # rateslib 3.0's SmithWilsonCurve with the coefficients of zero-coupon bonds at these
    # times (Act/365.25 days), alpha 0.12 and a UFR of 3.3%.
    times = np.array([91, 183, 365, 730, 1096, 1826, 2557, 3653, 7305, 10958]) / 365.25
    rates = np.array(
        [0.04, 0.041, 0.042, 0.04, 0.039, 0.04, 0.041, 0.043, 0.046, 0.045]
    )
    years = np.array([30, 300, 1500, 5000, 9000, 15000, 22000, 36500, 54000]) / 365.25
    curve = smith_wilson_model.get_smith_wilson_curve(
        times, np.eye(len(times)), (1 + rates) ** -times, years, 0.033, 0.12
    )

    # rateslib agrees to 1e-15; the values are rounded to 12 decimals.
    assert curve["Discount Factor"].to_numpy() == pytest.approx(
        [0.996814179098, 0.966850439919, 0.853171505351, 0.549230402935, 0.331672640163,
         0.179334800998, 0.095026835953, 0.026149185602, 0.005519156939],
        abs=1e-11,
    )  # fmt: skip


def test_smith_wilson_invalid():
    times, cash_flows, prices = eiopa_instruments()
    with pytest.raises(ValueError, match="convergence speed"):
        smith_wilson_model.get_smith_wilson_curve(
            times, cash_flows, prices, 10, EIOPA_UFR, 0.0
        )
    with pytest.raises(ValueError, match="row per instrument"):
        smith_wilson_model.get_smith_wilson_curve(
            times, cash_flows[:, :5], prices, 10, EIOPA_UFR, 0.1
        )


# The Svensson curve the synthetic Treasuries are priced on.
TRUE_CURVE = pd.Series(
    {
        "Beta 0": 0.05,
        "Beta 1": -0.01,
        "Beta 2": 0.01,
        "Beta 3": -0.01,
        "Tau 1": 1.5,
        "Tau 2": 8.0,
    }
)
PRICE_DATE = date(2026, 11, 19)


def synthetic_treasuries(mispriced: float = 0.0) -> pd.DataFrame:
    """Notes maturing every month up to 10 years and bonds every quarter up to 30 years, as
    the Treasury issues them, priced on TRUE_CURVE."""
    rows = []
    maturities = pd.date_range("2027-02-01", "2036-11-01", freq="MS").append(
        pd.date_range("2037-02-01", "2056-11-01", freq="3MS")
    ) + pd.Timedelta(days=14)
    for number, maturity in enumerate(maturities):
        coupon = 0.02 + 0.0025 * (number % 12)
        _, coupon_dates = bond_data_model.get_coupon_dates(PRICE_DATE, maturity.date())
        times = np.array([(day - PRICE_DATE).days / 365 for day in coupon_dates])
        flows = np.full(len(times), 100 * coupon / 2)
        flows[-1] += 100
        discount = fitted_curve_model.get_fitted_discount_factors(
            TRUE_CURVE, times, "svensson"
        )
        price = flows @ discount - bond_data_model.get_treasury_accrued_interest(
            coupon, PRICE_DATE, maturity.date()
        )
        if number == 20:
            price += mispriced
        rows.append(
            (
                f"91282C{number:03d}",
                "Note" if maturity.year < 2037 else "Bond",
                coupon,
                maturity,
                price,
            )
        )
    prices = pd.DataFrame(
        rows, columns=["CUSIP", "Security Type", "Coupon", "Maturity", "Price"]
    )
    # A bill and a TIPS, which the fit leaves out.
    extra = pd.DataFrame(
        [
            ("912797AA0", "Bill", 0.0, pd.Timestamp("2027-02-18"), 98.9),
            ("91282CTP0", "TIPS", 0.01, pd.Timestamp("2030-01-15"), 99.0),
        ],
        columns=prices.columns,
    )
    return pd.concat([prices, extra], ignore_index=True)


def bond_inputs(prices: pd.DataFrame):
    """The cash flow times, cash flows and prices with accrued interest of the notes and bonds."""
    bonds = prices[prices["Security Type"].isin(["Note", "Bond"])]
    schedules = []
    for coupon, maturity in zip(bonds["Coupon"], bonds["Maturity"], strict=True):
        _, coupon_dates = bond_data_model.get_coupon_dates(PRICE_DATE, maturity.date())
        flows = np.full(len(coupon_dates), 100 * coupon / 2)
        flows[-1] += 100
        schedules.append(
            ([(day - PRICE_DATE).days / 365 for day in coupon_dates], flows)
        )
    width = max(len(times) for times, _ in schedules)
    times, flows = np.zeros((len(schedules), width)), np.zeros((len(schedules), width))
    for row, (schedule, cash) in enumerate(schedules):
        times[row, : len(schedule)] = schedule
        flows[row, : len(cash)] = cash
    accrued = [
        bond_data_model.get_treasury_accrued_interest(
            coupon, PRICE_DATE, maturity.date()
        )
        for coupon, maturity in zip(bonds["Coupon"], bonds["Maturity"], strict=True)
    ]
    return times, flows, bonds["Price"].to_numpy() + np.array(accrued)


@pytest.mark.parametrize("method", ["nelson_siegel", "svensson"])
def test_fit_bond_curve_recovers_the_curve(method):
    times, flows, prices = bond_inputs(synthetic_treasuries())
    truth = TRUE_CURVE if method == "svensson" else TRUE_CURVE.drop(["Beta 3", "Tau 2"])
    if method == "nelson_siegel":
        # Priced on a Nelson-Siegel curve instead.
        prices = fitted_curve_model.get_fitted_bond_prices(
            truth, times, flows, "nelson_siegel"
        )

    fit = fitted_curve_model.fit_bond_curve(times, flows, prices, method=method)

    assert fit.to_numpy() == pytest.approx(truth.to_numpy(), abs=1e-6)
    # The same fit again, from the same fixed starts, gives the same parameters.
    again = fitted_curve_model.fit_bond_curve(times, flows, prices, method=method)
    assert again.to_numpy() == pytest.approx(fit.to_numpy(), abs=1e-12)


@pytest.mark.parametrize("method", ["exponential_spline", "cubic_spline"])
def test_fit_bond_curve_splines(method):
    times, flows, prices = bond_inputs(synthetic_treasuries())
    fit = fitted_curve_model.fit_bond_curve(times, flows, prices, method=method)
    years = np.array([1, 2, 5, 10, 20, 29.0])

    # The splines follow the Svensson curve to within a basis point (the cubic spline only
    # as finely as its knots, a year or two apart at the short end), start at d(0) = 1 and
    # price the bonds to within a cent.
    assert -np.log(
        fitted_curve_model.get_fitted_discount_factors(fit, years, method)
    ) / years == pytest.approx(
        -np.log(
            fitted_curve_model.get_fitted_discount_factors(
                TRUE_CURVE, years, "svensson"
            )
        )
        / years,
        abs=1e-4,
    )
    assert fitted_curve_model.get_fitted_discount_factors(
        fit, 0.0, method
    ) == pytest.approx(1.0)
    assert fitted_curve_model.get_fitted_bond_prices(
        fit, times, flows, method
    ) == pytest.approx(prices, abs=0.01)


def test_fit_bond_curve_rejects_an_unidentified_fit():
    # Prices of a flat curve leave the slope, curvatures and decays of Svensson free.
    times, flows, _ = bond_inputs(synthetic_treasuries())
    flat = pd.Series(
        {"Beta 0": 0.04, "Beta 1": 0.0, "Beta 2": 0.0, "Tau 1": 1.0}, dtype=float
    )
    prices = fitted_curve_model.get_fitted_bond_prices(
        flat, times, flows, "nelson_siegel"
    )

    with pytest.raises(ValueError, match="not identified"):
        fitted_curve_model.fit_bond_curve(times, flows, prices, method="svensson")


def test_fit_bond_curve_invalid():
    times, flows, prices = bond_inputs(synthetic_treasuries())
    with pytest.raises(InvalidArgumentError, match="method must be one of"):
        fitted_curve_model.fit_bond_curve(times, flows, prices, method="polynomial")
    with pytest.raises(ValueError, match="at least"):
        fitted_curve_model.fit_bond_curve(
            times[:3], flows[:3], prices[:3], method="svensson"
        )


LABELS = ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]


def government_curves() -> pd.DataFrame:
    """Daily humped par curves of the first weeks of November 2026."""
    dates = pd.bdate_range("2026-11-02", "2026-11-20").to_period("D")
    years = np.array([curve_model.maturity_in_years(label) for label in LABELS])
    shape = (
        0.042
        - 0.004 * np.exp(-years / 0.8)
        + 0.012 * (1 - np.exp(-years / 6))
        - 0.004 * (1 - np.exp(-years / 25))
    )
    curves = shape[None, :] + 0.0001 * np.arange(len(dates))[:, None]
    return pd.DataFrame(curves, index=dates, columns=LABELS)


@pytest.fixture
def fixedincome(monkeypatch):
    clear_active_cache()
    curves = government_curves()
    monkeypatch.setattr(
        FixedIncome,
        "_get_country_yield_curves",
        lambda self, start_date: {
            "United States": lambda: curves,
            "Germany": lambda: curves[LABELS[3:]] - 0.015,
        },
    )
    monkeypatch.setattr(
        treasurydirect_model,
        "get_treasury_prices",
        lambda price_date: (synthetic_treasuries(mispriced=0.5), PRICE_DATE),
    )
    yield FixedIncome(start_date="2026-11-02", end_date="2026-11-20", rounding=6)
    clear_active_cache()


def test_get_zero_coupon_curve_interpolation(recorder, fixedincome):
    recorder.capture(
        fixedincome.get_zero_coupon_curve(
            ["United States", "Germany"], interpolation="monotone_convex"
        )
    )
    recorder.capture(
        fixedincome.get_zero_coupon_curve(
            interpolation="natural_cubic", maturities="1.5,4,15,25,40"
        )
    )


def test_get_zero_coupon_curve_default_is_unchanged(fixedincome):
    default = fixedincome.get_zero_coupon_curve()
    linear_par = fixedincome.get_zero_coupon_curve(maturities=LABELS)

    # Reading the linear_par curve at its own maturities gives the bootstrap's zero rates.
    assert linear_par.to_numpy() == pytest.approx(default.to_numpy(), abs=1e-12)


def test_get_forward_rate_and_par_yield_interpolation(recorder, fixedincome):
    for interpolation in interpolation_model.INTERPOLATIONS:
        recorder.capture(
            fixedincome.get_forward_rate(
                near_maturity=[1, 4], far_maturity=[5, 15], interpolation=interpolation
            )
        )
        recorder.capture(
            fixedincome.get_par_yield(
                years_to_maturity=[2, 4.5, 15, 25], interpolation=interpolation
            )
        )


def test_get_forward_rate_default_reads_the_zero_coupon_curve(fixedincome):
    zero_curve = fixedincome._latest_zero_curve("United States")
    near, far = 4.0, 15.0
    r1, r2 = np.interp([near, far], zero_curve.index, zero_curve.to_numpy())

    assert fixedincome.get_forward_rate(near, far, rounding=12).loc[
        4, 15
    ] == pytest.approx(((1 + r2) ** far / (1 + r1) ** near) ** (1 / (far - near)) - 1)


def test_par_yields_reprice_the_quoted_maturities(fixedincome):
    curve = government_curves().iloc[-1]
    for interpolation in interpolation_model.INTERPOLATIONS:
        par = fixedincome.get_par_yield(
            years_to_maturity=[2, 5, 10, 30], interpolation=interpolation, rounding=12
        )
        assert par.to_numpy() == pytest.approx(
            curve[["2Y", "5Y", "10Y", "30Y"]].to_numpy(), abs=1e-10
        )


def test_get_instantaneous_forward_curve(recorder, fixedincome):
    recorder.capture(fixedincome.get_instantaneous_forward_curve())
    recorder.capture(
        fixedincome.get_instantaneous_forward_curve(
            maturities="6M,2Y,10Y,40Y", country="Germany", interpolation="log_linear"
        )
    )


def test_get_smith_wilson_curve(recorder, fixedincome):
    recorder.capture(fixedincome.get_smith_wilson_curve())
    recorder.capture(
        fixedincome.get_smith_wilson_curve(
            maturities=[10, 20, 60, 100],
            country="Germany",
            ultimate_forward_rate=0.0345,
            last_liquid_point=20,
        )
    )
    recorder.capture(fixedincome.get_smith_wilson_curve(convergence_speed=0.1))


def test_get_smith_wilson_curve_keeps_the_par_curve(fixedincome):
    curve = fixedincome.get_smith_wilson_curve(maturities=[2, 5, 10, 30], rounding=12)
    par = fixedincome.get_par_yield(
        years_to_maturity=[2, 5, 10, 30], interpolation="monotone_convex", rounding=12
    )
    zero_rates = fixedincome.get_zero_coupon_curve(
        interpolation="monotone_convex", maturities=[2, 5, 10, 30], rounding=12
    )["United States"].iloc[-1]

    # Both curves price the same par bonds exactly, so their zero rates are close; the
    # forward converges to ln(1 + UFR) beyond the convergence point.
    quoted = government_curves().iloc[-1][["2Y", "5Y", "10Y", "30Y"]]
    assert par.to_numpy() == pytest.approx(quoted.to_numpy())
    assert curve["Zero Rate"].to_numpy() == pytest.approx(
        zero_rates.to_numpy(), abs=2e-4
    )
    far = fixedincome.get_smith_wilson_curve(maturities=[70], rounding=12)
    assert abs(
        far["Instantaneous Forward Rate"].iloc[0] - np.log1p(0.033)
    ) == pytest.approx(0.0001, abs=1e-9)


# The nonlinear fits (Svensson, Nelson-Siegel and the exponential spline's decay) reproduce to
# about 1e-8 between runs and platforms, which can still flip the sixth decimal of a value on a
# rounding boundary, so they are recorded at four; the cubic spline is an exact linear solve.
def test_get_fitted_bond_curve(recorder, fixedincome):
    recorder.capture(fixedincome.get_fitted_bond_curve(rounding=4))
    recorder.capture(
        fixedincome.get_fitted_bond_curve(method="nelson_siegel", rounding=4)
    )


def test_get_fitted_bond_curve_splines(recorder, fixedincome):
    recorder.capture(
        fixedincome.get_fitted_bond_curve(method="cubic_spline", maturities="1,5,10,25")
    )
    recorder.capture(
        fixedincome.get_fitted_bond_curve(
            method="exponential_spline", residuals=True, rounding=4
        )
    )


def test_get_fitted_bond_curve_finds_the_rich_bond(fixedincome):
    bonds = fixedincome.get_fitted_bond_curve(residuals=True, rounding=10)

    # The bill, the TIPS and the note within three months of maturity are left out; the
    # bond priced 0.5 above the curve stands out.
    assert len(bonds) == len(synthetic_treasuries()) - 3
    assert bonds["Price Residual"].abs().idxmax() == "91282C020"
    assert bonds.loc["91282C020", "Price Residual"] == pytest.approx(0.5, abs=0.05)
    assert bonds.loc["91282C020", "Yield Residual"] < 0


def test_curve_construction_invalid(fixedincome):
    with pytest.raises(InvalidArgumentError, match="interpolation must be one of"):
        fixedincome.get_forward_rate(interpolation="cubic")
    with pytest.raises(InvalidArgumentError, match="method must be one of"):
        fixedincome.get_fitted_bond_curve(method="polynomial")
    assert fixedincome.get_smith_wilson_curve(last_liquid_point=0.01).empty


@pytest.mark.parametrize(
    "function",
    [
        curve_model.bootstrap_zero_rates,
        interpolation_model.bootstrap_zero_curve,
        lambda maturities, rates: interpolation_model.get_discount_factors(
            maturities, rates, [1.0]
        ),
        lambda maturities, rates: interpolation_model.get_instantaneous_forward_rates(
            maturities, rates, [1.0]
        ),
    ],
)
def test_a_curve_needs_one_rate_for_each_maturity(function):
    with pytest.raises(ValueError, match="for each maturity"):
        function(np.array([1.0, 3.0, 5.0]), np.array([0.04]))
    with pytest.raises(ValueError, match="for each maturity"):
        function(np.array([]), np.array([]))


@pytest.mark.parametrize("frequency", [0, -1])
def test_bootstrapping_refuses_a_frequency_below_one(frequency):
    with pytest.raises(ValueError, match="frequency must be at least 1"):
        curve_model.bootstrap_zero_rates(
            np.array([0.5, 2.0]), np.array([0.04, 0.042]), frequency
        )
    with pytest.raises(ValueError, match="frequency must be at least 1"):
        interpolation_model.bootstrap_zero_curve([0.5, 2.0], [0.04, 0.042], frequency)
