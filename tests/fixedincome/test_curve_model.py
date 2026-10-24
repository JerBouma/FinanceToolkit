"""Yield Curve Fitting, Bootstrapping and Principal Components Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.fixedincome import curve_model

# pylint: disable=missing-function-docstring

MATURITIES = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30.0])


def test_maturity_in_years():
    assert curve_model.maturity_in_years("3M") == pytest.approx(0.25)
    assert curve_model.maturity_in_years("1.5M") == pytest.approx(0.125)
    assert curve_model.maturity_in_years("10Y") == 10
    assert curve_model.maturity_in_years("52W") == pytest.approx(364 / 365)
    assert curve_model.maturity_in_years(7) == 7
    assert curve_model.maturity_in_years("0.5") == 0.5
    assert curve_model.maturity_in_years("3month") == pytest.approx(0.25)
    assert curve_model.maturity_in_years("2 years") == 2
    with pytest.raises(ValueError):
        curve_model.maturity_in_years("ten years")


@pytest.mark.parametrize(
    "parameters",
    [(0.045, -0.01, 0.02, -0.015, 1.5, 9.0), (0.03, 0.01, -0.02, 0.01, 0.8, 4.0)],
)
def test_svensson_fit_recovers_the_curve(parameters):
    yields = curve_model.nelson_siegel_svensson_yield(MATURITIES, *parameters)
    fit = curve_model.fit_nelson_siegel_svensson(MATURITIES, yields)

    assert fit["Root Mean Squared Error"] < 2e-5
    assert fit["Beta 0"] == pytest.approx(parameters[0], abs=5e-4)
    assert fit["Tau 2"] >= curve_model.TAU_RATIO * fit["Tau 1"]


def test_nelson_siegel_fit_recovers_the_parameters():
    yields = curve_model.nelson_siegel_svensson_yield(
        MATURITIES, 0.05, -0.02, 0.01, 0.0, 2.0, 1.0
    )
    fit = curve_model.fit_nelson_siegel_svensson(MATURITIES, yields, svensson=False)

    assert list(fit) == [
        "Beta 0",
        "Beta 1",
        "Beta 2",
        "Tau 1",
        "Root Mean Squared Error",
    ]
    assert fit["Beta 0"] == pytest.approx(0.05, abs=1e-5)
    assert fit["Beta 1"] == pytest.approx(-0.02, abs=1e-5)
    assert fit["Beta 2"] == pytest.approx(0.01, abs=1e-4)
    assert fit["Tau 1"] == pytest.approx(2.0, rel=1e-2)


def test_fit_of_a_history_handles_missing_maturities():
    yields = curve_model.nelson_siegel_svensson_yield(
        MATURITIES, 0.04, -0.01, 0.01, 0.0, 1.5, 1.0
    )
    curves = pd.DataFrame([yields, yields, yields], columns=MATURITIES)
    # The second curve misses two maturities and the third has too few to fit.
    curves.iloc[1, [0, 4]] = np.nan
    curves.iloc[2, 3:] = np.nan

    fits = curve_model.fit_nelson_siegel_curves(curves, svensson=False)

    assert fits.loc[0, "Beta 0"] == pytest.approx(0.04, abs=1e-5)
    assert fits.loc[1, "Beta 0"] == pytest.approx(0.04, abs=1e-5)
    assert fits.loc[2].isna().all()


def test_fit_raises_with_too_few_maturities():
    with pytest.raises(ValueError, match="maturities"):
        curve_model.fit_nelson_siegel_svensson(
            np.array([1, 2, 5.0]), np.array([0.03, 0.035, 0.04])
        )


def test_bootstrap_of_a_flat_par_curve():
    maturities = np.array([0.5, 1, 2, 5, 10, 30.0])
    zero_rates = curve_model.bootstrap_zero_rates(maturities, np.full(6, 0.05))

    # A flat par curve of 5% compounded semi-annually is a zero curve of 5.0625% compounded
    # annually, bills included.
    assert zero_rates == pytest.approx(np.full(6, 1.025**2 - 1), abs=1e-10)


def test_bootstrap_reprices_the_par_bonds():
    maturities = np.array([0.5, 1, 2, 3, 5, 7, 10.0])
    par_yields = np.array([0.03, 0.032, 0.035, 0.037, 0.04, 0.042, 0.045])
    zero_rates = curve_model.bootstrap_zero_rates(maturities, par_yields)

    for maturity, coupon in zip(maturities[2:], par_yields[2:], strict=True):
        times = np.arange(0.5, maturity + 1e-9, 0.5)
        discount = (1 + np.interp(times, maturities, zero_rates)) ** -times
        # The zero rates at the curve's own maturities reprice its par bonds exactly; at
        # the coupon dates in between they are interpolated, so allow a small difference.
        price = coupon / 2 * discount.sum() + discount[-1]
        assert price == pytest.approx(1, abs=2e-3)


def test_principal_components_of_level_and_slope_moves():
    rng = np.random.default_rng(0)
    maturities = np.array([1, 2, 5, 10, 30.0])
    level = rng.normal(0, 0.01, 500).cumsum()
    slope = rng.normal(0, 0.003, 500).cumsum()
    curves = pd.DataFrame(
        0.03 + level[:, None] + slope[:, None] * np.log(maturities)[None, :],
        columns=[f"{maturity:g}Y" for maturity in maturities],
    )

    loadings, scores, explained = curve_model.get_principal_components(curves, 2)

    assert explained.sum() == pytest.approx(1, abs=1e-6)
    assert explained["PC1"] > explained["PC2"]
    # The first component is a rise in all yields, the second moves the ends apart.
    assert (loadings["PC1"] > 0).all()
    assert loadings.loc["30Y", "PC2"] * loadings.loc["1Y", "PC2"] < 0
    assert len(scores) == len(curves) - 1


def test_the_yield_at_maturity_zero_is_the_short_rate():
    yields = curve_model.nelson_siegel_svensson_yield(
        [0.0, 1e-8], 0.04, -0.02, 0.01, 0.005, 1.5, 8.0
    )

    assert yields == pytest.approx([0.02, 0.02])


def test_the_level_stays_at_or_above_zero():
    # A curve whose unbounded fit offsets a negative level with the second curvature.
    maturities = np.array([1, 2, 3, 5, 7, 10, 20, 30.0])
    yields = (
        curve_model.nelson_siegel_svensson_yield(
            maturities, 0.02, 0.01, -0.03, 0.06, 1.0, 12.0
        )
        + np.array([0, 2, -1, 1, -2, 1, 0, -1]) * 1e-4
    )

    fit = curve_model.fit_nelson_siegel_svensson(maturities, yields)

    assert fit["Beta 0"] >= 0
    # Without a maturity under a year the first decay stays at least a third of a year,
    # so the short end is not bent far away from the shortest yield.
    assert fit["Tau 1"] >= 1 / 3 - 1e-9
    short_end = curve_model.nelson_siegel_svensson_yield(
        [0.25],
        *[
            fit[name]
            for name in ("Beta 0", "Beta 1", "Beta 2", "Beta 3", "Tau 1", "Tau 2")
        ],
    )[0]
    assert abs(short_end - yields[0]) < 0.01


def test_the_fit_is_polished_beyond_the_grid():
    maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30.0])
    yields = curve_model.nelson_siegel_svensson_yield(
        maturities, 0.045, -0.01, 0.02, -0.015, 1.792, 15.918
    )

    fit = curve_model.fit_nelson_siegel_svensson(maturities, yields)

    assert fit["Root Mean Squared Error"] < 1e-6


def test_bill_yields_as_bond_equivalent_yields():
    annual = curve_model.bill_yields_to_annual(
        np.array([0.25, 0.5, 1.0]), np.full(3, 0.04), 2, "bond-equivalent"
    )

    assert annual[0] == pytest.approx(1.01**4 - 1)
    assert annual[1] == pytest.approx(1.02**2 - 1)
    assert annual[2] == pytest.approx(1.02**2 - 1)
    assert curve_model.bill_yields_to_annual(
        np.array([0.25]), np.array([0.04])
    ) == pytest.approx([1.02**2 - 1])


def test_bill_yields_with_simple_interest_on_365_and_360_days():
    maturities = np.array([0.25, 0.5, 1.0])
    yields = np.full(3, 0.04)

    # Government of Canada bills: P = 100 / (1 + y t / 365), so the whole year is simple.
    canada = curve_model.bill_yields_to_annual(maturities, yields, 2, "act/365")
    assert canada == pytest.approx((1 + 0.04 * maturities) ** (1 / maturities) - 1)
    assert canada[2] == pytest.approx(0.04)

    # Swedish statsskuldväxlar on ACT/360 (Riksgälden's example settles 100 million SEK at
    # 3.498% over 371 days for 96,520,547 SEK, 1e8 / (1 + 0.03498 * 371 / 360)): a 6-month
    # bill earns the yield over 182.5 days of a 360-day year.
    sweden = curve_model.bill_yields_to_annual(maturities, yields, 1, "act/360")
    assert (1 + sweden[1]) ** 0.5 == pytest.approx(1 + 0.04 * 182.5 / 360)
    assert sweden[2] == pytest.approx(0.04 * 365 / 360)

    with pytest.raises(ValueError):
        curve_model.bill_yields_to_annual(maturities, yields, 2, "30/360")


def test_bootstrap_keeps_the_bill_convention_on_the_coupon_dates():
    # A flat curve of 4% Swedish yields: the 1-year bill earns 4% on a 360-day year, so the
    # 1-year zero rate is 4% * 365 / 360 and the 2-year bond is discounted with it.
    maturities = np.array([0.5, 1.0, 2.0])
    zero_rates = curve_model.bootstrap_zero_rates(
        maturities, np.full(3, 0.04), 1, "act/360"
    )
    one_year = 0.04 * 365 / 360
    discount_2 = (1 - 0.04 / (1 + one_year)) / 1.04

    assert zero_rates[1] == pytest.approx(one_year)
    assert zero_rates[2] == pytest.approx(discount_2 ** (-1 / 2) - 1)
