"""Cycle Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.technicals.cycle import cycle_model
from financetoolkit.utilities.error_model import InvalidArgumentError


def _cycle_prices() -> pd.Series:
    """A trend with a 20-period cycle and a smaller 7-period one, long enough for TA-Lib's lookback."""
    steps = np.arange(150)
    return pd.Series(
        100
        + 0.05 * steps
        + 3 * np.sin(2 * np.pi * steps / 20)
        + 0.5 * np.sin(2 * np.pi * steps / 7)
    )


def test_dominant_cycle_period_matches_talib():
    result = cycle_model.get_hilbert_dominant_cycle_period(_cycle_prices())

    assert result.iloc[:32].isna().all()
    assert result.iloc[32:].notna().all()
    # TA-Lib's HT_DCPERIOD; the period settles on the 20-period cycle.
    assert result.iloc[32] == pytest.approx(15.646696015449784)
    assert result.iloc[63] == pytest.approx(20.67183970960512)
    assert result.iloc[149] == pytest.approx(20.14381737990937)


def test_dominant_cycle_phase_matches_talib():
    result = cycle_model.get_hilbert_dominant_cycle_phase(_cycle_prices())

    assert result.iloc[:63].isna().all()
    assert result.iloc[63] == pytest.approx(18.673528277207062)
    assert result.iloc[149] == pytest.approx(165.24544650303184)
    assert result.dropna().between(-45, 315).all()


def test_phasor_components_match_talib():
    result = cycle_model.get_hilbert_phasor_components(_cycle_prices())

    assert list(result.columns) == ["In Phase", "Quadrature"]
    assert result.iloc[:32].isna().all().all()
    assert result.iloc[63].tolist() == pytest.approx(
        [1.8167821894897007, 2.9982875924747963]
    )
    assert result.iloc[149].tolist() == pytest.approx(
        [2.3371906315246505, -3.0862515574352436]
    )


def test_sine_wave_matches_talib():
    result = cycle_model.get_hilbert_sine_wave(_cycle_prices())

    assert list(result.columns) == ["Sine", "Lead Sine"]
    assert result.iloc[:63].isna().all().all()
    assert result.iloc[63].tolist() == pytest.approx(
        [0.32017532750058114, 0.896281627523067]
    )
    assert result.iloc[149].tolist() == pytest.approx(
        [0.2546788019409285, -0.5037053234110581]
    )


def test_instantaneous_trendline_matches_talib():
    result = cycle_model.get_hilbert_instantaneous_trendline(_cycle_prices())

    assert result.iloc[:63].isna().all()
    assert result.iloc[63] == pytest.approx(102.64828268146964)
    assert result.iloc[149] == pytest.approx(106.90944004585512)


def test_trend_mode_matches_talib():
    result = cycle_model.get_hilbert_trend_mode(_cycle_prices())

    # TA-Lib shows 0 during its lookback; the Toolkit leaves it empty.
    assert result.iloc[:63].isna().all()
    assert set(result.dropna().unique()) <= {0.0, 1.0}
    # TA-Lib's HT_TRENDMODE marks 52 of the 87 periods after its lookback as a trend.
    assert result.sum() == 52
    assert result.iloc[63] == 1
    assert result.iloc[149] == 1


def test_mesa_adaptive_moving_average_matches_talib():
    result = cycle_model.get_mesa_adaptive_moving_average(_cycle_prices())

    assert list(result.columns) == ["MAMA", "FAMA"]
    assert result.iloc[:32].isna().all().all()
    assert result["MAMA"].iloc[32] == pytest.approx(101.36021601058589)
    assert result.iloc[63].tolist() == pytest.approx(
        [102.99907871875766, 101.58197440707546]
    )
    assert result.iloc[149].tolist() == pytest.approx(
        [108.23342860292428, 106.45374060662009]
    )


def test_cycle_indicators_skip_missing_prices():
    prices = _cycle_prices()
    shifted = pd.concat([pd.Series([np.nan] * 5), prices], ignore_index=True)

    # Leading missing prices are left out, so the lookback counts from the first price, as
    # in TA-Lib.
    result = cycle_model.get_hilbert_dominant_cycle_period(shifted)
    expected = cycle_model.get_hilbert_dominant_cycle_period(prices)

    assert result.iloc[:37].isna().all()
    np.testing.assert_allclose(result.iloc[5:].to_numpy(), expected.to_numpy())

    # A series shorter than the lookback has no values.
    assert cycle_model.get_hilbert_trend_mode(prices.iloc[:40]).isna().all()
    assert (
        cycle_model.get_mesa_adaptive_moving_average(prices.iloc[:10])
        .isna()
        .all()
        .all()
    )


def test_swiss_army_knife_filters_match_talib():
    prices = _cycle_prices()

    # TA-Lib's SWAK_HP, SWAK_2PHP, SWAK_BP, SWAK_BUTTER and SWAK_GAUSS with a period of 20.
    expected = {
        "high_pass": [
            0.0,
            1.1809263632508717,
            -0.5002414658646386,
            -0.3983532283224965,
        ],
        "two_pole_high_pass": [
            0.0,
            0.8951209149289664,
            -0.6737373023236648,
            -0.7581624209599329,
        ],
        "band_pass": [0.0, 0.04168023215374073, 0.2327054799461198, 0.955742658385626],
        "butterworth": [
            100.0,
            100.04994903007196,
            102.39005265321173,
            109.26126235675781,
        ],
        "gaussian": [100.0, 100.19979612028784, 102.38746533742817, 109.30433044093685],
    }
    for filter_type, values in expected.items():
        result = cycle_model.get_swiss_army_knife_filter(prices, 20, filter_type)

        assert result.notna().all()
        assert result.iloc[[0, 1, 9, 149]].tolist() == pytest.approx(values)


def test_swiss_army_knife_filter_rejects_invalid_input():
    with pytest.raises(InvalidArgumentError, match="method must be one of"):
        cycle_model.get_swiss_army_knife_filter(_cycle_prices(), method="band_stop")
    with pytest.raises(ValueError, match="at least 5"):
        cycle_model.get_swiss_army_knife_filter(_cycle_prices(), window=4)
    with pytest.raises(ValueError, match="delta"):
        cycle_model.get_swiss_army_knife_filter(_cycle_prices(), delta=0.6)


def test_super_smoother_filter_matches_pandas_ta():
    prices = _cycle_prices()

    # pandas-ta-classic's ssf with three poles.
    three_poles = cycle_model.get_super_smoother_filter(prices, 10, 3)
    assert three_poles.iloc[[0, 1, 2, 63, 149]].tolist() == pytest.approx(
        [
            100.0,
            101.36796672435885,
            102.35081971296832,
            103.86503313418095,
            109.75173990544826,
        ]
    )

    # pandas-ta-classic's two-pole ssf applied to the average of the last two prices, Ehlers' 2013 form.
    two_poles = cycle_model.get_super_smoother_filter(prices, 10, 2)
    assert two_poles.iloc[[0, 1, 2, 63, 149]].tolist() == pytest.approx(
        [
            100.0,
            101.36796672435885,
            102.05495695336685,
            103.75969956047857,
            109.72074394230435,
        ]
    )

    with pytest.raises(ValueError, match="poles"):
        cycle_model.get_super_smoother_filter(prices, poles=4)


def test_even_better_sinewave_matches_pandas_ta():
    prices = _cycle_prices()
    # Flat before the 12th price, where pandas-ta-classic starts the filters.
    flat_start = prices.copy()
    flat_start.iloc[:11] = flat_start.iloc[0]

    result = cycle_model.get_even_better_sinewave(flat_start, 40, 10)

    assert result.iloc[:40].isna().all()
    assert result.iloc[[40, 63, 149]].tolist() == pytest.approx(
        [-0.7954201926449976, 0.94033245344123, 0.9850251715493064]
    )
    # From the first price the start differs, which fades.
    assert cycle_model.get_even_better_sinewave(prices).iloc[149] == pytest.approx(
        0.9850251715478667
    )


def test_detrended_synthetic_price_matches_pandas_ta_averages():
    result = cycle_model.get_detrended_synthetic_price(_cycle_prices(), 20)

    # pandas-ta-classic's ema(5) - ema(10).
    assert result.iloc[:9].isna().all()
    assert result.iloc[[9, 63, 149]].tolist() == pytest.approx(
        [0.09451196293056796, 0.9161444969996069, 0.4869817943215651]
    )


def test_mesa_sine_wave_matches_pandas_ta():
    result = cycle_model.get_mesa_sine_wave(_cycle_prices(), 5)

    assert list(result.columns) == ["Sine", "Lead Sine"]
    # The first full window ends on the 5th price, a price before pandas-ta-classic and Tulip start.
    assert result.iloc[:4].isna().all().all()
    assert result.iloc[4].notna().all()
    assert result.iloc[9].tolist() == pytest.approx(
        [-0.9816143074765472, -0.5591369676674672]
    )
    assert result.iloc[63].tolist() == pytest.approx(
        [0.7123444103506107, 0.007434764441586148]
    )
    assert result.dropna().abs().le(1).all().all()


def test_center_of_gravity_oscillator_matches_talib():
    result = cycle_model.get_center_of_gravity_oscillator(_cycle_prices(), 10)

    assert list(result.columns) == ["Center of Gravity", "Trigger"]
    assert result["Center of Gravity"].iloc[:9].isna().all()
    # TA-Lib's CG.
    assert result["Center of Gravity"].iloc[[9, 63, 149]].tolist() == pytest.approx(
        [-5.486768506497331, -5.443123592362679, -5.487616625075256]
    )
    assert result["Trigger"].iloc[64] == result["Center of Gravity"].iloc[63]

    # Flat prices balance in the middle of the window.
    flat = cycle_model.get_center_of_gravity_oscillator(pd.Series([10.0] * 12), 10)
    assert flat["Center of Gravity"].iloc[-1] == pytest.approx(-5.5)


def test_laguerre_relative_strength_index_matches_pandas_ta():
    result = cycle_model.get_laguerre_relative_strength_index(_cycle_prices(), 0.5)

    # pandas-ta-classic's lrsi divided by 100.
    assert np.isnan(result.iloc[0])
    assert result.iloc[[1, 2, 63, 149]].tolist() == pytest.approx(
        [0.7142857142857132, 0.8000000000000026, 1.0, 0.84145775443844]
    )

    # Without damping the elements are the last four prices; once they are equal the value
    # before is kept, as in Ehlers' listing.
    prices = pd.Series([10.0, 11.0, 12.0, 11.0, 12.0, 12.0, 12.0, 12.0])
    held = cycle_model.get_laguerre_relative_strength_index(prices, gamma=0)
    assert held.iloc[4] == pytest.approx(2 / 3)
    assert held.iloc[5:7].tolist() == pytest.approx([0.5, 1.0])
    assert held.iloc[7] == held.iloc[6]

    # Flat prices never move the elements apart, so there is no value to keep.
    assert (
        cycle_model.get_laguerre_relative_strength_index(pd.Series([5.0] * 6))
        .isna()
        .all()
    )
