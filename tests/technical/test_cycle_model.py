"""Cycle Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.technicals.cycle import cycle_model


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
