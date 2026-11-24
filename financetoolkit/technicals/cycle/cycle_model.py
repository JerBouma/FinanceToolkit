"""Cycle Model: indicators of the cycles in the price"""

__docformat__ = "google"

import math

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from financetoolkit.technicals.overlap import overlap_model
from financetoolkit.utilities.error_model import InvalidArgumentError, check_option

# The bar on which the filters start from zero and the bars left empty while they settle: the indicators that
# only need the period start on bar 12 and show a value from bar 32, those that also need the phase over a full
# cycle of up to 50 bars start on bar 37 and show a value from bar 63. These are the start and warm-up of TA-Lib's
# HT functions and MAMA.
PERIOD_START, PERIOD_LOOKBACK = 12, 32
PHASE_START, PHASE_LOOKBACK = 37, 63


def _get_hilbert_transform(prices: pd.Series, start: int) -> pd.DataFrame:
    """
    The Hilbert transform and homodyne discriminator of Ehlers (2001) that the HT indicators
    and MAMA share, in the steps Ehlers describes.

    1. Smooth the price with a 4-bar weighted moving average.
    2. Detrend it with a discrete Hilbert transform whose bandwidth follows the period of the
       bar before (0.075 * Period + 0.54). The detrended price lagged 3 bars is the in-phase
       component I1, its Hilbert transform the quadrature Q1.
    3. Advance both by 90 degrees (jI, jQ) and smooth the phasor I2 = I1 - jQ,
       Q2 = Q1 + jI.
    4. Multiply the phasor with the conjugate of the phasor of the bar before (the homodyne
       discriminator): the angle of the smoothed product is the phase change per bar, and
       360 degrees divided by it the period, limited to between 0.67 and 1.5 times the period
       before and to 6-50 bars, then smoothed twice.
    5. From the smoothed period follow the dominant cycle phase, a discrete Fourier transform
       of the smoothed price over one cycle, and the instantaneous trendline, the price
       averaged over one cycle and smoothed with a 4-bar weighted moving average.

    By convention every filter is zero before bar `start` and starts from there, as in Ehlers'
    listings and TA-Lib, so the values only settle after a number of bars; the callers leave
    the warm-up empty.

    Args:
        prices (pd.Series): The prices without missing values.
        start (int): The bar on which the filters start, PERIOD_START or PHASE_START.

    Returns:
        pd.DataFrame: Per bar the smoothed price, the in-phase and quadrature components, the
        phase of the phasor (degrees), the smoothed period, the dominant cycle phase
        (degrees) and the instantaneous trendline.
    """
    price = prices.to_numpy(dtype=float)
    bars = len(price)
    degrees = 180 / math.pi

    # 1. The 4-bar weighted moving average of the price.
    smooth = np.zeros(bars)
    smooth[3:] = (4 * price[3:] + 3 * price[2:-1] + 2 * price[1:-2] + price[:-3]) / 10
    smooth[:start] = 0.0

    def hilbert(x: np.ndarray, t: int, bandwidth: float) -> float:
        # The outer taps first, then the inner ones: the values depend on this order to the last digit.
        return (
            0.0962 * x[t] - 0.0962 * x[t - 6] - 0.5769 * x[t - 4] + 0.5769 * x[t - 2]
        ) * bandwidth

    detrender, i1, q1 = np.zeros(bars), np.zeros(bars), np.zeros(bars)
    smooth_period = np.full(bars, np.nan)
    i2 = q2 = re = im = period = smoothed = 0.0

    for t in range(start, bars):
        bandwidth = 0.075 * period + 0.54

        # 2. The in-phase and quadrature components.
        detrender[t] = hilbert(smooth, t, bandwidth)
        q1[t] = hilbert(detrender, t, bandwidth)
        i1[t] = detrender[t - 3]

        # 3. The phasor advanced by 90 degrees, smoothed against noise.
        ji = hilbert(i1, t, bandwidth)
        jq = hilbert(q1, t, bandwidth)
        i2_before, q2_before = i2, q2
        i2 = 0.2 * (i1[t] - jq) + 0.8 * i2
        q2 = 0.2 * (q1[t] + ji) + 0.8 * q2

        # 4. The homodyne discriminator and the period it measures.
        re = 0.2 * (i2 * i2_before + q2 * q2_before) + 0.8 * re
        im = 0.2 * (i2 * q2_before - q2 * i2_before) + 0.8 * im

        period_before = period
        if im != 0 and re != 0:
            period = 360 / (math.atan(im / re) * degrees)
        period = max(min(period, 1.5 * period_before), 0.67 * period_before)
        period = min(max(period, 6), 50)
        period = 0.2 * period + 0.8 * period_before
        smoothed = 0.33 * period + 0.67 * smoothed
        smooth_period[t] = smoothed

    # The phase of the phasor; by convention 0 where the in-phase component is 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        phase = np.where(i1 != 0, np.arctan(q1 / i1) * degrees, 0.0)

    # 5. The dominant cycle phase and the price averaged over one cycle.
    dc_phase = np.full(bars, np.nan)
    cycle_average = np.zeros(bars)
    angle = 0.0
    for t in range(start, bars):
        cycle = int(smooth_period[t] + 0.5)
        real_part = imag_part = 0.0
        for bar in range(cycle):
            real_part += math.sin(2 * math.pi * bar / cycle) * smooth[t - bar]
            imag_part += math.cos(2 * math.pi * bar / cycle) * smooth[t - bar]

        # Without an imaginary part the phase keeps its value and turns 90 degrees towards the real part.
        if imag_part != 0:
            angle = math.atan(real_part / imag_part) * degrees
        else:
            angle += 90 * np.sign(real_part)
        # A quarter cycle for the sine, and the lag of the smoothing.
        angle += 90 + 360 / smooth_period[t]
        if imag_part < 0:
            angle += 180
        if angle > 315:  # noqa: PLR2004
            angle -= 360
        dc_phase[t] = angle

        if cycle:
            cycle_average[t] = price[max(t - cycle + 1, 0) : t + 1].sum() / cycle

    trendline = np.full(bars, np.nan)
    trendline[3:] = (
        4 * cycle_average[3:]
        + 3 * cycle_average[2:-1]
        + 2 * cycle_average[1:-2]
        + cycle_average[:-3]
    ) / 10

    output = pd.DataFrame(
        {
            "Smoothed Price": smooth,
            "In Phase": i1,
            "Quadrature": q1,
            "Phase": phase,
            "Smoothed Period": smooth_period,
            "Dominant Cycle Phase": dc_phase,
            "Trendline": trendline,
        },
        index=prices.index,
    )
    output.iloc[:start] = np.nan

    return output


def get_hilbert_dominant_cycle_period(prices: pd.Series) -> pd.Series:
    """
    Calculate the Hilbert Transform Dominant Cycle Period (HT_DCPERIOD): the length, in periods, of the main cycle
    in the prices.

    The dominant cycle period is the length in bars of the cycle that dominates the price,
    measured with Ehlers' homodyne discriminator: the phase of the analytic signal (the
    in-phase and quadrature components of the Hilbert transform) advances 360 degrees per
    cycle, so the phase change per bar gives the period.

    The formula is as follows:

    - Smooth = (4 * Price + 3 * Price(t-1) + 2 * Price(t-2) + Price(t-3)) / 10
    - H(x) = (0.0962 * x + 0.5769 * x(t-2) - 0.5769 * x(t-4) - 0.0962 * x(t-6)) *
      (0.075 * Period(t-1) + 0.54)
    - Detrender = H(Smooth), Q1 = H(Detrender), I1 = Detrender(t-3)
    - I2 = 0.2 * (I1 - H(Q1)) + 0.8 * I2(t-1), Q2 = 0.2 * (Q1 + H(I1)) + 0.8 * Q2(t-1)
    - Re = 0.2 * (I2 * I2(t-1) + Q2 * Q2(t-1)) + 0.8 * Re(t-1)
    - Im = 0.2 * (I2 * Q2(t-1) - Q2 * I2(t-1)) + 0.8 * Im(t-1)
    - Period = 360 / arctan(Im / Re) in degrees, limited to between 0.67 and 1.5 times
      Period(t-1) and to between 6 and 50 bars, then 0.2 * Period + 0.8 * Period(t-1)
    - Dominant Cycle Period = 0.33 * Period + 0.67 * Dominant Cycle Period(t-1)

    Reference: Ehlers, J.F. (2001). "Rocket Science for Traders: Digital Signal Processing
    Applications." John Wiley & Sons.

    Referenced against:
    - TA-Lib, HT_DCPERIOD: https://ta-lib.org/functions/ht_dcperiod.html

    Shares similar logic:
    - pandas-ta-classic, ht_dcperiod: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/cycles/ht_dcperiod.py#L14

    Notes:
        - Equals TA-Lib's HT_DCPERIOD: the recursion starts from zero on bar 12 of the
          series and the first 32 bars are NaN, TA-Lib's lookback.

    Args:
        prices (pd.Series): Series of prices.

    Returns:
        pd.Series: The dominant cycle period in bars.
    """
    clean = prices.dropna()
    result = _get_hilbert_transform(clean, PERIOD_START)["Smoothed Period"]

    result.iloc[:PERIOD_LOOKBACK] = np.nan

    return result.reindex(prices.index)


def get_hilbert_dominant_cycle_phase(prices: pd.Series) -> pd.Series:
    """
    Calculate the Hilbert Transform Dominant Cycle Phase (HT_DCPHASE): where the prices are within their main cycle,
    in degrees.

    The dominant cycle phase is the position of the price within its dominant cycle, in
    degrees: the angle of the discrete Fourier transform of the smoothed price over one
    dominant cycle period, corrected for the lag of the smoothing. It runs from -45 to 315
    degrees and wraps around once per cycle.

    The formula is as follows:

    - N = the dominant cycle period (HT_DCPERIOD) rounded to the nearest bar
    - Real = Σ sin(360 * i / N) * Smooth(t-i), Imag = Σ cos(360 * i / N) * Smooth(t-i)
      for i from 0 to N - 1
    - Phase = arctan(Real / Imag) + 90 + 360 / Dominant Cycle Period, plus 180 when Imag is
      negative, minus 360 above 315

    Reference: Ehlers, J.F. (2001). "Rocket Science for Traders: Digital Signal Processing
    Applications." John Wiley & Sons.

    Referenced against:
    - TA-Lib, HT_DCPHASE: https://ta-lib.org/functions/ht_dcphase.html

    Shares similar logic:
    - pandas-ta-classic, ht_dcphase: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/cycles/ht_dcphase.py#L14

    Notes:
        - Equals TA-Lib's HT_DCPHASE, in degrees as in TA-Lib and Ehlers: the recursion
          starts from zero on bar 37 of the series and the first 63 bars are NaN, TA-Lib's
          lookback.

    Args:
        prices (pd.Series): Series of prices.

    Returns:
        pd.Series: The dominant cycle phase in degrees.
    """
    clean = prices.dropna()
    result = _get_hilbert_transform(clean, PHASE_START)["Dominant Cycle Phase"]

    result.iloc[:PHASE_LOOKBACK] = np.nan

    return result.reindex(prices.index)


def get_hilbert_phasor_components(prices: pd.Series) -> pd.DataFrame:
    """
    Calculate the Hilbert Transform Phasor Components (HT_PHASOR): the in-phase and quadrature parts of the prices'
    cycle, from which its phase and period follow.

    The phasor components are the in-phase component (the detrended, smoothed price lagged
    3 bars) and the quadrature component (its Hilbert transform, the same cycle shifted by
    90 degrees). Together they form the analytic signal whose angle is the phase of the
    cycle and whose length is its amplitude, in price units.

    The formula is as follows:

    - Smooth = (4 * Price + 3 * Price(t-1) + 2 * Price(t-2) + Price(t-3)) / 10
    - H(x) = (0.0962 * x + 0.5769 * x(t-2) - 0.5769 * x(t-4) - 0.0962 * x(t-6)) *
      (0.075 * Period(t-1) + 0.54)
    - In Phase = H(Smooth)(t-3)
    - Quadrature = H(H(Smooth))

    Reference: Ehlers, J.F. (2001). "Rocket Science for Traders: Digital Signal Processing
    Applications." John Wiley & Sons.

    Referenced against:
    - TA-Lib, HT_PHASOR: https://ta-lib.org/functions/ht_phasor.html

    Shares similar logic:
    - pandas-ta-classic, ht_phasor: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/cycles/ht_phasor.py#L14

    Notes:
        - Equals TA-Lib's HT_PHASOR: the recursion starts from zero on bar 12 of the
          series and the first 32 bars are NaN, TA-Lib's lookback.

    Args:
        prices (pd.Series): Series of prices.

    Returns:
        pd.DataFrame: The "In Phase" and "Quadrature" components.
    """
    clean = prices.dropna()
    result = _get_hilbert_transform(clean, PERIOD_START)[["In Phase", "Quadrature"]]

    result.iloc[:PERIOD_LOOKBACK] = np.nan

    return result.reindex(prices.index)


def get_hilbert_sine_wave(prices: pd.Series) -> pd.DataFrame:
    """
    Calculate the Hilbert Transform SineWave (HT_SINE): the sine of the phase of the prices' main cycle and its
    lead, whose crossings mark the turning points of the cycle.

    The sine wave is the sine of the dominant cycle phase, and the lead sine the sine of the
    phase advanced by 45 degrees. In a cycling market the two lines cross a little before the
    turning points of the cycle; in a trend the phase barely moves and the lines run
    side by side.

    The formula is as follows:

    - Sine = sin(Dominant Cycle Phase)
    - Lead Sine = sin(Dominant Cycle Phase + 45 degrees)

    Reference: Ehlers, J.F. (2001). "Rocket Science for Traders: Digital Signal Processing
    Applications." John Wiley & Sons.

    Referenced against:
    - TA-Lib, HT_SINE: https://ta-lib.org/functions/ht_sine.html

    Shares similar logic:
    - pandas-ta-classic, ht_sine: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/cycles/ht_sine.py#L14

    Notes:
        - Equals TA-Lib's HT_SINE: the recursion starts from zero on bar 37 of the series
          and the first 63 bars are NaN, TA-Lib's lookback.

    Args:
        prices (pd.Series): Series of prices.

    Returns:
        pd.DataFrame: The "Sine" and "Lead Sine", between -1 and 1.
    """
    clean = prices.dropna()
    phase = np.deg2rad(
        _get_hilbert_transform(clean, PHASE_START)["Dominant Cycle Phase"]
    )
    result = pd.DataFrame(
        {"Sine": np.sin(phase), "Lead Sine": np.sin(phase + np.pi / 4)}
    )

    result.iloc[:PHASE_LOOKBACK] = np.nan

    return result.reindex(prices.index)


def get_hilbert_instantaneous_trendline(prices: pd.Series) -> pd.Series:
    """
    Calculate the Hilbert Transform Instantaneous Trendline (HT_TRENDLINE): the average of the prices over their
    main cycle, a trendline without the cycle's swings.

    The instantaneous trendline is the price averaged over its dominant cycle period: the
    average over exactly one cycle removes the cycle and leaves the trend. The averages are
    smoothed with a 4-bar weighted moving average.

    The formula is as follows:

    - N = the dominant cycle period (HT_DCPERIOD) rounded to the nearest bar
    - Average = the simple average of the latest N prices
    - Trendline = (4 * Average + 3 * Average(t-1) + 2 * Average(t-2) + Average(t-3)) / 10

    Reference: Ehlers, J.F. (2001). "Rocket Science for Traders: Digital Signal Processing
    Applications." John Wiley & Sons.

    Referenced against:
    - TA-Lib, HT_TRENDLINE: https://ta-lib.org/functions/ht_trendline.html
    - pandas-ta-classic, ht_trendline: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/overlap/ht_trendline.py#L14

    Notes:
        - Equals TA-Lib's HT_TRENDLINE: the recursion starts from zero on bar 37 of the
          series and the first 63 bars are NaN, TA-Lib's lookback.

    Args:
        prices (pd.Series): Series of prices.

    Returns:
        pd.Series: The instantaneous trendline in price units.
    """
    clean = prices.dropna()
    result = _get_hilbert_transform(clean, PHASE_START)["Trendline"]

    result.iloc[:PHASE_LOOKBACK] = np.nan

    return result.reindex(prices.index)


def get_hilbert_trend_mode(prices: pd.Series) -> pd.Series:
    """
    Calculate the Hilbert Transform Trend vs Cycle Mode (HT_TRENDMODE): whether the prices are trending (1) or
    moving in a cycle (0).

    The trend mode is 1 when the price trends and 0 when it moves in a cycle. A crossing of
    the sine and lead sine marks a cycle, which lasts at least half a dominant cycle period;
    a phase that advances at the rate of the dominant cycle also marks a cycle; a smoothed
    price 1.5% or more away from the instantaneous trendline always marks a trend.

    The formula is as follows:

    - Trend = 1, and 0 (with the days in trend restarting at zero) when Sine and Lead Sine
      cross
    - Trend = 0 while the days in trend are fewer than half the dominant cycle period
    - Trend = 0 when the phase change is between 0.67 and 1.5 times 360 / dominant cycle
      period
    - Trend = 1 when |Smooth - Trendline| / Trendline >= 0.015

    Reference: Ehlers, J.F. (2001). "Rocket Science for Traders: Digital Signal Processing
    Applications." John Wiley & Sons.

    Referenced against:
    - TA-Lib, HT_TRENDMODE: https://ta-lib.org/functions/ht_trendmode.html

    Shares similar logic:
    - pandas-ta-classic, ht_trendmode: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/cycles/ht_trendmode.py#L14

    Notes:
        - Equals TA-Lib's HT_TRENDMODE after its lookback: the recursion starts from zero on
          bar 37 of the series and the first 63 bars are NaN, where TA-Lib shows 0. The sine,
          lead sine and phase before the first bar count as 0, so a crossing and the phase
          change on the first bar are measured from zero.

    Args:
        prices (pd.Series): Series of prices.

    Returns:
        pd.Series: 1 in a trend and 0 in a cycle.
    """
    clean = prices.dropna()
    hilbert = _get_hilbert_transform(clean, PHASE_START).iloc[PHASE_START:]

    phase = hilbert["Dominant Cycle Phase"]
    smooth_period = hilbert["Smoothed Period"]
    trendline = hilbert["Trendline"]
    sine = np.sin(np.deg2rad(phase))
    lead_sine = np.sin(np.deg2rad(phase + 45))
    sine_before = sine.shift(1, fill_value=0.0)
    lead_sine_before = lead_sine.shift(1, fill_value=0.0)

    crossing = ((sine > lead_sine) & (sine_before <= lead_sine_before)) | (
        (sine < lead_sine) & (sine_before >= lead_sine_before)
    )
    # The bars since the last crossing, counting the crossing itself (or since the first bar).
    days_in_trend = crossing.groupby(crossing.cumsum()).cumcount() + 1

    phase_change = phase - phase.shift(1, fill_value=0.0)
    cycle_rate = 360 / smooth_period
    with np.errstate(divide="ignore", invalid="ignore"):
        far_from_trendline = (trendline != 0) & (
            ((hilbert["Smoothed Price"] - trendline) / trendline).abs()
            >= 0.015  # noqa: PLR2004
        )

    cycle = (
        crossing
        | (days_in_trend < 0.5 * smooth_period)
        | ((0.67 * cycle_rate < phase_change) & (phase_change < 1.5 * cycle_rate))
    )
    trend_mode = pd.Series(
        np.where(far_from_trendline | ~cycle, 1.0, 0.0), index=hilbert.index
    )

    trend_mode.iloc[: PHASE_LOOKBACK - PHASE_START] = np.nan

    return trend_mode.reindex(prices.index)


def get_mesa_adaptive_moving_average(
    prices: pd.Series, fast_limit: float = 0.5, slow_limit: float = 0.05
) -> pd.DataFrame:
    """
    Calculate the MESA Adaptive Moving Average (MAMA) and the Following Adaptive Moving Average (FAMA): averages
    that adapt to the speed of the prices' cycle, so they follow fast moves and stay flat in a range.

    MAMA is an exponential moving average whose smoothing factor follows the rate at which
    the phase of the price cycle changes, measured with the Hilbert transform: a phase that
    jumps (the start of a move) gives the fast limit and MAMA follows the price, a phase that
    advances slowly gives a smoothing factor towards the slow limit and MAMA flattens. FAMA
    applies half the smoothing factor to MAMA, so it follows MAMA and crossings of the two
    signal changes of direction.

    The formula is as follows:

    - Phase = arctan(Q1 / I1) in degrees, with I1 and Q1 the phasor components (HT_PHASOR)
    - Delta Phase = Phase(t-1) - Phase, at least 1
    - Alpha = Fast Limit / Delta Phase, limited to between Slow Limit and Fast Limit
    - MAMA = Alpha * Price + (1 - Alpha) * MAMA(t-1)
    - FAMA = 0.5 * Alpha * MAMA + (1 - 0.5 * Alpha) * FAMA(t-1)

    Reference: Ehlers, J.F. (2001). "MESA Adaptive Moving Averages." Technical Analysis of
    Stocks & Commodities, September 2001.

    Referenced against:
    - TA-Lib, MAMA: https://ta-lib.org/functions/mama.html

    Shares similar logic:
    - pandas-ta-classic, mama: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/overlap/mama.py#L248

    Notes:
        - Equals TA-Lib's MAMA: the recursion, MAMA and FAMA start from zero on bar 12 of
          the series, as in TA-Lib and Ehlers' article, and the first 32 bars are NaN, TA-Lib's
          lookback. FAMA is then still several percent below the price and settles to within
          0.1% after about 80 bars.
        - The phase is 0 on a bar whose in-phase component is 0 and before the first bar,
          the convention TA-Lib uses.

    Args:
        prices (pd.Series): Series of prices.
        fast_limit (float): The largest smoothing factor. Defaults to 0.5.
        slow_limit (float): The smallest smoothing factor. Defaults to 0.05.

    Returns:
        pd.DataFrame: The "MAMA" and "FAMA" in price units.
    """
    clean = prices.dropna()
    phase = _get_hilbert_transform(clean, PERIOD_START)["Phase"].iloc[PERIOD_START:]

    delta_phase = (phase.shift(1, fill_value=0.0) - phase).clip(lower=1.0)
    alpha = (fast_limit / delta_phase).clip(lower=slow_limit).to_numpy()
    price = clean.iloc[PERIOD_START:].to_numpy(dtype=float)

    mama, fama = np.zeros(len(price)), np.zeros(len(price))
    mama_before = fama_before = 0.0
    for t, (a, p) in enumerate(zip(alpha, price, strict=True)):
        mama[t] = mama_before = a * p + (1 - a) * mama_before
        fama[t] = fama_before = 0.5 * a * mama_before + (1 - 0.5 * a) * fama_before

    result = pd.DataFrame({"MAMA": mama, "FAMA": fama}, index=phase.index)
    result.iloc[: PERIOD_LOOKBACK - PERIOD_START] = np.nan

    return result.reindex(prices.index)


def get_swiss_army_knife_filter(
    prices: pd.Series,
    window: int = 20,
    method: str = "band_pass",
    delta: float = 0.1,
) -> pd.Series:
    """
    Calculate Ehlers' Swiss Army Knife filter: a family of filters that remove the trend (high-pass), the noise
    (low-pass) or both (band-pass) from the prices.

    The Swiss Army Knife indicator is one second-order recursive filter whose coefficients
    select the response: a one-pole or two-pole high-pass filter that removes the trend and
    keeps the cycles shorter than the window, a band-pass filter that keeps only the cycles
    around the window, or a two-pole Butterworth or Gaussian low-pass filter that smooths
    away the cycles shorter than the window.

    The formula is as follows:

    - Filter = c0 * (b0 * Price + b1 * Price(t-1) + b2 * Price(t-2)) + a1 * Filter(t-1)
      + a2 * Filter(t-2), with b0 = 1 and the other coefficients 0 unless set below
    - One-pole alpha = (cos(360 / Window) + sin(360 / Window) - 1) / cos(360 / Window)
    - Two-pole alpha = -Beta + √(Beta² + 2 * Beta), Beta = 2.415 * (1 - cos(360 / Window))
    - High-pass: c0 = 1 - alpha / 2, b1 = -1, a1 = 1 - alpha (one-pole alpha)
    - Two-pole high-pass: c0 = (1 - alpha / 2)², b1 = -2, b2 = 1, a1 = 2 * (1 - alpha),
      a2 = -(1 - alpha)² (two-pole alpha)
    - Butterworth: c0 = alpha² / 4, b1 = 2, b2 = 1, a1 = 2 * (1 - alpha), a2 = -(1 - alpha)²
    - Gaussian: c0 = alpha², a1 = 2 * (1 - alpha), a2 = -(1 - alpha)²
    - Band-pass: Beta = cos(360 / Window), Gamma = 1 / cos(720 * Delta / Window),
      alpha = Gamma - √(Gamma² - 1), c0 = (1 - alpha) / 2, b2 = -1, a1 = Beta * (1 + alpha),
      a2 = -alpha

    Reference: Ehlers, J.F. (2006). "Swiss Army Knife Indicator." Technical Analysis of
    Stocks & Commodities, 24(1), 28-31, 50-53.

    Referenced against:
    - TA-Lib, SWAK_HP: https://ta-lib.org/functions/swak_hp.html
    - TA-Lib, SWAK_2PHP: https://ta-lib.org/functions/swak_2php.html
    - TA-Lib, SWAK_BP: https://ta-lib.org/functions/swak_bp.html
    - TA-Lib, SWAK_BUTTER: https://ta-lib.org/functions/swak_butter.html
    - TA-Lib, SWAK_GAUSS: https://ta-lib.org/functions/swak_gauss.html

    Notes:
        - The filter starts in the steady state of a constant price equal to the first
          price: the low-pass filters on that price, the high-pass and band-pass filters on
          zero. Ehlers' listing starts every filter from zero, which leaves the low-pass
          filters far below the price for many bars. TA-Lib starts in the same steady state,
          and the two are equal from the first price.

    Args:
        prices (pd.Series): Series of prices.
        window (int): The cutoff period of the high-pass and low-pass filters and the
            centre period of the band-pass filter, at least 5. Defaults to 20.
        method (str): The filter. Options are: "high_pass" (one-pole high-pass filter),
            "two_pole_high_pass" (two-pole high-pass filter), "band_pass" (band-pass filter),
            "butterworth" (two-pole Butterworth low-pass filter), "gaussian" (Gaussian low-pass filter).
            Defaults to "band_pass", the default of Ehlers' listing.
        delta (float): The half bandwidth of the band-pass filter as a fraction of its
            centre frequency, between 0 and 0.5. Defaults to 0.1.

    Returns:
        pd.Series: The filtered prices: in price units for the low-pass filters and
        centred on zero for the high-pass and band-pass filters.
    """
    if window < 5:  # noqa: PLR2004
        raise ValueError("The window must be at least 5 periods.")
    if not 0 < delta <= 0.5:  # noqa: PLR2004
        raise ValueError("The delta must be above 0 and at most 0.5.")

    angle = 2 * math.pi / window
    c0, b0, b1, b2, a1, a2 = 1.0, 1.0, 0.0, 0.0, 0.0, 0.0

    if method == "high_pass":
        alpha = (math.cos(angle) + math.sin(angle) - 1) / math.cos(angle)
        c0, b1, a1 = 1 - alpha / 2, -1.0, 1 - alpha
    elif method in ["two_pole_high_pass", "butterworth", "gaussian"]:
        beta = 2.415 * (1 - math.cos(angle))
        alpha = -beta + math.sqrt(beta * beta + 2 * beta)
        a1, a2 = 2 * (1 - alpha), -(1 - alpha) * (1 - alpha)
        if method == "two_pole_high_pass":
            c0, b1, b2 = (1 - alpha / 2) * (1 - alpha / 2), -2.0, 1.0
        elif method == "butterworth":
            c0, b1, b2 = alpha * alpha / 4, 2.0, 1.0
        else:
            c0 = alpha * alpha
    elif method == "band_pass":
        beta = math.cos(angle)
        gamma = 1 / math.cos(2 * angle * delta)
        alpha = gamma - math.sqrt(gamma * gamma - 1)
        c0, b2, a1, a2 = (1 - alpha) / 2, -1.0, beta * (1 + alpha), -alpha
    else:
        raise InvalidArgumentError(
            "method must be one of 'high_pass', 'two_pole_high_pass', 'band_pass', 'butterworth' "
            f"or 'gaussian', not {method!r}. Please pass one of these as method."
        )

    clean = prices.dropna()
    price = clean.to_numpy(dtype=float)
    result = np.full(len(price), np.nan)

    # The steady state of a constant first price: the filter's gain at zero frequency times that price.
    gain = c0 * (b0 + b1 + b2) / (1 - a1 - a2)
    price_1 = price_2 = price[0]
    filter_1 = filter_2 = gain * price[0]

    for t, value in enumerate(price):
        result[t] = (
            c0 * (b0 * value + b1 * price_1 + b2 * price_2)
            + a1 * filter_1
            + a2 * filter_2
        )
        price_1, price_2 = value, price_1
        filter_1, filter_2 = result[t], filter_1

    return pd.Series(result, index=clean.index).reindex(prices.index)


def get_super_smoother_filter(
    prices: pd.Series, window: int = 10, poles: int = 2
) -> pd.Series:
    """
    Calculate Ehlers' SuperSmoother filter: a smoothing of the prices that removes the noise of the shortest cycles
    with less lag than a moving average.

    The SuperSmoother is a low-pass filter built from an analog Butterworth filter: it removes
    the cycles shorter than the window (the aliasing noise) with less lag than a moving
    average of the same length. The two-pole filter of Ehlers (2013) also averages the last
    two prices, which removes the two-bar cycle completely; the three-pole filter of Ehlers
    (2004) is smoother at the cost of more lag.

    The formula is as follows:

    - Two poles: a = exp(-√2 * π / Window), b = 2 * a * cos(√2 * 180 / Window),
      c2 = b, c3 = -a², c1 = 1 - c2 - c3
    - Filter = c1 * (Price + Price(t-1)) / 2 + c2 * Filter(t-1) + c3 * Filter(t-2)
    - Three poles: a = exp(-π / Window), b = 2 * a * cos(√3 * 180 / Window), c = a²,
      c2 = b + c, c3 = -(c + b * c), c4 = c², c1 = 1 - c2 - c3 - c4
    - Filter = c1 * Price + c2 * Filter(t-1) + c3 * Filter(t-2) + c4 * Filter(t-3)

    Reference: Ehlers, J.F. (2013). "Cycle Analytics for Traders: Advanced Technical Trading
    Concepts." John Wiley & Sons; Ehlers, J.F. (2014). "Predictive And Successful
    Indicators." Technical Analysis of Stocks & Commodities, 32(1), 16-25; Ehlers, J.F.
    (2004). "Cybernetic Analysis for Stocks and Futures." John Wiley & Sons.

    Referenced against:
    - pandas-ta-classic, ssf: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/overlap/ssf.py#L28

    Notes:
        - The filter equals the price on its first two (three) prices and runs from there,
          as pandas-ta-classic starts it. Ehlers' listing starts it from zero.
        - pandas-ta-classic's two-pole ssf filters the price itself rather than the average of
          the last two prices; on that average the two are equal.
        - √2, √3 and π are exact, where Ehlers' listings write 1.414 and 3.14159.

    Args:
        prices (pd.Series): Series of prices.
        window (int): The critical period: cycles shorter than it are removed. Defaults to 10.
        poles (int): The number of poles. Options are: 2 (the two-pole SuperSmoother), 3 (the three-pole
            SuperSmoother). Defaults to 2.

    Returns:
        pd.Series: The smoothed prices in price units.
    """
    check_option("poles", poles, [2, 3])

    clean = prices.dropna()
    price = clean.to_numpy(dtype=float)
    result = price.copy()

    if poles == 2:  # noqa: PLR2004
        a = math.exp(-math.sqrt(2) * math.pi / window)
        b = 2 * a * math.cos(math.sqrt(2) * math.pi / window)
        c2, c3 = b, -a * a
        c1 = 1 - c2 - c3
        for t in range(2, len(price)):
            result[t] = (
                c1 * (price[t] + price[t - 1]) / 2
                + c2 * result[t - 1]
                + c3 * result[t - 2]
            )
    else:
        a = math.exp(-math.pi / window)
        b = 2 * a * math.cos(math.sqrt(3) * math.pi / window)
        c = a * a
        c2, c3, c4 = b + c, -(c + b * c), c * c
        c1 = 1 - c2 - c3 - c4
        for t in range(3, len(price)):
            result[t] = (
                c1 * price[t]
                + c2 * result[t - 1]
                + c3 * result[t - 2]
                + c4 * result[t - 3]
            )

    return pd.Series(result, index=clean.index).reindex(prices.index)


def get_even_better_sinewave(
    prices: pd.Series, window: int = 40, smooth_window: int = 10
) -> pd.Series:
    """
    Calculate Ehlers' Even Better Sinewave: an oscillator between -1 and 1 that shows whether the prices move in a
    cycle or a trend and where they are in the cycle.

    The Even Better Sinewave removes the trend with a one-pole high-pass filter, smooths the
    remaining cycles with a SuperSmoother and divides their average over three bars by the
    square root of their power over the same bars. The result swings between -1 and 1 like a
    sine wave in a cycling market and stays near 1 (or -1) in an up (or down) trend.

    The formula is as follows:

    - alpha = (1 - sin(360 / Window)) / cos(360 / Window)
    - HP = 0.5 * (1 + alpha) * (Price - Price(t-1)) + alpha * HP(t-1)
    - a = exp(-√2 * π / Smoothing Window), b = 2 * a * cos(√2 * 180 / Smoothing Window),
      c2 = b, c3 = -a², c1 = 1 - c2 - c3
    - Filter = c1 * (HP + HP(t-1)) / 2 + c2 * Filter(t-1) + c3 * Filter(t-2)
    - Wave = (Filter + Filter(t-1) + Filter(t-2)) / 3
    - Power = (Filter² + Filter(t-1)² + Filter(t-2)²) / 3
    - Even Better Sinewave = Wave / √Power

    Reference: Ehlers, J.F. (2013). "Cycle Analytics for Traders: Advanced Technical Trading
    Concepts." John Wiley & Sons.

    Referenced against:
    - pandas-ta-classic, ebsw: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/cycles/ebsw.py#L63

    Notes:
        - The high-pass filter and the SuperSmoother start from zero on the first price, as in
          Ehlers' listing, and the first `window` periods are NaN while they settle.
          pandas-ta-classic starts them on the 12th price; on prices that do not move before
          it the two are equal, and otherwise they converge as the start fades.
        - A bar without power (three filter values of zero) is NaN, where pandas-ta-classic
          gives 0.

    Args:
        prices (pd.Series): Series of prices.
        window (int): The cutoff period of the high-pass filter, the longest cycle kept.
            Defaults to 40.
        smooth_window (int): The critical period of the SuperSmoother. Defaults to 10.

    Returns:
        pd.Series: The Even Better Sinewave, between -1 and 1.
    """
    clean = prices.dropna()
    price = clean.to_numpy(dtype=float)
    bars = len(price)

    alpha = (1 - math.sin(2 * math.pi / window)) / math.cos(2 * math.pi / window)
    a = math.exp(-math.sqrt(2) * math.pi / smooth_window)
    b = 2 * a * math.cos(math.sqrt(2) * math.pi / smooth_window)
    c2, c3 = b, -a * a
    c1 = 1 - c2 - c3

    high_pass, smoothed = np.zeros(bars), np.zeros(bars)
    for t in range(1, bars):
        high_pass[t] = (
            0.5 * (1 + alpha) * (price[t] - price[t - 1]) + alpha * high_pass[t - 1]
        )
        smoothed[t] = (
            c1 * (high_pass[t] + high_pass[t - 1]) / 2
            + c2 * smoothed[t - 1]
            + c3 * (smoothed[t - 2] if t > 1 else 0.0)
        )

    smoothed_series = pd.Series(smoothed, index=clean.index)
    wave = smoothed_series.rolling(3, min_periods=1).sum() / 3
    power = (smoothed_series**2).rolling(3, min_periods=1).sum() / 3

    result = wave / np.sqrt(power.where(power > 0))
    result.iloc[:window] = np.nan

    return result.reindex(prices.index)


def get_detrended_synthetic_price(prices: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate Ehlers' Detrended Synthetic Price: the difference between a fast and a slow average tuned to the
    prices' cycle, which shows the cycle without the trend.

    The Detrended Synthetic Price is the difference between an exponential moving average
    over a quarter of the dominant cycle and one over half of it. The difference removes the
    trend and leaves a wave that is in phase with the dominant cycle of the price.

    The formula is as follows:

    - Detrended Synthetic Price = EMA(Price, Window / 4) - EMA(Price, Window / 2)

    Reference: Ehlers, J.F. (1992). "MESA and Trading Market Cycles." John Wiley & Sons.

    Referenced against:
    - pandas-ta-classic, dsp: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/cycles/dsp.py#L12

    Notes:
        - The averages span a quarter and half of the window rounded down, and are seeded
          with their simple average, as every exponential moving average of the Toolkit.
        - pandas-ta-classic's dsp subtracts one exponential moving average from the price,
          Price - EMA(Price, 14), a simpler detrender that also keeps the shorter cycles.

    Args:
        prices (pd.Series): Series of prices.
        window (int): The period of the dominant cycle, at least 4. Defaults to 20.

    Returns:
        pd.Series: The Detrended Synthetic Price in price units.
    """
    if window < 4:  # noqa: PLR2004
        raise ValueError("The window must be at least 4 periods.")

    quarter_cycle = overlap_model.get_exponential_moving_average(prices, window // 4)
    half_cycle = overlap_model.get_exponential_moving_average(prices, window // 2)

    return quarter_cycle - half_cycle


def get_mesa_sine_wave(prices: pd.Series, window: int = 5) -> pd.DataFrame:
    """
    Calculate the MESA Sine Wave: the sine of the phase of the prices' cycle over a fixed window and its lead, whose
    crossings mark the turning points of the cycle.

    The MESA Sine Wave is the sine of the phase of the price within a cycle of a fixed
    window, measured with a discrete Fourier transform over that window, and the lead sine
    the sine of the phase advanced by 45 degrees. In a cycling market the two lines cross a
    little before the turning points; in a trend they run side by side. It is Ehlers'
    sinewave indicator over a fixed window, where the Hilbert Transform SineWave measures the
    window as the dominant cycle.

    The formula is as follows:

    - Real = Σ sin(360 * i / Window) * Price(t-i), Imaginary = Σ cos(360 * i / Window) *
      Price(t-i) for i from 0 to Window - 1
    - Phase = arctan(Real / Imaginary) + 90 degrees, plus 180 when Imaginary is negative
    - Phase = 90 * sign(Real) + 90 degrees when |Imaginary| is at most 0.001
    - Sine = sin(Phase), Lead Sine = sin(Phase + 45 degrees)

    Reference: Ehlers, J.F. (2001). "Rocket Science for Traders: Digital Signal Processing
    Applications." John Wiley & Sons; Ehlers, J.F. (1992). "MESA and Trading Market Cycles."
    John Wiley & Sons.

    Referenced against:
    - Tulip Indicators, msw: https://tulipindicators.org/msw
    - pandas-ta-classic, msw: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/cycles/msw.py#L14

    Notes:
        - The first value is on the last price of the first full window; Tulip Indicators and
          pandas-ta-classic start one price later.
        - Without an imaginary part the phase is 90 degrees times the sign of the real part,
          as in Ehlers' sinewave indicator, where Tulip Indicators takes 180 degrees.
        - Tulip Indicators takes 3.1415926 for π, which moves its values by up to 3e-4.

    Args:
        prices (pd.Series): Series of prices.
        window (int): The number of periods of the cycle. Defaults to 5.

    Returns:
        pd.DataFrame: The "Sine" and "Lead Sine", between -1 and 1.
    """
    clean = prices.dropna()
    price = clean.to_numpy(dtype=float)
    sine = np.full(len(price), np.nan)
    lead_sine = np.full(len(price), np.nan)

    if len(price) >= window:
        # Each row holds a window with the latest price first, so column i is Price(t-i).
        windows = sliding_window_view(price, window)[:, ::-1]
        angles = 2 * np.pi * np.arange(window) / window
        real = windows @ np.sin(angles)
        imaginary = windows @ np.cos(angles)

        with np.errstate(divide="ignore", invalid="ignore"):
            phase = np.where(
                np.abs(imaginary) > 0.001,  # noqa: PLR2004
                np.degrees(np.arctan(real / imaginary)),
                90 * np.sign(real),
            )
        phase = phase + 90 + np.where(imaginary < 0, 180, 0)

        sine[window - 1 :] = np.sin(np.radians(phase))
        lead_sine[window - 1 :] = np.sin(np.radians(phase + 45))

    result = pd.DataFrame({"Sine": sine, "Lead Sine": lead_sine}, index=clean.index)

    return result.reindex(prices.index)


def get_center_of_gravity_oscillator(
    prices: pd.Series, window: int = 10
) -> pd.DataFrame:
    """
    Calculate Ehlers' Center of Gravity (CG) Oscillator: the weighted position of the prices within a window, which
    marks the turning points of the cycle with little lag.

    The Center of Gravity is the balance point of the prices in the window, with the latest
    price at position 1. When prices rise the weight shifts to the latest prices and the
    balance point moves towards them, so its negative rises with the price, smoothed and with
    almost no lag. The trigger is the previous value, and a crossing of the two marks a turn.

    The formula is as follows:

    - CG = -Σ (i + 1) * Price(t-i) / Σ Price(t-i), for i from 0 to Window - 1
    - Trigger = CG(t-1)

    Reference: Ehlers, J.F. (2002). "The Center Of Gravity Oscillator." Technical Analysis of
    Stocks & Commodities, 20(5), 20-24.

    Referenced against:
    - TA-Lib, CG: https://ta-lib.org/functions/cg.html
    - pandas-ta-classic, cg: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/cg.py#L12

    Notes:
        - The oscillator is not centred on zero: for flat prices it is -(Window + 1) / 2.
        - A window whose prices sum to zero is NaN, where Ehlers' listing keeps the value
          before.

    Args:
        prices (pd.Series): Series of prices.
        window (int): The number of periods, ideally half the dominant cycle. Defaults to 10.

    Returns:
        pd.DataFrame: The "Center of Gravity" and its "Trigger".
    """
    clean = prices.dropna()
    price = clean.to_numpy(dtype=float)
    center_of_gravity = np.full(len(price), np.nan)

    if len(price) >= window:
        # Each row holds a window with the latest price first, at position 1.
        windows = sliding_window_view(price, window)[:, ::-1]
        numerator = windows @ np.arange(1, window + 1)
        denominator = windows.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            center_of_gravity[window - 1 :] = np.where(
                denominator != 0, -numerator / denominator, np.nan
            )

    result = pd.DataFrame({"Center of Gravity": center_of_gravity}, index=clean.index)
    result["Trigger"] = result["Center of Gravity"].shift(1)

    return result.reindex(prices.index)


def get_laguerre_relative_strength_index(
    prices: pd.Series, gamma: float = 0.5
) -> pd.Series:
    """
    Calculate Ehlers' Laguerre Relative Strength Index (Laguerre RSI): a Relative Strength Index of
    Laguerre-filtered prices, which shows overbought and oversold levels with little lag.

    The Laguerre RSI is an RSI over four samples of a Laguerre filter instead of four prices.
    The first element of the filter is an exponential moving average and each next one an
    all-pass element that delays the low frequencies more than the high ones, which warps
    time: four elements see as far back as a much longer window, so the RSI is smooth yet
    reacts quickly. It moves between 0 and 1, and Ehlers buys when it crosses back above 0.2
    and sells when it crosses back below 0.8.

    The formula is as follows:

    - L0 = (1 - Gamma) * Price + Gamma * L0(t-1)
    - L1 = -Gamma * L0 + L0(t-1) + Gamma * L1(t-1)
    - L2 = -Gamma * L1 + L1(t-1) + Gamma * L2(t-1)
    - L3 = -Gamma * L2 + L2(t-1) + Gamma * L3(t-1)
    - CU = the sum of L0 - L1, L1 - L2 and L2 - L3 where positive
    - CD = the sum of L1 - L0, L2 - L1 and L3 - L2 where positive
    - Laguerre RSI = CU / (CU + CD)

    Reference: Ehlers, J.F. (2004). "Cybernetic Analysis for Stocks and Futures." John Wiley
    & Sons, chapter 14, "Time Warp - Without Space Travel."

    Referenced against:
    - pandas-ta-classic, lrsi: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/lrsi.py#L29

    Notes:
        - A decimal between 0 and 1, as in Ehlers' listing; pandas-ta-classic multiplies it
          by 100.
        - The filter elements start on the first price, as pandas-ta-classic starts them, so
          the first value is NaN. Ehlers' listing starts them from zero.
        - When the elements are equal (CU + CD is zero) the value before is kept, as in
          Ehlers' listing, where pandas-ta-classic gives 0.

    Args:
        prices (pd.Series): Series of prices.
        gamma (float): The damping factor of the Laguerre filter, between 0 and 1: a higher
            value smooths more and lags more. Defaults to 0.5.

    Returns:
        pd.Series: The Laguerre RSI, between 0 and 1.
    """
    if not 0 <= gamma < 1:
        raise ValueError("Gamma must be at least 0 and below 1.")

    clean = prices.dropna()
    price = clean.to_numpy(dtype=float)
    laguerre = np.zeros((len(price), 4))
    if len(price):
        laguerre[0] = price[0]

    for t in range(1, len(price)):
        laguerre[t, 0] = (1 - gamma) * price[t] + gamma * laguerre[t - 1, 0]
        for element in range(1, 4):
            laguerre[t, element] = (
                -gamma * laguerre[t, element - 1]
                + laguerre[t - 1, element - 1]
                + gamma * laguerre[t - 1, element]
            )

    differences = laguerre[:, :3] - laguerre[:, 1:]
    closes_up = np.clip(differences, 0, None).sum(axis=1)
    closes_down = np.clip(-differences, 0, None).sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        relative_strength = np.where(
            closes_up + closes_down != 0,
            closes_up / (closes_up + closes_down),
            np.nan,
        )
    result = pd.Series(relative_strength, index=clean.index).ffill()

    return result.reindex(prices.index)
