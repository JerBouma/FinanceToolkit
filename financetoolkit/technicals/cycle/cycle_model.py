"""Cycle Model: indicators of the cycles in the price"""

__docformat__ = "google"

import math

import numpy as np
import pandas as pd

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
    Calculate the Hilbert Transform Dominant Cycle Period (HT_DCPERIOD) of a price series.

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

    Referenced against TA-Lib's HT_DCPERIOD
    (https://ta-lib.org/functions/ht_dcperiod.html).

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
    Calculate the Hilbert Transform Dominant Cycle Phase (HT_DCPHASE) of a price series.

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

    Referenced against TA-Lib's HT_DCPHASE
    (https://ta-lib.org/functions/ht_dcphase.html).

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
    Calculate the Hilbert Transform Phasor Components (HT_PHASOR) of a price series.

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

    Referenced against TA-Lib's HT_PHASOR
    (https://ta-lib.org/functions/ht_phasor.html).

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
    Calculate the Hilbert Transform SineWave (HT_SINE) of a price series.

    The sine wave is the sine of the dominant cycle phase, and the lead sine the sine of the
    phase advanced by 45 degrees. In a cycling market the two lines cross a little before the
    turning points of the cycle; in a trend the phase barely moves and the lines run
    side by side.

    The formula is as follows:

    - Sine = sin(Dominant Cycle Phase)
    - Lead Sine = sin(Dominant Cycle Phase + 45 degrees)

    Reference: Ehlers, J.F. (2001). "Rocket Science for Traders: Digital Signal Processing
    Applications." John Wiley & Sons.

    Referenced against TA-Lib's HT_SINE
    (https://ta-lib.org/functions/ht_sine.html).

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
    Calculate the Hilbert Transform Instantaneous Trendline (HT_TRENDLINE) of a price series.

    The instantaneous trendline is the price averaged over its dominant cycle period: the
    average over exactly one cycle removes the cycle and leaves the trend. The averages are
    smoothed with a 4-bar weighted moving average.

    The formula is as follows:

    - N = the dominant cycle period (HT_DCPERIOD) rounded to the nearest bar
    - Average = the simple average of the latest N prices
    - Trendline = (4 * Average + 3 * Average(t-1) + 2 * Average(t-2) + Average(t-3)) / 10

    Reference: Ehlers, J.F. (2001). "Rocket Science for Traders: Digital Signal Processing
    Applications." John Wiley & Sons.

    Referenced against TA-Lib's HT_TRENDLINE
    (https://ta-lib.org/functions/ht_trendline.html).

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
    Calculate the Hilbert Transform Trend vs Cycle Mode (HT_TRENDMODE) of a price series.

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

    Referenced against TA-Lib's HT_TRENDMODE
    (https://ta-lib.org/functions/ht_trendmode.html).

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
    Calculate the MESA Adaptive Moving Average (MAMA) and the Following Adaptive Moving
    Average (FAMA) of a price series.

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

    Referenced against TA-Lib's MAMA (https://ta-lib.org/functions/mama.html).

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
