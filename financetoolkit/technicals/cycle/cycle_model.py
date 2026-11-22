"""Cycle Model: indicators of the cycles in the price"""

__docformat__ = "google"

import math

import numpy as np
import pandas as pd

# The coefficients of Ehlers' discrete Hilbert transform.
HILBERT_A = 0.0962
HILBERT_B = 0.5769

# The bar on which the recursion starts and the bars without a value (the lookback) of
# TA-Lib's functions: those that only need the period start after 12 bars (as in Ehlers'
# EasyLanguage listings) and show a value from bar 32, those that also need the phase over a
# full cycle of up to 50 bars start after 37 and show a value from bar 63.
PERIOD_START, PERIOD_LOOKBACK = 12, 32
PHASE_START, PHASE_LOOKBACK = 37, 63


def _get_hilbert_transform(prices: pd.Series, start: int) -> pd.DataFrame:
    """
    The Hilbert transform and homodyne discriminator of Ehlers (2001) that the HT
    indicators and MAMA share, calculated bar by bar as in TA-Lib (Mario Fortier).

    The prices are smoothed with a 4-bar weighted moving average and detrended with a
    discrete Hilbert transform whose gain follows the period of the bar before. The
    detrended price lagged 3 bars is the in-phase component (I1), its Hilbert transform the
    quadrature (Q1). Advancing both by 90 degrees (jI, jQ) gives the phasors I2 and Q2,
    whose product with the phasor of the bar before (the homodyne discriminator) gives the
    phase change per bar and so the period, limited to 6-50 bars and to between 0.67 and
    1.5 times the period before. From the smoothed period follow the dominant cycle phase,
    a discrete Fourier transform of the smoothed price over one cycle, and the
    instantaneous trendline, the price averaged over one cycle.

    The filters start from zero on bar `start`, as in TA-Lib, so the values only settle
    after a number of bars; the callers leave TA-Lib's lookback empty.

    Args:
        prices (pd.Series): The prices without missing values.
        start (int): The bar on which the recursion starts, 12 or 37 as in TA-Lib.

    Returns:
        pd.DataFrame: Per bar the smoothed price, in-phase and quadrature components, the
        phase of the phasor (degrees), the smoothed period, the dominant cycle phase
        (degrees) and the instantaneous trendline.
    """
    values = prices.to_numpy(dtype=float)
    length = len(values)
    a, b = HILBERT_A, HILBERT_B
    radians_to_degrees = 180 / math.pi

    columns = {
        name: np.full(length, np.nan)
        for name in (
            "Smoothed Price",
            "In Phase",
            "Quadrature",
            "Phase",
            "Smoothed Period",
            "Dominant Cycle Phase",
            "Trendline",
        )
    }

    # The inputs of the Hilbert transforms, zero before the start as TA-Lib's filters are.
    smooth = [0.0] * length
    detrender = [0.0] * length
    in_phase = [0.0] * length
    quadrature = [0.0] * length

    def hilbert(series: list[float], today: int, gain: float) -> float:
        # The discrete Hilbert transform, with a gain that follows the period of the bar before.
        return (
            -a * series[today - 6]
            + a * series[today]
            - b * series[today - 4]
            + b * series[today - 2]
        ) * gain

    period = smooth_period = dc_phase = 0.0
    i2_previous = q2_previous = real = imaginary = 0.0
    trend_averages = [0.0, 0.0, 0.0]

    for today in range(start, length):
        adjusted_period = 0.075 * period + 0.54

        smooth[today] = (
            4 * values[today]
            + 3 * values[today - 1]
            + 2 * values[today - 2]
            + values[today - 3]
        ) / 10

        detrender[today] = hilbert(smooth, today, adjusted_period)
        quadrature[today] = hilbert(detrender, today, adjusted_period)
        in_phase[today] = detrender[today - 3]
        j_in_phase = hilbert(in_phase, today, adjusted_period)
        j_quadrature = hilbert(quadrature, today, adjusted_period)

        # The phasor advanced by 90 degrees, smoothed against noise.
        q2 = 0.2 * (quadrature[today] + j_in_phase) + 0.8 * q2_previous
        i2 = 0.2 * (in_phase[today] - j_quadrature) + 0.8 * i2_previous

        # The homodyne discriminator: the phase change per bar is the angle of the product
        # of the phasor with the conjugate of the phasor of the bar before.
        real = 0.8 * real + 0.2 * (i2 * i2_previous + q2 * q2_previous)
        imaginary = 0.8 * imaginary + 0.2 * (i2 * q2_previous - q2 * i2_previous)
        i2_previous, q2_previous = i2, q2

        period_before = period
        if imaginary != 0 and real != 0:
            period = 360 / (math.atan(imaginary / real) * radians_to_degrees)
        period = min(period, 1.5 * period_before)
        period = max(period, 0.67 * period_before)
        period = min(max(period, 6), 50)
        period = 0.2 * period + 0.8 * period_before
        smooth_period = 0.33 * period + 0.67 * smooth_period

        # The dominant cycle phase: the discrete Fourier transform of the smoothed prices
        # over one cycle, shifted by 90 degrees and by the lag of the smoothing.
        cycle = int(smooth_period + 0.5)
        real_part = imaginary_part = 0.0
        for bar in range(cycle):
            angle = 2 * math.pi * bar / cycle
            real_part += math.sin(angle) * smooth[today - bar]
            imaginary_part += math.cos(angle) * smooth[today - bar]
        if abs(imaginary_part) > 0:
            dc_phase = math.atan(real_part / imaginary_part) * radians_to_degrees
        elif real_part != 0:
            dc_phase += math.copysign(90, real_part)
        dc_phase += 90 + 360 / smooth_period
        if imaginary_part < 0:
            dc_phase += 180
        if dc_phase > 315:  # noqa: PLR2004
            dc_phase -= 360

        # The instantaneous trendline: the price averaged over one cycle and smoothed with
        # a 4-bar weighted moving average.
        average = (
            values[max(today - cycle + 1, 0) : today + 1].sum() / cycle
            if cycle
            else 0.0
        )
        columns["Trendline"][today] = (
            4 * average
            + 3 * trend_averages[0]
            + 2 * trend_averages[1]
            + trend_averages[2]
        ) / 10
        trend_averages = [average, *trend_averages[:2]]

        columns["Phase"][today] = (
            math.atan(quadrature[today] / in_phase[today]) * radians_to_degrees
            if in_phase[today] != 0
            else 0.0
        )
        columns["Smoothed Period"][today] = smooth_period
        columns["Dominant Cycle Phase"][today] = dc_phase

    columns["Smoothed Price"][start:] = smooth[start:]
    columns["In Phase"][start:] = in_phase[start:]
    columns["Quadrature"][start:] = quadrature[start:]

    return pd.DataFrame(columns, index=prices.index)


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
          bar 37 of the series and the first 63 bars are NaN, where TA-Lib shows 0.

    Args:
        prices (pd.Series): Series of prices.

    Returns:
        pd.Series: 1 in a trend and 0 in a cycle.
    """
    clean = prices.dropna()
    hilbert = _get_hilbert_transform(clean, PHASE_START)

    phase = hilbert["Dominant Cycle Phase"].to_numpy()
    smooth_period = hilbert["Smoothed Period"].to_numpy()
    smooth = hilbert["Smoothed Price"].to_numpy()
    trendline = hilbert["Trendline"].to_numpy()
    sine = np.sin(np.deg2rad(phase))
    lead_sine = np.sin(np.deg2rad(phase + 45))

    trend_mode = np.full(len(clean), np.nan)
    days_in_trend = 0
    previous_phase = previous_sine = previous_lead_sine = 0.0

    for today in range(PHASE_START, len(clean)):
        trend = 1
        if (sine[today] > lead_sine[today] and previous_sine <= previous_lead_sine) or (
            sine[today] < lead_sine[today] and previous_sine >= previous_lead_sine
        ):
            days_in_trend = 0
            trend = 0
        days_in_trend += 1
        if days_in_trend < 0.5 * smooth_period[today]:
            trend = 0

        phase_change = phase[today] - previous_phase
        cycle_rate = 360 / smooth_period[today]
        if 0.67 * cycle_rate < phase_change < 1.5 * cycle_rate:
            trend = 0

        if (
            trendline[today] != 0
            and abs((smooth[today] - trendline[today]) / trendline[today])
            >= 0.015  # noqa: PLR2004
        ):
            trend = 1

        trend_mode[today] = trend
        previous_phase = phase[today]
        previous_sine, previous_lead_sine = sine[today], lead_sine[today]

    trend_mode[:PHASE_LOOKBACK] = np.nan

    return pd.Series(trend_mode, index=clean.index).reindex(prices.index)


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

    Args:
        prices (pd.Series): Series of prices.
        fast_limit (float): The largest smoothing factor. Defaults to 0.5.
        slow_limit (float): The smallest smoothing factor. Defaults to 0.05.

    Returns:
        pd.DataFrame: The "MAMA" and "FAMA" in price units.
    """
    clean = prices.dropna()
    phase = _get_hilbert_transform(clean, PERIOD_START)["Phase"].to_numpy()
    values = clean.to_numpy(dtype=float)

    mama = np.full(len(clean), np.nan)
    fama = np.full(len(clean), np.nan)
    mama_today = fama_today = previous_phase = 0.0

    for today in range(PERIOD_START, len(clean)):
        delta_phase = max(previous_phase - phase[today], 1.0)
        previous_phase = phase[today]
        alpha = max(fast_limit / delta_phase, slow_limit)

        mama_today = alpha * values[today] + (1 - alpha) * mama_today
        fama_today = 0.5 * alpha * mama_today + (1 - 0.5 * alpha) * fama_today
        mama[today], fama[today] = mama_today, fama_today

    result = pd.DataFrame({"MAMA": mama, "FAMA": fama}, index=clean.index)

    result.iloc[:PERIOD_LOOKBACK] = np.nan

    return result.reindex(prices.index)
