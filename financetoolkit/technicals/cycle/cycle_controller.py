"""Cycle Module"""

__docformat__ = "google"

from typing import Literal

import pandas as pd

from financetoolkit.helpers import handle_portfolio
from financetoolkit.technicals.cycle import cycle_model
from financetoolkit.technicals.helpers import (
    calculate_per_ticker,
    report_missing_values,
)
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import (
    InvalidArgumentError,
    check_option,
    check_period,
    handle_errors,
)
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import (
    apply_rounding,
    calculate_growth,
    calculate_standardization,
    finalize_dataset,
)

# The examples' output tables are wider than the line length.
# ruff: noqa: E501


@report_missing_values
@filter_shown_columns
@track_sources
class CycleTechnicals:
    """
    Cycle indicators of John Ehlers: those built on his Hilbert transform (the dominant cycle period and phase, the phasor components, the sine wave and the MESA Sine Wave, the trend vs cycle mode and the MESA adaptive moving average, MAMA and FAMA) and his filters and cycle oscillators (the Swiss Army Knife filters, the Even Better Sinewave, the Detrended Synthetic Price and the Center of Gravity Oscillator). His instantaneous trendline and SuperSmoother filter are methods of `get_moving_average` and his Laguerre RSI a method of `get_relative_strength_index`, which `collect_cycle_indicators` includes.

    The indicators are John Ehlers' (his books MESA and Trading Market Cycles, 1992, Rocket Science for Traders,
    2001, Cybernetic Analysis for Stocks and Futures, 2004, and Cycle Analytics for Traders, 2013, and his articles
    in Technical Analysis of Stocks & Commodities). The Hilbert transform indicators are referenced against TA-Lib's
    HT functions and MAMA (Mario Fortier), whose start and lookback they share, and the filters and oscillators
    against TA-Lib, pandas-ta-classic and Tulip Indicators.
    """

    @handle_errors
    def collect_cycle_indicators(
        self,
        period: str = "daily",
        window: int = 14,
        close_column: str = "Adj Close",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Collect the cycle indicators of every ticker in one table: the length, phase and turning points of the
        cycles in the prices.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            window (int, optional): The window size for calculating indicators.
                Defaults to 14.
            close_column (str, optional): The name of the column containing the close prices.
                Defaults to "Adj Close".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the indicator values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series or pd.DataFrame: Cycle indicators calculated based on the specified parameters.

        Notes:
        - The method calculates the cycle indicators for each asset in the Toolkit instance.
        - The Hilbert transform indicators adapt to the dominant cycle of the price and the filters and
          oscillators use their own default windows, so `window` is left unused.
        - When growth is requested, the method calculates the growth of the indicator values
          over the given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.collect_cycle_indicators().xs("AAPL", level=1, axis="columns")
        ```

        Which returns:

        | Date       |   Hilbert Dominant Cycle Period |   Hilbert Dominant Cycle Phase |   Hilbert In Phase |   Hilbert Quadrature |   Hilbert Sine |   Hilbert Lead Sine |   Hilbert Trend Mode |   Hilbert Instantaneous Trendline |   MESA Adaptive Moving Average (MAMA) |   Following Adaptive Moving Average (FAMA) |   Swiss Army Knife Filter |   SuperSmoother Filter |   Even Better Sinewave |   Detrended Synthetic Price |   MESA Sine Wave Sine |   MESA Sine Wave Lead Sine |   Center of Gravity Oscillator |   Center of Gravity Trigger |   Laguerre RSI |
        |:-----------|--------------------------------:|-------------------------------:|-------------------:|---------------------:|---------------:|--------------------:|---------------------:|----------------------------------:|--------------------------------------:|-------------------------------------------:|--------------------------:|-----------------------:|-----------------------:|----------------------------:|----------------------:|---------------------------:|-------------------------------:|----------------------------:|---------------:|
        | 2025-12-17 |                         18.3901 |                       282.339  |            -2.1962 |               4.3125 |        -0.9769 |             -0.5397 |                    0 |                           276.437 |                               276.673 |                                    273.841 |                   -2.8349 |                274.241 |                -0.9054 |                     -1.3767 |               -0.7739 |                    -0.0994 |                        -5.5219 |                     -5.5285 |         0      |
        | 2025-12-18 |                         18.2309 |                       303.757  |            -0.8122 |               1.9647 |        -0.8314 |             -0.195  |                    1 |                           276.799 |                               276.073 |                                    273.964 |                   -2.5273 |                272.939 |                -0.9568 |                     -1.5143 |               -0.6032 |                     0.1375 |                        -5.5252 |                     -5.5219 |         0      |
        | 2025-12-19 |                         18.064  |                       -35.1903 |            -0.6121 |              -1.6204 |        -0.5763 |              0.1704 |                    1 |                           276.957 |                               274.397 |                                    274.072 |                   -2.0487 |                272.161 |                -0.9881 |                     -1.2741 |               -0.2922 |                     0.4696 |                        -5.5254 |                     -5.5252 |         0      |
        | 2025-12-22 |                         17.9853 |                       -17.9254 |            -2.3867 |              -3.8683 |        -0.3078 |              0.4551 |                    1 |                           276.873 |                               274.178 |                                    274.075 |                   -1.347  |                271.642 |                -0.9995 |                     -1.4735 |                0.1788 |                     0.8222 |                        -5.5263 |                     -5.5254 |         0      |
        | 2025-12-23 |                         18.0561 |                        -3.6741 |            -3.886  |              -1.9043 |        -0.0641 |              0.6603 |                    1 |                           276.637 |                               274.04  |                                    274.074 |                   -0.5972 |                271.196 |                -0.9999 |                     -1.2831 |                0.8774 |                     0.9597 |                        -5.5259 |                     -5.5263 |         0      |
        | 2025-12-24 |                         18.3028 |                         9.1698 |            -4.0015 |              -0.3152 |         0.1594 |              0.8108 |                    1 |                           276.361 |                               273.981 |                                    274.072 |                    0.2127 |                271.251 |                -0.9962 |                     -0.8825 |                0.6824 |                    -0.0344 |                        -5.519  |                     -5.5259 |         0.1145 |
        | 2025-12-26 |                         18.6492 |                        13.3466 |            -4.0726 |               2.8793 |         0.2308 |              0.8512 |                    0 |                           276.11  |                               273.904 |                                    274.067 |                    1.0468 |                271.629 |                -0.9816 |                     -0.6725 |                0.9729 |                     0.5244 |                        -5.5128 |                     -5.519  |         0.1222 |
        | 2025-12-29 |                         18.9902 |                        23.5747 |            -1.6754 |               4.2075 |         0.3999 |              0.9309 |                    0 |                           275.772 |                               273.85  |                                    274.062 |                    1.7167 |                272.038 |                -0.9626 |                     -0.4628 |                0.3714 |                    -0.3939 |                        -5.5043 |                     -5.5128 |         0.203  |
        | 2025-12-30 |                         19.3292 |                        29.9641 |            -1.2128 |               1.7906 |         0.4995 |              0.9658 |                    1 |                           275.284 |                               272.991 |                                    273.794 |                    2.1545 |                272.316 |                -0.9522 |                     -0.423  |               -0.6187 |                    -0.993  |                        -5.4973 |                     -5.5043 |         0.2628 |
        | 2025-12-31 |                         19.6066 |                        33.1911 |            -0.9892 |               3.5115 |         0.5474 |              0.9788 |                    1 |                           274.885 |                               272.887 |                                    273.772 |                    2.3417 |                272.229 |                -0.9772 |                     -0.5599 |               -0.7431 |                    -0.0523 |                        -5.4957 |                     -5.4973 |         0.3273 |
        """
        self._check_arguments(period, close_column, window=window)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday" and self._historical_data[period].empty:
            raise InvalidArgumentError(
                "Please define the 'intraday_period' parameter when initializing the Toolkit."
            )

        cycle_indicators: dict = {}

        cycle_indicators["Hilbert Dominant Cycle Period"] = (
            self.get_hilbert_dominant_cycle_period(
                period=period, close_column=close_column
            )
        )
        cycle_indicators["Hilbert Dominant Cycle Phase"] = (
            self.get_hilbert_dominant_cycle_phase(
                period=period, close_column=close_column
            )
        )

        phasor_components = self.get_hilbert_phasor_components(
            period=period, close_column=close_column
        )
        cycle_indicators["Hilbert In Phase"] = phasor_components["In Phase"]
        cycle_indicators["Hilbert Quadrature"] = phasor_components["Quadrature"]

        sine_wave = self.get_hilbert_sine_wave(period=period, close_column=close_column)
        cycle_indicators["Hilbert Sine"] = sine_wave["Sine"]
        cycle_indicators["Hilbert Lead Sine"] = sine_wave["Lead Sine"]

        cycle_indicators["Hilbert Trend Mode"] = self.get_hilbert_trend_mode(
            period=period, close_column=close_column
        )
        cycle_indicators["Hilbert Instantaneous Trendline"] = self.get_moving_average(
            period=period, close_column=close_column, method="hilbert_trendline"
        )

        mesa_adaptive_moving_average = self.get_mesa_adaptive_moving_average(
            period=period, close_column=close_column
        )
        cycle_indicators["MESA Adaptive Moving Average (MAMA)"] = (
            mesa_adaptive_moving_average["MAMA"]
        )
        cycle_indicators["Following Adaptive Moving Average (FAMA)"] = (
            mesa_adaptive_moving_average["FAMA"]
        )

        cycle_indicators["Swiss Army Knife Filter"] = self.get_swiss_army_knife_filter(
            period=period, close_column=close_column
        )
        cycle_indicators["SuperSmoother Filter"] = self.get_moving_average(
            period=period, close_column=close_column, method="super_smoother"
        )
        cycle_indicators["Even Better Sinewave"] = self.get_even_better_sinewave(
            period=period, close_column=close_column
        )
        cycle_indicators["Detrended Synthetic Price"] = (
            self.get_detrended_synthetic_price(period=period, close_column=close_column)
        )

        mesa_sine_wave = self.get_hilbert_sine_wave(
            period=period, close_column=close_column, method="mesa"
        )
        cycle_indicators["MESA Sine Wave Sine"] = mesa_sine_wave["Sine"]
        cycle_indicators["MESA Sine Wave Lead Sine"] = mesa_sine_wave["Lead Sine"]

        center_of_gravity = self.get_center_of_gravity_oscillator(
            period=period, close_column=close_column
        )
        cycle_indicators["Center of Gravity Oscillator"] = center_of_gravity[
            "Center of Gravity"
        ]
        cycle_indicators["Center of Gravity Trigger"] = center_of_gravity["Trigger"]

        cycle_indicators["Laguerre RSI"] = self.get_relative_strength_index(
            period=period, close_column=close_column, method="laguerre"
        )

        cycle_collection = pd.concat(cycle_indicators, axis=1)

        cycle_collection = apply_rounding(
            cycle_collection,
            rounding if rounding is not None else self._rounding,
        ).loc[self._start_date : self._end_date]

        if growth:
            cycle_collection = calculate_growth(
                dataset=cycle_collection,
                lag=lag,
                rounding=rounding if rounding is not None else self._rounding,
                axis="index",
            )

        if standardize:
            cycle_collection = calculate_standardization(
                dataset=cycle_collection,
                rounding=rounding if rounding is not None else self._rounding,
                axis="rows",
            )

        if len(self._tickers) == 1:
            return cycle_collection.xs(self._tickers[0], level=1, axis="columns")

        return cycle_collection

    @handle_portfolio
    @handle_errors
    def get_hilbert_dominant_cycle_period(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Hilbert Transform Dominant Cycle Period (HT_DCPERIOD): the length, in periods, of the main
        cycle in the prices.

        The dominant cycle period is the length, in periods, of the cycle that dominates the price,
        measured with Ehlers' homodyne discriminator. The smoothed and detrended price is split by the
        Hilbert transform into an in-phase and a quadrature component, the two sides of an analytic
        signal whose angle advances 360 degrees per cycle. Multiplying the signal with that of the
        period before gives the change of the angle per period, and 360 degrees divided by that change
        is the period of the cycle. A short period means the price swings quickly, a long one that it
        moves in slow waves or trends.

        The formula is as follows:

        - Smooth = (4 * Price + 3 * Price(t-1) + 2 * Price(t-2) + Price(t-3)) / 10
        - H(x) = (0.0962 * x + 0.5769 * x(t-2) - 0.5769 * x(t-4) - 0.0962 * x(t-6)) * (0.075 * Period(t-1) + 0.54)
        - Detrender = H(Smooth), Q1 = H(Detrender), I1 = Detrender(t-3)
        - I2 = 0.2 * (I1 - H(Q1)) + 0.8 * I2(t-1) and Q2 = 0.2 * (Q1 + H(I1)) + 0.8 * Q2(t-1)
        - Re = 0.2 * (I2 * I2(t-1) + Q2 * Q2(t-1)) + 0.8 * Re(t-1)
        - Im = 0.2 * (I2 * Q2(t-1) - Q2 * I2(t-1)) + 0.8 * Im(t-1)
        - Period = 360 / arctan(Im / Re) in degrees, limited to between 0.67 and 1.5 times Period(t-1) and to
          between 6 and 50 periods, then smoothed as 0.2 * Period + 0.8 * Period(t-1)
        - Dominant Cycle Period = 0.33 * Period + 0.67 * Dominant Cycle Period(t-1)

        For more information, see the following sources:

        - Ehlers, J.F. (2001). "Rocket Science for Traders: Digital Signal Processing Applications."
          John Wiley & Sons.

        Referenced against:
        - TA-Lib, HT_DCPERIOD: https://ta-lib.org/functions/ht_dcperiod.html

        Shares similar logic:
        - pandas-ta-classic, ht_dcperiod: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/cycles/ht_dcperiod.py#L14

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the dominant cycle period.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame or pd.Series: The dominant cycle period in periods.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the dominant cycle period for each asset in the Toolkit instance.
        - Equals TA-Lib's HT_DCPERIOD and pandas-ta-classic's ht_dcperiod.
        - The recursion starts from zero on the 12th period of each ticker's own history, as in TA-Lib,
          and the first 32 periods (TA-Lib's lookback) are NaN.
        - The Hilbert transform and the homodyne discriminator are shared with the other Hilbert Transform
          indicators and MAMA.
        - When growth is requested, the method calculates the growth of the dominant cycle period over the
          given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_hilbert_dominant_cycle_period().tail(5)
        ```

        Which returns:

        | Date       |    AAPL |    MSFT |   Benchmark |
        |:-----------|--------:|--------:|------------:|
        | 2025-12-24 | 18.3028 | 16.1896 |     18.5522 |
        | 2025-12-26 | 18.6492 | 15.902  |     19.001  |
        | 2025-12-29 | 18.9902 | 15.6336 |     19.0715 |
        | 2025-12-30 | 19.3292 | 15.4975 |     18.9697 |
        | 2025-12-31 | 19.6066 | 15.3876 |     18.9366 |
        """
        self._check_arguments(period, close_column)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._get_historical_data(period, close_column)

        dominant_cycle_period = calculate_per_ticker(
            historical_data,
            [close_column],
            cycle_model.get_hilbert_dominant_cycle_period,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=dominant_cycle_period,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            apply_slice=False,
        )

    @handle_portfolio
    @handle_errors
    def get_hilbert_dominant_cycle_phase(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Hilbert Transform Dominant Cycle Phase (HT_DCPHASE): where the prices are within their main
        cycle, in degrees.

        The dominant cycle phase is the position of the price within its dominant cycle, in degrees.
        It is the angle of the discrete Fourier transform of the smoothed price over exactly one dominant
        cycle period, corrected for the lag of the smoothing. The phase advances by about 360 degrees
        divided by the period each period while the price cycles, and barely moves (or moves backwards)
        while it trends. It runs from -45 to 315 degrees and wraps around once per cycle.

        The formula is as follows:

        - Smooth = (4 * Price + 3 * Price(t-1) + 2 * Price(t-2) + Price(t-3)) / 10
        - H(x) = (0.0962 * x + 0.5769 * x(t-2) - 0.5769 * x(t-4) - 0.0962 * x(t-6)) * (0.075 * Period(t-1) + 0.54)
        - Detrender = H(Smooth), Q1 = H(Detrender), I1 = Detrender(t-3)
        - I2 = 0.2 * (I1 - H(Q1)) + 0.8 * I2(t-1) and Q2 = 0.2 * (Q1 + H(I1)) + 0.8 * Q2(t-1)
        - Re = 0.2 * (I2 * I2(t-1) + Q2 * Q2(t-1)) + 0.8 * Re(t-1)
        - Im = 0.2 * (I2 * Q2(t-1) - Q2 * I2(t-1)) + 0.8 * Im(t-1)
        - Period = 360 / arctan(Im / Re) in degrees, limited to between 0.67 and 1.5 times Period(t-1) and to
          between 6 and 50 periods, then smoothed as 0.2 * Period + 0.8 * Period(t-1)
        - Dominant Cycle Period = 0.33 * Period + 0.67 * Dominant Cycle Period(t-1)
        - N = the Dominant Cycle Period rounded to the nearest whole period
        - Real = Σ sin(360 * i / N) * Smooth(t-i) and Imag = Σ cos(360 * i / N) * Smooth(t-i) for i = 0 to N - 1
        - Dominant Cycle Phase = arctan(Real / Imag) + 90 + 360 / Dominant Cycle Period in degrees, plus 180 when
          Imag is negative and minus 360 when above 315

        For more information, see the following sources:

        - Ehlers, J.F. (2001). "Rocket Science for Traders: Digital Signal Processing Applications."
          John Wiley & Sons.

        Referenced against:
        - TA-Lib, HT_DCPHASE: https://ta-lib.org/functions/ht_dcphase.html

        Shares similar logic:
        - pandas-ta-classic, ht_dcphase: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/cycles/ht_dcphase.py#L14

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the dominant cycle phase.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame or pd.Series: The dominant cycle phase in degrees.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the dominant cycle phase for each asset in the Toolkit instance.
        - Equals TA-Lib's HT_DCPHASE and pandas-ta-classic's ht_dcphase, in degrees as in TA-Lib and Ehlers
          (multiply by π / 180 for radians).
        - The recursion starts from zero on the 37th period of each ticker's own history, as in TA-Lib,
          and the first 63 periods (TA-Lib's lookback) are NaN.
        - The phase is an angle that wraps from 315 to -45 degrees, so its growth and its standardized
          values jump at every new cycle.
        - When growth is requested, the method calculates the growth of the dominant cycle phase over the
          given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_hilbert_dominant_cycle_phase().tail(5)
        ```

        Which returns:

        | Date       |    AAPL |     MSFT |   Benchmark |
        |:-----------|--------:|---------:|------------:|
        | 2025-12-24 |  9.1698 |  54.627  |    -17.3216 |
        | 2025-12-26 | 13.3466 |  80.4208 |     37.7222 |
        | 2025-12-29 | 23.5747 | 104.685  |     74.4394 |
        | 2025-12-30 | 29.9641 | 136.968  |     97.2947 |
        | 2025-12-31 | 33.1911 | 164.331  |    115.377  |
        """
        self._check_arguments(period, close_column)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._get_historical_data(period, close_column)

        dominant_cycle_phase = calculate_per_ticker(
            historical_data,
            [close_column],
            cycle_model.get_hilbert_dominant_cycle_phase,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=dominant_cycle_phase,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            apply_slice=False,
        )

    @handle_portfolio
    @handle_errors
    def get_hilbert_phasor_components(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Hilbert Transform Phasor Components (HT_PHASOR): the in-phase and quadrature parts of the
        prices' cycle, from which its phase and period follow.

        The phasor components are the in-phase component, the smoothed and detrended price lagged three
        periods, and the quadrature component, its Hilbert transform: the same cycle shifted by 90
        degrees. Together they form the analytic signal (the phasor) of the price cycle, whose angle is
        the phase of the cycle and whose length is its amplitude, in price units.

        The formula is as follows:

        - Smooth = (4 * Price + 3 * Price(t-1) + 2 * Price(t-2) + Price(t-3)) / 10
        - H(x) = (0.0962 * x + 0.5769 * x(t-2) - 0.5769 * x(t-4) - 0.0962 * x(t-6)) * (0.075 * Period(t-1) + 0.54)
        - In Phase = H(Smooth)(t-3)
        - Quadrature = H(H(Smooth))
        - Period = the period of the homodyne discriminator before smoothing (see the Dominant Cycle Period)

        For more information, see the following sources:

        - Ehlers, J.F. (2001). "Rocket Science for Traders: Digital Signal Processing Applications."
          John Wiley & Sons.

        Referenced against:
        - TA-Lib, HT_PHASOR: https://ta-lib.org/functions/ht_phasor.html

        Shares similar logic:
        - pandas-ta-classic, ht_phasor: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/cycles/ht_phasor.py#L14

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the phasor components.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The In Phase and Quadrature components in price units.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the phasor components for each asset in the Toolkit instance.
        - Equals TA-Lib's HT_PHASOR and pandas-ta-classic's ht_phasor.
        - The recursion starts from zero on the 12th period of each ticker's own history, as in TA-Lib,
          and the first 32 periods (TA-Lib's lookback) are NaN.
        - When growth is requested, the method calculates the growth of the phasor components over the
          given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_hilbert_phasor_components().xs("AAPL", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |   In Phase |   Quadrature |
        |:-----------|-----------:|-------------:|
        | 2025-12-24 |    -4.0015 |      -0.3152 |
        | 2025-12-26 |    -4.0726 |       2.8793 |
        | 2025-12-29 |    -1.6754 |       4.2075 |
        | 2025-12-30 |    -1.2128 |       1.7906 |
        | 2025-12-31 |    -0.9892 |       3.5115 |
        """
        self._check_arguments(period, close_column)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._get_historical_data(period, close_column)

        phasor_components = calculate_per_ticker(
            historical_data,
            [close_column],
            cycle_model.get_hilbert_phasor_components,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=phasor_components,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            apply_slice=False,
        )

    @handle_portfolio
    @handle_errors
    def get_hilbert_sine_wave(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        method: Literal["standard", "mesa"] = "standard",
        median_price: bool = True,
        window: int = 5,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Hilbert Transform SineWave (HT_SINE), or the MESA Sine Wave over a fixed window: the sine of
        the phase of the prices' main cycle and its lead, whose crossings mark the turning points of the cycle.

        The sine wave is the sine of the dominant cycle phase and the lead sine the sine of the phase
        advanced by 45 degrees. While the price cycles, the two lines cross shortly before the turning
        points of the cycle: the sine crossing above the lead sine marks a low, crossing below a high.
        While the price trends, the phase barely moves and the two lines run apart without crossing.
        The MESA Sine Wave is Ehlers' sinewave indicator over a fixed window: the
        phase of the price within a cycle of that window, measured with a discrete Fourier transform,
        where the Hilbert Transform SineWave measures the window as the dominant cycle.

        The formula is as follows, per method:

        "standard":

        - Smooth = (4 * Price + 3 * Price(t-1) + 2 * Price(t-2) + Price(t-3)) / 10
        - H(x) = (0.0962 * x + 0.5769 * x(t-2) - 0.5769 * x(t-4) - 0.0962 * x(t-6)) * (0.075 * Period(t-1) + 0.54)
        - Detrender = H(Smooth), Q1 = H(Detrender), I1 = Detrender(t-3)
        - I2 = 0.2 * (I1 - H(Q1)) + 0.8 * I2(t-1) and Q2 = 0.2 * (Q1 + H(I1)) + 0.8 * Q2(t-1)
        - Re = 0.2 * (I2 * I2(t-1) + Q2 * Q2(t-1)) + 0.8 * Re(t-1)
        - Im = 0.2 * (I2 * Q2(t-1) - Q2 * I2(t-1)) + 0.8 * Im(t-1)
        - Period = 360 / arctan(Im / Re) in degrees, limited to between 0.67 and 1.5 times Period(t-1) and to
          between 6 and 50 periods, then smoothed as 0.2 * Period + 0.8 * Period(t-1)
        - Dominant Cycle Period = 0.33 * Period + 0.67 * Dominant Cycle Period(t-1)
        - N = the Dominant Cycle Period rounded to the nearest whole period
        - Real = Σ sin(360 * i / N) * Smooth(t-i) and Imag = Σ cos(360 * i / N) * Smooth(t-i) for i = 0 to N - 1
        - Dominant Cycle Phase = arctan(Real / Imag) + 90 + 360 / Dominant Cycle Period in degrees, plus 180 when
          Imag is negative and minus 360 when above 315
        - Sine = sin(Dominant Cycle Phase)
        - Lead Sine = sin(Dominant Cycle Phase + 45 degrees)

        "mesa":

        - Real = Σ sin(360 * i / Window) * Price(t-i) and Imaginary = Σ cos(360 * i / Window) * Price(t-i) for
          i = 0 to Window - 1
        - Phase = arctan(Real / Imaginary) + 90 degrees, plus 180 when Imaginary is negative, or 90 * sign(Real) + 90
          when |Imaginary| is at most 0.001
        - Sine = sin(Phase)
        - Lead Sine = sin(Phase + 45 degrees)

        For more information, see the following sources:

        - Ehlers, J.F. (2001). "Rocket Science for Traders: Digital Signal Processing Applications."
          John Wiley & Sons.
        - Ehlers, J.F. (1992). "MESA and Trading Market Cycles." John Wiley & Sons.

        Referenced against:
        - TA-Lib, HT_SINE: https://ta-lib.org/functions/ht_sine.html
        - Tulip Indicators, msw (MESA Sine Wave): https://tulipindicators.org/msw
        - pandas-ta-classic, msw (MESA Sine Wave): https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/cycles/msw.py#L14

        Shares similar logic:
        - pandas-ta-classic, ht_sine: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/cycles/ht_sine.py#L14

        Also known as: HT_SINE, Hilbert Transform SineWave, sine wave, lead sine, MSW, MESA Sine Wave, MESA sinewave.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the high and low of the median price are
                adjusted for dividends in the same way.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the sine wave.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            method (str, optional): The sine wave. Options are: "standard" (the Hilbert Transform SineWave),
                "mesa" (the MESA Sine Wave over a fixed window). Defaults to "standard".
            median_price (bool, optional): Whether to filter the median price (High + Low) / 2, as
                Ehlers does, or the close of `close_column`, as Tulip Indicators and
                pandas-ta-classic do. Only used when method="mesa". Defaults to True.
            window (int, optional): The number of periods of the cycle. Only used when
                method="mesa". Defaults to 5.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Sine and Lead Sine, between -1 and 1.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the sine wave for each asset in the Toolkit instance.
        - Equals TA-Lib's HT_SINE and pandas-ta-classic's ht_sine.
        - The recursion starts from zero on the 37th period of each ticker's own history, as in TA-Lib,
          and the first 63 periods (TA-Lib's lookback) are NaN.
        - MESA Sine Wave: equals pandas-ta-classic's msw on the close (not the median price), and Tulip
          Indicators' msw to 1e-4, as Tulip takes 3.1415926 for π. Both start one period later, on the
          period after the first full window. Without an imaginary part the phase is 90 degrees times
          the sign of the real part, as in Ehlers' sinewave indicator, where Tulip Indicators and
          pandas-ta-classic take 180 degrees.
        - When growth is requested, the method calculates the growth of the sine wave over the
          given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_hilbert_sine_wave().xs("AAPL", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |   Lead Sine |   Sine |
        |:-----------|------------:|-------:|
        | 2025-12-24 |      0.8108 | 0.1594 |
        | 2025-12-26 |      0.8512 | 0.2308 |
        | 2025-12-29 |      0.9309 | 0.3999 |
        | 2025-12-30 |      0.9658 | 0.4995 |
        | 2025-12-31 |      0.9788 | 0.5474 |
        """
        self._check_arguments(period, close_column, window=window)
        methods = ("standard", "mesa")
        check_option("method", method, methods)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._get_historical_data(period, close_column)

        if method == "mesa":
            # The median price (High + Low) / 2, or the close: the average of the price columns.
            sine_wave = calculate_per_ticker(
                historical_data,
                ["High", "Low"] if median_price else [close_column],
                lambda *prices: cycle_model.get_mesa_sine_wave(
                    sum(prices) / len(prices), window
                ),
            )
        else:
            sine_wave = calculate_per_ticker(
                historical_data,
                [close_column],
                cycle_model.get_hilbert_sine_wave,
            )
        sine_wave = sine_wave.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=sine_wave,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            apply_slice=False,
        )

    @handle_portfolio
    @handle_errors
    def get_hilbert_trend_mode(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Hilbert Transform Trend vs Cycle Mode (HT_TRENDMODE): whether the prices are trending (1) or
        moving in a cycle (0).

        The trend mode tells whether the price trends (1) or moves in a cycle (0). Ehlers' rules: a
        crossing of the sine and the lead sine starts a cycle, which lasts at least half a dominant
        cycle period; a phase that advances at the rate of the dominant cycle is a cycle; and a smoothed
        price 1.5% or more away from the instantaneous trendline is always a trend. Cycle indicators
        (oscillators) fit the cycle mode, trend-following indicators the trend mode.

        The formula is as follows:

        - Trend = 1, or 0 with the periods in trend restarting from zero when the Sine and the Lead Sine
          cross (see the Hilbert Transform SineWave)
        - Trend = 0 while the periods in trend are fewer than half the Dominant Cycle Period
        - Trend = 0 when the change of the Dominant Cycle Phase is between 0.67 and 1.5 times 360 / Dominant
          Cycle Period
        - Trend = 1 when |Smooth - Trendline| / Trendline >= 0.015 (see the Instantaneous Trendline)

        For more information, see the following sources:

        - Ehlers, J.F. (2001). "Rocket Science for Traders: Digital Signal Processing Applications."
          John Wiley & Sons.

        Referenced against:
        - TA-Lib, HT_TRENDMODE: https://ta-lib.org/functions/ht_trendmode.html

        Shares similar logic:
        - pandas-ta-classic, ht_trendmode: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/cycles/ht_trendmode.py#L14

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the trend mode.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame or pd.Series: 1 when the price trends and 0 when it cycles.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the trend mode for each asset in the Toolkit instance.
        - Equals TA-Lib's HT_TRENDMODE and pandas-ta-classic's ht_trendmode after the lookback.
        - The recursion starts from zero on the 37th period of each ticker's own history, as in TA-Lib,
          and the first 63 periods (TA-Lib's lookback) are NaN, where TA-Lib returns 0.
        - When growth is requested, the method calculates the growth of the trend mode over the
          given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_hilbert_trend_mode().tail(5)
        ```

        Which returns:

        | Date       |   AAPL |   MSFT |   Benchmark |
        |:-----------|-------:|-------:|------------:|
        | 2025-12-24 |      1 |      0 |           0 |
        | 2025-12-26 |      0 |      0 |           0 |
        | 2025-12-29 |      0 |      0 |           0 |
        | 2025-12-30 |      1 |      0 |           0 |
        | 2025-12-31 |      1 |      0 |           0 |
        """
        self._check_arguments(period, close_column)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._get_historical_data(period, close_column)

        trend_mode = calculate_per_ticker(
            historical_data,
            [close_column],
            cycle_model.get_hilbert_trend_mode,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=trend_mode,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            apply_slice=False,
        )

    @handle_portfolio
    @handle_errors
    def get_mesa_adaptive_moving_average(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        fast_limit: float = 0.5,
        slow_limit: float = 0.05,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the MESA Adaptive Moving Average (MAMA) and the Following Adaptive Moving Average (FAMA): averages
        that adapt to the speed of the prices' cycle, so they follow fast moves and stay flat in a range.

        MAMA is an exponential moving average whose smoothing factor follows the rate at which the phase
        of the price cycle changes, measured with the Hilbert transform. When the phase jumps, as at the
        start of a move, the smoothing factor is the fast limit and MAMA follows the price closely; when
        the phase advances slowly, the factor falls towards the slow limit and MAMA flattens. FAMA applies
        half the smoothing factor to MAMA, so that it follows MAMA, and Ehlers reads the crossings of the
        two lines as changes of direction.

        The formula is as follows:

        - Smooth = (4 * Price + 3 * Price(t-1) + 2 * Price(t-2) + Price(t-3)) / 10
        - I1 and Q1 = the In Phase and Quadrature components (see the Hilbert Transform Phasor Components)
        - Phase = arctan(Q1 / I1) in degrees
        - Delta Phase = Phase(t-1) - Phase, at least 1
        - Alpha = Fast Limit / Delta Phase, at least the Slow Limit
        - MAMA = Alpha * Price + (1 - Alpha) * MAMA(t-1)
        - FAMA = 0.5 * Alpha * MAMA + (1 - 0.5 * Alpha) * FAMA(t-1)

        For more information, see the following sources:

        - Ehlers, J.F. (2001). "MESA Adaptive Moving Averages." Technical Analysis of Stocks & Commodities,
          September 2001.
        - Ehlers, J.F. (2001). "Rocket Science for Traders: Digital Signal Processing Applications."
          John Wiley & Sons.

        Referenced against:
        - TA-Lib, MAMA: https://ta-lib.org/functions/mama.html

        Shares similar logic:
        - pandas-ta-classic, mama: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/overlap/mama.py#L248

        Also known as: MAMA, FAMA, MESA adaptive moving average.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            fast_limit (float, optional): The largest smoothing factor, used when the phase jumps.
                Defaults to 0.5, Ehlers' value.
            slow_limit (float, optional): The smallest smoothing factor, used when the phase advances
                slowly. Defaults to 0.05, Ehlers' value.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the MESA adaptive moving average.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The MAMA and FAMA in price units.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the MESA adaptive moving average for each asset in the Toolkit instance.
        - Equals TA-Lib's MAMA and pandas-ta-classic's mama on the close. Ehlers applies it to the median
          price (High + Low) / 2 in his article.
        - The recursion starts from zero on the 12th period of each ticker's own history, as in TA-Lib,
          and the first 32 periods (TA-Lib's lookback) are NaN. MAMA and FAMA also start from zero there, as in
          TA-Lib and Ehlers' article. MAMA has caught up with the price by the end of the lookback, but FAMA, at half
          the smoothing factor, is then still several percent below it and settles to within 0.1% after about
          80 periods. The Toolkit calculates the technical indicators from a year before `start_date`, so with
          daily data this warm-up falls before the dates shown.
        - With the default `close_column` of "Adj Close" the prices are adjusted for dividends.
        - When growth is requested, the method calculates the growth of the MESA adaptive moving average over
          the given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_mesa_adaptive_moving_average().xs("AAPL", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |    FAMA |    MAMA |
        |:-----------|--------:|--------:|
        | 2025-12-24 | 274.072 | 273.981 |
        | 2025-12-26 | 274.067 | 273.904 |
        | 2025-12-29 | 274.062 | 273.85  |
        | 2025-12-30 | 273.794 | 272.991 |
        | 2025-12-31 | 273.772 | 272.887 |
        """
        self._check_arguments(period, close_column)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._get_historical_data(period, close_column)

        mesa_adaptive_moving_average = calculate_per_ticker(
            historical_data,
            [close_column],
            cycle_model.get_mesa_adaptive_moving_average,
            fast_limit,
            slow_limit,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=mesa_adaptive_moving_average,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            apply_slice=False,
        )

    @handle_portfolio
    @handle_errors
    def get_swiss_army_knife_filter(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        median_price: bool = True,
        window: int = 20,
        method: str = "band_pass",
        delta: float = 0.1,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate Ehlers' Swiss Army Knife filter: a family of filters that remove the trend (high-pass), the noise
        (low-pass) or both (band-pass) from the prices.

        The Swiss Army Knife indicator of Ehlers (2006) is one second-order recursive filter whose coefficients
        select its response. The high-pass filters remove the trend and keep the cycles shorter than the window,
        the two-pole one with a steeper cut than the one-pole one. The band-pass filter keeps only the cycles around
        the window, which shows how strongly that cycle is present and where it turns. The Butterworth and Gaussian
        filters are low-pass filters that smooth away the cycles shorter than the window with little lag.

        The formula is as follows:

        - Filter = c0 * (b0 * Price + b1 * Price(t-1) + b2 * Price(t-2)) + a1 * Filter(t-1) + a2 * Filter(t-2),
          with b0 = 1 and every other coefficient 0 unless set below
        - One-pole alpha = (cos(360 / Window) + sin(360 / Window) - 1) / cos(360 / Window)
        - Two-pole alpha = -Beta + √(Beta² + 2 * Beta), with Beta = 2.415 * (1 - cos(360 / Window))
        - High-pass: c0 = 1 - alpha / 2, b1 = -1 and a1 = 1 - alpha, with the one-pole alpha
        - Two-pole high-pass: c0 = (1 - alpha / 2)², b1 = -2, b2 = 1, a1 = 2 * (1 - alpha) and a2 = -(1 - alpha)²
        - Butterworth: c0 = alpha² / 4, b1 = 2, b2 = 1, a1 = 2 * (1 - alpha) and a2 = -(1 - alpha)²
        - Gaussian: c0 = alpha², a1 = 2 * (1 - alpha) and a2 = -(1 - alpha)²
        - Band-pass: Beta = cos(360 / Window), Gamma = 1 / cos(720 * Delta / Window), alpha = Gamma - √(Gamma² - 1),
          c0 = (1 - alpha) / 2, b2 = -1, a1 = Beta * (1 + alpha) and a2 = -alpha

        For more information, see the following sources:

        - Ehlers, J.F. (2006). "Swiss Army Knife Indicator." Technical Analysis of Stocks & Commodities, 24(1),
          28-31, 50-53.

        Referenced against:
        - TA-Lib, SWAK_HP: https://ta-lib.org/functions/swak_hp.html
        - TA-Lib, SWAK_2PHP: https://ta-lib.org/functions/swak_2php.html
        - TA-Lib, SWAK_BP: https://ta-lib.org/functions/swak_bp.html
        - TA-Lib, SWAK_BUTTER: https://ta-lib.org/functions/swak_butter.html
        - TA-Lib, SWAK_GAUSS: https://ta-lib.org/functions/swak_gauss.html

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the high and low prices are adjusted
                for dividends in the same way.
            median_price (bool, optional): Whether to filter the median price (High + Low) / 2, as Ehlers does,
                or the close of `close_column`, as TA-Lib's SWAK functions do. Defaults to True.
            window (int, optional): The cutoff period of the high-pass and low-pass filters and the centre period of
                the band-pass filter, at least 5. Defaults to 20.
            method (str, optional): The filter. Options are: "high_pass" (one-pole high-pass filter),
                "two_pole_high_pass" (two-pole high-pass filter), "band_pass" (band-pass filter),
                "butterworth" (two-pole Butterworth low-pass filter), "gaussian" (Gaussian low-pass filter).
                Defaults to "band_pass", the default of Ehlers' listing.
            delta (float, optional): The half bandwidth of the band-pass filter as a fraction of its centre
                frequency, above 0 and at most 0.5. Defaults to 0.1.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Swiss Army Knife filter.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame or pd.Series: The filtered prices, in price units for the Butterworth and Gaussian filters
            and centred on zero for the high-pass and band-pass filters.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Swiss Army Knife filter for each asset in the Toolkit instance.
        - Equals TA-Lib's SWAK functions on the close (not the median price), from the first period on.
        - The filter starts in the steady state of a constant price equal to the first price of each ticker's own
          history: the low-pass filters on that price, the high-pass and band-pass filters on zero, as in TA-Lib.
          Ehlers' listing starts every filter from zero, which keeps the low-pass filters far below the price for
          many periods.
        - When growth is requested, the method calculates the growth of the Swiss Army Knife filter over the
          given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_swiss_army_knife_filter().tail(5)
        ```

        Which returns:

        | Date       |   AAPL |   MSFT |   Benchmark |
        |:-----------|-------:|-------:|------------:|
        | 2025-12-24 | 0.2127 | 1.3624 |     -0.207  |
        | 2025-12-26 | 1.0468 | 2.0027 |      1.5323 |
        | 2025-12-29 | 1.7167 | 2.3917 |      2.9695 |
        | 2025-12-30 | 2.1545 | 2.5467 |      3.9497 |
        | 2025-12-31 | 2.3417 | 2.4341 |      4.4016 |
        """
        self._check_arguments(period, close_column, window=window)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._get_historical_data(period, close_column)

        # The median price (High + Low) / 2, or the close: the average of the price columns.
        price_columns = ["High", "Low"] if median_price else [close_column]

        swiss_army_knife_filter = calculate_per_ticker(
            historical_data,
            price_columns,
            lambda *prices: cycle_model.get_swiss_army_knife_filter(
                sum(prices) / len(prices), window, method, delta
            ),
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=swiss_army_knife_filter,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            apply_slice=False,
        )

    @handle_portfolio
    @handle_errors
    def get_even_better_sinewave(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 40,
        smooth_window: int = 10,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate Ehlers' Even Better Sinewave: an oscillator between -1 and 1 that shows whether the prices move in
        a cycle or a trend and where they are in the cycle.

        The Even Better Sinewave of Ehlers (2013) removes the trend with a one-pole high-pass filter, smooths the
        remaining cycles with a SuperSmoother and divides their average over three periods by the square root of
        their power over the same periods. The result swings between -1 and 1 like a sine wave while the price
        cycles and stays near 1 (or -1) while it trends up (or down), so it tells a cycle from a trend and marks
        the turns of the cycle.

        The formula is as follows:

        - alpha = (1 - sin(360 / Window)) / cos(360 / Window)
        - HP = 0.5 * (1 + alpha) * (Price - Price(t-1)) + alpha * HP(t-1)
        - a = exp(-√2 * π / Smoothing Window), b = 2 * a * cos(√2 * 180 / Smoothing Window), c2 = b, c3 = -a² and
          c1 = 1 - c2 - c3
        - Filter = c1 * (HP + HP(t-1)) / 2 + c2 * Filter(t-1) + c3 * Filter(t-2)
        - Wave = (Filter + Filter(t-1) + Filter(t-2)) / 3
        - Power = (Filter² + Filter(t-1)² + Filter(t-2)²) / 3
        - Even Better Sinewave = Wave / √Power

        For more information, see the following sources:

        - Ehlers, J.F. (2013). "Cycle Analytics for Traders: Advanced Technical Trading Concepts." John Wiley & Sons.

        Referenced against:
        - pandas-ta-classic, ebsw: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/cycles/ebsw.py#L63

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The cutoff period of the high-pass filter, the longest cycle kept. Defaults to 40.
            smooth_window (int, optional): The critical period of the SuperSmoother. Defaults to 10.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Even Better Sinewave.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame or pd.Series: The Even Better Sinewave, between -1 and 1.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Even Better Sinewave for each asset in the Toolkit instance.
        - The filters start from zero on the first period of each ticker's own history, as in Ehlers' listing, and
          the first `window` periods are NaN while they settle. pandas-ta-classic starts them on the 12th period;
          the two converge as the start fades and are equal on prices that do not move before it.
        - A period without power (three filter values of zero) is NaN, where pandas-ta-classic gives 0.
        - When growth is requested, the method calculates the growth of the Even Better Sinewave over the
          given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_even_better_sinewave().tail(5)
        ```

        Which returns:

        | Date       |    AAPL |   MSFT |   Benchmark |
        |:-----------|--------:|-------:|------------:|
        | 2025-12-24 | -0.9962 | 0.6793 |      0.5028 |
        | 2025-12-26 | -0.9816 | 0.9452 |      0.8897 |
        | 2025-12-29 | -0.9626 | 0.9932 |      0.9801 |
        | 2025-12-30 | -0.9522 | 0.9992 |      0.9989 |
        | 2025-12-31 | -0.9772 | 0.9868 |      0.9863 |
        """
        self._check_arguments(
            period, close_column, window=window, smooth_window=smooth_window
        )
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._get_historical_data(period, close_column)

        even_better_sinewave = calculate_per_ticker(
            historical_data,
            [close_column],
            cycle_model.get_even_better_sinewave,
            window,
            smooth_window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=even_better_sinewave,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            apply_slice=False,
        )

    @handle_portfolio
    @handle_errors
    def get_detrended_synthetic_price(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 20,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate Ehlers' Detrended Synthetic Price: the difference between a fast and a slow average tuned to the
        prices' cycle, which shows the cycle without the trend.

        The Detrended Synthetic Price of Ehlers (1992) is the difference between an exponential moving average over
        a quarter of the dominant cycle and one over half of it. The difference removes the trend and leaves a
        wave in phase with the dominant cycle of the price: above zero in the rising half of the cycle and below
        zero in the falling half.

        The formula is as follows:

        - Detrended Synthetic Price = EMA(Price, Window / 4) - EMA(Price, Window / 2)

        For more information, see the following sources:

        - Ehlers, J.F. (1992). "MESA and Trading Market Cycles." John Wiley & Sons.

        Referenced against:
        - pandas-ta-classic, dsp: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/cycles/dsp.py#L12

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The period of the dominant cycle, at least 4. Defaults to 20.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Detrended Synthetic Price.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame or pd.Series: The Detrended Synthetic Price in price units.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Detrended Synthetic Price for each asset in the Toolkit instance.
        - The averages span a quarter and a half of the window, rounded down, and are seeded with their simple
          average as every exponential moving average of the Toolkit; they agree with pandas-ta-classic's ema.
        - pandas-ta-classic's dsp subtracts one exponential moving average from the price, Price - EMA(Price, 14),
          a simpler detrender that also keeps the shorter cycles.
        - When growth is requested, the method calculates the growth of the Detrended Synthetic Price over the
          given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_detrended_synthetic_price().tail(5)
        ```

        Which returns:

        | Date       |    AAPL |   MSFT |   Benchmark |
        |:-----------|--------:|-------:|------------:|
        | 2025-12-24 | -0.8825 | 1.3797 |      2.6009 |
        | 2025-12-26 | -0.6725 | 1.4629 |      2.902  |
        | 2025-12-29 | -0.4628 | 1.328  |      2.5204 |
        | 2025-12-30 | -0.423  | 1.231  |      2.0333 |
        | 2025-12-31 | -0.5599 | 0.5235 |      0.8791 |
        """
        self._check_arguments(period, close_column, window=window)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._get_historical_data(period, close_column)

        detrended_synthetic_price = calculate_per_ticker(
            historical_data,
            [close_column],
            cycle_model.get_detrended_synthetic_price,
            window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=detrended_synthetic_price,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            apply_slice=False,
        )

    @handle_portfolio
    @handle_errors
    def get_center_of_gravity_oscillator(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        median_price: bool = True,
        window: int = 10,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate Ehlers' Center of Gravity (CG) Oscillator: the weighted position of the prices within a window,
        which marks the turning points of the cycle with little lag.

        The Center of Gravity Oscillator of Ehlers (2002) is the balance point of the prices in the window, with the
        latest price at position 1, turned negative. When prices rise, the weight shifts to the latest prices and
        the balance point moves towards them, so the oscillator rises with the price, smoothed and with almost no
        lag. The trigger is the previous value, and a crossing of the two marks a turn. Ehlers sets the window to
        half the dominant cycle.

        The formula is as follows:

        - Center of Gravity = -Σ (i + 1) * Price(t-i) / Σ Price(t-i) for i = 0 to Window - 1
        - Trigger = Center of Gravity(t-1)

        For more information, see the following sources:

        - Ehlers, J.F. (2002). "The Center Of Gravity Oscillator." Technical Analysis of Stocks & Commodities,
          20(5), 20-24.

        Referenced against:
        - TA-Lib, CG: https://ta-lib.org/functions/cg.html
        - pandas-ta-classic, cg: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/cg.py#L12

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the high and low prices are adjusted
                for dividends in the same way.
            median_price (bool, optional): Whether to filter the median price (High + Low) / 2, as Ehlers does,
                or the close of `close_column`, as TA-Lib and pandas-ta-classic do. Defaults to True.
            window (int, optional): The number of periods, ideally half the dominant cycle. Defaults to 10.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Center of Gravity Oscillator.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Center of Gravity and its Trigger.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Center of Gravity Oscillator for each asset in the Toolkit instance.
        - Equals TA-Lib's CG and pandas-ta-classic's cg on the close (not the median price).
        - The oscillator is not centred on zero: for flat prices it is -(Window + 1) / 2.
        - A window whose prices sum to zero is NaN, where Ehlers' listing keeps the value before.
        - When growth is requested, the method calculates the growth of the Center of Gravity Oscillator over the
          given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_center_of_gravity_oscillator().xs("AAPL", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |   Center of Gravity |   Trigger |
        |:-----------|--------------------:|----------:|
        | 2025-12-24 |             -5.519  |   -5.5259 |
        | 2025-12-26 |             -5.5128 |   -5.519  |
        | 2025-12-29 |             -5.5043 |   -5.5128 |
        | 2025-12-30 |             -5.4973 |   -5.5043 |
        | 2025-12-31 |             -5.4957 |   -5.4973 |
        """
        self._check_arguments(period, close_column, window=window)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._get_historical_data(period, close_column)

        # The median price (High + Low) / 2, or the close: the average of the price columns.
        price_columns = ["High", "Low"] if median_price else [close_column]

        center_of_gravity = calculate_per_ticker(
            historical_data,
            price_columns,
            lambda *prices: cycle_model.get_center_of_gravity_oscillator(
                sum(prices) / len(prices), window
            ),
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=center_of_gravity,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            apply_slice=False,
        )
