"""Labeling Module"""

__docformat__ = "google"

import functools
from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd

from financetoolkit.technicals.helpers import apply_on_own_closes, report_missing_values
from financetoolkit.technicals.labeling import bet_sizing_model, labeling_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import (
    check_option,
    handle_errors,
    validate_window,
)
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import apply_rounding

# The examples' output tables are wider than the line length.
# ruff: noqa: E501

# The columns of a Toolkit that are not tickers of their own, left out of the cross-sections.
NON_TICKER_COLUMNS = ("Benchmark", "Portfolio")


def _per_ticker(
    closes: pd.DataFrame, function: Callable[[pd.Series], pd.DataFrame]
) -> pd.DataFrame:
    """
    Applies a calculation returning several columns to each ticker's closes over the dates the
    ticker has a close, so its periods are its own trading days, and places the result back on
    every date, NaN on the dates without a close.

    Args:
        closes (pd.DataFrame): The closes, a column per ticker, NaN on the dates a ticker did not
            trade.
        function (Callable[[pd.Series], pd.DataFrame]): The calculation over one ticker's closes.

    Returns:
        pd.DataFrame: The result with (measure, ticker) columns.
    """
    results = {
        ticker: function(closes[ticker].dropna()).reindex(closes.index)
        for ticker in closes.columns
    }
    measures = next(iter(results.values())).columns
    return pd.concat(results, axis=1).swaplevel(0, 1, axis=1)[list(measures)]


@report_missing_values
@filter_shown_columns
@track_sources
class LabelingTechnicals:
    """
    Labels and sample weights for financial machine learning (López de Prado): the CUSUM and
    z-score event filters, triple-barrier labels and meta-labels, fixed-horizon and relative
    labels, trend-scanning, tail-set and matrix flag labels, the concurrency and uniqueness of
    labels and sample weights by return attribution and time decay. Apart from the event filters
    and the matrix flags, labels look forward by construction: they describe what happens after
    a date, which is why they are not part of the indicator collections or the factor analysis.
    """

    def _get_labels_or_default(
        self,
        labels: pd.DataFrame | None,
        period: str,
        close_column: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        The closes and the holding periods of the labels, from the labels given or else from the
        triple-barrier labels with their defaults.

        Args:
            labels (pd.DataFrame | None): Labels with a Holding Period per date and ticker, or
                None for the default triple-barrier labels.
            period (str): The period of the prices.
            close_column (str): The close column.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: The closes and the holding periods, a column per
            ticker on the dates of the closes.
        """
        closes = self._close_prices(period, close_column, traded_only=True)
        if labels is None:
            labels = self.get_triple_barrier_labels(
                period=period, close_column=close_column, rounding=15
            )
        if "Holding Period" not in labels.columns.get_level_values(0):
            raise ValueError(
                "The labels need a 'Holding Period' column per ticker, such as the result of "
                "get_triple_barrier_labels, get_fixed_horizon_labels or get_trend_scanning_labels."
            )
        holding_periods = labels["Holding Period"].reindex(index=closes.index)
        tickers = [ticker for ticker in closes.columns if ticker in holding_periods]
        return closes[tickers], holding_periods[tickers]

    @handle_errors
    def get_event_filter(
        self,
        method: Literal["cusum", "z_score"] = "cusum",
        period: str = "daily",
        close_column: str = "Adj Close",
        threshold: float | None = None,
        span: int = 100,
        window: int = 20,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the events of an event filter: the dates something happens that is worth labeling, such as a run
        of returns in one direction.

        Sampling on events rather than on every date keeps the labels of a quiet market from outnumbering those of
        the moves that matter, and the events are the dates `get_triple_barrier_labels(events=...)` labels.

        - CUSUM filter: the symmetric CUSUM filter of López de Prado, after the quality control
          charts of Page (1954) and the trading rule of Lam and Yam (1997). It sums the log
          returns since the last event of each direction and reports an event when a sum moves
          beyond `threshold` daily volatilities, after which that sum starts again from zero. A
          price hovering around a level triggers no events, unlike a band crossing, as it takes
          a full run of the threshold to set one off.
        - Z-score filter: the z-score filter: an event when the price lies at least `threshold`
          standard deviations from its average over `window` periods, a filter after the
          z-score peak detection of van Brakel (2014).

        The formula is as follows, per method:

        - CUSUM filter: y_t = ln(p_t / p_t-1), S+_t = max(0, S+_t-1 + y_t), S-_t = min(0, S-_t-1 + y_t),
          h_t = threshold x sigma_t (the AFML daily volatility, an EWM standard deviation of the
          returns with `span`); an event of -1 when S-_t < -h_t (S-_t resets) or else of 1 when
          S+_t > h_t (S+_t resets)
        - Z-score filter: z_t = (p_t - mean(p, window)) / std(p, window); an event of 1 when
          z_t >= threshold and of -1 when z_t <= -threshold

        Each value uses the prices up to its date only.

        For more information, see the following books and papers:

        - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, section
          2.5.2.1 (Snippet 2.4).
        - Lam, K. and Yam, H.C. (1997). "CUSUM Techniques for Technical Trading in Financial
          Markets." Financial Engineering and the Japanese Markets, 4(3), 257-274.
        - Page, E.S. (1954). "Continuous Inspection Schemes." Biometrika, 41(1/2), 100-115.
        - van Brakel, J.P.G. (2014). "Robust peak detection algorithm using z-scores." Stack
          Overflow (https://stackoverflow.com/questions/22583391).

        Referenced against:
        - RiskLabAI, symmetric_cusum_filter: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/data/labeling/labeling.py#L78

        Also known as: CUSUM filter, symmetric CUSUM filter, event-based sampling, z-score filter,
        event filter.

        Args:
            method (str, optional): The filter. Options are: "cusum" (the symmetric CUSUM filter),
                "z_score" (the z-score filter). Defaults to "cusum".
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            threshold (float | None, optional): The threshold in volatilities: the multiple of
                the daily volatility the CUSUM sums need to pass, or the number of standard
                deviations of the z-score. Defaults to None, which uses 1 for "cusum" (the daily
                volatility itself) and 3 for "z_score".
            span (int, optional): The span of the daily volatility of the CUSUM filter. Defaults
                to 100.
            window (int, optional): The window of the average and standard deviation of the
                z-score filter. Defaults to 20.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The events per date and ticker: 1 for an upward event, -1 for a
            downward event and 0 otherwise.

        Notes:
        - The CUSUM sums start on each ticker's first return and the threshold needs two returns
          for its volatility, so the first event comes after the third price.
        - The z-score filter reports prices both above and below the band and takes one window
          for the average and the standard deviation.
        - The CUSUM test of `toolkit.econometrics.get_cusum_test` is a different tool: a test of
          a stable mean over the whole sample rather than a filter that marks dates.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2024-01-01",
            end_date="2025-12-31",
        )

        events = toolkit.technicals.get_event_filter(threshold=2)

        events[events.ne(0).any(axis=1)].tail()
        ```

        Which returns:

        | Date       |   AAPL |   MSFT |   Benchmark |
        |:-----------|-------:|-------:|------------:|
        | 2025-12-08 |     -1 |      1 |           0 |
        | 2025-12-10 |      0 |     -1 |           0 |
        | 2025-12-17 |      0 |      0 |          -1 |
        | 2025-12-22 |     -1 |      0 |           1 |
        | 2025-12-23 |      0 |      1 |           0 |
        """
        self._check_arguments(period, close_column, span=span, window=window)
        check_option("method", method, ("cusum", "z_score"))
        closes = self._close_prices(period, close_column, traded_only=True)

        if method == "cusum":
            multiple = 1.0 if threshold is None else threshold
            events = apply_on_own_closes(
                closes,
                lambda prices: labeling_model.get_cusum_events(
                    prices,
                    multiple * labeling_model.get_daily_volatility(prices, span),
                ),
            )
        else:
            events = apply_on_own_closes(
                closes,
                lambda prices: labeling_model.get_z_score_events(
                    prices, window, 3.0 if threshold is None else threshold
                ),
            )

        return events.loc[self._start_date : self._end_date]

    @handle_errors
    def get_triple_barrier_labels(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        horizon: int = 10,
        profit_taking: float = 1.0,
        stop_loss: float = 1.0,
        span: int = 100,
        minimum_target: float = 0.0,
        events: str | pd.DataFrame | None = None,
        side: str | pd.DataFrame | None = None,
        vertical_barrier_zero: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the triple-barrier labels of López de Prado: for every date (or event) the first of three barriers
        the price reaches.

        A profit-taking barrier lies `profit_taking` daily volatilities above the price, a stop-loss barrier
        `stop_loss` daily volatilities below it and a vertical barrier `horizon` periods later. The label is 1 when
        the upper barrier is reached first, -1 for the lower barrier and the sign of the return at the vertical
        barrier. Unlike a label on the return over a fixed horizon, it follows the path of the price the way a trade
        with a profit target, a stop-loss and a time limit does, and its barriers widen and narrow with the
        volatility.

        With a `side`, such as the crossovers of `get_signals`, the labels are meta-labels: the
        returns are those of the bet (multiplied by the side) and the label is 1 when the bet
        made money and 0 when it did not. A second model trained on them learns when to act on
        the first model's signals and how much to bet, rather than which side to take.

        The formula is as follows:

        - sigma_t = the EWM standard deviation of the period returns with `span`, up to t
        - R_t,k = (p_t+k / p_t - 1) x side_t, for k = 1..horizon (side 1 without a side)
        - upper_t = profit_taking x sigma_t and lower_t = -stop_loss x sigma_t
        - k* = the first k with R_t,k > upper_t or R_t,k < lower_t, else horizon
        - Label = 1 (upper), -1 (lower) or sign(R_t,k*) (vertical); with a side 1 when
          R_t,k* > 0, else 0

        Labels look forward by construction: the label of a date describes the prices after it,
        up to `horizon` periods later, and is only known once a barrier is reached. The volatility
        of the barriers uses the returns up to the date only. Train a model on the labels with
        the features of the date itself and leave the labels' spans out of the test set (purging,
        see `get_label_uniqueness`).

        For more information, see the following book:

        - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, chapter 3
          (Snippets 3.1 to 3.7).

        Referenced against:
        - RiskLabAI, triple_barrier: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/data/labeling/labeling.py#L193
        - RiskLabAI, meta_labeling: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/data/labeling/labeling.py#L402

        Also known as: triple barrier method, triple-barrier labeling, meta-labeling, meta labels,
        profit-taking and stop-loss labels.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            horizon (int, optional): The number of periods to the vertical barrier. Defaults to 10.
            profit_taking (float, optional): The multiple of the volatility of the upper barrier,
                0 for no upper barrier. Defaults to 1.0.
            stop_loss (float, optional): The multiple of the volatility of the lower barrier, 0 for
                no lower barrier. Defaults to 1.0.
            span (int, optional): The span of the exponentially weighted volatility. Defaults to
                100, as López de Prado.
            minimum_target (float, optional): The volatility a date needs to exceed to be labeled.
                Defaults to 0.0.
            events (str | pd.DataFrame | None, optional): The dates to label. Options are: "cusum" (the events
                of the CUSUM filter of `get_event_filter` with its defaults), "z_score" (the events of its
                z-score filter with its defaults), or a DataFrame per date and ticker that is non-zero on the
                dates to label, such as the result of `get_event_filter` with other settings. Defaults to None,
                which labels every date.
            side (str | pd.DataFrame | None, optional): The side of the bets for meta-labels: the
                name of a signal of `get_signals`, such as "Moving Average Crossover" or "MACD
                Crossover", whose crossovers are the bets, or a DataFrame per date and ticker of
                1 (long), -1 (short) and 0 (no bet). Only the dates with a bet are labeled.
                Defaults to None, which gives the triple-barrier labels themselves.
            vertical_barrier_zero (bool, optional): Whether a date whose vertical barrier is
                reached first is labeled 0 rather than with the sign of its return. Defaults
                to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Label, Return (of the bet for meta-labels), Barrier (1 upper, -1
            lower, 0 vertical), Holding Period (the periods to the barrier) and Target (the
            volatility) per date, with the Side for meta-labels, a column per measure and ticker.

        Notes:
        - The holding period counts each ticker's own trading days. A date whose barriers are not
          reached before the end of the data and whose vertical barrier lies beyond it has no
          label yet (NaN), so the last `horizon` dates are mostly empty.
        - The volatility is that of each period's return. López de Prado's Snippet 3.1 measures
          the return over the bar a calendar day earlier, written for intraday bars.
        - The vertical barrier lies `horizon` periods ahead, where López de Prado's Snippet 3.4
          places it a number of calendar days ahead. Without a side the two multiples apply as
          given, where Snippet 3.3 takes the profit-taking multiple for both barriers. The label
          is that of the barrier reached first. The return of a short bet is minus the price
          return, as in López de Prado's Snippet 3.7.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2024-01-01",
            end_date="2025-12-31",
        )

        labels = toolkit.technicals.get_triple_barrier_labels()

        labels.xs("AAPL", level=1, axis="columns").dropna().tail()
        ```

        Which returns:

        | Date       |   Label |   Return |   Barrier |   Holding Period |   Target |
        |:-----------|--------:|---------:|----------:|-----------------:|---------:|
        | 2025-12-10 |      -1 |  -0.0168 |        -1 |                3 |   0.0143 |
        | 2025-12-11 |      -1 |  -0.0223 |        -1 |                4 |   0.0142 |
        | 2025-12-12 |      -1 |  -0.015  |        -1 |                1 |   0.014  |
        | 2025-12-15 |      -1 |  -0.0038 |         0 |               10 |   0.0141 |
        | 2025-12-16 |      -1 |  -0.01   |         0 |               10 |   0.014  |
        """
        self._check_arguments(period, close_column, horizon=horizon, span=span)
        closes = self._close_prices(period, close_column, traded_only=True)

        if isinstance(events, str):
            events = self.get_event_filter(
                method=events, period=period, close_column=close_column
            )
            if events.empty:
                raise ValueError("events must be 'cusum', 'z_score' or a DataFrame.")
        if isinstance(side, str):
            signals = self.get_signals(period=period, close_column=close_column)
            if side not in signals.columns.get_level_values(0):
                raise ValueError(
                    f"side must be a signal of get_signals, such as 'Moving Average Crossover', "
                    f"or a DataFrame, not {side!r}."
                )
            side = signals[side]

        labels = _per_ticker(
            closes,
            functools.partial(
                labeling_model.get_volatility_triple_barrier_labels,
                span=span,
                horizon=horizon,
                profit_taking=profit_taking,
                stop_loss=stop_loss,
                events=events,
                side=side,
                vertical_barrier_zero=vertical_barrier_zero,
                minimum_target=minimum_target,
            ),
        ).loc[self._start_date : self._end_date]

        return apply_rounding(
            labels, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_fixed_horizon_labels(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        horizon: int = 1,
        threshold: float = 0.0,
        volatility_scaled: bool = False,
        span: int = 100,
        relative_to: str | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the fixed-horizon labels: the sign of the return over the next `horizon` periods, 0 when it stays
        within the threshold, the most common labels of financial machine learning.

        The return can be the ticker's own or relative: its excess over the mean or median return of the tickers (a
        cross-sectional label, for which only the ranking matters) or over the Benchmark.

        The formula is as follows:

        - r_t = p_t+horizon / p_t - 1
        - x_t = r_t, or r_t minus the mean or median r_t of the tickers, or minus the Benchmark's r_t
        - tau_t = threshold, or threshold x sigma_t x sqrt(horizon) when volatility scaled
        - Label = 1 when x_t > tau_t, -1 when x_t < -tau_t, else 0

        Labels look forward by construction: the label of a date is the return of the `horizon`
        periods after it. The volatility of a scaled threshold uses the returns up to the date
        only. López de Prado points out that a fixed threshold ignores the volatility, so that
        quiet periods are labeled 0 and volatile ones rarely, which the scaled threshold and the
        triple-barrier labels (`get_triple_barrier_labels`) address.

        For more information, see the following books:

        - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, section 3.2.
        - López de Prado, M. (2020). "Machine Learning for Asset Managers." Cambridge University
          Press, section 5.2.

        Also known as: fixed-time horizon labels, fixed horizon method, return labels, excess
        return labels, relative labels.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            horizon (int, optional): The number of periods ahead. Defaults to 1.
            threshold (float, optional): The threshold of the 0 label, a return, or a number of
                volatilities when volatility scaled. Defaults to 0.0.
            volatility_scaled (bool, optional): Whether the threshold is a multiple of the
                volatility over the horizon, sigma_t x sqrt(horizon). Defaults to False.
            span (int, optional): The span of the exponentially weighted volatility of a scaled
                threshold. Defaults to 100.
            relative_to (str | None, optional): What the return is measured against. Options are: "mean" (the
                excess over the mean return of the tickers), "median" (the excess over the median return of the
                tickers), "benchmark" (the excess over the Benchmark). Defaults to None, which labels the return
                itself.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Return (the excess return when relative), Label and Holding Period
            (the horizon) per date, a column per measure and ticker.

        Notes:
        - The horizon counts each ticker's own trading days, and the last `horizon` dates have no
          label.
        - A relative label leaves out the Benchmark (and a Portfolio) as tickers; the mean or
          median is that of the tickers with a return on the date.
        - The volatility of a scaled threshold is the one known at the date, not that of the
          forward returns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "NVDA", "AMZN"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2024-01-01",
            end_date="2025-12-31",
        )

        labels = toolkit.technicals.get_fixed_horizon_labels(horizon=5, relative_to="mean")

        labels.dropna()["Label"].tail()
        ```

        Which returns:

        | Date       |   AAPL |   MSFT |   NVDA |   AMZN |
        |:-----------|-------:|-------:|-------:|-------:|
        | 2025-12-17 |     -1 |     -1 |      1 |      1 |
        | 2025-12-18 |     -1 |     -1 |      1 |     -1 |
        | 2025-12-19 |     -1 |     -1 |      1 |      1 |
        | 2025-12-22 |     -1 |     -1 |      1 |      1 |
        | 2025-12-23 |      1 |      1 |     -1 |      1 |
        """
        self._check_arguments(period, close_column, horizon=horizon, span=span)
        check_option("relative_to", relative_to, (None, "mean", "median", "benchmark"))
        closes = self._close_prices(period, close_column, traded_only=True)
        forward_returns = apply_on_own_closes(
            closes, lambda prices: labeling_model.get_forward_returns(prices, horizon)
        )
        threshold_values: float | pd.DataFrame = threshold
        if volatility_scaled:
            threshold_values = (
                threshold
                * np.sqrt(horizon)
                * apply_on_own_closes(
                    closes,
                    lambda prices: labeling_model.get_daily_volatility(prices, span),
                )
            )

        benchmark_returns = None
        if relative_to is not None:
            if relative_to == "benchmark":
                if "Benchmark" not in forward_returns:
                    raise ValueError(
                        "relative_to='benchmark' needs the Benchmark of the Toolkit."
                    )
                benchmark_returns = forward_returns["Benchmark"]
            tickers = [
                ticker
                for ticker in forward_returns.columns
                if ticker not in NON_TICKER_COLUMNS
            ]
            forward_returns = forward_returns[tickers]
            if isinstance(threshold_values, pd.DataFrame):
                threshold_values = threshold_values[tickers]

        labels = labeling_model.get_fixed_horizon_labels(
            forward_returns, threshold_values, relative_to, benchmark_returns
        )
        holding_period = pd.DataFrame(
            float(horizon), index=labels.index, columns=forward_returns.columns
        ).where(labels["Return"].notna())
        labels = pd.concat(
            [labels, pd.concat({"Holding Period": holding_period}, axis=1)], axis=1
        ).loc[self._start_date : self._end_date]

        return apply_rounding(
            labels, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_trend_scanning_labels(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        minimum_window: int = 5,
        maximum_window: int = 20,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate López de Prado's trend-scanning labels: the sign of the most significant linear trend in the
        prices after every date.

        For every date it fits a linear trend to the prices of each forward window from `minimum_window` to
        `maximum_window` periods and takes the window whose slope is the most significant. The label is the sign of
        that slope and its t-value the strength of the trend. Unlike a fixed horizon, the method lets the data
        decide how long the trend after a date lasts, and labels the dates without a clear trend with a small
        t-value rather than a forced sign.

        The formula is as follows:

        - for L = minimum_window..maximum_window: OLS of p_t+l = b0 + b1 x l + e, l = 0..L-1
        - t(b1) = b1 / sqrt(SSR / (L - 2) / SUM((l - mean(l))^2))
        - L* = the L with the largest |t(b1)|
        - Label = sign(t(b1) at L*), t-Value = t(b1) at L*, Holding Period = L* - 1 and
          Return = p_t+L*-1 / p_t - 1

        Labels look forward by construction: the label of a date describes the prices of up to
        `maximum_window` periods after it.

        For more information, see the following book:

        - López de Prado, M. (2020). "Machine Learning for Asset Managers." Cambridge University
          Press, section 5.4 (Snippets 5.1 and 5.2).

        Referenced against:
        - RiskLabAI, find_trend_using_trend_scanning: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/data/labeling/financial_labels.py#L51

        Also known as: trend scanning, trend-scanning method, t-value labels.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            minimum_window (int, optional): The shortest forward window in prices, at least 3.
                Defaults to 5.
            maximum_window (int, optional): The longest forward window in prices, at least the
                minimum window. A maximum window longer than the data leaves every date without a
                label, which is logged. Defaults to 20.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Label, t-Value, Return and Holding Period per date, a column per
            measure and ticker.

        Notes:
        - The windows count each ticker's own trading days. A date gets a label once all its
          windows are complete, so the last `maximum_window - 1` dates have none.
        - The trend is fitted on the prices, as in López de Prado's Snippet 5.1; a tie between
          windows goes to the shortest.
        - The windows run up to and including `maximum_window`; the span of López de Prado's
          Snippet 5.2 is a Python range, which leaves its end out.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2024-01-01",
            end_date="2025-12-31",
        )

        labels = toolkit.technicals.get_trend_scanning_labels()

        labels.xs("MSFT", level=1, axis="columns").dropna().tail()
        ```

        Which returns:

        | Date       |   Label |   t-Value |   Return |   Holding Period |
        |:-----------|--------:|----------:|---------:|-----------------:|
        | 2025-11-26 |      -1 |   -2.9557 |  -0.0193 |               14 |
        | 2025-11-28 |      -1 |   -3.0081 |  -0.0323 |               13 |
        | 2025-12-01 |      -1 |   -2.352  |  -0.0218 |               12 |
        | 2025-12-02 |      -1 |   -2.0796 |  -0.0283 |               11 |
        | 2025-12-03 |       1 |    7.1482 |   0.0299 |                4 |
        """
        self._check_arguments(
            period,
            close_column,
            minimum_window=minimum_window,
            maximum_window=maximum_window,
        )
        validate_window("minimum_window", minimum_window, 3)
        validate_window("maximum_window", maximum_window, minimum_window)
        closes = self._close_prices(period, close_column, traded_only=True)
        labels = _per_ticker(
            closes,
            lambda prices: labeling_model.get_trend_scanning_labels(
                prices, minimum_window, maximum_window
            ),
        ).loc[self._start_date : self._end_date]

        return apply_rounding(
            labels, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_tail_set_labels(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        horizon: int = 1,
        quantile: float = 0.1,
        volatility_adjusted: bool = True,
        span: int = 100,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the tail-set labels: per date, rank the tickers on their volatility-adjusted return over the next
        `horizon` periods and label the top `quantile` 1, the bottom `quantile` -1 and the rest 0.

        A model trained on them learns to tell the best from the worst performers of a universe, the question a
        long-short portfolio asks, without being distracted by the noisy middle. The labels need a universe of
        tickers, such as the constituents of an index.

        The formula is as follows:

        - Score_t = (p_t+horizon / p_t - 1) / sigma_t, sigma_t the EWM volatility up to t (or
          the return itself without volatility adjustment)
        - q = (rank - 1) / (N - 1), the percentile rank among the N tickers with a score
        - Label = 1 when q >= 1 - quantile, -1 when q <= quantile, else 0

        Labels look forward by construction: the score of a date is the return of the `horizon`
        periods after it, divided by the volatility known at the date.

        The tail sets follow Huerta, Corbacho and Elkan (2013), who label the stocks with the
        highest and lowest volatility-adjusted returns and leave the middle out.

        For more information, see the following sources:

        - Huerta, R., Corbacho, F. and Elkan, C. (2013). "Nonlinear Support Vector Machines Can
          Systematically Identify Stocks with High and Low Future Returns." Algorithmic Finance,
          2(1), 45-58.
        - López de Prado, M. (2020). "Machine Learning for Asset Managers." Cambridge University
          Press, chapter 5.

        Also known as: tail sets, tail set labels, cross-sectional quantile labels.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            horizon (int, optional): The number of periods of the forward return. Defaults to 1.
            quantile (float, optional): The share of the tickers in each tail, up to 0.5. Defaults
                to 0.1.
            volatility_adjusted (bool, optional): Whether the returns are divided by the
                volatility. Defaults to True.
            span (int, optional): The span of the exponentially weighted volatility. Defaults to 100.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Score and Label per date, a column per measure and ticker.

        Notes:
        - The Benchmark (and a Portfolio) are left out of the ranking; it needs at least two
          tickers, and with fewer than 1 / quantile tickers each tail holds a single ticker.
        - Tickers with an equal score share their average rank. A ticker exactly on the
          boundary of a tail belongs to the tail on both sides.
        - The forward return is labeled, adjusted by the volatility known at the date.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "ORCL", "AMD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2024-01-01",
            end_date="2025-12-31",
        )

        labels = toolkit.technicals.get_tail_set_labels()

        labels["Label"].dropna().tail()
        ```

        Which returns:

        | Date       |   AAPL |   MSFT |   NVDA |   AMZN |   GOOGL |   META |   TSLA |   AVGO |   ORCL |   AMD |
        |:-----------|-------:|-------:|-------:|-------:|--------:|-------:|-------:|-------:|-------:|------:|
        | 2025-12-23 |      1 |      0 |     -1 |      0 |       0 |      0 |      0 |      0 |      0 |     0 |
        | 2025-12-24 |      0 |      0 |      1 |      0 |       0 |      0 |     -1 |      0 |      0 |     0 |
        | 2025-12-26 |      1 |      0 |      0 |      0 |       0 |      0 |     -1 |      0 |      0 |     0 |
        | 2025-12-29 |      0 |      0 |      0 |      0 |       0 |      1 |     -1 |      0 |      0 |     0 |
        | 2025-12-30 |      0 |     -1 |      0 |      0 |       0 |      0 |      0 |      0 |      0 |     1 |
        """
        self._check_arguments(period, close_column, horizon=horizon, span=span)
        closes = self._close_prices(period, close_column, traded_only=True)
        closes = closes[
            [ticker for ticker in closes.columns if ticker not in NON_TICKER_COLUMNS]
        ]
        if closes.shape[1] < 2:  # noqa: PLR2004
            raise ValueError("The tail-set labels need at least two tickers to rank.")

        scores = apply_on_own_closes(
            closes,
            lambda prices: labeling_model.get_forward_returns(prices, horizon),
        )
        if volatility_adjusted:
            scores = scores / apply_on_own_closes(
                closes,
                lambda prices: labeling_model.get_daily_volatility(prices, span),
            )
        labels = pd.concat(
            {
                "Score": scores,
                "Label": labeling_model.get_tail_set_labels(scores, quantile),
            },
            axis=1,
        ).loc[self._start_date : self._end_date]

        return apply_rounding(
            labels, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_matrix_flag_labels(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 40,
        template: str = "leigh_bull",
        threshold: float | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the matrix flag labels: how closely the prices of the last `window` periods fit a bull (or bear)
        flag, a sharp move followed by a consolidation that chartists read as the pause before the move continues.

        The window's prices are laid out as a 10 x 10 grid of their price deciles (rows) over ten consecutive parts
        of the window (columns), and a template of weights rewards the cells the flag passes through and penalises
        the others. The score is the sum of the weighted grid; with a threshold the label is 1 from that score.

        The templates are those of Leigh, Modani, Purvis and Roberts (2002), "leigh_bull", and of
        Cervelló-Royo, Guijarro and Michniuk (2015), "cervello_royo_bull", who proposed a flag
        whose weights are concentrated in the consolidation. The bear templates are the bull
        templates upside down, the highest decile becoming the lowest.

        The formula is as follows:

        - D[d, j] = the share of the prices of part j of the window in price decile d of the
          whole window, the highest decile in the top row
        - Score = SUM(W x D), W the template
        - Label = 1 when Score >= threshold, else 0

        Unlike the other labels, the score looks back: it uses the prices of the window up to and
        including the date, so a flag is known at the close of the date.

        For more information, see the following papers:

        - Leigh, W., Modani, N., Purvis, R. and Roberts, T. (2002). "Stock Market Trading Rule
          Discovery Using Technical Charting Heuristics." Expert Systems with Applications,
          23(2), 155-159.
        - Cervelló-Royo, R., Guijarro, F. and Michniuk, K. (2015). "Stock Market Trading Rule
          Based on Pattern Recognition and Technical Analysis: Forecasting the DJIA Index with
          Intraday Data." Expert Systems with Applications, 42(14), 5963-5975.

        Also known as: matrix flags, bull flag, bear flag, flag pattern, template matching.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The number of prices of the window, at least 10. Defaults to 40.
            template (str, optional): The template of weights. Options are: "leigh_bull" (the bull flag of
                Leigh, Modani, Purvis and Roberts, 2002), "leigh_bear" (the Leigh bull flag upside down),
                "cervello_royo_bull" (the bull flag of Cervelló-Royo, Guijarro and Michniuk, 2015),
                "cervello_royo_bear" (the Cervelló-Royo bull flag upside down). Defaults to "leigh_bull".
            threshold (float | None, optional): The score from which the label is 1. Defaults to
                None, which returns the score only.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Score, and the Label with a threshold, per date, a column per
            measure and ticker.

        Notes:
        - The window counts each ticker's own trading days; a window that is not a multiple of
          ten gives its first parts one price more.
        - A price on the boundary of two deciles belongs to the lower one.
        - The window ends at the date, and the scores below the threshold are labeled 0, since
          a missing flag is no sign of the opposite move.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2024-01-01",
            end_date="2025-12-31",
        )

        flags = toolkit.technicals.get_matrix_flag_labels(threshold=2.5)

        flags.xs("MSFT", level=1, axis="columns").tail()
        ```

        Which returns:

        | Date       |   Score |   Label |
        |:-----------|--------:|--------:|
        | 2025-12-24 |   3.25  |       1 |
        | 2025-12-26 |   3.75  |       1 |
        | 2025-12-29 |   4.125 |       1 |
        | 2025-12-30 |   4.75  |       1 |
        | 2025-12-31 |   4.125 |       1 |
        """
        self._check_arguments(period, close_column, window=window)
        check_option("template", template, labeling_model.MATRIX_FLAG_TEMPLATES)
        closes = self._close_prices(period, close_column, traded_only=True)
        flags = _per_ticker(
            closes,
            lambda prices: labeling_model.get_matrix_flag_labels(
                prices, window, template, threshold
            ),
        ).loc[self._start_date : self._end_date]

        return apply_rounding(
            flags, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_label_uniqueness(
        self,
        labels: pd.DataFrame | None = None,
        period: str = "daily",
        close_column: str = "Adj Close",
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the concurrency and the average uniqueness of labels: how many labels span each date and how much
        of its span each label shares with others.

        Labels that span several periods overlap: the triple-barrier labels of two consecutive days share most of
        their prices, so they are not independent observations. The concurrency of a date is the number of labels
        whose span includes it, and the average uniqueness of a label is the mean of one over the concurrency across
        its span: 1 for a label that overlaps no other and 1/10 for a label sharing its whole span with nine others.
        López de Prado uses it to weigh the labels, to bootstrap them in proportion to their uniqueness (sequential
        bootstrap) and to set the number of samples a bagging classifier draws.

        The formula is as follows:

        - c_t = SUM_i 1[t0_i <= t <= t1_i], t1_i the date `Holding Period` periods after t0_i
        - u_i = mean of 1 / c_t over t0_i <= t <= t1_i

        Both look forward by construction, as the spans of the labels do: the uniqueness of a
        label depends on the labels after it.

        For more information, see the following book:

        - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, sections
          4.3 and 4.4 (Snippets 4.1 and 4.2).

        Referenced against:
        - RiskLabAI, calculate_average_uniqueness: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/data/weights/sample_weights.py#L71

        Also known as: label concurrency, number of concurrent events, average uniqueness,
        sample uniqueness, overlapping labels.

        Args:
            labels (pd.DataFrame | None, optional): Labels with a Holding Period per date and
                ticker, such as the result of `get_triple_barrier_labels`,
                `get_fixed_horizon_labels` or `get_trend_scanning_labels`. Defaults to None, which
                uses the triple-barrier labels with their defaults.
            period (str, optional): The time period of the labels. Options are: "intraday" (intraday prices),
                "daily" (daily prices), "weekly" (weekly prices), "monthly" (monthly prices),
                "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Concurrency per date and the Average Uniqueness per label date, a
            column per measure and ticker.

        Notes:
        - The spans count each ticker's own trading days and include both the label's date and
          the date of its barrier, as in López de Prado's Snippet 4.1.
        - Dates without a label (such as the last ones, whose outcome is not known yet) count
          for no span.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2024-01-01",
            end_date="2025-12-31",
        )

        uniqueness = toolkit.technicals.get_label_uniqueness()

        uniqueness.xs("AAPL", level=1, axis="columns").dropna().tail()
        ```

        Which returns:

        | Date       |   Concurrency |   Average Uniqueness |
        |:-----------|--------------:|---------------------:|
        | 2025-12-10 |             5 |               0.1586 |
        | 2025-12-11 |             6 |               0.1669 |
        | 2025-12-12 |             7 |               0.1339 |
        | 2025-12-15 |             8 |               0.4114 |
        | 2025-12-16 |             5 |               0.4909 |
        """
        self._check_arguments(period, close_column)
        closes, holding_periods = self._get_labels_or_default(
            labels, period, close_column
        )

        uniqueness = _per_ticker(
            closes,
            lambda prices: labeling_model.get_label_uniqueness(
                holding_periods[prices.name].reindex(prices.index)
            ),
        ).loc[self._start_date : self._end_date]

        return apply_rounding(
            uniqueness, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_sample_weights(
        self,
        method: Literal["return", "time_decay"] = "return",
        labels: pd.DataFrame | None = None,
        decay: float = 1.0,
        period: str = "daily",
        close_column: str = "Adj Close",
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the sample weights of labels for training a model: how much each label should count, by its
        uniqueness, its return or its age, as López de Prado proposes for labels that overlap.

        - Return attribution: the weight by return attribution. Each period's log return is shared among
          the labels whose span includes it, and a label's weight is the absolute sum of its
          shares. Labels whose spans saw large moves, and that share them with few other labels,
          count most, while a label over a flat stretch counts little. The weights add up to the
          number of labels.
        - Time decay: the weight by time decay. The weights fall linearly with the cumulative
          average uniqueness from 1 for the newest label to `decay` for the oldest, so that
          older observations count less, measured in unique information rather than in dates.

        The formula is as follows, per method:

        - Return attribution: r_t = ln(p_t / p_t-1), w~_i = |SUM_{t0_i < t <= t1_i} r_t / c_t| and
          w_i = w~_i x N / SUM_j w~_j, with c_t the concurrency
        - Time decay: x_i the cumulative average uniqueness up to label i and X the total,
          slope = (1 - decay) / X when decay >= 0, else 1 / ((decay + 1) x X), and
          w_i = max(0, 1 - slope x X + slope x x_i)

        The weights look forward by construction, as the spans of the labels do.

        For more information, see the following book:

        - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, sections
          4.6 and 4.7 (Snippets 4.10 and 4.11).

        Referenced against:
        - RiskLabAI, sample_weight_absolute_return_meta_labeling: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/data/weights/sample_weights.py#L200
        - RiskLabAI, calculate_time_decay: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/data/weights/sample_weights.py#L279

        Also known as: sample weights, return attribution, time decay, time-decay weights.

        Args:
            method (str, optional): The weighting. Options are: "return" (return attribution),
                "time_decay" (time decay). Defaults to "return".
            labels (pd.DataFrame | None, optional): Labels with a Holding Period per date and
                ticker, such as the result of `get_triple_barrier_labels`,
                `get_fixed_horizon_labels` or `get_trend_scanning_labels`. Defaults to None, which
                uses the triple-barrier labels with their defaults.
            decay (float, optional): The weight of the oldest label of "time_decay", in (-1, 1]: 1
                for no decay, 0 for a decay to zero and a negative value to give the oldest share
                (-decay) of the labels no weight. Defaults to 1.0.
            period (str, optional): The time period of the labels. Options are: "intraday" (intraday prices),
                "daily" (daily prices), "weekly" (weekly prices), "monthly" (monthly prices),
                "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The weight of the label at each date, a column per ticker.

        Notes:
        - The weights are calculated per ticker, over the labels from the start date of the
          Toolkit, so each ticker's weights add up to its number of labels ("return").
        - López de Prado's Snippet 4.10 also adds the return into the label's date,
          the period before the label starts; the sum here starts after it, with the returns the
          label earns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2024-01-01",
            end_date="2025-12-31",
        )

        weights = toolkit.technicals.get_sample_weights()

        weights.dropna().tail()
        ```

        Which returns:

        | Date       |   AAPL |   MSFT |   Benchmark |
        |:-----------|-------:|-------:|------------:|
        | 2025-12-10 | 0.3814 | 1.1291 |      0.3261 |
        | 2025-12-11 | 0.5911 | 1.0596 |      0.4031 |
        | 2025-12-12 | 0.326  | 0.7975 |      0.6633 |
        | 2025-12-15 | 0.1058 | 0.7868 |      0.5882 |
        | 2025-12-16 | 0.7306 | 0.6239 |      0.4716 |
        """
        self._check_arguments(period, close_column)
        check_option("method", method, ("return", "time_decay"))
        if method == "time_decay" and not -1 < decay <= 1:
            raise ValueError(f"The decay must lie in (-1, 1], not {decay}.")
        closes, holding_periods = self._get_labels_or_default(
            labels, period, close_column
        )

        weights = apply_on_own_closes(
            closes,
            lambda prices: labeling_model.get_sample_weights(
                prices,
                holding_periods[prices.name].reindex(prices.index),
                method,
                decay,
            ),
        ).loc[self._start_date : self._end_date]

        return apply_rounding(
            weights, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_bet_sizes(
        self,
        method: Literal["budget", "reserve", "probability"] = "budget",
        labels: pd.DataFrame | None = None,
        step_size: float | None = None,
        minimum_labels: int = 20,
        runs: int = 100,
        seed: int = 0,
        period: str = "daily",
        close_column: str = "Adj Close",
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate López de Prado's bet size for meta-labelled bets: how much of the budget to put on each date, from
        -1 (fully short) to 1 (fully long).

        It follows from the sides of the bets a primary model takes and, for the probability bet size, how often its
        bets have paid off. A model that only says which side to take does not say how much to bet; sizing the bets
        keeps a strategy from betting the same on a weak signal as on a strong one.

        - Budget bet size: the open long bets as a share of the most long bets ever open, minus the same
          share of the short bets, a size that spreads the budget over the bets the strategy
          usually has open.
        - Reserve bet size: the net number of open bets (long minus short) against how many the
          strategy usually has open, through a mixture of two normal distributions fitted to it
          with the EF3M algorithm (López de Prado and Foreman, 2014). An unusually large net
          count gives a bet close to 1 and a usual one a small bet, keeping a reserve for stronger
          signals.
        - Probability bet size: every bet sized by the probability that it pays off, taken as the hit
          rate of the bets closed before it was placed (the share whose meta-label is 1), and the
          sizes averaged over the bets still open. A hit rate of 50% or less gives no bet.

        The formula is as follows, per method:

        - Long Bets_t and Short Bets_t, the bets open on t: placed on or before t and closing
          after t, `Holding Period` periods after they were placed
        - Budget bet size: m_t = Long Bets_t / max(Long Bets) - Short Bets_t / max(Short Bets)
        - Reserve bet size: c_t = Long Bets_t - Short Bets_t, F the mixture of two normal distributions
          that matches the first five moments of c_t on the dates bets are placed, and
          m_t = (F(c_t) - F(0)) / (1 - F(0)) when c_t >= 0, (F(c_t) - F(0)) / F(0) otherwise
        - Probability bet size: p = the hit rate known when bet i is placed, z_i = (p - 0.5) /
          SQRT(p x (1 - p)) and m_i = side_i x (2 x N(z_i) - 1) when p > 0.5, else 0; m_t = the
          average of m_i over the bets open on t (0 without an open bet)
        - with a `step_size`: m*_t = round(m_t / step_size) x step_size, capped at -1 and 1

        Where N is the standard normal distribution function. The bets come from the meta-labels:
        the Side is the side of a bet, the Holding Period the periods until it closes at a
        barrier and the Label (1 or 0) whether it made money.

        For more information, see the following book and paper:

        - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, chapter 10
          (Snippets 10.1 to 10.3).
        - López de Prado, M. and Foreman, M. (2014). "A mixture of two Gaussians approach to
          mathematical portfolio oversight: the EF3M algorithm." Quantitative Finance, 14(5),
          913-930.

        Referenced against:
        - RiskLabAI, probability_bet_size: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/backtest/bet_sizing.py#L33
        - RiskLabAI, avg_active_signals: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/backtest/bet_sizing.py#L166

        Also known as: bet sizing, position sizing from meta-labels, budget bet size, reserve bet
        size, EF3M, bet size from predicted probabilities.

        Args:
            method (str, optional): The bet size. Options are: "budget" (the budget bet size), "reserve" (the
                reserve bet size), "probability" (the probability bet size). Defaults to "budget".
            labels (pd.DataFrame | None, optional): Meta-labels with a Side, Holding Period and
                Label per date and ticker, such as the result of `get_triple_barrier_labels` with a
                side. Defaults to None, which uses the triple-barrier meta-labels of the CUSUM
                filter's events with the direction of each event as the side (a trend-following
                primary model) and the other defaults.
            step_size (float | None, optional): The step the bet sizes are rounded to, between 0
                and 1, so small changes of the size do not trade. Defaults to None, which keeps
                them unrounded.
            minimum_labels (int, optional): The number of closed bets the hit rate of
                "probability" needs before it sizes a bet; earlier bets are left out. Defaults
                to 20.
            runs (int, optional): The number of random starting points of the EF3M fit of
                "reserve". Defaults to 100.
            seed (int, optional): The seed of the starting points of the EF3M fit. Defaults to 0.
            period (str, optional): The time period of the labels. Options are: "intraday" (intraday prices),
                "daily" (daily prices), "weekly" (weekly prices), "monthly" (monthly prices),
                "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Long Bets, Short Bets and Bet Size per date, with the Hit Rate for
            "probability", a column per measure and ticker.

        Notes:
        - The bets are those whose barrier was reached; a bet still open at the end of the data
          has no Holding Period yet and is left out.
        - A bet counts from the date it is placed up to, but not including, the date it closes.
          The "budget" maximum and the "reserve" fit cover each ticker's whole sample, as in
          López de Prado, so like the labels themselves they describe the bets of the sample
          rather than a size known on the date. The hit rate of "probability" uses only the bets
          closed by the date.
        - The "reserve" fit solves the five moment equations of EF3M with Powell's hybrid method
          from EF3M's random starting points rather than with EF3M's fixed-point iteration, whose
          solution is often a repelling point it cannot reach, so where it stops differs per
          seed. Here every seed reaches the same exact solution, so `seed` only matters when
          the moments allow several.
        - "probability" sizes the bets with the hit rate of the past bets as the predicted
          probability, a simple secondary model; `bet_sizing_model.get_probability_bet_size`
          takes the probabilities of a classifier of your own.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2024-01-01",
            end_date="2025-12-31",
        )

        bet_sizes = toolkit.technicals.get_bet_sizes(method="reserve")

        bet_sizes.xs("AAPL", level=1, axis="columns").tail()
        ```

        Which returns:

        | Date       |   Long Bets |   Short Bets |   Bet Size |
        |:-----------|------------:|-------------:|-----------:|
        | 2025-12-24 |           0 |            1 |     -0.534 |
        | 2025-12-26 |           0 |            1 |     -0.534 |
        | 2025-12-29 |           0 |            1 |     -0.534 |
        | 2025-12-30 |           0 |            0 |      0     |
        | 2025-12-31 |           0 |            0 |      0     |
        """
        self._check_arguments(period, close_column)
        check_option("method", method, ("budget", "reserve", "probability"))
        if step_size is not None and not 0 < step_size <= 1:
            raise ValueError(f"step_size must lie in (0, 1], not {step_size}.")

        closes = self._close_prices(period, close_column, traded_only=True)
        if labels is None:
            events = self.get_event_filter(
                method="cusum", period=period, close_column=close_column
            )
            labels = self.get_triple_barrier_labels(
                period=period,
                close_column=close_column,
                events=events,
                side=events,
                rounding=15,
            )
        measures = labels.columns.get_level_values(0)
        needed = ["Side", "Holding Period"] + (
            ["Label"] if method == "probability" else []
        )
        if any(measure not in measures for measure in needed):
            raise ValueError(
                f"The labels need {', '.join(needed)} columns per ticker, such as the result of "
                "get_triple_barrier_labels with a side (meta-labels)."
            )

        bet_sizes = _per_ticker(
            closes,
            functools.partial(
                bet_sizing_model.get_bet_sizes,
                labels=labels,
                method=method,
                step_size=step_size,
                minimum_labels=minimum_labels,
                runs=runs,
                seed=seed,
            ),
        ).loc[self._start_date : self._end_date]

        return apply_rounding(
            bet_sizes, rounding if rounding is not None else self._rounding
        )
