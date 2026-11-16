"""Option Market Metrics Module"""

__docformat__ = "google"

import numpy as np
import pandas as pd

from financetoolkit.options import options_model
from financetoolkit.options.base.base_controller import OptionsBase, _days_to_expiration
from financetoolkit.options.market import market_metrics_model
from financetoolkit.utilities import logger_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import (
    apply_rounding,
    to_period_index,
)

# ruff: noqa: E501

logger = logger_model.get_logger()

# What the market metrics take from the historical data while the option chains are live.
_CHAIN_INPUTS_FROM_HISTORY = (
    "the risk free rate, the dividend yield and, when a chain quotes no price of "
    "the underlying, the stock price come from that date"
)


@filter_shown_columns
@track_sources
class MarketOptions(OptionsBase):
    """
    Metrics from the option chains: the put-call ratio, max pain, gamma exposure, the implied volatility term structure and skew, the implied move, the model-free implied volatility and the volatility risk premium.
    """

    def _expiries(
        self,
        ticker: str,
        expiration_dates: list[str] | None,
        number_of_expirations: int,
    ) -> list[str]:
        """The expiries of a ticker to use: as given, otherwise the nearest ones."""
        available = options_model.get_option_expiry_dates(ticker)
        if expiration_dates is not None:
            invalid = [date for date in expiration_dates if date not in available]
            if invalid:
                raise ValueError(
                    f"The expiration date(s) {', '.join(invalid)} are not available for "
                    f"{ticker}. Choose from {', '.join(available)}"
                )
            return list(expiration_dates)
        upcoming = [date for date in available if _days_to_expiration(date) > 0]
        return upcoming[:number_of_expirations]

    def _warn_if_history_is_stale(self, purpose: str) -> None:
        """
        Warn when the historical data ends more than 5 days before today. The option chains
        are always live, so history that ends earlier (an end_date in the past) mixes old
        prices, rates and dividends with today's option quotes.
        """
        last_date = to_period_index(self._daily_historical.index)[-1].to_timestamp()
        lag = (pd.Timestamp.today().normalize() - last_date).days
        if lag > 5:  # noqa: PLR2004
            logger.warning(
                "The historical data ends on %s, %d days before today, while the option "
                "chains are live: %s. Leave end_date at today for these metrics.",
                last_date.strftime("%Y-%m-%d"),
                lag,
                purpose,
            )

    def _expiry_inputs(
        self, ticker: str, expiration_date: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, float, float, float, float]:
        """
        The calls, puts, stock price, time to expiration, risk free rate and dividend yield.
        The stock price is the one quoted with the chain when available (see _spot_price).
        """
        calls = options_model.get_option_chains([ticker], expiration_date, False)
        puts = options_model.get_option_chains([ticker], expiration_date, True)
        calls = calls.loc[ticker] if not calls.empty else calls
        puts = puts.loc[ticker] if not puts.empty else puts

        # Outside market hours Yahoo Finance often returns chains without bids, while Cboe's
        # delayed quotes keep the last ones; the better quoted chain is used.
        def quoted(chain: pd.DataFrame) -> int:
            return int((pd.to_numeric(chain.get("Bid"), errors="coerce") > 0).sum())

        if quoted(calls) + quoted(puts) < (len(calls) + len(puts)) / 2:
            cboe_calls = options_model._get_cboe_option_chain(  # noqa: SLF001
                ticker, expiration_date, False
            )
            cboe_puts = options_model._get_cboe_option_chain(  # noqa: SLF001
                ticker, expiration_date, True
            )
            if quoted(cboe_calls) + quoted(cboe_puts) > quoted(calls) + quoted(puts):
                calls, puts = cboe_calls, cboe_puts
        stock_price = self._spot_price(ticker, calls, puts)
        time_to_expiration = max(_days_to_expiration(expiration_date), 0.5) / 365
        risk_free_rate = float(self._risk_free_rate.dropna().iloc[-1])
        dividend_yield = (
            float(self._dividend_yield[ticker].iloc[-1])
            if not self._dividend_yield[ticker].empty
            else 0.0
        )
        if np.isnan(dividend_yield):
            dividend_yield = 0.0
        return (
            calls,
            puts,
            stock_price,
            time_to_expiration,
            risk_free_rate,
            dividend_yield,
        )

    def _per_expiry(
        self,
        calculate,
        expiration_dates: list[str] | dict[str, list[str]] | None,
        number_of_expirations: int,
        rounding: int | None,
    ) -> pd.DataFrame:
        """
        Run a calculation on every expiry of every ticker, rows (ticker, expiration). The
        expiries are the same for every ticker, or given per ticker as a dictionary, which
        then also selects the tickers.
        """
        self._warn_if_history_is_stale(_CHAIN_INPUTS_FROM_HISTORY)
        per_ticker = (
            expiration_dates
            if isinstance(expiration_dates, dict)
            else dict.fromkeys(self._tickers, expiration_dates)
        )
        rows = {}
        for ticker, ticker_expiration_dates in per_ticker.items():
            for expiration_date in self._expiries(
                ticker, ticker_expiration_dates, number_of_expirations
            ):
                inputs = self._expiry_inputs(ticker, expiration_date)
                if inputs[0].empty or inputs[1].empty:
                    continue
                result = calculate(*inputs)
                rows[(ticker, expiration_date)] = pd.concat(
                    [
                        pd.Series(
                            {"Days to Expiration": _days_to_expiration(expiration_date)}
                        ),
                        result,
                    ]
                )

        if not rows:
            raise ValueError("No option chains are available for these tickers.")

        table = pd.DataFrame(rows).T.astype(float)
        table.index.names = ["Ticker", "Expiration"]
        return apply_rounding(
            table, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_put_call_ratio(
        self,
        expiration_dates: list[str] | None = None,
        number_of_expirations: int = 6,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the put-call ratio of every expiry, by volume and by open interest. The
        ratio is the number of puts divided by the number of calls: a ratio above one means
        more puts are traded or held than calls, which is usually read as hedging demand or
        bearish sentiment, and an unusually low ratio as complacency. The volume ratio
        reflects today's trading; the open interest ratio the positions that remain.

        Also known as: PCR, put/call ratio, options sentiment.

        Args:
            expiration_dates (list[str] | None, optional): The expiries to use. Defaults to
                None, the nearest `number_of_expirations` expiries.
            number_of_expirations (int, optional): The number of nearest expiries to use when
                no dates are given. Defaults to 6.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per ticker and expiry, and over all of them ("All"), the Days to
            Expiration, Call Volume, Put Volume, Volume Put-Call Ratio, Call Open Interest,
            Put Open Interest and Open Interest Put-Call Ratio.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_put_call_ratio(number_of_expirations=3)
        ```

        Which returns:

        |                        |   Days to Expiration |   Call Volume |   Put Volume |   Volume Put-Call Ratio |   Call Open Interest |   Put Open Interest |   Open Interest Put-Call Ratio |
        |:-----------------------|---------------------:|--------------:|-------------:|------------------------:|---------------------:|--------------------:|-------------------------------:|
        | ('AAPL', '2026-10-26') |                    2 |         90053 |        85187 |                  0.946  |                14688 |               22097 |                         1.5044 |
        | ('AAPL', '2026-10-28') |                    4 |         27330 |        21152 |                  0.7739 |                17505 |               10137 |                         0.5791 |
        | ('AAPL', '2026-10-30') |                    6 |         76540 |       134401 |                  1.756  |                81432 |               86739 |                         1.0652 |
        | ('AAPL', 'All')        |                  nan |        193923 |       240740 |                  1.2414 |               113625 |              118973 |                         1.0471 |
        | ('TSLA', '2026-10-26') |                    2 |        149535 |       129459 |                  0.8657 |                68890 |               34714 |                         0.5039 |
        | ('TSLA', '2026-10-28') |                    4 |         32602 |        24491 |                  0.7512 |                14542 |               15028 |                         1.0334 |
        | ('TSLA', '2026-10-30') |                    6 |        121542 |       101610 |                  0.836  |               104417 |               97818 |                         0.9368 |
        | ('TSLA', 'All')        |                  nan |        303679 |       255560 |                  0.8415 |               187849 |              147560 |                         0.7855 |
        """
        table = self._per_expiry(
            lambda calls, puts, *_: market_metrics_model.get_put_call_ratio(
                calls, puts
            ),
            expiration_dates,
            number_of_expirations,
            None,
        )
        # Each ticker's expiries are followed by an "All" row with the summed volume and
        # open interest and the ratios of those sums.
        parts = []
        for ticker in table.index.get_level_values(0).unique():
            total = table.loc[ticker].sum()
            total["Days to Expiration"] = np.nan
            for ratio, side in (
                ("Volume Put-Call Ratio", "Volume"),
                ("Open Interest Put-Call Ratio", "Open Interest"),
            ):
                calls = total[f"Call {side}"]
                total[ratio] = total[f"Put {side}"] / calls if calls else np.nan
            parts += [
                table.loc[[ticker]],
                pd.DataFrame(
                    [total], index=pd.MultiIndex.from_tuples([(ticker, "All")])
                ),
            ]
        combined = pd.concat(parts)
        combined.index.names = ["Ticker", "Expiration"]
        return apply_rounding(
            combined, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_max_pain(
        self,
        expiration_dates: list[str] | None = None,
        number_of_expirations: int = 6,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the max pain strike of every expiry: the expiry price at which the
        options outstanding would pay their holders the least, the sum over every call of
        open interest times max(price - strike, 0) and over every put of open interest
        times max(strike - price, 0) being smallest. Option writers, often market makers,
        profit most there, and some traders expect the price to drift towards it as the
        expiry nears; the evidence for that is mixed, so it is best read as a description
        of where open interest is concentrated.

        Also known as: max pain theory, option pain, maximum pain.

        Args:
            expiration_dates (list[str] | None, optional): The expiries to use. Defaults to
                None, the nearest `number_of_expirations` expiries.
            number_of_expirations (int, optional): The number of nearest expiries to use when
                no dates are given. Defaults to 6.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per ticker and expiry the Days to Expiration, Max Pain strike,
            Stock Price and the Distance to Max Pain (as a fraction of the price).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_max_pain(number_of_expirations=4)
        ```

        Which returns:

        |                        |   Days to Expiration |   Max Pain |   Stock Price |   Distance to Max Pain |
        |:-----------------------|---------------------:|-----------:|--------------:|-----------------------:|
        | ('AAPL', '2026-10-26') |                    2 |      345   |        347.73 |                -0.0079 |
        | ('AAPL', '2026-10-28') |                    4 |      345   |        347.73 |                -0.0079 |
        | ('AAPL', '2026-10-30') |                    6 |      342.5 |        347.73 |                -0.015  |
        | ('AAPL', '2026-11-06') |                   13 |      340   |        347.73 |                -0.0222 |
        """

        def calculate(calls, puts, stock_price, *_):
            max_pain = market_metrics_model.get_max_pain(calls, puts)
            return pd.Series(
                {
                    "Max Pain": max_pain,
                    "Stock Price": stock_price,
                    "Distance to Max Pain": max_pain / stock_price - 1,
                }
            )

        return self._per_expiry(
            calculate, expiration_dates, number_of_expirations, rounding
        )

    @handle_errors
    def get_gamma_exposure(
        self,
        expiration_dates: list[str] | None = None,
        number_of_expirations: int = 6,
        by_strike: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the dealer gamma exposure (GEX): the dollar value of shares market makers
        would have to buy or sell for every 1% move in the underlying to keep their option
        books delta-neutral. Following the common convention, dealers are assumed long the
        calls and short the puts that investors trade:

            - Gamma Exposure = Gamma x Open Interest x 100 x Price² x 0.01
            - positive for calls, negative for puts, summed over strikes and expiries

        With positive exposure dealers buy dips and sell rallies, which dampens volatility;
        with negative exposure they trade in the direction of the move and amplify it. The
        gamma flip point is the price at which the total exposure changes sign.

        For more information, see the following sources:

        - SqueezeMetrics (2017). "Gamma Exposure (GEX)." White paper.
        - Perfiliev, S. (2021). "How to Calculate Gamma Exposure (GEX) and Zero Gamma
          Level." https://perfiliev.com/blog/how-to-calculate-gamma-exposure-and-zero-gamma-level/

        Also known as: GEX, dealer gamma, gamma positioning, net gamma.

        Args:
            expiration_dates (list[str] | None, optional): The expiries to include. Defaults
                to None, the nearest `number_of_expirations` expiries.
            number_of_expirations (int, optional): The number of nearest expiries to include
                when no dates are given. Defaults to 6.
            by_strike (bool, optional): Whether to return the exposure per strike instead of
                the totals. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per ticker the Stock Price, Call, Put and Net Gamma Exposure (in
            dollars per 1% move) and the Gamma Flip Point; with by_strike=True the Call, Put
            and Net Gamma Exposure per ticker and strike.

        Notes:
        - The assumption about who holds which side is the main uncertainty: when
          investors sell calls (covered calls) the sign of that part flips.
        - Gamma uses each option's own implied volatility (Black-Scholes).
        - The stock price is the price of the underlying quoted with the live option chain
          (Yahoo Finance's regular market price or Cboe's current price), so both come from
          the same moment; when the chain quotes none, the traded Close of the last
          historical date is used. The risk free rate and dividend yield come from the
          historical data as well, so leave end_date at today: a warning is logged when the
          history ends more than 5 days before today.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["QQQ", "AAPL"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_gamma_exposure()
        ```

        Which returns:

        |      |   Stock Price |   Call Gamma Exposure |   Put Gamma Exposure |   Net Gamma Exposure |   Gamma Flip Point |
        |:-----|--------------:|----------------------:|---------------------:|---------------------:|-------------------:|
        | QQQ  |        704.4  |           1.83881e+09 |         -6.41458e+09 |         -4.57577e+09 |            727.993 |
        | AAPL |        347.73 |           7.48337e+08 |         -4.80606e+08 |          2.67731e+08 |            342.412 |
        """
        self._warn_if_history_is_stale(_CHAIN_INPUTS_FROM_HISTORY)
        summary, strikes = {}, {}
        for ticker in self._tickers:
            expiries = []
            per_strike = []
            stock_price = risk_free_rate = dividend_yield = np.nan
            for expiration_date in self._expiries(
                ticker, expiration_dates, number_of_expirations
            ):
                (
                    calls,
                    puts,
                    stock_price,
                    time_to_expiration,
                    risk_free_rate,
                    dividend_yield,
                ) = self._expiry_inputs(ticker, expiration_date)
                if calls.empty or puts.empty:
                    continue
                expiries.append((calls, puts, time_to_expiration))
                per_strike.append(
                    market_metrics_model.get_gamma_exposure(
                        calls,
                        puts,
                        stock_price,
                        time_to_expiration,
                        risk_free_rate,
                        dividend_yield,
                    )
                )
            if not per_strike:
                continue
            combined = pd.concat(per_strike).groupby(level=0).sum()
            strikes[ticker] = combined
            summary[ticker] = pd.Series(
                {
                    "Stock Price": stock_price,
                    **combined.sum(),
                    "Gamma Flip Point": market_metrics_model.get_gamma_flip(
                        expiries, stock_price, risk_free_rate, dividend_yield
                    ),
                }
            )

        if not summary:
            raise ValueError("No option chains are available for these tickers.")

        result = (
            pd.concat(strikes, names=["Ticker", "Strike Price"])
            if by_strike
            else pd.DataFrame(summary).T
        )
        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_implied_volatility_term_structure(
        self,
        expiration_dates: list[str] | None = None,
        number_of_expirations: int = 10,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the term structure of implied volatility: the at-the-money implied
        volatility of every expiry, interpolated at the current price and averaged over
        calls and puts. Normally longer expiries carry a higher volatility (contango); an
        inverted curve, with short-dated options dearer than long-dated ones, shows that
        the market expects turbulence soon, as around earnings or in a sell-off.

        Also known as: volatility term structure, IV curve, volatility curve.

        Args:
            expiration_dates (list[str] | None, optional): The expiries to use. Defaults to
                None, the nearest `number_of_expirations` expiries.
            number_of_expirations (int, optional): The number of nearest expiries to use when
                no dates are given. Defaults to 10.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per ticker and expiry the Days to Expiration and the At-the-Money
            Volatility (annualized, as a decimal).

        Notes:
        - The stock price is the price of the underlying quoted with the live option chain
          (Yahoo Finance's regular market price or Cboe's current price), so both come from
          the same moment; when the chain quotes none, the traded Close of the last
          historical date is used. The risk free rate and dividend yield come from the
          historical data as well, so leave end_date at today: a warning is logged when the
          history ends more than 5 days before today.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_implied_volatility_term_structure(number_of_expirations=5)
        ```

        Which returns:

        |                        |   Days to Expiration |   At-the-Money Volatility |
        |:-----------------------|---------------------:|--------------------------:|
        | ('AAPL', '2026-10-26') |                    2 |                    0.1637 |
        | ('AAPL', '2026-10-28') |                    4 |                    0.2136 |
        | ('AAPL', '2026-10-30') |                    6 |                    0.2273 |
        | ('AAPL', '2026-11-06') |                   13 |                    0.319  |
        | ('AAPL', '2026-11-13') |                   20 |                    0.2967 |
        | ('TSLA', '2026-10-26') |                    2 |                    0.297  |
        | ('TSLA', '2026-10-28') |                    4 |                    0.3781 |
        | ('TSLA', '2026-10-30') |                    6 |                    0.4003 |
        | ('TSLA', '2026-11-06') |                   13 |                    0.4124 |
        | ('TSLA', '2026-11-13') |                   20 |                    0.4058 |
        """
        return self._per_expiry(
            lambda calls, puts, stock_price, *_: pd.Series(
                {
                    "At-the-Money Volatility": market_metrics_model.get_at_the_money_volatility(
                        calls, puts, stock_price
                    )
                }
            ),
            expiration_dates,
            number_of_expirations,
            rounding,
        )

    @handle_errors
    def get_volatility_skew(
        self,
        expiration_dates: list[str] | None = None,
        number_of_expirations: int = 6,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the volatility skew of every expiry from the 25-delta options, the
        standard way to compare skew across underlyings and dates:

            - 25-Delta Risk Reversal = IV(25-delta call) - IV(25-delta put)
            - 25-Delta Butterfly = (IV(25-delta call) + IV(25-delta put)) / 2 - ATM IV

        For equities the risk reversal is usually negative: out-of-the-money puts are
        dearer than calls because investors pay for crash protection. A more negative risk
        reversal means more fear of a fall; the butterfly measures how much both tails cost
        relative to the centre. The deltas follow from each option's implied volatility, and
        the volatility at exactly 25 delta is interpolated.

        For more information, see the following book:

        - Gatheral, J. (2006). "The Volatility Surface: A Practitioner's Guide." Wiley.

        Also known as: risk reversal, volatility smile, skew, butterfly spread volatility.

        Args:
            expiration_dates (list[str] | None, optional): The expiries to use. Defaults to
                None, the nearest `number_of_expirations` expiries.
            number_of_expirations (int, optional): The number of nearest expiries to use when
                no dates are given. Defaults to 6.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per ticker and expiry the Days to Expiration, the 25-Delta Put,
            At-the-Money and 25-Delta Call Volatility, the 25-Delta Risk Reversal and the
            25-Delta Butterfly.

        Notes:
        - The stock price is the price of the underlying quoted with the live option chain
          (Yahoo Finance's regular market price or Cboe's current price), so both come from
          the same moment; when the chain quotes none, the traded Close of the last
          historical date is used. The risk free rate and dividend yield come from the
          historical data as well, so leave end_date at today: a warning is logged when the
          history ends more than 5 days before today.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["QQQ", "AAPL"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_volatility_skew(number_of_expirations=3)
        ```

        Which returns:

        |                        |   Days to Expiration |   25-Delta Put Volatility |   At-the-Money Volatility |   25-Delta Call Volatility |   25-Delta Risk Reversal |   25-Delta Butterfly |
        |:-----------------------|---------------------:|--------------------------:|--------------------------:|---------------------------:|-------------------------:|---------------------:|
        | ('QQQ', '2026-10-26')  |                    2 |                    0.1908 |                    0.1938 |                     0.1855 |                  -0.0052 |              -0.0056 |
        | ('QQQ', '2026-10-27')  |                    3 |                    0.2287 |                    0.2253 |                     0.213  |                  -0.0157 |              -0.0044 |
        | ('QQQ', '2026-10-28')  |                    4 |                    0.2629 |                    0.2545 |                     0.2396 |                  -0.0233 |              -0.0033 |
        | ('AAPL', '2026-10-26') |                    2 |                    0.1716 |                    0.1637 |                     0.1555 |                  -0.0161 |              -0.0002 |
        | ('AAPL', '2026-10-28') |                    4 |                    0.2238 |                    0.2136 |                     0.2031 |                  -0.0207 |              -0.0002 |
        | ('AAPL', '2026-10-30') |                    6 |                    0.2349 |                    0.2273 |                     0.215  |                  -0.0199 |              -0.0023 |
        """
        return self._per_expiry(
            market_metrics_model.get_volatility_skew,
            expiration_dates,
            number_of_expirations,
            rounding,
        )

    @handle_errors
    def get_implied_move(
        self,
        expiration_dates: list[str] | None = None,
        number_of_expirations: int = 4,
        around_earnings: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the move the options market expects up to every expiry, from the price of
        the at-the-money straddle (buying the call and the put at the strike closest to the
        current price):

            - Implied Move = (Call Price + Put Price) / Stock Price
            - Expected Range = Stock Price ± Straddle Price

        The straddle pays the absolute move, so its price is the market's expected size of
        the move, regardless of direction. With around_earnings=True the first expiry after
        the next earnings date is used, which gives the move the market prices for the
        earnings announcement (an expiry on the announcement day itself settles at that
        day's close, before the usual after-the-close announcement).

        Also known as: expected move, straddle move, earnings move, implied earnings move.

        Args:
            expiration_dates (list[str] | None, optional): The expiries to use. Defaults to
                None, the nearest `number_of_expirations` expiries.
            number_of_expirations (int, optional): The number of nearest expiries to use when
                no dates are given. Defaults to 4.
            around_earnings (bool, optional): Whether to use the first expiry after the next
                earnings date instead. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per ticker and expiry the Days to Expiration, Straddle Strike,
            Straddle Price, Implied Move (as a fraction of the price) and the Lower and Upper
            Price of the expected range.

        Notes:
        - The move up to an expiry after earnings also includes the ordinary daily moves
          until then; the closer the expiry is to the announcement, the purer the measure.
        - The stock price is the price of the underlying quoted with the live option chain
          (Yahoo Finance's regular market price or Cboe's current price), so both come from
          the same moment; when the chain quotes none, the traded Close of the last
          historical date is used. The risk free rate and dividend yield come from the
          historical data as well, so leave end_date at today: a warning is logged when the
          history ends more than 5 days before today.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "NVDA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_implied_move(around_earnings=True)
        ```

        Which returns:

        |                        | Earnings Date   |   Days to Expiration |   Straddle Strike |   Straddle Price |   Implied Move |   Lower Price |   Upper Price |
        |:-----------------------|:----------------|---------------------:|------------------:|-----------------:|---------------:|--------------:|--------------:|
        | ('AAPL', '2026-11-06') | 2026-11-02      |                   13 |               350 |            16.85 |         0.0485 |        330.88 |        364.58 |
        | ('NVDA', '2026-11-20') | 2026-11-18      |                   27 |               210 |            20.5  |         0.0972 |        190.46 |        231.46 |
        """
        earnings_dates = None
        if around_earnings:
            if self._earnings_function is None:
                raise ValueError("Earnings dates are not available for this Toolkit.")
            earnings_dates = self._earnings_function()
            # Each ticker uses the first expiry after its own earnings date.
            expiration_dates = {}
            for ticker in self._tickers:
                earnings_date = earnings_dates.get(ticker)
                if not earnings_date:
                    logger.warning("No upcoming earnings date is known for %s.", ticker)
                    continue
                after = [
                    date
                    for date in options_model.get_option_expiry_dates(ticker)
                    if date > earnings_date
                ]
                if after:
                    expiration_dates[ticker] = [after[0]]
            if not expiration_dates:
                raise ValueError("No expiry after an upcoming earnings date was found.")

        table = self._per_expiry(
            lambda calls, puts, stock_price, *_: market_metrics_model.get_implied_move(
                calls, puts, stock_price
            ),
            expiration_dates,
            number_of_expirations,
            rounding,
        )
        if earnings_dates is not None:
            table.insert(
                0,
                "Earnings Date",
                table.index.get_level_values("Ticker").map(earnings_dates),
            )
        return table

    def _model_free_volatility(self, ticker: str, days: int) -> pd.Series:
        """The constant-maturity model-free implied volatility of one ticker."""
        usable = [
            date
            for date in options_model.get_option_expiry_dates(ticker)
            if _days_to_expiration(date) >= market_metrics_model.MINIMUM_DAYS
        ]
        if len(usable) < 2:  # noqa: PLR2004
            raise ValueError(
                f"{ticker} has too few expiries for a model-free volatility."
            )

        # Beyond the furthest expiry there is nothing to interpolate towards, and holding
        # the last expiry's variance would present it as the volatility of a longer span.
        if days > _days_to_expiration(usable[-1]):
            raise ValueError(
                f"The longest usable expiry of {ticker} is {usable[-1]} "
                f"({_days_to_expiration(usable[-1])} days), shorter than days={days}."
            )

        # A target shorter than the nearest usable expiry is extrapolated from the two
        # nearest terms, as the Cboe does when no expiry lies before the 30 days.
        near_candidates = [date for date in usable if _days_to_expiration(date) <= days]
        near = near_candidates[-1] if near_candidates else usable[0]
        following = [date for date in usable if date > near]
        next_term = following[0] if following else near

        terms = {}
        for label, date in (("Near-Term", near), ("Next-Term", next_term)):
            calls, puts, _, time, rate, _ = self._expiry_inputs(ticker, date)
            # A ValueError reaches the per-ticker warning, so the other tickers still answer.
            if calls.empty or puts.empty:
                raise ValueError(f"{ticker} has no option chain for {date}.")
            variance, forward = market_metrics_model.get_model_free_variance(
                calls, puts, time, rate
            )
            terms[label] = (date, variance, forward, _days_to_expiration(date))

        near_date, near_variance, forward, near_days = terms["Near-Term"]
        next_date, next_variance, _, next_days = terms["Next-Term"]
        volatility = market_metrics_model.get_constant_maturity_volatility(
            near_variance, near_days, next_variance, next_days, days
        )
        return pd.Series(
            {
                "Near-Term Expiration": near_date,
                "Next-Term Expiration": next_date,
                "Near-Term Volatility": (
                    np.sqrt(near_variance) if near_variance > 0 else np.nan
                ),
                "Next-Term Volatility": (
                    np.sqrt(next_variance) if next_variance > 0 else np.nan
                ),
                "Forward Price": forward,
                "Model-Free Implied Volatility": volatility,
            }
        )

    @handle_errors
    def get_model_free_implied_volatility(
        self,
        days: int = 30,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the model-free implied volatility with the Cboe VIX methodology, for any
        ticker with listed options. Rather than inverting a pricing model, it reads the
        expected variance directly from the prices of out-of-the-money options across all
        strikes, weighted by 1/K², for the two expiries around the target maturity, and
        interpolates their variances to a constant maturity:

            - σ² = (2/T) Σ (ΔK_i / K_i²) e^(RT) Q(K_i) - (1/T) (F/K_0 - 1)²
            - Volatility = sqrt(interpolated variance at `days`)

        Applied to the S&P 500 index options ("^SPX") with days=30 it approximates the VIX,
        within about a tenth of a volatility point: the Cboe counts the time to expiry in
        minutes, uses Treasury rates for each expiry and the Friday expiries between 23 and
        37 days away. For a single stock it gives the equivalent volatility index.

        For more information, see the following papers:

        - Cboe Global Markets (2023). "Cboe Volatility Index (VIX) White Paper."
        - Britten-Jones, M. and Neuberger, A. (2000). "Option Prices, Implied Price
          Processes, and Stochastic Volatility." The Journal of Finance, 55(2), 839-866.

        Also known as: VIX methodology, model-free volatility, variance swap volatility.

        Args:
            days (int, optional): The constant maturity in calendar days. Defaults to 30.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per ticker the Near-Term and Next-Term Expiration and Volatility,
            the Forward Price and the Model-Free Implied Volatility (annualized, as a
            decimal).

        Notes:
        - Options without a bid are left out of the strip, as in the Cboe method. Outside
          market hours, when bids are missing, an expiry has no strip and the result is
          NaN.
        - A target shorter than the nearest usable expiry (at least 7 days away) is
          extrapolated from the two nearest terms, as in the Cboe method. A target beyond
          the longest usable expiry raises a ValueError naming the ticker and that expiry,
          since there is no later term to interpolate towards.
        - The risk free rate comes from the historical data, so leave end_date at today: a
          warning is logged when the history ends more than 5 days before today.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["^SPX", "AAPL"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_model_free_implied_volatility()
        ```

        Which returns:

        |      | Near-Term Expiration   | Next-Term Expiration   |   Near-Term Volatility |   Next-Term Volatility |   Forward Price |   Model-Free Implied Volatility |
        |:-----|:-----------------------|:-----------------------|-----------------------:|-----------------------:|----------------:|--------------------------------:|
        | ^SPX | 2026-11-23             | 2026-11-24             |                 0.1955 |                 0.1965 |        7649.3   |                          0.1955 |
        | AAPL | 2026-11-20             | 2026-11-27             |                 0.297  |                 0.2817 |         348.495 |                          0.2897 |
        """
        self._warn_if_history_is_stale("the risk free rate comes from that date")
        rows, errors = {}, []
        for ticker in self._tickers:
            try:
                rows[ticker] = self._model_free_volatility(ticker, days)
            except ValueError as error:
                logger.warning("%s", error)
                errors.append(str(error))

        if not rows:
            raise ValueError(
                "No model-free implied volatility could be calculated. "
                + " ".join(errors)
            )

        table = pd.DataFrame(rows).T
        numeric = table.columns.drop(["Near-Term Expiration", "Next-Term Expiration"])
        table[numeric] = apply_rounding(
            table[numeric].astype(float),
            rounding if rounding is not None else self._rounding,
        )
        return table

    @handle_errors
    def get_volatility_risk_premium(
        self,
        days: int = 30,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the volatility risk premium: the model-free implied volatility over the
        coming `days` minus the volatility the stock actually realized over the same span
        in the past, and the same in variance units. Implied volatility usually exceeds realized volatility, because
        investors pay a premium for protection, which is the return option sellers earn. A
        premium that turns negative means the market prices less movement than it has just
        seen, often after a shock.

            - Realized Volatility = standard deviation of the daily returns over the last
              `days` calendar days, annualized
            - Volatility Risk Premium = Implied Volatility - Realized Volatility
            - Variance Risk Premium = Implied Volatility² - Realized Volatility²

        Both compare the implied figure with the realized one of the past, which is known
        today, as Bollerslev, Tauchen and Zhou (2009) do. Carr and Wu (2009) measure the
        premium after the fact instead: the variance realized over the coming `days` minus
        the variance swap rate, which is negative on average.

        For more information, see the following papers:

        - Bollerslev, T., Tauchen, G. and Zhou, H. (2009). "Expected Stock Returns and
          Variance Risk Premia." The Review of Financial Studies, 22(11), 4463-4492.
        - Carr, P. and Wu, L. (2009). "Variance Risk Premiums." The Review of Financial
          Studies, 22(3), 1311-1341.

        Also known as: VRP, implied minus realized volatility, variance risk premium.

        Args:
            days (int, optional): The horizon in calendar days of both volatilities.
                Defaults to 30.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per ticker the Implied Volatility, Realized Volatility, Volatility
            Risk Premium, Variance Risk Premium and the ratio of implied to realized
            volatility.

        Notes:
        - The realized volatility covers the last `days` calendar days of the historical
          data, while the implied volatility comes from the live option chains. Leave
          end_date at today so both describe the same moment: a warning is logged when the
          history ends more than 5 days before today.
        - A `days` beyond the longest usable expiry raises a ValueError, see
          get_model_free_implied_volatility.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["QQQ", "AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_volatility_risk_premium()
        ```

        Which returns:

        |      |   Implied Volatility |   Realized Volatility |   Volatility Risk Premium |   Variance Risk Premium |   Implied to Realized Ratio |
        |:-----|---------------------:|----------------------:|--------------------------:|------------------------:|----------------------------:|
        | QQQ  |               0.2846 |                0.1769 |                    0.1077 |                  0.0497 |                      1.609  |
        | AAPL |               0.2897 |                0.1641 |                    0.1256 |                  0.057  |                      1.7653 |
        | TSLA |               0.4357 |                0.3516 |                    0.0841 |                  0.0662 |                      1.2392 |
        """
        self._warn_if_history_is_stale(
            "the realized volatility covers the days up to that date instead of the days "
            "up to today, and the risk free rate comes from that date"
        )
        rows, errors = {}, []
        for ticker in self._tickers:
            try:
                implied = float(
                    self._model_free_volatility(ticker, days)[
                        "Model-Free Implied Volatility"
                    ]
                )
            except ValueError as error:
                logger.warning("%s", error)
                errors.append(str(error))
                continue
            returns = self._daily_historical["Return"][ticker].dropna()
            dates = to_period_index(returns.index).to_timestamp()
            recent = returns[dates > dates.max() - pd.Timedelta(days=days)]
            realized = float(recent.std() * np.sqrt(252))
            rows[ticker] = {
                "Implied Volatility": implied,
                "Realized Volatility": realized,
                "Volatility Risk Premium": implied - realized,
                "Variance Risk Premium": implied**2 - realized**2,
                "Implied to Realized Ratio": (
                    implied / realized if realized > 0 else np.nan
                ),
            }

        if not rows:
            raise ValueError(
                "No volatility risk premium could be calculated. " + " ".join(errors)
            )

        return apply_rounding(
            pd.DataFrame(rows).T, rounding if rounding is not None else self._rounding
        )
