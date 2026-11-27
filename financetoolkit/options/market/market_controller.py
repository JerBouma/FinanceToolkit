"""Option Market Metrics Module"""

__docformat__ = "google"

from functools import partial

import numpy as np
import pandas as pd

from financetoolkit.options import options_model
from financetoolkit.options.helpers import (
    get_days_to_expiration,
    get_time_to_expiration,
    validate_positive_inputs,
)
from financetoolkit.options.market import hedging_model, market_metrics_model
from financetoolkit.utilities import logger_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import (
    InvalidArgumentError,
    handle_errors,
    validate_window,
)
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

# Above this bid-ask spread, as a fraction of the straddle, the middle of the quotes says
# little about where the straddle trades.
_WIDE_STRADDLE_SPREAD = 0.5


def _implied_move_noting_quotes(
    unquoted: list[str],
    ticker: str,
    expiration_date: str,
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    stock_price: float,
    *_,
) -> pd.Series:
    """
    The implied move of one expiry, with a warning when its straddle is priced from a last
    trade or its bid-ask spread is wider than half the straddle. An expiry whose chain has no
    bid at all (Yahoo Finance outside US market hours) is added to `unquoted` instead, so that
    the caller warns once for all of them.
    """
    move = market_metrics_model.get_implied_move(calls, puts, stock_price)
    if np.isnan(move["Straddle Strike"]):
        return move
    bids = pd.concat(
        [chain.get("Bid", pd.Series(dtype=float)) for chain in (calls, puts)]
    )
    if not (pd.to_numeric(bids, errors="coerce") > 0).any():
        unquoted.append(f"{ticker} {expiration_date}")
        return move
    if (
        move["Last Trade Legs"] == 0
        and not move["Bid-Ask Spread"] > _WIDE_STRADDLE_SPREAD
    ):
        return move

    strike = f"{move['Straddle Strike']:g}"
    legs = market_metrics_model.get_straddle_quotes(
        calls, puts, move["Straddle Strike"]
    )
    if move["Last Trade Legs"] == 0:
        reasons = [
            f"the bid-ask spread of the {strike} straddle is "
            f"{move['Bid-Ask Spread']:.0%} of its price"
        ]
    else:
        reasons = []
        for leg, quote in legs.iterrows():
            if quote["Last Trade"]:
                reason = (
                    f"the {strike} {leg.lower()} has no two-sided quote and is priced at "
                    "its last trade"
                )
                if isinstance(quote["Last Trade Date"], str):
                    reason += f" of {quote['Last Trade Date']}"
                if pd.notna(quote["Open Interest"]):
                    reason += f" (open interest {quote['Open Interest']:g})"
            else:
                reason = (
                    f"the {leg.lower()}'s bid-ask spread is "
                    f"{quote['Bid-Ask Spread'] / move['Straddle Price']:.0%} of the straddle"
                )
            reasons.append(reason)
    logger.warning(
        "%s %s: %s, so the implied move may not reflect the live market.",
        ticker,
        expiration_date,
        ", and ".join(reasons),
    )
    return move


@filter_shown_columns
@track_sources
class MarketOptions:
    """
    Metrics from the option chains: the put-call ratio, max pain, gamma exposure, the implied volatility term structure and skew, the implied move, the model-free implied volatility, the volatility risk premium, the risk-neutral moments and the SKEW index.
    """

    def _expiries(
        self,
        ticker: str,
        expiration_dates: list[str] | None,
        number_of_expirations: int,
    ) -> list[str]:
        """The expiries of a ticker to use: as given, otherwise the nearest ones."""
        validate_window("number_of_expirations", number_of_expirations)
        available = options_model.get_option_expiry_dates(ticker)
        if expiration_dates is not None:
            invalid = [date for date in expiration_dates if date not in available]
            if invalid:
                raise ValueError(
                    f"The expiration date(s) {', '.join(invalid)} are not available for "
                    f"{ticker}. Choose from {', '.join(available)}"
                )
            # An expiry given for today has settled once the market closed.
            open_dates = []
            for date in expiration_dates:
                if get_time_to_expiration(date, self._current_time()) <= 0:
                    logger.warning(
                        "The options of %s expiring on %s have expired and are left out.",
                        ticker,
                        date,
                    )
                else:
                    open_dates.append(date)
            return open_dates
        now = self._current_time()
        upcoming = [date for date in available if get_days_to_expiration(date, now) > 0]
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
        The stock price is the one quoted with the chain when available (see _spot_price),
        the time to expiration runs from _current_time to 16:00 New York time on the
        expiration date, and the dividend yield is the continuously compounded trailing
        yield on the last historical date (see _dividend_yield_on).
        """
        calls = options_model.get_option_chains([ticker], expiration_date, False)
        puts = options_model.get_option_chains([ticker], expiration_date, True)
        calls = calls.loc[ticker] if not calls.empty else calls
        puts = puts.loc[ticker] if not puts.empty else puts
        stock_price = self._spot_price(ticker, calls, puts)
        # The time left up to the close on the expiration date, so that an expiry this
        # afternoon is not priced as a whole day away.
        time_to_expiration = get_time_to_expiration(
            expiration_date, self._current_time()
        )
        if time_to_expiration <= 0:
            raise ValueError(
                f"The options of {ticker} expiring on {expiration_date} have expired."
            )
        risk_free_rate = float(self._risk_free_rate.dropna().iloc[-1])
        # The trailing dividend yield on the last historical date, continuously compounded
        # as ln(1 + q) like the risk free rate, and 0 for a ticker without dividend data.
        dividend_yield = (
            self._dividend_yield_on(ticker, self._daily_historical.index[-1])
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
        with_ticker: bool = False,
    ) -> pd.DataFrame:
        """
        Run a calculation on every expiry of every ticker, rows (ticker, expiration). The
        expiries are the same for every ticker, or given per ticker as a dictionary, which
        then also selects the tickers. With with_ticker the calculation also receives the
        ticker and the expiration date, before the chain inputs.
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
                result = (
                    calculate(ticker, expiration_date, *inputs)
                    if with_ticker
                    else calculate(*inputs)
                )
                rows[(ticker, expiration_date)] = pd.concat(
                    [
                        pd.Series(
                            {
                                "Days to Expiration": get_days_to_expiration(
                                    expiration_date, self._current_time()
                                )
                            }
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
        Calculate the put-call ratio of every expiry, by volume and by open interest:
        how many puts are traded or held per call, which is usually read as hedging
        demand or bearish sentiment when above one.

        The ratio is the number of puts divided by the number of calls, and an unusually
        low ratio is read as complacency. The volume ratio reflects today's trading; the
        open interest ratio the positions that remain.

        Also known as: PCR, put/call ratio, options sentiment.

        Args:
            expiration_dates (list[str] | None, optional): The expiries to use. Defaults to
                None, the nearest `number_of_expirations` expiries.
            number_of_expirations (int, optional): The number of nearest expiries to use when
                no dates are given, at least 1. Defaults to 6.
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

        |                 |   Days to Expiration |   Call Volume |   Put Volume |   Volume Put-Call Ratio |   Call Open Interest |   Put Open Interest |   Open Interest Put-Call Ratio |
        |:----------------|---------------------:|--------------:|-------------:|------------------------:|---------------------:|--------------------:|-------------------------------:|
        | ('AAPL', 'All') |                  nan |        117895 |       172939 |                  1.4669 |               467397 |              454917 |                         0.9733 |
        | ('TSLA', 'All') |                  nan |        222838 |       192067 |                  0.8619 |               383562 |              508747 |                         1.3264 |
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
                no dates are given, at least 1. Defaults to 6.
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
        | ('AAPL', '2026-11-18') |                    2 |      355   |        352.78 |                 0.0063 |
        | ('AAPL', '2026-11-20') |                    4 |      337.5 |        352.78 |                -0.0433 |
        | ('AAPL', '2026-11-23') |                    7 |      355   |        352.78 |                 0.0063 |
        | ('AAPL', '2026-11-25') |                    9 |      357.5 |        352.78 |                 0.0134 |
        """

        return self._per_expiry(
            lambda calls, puts, stock_price, *_: market_metrics_model.get_max_pain_distance(
                calls, puts, stock_price
            ),
            expiration_dates,
            number_of_expirations,
            rounding,
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
                when no dates are given, at least 1. Defaults to 6.
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
            dollars per 1% move) and the Gamma Flip Point; or, when returned per strike, the
            Call, Put and Net Gamma Exposure per ticker and strike.

        Notes:
        - The assumption about who holds which side is the main uncertainty: when
          investors sell calls (covered calls) the sign of that part flips.
        - Gamma uses each option's own implied volatility (Black-Scholes).
        - The stock price is the price of the underlying quoted with the live option chain
          (Yahoo Finance's regular market price), so both come from the same moment; when
          the chain quotes none, the traded Close of the last historical date is used. The
          risk free rate and dividend yield come from the historical data as well, so
          leave end_date at today: a warning is logged when the history ends more than 5
          days before today.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["QQQ", "AAPL"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_gamma_exposure()
        ```

        Which returns:

        |      |   Stock Price |   Call Gamma Exposure |   Put Gamma Exposure |   Net Gamma Exposure |   Gamma Flip Point |
        |:-----|--------------:|----------------------:|---------------------:|---------------------:|-------------------:|
        | QQQ  |        661.75 |           1.57107e+09 |         -4.27908e+09 |         -2.708e+09   |            689.003 |
        | AAPL |        352.78 |           1.02921e+09 |         -6.02256e+08 |          4.26958e+08 |            346.506 |
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
                no dates are given, at least 1. Defaults to 10.
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
          (Yahoo Finance's regular market price), so both come from the same moment; when
          the chain quotes none, the traded Close of the last historical date is used. The
          risk free rate and dividend yield come from the historical data as well, so
          leave end_date at today: a warning is logged when the history ends more than 5
          days before today.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_implied_volatility_term_structure(number_of_expirations=5)
        ```

        Which returns:

        |                        |   Days to Expiration |   At-the-Money Volatility |
        |:-----------------------|---------------------:|--------------------------:|
        | ('AAPL', '2026-11-18') |                    2 |                    0.2889 |
        | ('AAPL', '2026-11-20') |                    4 |                    0.2983 |
        | ('AAPL', '2026-11-23') |                    7 |                    0.2516 |
        | ('AAPL', '2026-11-25') |                    9 |                    0.2711 |
        | ('AAPL', '2026-11-27') |                   11 |                    0.2508 |
        | ('TSLA', '2026-11-18') |                    2 |                    0.6121 |
        | ('TSLA', '2026-11-20') |                    4 |                    0.6225 |
        | ('TSLA', '2026-11-23') |                    7 |                    0.526  |
        | ('TSLA', '2026-11-25') |                    9 |                    0.5349 |
        | ('TSLA', '2026-11-27') |                   11 |                    0.5078 |
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
        Calculate the volatility skew of every expiry: how much more the market pays for
        downside than for upside protection, from the 25-delta options, the standard way
        to compare skew across underlyings and dates:

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
                no dates are given, at least 1. Defaults to 6.
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
          (Yahoo Finance's regular market price), so both come from the same moment; when
          the chain quotes none, the traded Close of the last historical date is used. The
          risk free rate and dividend yield come from the historical data as well, so
          leave end_date at today: a warning is logged when the history ends more than 5
          days before today.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["QQQ", "AAPL"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_volatility_skew(number_of_expirations=3)
        ```

        Which returns:

        |                        |   Days to Expiration |   25-Delta Put Volatility |   At-the-Money Volatility |   25-Delta Call Volatility |   25-Delta Risk Reversal |   25-Delta Butterfly |
        |:-----------------------|---------------------:|--------------------------:|--------------------------:|---------------------------:|-------------------------:|---------------------:|
        | ('QQQ', '2026-11-17')  |                    1 |                    0.4167 |                    0.3999 |                     0.3738 |                  -0.0429 |              -0.0047 |
        | ('QQQ', '2026-11-18')  |                    2 |                    0.4336 |                    0.4161 |                     0.3949 |                  -0.0387 |              -0.0019 |
        | ('QQQ', '2026-11-19')  |                    3 |                    0.4302 |                    0.4128 |                     0.3904 |                  -0.0398 |              -0.0025 |
        | ('AAPL', '2026-11-18') |                    2 |                    0.3172 |                    0.2889 |                     0.2724 |                  -0.0448 |               0.0059 |
        | ('AAPL', '2026-11-20') |                    4 |                    0.3142 |                    0.2983 |                     0.2728 |                  -0.0414 |              -0.0048 |
        | ('AAPL', '2026-11-23') |                    7 |                    0.27   |                    0.2516 |                     0.2324 |                  -0.0376 |              -0.0004 |
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
        Calculate the move the options market expects up to every expiry: the size of
        the move, in either direction, that the price of the at-the-money straddle
        (buying the call and the put at the strike closest to the current price) prices
        in:

            - Implied Move = (Call Price + Put Price) / Stock Price
            - Expected Range = Stock Price ± Straddle Price
            - Bid-Ask Spread = (Call Ask − Call Bid + Put Ask − Put Bid) / Straddle Price

        The straddle pays the absolute move, so its price is the market's expected size of
        the move, regardless of direction. Around earnings the first expiry after the next
        earnings date is used, which gives the move the market prices for the
        earnings announcement (an expiry on the announcement day itself settles at that
        day's close, before the usual after-the-close announcement).

        Also known as: expected move, straddle move, earnings move, implied earnings move.

        Args:
            expiration_dates (list[str] | None, optional): The expiries to use. Defaults to
                None, the nearest `number_of_expirations` expiries.
            number_of_expirations (int, optional): The number of nearest expiries to use when
                no dates are given, at least 1. Defaults to 4.
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
            Straddle Price, Bid-Ask Spread (as a fraction of the straddle price), Last Trade
            Legs, Implied Move (as a fraction of the price) and the Lower and Upper Price of
            the expected range.

        Notes:
        - The move up to an expiry after earnings also includes the ordinary daily moves
          until then; the closer the expiry is to the announcement, the purer the measure.
        - The prices are the middle of the bid and ask, or the last traded price of an option
          without a two-sided quote that is still held or traded. The Bid-Ask Spread shows
          how firmly the straddle is quoted (NaN when a leg has no two-sided quote) and the
          Last Trade Legs how many of the two legs (0, 1 or 2) are priced from their last
          trade, which can be days or weeks old.
        - A warning names the ticker, expiry and strike when a leg is priced from its last
          trade (with its last trade date and open interest) or when the bid-ask spread is
          wider than half the straddle (above 0.5), as the implied move then may not reflect
          the live market. Outside US market hours Yahoo Finance reports a bid and ask of 0,
          so every leg is priced from its last trade; one warning then names the expiries.
        - The stock price is the price of the underlying quoted with the live option chain
          (Yahoo Finance's regular market price), so both come from the same moment; when
          the chain quotes none, the traded Close of the last historical date is used. The
          risk free rate and dividend yield come from the historical data as well, so
          leave end_date at today: a warning is logged when the history ends more than 5
          days before today.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "NVDA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_implied_move(around_earnings=True)
        ```

        Which returns:

        |                        | Earnings Date   |   Days to Expiration |   Straddle Strike |   Straddle Price |   Bid-Ask Spread |   Last Trade Legs |   Implied Move |   Lower Price |   Upper Price |
        |:-----------------------|:----------------|---------------------:|------------------:|-----------------:|-----------------:|------------------:|---------------:|--------------:|--------------:|
        | ('AAPL', '2027-02-19') | 2027-02-04      |                   85 |               360 |           36.925 |           0.0447 |                 0 |         0.1038 |       318.655 |       392.505 |
        | ('NVDA', '2027-03-19') | 2027-02-24      |                  113 |               150 |           38.725 |           0.0374 |                 0 |         0.2594 |       110.575 |       188.025 |
        """
        earnings_dates = None
        if around_earnings:
            if self._earnings_function is None:
                raise InvalidArgumentError(
                    "The earnings dates come from the Toolkit. Please use this method from a "
                    "Toolkit (toolkit.options), pass earnings_function when creating the Options "
                    "class, or set around_earnings=False."
                )
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

        unquoted: list[str] = []
        table = self._per_expiry(
            partial(_implied_move_noting_quotes, unquoted),
            expiration_dates,
            number_of_expirations,
            rounding,
            with_ticker=True,
        )
        if unquoted:
            logger.warning(
                "The option chains of %s have no bid quotes, so their implied moves are priced "
                "from the last trades, which may not reflect the live market. Outside US market "
                "hours Yahoo Finance reports a bid and ask of 0; try again during market hours.",
                ", ".join(unquoted),
            )
        if earnings_dates is not None:
            table.insert(
                0,
                "Earnings Date",
                table.index.get_level_values("Ticker").map(earnings_dates),
            )
        return table

    def _constant_maturity_terms(
        self, ticker: str, days: int
    ) -> list[tuple[str, pd.DataFrame, pd.DataFrame, float, float]]:
        """
        The near and next term around `days` that the Cboe VIX and SKEW methodologies
        interpolate between: per term the expiration date, calls, puts, time to expiration
        and risk free rate. The terms have at least MINIMUM_DAYS to expiration.
        """
        now = self._current_time()
        usable = [
            date
            for date in options_model.get_option_expiry_dates(ticker)
            if get_days_to_expiration(date, now) >= market_metrics_model.MINIMUM_DAYS
        ]
        if len(usable) < 2:  # noqa: PLR2004
            raise ValueError(
                f"{ticker} has too few expiries to interpolate to {days} days."
            )

        # Beyond the furthest expiry there is nothing to interpolate towards, and holding
        # the last expiry's value would present it as the value of a longer span.
        if days > get_days_to_expiration(usable[-1], now):
            raise ValueError(
                f"The longest usable expiry of {ticker} is {usable[-1]} "
                f"({get_days_to_expiration(usable[-1], now)} days), shorter than days={days}."
            )

        # A target shorter than the nearest usable expiry is extrapolated from the two
        # nearest terms, as the Cboe does when no expiry lies before the 30 days.
        near_candidates = [
            date for date in usable if get_days_to_expiration(date, now) <= days
        ]
        near = near_candidates[-1] if near_candidates else usable[0]
        following = [date for date in usable if date > near]
        next_term = following[0] if following else near

        terms = []
        for date in (near, next_term):
            calls, puts, _, time, rate, _ = self._expiry_inputs(ticker, date)
            # A ValueError reaches the per-ticker warning, so the other tickers still answer.
            if calls.empty or puts.empty:
                raise ValueError(f"{ticker} has no option chain for {date}.")
            # Without bids on both sides of the forward there is no strip (the Cboe methodology selects the strikes on their bids). The last traded prices are no substitute: they are of different moments of the session, so put-call parity and the tails no longer agree.  # noqa: E501
            strip, _ = market_metrics_model.get_model_free_strip(
                calls, puts, time, rate
            )
            if strip.empty:
                raise ValueError(
                    f"{ticker} has no bid quotes for its {date} options to build the strip from. "
                    "Outside US market hours Yahoo Finance reports a bid and ask of 0; try again "
                    "during market hours."
                )
            terms.append((date, calls, puts, time, rate))
        return terms

    def _model_free_volatility(self, ticker: str, days: int) -> pd.Series:
        """The constant-maturity model-free implied volatility of one ticker."""
        near, following = self._constant_maturity_terms(ticker, days)
        near_date, near_calls, near_puts, near_time, near_rate = near
        next_date, next_calls, next_puts, next_time, next_rate = following
        near_variance, forward = market_metrics_model.get_model_free_variance(
            near_calls, near_puts, near_time, near_rate
        )
        next_variance, _ = market_metrics_model.get_model_free_variance(
            next_calls, next_puts, next_time, next_rate
        )
        # The interpolation weighs the terms by the same time to expiration as the
        # variance, in days with the fraction of the day left (the Cboe counts minutes).
        volatility = market_metrics_model.get_constant_maturity_volatility(
            near_variance, near_time * 365, next_variance, next_time * 365, days
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
        Calculate the model-free implied volatility with the Cboe VIX methodology, for
        any ticker with listed options: the volatility the options market expects over
        the coming days, without a pricing model.

        Rather than inverting a pricing model, it reads the expected variance directly
        from the prices of out-of-the-money options across all strikes, weighted by
        1/K², for the two expiries around the target maturity, and interpolates their
        variances to a constant maturity:

            - σ² = (2/T) Σ (ΔK_i / K_i²) e^(RT) Q(K_i) - (1/T) (F/K_0 - 1)²
            - Volatility = sqrt(interpolated variance at `days`)

        Applied to the S&P 500 index options ("^SPX") over 30 days it approximates the VIX,
        within about a tenth of a volatility point: the Cboe counts the time to expiry in
        minutes, uses Treasury rates for each expiry and the Friday expiries between 23 and
        37 days away. For a single stock it gives the equivalent volatility index.

        For more information, see the following papers:

        - Cboe Global Markets (2023). "Cboe Volatility Index (VIX) White Paper."
        - Britten-Jones, M. and Neuberger, A. (2000). "Option Prices, Implied Price
          Processes, and Stochastic Volatility." The Journal of Finance, 55(2), 839-866.

        Also known as: VIX methodology, model-free volatility, variance swap volatility.

        Args:
            days (int, optional): The constant maturity in calendar days, at least 1. Defaults to 30.
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
          market hours Yahoo Finance reports a bid and ask of 0, so a ValueError names the
          ticker and expiry. Last traded prices are not used instead: they are of different
          moments of the session, so put-call parity and the tails no longer agree.
        - A target shorter than the nearest usable expiry (at least 7 days away) is
          extrapolated from the two nearest terms, as in the Cboe method. A target beyond
          the longest usable expiry raises a ValueError naming the ticker and that expiry,
          since there is no later term to interpolate towards.
        - The time to each expiry runs from the current time to 16:00 New York time on the
          expiration date, in minutes over the 525,600 minutes of a year as in the Cboe method, and
          the variance of a term and its weight in the interpolation use that same time.
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
        | ^SPX | 2026-12-16             | 2026-12-17             |                 0.2274 |                 0.2284 |        7436.64  |                          0.2276 |
        | AAPL | 2026-12-11             | 2026-12-18             |                 0.2655 |                 0.2724 |         353.871 |                          0.2709 |
        """
        validate_window("days", days)
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
            days (int, optional): The horizon in calendar days of both volatilities, at least 1.
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
        | QQQ  |               0.3272 |                0.3555 |                   -0.0283 |                 -0.0193 |                      0.9203 |
        | AAPL |               0.2709 |                0.2745 |                   -0.0036 |                 -0.002  |                      0.9867 |
        | TSLA |               0.5498 |                0.5943 |                   -0.0445 |                 -0.0509 |                      0.9251 |
        """
        validate_window("days", days)
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

    @handle_errors
    def get_risk_neutral_moments(
        self,
        expiration_dates: list[str] | None = None,
        number_of_expirations: int = 4,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the risk-neutral (option-implied) volatility, skewness and kurtosis of
        the log return up to every expiry: the shape of the return distribution that the
        option prices imply, with the method of Bakshi, Kapadia and Madan (2003).

        Any payoff can be built from a position in the underlying and a strip of
        out-of-the-money options (Carr and Madan, 2001), so the prices of the quadratic,
        cubic and quartic contracts on the log return follow from the option prices, and
        with them the moments the market prices in, without a pricing model. A negative
        skewness means the options price a larger chance of a fall than of a rise of the
        same size; a kurtosis above 3 means fatter tails than a normal distribution.

        The formula is as follows, with R = ln(S_T / K_0), y_i = ln(K_i / K_0) and the
        strikes K_i, prices Q(K_i) and intervals ΔK_i of the VIX methodology:

            - V = Σ 2 (1 - y_i) Q(K_i) ΔK_i / K_i²
            - W = Σ (6 y_i - 3 y_i²) Q(K_i) ΔK_i / K_i²
            - X = Σ (12 y_i² - 4 y_i³) Q(K_i) ΔK_i / K_i²
            - μ = F / K_0 - 1 - e^(rT) V / 2 - e^(rT) W / 6 - e^(rT) X / 24
            - Skewness = (e^(rT) W - 3 μ e^(rT) V + 2 μ³) / (e^(rT) V - μ²)^(3/2)
            - Kurtosis = (e^(rT) X - 4 μ e^(rT) W + 6 e^(rT) μ² V - 3 μ⁴) / (e^(rT) V - μ²)²
            - Volatility = sqrt((e^(rT) V - μ²) / T)
            - Expected Log Return = E[ln(S_T / F)] = μ - ln(F / K_0)

        The integrals of the paper are discretised as in the Cboe VIX methodology: the
        forward price F follows from put-call parity at the strike where call and put
        prices are closest, K_0 is the first strike at or below it, Q(K_i) is the put price
        below K_0, the call price above and their average at K_0, and ΔK_i is half the
        distance between the neighbouring strikes (the distance to the one neighbour at the
        ends). Puts and calls are added away from K_0 while they have a bid, up to two
        consecutive strikes without one. The paper expands around the spot price without
        dividends (F / S = e^(rT)); the expansion around K_0 keeps the replication exact and
        takes in dividends through the forward price, and leaves the skewness and kurtosis
        unchanged. r is the continuously compounded risk free rate.

        For more information, see the following papers:

        - Bakshi, G., Kapadia, N. and Madan, D. (2003). "Stock Return Characteristics, Skew
          Laws, and the Differential Pricing of Individual Equity Options." The Review of
          Financial Studies, 16(1), 101-143.
        - Carr, P. and Madan, D. (2001). "Optimal Positioning in Derivative Securities."
          Quantitative Finance, 1(1), 19-37.
        - Cboe Global Markets (2023). "Cboe Volatility Index (VIX) White Paper."

        Also known as: option-implied moments, implied skewness, implied kurtosis, BKM moments.

        Args:
            expiration_dates (list[str] | None, optional): The expiries to use. Defaults to
                None, the nearest `number_of_expirations` expiries.
            number_of_expirations (int, optional): The number of nearest expiries to use when
                no dates are given, at least 1. Defaults to 4.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per ticker and expiry the Days to Expiration, Forward Price,
            Expected Log Return (relative to the forward price, over the time to
            expiration), Volatility (annualized), Skewness and Kurtosis of the log return.

        Notes:
        - The sums approximate the integrals over the strikes, so the moments depend on how
          finely the strikes are spaced and how far they reach into the tails; the kurtosis is
          the most sensitive. On a Black-Scholes chain (20% volatility, 30 days) with strikes
          from 50% to 150% of the price, the kurtosis is within 0.031 of the normal value of 3
          (and the skewness within 0.001 of 0) for strikes 1% of the price apart, and within
          0.001 for strikes 0.1% apart; strikes from 10% to 190% give the same values, as the
          options beyond 50% and 150% are worth almost nothing. The reach matters when the
          strip ends within a few standard deviations of the price: from 80% to 120% (3.5
          standard deviations) the kurtosis is 0.008 low at the finest spacing.
        - Prices are the middle of the bid and ask, or the last price of an option with
          volume or open interest when there is no two-sided quote. Outside market hours,
          when bids are missing, an expiry has no strip and its moments are NaN.
        - The time to each expiry runs from the current time to 16:00 New York time on the
          expiration date, in minutes over the 525,600 minutes of a year. The risk free rate
          comes from the historical data, so leave end_date at today: a warning is logged
          when the history ends more than 5 days before today.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["^SPX", "AAPL"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_risk_neutral_moments(
            number_of_expirations=3, show_columns=["Volatility", "Skewness", "Kurtosis"]
        )
        ```

        Which returns:

        |                        |   Volatility |   Skewness |   Kurtosis |
        |:-----------------------|-------------:|-----------:|-----------:|
        | ('^SPX', '2026-11-19') |       0.2852 |    -0.9794 |    10.6888 |
        | ('^SPX', '2026-11-20') |       0.2745 |    -2.5062 |    28.0073 |
        | ('^SPX', '2026-11-23') |       0.2381 |    -3.069  |    41.0756 |
        | ('AAPL', '2026-11-20') |       0.3401 |    -1.4245 |    12.0253 |
        | ('AAPL', '2026-11-23') |       0.2615 |    -1.4104 |     9.5121 |
        | ('AAPL', '2026-11-25') |       0.2763 |    -1.4946 |     8.8768 |
        """

        return self._per_expiry(
            lambda calls, puts, _, time, rate, *__: market_metrics_model.get_chain_risk_neutral_moments(
                calls, puts, time, rate
            ),
            expiration_dates,
            number_of_expirations,
            rounding,
        )

    def _skew_index(self, ticker: str, days: int) -> pd.Series:
        """The constant-maturity Cboe SKEW-style index of one ticker."""
        terms = {}
        for label, (date, calls, puts, time, rate) in zip(
            ("Near-Term", "Next-Term"),
            self._constant_maturity_terms(ticker, days),
            strict=True,
        ):
            strip, forward = market_metrics_model.get_model_free_strip(
                calls, puts, time, rate
            )
            skewness = market_metrics_model.get_cboe_skewness(
                strip.index.to_numpy(dtype=float),
                strip["Call Price"].to_numpy(),
                strip["Put Price"].to_numpy(),
                forward,
                rate,
                time,
            )
            # The weights use the time to expiration in days with the fraction of the day
            # left, proportional to the minutes the Cboe counts.
            terms[label] = (date, skewness, forward, time * 365)

        near_date, near_skewness, forward, near_days = terms["Near-Term"]
        next_date, next_skewness, _, next_days = terms["Next-Term"]
        skewness, index = market_metrics_model.get_skew_index(
            near_skewness, near_days, next_skewness, next_days, days
        )
        return pd.Series(
            {
                "Near-Term Expiration": near_date,
                "Next-Term Expiration": next_date,
                "Near-Term Skewness": near_skewness,
                "Next-Term Skewness": next_skewness,
                "Forward Price": forward,
                "Skewness": skewness,
                "SKEW Index": index,
            }
        )

    @handle_errors
    def get_skew_index(
        self,
        days: int = 30,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate a SKEW index with the Cboe SKEW Index methodology, for any ticker with
        listed options: the risk-neutral skewness S of the log return over the coming
        `days`, read from the prices of out-of-the-money options and expressed as
        100 - 10 S. A SKEW of 100 means the options price a symmetric distribution; the
        higher the index, the more the options price a large fall (tail risk). The Cboe
        SKEW of the S&P 500 has mostly ranged between 110 and 150.

        The formula is as follows, per expiry with R = ln(S_T / F_0), y_i = ln(K_i / F_0)
        and the strikes K_i, prices Q(K_i) and intervals ΔK_i of the VIX methodology:

            - P_1 = -e^(rT) Σ Q(K_i) ΔK_i / K_i² + ε_1
            - P_2 = e^(rT) Σ 2 (1 - y_i) Q(K_i) ΔK_i / K_i² + ε_2
            - P_3 = e^(rT) Σ 3 (2 y_i - y_i²) Q(K_i) ΔK_i / K_i² + ε_3
            - ε_1 = -(1 + ln(F_0 / K_0) - F_0 / K_0)
            - ε_2 = 2 ln(K_0 / F_0) (F_0 / K_0 - 1) + ln(K_0 / F_0)²
            - ε_3 = 3 ln(K_0 / F_0)² (ln(K_0 / F_0) / 3 - 1 + F_0 / K_0)
            - S = (P_3 - 3 P_1 P_2 + 2 P_1³) / (P_2 - P_1²)^(3/2)

        and over the near and next term around `days`:

            - w = (T_next - T_days) / (T_next - T_near)
            - Skewness = w S_near + (1 - w) S_next
            - SKEW Index = 100 - 10 Skewness

        P_1, P_2 and P_3 are the prices of R, R² and R³, the ε terms correct for the
        expansion around the forward price F_0 while the puts and calls are split at K_0,
        the first strike at or below it, and r is the continuously compounded risk free
        rate. The terms, the forward price, K_0 and the strikes used are those of the VIX
        methodology (see get_model_free_implied_volatility), as the white paper prescribes.
        ε_2 holds the full ln(K_0 / F_0)² of the expansion of R² around K_0, where the white
        paper prints half of it (see market_metrics_model.get_cboe_skewness); for index
        options the difference is negligible.
        Unlike the variance of the VIX, the skewness is interpolated linearly in time.
        Applied to the S&P 500 index options ("^SPX") over 30 days it approximates the Cboe
        SKEW Index, which uses the Cboe's own quotes and Treasury bill rates per expiry.

        For more information, see the following papers:

        - Cboe Global Markets (2010). "The Cboe SKEW Index - SKEW." White paper.
        - Bakshi, G., Kapadia, N. and Madan, D. (2003). "Stock Return Characteristics, Skew
          Laws, and the Differential Pricing of Individual Equity Options." The Review of
          Financial Studies, 16(1), 101-143.

        Also known as: Cboe SKEW, tail risk index, black swan index, implied skewness index.

        Args:
            days (int, optional): The constant maturity in calendar days, at least 1. Defaults to 30.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per ticker the Near-Term and Next-Term Expiration and Skewness, the
            Forward Price of the near term, the Skewness at `days` and the SKEW Index.

        Notes:
        - The skewness depends on how far the strikes reach into the tails, so a stock with
          few listed strikes shows a SKEW closer to 100 than its options imply.
        - A target shorter than the nearest usable expiry (at least 7 days away) is
          extrapolated from the two nearest terms, as in the Cboe method. A target beyond
          the longest usable expiry raises a ValueError naming the ticker and that expiry.
        - Outside market hours (at the weekend) Yahoo Finance reports a bid and ask of 0, so
          a ValueError names the ticker and expiry instead of a NaN index. Last traded
          prices are not used: they are of different moments of the session, which gives a
          forward price and tails that do not agree (for ^SPX a skewness of -12 on a Sunday).
        - The near and next term are the expiries just before and after `days` (at least 7
          days away), whichever weekday they expire on. The Cboe VIX only uses the SPX
          options that expire on a Friday, more than 23 and less than 37 days away, so for
          ^SPX the terms can be daily expiries with fewer strikes than the Cboe's.
        - The time to each expiry runs from the current time to 16:00 New York time on the
          expiration date, in minutes over the 525,600 minutes of a year. The risk free rate
          comes from the historical data, so leave end_date at today: a warning is logged
          when the history ends more than 5 days before today.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["^SPX", "AAPL"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_skew_index(
            show_columns=["Near-Term Expiration", "Next-Term Expiration", "Skewness", "SKEW Index"]
        )
        ```

        Which returns:

        |      | Near-Term Expiration   | Next-Term Expiration   |   Skewness |   SKEW Index |
        |:-----|:-----------------------|:-----------------------|-----------:|-------------:|
        | ^SPX | 2026-12-18             | 2026-12-21             |    -4.2626 |      142.626 |
        | AAPL | 2026-12-18             | 2026-12-24             |    -1.7288 |      117.288 |
        """
        validate_window("days", days)
        self._warn_if_history_is_stale("the risk free rate comes from that date")
        rows, errors = {}, []
        for ticker in self._tickers:
            try:
                rows[ticker] = self._skew_index(ticker, days)
            except ValueError as error:
                logger.warning("%s", error)
                errors.append(str(error))

        if not rows:
            raise ValueError("No SKEW index could be calculated. " + " ".join(errors))

        table = pd.DataFrame(rows).T
        numeric = table.columns.drop(["Near-Term Expiration", "Next-Term Expiration"])
        table[numeric] = apply_rounding(
            table[numeric].astype(float),
            rounding if rounding is not None else self._rounding,
        )
        return table

    @handle_errors
    def get_variance_swap(
        self,
        expiration_dates: list[str] | None = None,
        number_of_expirations: int = 4,
        strike: float | None = None,
        trade_date: str | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the fair strike of a variance swap up to every expiry from the option
        chain: the variance at which a swap on the stock's realized variance is worth
        nothing today, replicated with the discrete strike weights of Demeterfi, Derman,
        Kamal and Zou (1999), and with a strike the value of a swap, also of one traded
        in the past.

        A variance swap pays the variance the stock realizes up to maturity minus the
        strike; the fair strike is the price of a portfolio of out-of-the-money options
        that replicates the log contract, whose payoff grows with the realized variance
        whatever the path:

            - f(S) = (2/T) [(S - S*) / S* - ln(S / S*)]
            - w(K_i) = the change in the slope of f at K_i, with f linear between the strikes
            - Fair Variance = (2/T) [ln(F / S*) - (F / S* - 1)] + e^(rT) [Σ w(K_p) P(K_p) + Σ w(K_c) C(K_c)]
            - Fair Volatility = sqrt(Fair Variance)
            - Realized Variance = (252 / n) Σ ln(S_i / S_(i-1))², over the n returns since trade_date
            - Value = e^(-r (T - t)) [(t / T) Realized Variance + ((T - t) / T) Fair Variance - Strike²]

        with the forward price F and the boundary strike S* = K_0, the highest strike at or
        below it, from put-call parity as in the Cboe VIX methodology, puts below S* and calls
        above it. The Cboe Variance column is the VIX methodology's variance of the same
        expiry (see get_model_free_implied_volatility): it discretises the same integral with
        the midpoint rule, so the two agree when the strikes are close together and the Fair
        Variance replicates the payoff at the quoted strikes exactly when they are not.

        For more information, see the following papers:

        - Demeterfi, K., Derman, E., Kamal, M. and Zou, J. (1999). "More Than You Ever Wanted
          to Know About Volatility Swaps." Goldman Sachs Quantitative Strategies Research
          Notes.
        - Carr, P. and Madan, D. (1998). "Towards a Theory of Volatility Trading." In Volatility:
          New Estimation Techniques for Pricing Derivatives, Risk Books, 417-427.

        Also known as: variance swap rate, variance strike, fair variance, DDKZ replication.

        Referenced against:
        - QuantLib, ReplicatingVarianceSwapEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_replicating_variance_swap_engine.html
        - FinancePy, EquityVarianceSwap.fair_strike: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/equity/equity_variance_swap.py#L102
        - FinancePy, EquityVarianceSwap.value: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/equity/equity_variance_swap.py#L61

        Shares similar logic:
        - QuantLib, VarianceSwap: https://www.quantlib.org/reference/class_quant_lib_1_1_variance_swap.html

        Args:
            expiration_dates (list[str] | None, optional): The expiries to use, the maturities
                of the swaps. Defaults to None, the nearest `number_of_expirations` expiries.
            number_of_expirations (int, optional): The number of nearest expiries to use when
                no dates are given, at least 1. Defaults to 4.
            strike (float | None, optional): The volatility strike of the swap as a decimal
                (0.25 for 25%), whose square is the variance strike. Defaults to None, which
                gives the fair strike only.
            trade_date (str | None, optional): The date the swap was traded, whose returns
                since then count towards the realized variance. Requires a strike. Defaults to
                None, a swap traded today.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per ticker and expiry the Days to Expiration, Forward Price, Boundary
            Strike, Fair Variance, Fair Volatility and Cboe Variance (annualized, as decimals),
            and with a strike the Variance Strike, the Realized Variance and its share of the
            swap (Elapsed Fraction) and the Variance Swap Value per unit of variance notional.

        Notes:
        - The value is per unit of variance notional, paying σ² - K² on decimal variances. A
          vega notional N_vega, the profit per volatility point, is N_vega / (2 Strike) units of
          variance notional, so Value × 100 / (2 Strike) is the value per unit of vega notional
          in volatility points.
        - The fair strike uses the strip of the Cboe methodology: options with a bid, walking
          away from K_0 until two strikes in a row have none. Outside US market hours Yahoo
          Finance reports a bid of 0, and the fair strike is NaN.
        - The realized variance counts the daily log returns of the Adjusted Close since
          trade_date, without subtracting their mean, as variance swaps settle. The share t/T
          counts those returns against the returns still to come, the weekdays up to the
          expiry, and the value discounts over the time to expiration with the risk free rate
          of the historical data (continuously compounded). Leave end_date at today: a warning
          is logged when the history ends more than 5 days before today.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "^SPX"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_variance_swap(
            number_of_expirations=2,
            strike=0.3,
            show_columns=["Forward Price", "Fair Variance", "Fair Volatility", "Cboe Variance", "Variance Swap Value"],
        )
        ```

        Which returns:

        |                        |   Forward Price |   Fair Variance |   Fair Volatility |   Cboe Variance |   Variance Swap Value |
        |:-----------------------|----------------:|----------------:|------------------:|----------------:|----------------------:|
        | ('AAPL', '2026-11-25') |          349.55 |          0.1281 |            0.358  |          0.1281 |                0.0381 |
        | ('AAPL', '2026-11-27') |          349.65 |          0.0887 |            0.2979 |          0.0887 |               -0.0013 |
        | ('^SPX', '2026-11-24') |         7253.05 |          0.1076 |            0.328  |          0.1076 |                0.0176 |
        | ('^SPX', '2026-11-25') |         7255    |          0.1055 |            0.3248 |          0.1055 |                0.0155 |
        """
        if trade_date is not None and strike is None:
            raise InvalidArgumentError(
                "A swap traded on an earlier date needs its strike: the option chains of that "
                "date are not available to find its fair strike."
            )
        if strike is not None and strike <= 0:
            raise InvalidArgumentError("The strike must be a positive volatility.")

        prices = self._daily_historical["Adj Close"]

        return self._per_expiry(
            lambda ticker, expiration_date, calls, puts, _, time, rate, __: (
                market_metrics_model.get_variance_swap_terms(
                    calls,
                    puts,
                    time,
                    rate,
                    prices,
                    ticker,
                    expiration_date,
                    strike,
                    trade_date,
                )
            ),
            expiration_dates,
            number_of_expirations,
            rounding,
            with_ticker=True,
        )

    def _hedge_inputs(
        self, ticker: str, date: pd.Period, volatility: float | None
    ) -> tuple[float, float, float]:
        """The volatility, risk free rate and dividend yield of a ticker on a date."""
        if volatility is None:
            volatility = float(
                self._trailing_volatility[ticker].loc[:date].dropna().iloc[-1]
            )
        risk_free_rate = float(self._risk_free_rate.loc[:date].dropna().iloc[-1])
        dividend_yield = self._dividend_yield_on(ticker, date)
        return (
            volatility,
            risk_free_rate,
            0.0 if np.isnan(dividend_yield) else dividend_yield,
        )

    @handle_errors
    def get_delta_hedging_backtest(
        self,
        days: int = 63,
        start_date: str | None = None,
        moneyness: float = 1.0,
        volatility: float | None = None,
        put_option: bool = False,
        transaction_cost: float = 0.0,
        rebalancing_days: int = 1,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the realized profit and loss of selling a European option on each
        ticker and delta hedging it on the ticker's own prices until it expires `days`
        trading days later: how much the hedged seller gained or lost when the
        volatility the stock realized differed from the one the option was sold at
        (Hull, 2022, chapter 19).

        The option is sold at its Black-Scholes value, the hedge holds its delta in
        shares, pays a proportional cost on every trade, receives the dividends on the
        shares held and keeps the rest in cash at the risk free rate:

            - Option Value = Black-Scholes value at the volatility σ on the start date
            - Cash_(i+1) = Cash_i e^(r Δt) + Δ_i D_(i+1) - (Δ_(i+1) - Δ_i) S_(i+1) - k |Δ_(i+1) - Δ_i| S_(i+1)
            - Profit and Loss = e^(-rT) [Cash_T + Δ_T S_T - Payoff(S_T)]

        The profit and loss shows what the volatility the option was sold at earned against
        the volatility the stock then realized: the seller gains about ½ Γ S² (σ² - σ²_realized)
        per unit of time (Wilmott, 2006), so a positive result means the stock moved less than
        the option priced, less the costs of hedging (Leland, 1985).

        For more information, see the following sources:

        - Hull, J. C. (2022). "Options, Futures, and Other Derivatives." 11th edition, chapter 19.
        - Leland, H. E. (1985). "Option Pricing and Replication with Transactions Costs." The
          Journal of Finance, 40(5), 1283-1301.
        - Wilmott, P. (2006). "Paul Wilmott on Quantitative Finance." 2nd edition, chapter 12.

        Also known as: delta hedging P&L, hedged option P&L, gamma scalping backtest,
        realized hedging error.

        Args:
            days (int, optional): The trading days from the sale to expiration, at least 1. Defaults to 63,
                about three months.
            start_date (str | None, optional): The date the option is sold, or the last
                trading day before it. Defaults to None, `days` trading days before the end
                of the historical data, so the option expires on its last date.
            moneyness (float, optional): The strike as a fraction of the stock price on the
                start date. Defaults to 1.0, at the money.
            volatility (float | None, optional): The volatility the option is sold and hedged
                at, such as an implied volatility, for every ticker, 0 or above (at 0 the option
                is sold at its discounted intrinsic value and the delta is 0 or 1). Defaults to
                None, the volatility of the year of daily returns up to the start date, which is
                known on that date.
            put_option (bool, optional): Whether the option is a put. Defaults to False.
            transaction_cost (float, optional): The cost of a trade as a fraction of its value
                (0.001 is 10 basis points), 0 or above. Defaults to 0.0.
            rebalancing_days (int, optional): The trading days between rebalances. Defaults to
                1, daily.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per ticker the Start Date, Expiration Date, Stock Price, Strike Price,
            Volatility, Realized Volatility, Option Value, Payoff, Transaction Costs and Profit
            and Loss. The costs and the profit and loss are discounted to the start date.

        Raises:
            InvalidArgumentError: When the days or rebalancing days are below 1, the moneyness
                is not positive, or the volatility or transaction cost is negative.

        Notes:
        - Time runs on each ticker's own trading days, 252 to a year, the same clock as the
          volatility, so the option's time value and the variance the stock realizes count
          the same days.
        - The stock follows the traded Close and the hedge receives the cash dividends on
          their ex-dates; the option is valued with the trailing dividend yield and the risk
          free rate on the start date, both continuously compounded.
        - At expiration the hedge moves to the position the option settles with (one share
          for a call in the money, none out of the money), and that trade pays the cost as
          well, as for physical settlement.
        - The Realized Volatility is the square root of the mean squared daily log return of
          the Adjusted Close over the option's life, annualized, as a variance swap settles.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY", start_date="2023-01-01", end_date="2025-12-31"
        )

        toolkit.options.get_delta_hedging_backtest(
            start_date="2025-06-02",
            transaction_cost=0.0005,
            show_columns=["Volatility", "Realized Volatility", "Option Value", "Payoff", "Transaction Costs", "Profit and Loss"],
        )
        ```

        Which returns:

        |      |   Volatility |   Realized Volatility |   Option Value |   Payoff |   Transaction Costs |   Profit and Loss |
        |:-----|-------------:|----------------------:|---------------:|---------:|--------------------:|------------------:|
        | AAPL |       0.3304 |                0.2141 |        14.2012 |    28.02 |              0.2146 |            2.9596 |
        | MSFT |       0.2574 |                0.1516 |        25.7479 |    43.15 |              0.3291 |            5.2927 |
        """
        if days < 1 or rebalancing_days < 1:
            raise InvalidArgumentError(
                "The days and rebalancing days must be at least 1."
            )
        if moneyness <= 0:
            raise InvalidArgumentError("The moneyness must be positive.")
        # Checked once for every ticker. A volatility of zero is valid: the option is then worth
        # its discounted intrinsic value and the delta steps from 0 to 1 at the forward price.
        try:
            validate_positive_inputs(
                allow_zero=True,
                volatility=volatility,
                transaction_cost=transaction_cost,
            )
        except ValueError as error:
            raise InvalidArgumentError(str(error)) from None

        dividends = (
            self._daily_historical["Dividends"]
            if "Dividends" in self._daily_historical.columns.get_level_values(0)
            else None
        )
        rows = {}
        for ticker in self._tickers:
            history = self._prices[ticker].dropna()
            if start_date is None:
                start = len(history) - 1 - days
            else:
                start = (
                    history.index.searchsorted(
                        pd.Period(start_date, freq="D"), side="right"
                    )
                    - 1
                )
            if start < 0 or start + days >= len(history):
                logger.warning(
                    "The historical data of %s does not cover %d trading days after the start "
                    "date, so it is left out.",
                    ticker,
                    days,
                )
                continue

            path = history.iloc[start : start + days + 1]
            date = path.index[0]
            try:
                sigma, rate, dividend_yield = self._hedge_inputs(
                    ticker, date, volatility
                )
            except IndexError:
                # Too few prices before the start date for a volatility leaves this ticker out only.  # noqa: E501
                logger.warning(
                    "No delta hedging backtest for %s: it has too few prices for a "
                    "volatility on the start date.",
                    ticker,
                )
                continue
            paid = None
            if dividends is not None:
                paid = (
                    dividends[ticker].reindex(path.index).fillna(0).to_numpy(copy=True)
                )
                # A dividend going ex on the start date belongs to the seller of the shares.
                paid[0] = 0.0
            result = hedging_model.get_delta_hedging_profit_and_loss(
                stock_prices=path.to_numpy(),
                strike_price=moneyness * float(path.iloc[0]),
                time_to_expiration=days / 252,
                risk_free_rate=rate,
                volatility=sigma,
                dividend_yield=dividend_yield,
                put_option=put_option,
                transaction_cost=transaction_cost,
                rebalancing_interval=rebalancing_days,
                dividends=paid,
            ).iloc[0]
            adjusted = self._daily_historical["Adj Close"][ticker].loc[
                path.index[0] : path.index[-1]
            ]
            rows[ticker] = {
                "Start Date": str(date),
                "Expiration Date": str(path.index[-1]),
                "Stock Price": float(path.iloc[0]),
                "Strike Price": moneyness * float(path.iloc[0]),
                "Volatility": sigma,
                "Realized Volatility": float(
                    np.sqrt(
                        market_metrics_model.get_realized_variance(adjusted.dropna())
                    )
                ),
                **result.to_dict(),
            }

        if not rows:
            raise ValueError(
                f"No ticker has {days} trading days of prices after the start date."
            )

        table = pd.DataFrame(rows).T
        numeric = table.columns.drop(["Start Date", "Expiration Date"])
        table[numeric] = apply_rounding(
            table[numeric].astype(float),
            rounding if rounding is not None else self._rounding,
        )
        return table

    @handle_errors
    def get_delta_hedging_simulation(
        self,
        days: int = 63,
        start_date: str | None = None,
        moneyness: float = 1.0,
        volatility: float | None = None,
        implied_volatility: float | None = None,
        put_option: bool = False,
        transaction_cost: float = 0.0,
        rebalancing_days: int = 1,
        simulations: int = 10_000,
        seed: int | None = 0,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the distribution of the hedging error of selling a European option and
        delta hedging it: how much the hedged seller can gain or lose across simulated
        Geometric Brownian Motion paths of the stock from its price on the start date.

        Each path is hedged as in get_delta_hedging_backtest: the option is sold at its
        Black-Scholes value at the implied volatility, the hedge holds its delta,
        rebalanced every `rebalancing_days` trading days at a proportional cost, while
        the stock moves with the volatility:

            - S(t + Δt) = S(t) e^((r - q - σ² / 2) Δt + σ sqrt(Δt) Z), Z ~ N(0, 1), Δt = 1 / 252
            - Profit and Loss = e^(-rT) [Cash_T + Δ_T S_T - Payoff(S_T)], per path
            - Approximate Standard Deviation = sqrt(π / 4) × Vega × σ / sqrt(N)

        Hedging at discrete times leaves an error whose mean is near zero and whose standard
        deviation falls with the square root of the number of rebalances N, as Boyle and
        Emanuel (1980) and Derman and Kamal (1999) derive for an at-the-money option; the last
        column gives their approximation. An implied volatility above the volatility gives the
        seller a profit on average, transaction costs a loss that grows as the hedge is
        rebalanced more often (Leland, 1985).

        For more information, see the following papers:

        - Boyle, P. P. and Emanuel, D. (1980). "Discretely Adjusted Option Hedges." Journal of
          Financial Economics, 8(3), 259-282.
        - Derman, E. and Kamal, M. (1999). "When You Cannot Hedge Continuously: The
          Corrections of Black-Scholes." Risk, 12(1), 82-85.
        - Leland, H. E. (1985). "Option Pricing and Replication with Transactions Costs." The
          Journal of Finance, 40(5), 1283-1301.

        FinancePy's simulate_hedge_paths reports the profit and loss at expiration rather
        than discounted and closes the hedge in cash at expiration.

        Also known as: hedging error distribution, discrete hedging simulation, delta hedging
        Monte Carlo.

        Referenced against:
        - FinancePy, simulate_hedge_paths: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/black_scholes_hedging_sim.py#L592

        Args:
            days (int, optional): The trading days to expiration, at least 1. Defaults to 63, about three
                months.
            start_date (str | None, optional): The date of the stock price, volatility, risk
                free rate and dividend yield, or the last trading day before it. Defaults to
                None, the last date of the historical data.
            moneyness (float, optional): The strike as a fraction of the stock price. Defaults
                to 1.0, at the money.
            volatility (float | None, optional): The volatility the stock moves with, for every
                ticker, 0 or above (at 0 every path grows at the risk free rate less the
                dividend yield). Defaults to None, the volatility of the year of daily returns
                up to the start date.
            implied_volatility (float | None, optional): The volatility the option is sold and
                hedged at, 0 or above (at 0 the option is sold at its discounted intrinsic value
                and the delta is 0 or 1). Defaults to None, the volatility the stock moves with.
            put_option (bool, optional): Whether the option is a put. Defaults to False.
            transaction_cost (float, optional): The cost of a trade as a fraction of its value
                (0.001 is 10 basis points), 0 or above. Defaults to 0.0.
            rebalancing_days (int, optional): The trading days between rebalances. Defaults to
                1, daily.
            simulations (int, optional): The number of simulated paths, at least 1. Defaults to 10,000.
            seed (int | None, optional): The seed of the random numbers, None for a different
                draw on every call. Defaults to 0.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per ticker the Stock Price, Strike Price, Volatility, Implied
            Volatility, Option Value, the Mean, Standard Deviation and 5% and 95% Quantiles of
            the Profit and Loss, the Mean Transaction Costs (both discounted to the start date)
            and the Approximate Standard Deviation.

        Raises:
            InvalidArgumentError: When the days or rebalancing days are below 1, the moneyness
                is not positive, the simulations are not a whole number of at least 1, or the
                volatility, implied volatility or transaction cost is negative.

        Notes:
        - The paths drift at the risk free rate. The real-world drift barely moves the hedging
          error, since the hedge removes the exposure to the direction of the stock.
        - The dividend yield is the trailing yield on the start date, continuously
          compounded, paid on the shares held; the risk free rate is continuously compounded.
        - The approximate standard deviation holds near the money without costs; the
          simulated one also covers other strikes and costs.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY", start_date="2024-01-01", end_date="2025-12-31"
        )

        toolkit.options.get_delta_hedging_simulation(
            start_date="2025-12-31",
            rebalancing_days=5,
            transaction_cost=0.0005,
            show_columns=["Option Value", "Mean Profit and Loss", "Standard Deviation", "Approximate Standard Deviation"],
        )
        ```

        Which returns:

        |      |   Option Value |   Mean Profit and Loss |   Standard Deviation |   Approximate Standard Deviation |
        |:-----|---------------:|-----------------------:|---------------------:|---------------------------------:|
        | AAPL |        18.7557 |                -0.2161 |               4.1012 |                           4.2758 |
        | MSFT |        25.3213 |                -0.3942 |               5.4501 |                           5.6839 |
        """
        if days < 1 or rebalancing_days < 1:
            raise InvalidArgumentError(
                "The days and rebalancing days must be at least 1."
            )
        if moneyness <= 0:
            raise InvalidArgumentError("The moneyness must be positive.")
        validate_window("simulations", simulations)
        # Checked once for every ticker rather than reported per ticker. A volatility of zero
        # is valid: the stock then grows at the risk free rate less the dividend yield, and an
        # option sold at zero volatility is worth its discounted intrinsic value.
        try:
            validate_positive_inputs(
                allow_zero=True,
                volatility=volatility,
                implied_volatility=implied_volatility,
                transaction_cost=transaction_cost,
            )
        except ValueError as error:
            raise InvalidArgumentError(str(error)) from None

        rows = {}
        for ticker in self._tickers:
            try:
                history = self._prices[ticker].dropna()
                if start_date is None:
                    date = history.index[-1]
                else:
                    position = (
                        history.index.searchsorted(
                            pd.Period(start_date, freq="D"), side="right"
                        )
                        - 1
                    )
                    if position < 0:
                        logger.warning(
                            "%s has no price on or before %s, so it is left out.",
                            ticker,
                            start_date,
                        )
                        continue
                    date = history.index[position]
                stock_price = float(history.loc[date])
                sigma, rate, dividend_yield = self._hedge_inputs(
                    ticker, date, volatility
                )
                implied = sigma if implied_volatility is None else implied_volatility
                strike_price = moneyness * stock_price
                simulation = hedging_model.get_delta_hedging_simulation(
                    stock_price=stock_price,
                    strike_price=strike_price,
                    time_to_expiration=days / 252,
                    risk_free_rate=rate,
                    volatility=sigma,
                    implied_volatility=implied,
                    dividend_yield=dividend_yield,
                    put_option=put_option,
                    transaction_cost=transaction_cost,
                    steps=days,
                    rebalancing_interval=rebalancing_days,
                    simulations=simulations,
                    seed=seed,
                )
                profit = simulation["Profit and Loss"]
                rows[ticker] = {
                    "Stock Price": stock_price,
                    "Strike Price": strike_price,
                    "Volatility": sigma,
                    "Implied Volatility": implied,
                    "Option Value": float(simulation["Option Value"].iloc[0]),
                    "Mean Profit and Loss": float(profit.mean()),
                    "Standard Deviation": float(profit.std()),
                    "5% Quantile": float(profit.quantile(0.05)),
                    "95% Quantile": float(profit.quantile(0.95)),
                    "Mean Transaction Costs": float(
                        simulation["Transaction Costs"].mean()
                    ),
                    "Approximate Standard Deviation": hedging_model.get_hedging_error_approximation(
                        stock_price=stock_price,
                        strike_price=strike_price,
                        time_to_expiration=days / 252,
                        risk_free_rate=rate,
                        volatility=sigma,
                        rebalances=int(np.ceil(days / rebalancing_days)),
                        dividend_yield=dividend_yield,
                    ),
                }
            except InvalidArgumentError:
                raise
            except ValueError as error:
                # One ticker the formula cannot value (such as a volatility of zero from constant
                # prices) is left out rather than emptying the result of every ticker.
                logger.warning("No delta hedging simulation for %s: %s", ticker, error)
            except (IndexError, KeyError):
                # Too few prices for a volatility (or none at all) leaves this ticker out only.
                logger.warning(
                    "No delta hedging simulation for %s: it has too few prices for a "
                    "volatility on the start date.",
                    ticker,
                )

        if not rows:
            raise ValueError(
                "No ticker has a price and volatility to simulate the hedge from."
            )

        return apply_rounding(
            pd.DataFrame(rows).T, rounding if rounding is not None else self._rounding
        )
