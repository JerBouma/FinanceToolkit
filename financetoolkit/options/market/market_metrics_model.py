"""Option Market Metrics Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd

from financetoolkit.options.greeks import greeks_model
from financetoolkit.utilities import logger_model

logger = logger_model.get_logger()

# One option contract covers one hundred shares.
CONTRACT_SIZE = 100

# The delta that defines the wings of the volatility skew.
SKEW_DELTA = 0.25

# The expiries used have at least a week to expiration; the Cboe uses the Friday expiries
# between 23 and 37 days away for the VIX.
MINIMUM_DAYS = 7


def get_mid_prices(chain: pd.DataFrame) -> pd.Series:
    """
    Calculate the price of every option: the middle of the bid and ask when both are quoted,
    the last traded price otherwise (e.g. outside market hours, when quotes are zero) for an
    option that is still held or traded, which has an open interest or a volume above zero.

    An option without a two-sided quote, open interest or volume has no price: its last
    trade can be weeks old, from when the underlying stood elsewhere, and solving an implied
    volatility from it gives values such as 276% that the market does not quote.

    Args:
        chain (pd.DataFrame): The options of one expiry, indexed by strike, with Bid, Ask and
            Last Price columns, and the Volume and Open Interest columns of the chain. A chain
            without both of these keeps the last traded price wherever there is no quote.

    Returns:
        pd.Series: The price per strike, NaN for an option without a quote or activity.
    """
    bid, ask, quoted = _two_sided_quotes(chain)
    last = pd.to_numeric(chain.get("Last Price"), errors="coerce")

    activity = [
        pd.to_numeric(chain[column], errors="coerce") > 0
        for column in ("Volume", "Open Interest")
        if column in chain.columns
    ]
    if activity:
        last = last.where(np.logical_or.reduce(activity))

    return ((bid + ask) / 2).where(quoted, last)


def _two_sided_quotes(chain: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """The bid and ask of every option and whether both are quoted: above zero, the ask at or above the bid."""  # noqa: E501
    bid = pd.to_numeric(chain.get("Bid"), errors="coerce")
    ask = pd.to_numeric(chain.get("Ask"), errors="coerce")
    return bid, ask, (bid > 0) & (ask > 0) & (ask >= bid)


def _positive_volatility(chain: pd.DataFrame) -> pd.Series:
    """The quoted implied volatilities above zero, averaged per strike."""
    volatility = pd.to_numeric(chain["Implied Volatility"], errors="coerce")
    return volatility[volatility > 0].groupby(level=0).mean()


def get_put_call_ratio(calls: pd.DataFrame, puts: pd.DataFrame) -> pd.Series:
    """
    Calculate the put-call ratio by volume and by open interest: how many puts are traded or
    held per call.

    A ratio above one means more puts than calls are traded or held, usually read as hedging
    demand or bearish sentiment.

    Args:
        calls (pd.DataFrame): The calls of one expiry, with Volume and Open Interest columns.
        puts (pd.DataFrame): The puts of the same expiry.

    Returns:
        pd.Series: The Call Volume, Put Volume, Volume Put-Call Ratio, Call Open Interest,
        Put Open Interest and Open Interest Put-Call Ratio.
    """

    def total(chain: pd.DataFrame, column: str) -> float:
        return float(pd.to_numeric(chain[column], errors="coerce").sum())

    call_volume, put_volume = total(calls, "Volume"), total(puts, "Volume")
    call_interest = total(calls, "Open Interest")
    put_interest = total(puts, "Open Interest")

    return pd.Series(
        {
            "Call Volume": call_volume,
            "Put Volume": put_volume,
            "Volume Put-Call Ratio": (
                put_volume / call_volume if call_volume else np.nan
            ),
            "Call Open Interest": call_interest,
            "Put Open Interest": put_interest,
            "Open Interest Put-Call Ratio": (
                put_interest / call_interest if call_interest else np.nan
            ),
        }
    )


def get_max_pain(calls: pd.DataFrame, puts: pd.DataFrame) -> float:
    """
    Calculate the max pain strike: the expiry price at which the options outstanding would
    pay their holders the least, the sum over all calls of OI x max(S - K, 0) and over all
    puts of OI x max(K - S, 0) being smallest.

    Some traders expect the price to gravitate there as expiry nears.

    Args:
        calls (pd.DataFrame): The calls of one expiry, indexed by strike, with Open Interest.
        puts (pd.DataFrame): The puts of the same expiry.

    Returns:
        float: The max pain strike, or NaN when there is no open interest.
    """
    call_interest = pd.to_numeric(calls["Open Interest"], errors="coerce").fillna(0)
    put_interest = pd.to_numeric(puts["Open Interest"], errors="coerce").fillna(0)
    strikes = np.unique(np.concatenate([call_interest.index, put_interest.index]))

    if call_interest.sum() + put_interest.sum() == 0:
        return np.nan

    call_strikes = call_interest.index.to_numpy(dtype=float)
    put_strikes = put_interest.index.to_numpy(dtype=float)
    payout = [
        float(
            (call_interest.to_numpy() * np.maximum(price - call_strikes, 0)).sum()
            + (put_interest.to_numpy() * np.maximum(put_strikes - price, 0)).sum()
        )
        for price in strikes
    ]

    return float(strikes[int(np.argmin(payout))])


def get_max_pain_distance(
    calls: pd.DataFrame, puts: pd.DataFrame, stock_price: float
) -> pd.Series:
    """
    Calculate the max pain strike of one expiry and how far it lies from the stock price.

    Args:
        calls (pd.DataFrame): The calls of one expiry, indexed by strike, with Open Interest.
        puts (pd.DataFrame): The puts of the same expiry.
        stock_price (float): The price of the underlying.

    Returns:
        pd.Series: The Max Pain strike, the Stock Price and the Distance to Max Pain as a
        fraction of the price.
    """
    max_pain = get_max_pain(calls, puts)
    return pd.Series(
        {
            "Max Pain": max_pain,
            "Stock Price": stock_price,
            "Distance to Max Pain": max_pain / stock_price - 1,
        }
    )


def get_gamma_exposure(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    stock_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
) -> pd.DataFrame:
    """
    Calculate the dealer gamma exposure per strike: the dollar amount of shares market
    makers would have to buy or sell for a 1% move in the underlying to stay delta-hedged,
    assuming they are long the calls and short the puts that investors trade (the common
    convention):

        Gamma Exposure = ±Gamma x Open Interest x 100 x Price² x 0.01

    positive for calls and negative for puts. Positive total exposure means dealers trade
    against moves, which dampens volatility; negative exposure means they trade with it.

    For more information, see the following sources:

    - SqueezeMetrics (2017). "Gamma Exposure (GEX)." White paper.
    - Perfiliev, S. (2021). "How to Calculate Gamma Exposure (GEX) and Zero Gamma Level."
      https://perfiliev.com/blog/how-to-calculate-gamma-exposure-and-zero-gamma-level/

    Args:
        calls (pd.DataFrame): The calls of one expiry, indexed by strike, with Open Interest
            and Implied Volatility.
        puts (pd.DataFrame): The puts of the same expiry.
        stock_price (float): The price of the underlying.
        time_to_expiration (float): The time to expiration in years.
        risk_free_rate (float): The risk free rate.
        dividend_yield (float, optional): The dividend yield. Defaults to 0.

    Returns:
        pd.DataFrame: The Call, Put and Net Gamma Exposure per strike.
    """

    def exposure(chain: pd.DataFrame, sign: int) -> pd.Series:
        volatility = pd.to_numeric(chain["Implied Volatility"], errors="coerce")
        interest = pd.to_numeric(chain["Open Interest"], errors="coerce").fillna(0)
        valid = volatility > 0
        strikes = chain.index.to_numpy(dtype=float)
        gamma = np.where(
            valid,
            greeks_model.get_gamma(
                stock_price=float(stock_price),
                strike_price=strikes,
                time_to_expiration=float(time_to_expiration),
                risk_free_rate=float(risk_free_rate),
                volatility=volatility.where(valid, 1.0).to_numpy(dtype=float),
                dividend_yield=float(dividend_yield),
            ),
            0.0,
        )
        values = (
            sign * gamma * interest.to_numpy() * CONTRACT_SIZE * stock_price**2 * 0.01
        )
        return pd.Series(values, index=chain.index).groupby(level=0).sum()

    call_exposure = exposure(calls, 1)
    put_exposure = exposure(puts, -1)
    table = pd.DataFrame(
        {"Call Gamma Exposure": call_exposure, "Put Gamma Exposure": put_exposure}
    ).fillna(0)
    table["Net Gamma Exposure"] = (
        table["Call Gamma Exposure"] + table["Put Gamma Exposure"]
    )
    table.index.name = "Strike Price"

    return table.sort_index()


def get_gamma_flip(
    expiries: list[tuple[pd.DataFrame, pd.DataFrame, float]],
    stock_price: float,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
    price_range: float = 0.2,
    points: int = 201,
) -> float:
    """
    Calculate the gamma flip point: the price of the underlying at which the total dealer
    gamma exposure over all expiries changes sign, found by recalculating the exposure over
    a range of prices.

    Above it dealers dampen moves, below it they amplify them.

    Args:
        expiries (list[tuple[pd.DataFrame, pd.DataFrame, float]]): Per expiry the calls, the
            puts and the time to expiration in years.
        stock_price (float): The price of the underlying.
        risk_free_rate (float): The risk free rate.
        dividend_yield (float, optional): The dividend yield. Defaults to 0.
        price_range (float, optional): How far around the current price to search, as a
            fraction. Defaults to 0.2.
        points (int, optional): The number of prices to evaluate. Defaults to 201.

    Returns:
        float: The price where the exposure crosses zero closest to the current price, or
        NaN when it does not cross within the range.
    """
    prices = np.linspace(
        stock_price * (1 - price_range), stock_price * (1 + price_range), points
    )
    totals = np.array(
        [
            sum(
                get_gamma_exposure(
                    calls, puts, price, time, risk_free_rate, dividend_yield
                )["Net Gamma Exposure"].sum()
                for calls, puts, time in expiries
            )
            for price in prices
        ]
    )
    crossings = np.where(np.diff(np.sign(totals)) != 0)[0]

    if crossings.size == 0:
        return np.nan

    # Linear interpolation between the two prices around each crossing.
    flips: list[float] = [
        float(
            prices[i]
            - totals[i] * (prices[i + 1] - prices[i]) / (totals[i + 1] - totals[i])
        )
        for i in crossings
    ]
    return min(flips, key=lambda flip: abs(flip - stock_price))


def get_at_the_money_volatility(
    calls: pd.DataFrame, puts: pd.DataFrame, stock_price: float
) -> float:
    """
    Calculate the at-the-money implied volatility: the implied volatility of calls and puts
    interpolated at the current price and averaged.

    Args:
        calls (pd.DataFrame): The calls of one expiry, indexed by strike.
        puts (pd.DataFrame): The puts of the same expiry.
        stock_price (float): The price of the underlying.

    Returns:
        float: The at-the-money implied volatility, or NaN.
    """
    values = []
    for chain in (calls, puts):
        volatility = _positive_volatility(chain).sort_index()
        if len(volatility) >= 2:  # noqa: PLR2004
            values.append(
                float(
                    np.interp(
                        stock_price,
                        volatility.index.to_numpy(dtype=float),
                        volatility.to_numpy(),
                    )
                )
            )
    return float(np.mean(values)) if values else np.nan


def get_volatility_skew(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    stock_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
) -> pd.Series:
    """
    Calculate the volatility skew of one expiry: how much more the market pays for downside
    than for upside protection, measured on the 25-delta options:

    - 25-Delta Risk Reversal = IV(25-delta call) - IV(25-delta put): negative when puts are
      dearer than calls, the usual case for equities as investors pay for crash protection.
    - 25-Delta Butterfly = (IV(25-delta call) + IV(25-delta put)) / 2 - ATM IV: the
      curvature of the smile, the price of tail risk on both sides.

    The deltas follow from each option's own implied volatility (Black-Scholes), and the
    implied volatility at exactly 25 delta is interpolated.

    Args:
        calls (pd.DataFrame): The calls of one expiry, indexed by strike.
        puts (pd.DataFrame): The puts of the same expiry.
        stock_price (float): The price of the underlying.
        time_to_expiration (float): The time to expiration in years.
        risk_free_rate (float): The risk free rate.
        dividend_yield (float, optional): The dividend yield. Defaults to 0.

    Returns:
        pd.Series: The 25-Delta Put Volatility, At-the-Money Volatility, 25-Delta Call
        Volatility, 25-Delta Risk Reversal and 25-Delta Butterfly.
    """

    def wing(chain: pd.DataFrame, put_option: bool) -> float:
        volatility = _positive_volatility(chain)
        if len(volatility) < 2:  # noqa: PLR2004
            return np.nan
        delta = np.abs(
            greeks_model.get_delta(
                stock_price=float(stock_price),
                strike_price=volatility.index.to_numpy(dtype=float),
                time_to_expiration=float(time_to_expiration),
                risk_free_rate=float(risk_free_rate),
                volatility=volatility.to_numpy(dtype=float),
                dividend_yield=float(dividend_yield),
                put_option=put_option,
            )
        )
        order = np.argsort(delta)
        delta, values = delta[order], volatility.to_numpy()[order]
        if not delta.min() <= SKEW_DELTA <= delta.max():
            return np.nan
        return float(np.interp(SKEW_DELTA, delta, values))

    put_wing = wing(puts, put_option=True)
    call_wing = wing(calls, put_option=False)
    at_the_money = get_at_the_money_volatility(calls, puts, stock_price)

    return pd.Series(
        {
            "25-Delta Put Volatility": put_wing,
            "At-the-Money Volatility": at_the_money,
            "25-Delta Call Volatility": call_wing,
            "25-Delta Risk Reversal": call_wing - put_wing,
            "25-Delta Butterfly": (call_wing + put_wing) / 2 - at_the_money,
        }
    )


def get_straddle_quotes(
    calls: pd.DataFrame, puts: pd.DataFrame, strike: float
) -> pd.DataFrame:
    """
    Show how the call and the put at one strike are priced: from a two-sided quote, or
    from their last trade when there is none (see get_mid_prices), with the bid-ask spread
    of a quoted leg and the last trade date and open interest the chain reports.

    Strikes listed more than once are combined as get_implied_move averages their prices:
    a leg is priced from its last trade when any of its priced options at the strike is
    (and then has no spread), otherwise its spread is their average spread; the last trade
    date is the latest and the open interest the sum.

    Args:
        calls (pd.DataFrame): The calls of one expiry, indexed by strike, with Bid, Ask and
            Last Price columns, and where the chain has them the Volume, Open Interest and
            Last Trade Date columns.
        puts (pd.DataFrame): The puts of the same expiry.
        strike (float): The strike of the call and the put.

    Returns:
        pd.DataFrame: Per leg ("Call" and "Put") whether it is priced from its Last Trade
        (1 or 0), its Bid-Ask Spread (in price, NaN without a two-sided quote), the Last
        Trade Date and the Open Interest (NaN when the chain has no such column).
    """
    legs = {}
    for leg, chain in (("Call", calls), ("Put", puts)):
        bid, ask, quoted = _two_sided_quotes(chain)
        at_strike = pd.Series(chain.index == strike, index=chain.index)
        priced = get_mid_prices(chain).notna() & at_strike
        last_trade = priced & ~quoted
        open_interest = pd.to_numeric(
            chain.get("Open Interest", pd.Series(np.nan, index=chain.index)),
            errors="coerce",
        )
        trade_dates = chain.get("Last Trade Date", pd.Series(np.nan, index=chain.index))
        legs[leg] = {
            "Last Trade": float(last_trade.any()),
            "Bid-Ask Spread": (
                np.nan if last_trade.any() else (ask - bid)[priced & quoted].mean()
            ),
            "Last Trade Date": trade_dates[at_strike].dropna().max(),
            "Open Interest": open_interest[at_strike].sum(min_count=1),
        }
    return pd.DataFrame.from_dict(legs, orient="index")


def get_implied_move(
    calls: pd.DataFrame, puts: pd.DataFrame, stock_price: float
) -> pd.Series:
    """
    Calculate the move the options market expects up to an expiry: the price of the
    at-the-money straddle (a call and a put at the strike closest to the current price)
    relative to the price.

    The straddle prices the expected absolute move, √(2/π) ≈ 0.8 standard deviations
    (Brenner and Subrahmanyam, 1988), so under a lognormal model about 57% of outcomes fall
    within one implied move and one standard deviation is about 1.25 times the straddle.

    The formula is as follows:

        - Straddle Price = Call Price + Put Price
        - Bid-Ask Spread = (Call Ask − Call Bid + Put Ask − Put Bid) / Straddle Price
        - Implied Move = Straddle Price / Stock Price

    Args:
        calls (pd.DataFrame): The calls of one expiry, indexed by strike.
        puts (pd.DataFrame): The puts of the same expiry.
        stock_price (float): The price of the underlying.

    Returns:
        pd.Series: The Straddle Strike, Straddle Price, Bid-Ask Spread (as a fraction of
        the straddle price), Last Trade Legs, Implied Move (as a fraction of the price) and
        the Lower and Upper Price of the expected range.

    Notes:
    - The prices are the middle of the bid and ask, or the last traded price of an option
      without a two-sided quote that is still held or traded (see get_mid_prices).
    - The Bid-Ask Spread shows how firmly the straddle is quoted: both legs' spreads as a
      fraction of the straddle price, NaN when a leg has no two-sided quote. Above 0.5 the
      spread is wider than half the straddle, and the middle can lie far from where it trades.
    - The Last Trade Legs counts how many of the two legs (0, 1 or 2) are priced from their
      last trade instead of a quote. A last trade can be days or weeks old, from when the
      underlying stood elsewhere, so the implied move then may not reflect the live market.
    """
    call_prices = get_mid_prices(calls).groupby(level=0).mean()
    put_prices = get_mid_prices(puts).groupby(level=0).mean()
    strikes = call_prices.dropna().index.intersection(put_prices.dropna().index)

    if strikes.empty:
        return pd.Series(
            dict.fromkeys(
                [
                    "Straddle Strike",
                    "Straddle Price",
                    "Bid-Ask Spread",
                    "Last Trade Legs",
                    "Implied Move",
                    "Lower Price",
                    "Upper Price",
                ],
                np.nan,
            )
        )

    strike = float(
        strikes[np.argmin(np.abs(strikes.to_numpy(dtype=float) - stock_price))]
    )
    straddle = float(call_prices[strike] + put_prices[strike])
    legs = get_straddle_quotes(calls, puts, strike)

    return pd.Series(
        {
            "Straddle Strike": strike,
            "Straddle Price": straddle,
            # NaN unless both legs are quoted: a spread of one leg alone understates it.
            "Bid-Ask Spread": legs["Bid-Ask Spread"].astype(float).sum(min_count=2)
            / straddle,
            "Last Trade Legs": legs["Last Trade"].astype(float).sum(),
            "Implied Move": straddle / stock_price,
            "Lower Price": stock_price - straddle,
            "Upper Price": stock_price + straddle,
        }
    )


def get_model_free_strip(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    time_to_expiration: float,
    risk_free_rate: float,
) -> tuple[pd.DataFrame, float]:
    """
    Calculate the strikes of one expiry that the Cboe VIX and SKEW methodologies use, and
    the forward price: the out-of-the-money options with a bid around the forward that the
    model-free measures sum over.

    The forward price follows from put-call parity at the strike where call and put prices
    are closest, F = K + e^(RT) (C - P). K_0 is the first strike at or below the forward;
    from there the puts below and the calls above are added strike by strike while they have
    a bid, stopping after two consecutive strikes without one.

    Args:
        calls (pd.DataFrame): The calls of one expiry, indexed by strike, with Bid and Ask.
        puts (pd.DataFrame): The puts of the same expiry.
        time_to_expiration (float): The time to expiration in years.
        risk_free_rate (float): The continuously compounded risk free rate.

    Returns:
        tuple[pd.DataFrame, float]: The Call Price and Put Price (see get_mid_prices) of the
        strikes used, indexed by strike in ascending order, and the forward price. Without
        a strip of puts and calls with a bid on both sides of K_0 the strikes are empty,
        and without three strikes quoted on both sides the forward price is NaN as well.
    """
    call_prices = get_mid_prices(calls).groupby(level=0).mean()
    put_prices = get_mid_prices(puts).groupby(level=0).mean()
    call_bids = pd.to_numeric(calls["Bid"], errors="coerce").groupby(level=0).max()
    put_bids = pd.to_numeric(puts["Bid"], errors="coerce").groupby(level=0).max()
    empty = pd.DataFrame(columns=["Call Price", "Put Price"], dtype=float)

    common = call_prices.dropna().index.intersection(put_prices.dropna().index)
    if len(common) < 3:  # noqa: PLR2004
        return empty, np.nan

    difference = (call_prices[common] - put_prices[common]).abs()
    parity_strike = float(difference.idxmin())
    growth = np.exp(risk_free_rate * time_to_expiration)
    forward = parity_strike + growth * float(
        call_prices[parity_strike] - put_prices[parity_strike]
    )

    strikes = np.array(
        sorted(set(call_prices.index) | set(put_prices.index)), dtype=float
    )
    below = strikes[strikes <= forward]
    if below.size == 0:
        return empty, float(forward)
    k0 = float(below.max())

    def selected(side_strikes: np.ndarray, bids: pd.Series) -> list[float]:
        # Walk away from K0 and stop after two strikes in a row without a bid.
        chosen, misses = [], 0
        for strike in side_strikes:
            if bids.get(strike, 0) > 0:
                chosen.append(float(strike))
                misses = 0
            else:
                misses += 1
                if misses >= 2:  # noqa: PLR2004
                    break
        return chosen

    puts_used = selected(strikes[strikes < k0][::-1], put_bids)[::-1]
    calls_used = selected(strikes[strikes > k0], call_bids)
    # Without a strip of bid options on both sides there is nothing to measure: K0 alone
    # gives only the correction terms.
    if not puts_used or not calls_used:
        return empty, float(forward)

    strikes_used = pd.Index(puts_used + [k0] + calls_used, name="Strike Price")
    strip = pd.DataFrame(
        {
            "Call Price": call_prices.reindex(strikes_used),
            "Put Price": put_prices.reindex(strikes_used),
        },
        index=strikes_used,
    )
    return strip, float(forward)


def get_out_of_the_money_prices(
    strike_prices: np.ndarray,
    call_prices: np.ndarray,
    put_prices: np.ndarray,
    forward_price: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Calculate the out-of-the-money option prices Q(K_i) and the strike intervals ΔK_i: the
    terms that discretise the integrals over option prices of the Cboe VIX and SKEW
    methodologies:

        - K_0 = the highest strike at or below the forward price
        - Q(K_i) = the put price for K_i < K_0, the call price for K_i > K_0 and the average
          of both at K_0
        - ΔK_i = (K_(i+1) - K_(i-1)) / 2, and the distance to the neighbouring strike at the
          lowest and highest strike

    Strikes without a price for their side are left out before the intervals are taken, so
    that their neighbours cover the gap.

    Args:
        strike_prices (np.ndarray): The strikes.
        call_prices (np.ndarray): The call price at each strike, NaN where there is none.
        put_prices (np.ndarray): The put price at each strike, NaN where there is none.
        forward_price (float): The forward price of the underlying for the expiry.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, float]: The strikes in ascending order, their
        prices Q(K_i) and intervals ΔK_i, and K_0. The arrays are empty when fewer than three
        strikes are priced or no strike lies at or below the forward price.
    """
    strikes = np.asarray(strike_prices, dtype=float)
    order = np.argsort(strikes)
    strikes = strikes[order]
    calls = np.asarray(call_prices, dtype=float)[order]
    puts = np.asarray(put_prices, dtype=float)[order]
    nothing = (np.array([]), np.array([]), np.array([]), np.nan)

    below = strikes[strikes <= forward_price]
    if below.size == 0:
        return nothing
    k0 = float(below.max())

    # At K0 the average of the put and the call, or the one of the two that has a price.
    average = np.where(
        np.isnan(puts), calls, np.where(np.isnan(calls), puts, (puts + calls) / 2)
    )
    prices = np.where(strikes < k0, puts, np.where(strikes == k0, average, calls))
    priced = np.isfinite(prices)
    strikes, prices = strikes[priced], prices[priced]
    if strikes.size < 3:  # noqa: PLR2004
        return nothing

    spacing = np.empty_like(strikes)
    spacing[1:-1] = (strikes[2:] - strikes[:-2]) / 2
    spacing[0] = strikes[1] - strikes[0]
    spacing[-1] = strikes[-1] - strikes[-2]
    return strikes, prices, spacing, k0


def get_model_free_variance(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    time_to_expiration: float,
    risk_free_rate: float,
) -> tuple[float, float]:
    """
    Calculate the model-free implied variance of one expiry, as in the Cboe VIX methodology:
    the variance the option prices imply without a pricing model, a weighted sum of
    out-of-the-money option prices,

        σ² = (2/T) Σ (ΔK_i / K_i²) e^(RT) Q(K_i) - (1/T) (F/K_0 - 1)²,

    with the forward price F, K_0 and the strikes from get_model_free_strip, and Q(K_i) and
    ΔK_i from get_out_of_the_money_prices. R is the continuously compounded risk free rate.

    Args:
        calls (pd.DataFrame): The calls of one expiry, indexed by strike, with Bid and Ask.
        puts (pd.DataFrame): The puts of the same expiry.
        time_to_expiration (float): The time to expiration in years.
        risk_free_rate (float): The continuously compounded risk free rate.

    Returns:
        tuple[float, float]: The annualized variance and the forward price.
    """
    strip, forward = get_model_free_strip(
        calls, puts, time_to_expiration, risk_free_rate
    )
    if strip.empty:
        return np.nan, forward

    strikes, prices, spacing, k0 = get_out_of_the_money_prices(
        strip.index.to_numpy(dtype=float),
        strip["Call Price"].to_numpy(),
        strip["Put Price"].to_numpy(),
        forward,
    )
    if strikes.size == 0:
        return np.nan, forward

    growth = np.exp(risk_free_rate * time_to_expiration)
    contributions = spacing / strikes**2 * growth * prices
    variance = (2 / time_to_expiration) * float(contributions.sum()) - (
        1 / time_to_expiration
    ) * (forward / k0 - 1) ** 2

    return float(variance), float(forward)


def get_constant_maturity_volatility(
    near_variance: float,
    near_days: float,
    next_variance: float,
    next_days: float,
    target_days: int = 30,
) -> float:
    """
    Calculate the constant maturity volatility: the variances of two expiries interpolated
    to a constant maturity, as the Cboe does for the 30-day VIX, returned as an annualized
    volatility.

    Args:
        near_variance (float): The annualized variance of the near expiry.
        near_days (float): The days to the near expiry.
        next_variance (float): The annualized variance of the next expiry.
        next_days (float): The days to the next expiry.
        target_days (int, optional): The constant maturity in days. Defaults to 30.

    Returns:
        float: The annualized volatility at the constant maturity.
    """
    t1, t2, t = near_days / 365, next_days / 365, target_days / 365
    # Both terms need a valid variance; one alone, weighted, looks like a valid figure.
    if not (
        np.isfinite(near_variance)
        and near_variance > 0
        and np.isfinite(next_variance)
        and next_variance > 0
    ):
        return np.nan
    if next_days == near_days:
        return float(np.sqrt(near_variance))
    weight_near = (next_days - target_days) / (next_days - near_days)
    weight_next = (target_days - near_days) / (next_days - near_days)
    variance = (t1 * near_variance * weight_near + t2 * next_variance * weight_next) / t
    return float(np.sqrt(variance)) if variance > 0 else np.nan


def get_risk_neutral_moments(
    strike_prices: np.ndarray,
    call_prices: np.ndarray,
    put_prices: np.ndarray,
    forward_price: float,
    risk_free_rate: float,
    time_to_expiration: float,
) -> pd.Series:
    """
    Calculate the risk-neutral volatility, skewness and kurtosis of the log return up to an
    expiry: the shape of the return distribution that option prices imply, from the prices
    of the quadratic, cubic and quartic contracts of Bakshi, Kapadia and Madan (2003).

    Each contract is a strip of out-of-the-money options (Carr and Madan, 2001). With R =
    ln(S_T / K_0), y_i = ln(K_i / K_0) and Q(K_i) and ΔK_i from get_out_of_the_money_prices,
    the integrals over option prices are summed like the VIX:

        - V = Σ 2 (1 - y_i) Q(K_i) ΔK_i / K_i²
        - W = Σ (6 y_i - 3 y_i²) Q(K_i) ΔK_i / K_i²
        - X = Σ (12 y_i² - 4 y_i³) Q(K_i) ΔK_i / K_i²
        - μ = F / K_0 - 1 - e^(rT) V / 2 - e^(rT) W / 6 - e^(rT) X / 24
        - Skewness = (e^(rT) W - 3 μ e^(rT) V + 2 μ³) / (e^(rT) V - μ²)^(3/2)
        - Kurtosis = (e^(rT) X - 4 μ e^(rT) W + 6 e^(rT) μ² V - 3 μ⁴) / (e^(rT) V - μ²)²
        - Volatility = sqrt((e^(rT) V - μ²) / T)

    e^(rT) V, e^(rT) W and e^(rT) X are E[R²], E[R³] and E[R⁴] under the risk-neutral
    measure, and μ = E[R] follows from the expansion of E[e^R] = F / K_0 up to the fourth
    power, as in the paper. Bakshi, Kapadia and Madan expand around the spot price without
    dividends, where F / S = e^(rT); expanding around K_0, the strike that separates the
    puts from the calls, keeps the replication exact and takes in dividends through the
    forward price. Skewness and kurtosis do not depend on that choice, since moments about
    the mean are the same for every expansion point, and the expected log return is given
    relative to the forward price, E[ln(S_T / F)] = μ - ln(F / K_0), which is -σ²T / 2 for
    a lognormal price.

    For more information, see the following papers:

    - Bakshi, G., Kapadia, N. and Madan, D. (2003). "Stock Return Characteristics, Skew
      Laws, and the Differential Pricing of Individual Equity Options." The Review of
      Financial Studies, 16(1), 101-143.
    - Carr, P. and Madan, D. (2001). "Optimal Positioning in Derivative Securities."
      Quantitative Finance, 1(1), 19-37.

    Args:
        strike_prices (np.ndarray): The strikes.
        call_prices (np.ndarray): The call price at each strike, NaN where there is none.
        put_prices (np.ndarray): The put price at each strike, NaN where there is none.
        forward_price (float): The forward price of the underlying for the expiry.
        risk_free_rate (float): The continuously compounded risk free rate.
        time_to_expiration (float): The time to expiration in years.

    Returns:
        pd.Series: The Expected Log Return (E[ln(S_T / F)], over the time to expiration),
        the annualized Volatility, the Skewness and the Kurtosis (3 for a normal
        distribution) of the log return. NaN when fewer than three strikes are priced or the variance is not positive.
    """
    columns = ["Expected Log Return", "Volatility", "Skewness", "Kurtosis"]
    strikes, prices, spacing, k0 = get_out_of_the_money_prices(
        strike_prices, call_prices, put_prices, forward_price
    )
    if strikes.size == 0:
        return pd.Series(dict.fromkeys(columns, np.nan))

    growth = np.exp(risk_free_rate * time_to_expiration)
    log_moneyness = np.log(strikes / k0)
    weights = prices * spacing / strikes**2
    # The second derivatives of R², R³ and R⁴ with respect to the price at strike K, times K².
    quadratic = float(np.sum(2 * (1 - log_moneyness) * weights))
    cubic = float(np.sum((6 * log_moneyness - 3 * log_moneyness**2) * weights))
    quartic = float(np.sum((12 * log_moneyness**2 - 4 * log_moneyness**3) * weights))

    mean = (
        forward_price / k0
        - 1
        - growth * quadratic / 2
        - growth * cubic / 6
        - growth * quartic / 24
    )
    variance = growth * quadratic - mean**2
    if not variance > 0:
        return pd.Series(dict.fromkeys(columns, np.nan))

    skewness = (
        growth * cubic - 3 * mean * growth * quadratic + 2 * mean**3
    ) / variance**1.5
    kurtosis = (
        growth * quartic
        - 4 * mean * growth * cubic
        + 6 * growth * mean**2 * quadratic
        - 3 * mean**4
    ) / variance**2

    return pd.Series(
        {
            "Expected Log Return": mean - np.log(forward_price / k0),
            "Volatility": float(np.sqrt(variance / time_to_expiration)),
            "Skewness": skewness,
            "Kurtosis": kurtosis,
        }
    )


def get_chain_risk_neutral_moments(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    time_to_expiration: float,
    risk_free_rate: float,
) -> pd.Series:
    """
    Calculate the forward price and the risk-neutral volatility, skewness and kurtosis of one
    expiry from its option chain, through the model-free strip of out-of-the-money options.

    Args:
        calls (pd.DataFrame): The calls of one expiry, indexed by strike.
        puts (pd.DataFrame): The puts of the same expiry.
        time_to_expiration (float): The time to expiration in years.
        risk_free_rate (float): The continuously compounded risk free rate.

    Returns:
        pd.Series: The Forward Price followed by the moments of get_risk_neutral_moments.
    """
    strip, forward = get_model_free_strip(
        calls, puts, time_to_expiration, risk_free_rate
    )
    moments = get_risk_neutral_moments(
        strip.index.to_numpy(dtype=float),
        strip["Call Price"].to_numpy(),
        strip["Put Price"].to_numpy(),
        forward,
        risk_free_rate,
        time_to_expiration,
    )
    return pd.concat([pd.Series({"Forward Price": forward}), moments])


def get_cboe_skewness(
    strike_prices: np.ndarray,
    call_prices: np.ndarray,
    put_prices: np.ndarray,
    forward_price: float,
    risk_free_rate: float,
    time_to_expiration: float,
) -> float:
    """
    Calculate the price of skewness S of one expiry as in the Cboe SKEW Index white paper
    (2010): the risk-neutral skewness of the log return R = ln(S_T / F_0), from the prices
    P_1, P_2 and P_3 of R, R² and R³.

    Each price is a strip of out-of-the-money options. With y_i = ln(K_i / F_0), Q(K_i) and
    ΔK_i from get_out_of_the_money_prices and K_0 the first strike at or below the forward
    price F_0:

        - P_1 = -e^(rT) Σ Q(K_i) ΔK_i / K_i² + ε_1
        - P_2 = e^(rT) Σ 2 (1 - y_i) Q(K_i) ΔK_i / K_i² + ε_2
        - P_3 = e^(rT) Σ 3 (2 y_i - y_i²) Q(K_i) ΔK_i / K_i² + ε_3
        - ε_1 = -(1 + ln(F_0 / K_0) - F_0 / K_0)
        - ε_2 = 2 ln(K_0 / F_0) (F_0 / K_0 - 1) + ln(K_0 / F_0)²
        - ε_3 = 3 ln(K_0 / F_0)² (ln(K_0 / F_0) / 3 - 1 + F_0 / K_0)
        - S = (P_3 - 3 P_1 P_2 + 2 P_1³) / (P_2 - P_1²)^(3/2)

    The ε terms correct for the expansion around F_0 while the puts and calls are split at
    K_0: ε_n = f(K_0) + f'(K_0) (F_0 - K_0) for the payoff f(S) = ln(S / F_0)^n. The white
    paper prints ε_2 with ln(K_0 / F_0)² / 2, half of the f(K_0) = ln(K_0 / F_0)² of that
    expansion; with the full term the skewness does not depend on whether the expansion is
    centred on F_0 or K_0. The difference is negligible for index options with strikes close
    together (below 10⁻⁴ of P_2), but reaches several percent of P_2 for a stock whose
    forward price lies far from the nearest strike. Unlike the mean of Bakshi, Kapadia and
    Madan (2003), P_1 is the exact price of the log contract, without a truncated expansion.

    For more information, see the following papers:

    - Cboe Global Markets (2010). "The Cboe SKEW Index - SKEW." White paper.
    - Bakshi, G., Kapadia, N. and Madan, D. (2003). "Stock Return Characteristics, Skew
      Laws, and the Differential Pricing of Individual Equity Options." The Review of
      Financial Studies, 16(1), 101-143.

    Args:
        strike_prices (np.ndarray): The strikes.
        call_prices (np.ndarray): The call price at each strike, NaN where there is none.
        put_prices (np.ndarray): The put price at each strike, NaN where there is none.
        forward_price (float): The forward price F_0 of the underlying for the expiry.
        risk_free_rate (float): The continuously compounded risk free rate.
        time_to_expiration (float): The time to expiration in years.

    Returns:
        float: The skewness S, NaN when fewer than three strikes are priced or the variance
        is not positive.
    """
    strikes, prices, spacing, k0 = get_out_of_the_money_prices(
        strike_prices, call_prices, put_prices, forward_price
    )
    if strikes.size == 0:
        return np.nan

    growth = np.exp(risk_free_rate * time_to_expiration)
    log_moneyness = np.log(strikes / forward_price)
    weights = prices * spacing / strikes**2
    # ln(K_0 / F_0) and F_0 / K_0 of the correction terms.
    log_k0 = np.log(k0 / forward_price)
    ratio = forward_price / k0
    epsilon_1 = -(1 - log_k0 - ratio)
    epsilon_2 = 2 * log_k0 * (ratio - 1) + log_k0**2
    epsilon_3 = 3 * log_k0**2 * (log_k0 / 3 - 1 + ratio)

    first = -growth * float(np.sum(weights)) + epsilon_1
    second = growth * float(np.sum(2 * (1 - log_moneyness) * weights)) + epsilon_2
    third = (
        growth * float(np.sum(3 * (2 * log_moneyness - log_moneyness**2) * weights))
        + epsilon_3
    )

    variance = second - first**2
    if not variance > 0:
        return np.nan
    return float((third - 3 * first * second + 2 * first**3) / variance**1.5)


def get_skew_index(
    near_skewness: float,
    near_days: float,
    next_skewness: float,
    next_days: float,
    target_days: int = 30,
) -> tuple[float, float]:
    """
    Calculate the SKEW index: the skewness of two expiries interpolated to a constant
    maturity and turned into an index as the Cboe does for the SKEW Index:

        - w = (T_next - T_target) / (T_next - T_near)
        - S = w S_near + (1 - w) S_next
        - SKEW = 100 - 10 S

    Unlike the variance of the VIX, the skewness is interpolated linearly in time. A target
    outside the two expiries is extrapolated with the same weights. A SKEW of 100 means a
    symmetric distribution of log returns; equity indices usually lie between 110 and 150.

    Args:
        near_skewness (float): The skewness S of the near expiry.
        near_days (float): The days to the near expiry.
        next_skewness (float): The skewness S of the next expiry.
        next_days (float): The days to the next expiry.
        target_days (int, optional): The constant maturity in days. Defaults to 30.

    Returns:
        tuple[float, float]: The skewness at the constant maturity and the SKEW index. NaN
        when either expiry has no skewness.
    """
    if not (np.isfinite(near_skewness) and np.isfinite(next_skewness)):
        return np.nan, np.nan
    if next_days == near_days:
        skewness = float(near_skewness)
    else:
        weight = (next_days - target_days) / (next_days - near_days)
        skewness = float(weight * near_skewness + (1 - weight) * next_skewness)
    return skewness, 100 - 10 * skewness


def get_variance_swap_weights(
    strike_prices: np.ndarray,
    boundary_strike: float,
    time_to_expiration: float,
) -> np.ndarray:
    """
    Calculate the variance swap weights: the number of options at each strike that
    replicates the log contract of a variance swap, from Demeterfi, Derman, Kamal and Zou
    (1999), appendix A.

    The payoff

        - f(S_T) = (2/T) [(S_T - S*) / S* - ln(S_T / S*)]

    is zero with a zero slope at the boundary strike S*, and convex on both sides. Calls at
    the strikes K_0 = S* < K_1 < ... < K_n replicate it above S*, puts at S* = K_0 > K_1 > ...
    > K_n below it, piecewise linear between the strikes:

        - w(K_0) = [f(K_1) - f(K_0)] / |K_1 - K_0|
        - w(K_i) = [f(K_(i+1)) - f(K_i)] / |K_(i+1) - K_i| - Σ_(j<i) w(K_j)

    so each weight is the change in the slope of the payoff at its strike. The weight of the
    outermost strike K_n uses K_(n+1), one interval of the last width further out (half the
    strike for puts whose interval would reach zero), so that the replicated payoff keeps
    growing beyond the last quoted strike as in the paper's example.

    For finely spaced strikes w(K_i) approaches (2/T) ΔK_i / K_i², the weight of the Cboe VIX
    methodology (see get_model_free_variance).

    Args:
        strike_prices (np.ndarray): The strikes of one side, starting at the boundary strike
            and moving away from it: ascending for calls, descending for puts.
        boundary_strike (float): The strike S* that separates the puts from the calls.
        time_to_expiration (float): The time to expiration T in years.

    Returns:
        np.ndarray: The weight of each strike, in the order given. Zero weights when the
        side has fewer than two strikes.
    """
    strikes = np.asarray(strike_prices, dtype=float)
    if strikes.size < 2:  # noqa: PLR2004
        return np.zeros(strikes.size)

    last_interval = strikes[-1] - strikes[-2]
    beyond = strikes[-1] + last_interval
    if beyond <= 0:
        beyond = strikes[-1] / 2
    extended = np.append(strikes, beyond)

    payoff = (2 / time_to_expiration) * (
        (extended - boundary_strike) / boundary_strike
        - np.log(extended / boundary_strike)
    )
    slopes = np.diff(payoff) / np.abs(np.diff(extended))
    return np.concatenate([slopes[:1], np.diff(slopes)])


def get_variance_swap_fair_strike(
    strike_prices: np.ndarray,
    call_prices: np.ndarray,
    put_prices: np.ndarray,
    forward_price: float,
    risk_free_rate: float,
    time_to_expiration: float,
    boundary_strike: float | None = None,
) -> float:
    """
    Calculate the fair variance strike of a variance swap: the annualized variance that
    makes the swap worth nothing today, replicated with the discrete strikes of Demeterfi,
    Derman, Kamal and Zou (1999):

        - K_var = (2/T) [ln(F / S*) - (F / S* - 1)] + e^(rT) [Σ w(K_p) P(K_p) + Σ w(K_c) C(K_c)]

    with the forward price F, the boundary strike S*, the puts at the strikes up to S* and the
    calls from S* and the weights from get_variance_swap_weights. Without dividends F = S_0
    e^(rT), and the first term is the paper's (2/T) [rT - (S_0 e^(rT) / S* - 1) - ln(S* / S_0)].

    The Cboe VIX methodology (get_model_free_variance) discretises the same log contract with
    the midpoint rule, weights (2/T) ΔK / K², and approximates the first term by -(1/T)
    (F / K_0 - 1)². The weights here replicate the payoff exactly at every strike, so the two
    agree for finely spaced strikes and differ by the discretisation error otherwise.

    Referenced against:
    - QuantLib, ReplicatingVarianceSwapEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_replicating_variance_swap_engine.html
    - FinancePy, EquityVarianceSwap.fair_strike: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/equity/equity_variance_swap.py#L102

    Args:
        strike_prices (np.ndarray): The strikes.
        call_prices (np.ndarray): The call price at each strike, NaN where there is none.
        put_prices (np.ndarray): The put price at each strike, NaN where there is none.
        forward_price (float): The forward price F of the underlying for the expiry.
        risk_free_rate (float): The continuously compounded risk free rate.
        time_to_expiration (float): The time to expiration T in years.
        boundary_strike (float | None, optional): The strike S* between the puts and the calls,
            one of the strikes. Defaults to None, the highest strike at or below the forward
            price (K_0 of the Cboe methodology).

    Returns:
        float: The fair variance strike, annualized, as a decimal (0.04 is a volatility of
        20%). NaN when the boundary strike lacks a put or call price or either side has
        fewer than two priced strikes.
    """
    strikes = np.asarray(strike_prices, dtype=float)
    order = np.argsort(strikes)
    strikes = strikes[order]
    calls = np.asarray(call_prices, dtype=float)[order]
    puts = np.asarray(put_prices, dtype=float)[order]

    if boundary_strike is None:
        below = strikes[strikes <= forward_price]
        if below.size == 0:
            return np.nan
        boundary_strike = float(below.max())
    at_boundary = strikes == boundary_strike
    if not (
        np.isfinite(calls[at_boundary]).any() and np.isfinite(puts[at_boundary]).any()
    ):
        return np.nan

    # Strikes without a price for their side are left out, so their neighbours cover the gap.
    call_side = (strikes >= boundary_strike) & np.isfinite(calls)
    put_side = (strikes <= boundary_strike) & np.isfinite(puts)
    call_strikes, put_strikes = strikes[call_side], strikes[put_side][::-1]
    if call_strikes.size < 2 or put_strikes.size < 2:  # noqa: PLR2004
        return np.nan

    call_weights = get_variance_swap_weights(
        call_strikes, boundary_strike, time_to_expiration
    )
    put_weights = get_variance_swap_weights(
        put_strikes, boundary_strike, time_to_expiration
    )
    portfolio = float(
        call_weights @ calls[call_side] + put_weights @ puts[put_side][::-1]
    )
    ratio = forward_price / boundary_strike

    return float(
        (2 / time_to_expiration) * (np.log(ratio) - (ratio - 1))
        + np.exp(risk_free_rate * time_to_expiration) * portfolio
    )


def get_realized_variance(
    prices: pd.Series | pd.DataFrame, trading_days: int = 252
) -> float | pd.Series:
    """
    Calculate the realized variance as a variance swap settles it: the annualized mean of
    the squared daily log returns, without subtracting their mean,

        - σ²_realized = (A / N) Σ ln(S_i / S_(i-1))²

    with N the number of returns and A the trading days per year. Each column keeps its own
    history: missing prices are left out rather than counted as a zero return.

    Shares similar logic:
    - FinancePy, EquityVarianceSwap.realised_variance: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/equity/equity_variance_swap.py#L223

    Args:
        prices (pd.Series | pd.DataFrame): The prices, one column per ticker.
        trading_days (int, optional): The number of returns per year A. Defaults to 252.

    Returns:
        float | pd.Series: The realized variance as a decimal, per column for a DataFrame.
        NaN without any return.
    """
    squared = np.log(prices.where(prices > 0)).diff().iloc[1:] ** 2
    return squared.mean() * trading_days


def get_variance_swap_value(
    realized_variance: float,
    fair_variance: float,
    variance_strike: float,
    elapsed_fraction: float,
    remaining_time: float,
    risk_free_rate: float,
    variance_notional: float = 1.0,
) -> float:
    """
    Calculate the value of a variance swap to the buyer of variance (who receives the
    realized variance and pays the strike) during its life: what the swap is worth from the
    variance realized so far and the fair variance of the remaining time (Demeterfi, Derman,
    Kamal and Zou, 1999).

    The variance it settles on is the realized variance so far and the fair variance of the
    remaining time, weighted by their share of the whole period:

        - Value = N e^(-r (T - t)) [(t / T) σ²_realized + ((T - t) / T) K_var(t, T) - K]

    with the variance notional N, the variance strike K, the realized variance up to t and the
    fair variance strike K_var(t, T) of the remaining time. At the start (t = 0) the value is
    N e^(-rT) (K_var - K), zero for a swap struck at its fair strike.

    A variance notional N pays N (σ² - K) on decimal variances. Quotes in vega notional
    N_vega, the profit of one volatility point (0.01 of decimal volatility) near the strike,
    translate to N = 100 N_vega / (2 sqrt(K)): a vega notional of 100,000 at a 20% strike is
    a variance notional of 25,000,000, which pays 102,500 when 21% is realized.

    Referenced against:
    - FinancePy, EquityVarianceSwap.value: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/equity/equity_variance_swap.py#L61

    Args:
        realized_variance (float): The annualized realized variance since the start.
        fair_variance (float): The fair variance strike of the remaining time.
        variance_strike (float): The variance strike K of the swap, the square of its
            volatility strike.
        elapsed_fraction (float): The share t / T of the swap's observations already made.
        remaining_time (float): The time to maturity T - t in years, for discounting.
        risk_free_rate (float): The continuously compounded risk free rate.
        variance_notional (float, optional): The variance notional N. Defaults to 1.0.

    Returns:
        float: The present value of the swap to the buyer of variance.
    """
    if not 0 <= elapsed_fraction <= 1:
        raise ValueError("The elapsed fraction of the swap must be between 0 and 1.")
    realized_part = (
        elapsed_fraction * realized_variance if elapsed_fraction > 0 else 0.0
    )
    expected_variance = realized_part + (1 - elapsed_fraction) * fair_variance
    return float(
        variance_notional
        * np.exp(-risk_free_rate * remaining_time)
        * (expected_variance - variance_strike)
    )


def get_variance_swap_terms(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    time_to_expiration: float,
    risk_free_rate: float,
    prices: pd.DataFrame,
    ticker: str,
    expiration_date: str,
    strike: float | None = None,
    trade_date: str | None = None,
) -> pd.Series:
    """
    Calculate the fair strike of a variance swap on one expiry and, given a strike, the
    swap's value from the variance realized since the trade date.

    Args:
        calls (pd.DataFrame): The calls of one expiry, indexed by strike.
        puts (pd.DataFrame): The puts of the same expiry.
        time_to_expiration (float): The time to expiration in years.
        risk_free_rate (float): The continuously compounded risk free rate.
        prices (pd.DataFrame): The daily (adjusted) close prices per ticker, with a daily
            PeriodIndex, for the realized variance.
        ticker (str): The ticker of the underlying.
        expiration_date (str): The expiration date of the swap.
        strike (float | None, optional): The volatility strike of the swap. Defaults to None,
            which returns the fair strike only.
        trade_date (str | None, optional): The date the swap was traded on. Defaults to None,
            a swap that starts today.

    Returns:
        pd.Series: The Forward Price, Boundary Strike, Fair Variance, Fair Volatility and Cboe
        Variance and, given a strike, the Variance Strike, Realized Variance, Elapsed Fraction
        and Variance Swap Value.
    """
    strip, forward = get_model_free_strip(
        calls, puts, time_to_expiration, risk_free_rate
    )
    if strip.empty:
        logger.warning(
            "%s has no bid quotes for its %s options to build the strip from.",
            ticker,
            expiration_date,
        )
    below = strip.index[strip.index <= forward]
    boundary = float(below.max()) if len(below) else np.nan
    fair = (
        np.nan
        if strip.empty
        else get_variance_swap_fair_strike(
            strip.index.to_numpy(dtype=float),
            strip["Call Price"].to_numpy(),
            strip["Put Price"].to_numpy(),
            forward,
            risk_free_rate,
            time_to_expiration,
            boundary,
        )
    )
    cboe, _ = get_model_free_variance(calls, puts, time_to_expiration, risk_free_rate)
    result = {
        "Forward Price": forward,
        "Boundary Strike": boundary,
        "Fair Variance": fair,
        "Fair Volatility": np.sqrt(fair) if fair > 0 else np.nan,
        "Cboe Variance": cboe,
    }
    if strike is None:
        return pd.Series(result)

    history = prices[ticker].dropna()
    last_date = history.index[-1].to_timestamp()
    # The weekdays after the last historical date up to and including the expiry.
    remaining = int(
        np.busday_count(
            (last_date + pd.Timedelta(days=1)).date(),
            (pd.Timestamp(expiration_date) + pd.Timedelta(days=1)).date(),
        )
    )
    realized, observed = np.nan, 0
    if trade_date is not None:
        since = history[history.index >= pd.Period(trade_date, freq="D")]
        if since.empty:
            raise ValueError(
                f"The historical data of {ticker} ends before the trade date {trade_date}."
            )
        observed = len(since) - 1
        realized = float(get_realized_variance(since))
    elapsed = observed / (observed + remaining) if observed + remaining else 1.0
    result.update(
        {
            "Variance Strike": strike**2,
            "Realized Variance": realized,
            "Elapsed Fraction": elapsed,
            "Variance Swap Value": get_variance_swap_value(
                realized_variance=realized,
                fair_variance=fair,
                variance_strike=strike**2,
                elapsed_fraction=elapsed,
                remaining_time=time_to_expiration,
                risk_free_rate=risk_free_rate,
            ),
        }
    )
    return pd.Series(result)
