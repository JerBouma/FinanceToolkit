"""Option Market Metrics Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd

from financetoolkit.options import greeks_model

# pylint: disable=too-many-arguments,too-many-locals

# One option contract covers one hundred shares.
CONTRACT_SIZE = 100

# The delta that defines the wings of the volatility skew.
SKEW_DELTA = 0.25

# The expiries used have at least a week to expiration; the Cboe uses the Friday expiries
# between 23 and 37 days away for the VIX.
MINIMUM_DAYS = 7


def get_mid_prices(chain: pd.DataFrame) -> pd.Series:
    """
    The price of every option: the middle of the bid and ask when both are quoted, the
    last traded price otherwise (e.g. outside market hours, when quotes are zero).

    Args:
        chain (pd.DataFrame): The options of one expiry, indexed by strike, with Bid, Ask and
            Last Price columns.

    Returns:
        pd.Series: The price per strike.
    """
    bid = pd.to_numeric(chain.get("Bid"), errors="coerce")
    ask = pd.to_numeric(chain.get("Ask"), errors="coerce")
    last = pd.to_numeric(chain.get("Last Price"), errors="coerce")
    quoted = (bid > 0) & (ask > 0) & (ask >= bid)
    return ((bid + ask) / 2).where(quoted, last)


def _positive_volatility(chain: pd.DataFrame) -> pd.Series:
    """The quoted implied volatilities above zero, averaged per strike."""
    volatility = pd.to_numeric(chain["Implied Volatility"], errors="coerce")
    return volatility[volatility > 0].groupby(level=0).mean()


def get_put_call_ratio(calls: pd.DataFrame, puts: pd.DataFrame) -> pd.Series:
    """
    The put-call ratio by volume and by open interest. A ratio above one means more puts
    than calls are traded or held, usually read as hedging demand or bearish sentiment.

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
    The max pain strike: the expiry price at which the options outstanding would pay their
    holders the least, the sum over all calls of OI x max(S - K, 0) and over all puts of
    OI x max(K - S, 0) being smallest. Some traders expect the price to gravitate there as
    expiry nears.

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


def get_gamma_exposure(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    stock_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
) -> pd.DataFrame:
    """
    The dealer gamma exposure per strike: the dollar amount of shares market makers would
    have to buy or sell for a 1% move in the underlying to stay delta-hedged, assuming they
    are long the calls and short the puts that investors trade (the common convention):

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
    The gamma flip point: the price of the underlying at which the total dealer gamma
    exposure over all expiries changes sign, found by recalculating the exposure over a
    range of prices. Above it dealers dampen moves, below it they amplify them.

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
    The at-the-money implied volatility: the implied volatility of calls and puts
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
    The volatility skew of one expiry, measured on the 25-delta options:

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


def get_implied_move(
    calls: pd.DataFrame, puts: pd.DataFrame, stock_price: float
) -> pd.Series:
    """
    The move the options market expects up to an expiry: the price of the at-the-money
    straddle (a call and a put at the strike closest to the current price) relative to the
    price. The straddle prices the expected absolute move, √(2/π) ≈ 0.8 standard
    deviations (Brenner and Subrahmanyam, 1988), so under a lognormal model about 57% of
    outcomes fall within one implied move and one standard deviation is about 1.25 times
    the straddle.

    Args:
        calls (pd.DataFrame): The calls of one expiry, indexed by strike.
        puts (pd.DataFrame): The puts of the same expiry.
        stock_price (float): The price of the underlying.

    Returns:
        pd.Series: The Straddle Strike, Straddle Price, Implied Move (as a fraction of the
        price) and the Lower and Upper Price of the expected range.
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

    return pd.Series(
        {
            "Straddle Strike": strike,
            "Straddle Price": straddle,
            "Implied Move": straddle / stock_price,
            "Lower Price": stock_price - straddle,
            "Upper Price": stock_price + straddle,
        }
    )


def get_model_free_variance(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    time_to_expiration: float,
    risk_free_rate: float,
) -> tuple[float, float]:
    """
    The model-free implied variance of one expiry, as in the Cboe VIX methodology: the
    forward price follows from put-call parity at the strike where call and put prices are
    closest, and the variance is a weighted sum of out-of-the-money option prices,

        σ² = (2/T) Σ (ΔK_i / K_i²) e^(RT) Q(K_i) - (1/T) (F/K_0 - 1)²,

    with K_0 the first strike below the forward and Q the price of the out-of-the-money put
    (below K_0) or call (above), averaged at K_0. Quotes stop after two consecutive strikes
    without a bid, as in the methodology.

    Args:
        calls (pd.DataFrame): The calls of one expiry, indexed by strike, with Bid and Ask.
        puts (pd.DataFrame): The puts of the same expiry.
        time_to_expiration (float): The time to expiration in years.
        risk_free_rate (float): The risk free rate.

    Returns:
        tuple[float, float]: The annualized variance and the forward price.
    """
    call_prices = get_mid_prices(calls).groupby(level=0).mean()
    put_prices = get_mid_prices(puts).groupby(level=0).mean()
    call_bids = pd.to_numeric(calls["Bid"], errors="coerce").groupby(level=0).max()
    put_bids = pd.to_numeric(puts["Bid"], errors="coerce").groupby(level=0).max()

    common = call_prices.dropna().index.intersection(put_prices.dropna().index)
    if len(common) < 3:  # noqa: PLR2004
        return np.nan, np.nan

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
        return np.nan, forward
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
    # Without a strip of bid options on both sides there is no variance to measure: K0
    # alone gives only the (negative) correction term.
    if not puts_used or not calls_used:
        return np.nan, float(forward)

    # Puts below K0, calls above and the average of both at K0.
    prices = np.array(
        [put_prices.get(strike, np.nan) for strike in puts_used]
        + [np.nanmean([put_prices.get(k0, np.nan), call_prices.get(k0, np.nan)])]
        + [call_prices.get(strike, np.nan) for strike in calls_used],
        dtype=float,
    )

    # Both sides hold at least one strike, so there are always three or more.
    strikes_used = np.array(puts_used + [k0] + calls_used)
    spacing = np.empty_like(strikes_used)
    spacing[1:-1] = (strikes_used[2:] - strikes_used[:-2]) / 2
    spacing[0] = strikes_used[1] - strikes_used[0]
    spacing[-1] = strikes_used[-1] - strikes_used[-2]

    contributions = spacing / strikes_used**2 * growth * prices
    variance = (2 / time_to_expiration) * float(np.nansum(contributions)) - (
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
    Interpolate the variances of two expiries to a constant maturity, as the Cboe does for
    the 30-day VIX, and return it as an annualized volatility.

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
