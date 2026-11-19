"""Option Market Metrics Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd

from financetoolkit.options.greeks import greeks_model

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
    last traded price otherwise (e.g. outside market hours, when quotes are zero) for an
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
    bid = pd.to_numeric(chain.get("Bid"), errors="coerce")
    ask = pd.to_numeric(chain.get("Ask"), errors="coerce")
    last = pd.to_numeric(chain.get("Last Price"), errors="coerce")

    activity = [
        pd.to_numeric(chain[column], errors="coerce") > 0
        for column in ("Volume", "Open Interest")
        if column in chain.columns
    ]
    if activity:
        last = last.where(np.logical_or.reduce(activity))

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


def get_model_free_strip(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    time_to_expiration: float,
    risk_free_rate: float,
) -> tuple[pd.DataFrame, float]:
    """
    The strikes of one expiry that the Cboe VIX and SKEW methodologies use, and the forward
    price. The forward price follows from put-call parity at the strike where call and put
    prices are closest, F = K + e^(RT) (C - P). K_0 is the first strike at or below the
    forward; from there the puts below and the calls above are added strike by strike
    while they have a bid, stopping after two consecutive strikes without one.

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
    The out-of-the-money option prices Q(K_i) and the strike intervals ΔK_i that discretise
    the integrals over option prices of the Cboe VIX and SKEW methodologies:

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
    The model-free implied variance of one expiry, as in the Cboe VIX methodology: a
    weighted sum of out-of-the-money option prices,

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


def get_risk_neutral_moments(
    strike_prices: np.ndarray,
    call_prices: np.ndarray,
    put_prices: np.ndarray,
    forward_price: float,
    risk_free_rate: float,
    time_to_expiration: float,
) -> pd.Series:
    """
    The risk-neutral volatility, skewness and kurtosis of the log return up to an expiry,
    from the prices of the quadratic, cubic and quartic contracts of Bakshi, Kapadia and
    Madan (2003), each a strip of out-of-the-money options (Carr and Madan, 2001). With
    R = ln(S_T / K_0), y_i = ln(K_i / K_0) and Q(K_i) and ΔK_i from
    get_out_of_the_money_prices, the integrals over option prices are summed like the VIX:

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


def get_cboe_skewness(
    strike_prices: np.ndarray,
    call_prices: np.ndarray,
    put_prices: np.ndarray,
    forward_price: float,
    risk_free_rate: float,
    time_to_expiration: float,
) -> float:
    """
    The price of skewness S of one expiry as in the Cboe SKEW Index white paper (2010): the
    risk-neutral skewness of the log return R = ln(S_T / F_0) from the prices P_1, P_2 and
    P_3 of R, R² and R³, each a strip of out-of-the-money options. With y_i = ln(K_i / F_0),
    Q(K_i) and ΔK_i from get_out_of_the_money_prices and K_0 the first strike at or below
    the forward price F_0:

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
    Interpolate the skewness of two expiries to a constant maturity and turn it into an
    index as the Cboe does for the SKEW Index:

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
