"""Futures Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd

# The delivery months each commodity lists for the months ahead: every calendar month for
# energy and metals (CME lists those monthly for over two years ahead) and the crop cycle
# for grains: corn and wheat March, May, July, September and December, soybeans January,
# March, May, July, August, September and November.
CONTRACT_MONTHS: dict[str, tuple[int, ...]] = {
    "Crude Oil": tuple(range(1, 13)),
    "Natural Gas": tuple(range(1, 13)),
    "Gold": tuple(range(1, 13)),
    "Silver": tuple(range(1, 13)),
    "Copper": tuple(range(1, 13)),
    "Corn": (3, 5, 7, 9, 12),
    "Wheat": (3, 5, 7, 9, 12),
    "Soybeans": (1, 3, 5, 7, 8, 9, 11),
}

# The months of the active cycle that the generic (nearby) series of gold and silver roll
# through, which are the months Yahoo Finance carries. COMEX lists every month, but the
# other months trade little and only come with prices shortly before delivery.
ACTIVE_MONTHS: dict[str, tuple[int, ...]] = {
    "Gold": (2, 4, 6, 8, 10, 12),
    "Silver": (3, 5, 7, 9, 12),
}

# The last trading day rule of each commodity's contracts (CME Group contract specifications).
LAST_TRADING_DAY_RULES: dict[str, str] = {
    "Crude Oil": "crude_oil",
    "Natural Gas": "natural_gas",
    "Gold": "metals",
    "Silver": "metals",
    "Copper": "metals",
    "Corn": "grains",
    "Wheat": "grains",
    "Soybeans": "grains",
}


def get_last_trading_day(
    rule: str, delivery_month: pd.Period, business_days: pd.DatetimeIndex
) -> pd.Timestamp:
    """
    Calculates the last trading day of a futures contract from the rule in its CME Group
    contract specification:

    - crude_oil (NYMEX WTI Light Sweet Crude Oil, CL): three business days before the 25th
      calendar day of the month before the delivery month, or four business days before
      it when the 25th is not a business day.
    - natural_gas (NYMEX Henry Hub Natural Gas, NG): the third last business day of the
      month before the delivery month.
    - metals (COMEX Gold GC, Silver SI and Copper HG): the third last business day of the
      delivery month.
    - grains (CBOT Corn ZC, Wheat ZW and Soybeans ZS): the business day before the 15th
      calendar day of the delivery month.

    Args:
        rule (str): The rule, "crude_oil", "natural_gas", "metals" or "grains".
        delivery_month (pd.Period): The contract's delivery month.
        business_days (pd.DatetimeIndex): The exchange's business days, covering at least
            the month before the delivery month and the delivery month itself.

    Raises:
        ValueError: If the rule is not one of the above.

    Returns:
        pd.Timestamp: The last day the contract trades.
    """
    delivery_month = pd.Period(delivery_month, freq="M")
    prior_month = delivery_month - 1

    if rule == "crude_oil":
        day_25 = pd.Timestamp(prior_month.year, prior_month.month, 25)
        before = business_days[business_days < day_25]

        return before[-3] if day_25 in business_days else before[-4]

    if rule == "natural_gas":
        return business_days[business_days.to_period("M") == prior_month][-3]

    if rule == "metals":
        return business_days[business_days.to_period("M") == delivery_month][-3]

    if rule == "grains":
        day_15 = pd.Timestamp(delivery_month.year, delivery_month.month, 15)

        return business_days[business_days < day_15][-1]

    raise ValueError(
        f"The last trading day rule '{rule}' is not supported. Choose one of: "
        "crude_oil, natural_gas, metals or grains."
    )


def get_generic_curve(
    forward_curve: pd.DataFrame,
    last_trading_days: pd.Series,
    contracts: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Arranges dated futures contracts by their position on the curve of each day: the
    front (the nearest contract that has not expired), the next one (+1) and so on, as
    the curve was listed on that day. This is the generic (nearby) series (CL1, CL2, ...
    in Bloomberg notation) with the contract that fills each position on each day, rolling
    to the next contract the day after the front's last trading day. A position is a rank
    on the curve, not a fixed time to maturity: its maturity shortens as the front nears
    expiry and lengthens again at the roll, as nothing is interpolated.

    A contract is on the curve of a day from its first price up to and including its last
    trading day. A contract without prices (one that expired and is no longer published, or
    one the source misses) is on the curve of every day up to its last trading day, so that the positions after
    it stay right; its own position has no price (NaN).

    Args:
        forward_curve (pd.DataFrame): The daily prices of each contract, indexed by date
            with one column per contract labelled by its delivery month (e.g. "2026-12").
            Expired contracts without prices are all-NaN columns.
        last_trading_days (pd.Series): The last trading day of each contract, indexed by
            the same labels.
        contracts (int, optional): The number of positions to return, the front included.
            Defaults to 12.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The price in each position ("Front", "+1",
            "+2", ...) per date, and the delivery month of the contract in that position.
            Positions no known contract fills, or whose contract has no price that day,
            are NaN.
    """
    forward_curve = forward_curve.sort_index(axis=1)
    labels = forward_curve.columns.to_numpy(dtype=object)
    dates = forward_curve.index.to_timestamp().to_numpy()
    prices = forward_curve.to_numpy(dtype=float)

    first_prices = np.array(
        [
            (
                np.datetime64("NaT", "ns")
                if (first := forward_curve[label].first_valid_index()) is None
                else first.to_timestamp().to_datetime64()
            )
            for label in labels
        ],
        dtype="datetime64[ns]",
    )
    expirations = (
        pd.to_datetime(last_trading_days.reindex(labels))
        .to_numpy()
        .astype("datetime64[ns]")
    )

    listed = np.isnat(first_prices)[None, :] | (dates[:, None] >= first_prices[None, :])
    on_curve = listed & (dates[:, None] <= expirations[None, :])
    position = np.cumsum(on_curve, axis=1) - 1

    columns = ["Front"] + [f"+{number}" for number in range(1, contracts)]
    position_prices = np.full((len(dates), contracts), np.nan)
    position_labels = np.full((len(dates), contracts), np.nan, dtype=object)
    rows = np.arange(len(dates))

    for number in range(contracts):
        in_position = on_curve & (position == number)
        filled = in_position.any(axis=1)
        contract = in_position.argmax(axis=1)
        position_prices[filled, number] = prices[rows[filled], contract[filled]]
        position_labels[filled, number] = labels[contract[filled]]

    return (
        pd.DataFrame(position_prices, index=forward_curve.index, columns=columns),
        pd.DataFrame(position_labels, index=forward_curve.index, columns=columns),
    )


def get_term_structure(
    position_prices: pd.DataFrame,
    position_labels: pd.DataFrame,
    months: int = 12,
) -> pd.DataFrame:
    """
    Summarises the term structure of a futures curve per day with the 12-month calendar
    spread: the front contract against the last contract of the 12-month strip that starts
    with it (the furthest contract delivering less than `months` months after the front,
    e.g. November 2027 for a December 2026 front of a monthly listed commodity, September
    2027 for corn's December 2026 front).

    The formulas are as follows:

        Spread = Front - Far
        Slope = Far / Front - 1
        Roll Yield = ln(Front / Far) x 12 / Months

    The spread follows the exchange quote of a calendar spread, nearby minus deferred, so
    that it is positive in backwardation. The slope is the percentage the far contract lies
    above the front, negative in backwardation. The roll yield is the annualised log basis
    of Fama and French (1987) and Szymanowska, De Roon, Nijman and Van den Goorbergh (2014):
    the continuously compounded yearly return a long position earns from the curve alone as
    the far contract rolls up (backwardation, positive) or down (contango, negative) to the
    front's price, with Months the months between both delivery months. The curve is in
    Backwardation when the front is above the far contract, in Contango when it is below
    and Flat when both trade at the same price.

    Args:
        position_prices (pd.DataFrame): The price in each position of the curve per date,
            the front first (see get_generic_curve).
        position_labels (pd.DataFrame): The delivery month ("YYYY-MM") of the contract in
            each position per date.
        months (int, optional): The length of the strip in months. Defaults to 12.

    Returns:
        pd.DataFrame: Per date the Front Contract, Front, Far Contract, Far, Months,
            Spread, Slope, Roll Yield (decimals) and Structure ("Backwardation", "Contango"
            or "Flat").
    """
    delivery = position_labels.map(
        lambda label: (
            pd.Period(label, freq="M").ordinal if isinstance(label, str) else np.nan
        )
    ).astype(float)
    months_from_front = delivery.sub(delivery.iloc[:, 0], axis=0)

    # The far contract is the furthest one that delivers within the strip of the front.
    in_strip = months_from_front.lt(months) & months_from_front.gt(0)
    far_position = in_strip.to_numpy()[:, ::-1].argmax(axis=1)
    far_position = in_strip.shape[1] - 1 - far_position
    has_far = in_strip.any(axis=1).to_numpy()
    rows = np.arange(len(position_prices))

    front = position_prices.iloc[:, 0]
    far = pd.Series(
        np.where(has_far, position_prices.to_numpy()[rows, far_position], np.nan),
        index=position_prices.index,
    )
    months_between = pd.Series(
        np.where(has_far, months_from_front.to_numpy()[rows, far_position], np.nan),
        index=position_prices.index,
    )
    far_contract = pd.Series(
        np.where(has_far, position_labels.to_numpy()[rows, far_position], np.nan),
        index=position_prices.index,
        dtype=object,
    )

    structure = pd.Series(np.nan, index=position_prices.index, dtype=object)
    structure[front > far] = "Backwardation"
    structure[front < far] = "Contango"
    structure[front == far] = "Flat"

    return pd.DataFrame(
        {
            "Front Contract": position_labels.iloc[:, 0],
            "Front": front,
            "Far Contract": far_contract,
            "Far": far,
            "Months": months_between,
            "Spread": front - far,
            "Slope": far / front - 1,
            "Roll Yield": np.log(front / far) * 12 / months_between,
            "Structure": structure,
        }
    )
