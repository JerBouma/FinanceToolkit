"""Trades Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd

from financetoolkit.portfolio.helpers import get_transaction_order, to_timestamps
from financetoolkit.portfolio.overview_model import match_lots
from financetoolkit.utilities.error_model import check_option

TRADE_COLUMNS = [
    "Ticker",
    "Direction",
    "Status",
    "Entry Date",
    "Entry Price",
    "Exit Date",
    "Exit Price",
    "Size",
    "Costs",
    "Gross PnL",
    "PnL",
    "Return",
    "Duration (Days)",
    "Duration (Periods)",
]

TRADE_STATISTICS_GROUPS = ("direction", "ticker")


def get_trades(
    transactions: pd.DataFrame,
    last_prices: pd.Series,
    trading_dates: pd.Index,
    method: str = "FIFO",
    purchases_first: bool = True,
) -> pd.DataFrame:
    """
    The trade list (round trips) of a set of transactions.

    Every lot a transaction opened, paired with the transaction that closed it, and the lots
    still open marked to the last price. Each ticker is matched on its own with
    `overview_model.match_lots`, the lot matching behind the realized profit and loss of the
    Portfolio, with short selling: a sale of more units than are held opens a short trade
    that later purchases close.

    A transaction that closes part of a lot splits it, so one purchase sold in two parts is
    two trades, and a sale that closes two purchases is two trades as well. The costs of a
    transaction are spread over its units, so a trade carries the share of the costs of its
    entry and of its exit that belongs to its units.

    The formula is as follows:

    - Gross PnL = Size × (Exit Price − Entry Price) × Direction, with Direction 1 for a long
      and −1 for a short trade
    - Costs = |Entry Costs| × Size / |Entry Volume| + |Exit Costs| × Size / |Exit Volume|
    - PnL = Gross PnL − Costs
    - Return = PnL / (Size × Entry Price)
    - Duration (Days) = Exit Date − Entry Date in calendar days, Duration (Periods) the number
      of trading dates from the entry to the exit

    Shares similar logic:
    - vectorbt, Trades: https://vectorbt.dev/api/portfolio/trades/#vectorbt.portfolio.trades.Trades
    - backtesting.py, Trade: https://kernc.github.io/backtesting.py/doc/backtesting/backtesting.html#backtesting.backtesting.Trade
    - backtrader, TradeAnalyzer: https://www.backtrader.com/docu/analyzers-reference/#tradeanalyzer

    Args:
        transactions (pd.DataFrame): One row per transaction with the columns "Date", "Ticker",
            "Volume" (negative for a sale), "Price" and "Costs" (the sign is ignored).
        last_prices (pd.Series): The last price of every ticker, at which open lots are marked.
        trading_dates (pd.Index): The trading dates, of the same type as the transaction dates,
            on which the durations in periods are counted.
        method (str, optional): Which lots a sale closes first. Options are: "FIFO" (the oldest lots
            close first), "LIFO" (the newest lots close first). Defaults to "FIFO".
        purchases_first (bool, optional): Whether the purchases of a day come before its
            sales. False keeps the order of the rows within a day. Defaults to True.

    Returns:
        pd.DataFrame: One row per trade, numbered from 1 in the order of their entry, with the
        Ticker, Direction ("Long" or "Short"), Status ("Closed" or "Open"), Entry Date, Entry
        Price, Exit Date, Exit Price, Size, Costs, Gross PnL, PnL, Return, Duration (Days) and
        Duration (Periods). An open trade exits at the last trading date and the last price,
        without exit costs.

    Notes:
    - vectorbt's Trades define the PnL and the return the same way.
    """
    method = method.upper()
    check_option("method", method, ("FIFO", "LIFO"))

    ordered = transactions.iloc[
        get_transaction_order(
            pd.Index(transactions["Date"]), transactions["Volume"], purchases_first
        )
    ]
    last_date = trading_dates[-1]
    rows = []

    for ticker, group in ordered.groupby("Ticker", sort=False, observed=True):
        dates = group["Date"].to_numpy()
        volumes = group["Volume"].to_numpy(dtype=float)
        prices = group["Price"].to_numpy(dtype=float)
        costs = np.abs(np.nan_to_num(group["Costs"].to_numpy(dtype=float)))
        # The costs of a transaction per unit traded, so every part of it carries its share.
        unit_costs = np.divide(
            costs, np.abs(volumes), out=np.zeros(len(costs)), where=volumes != 0
        )

        matches, open_lots, _ = match_lots(volumes, method=method, short_selling=True)

        for opening, closing, units in matches:
            rows.append(
                (
                    ticker,
                    np.sign(volumes[opening]),
                    "Closed",
                    dates[opening],
                    prices[opening],
                    dates[closing],
                    prices[closing],
                    units,
                    units * (unit_costs[opening] + unit_costs[closing]),
                )
            )
        for opening, units in open_lots:
            rows.append(
                (
                    ticker,
                    np.sign(units),
                    "Open",
                    dates[opening],
                    prices[opening],
                    last_date,
                    last_prices.get(ticker, np.nan),
                    abs(units),
                    abs(units) * unit_costs[opening],
                )
            )

    trades = pd.DataFrame(
        rows,
        columns=[
            "Ticker",
            "Direction",
            "Status",
            "Entry Date",
            "Entry Price",
            "Exit Date",
            "Exit Price",
            "Size",
            "Costs",
        ],
    )
    direction = trades["Direction"].astype(float)
    trades["Gross PnL"] = (
        trades["Size"] * (trades["Exit Price"] - trades["Entry Price"]) * direction
    )
    trades["PnL"] = trades["Gross PnL"] - trades["Costs"]
    entry_value = trades["Size"] * trades["Entry Price"]
    trades["Return"] = trades["PnL"] / entry_value.where(entry_value != 0)
    trades["Duration (Days)"] = (
        to_timestamps(trades["Exit Date"]) - to_timestamps(trades["Entry Date"])
    ).days.astype(float)
    # A transaction on a date without trading, such as a weekend, counts from the next trading date.
    entry_periods = trading_dates.get_indexer(trades["Entry Date"], method="backfill")
    exit_periods = trading_dates.get_indexer(trades["Exit Date"], method="backfill")
    entry_periods[entry_periods == -1] = len(trading_dates) - 1
    exit_periods[exit_periods == -1] = len(trading_dates) - 1
    trades["Duration (Periods)"] = exit_periods - entry_periods
    trades["Direction"] = np.where(direction > 0, "Long", "Short")

    trades = trades.sort_values(
        ["Entry Date", "Exit Date", "Ticker"], kind="stable"
    ).reset_index(drop=True)
    trades.index = pd.RangeIndex(1, len(trades) + 1, name="Trade")

    return trades[TRADE_COLUMNS]


def get_trades_from_positions(
    prices: pd.Series | pd.DataFrame,
    positions: pd.Series | pd.DataFrame,
    fees: float = 0.0,
    method: str = "FIFO",
) -> pd.DataFrame:
    """
    The trade list (round trips) of a position series.

    The positions can be the signals of a trading rule (1 long, −1 short, 0 flat), the pairs
    trading positions of `toolkit.econometrics.get_pairs_trading` or a number of units per
    date: every change in the position is a transaction at that date's price, and the
    transactions are matched into trades as `get_trades` does. A change from long to short
    closes the long trade and opens a short one on the same date.

    The position on a date is traded at that date's price, the close for daily prices, so a
    signal that is only known at the close has to be shifted a period to be traded at the
    next close without look-ahead.

    Shares similar logic:
    - vectorbt, Portfolio.from_orders (target amounts): https://vectorbt.dev/api/portfolio/base/#vectorbt.portfolio.base.Portfolio.from_orders
    - backtesting.py, Trade: https://kernc.github.io/backtesting.py/doc/backtesting/backtesting.html#backtesting.backtesting.Trade

    Args:
        prices (pd.Series | pd.DataFrame): The price per date, one column per ticker.
        positions (pd.Series | pd.DataFrame): The position per date in units (or 1, 0 and −1
            for one unit), one column per ticker with the same names as `prices`. A missing
            position is flat.
        fees (float, optional): The costs as a fraction of the value traded, 0.001 for 0.1%.
            Defaults to 0.0.
        method (str, optional): Which lots a smaller position closes first. Options are: "FIFO" (the
            oldest lots close first), "LIFO" (the newest lots close first). Defaults to "FIFO".

    Returns:
        pd.DataFrame: The trade list as `get_trades` returns it, with an open trade marked to
        the last price.

    Notes:
    - vectorbt's Portfolio.from_orders with target amounts and backtesting.py both trade at
      the close as well.
    """
    prices = prices.to_frame() if isinstance(prices, pd.Series) else prices
    positions = (
        positions.to_frame(name=prices.columns[0])
        if isinstance(positions, pd.Series)
        else positions
    )
    missing = positions.columns.difference(prices.columns)
    if not missing.empty:
        raise ValueError(
            f"No prices for the positions of {', '.join(map(str, missing))}."
        )

    held = positions.reindex(prices.index).fillna(0.0).astype(float)
    # The first position is bought on the first date, every later change on its date.
    changes = held.diff()
    changes.iloc[0] = held.iloc[0]

    transactions = []
    for ticker in held.columns:
        volume = changes[ticker]
        traded = volume[volume != 0]
        price = prices.loc[traded.index, ticker]
        if price.isna().any():
            raise ValueError(
                f"The position of {ticker} changes on dates without a price: "
                f"{', '.join(map(str, price.index[price.isna()][:5]))}."
            )
        transactions.append(
            pd.DataFrame(
                {
                    "Date": traded.index,
                    "Ticker": ticker,
                    "Volume": traded.to_numpy(),
                    "Price": price.to_numpy(),
                    "Costs": fees * np.abs(traded.to_numpy()) * price.to_numpy(),
                }
            )
        )

    return get_trades(
        pd.concat(transactions, ignore_index=True),
        last_prices=prices.ffill().iloc[-1],
        trading_dates=prices.index,
        method=method,
        purchases_first=False,
    )


def _trade_statistics(trades: pd.DataFrame, trading_dates: pd.Index) -> pd.Series:
    """The statistics of one set of trades (see `get_trade_statistics`)."""
    closed = trades[trades["Status"] == "Closed"].sort_values(
        ["Exit Date", "Entry Date"], kind="stable"
    )
    pnl = closed["PnL"].to_numpy(dtype=float)
    returns = closed["Return"].to_numpy(dtype=float)
    days = closed["Duration (Days)"].to_numpy(dtype=float)
    wins, losses = pnl > 0, pnl < 0
    count = len(pnl)

    def mean(values: np.ndarray) -> float:
        return float(np.mean(values)) if values.size else np.nan

    win_rate = wins.sum() / count if count else np.nan
    average_win, average_loss = mean(pnl[wins]), mean(pnl[losses])
    gross_profit, gross_loss = pnl[wins].sum(), -pnl[losses].sum()
    standard_deviation = np.std(pnl, ddof=1) if count > 1 else np.nan

    # The streaks run over the closed trades in the order they closed; a trade without a profit or loss ends both.
    longest_wins = longest_losses = current_streak = 0
    for value in pnl:
        if value > 0:
            current_streak = max(current_streak, 0) + 1
        elif value < 0:
            current_streak = min(current_streak, 0) - 1
        else:
            current_streak = 0
        longest_wins = max(longest_wins, current_streak)
        longest_losses = max(longest_losses, -current_streak)

    # A trade holds its position from the close it enters at up to the close it exits at, so it covers the periods from its entry up to its exit; an open trade covers the last period too.  # noqa: E501
    # Counted with a difference array: +1 on the entry date and -1 on the first date the trade no longer covers.  # noqa: E501
    entries = trading_dates.get_indexer(trades["Entry Date"], method="backfill")
    exits = trading_dates.get_indexer(trades["Exit Date"], method="backfill")
    entries[entries == -1] = len(trading_dates) - 1
    exits[exits == -1] = len(trading_dates) - 1
    ends = exits + (trades["Status"].to_numpy() == "Open")
    changes = np.zeros(len(trading_dates) + 1, dtype=int)
    np.add.at(changes, entries, 1)
    np.add.at(changes, ends, -1)
    held = np.cumsum(changes[:-1])
    covered = (held > 0).sum()

    return pd.Series(
        {
            "Trades": len(trades),
            "Closed Trades": count,
            "Open Trades": (trades["Status"] == "Open").sum(),
            "Long Trades": (trades["Direction"] == "Long").sum(),
            "Short Trades": (trades["Direction"] == "Short").sum(),
            "Winning Trades": wins.sum(),
            "Losing Trades": losses.sum(),
            "Win Rate": win_rate,
            "Gross PnL": closed["Gross PnL"].sum(),
            "Costs": closed["Costs"].sum(),
            "Net PnL": pnl.sum(),
            "Open PnL": trades.loc[trades["Status"] == "Open", "PnL"].sum(),
            "Best Trade": np.nanmax(returns) if count else np.nan,
            "Worst Trade": np.nanmin(returns) if count else np.nan,
            "Average Trade": mean(returns),
            "Average Winning Trade": mean(returns[wins]),
            "Average Losing Trade": mean(returns[losses]),
            "Average PnL": mean(pnl),
            "Average Winning PnL": average_win,
            "Average Losing PnL": average_loss,
            "Payoff Ratio": average_win / -average_loss if losses.any() else np.nan,
            "Profit Factor": gross_profit / gross_loss if gross_loss > 0 else np.nan,
            "Expectancy": (
                win_rate * np.nan_to_num(average_win)
                - losses.sum() / count * np.nan_to_num(-average_loss)
                if count
                else np.nan
            ),
            "Expectancy Ratio": mean(pnl) / -average_loss if losses.any() else np.nan,
            "SQN": (
                np.sqrt(count) * mean(pnl) / standard_deviation
                if standard_deviation > 0
                else np.nan
            ),
            "Kelly Criterion": (
                win_rate - (1 - win_rate) / (average_win / -average_loss)
                if wins.any() and losses.any()
                else np.nan
            ),
            "Average Duration": mean(days),
            "Average Winning Duration": mean(days[wins]),
            "Average Losing Duration": mean(days[losses]),
            "Maximum Duration": np.max(days) if count else np.nan,
            "Minimum Duration": np.min(days) if count else np.nan,
            "Longest Winning Streak": longest_wins,
            "Longest Losing Streak": longest_losses,
            "Current Streak": current_streak,
            "Coverage": np.mean(held > 0),
            "Overlap Coverage": np.mean(held[held > 0] > 1) if covered else np.nan,
        },
        dtype=float,
    )


def get_trade_statistics(
    trades: pd.DataFrame, trading_dates: pd.Index, by: str | None = None
) -> pd.DataFrame:
    """
    The statistics of a trade list (`get_trades` or `get_trades_from_positions`).

    How many trades there were, how many made money, how much they made and lost, how long
    they lasted and how they followed each other. Every statistic but the counts, the Open
    PnL and the coverage is calculated on the closed trades, in the order they closed.

    The formula is as follows:

    - Win Rate = W = number of trades with PnL > 0 / number of closed trades (a trade without
      a profit or loss counts as neither a win nor a loss)
    - Payoff Ratio = R = mean(PnL⁺) / |mean(PnL⁻)|
    - Profit Factor = Σ PnL⁺ / |Σ PnL⁻|
    - Expectancy = W × mean(PnL⁺) − L × |mean(PnL⁻)|, with L the share of losing trades, which
      equals the mean PnL per trade
    - Expectancy Ratio = Expectancy / |mean(PnL⁻)|, the expectancy in multiples of the
      average loss
    - SQN = √n × mean(PnL) / std(PnL), with n closed trades and ddof=1
    - Kelly Criterion = W − (1 − W) / R
    - Coverage = share of trading dates with at least one trade held, Overlap Coverage =
      share of those dates with more than one, as vectorbt defines them for its ranges
      (Ranges.coverage)

    Tharp defines the expectancy and the System Quality Number on R-multiples, the PnL of
    every trade divided by the risk taken on it (the distance to its initial stop). A trade
    list has no stops, so the PnL in currency takes its place: the SQN is the same for any
    constant risk per trade, since a ratio of a mean to a standard deviation does not change
    when every PnL is divided by the same amount. With the average loss as the risk, the
    Expectancy Ratio is the expectancy in R, Tharp's measure for a system whose losses are
    its stops. Tharp's later scoring caps n at 100 (√min(n, 100)) so a system is not ranked
    higher for trading more; here n is the number of trades, as in vectorbt and
    backtesting.py. backtrader divides the standard deviation by n rather than n − 1, which
    gives a slightly higher SQN for few trades.

    The definitions of the libraries differ in places, and these follow the trades in
    currency. The expectancy here is vectorbt's, in currency per trade; backtesting.py's
    "Expectancy [%]" is the mean trade return, here the "Average Trade". backtesting.py takes
    the profit factor from the trade returns rather than the PnL and averages the trade
    returns geometrically. The Kelly fraction is Kelly's f* = W − (1 − W)/R for a bet that wins
    R times what it loses, as backtesting.py calculates it from the PnL of the trades.

    For more information, see the following sources:

    - Tharp, V.K. (2008). "Van Tharp's Definitive Guide to Position Sizing." International
      Institute of Trading Mastery.
    - Tharp, V.K. (2006). "Trade Your Way to Financial Freedom." 2nd edition, McGraw-Hill.
    - Kelly, J.L. (1956). "A New Interpretation of Information Rate." Bell System Technical
      Journal, 35(4), 917-926.

    Shares similar logic:
    - vectorbt, Trades.stats: https://vectorbt.dev/api/portfolio/trades/#vectorbt.portfolio.trades.Trades
    - vectorbt, Ranges.coverage (coverage): https://vectorbt.dev/api/generic/ranges/#vectorbt.generic.ranges.Ranges.coverage
    - backtesting.py, Backtest.run (statistics): https://kernc.github.io/backtesting.py/doc/backtesting/backtesting.html#backtesting.backtesting.Backtest.run
    - backtrader, TradeAnalyzer: https://www.backtrader.com/docu/analyzers-reference/#tradeanalyzer
    - backtrader, SQN: https://www.backtrader.com/docu/analyzers-reference/#sqn

    Args:
        trades (pd.DataFrame): The trade list, as `get_trades` returns it.
        trading_dates (pd.Index): The trading dates the trades took place in, of the same type
            as their dates, over which the coverage is measured.
        by (str | None, optional): How the trades are split. Options are: "direction" (the long and
            short trades next to all of them), "ticker" (every ticker next to all of them). Defaults
            to None, all trades together.

    Returns:
        pd.DataFrame: The statistics as rows, with a column "All" and one per direction or
        ticker. Returns are decimals, PnL is in the currency of the prices and durations are in
        calendar days. The Current Streak is positive for winning trades and negative for
        losing ones.
    """
    if by is not None and by not in TRADE_STATISTICS_GROUPS:
        raise ValueError(f"by must be None, 'direction' or 'ticker', not '{by}'.")

    statistics = {"All": _trade_statistics(trades, trading_dates)}
    if by == "direction":
        for direction in ("Long", "Short"):
            statistics[direction] = _trade_statistics(
                trades[trades["Direction"] == direction], trading_dates
            )
    elif by == "ticker":
        for ticker, group in trades.groupby("Ticker", sort=True):
            statistics[ticker] = _trade_statistics(group, trading_dates)

    return pd.DataFrame(statistics)


def get_flattening_and_flips(positions: pd.Series) -> pd.Series:
    """
    The dates on which a position is flattened (goes to zero) or flips (changes sign), the
    dates on which a bet ends, as López de Prado defines a bet.

    A bet is a position held from the date it opens until it is flattened or flipped. The
    number of bets measures how often a strategy takes an independent position.

    For more information, see the following source:

    - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, section
      14.3 (Snippet 14.1).

    Args:
        positions (pd.Series): The position per date, in units or as a signal.

    Returns:
        pd.Series: "Flattening" or "Flip" on every date on which a bet ends, and "Open" on the
        last date when the position is still held, which ends the last bet.

    Notes:
        - López de Prado's snippet adds the last date in every case; here it ends a bet only
          when a position is held on it, so a strategy that is flat at the end has no bet
          that ends there. The snippet also counts the first date as a flattening when the
          position starts flat (its previous position is missing, not zero); here it does not.
    """
    held = positions.fillna(0.0).astype(float)
    previous = held.shift(1).fillna(0.0)
    events = pd.Series(pd.NA, index=held.index, dtype=object)
    events[(held == 0) & (previous != 0)] = "Flattening"
    events[held * previous < 0] = "Flip"
    if len(held) and held.iloc[-1] != 0:
        events.iloc[-1] = "Open"
    return events.dropna().rename("Event")


def get_average_holding_period(positions: pd.Series) -> float:
    """
    The average holding period of a position series in calendar days, as López de Prado
    defines it: the time every unit was held, weighted by the units.

    Each increase of the position moves its average entry time towards the date of the
    increase, weighted by the units added; each decrease closes units that were held since
    the average entry time, and a flip closes the whole position and opens a new one at the
    date of the flip. Units still held at the end are not counted, as their holding period
    has not ended.

    The formula is as follows:

    - Average Entry Time_t = (Average Entry Time_t-1 × |p_t-1| + t × |Δp_t|) / |p_t| when the
      position grows
    - Holding Period = Σ (t − Average Entry Time) × |units closed at t| / Σ |units closed|

    Over a position that ends flat, this equals the average duration of its FIFO trades
    weighted by their size, as both add up the same entry and exit times of every unit.

    For more information, see the following source:

    - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, section
      14.3 (Snippet 14.2).

    Args:
        positions (pd.Series): The position per date, in units or as a signal, on a
            DatetimeIndex or PeriodIndex.

    Returns:
        float: The average holding period in calendar days, NaN when no units were closed.
    """
    held = positions.fillna(0.0).to_numpy(dtype=float)
    times = to_timestamps(positions.index)
    elapsed = ((times - times[0]) / pd.Timedelta(days=1)).to_numpy(dtype=float)

    entry_time = 0.0
    weighted_time = closed_units = 0.0
    for step in range(1, len(held)):
        previous, current = held[step - 1], held[step]
        change = current - previous
        if change * previous >= 0:  # the position grows, or opens from flat, or stays
            if current != 0:
                entry_time = (
                    entry_time * abs(previous) + elapsed[step] * abs(change)
                ) / abs(current)
        elif (
            current * previous < 0
        ):  # a flip closes all units and opens the new position now
            weighted_time += (elapsed[step] - entry_time) * abs(previous)
            closed_units += abs(previous)
            entry_time = elapsed[step]
        else:  # the position shrinks towards zero
            weighted_time += (elapsed[step] - entry_time) * abs(change)
            closed_units += abs(change)

    return weighted_time / closed_units if closed_units > 0 else np.nan
