"""Trades Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.portfolio import trades_model

DATES = pd.date_range("2024-01-01", periods=10, freq="D")


def _transactions(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["Date", "Ticker", "Volume", "Price", "Costs"])


def test_a_long_round_trip():
    trades = trades_model.get_trades(
        _transactions(
            [(DATES[0], "A", 10, 100.0, -2.0), (DATES[3], "A", -10, 110.0, -3.0)]
        ),
        last_prices=pd.Series({"A": 120.0}),
        trading_dates=DATES,
    )

    trade = trades.loc[1]
    assert len(trades) == 1
    assert (trade["Direction"], trade["Status"]) == ("Long", "Closed")
    assert trade["Size"] == 10
    # 10 x (110 - 100) = 100, less the costs of 2 + 3.
    assert trade["Gross PnL"] == pytest.approx(100.0)
    assert trade["Costs"] == pytest.approx(5.0)
    assert trade["PnL"] == pytest.approx(95.0)
    assert trade["Return"] == pytest.approx(95.0 / 1000.0)
    assert trade["Duration (Days)"] == 3
    assert trade["Duration (Periods)"] == 3


def test_a_short_round_trip():
    trades = trades_model.get_trades(
        _transactions([(DATES[1], "B", -5, 50.0, 0.0), (DATES[3], "B", 5, 45.0, -1.0)]),
        last_prices=pd.Series({"B": 40.0}),
        trading_dates=DATES,
    )

    trade = trades.loc[1]
    assert (trade["Direction"], trade["Status"]) == ("Short", "Closed")
    # A short gains when the price falls: 5 x (50 - 45) = 25, less a cost of 1.
    assert trade["Gross PnL"] == pytest.approx(25.0)
    assert trade["PnL"] == pytest.approx(24.0)
    assert trade["Return"] == pytest.approx(24.0 / 250.0)


def test_a_partial_close_splits_the_lot_and_its_costs():
    trades = trades_model.get_trades(
        _transactions(
            [
                (DATES[0], "A", 10, 100.0, -2.0),
                (DATES[2], "A", -4, 110.0, -1.0),
                (DATES[4], "A", -6, 95.0, -1.0),
            ]
        ),
        last_prices=pd.Series({"A": 90.0}),
        trading_dates=DATES,
    )

    assert trades["Size"].tolist() == [4, 6]
    # The purchase cost of 2 is spread over its 10 units: 0.8 for the first part, 1.2 for the second.
    assert trades["Costs"].tolist() == pytest.approx([0.8 + 1.0, 1.2 + 1.0])
    assert trades["PnL"].tolist() == pytest.approx([40.0 - 1.8, -30.0 - 2.2])
    assert trades["Return"].tolist() == pytest.approx([38.2 / 400.0, -32.2 / 600.0])
    assert trades["Duration (Days)"].tolist() == [2, 4]


def test_an_open_lot_is_marked_to_the_last_price():
    trades = trades_model.get_trades(
        _transactions(
            [(DATES[0], "A", 10, 100.0, -1.0), (DATES[2], "A", -4, 120.0, 0.0)]
        ),
        last_prices=pd.Series({"A": 130.0}),
        trading_dates=DATES,
    )

    open_trade = trades.loc[2]
    assert trades["Status"].tolist() == ["Closed", "Open"]
    assert open_trade["Exit Date"] == DATES[-1]
    assert open_trade["Exit Price"] == 130.0
    # Six units held since 100 are worth 30 more each; they carry 0.6 of the purchase cost and no exit cost.
    assert open_trade["Costs"] == pytest.approx(0.6)
    assert open_trade["PnL"] == pytest.approx(180.0 - 0.6)
    assert open_trade["Duration (Periods)"] == 9


def test_lifo_closes_the_newest_lot_first():
    transactions = _transactions(
        [
            (DATES[0], "A", 5, 10.0, 0.0),
            (DATES[1], "A", 5, 20.0, 0.0),
            (DATES[2], "A", -5, 30.0, 0.0),
        ]
    )
    last = pd.Series({"A": 30.0})

    fifo = trades_model.get_trades(transactions, last, DATES, method="FIFO")
    lifo = trades_model.get_trades(transactions, last, DATES, method="LIFO")

    assert fifo.loc[fifo["Status"] == "Closed", "Entry Price"].tolist() == [10.0]
    assert lifo.loc[lifo["Status"] == "Closed", "Entry Price"].tolist() == [20.0]
    with pytest.raises(ValueError, match="FIFO"):
        trades_model.get_trades(transactions, last, DATES, method="AVERAGE")


def test_a_flip_of_a_position_series_closes_one_trade_and_opens_the_next():
    prices = pd.Series([10.0, 11.0, 12.0, 9.0, 9.0], index=DATES[:5], name="X")
    positions = pd.Series([0, 1, -1, 0, 0], index=DATES[:5], name="X")

    trades = trades_model.get_trades_from_positions(prices, positions, fees=0.01)

    assert trades["Direction"].tolist() == ["Long", "Short"]
    assert trades["Entry Price"].tolist() == [11.0, 12.0]
    assert trades["Exit Price"].tolist() == [12.0, 9.0]
    # Fees of 1% of the value traded at entry and exit: 0.11 + 0.12 and 0.12 + 0.09.
    assert trades["Costs"].tolist() == pytest.approx([0.23, 0.21])
    assert trades["PnL"].tolist() == pytest.approx([1.0 - 0.23, 3.0 - 0.21])


def test_the_trade_statistics_by_hand():
    trades = trades_model.get_trades(
        _transactions(
            [
                (DATES[0], "A", 10, 100.0, -2.0),
                (DATES[2], "A", -4, 110.0, -1.0),
                (DATES[4], "A", -6, 95.0, -1.0),
                (DATES[5], "B", -5, 50.0, 0.0),
                (DATES[7], "B", 5, 45.0, -1.0),
            ]
        ),
        last_prices=pd.Series({"A": 90.0, "B": 40.0}),
        trading_dates=DATES,
    )
    statistics = trades_model.get_trade_statistics(trades, DATES, by="direction")
    every = statistics["All"]
    pnl = np.array([38.2, -32.2, 24.0])

    assert statistics.columns.tolist() == ["All", "Long", "Short"]
    assert every["Win Rate"] == pytest.approx(2 / 3)
    assert every["Net PnL"] == pytest.approx(30.0)
    assert every["Gross PnL"] == pytest.approx(35.0)
    assert every["Costs"] == pytest.approx(5.0)
    assert every["Profit Factor"] == pytest.approx((38.2 + 24.0) / 32.2)
    assert every["Expectancy"] == pytest.approx(pnl.mean())
    assert every["Expectancy Ratio"] == pytest.approx(10.0 / 32.2)
    assert every["SQN"] == pytest.approx(np.sqrt(3) * pnl.mean() / pnl.std(ddof=1))
    payoff = 31.1 / 32.2
    assert every["Payoff Ratio"] == pytest.approx(payoff)
    assert every["Kelly Criterion"] == pytest.approx(2 / 3 - (1 / 3) / payoff)
    assert every["Best Trade"] == pytest.approx(0.096)
    assert every["Worst Trade"] == pytest.approx(-32.2 / 600.0)
    assert every["Average Duration"] == pytest.approx(8 / 3)
    # Closed in the order win, loss, win.
    assert every["Longest Winning Streak"] == 1
    assert every["Longest Losing Streak"] == 1
    assert every["Current Streak"] == 1
    # A is held on days 0 to 3 (both lots on days 0 and 1), B on days 5 and 6.
    assert every["Coverage"] == pytest.approx(6 / 10)
    assert every["Overlap Coverage"] == pytest.approx(2 / 6)
    assert statistics.loc["Short Trades", "Short"] == 1
    assert statistics.loc["Win Rate", "Long"] == pytest.approx(0.5)


def test_streaks_and_a_trade_without_profit():
    pnl = [5.0, 3.0, 0.0, -1.0, -2.0, -4.0, 2.0, -1.0, -1.0]
    trades = pd.DataFrame(
        {
            "Ticker": "A",
            "Direction": "Long",
            "Status": "Closed",
            "Entry Date": DATES[0],
            "Exit Date": DATES[: len(pnl)],
            "Gross PnL": pnl,
            "Costs": 0.0,
            "PnL": pnl,
            "Return": np.array(pnl) / 100,
            "Duration (Days)": 1.0,
        }
    )

    every = trades_model.get_trade_statistics(trades, DATES)["All"]

    assert every["Winning Trades"] == 3
    assert every["Losing Trades"] == 5
    # The trade without profit counts in the closed trades but is neither a win nor a loss.
    assert every["Win Rate"] == pytest.approx(3 / 9)
    assert every["Longest Winning Streak"] == 2
    assert every["Longest Losing Streak"] == 3
    assert every["Current Streak"] == -2


def test_the_position_timing_by_hand():
    positions = pd.Series([0, 1, 1, -1, -1, 0, 0, 2, 2, 1], index=DATES)

    events = trades_model.get_flattening_and_flips(positions)

    assert events.to_dict() == {
        DATES[3]: "Flip",
        DATES[5]: "Flattening",
        DATES[9]: "Open",
    }
    # The long unit is held from day 1 to 3, the short one from day 3 to 5, and one of the two
    # units bought on day 7 is sold on day 9: (2 + 2 + 2) / 3. The unit still held has no holding period yet.
    assert trades_model.get_average_holding_period(positions) == pytest.approx(2.0)
    # Adding to a position moves its average entry time: units bought on days 0 and 2, all sold on day 4.
    added = pd.Series([1, 1, 2, 2, 0], index=DATES[:5])
    assert trades_model.get_average_holding_period(added) == pytest.approx((4 + 2) / 2)
    assert np.isnan(
        trades_model.get_average_holding_period(pd.Series(1.0, index=DATES))
    )
