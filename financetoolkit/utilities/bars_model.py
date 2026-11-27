"""Bars Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd

BAR_TYPES = ("volume", "dollar")


def get_bars(
    prices: pd.DataFrame,
    threshold: float,
    bar_type: str = "volume",
) -> pd.DataFrame:
    """
    Group consecutive days into volume or dollar bars: a bar closes on the day the volume (or
    the dollar volume) traded since the previous bar reaches `threshold`.

    López de Prado (2018, section 2.3.1) samples prices on a volume or dollar clock rather
    than a time clock: a bar holds the same amount of trading instead of the same amount of
    time, so more bars form when the market is active. Their returns are closer to
    independent and normally distributed than those of time bars, and dollar bars keep their
    meaning when the price changes a lot (or the share count changes through a split or
    buyback). The book forms them from ticks; on daily data each day is one tick, so a bar is
    a whole number of days:

    - Measure_t = Volume_t (volume bars) or Close_t * Volume_t (dollar bars)
    - A bar closes on day t when Σ Measure since the last bar ≥ threshold
    - Open = the Open of the first day, High = the highest High, Low = the lowest Low,
      Close = the Close of the last day
    - Volume = Σ Volume_t, Dollar Volume = Σ Close_t * Volume_t
    - VWAP = Dollar Volume / Volume

    The count restarts at zero after every bar, so the excess of the day that closes a bar
    does not carry over to the next one, as in the book. The days after the last full bar
    form no bar. A day without a Close or a Volume (a day the asset did not trade) is left
    out.

    The dollar volume uses the Close as the price of all of a day's volume (the book sums the
    price times the size of every trade), so the VWAP is a Close-weighted average price.

    Tick bars, and the imbalance and run bars of the book, need the individual trades and
    are not available from daily data.

    For more information about the method, see the following source:

    - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, sections
    2.3.1.3 and 2.3.1.4.

    Args:
        prices (pd.DataFrame): The daily prices of one asset with the columns Open, High,
            Low, Close and Volume (the Open, High and Low are optional) and the dates as the
            index.
        threshold (float): The volume (or dollar volume) at which a bar closes.
        bar_type (str, optional): "volume" for volume bars or "dollar" for dollar bars.
            Defaults to "volume".

    Returns:
        pd.DataFrame: One row per bar with the date of its last day as the index and the
        columns Start Date, Days, Open, High, Low, Close, Volume, Dollar Volume and VWAP.

    Raises:
        ValueError: If `bar_type` is not volume or dollar, or `threshold` is not positive.

    As an example:

    ```python
    from financetoolkit import Toolkit
    from financetoolkit.utilities import bars_model

    toolkit = Toolkit(
        ["AAPL"],
        api_key="FINANCIAL_MODELING_PREP_KEY",
        start_date="2025-01-01",
        end_date="2025-03-31",
    )

    prices = toolkit.get_historical_data().xs("AAPL", axis=1, level=1)

    bars_model.get_bars(prices, threshold=5e10, bar_type="dollar")
    ```

    Which returns:

    | Date       | Start Date   |   Days |   Open |   High |    Low |   Close |    Volume |   Dollar Volume |    VWAP |
    |:-----------|:-------------|-------:|-------:|-------:|-------:|--------:|----------:|----------------:|--------:|
    | 2025-01-08 | 2025-01-02   |      5 | 248.93 | 249.1  | 240.05 |  242.7  | 219515300 |     5.34506e+10 | 243.494 |
    | 2025-01-16 | 2025-01-10   |      5 | 240.01 | 240.16 | 228.03 |  228.26 | 262368000 |     6.13037e+10 | 233.655 |
    | 2025-01-22 | 2025-01-17   |      3 | 232.12 | 232.29 | 219.38 |  223.83 | 230685200 |     5.19388e+10 | 225.15  |
    | 2025-01-28 | 2025-01-23   |      4 | 224.74 | 240.19 | 221.41 |  238.26 | 285503700 |     6.55011e+10 | 229.423 |
    | 2025-02-03 | 2025-01-29   |      4 | 234.12 | 247.19 | 225.7  |  228.01 | 275167500 |     6.45971e+10 | 234.756 |
    | 2025-02-11 | 2025-02-04   |      6 | 227.25 | 235.23 | 226.65 |  232.62 | 241154100 |     5.57547e+10 | 231.199 |
    | 2025-02-19 | 2025-02-12   |      5 | 231.2  | 246.01 | 230.68 |  244.87 | 220780300 |     5.34909e+10 | 242.281 |
    | 2025-02-26 | 2025-02-20   |      5 | 244.94 | 250    | 239.13 |  240.36 | 229287600 |     5.62311e+10 | 245.243 |
    | 2025-03-05 | 2025-02-27   |      5 | 239.41 | 244.03 | 229.23 |  235.74 | 246196700 |     5.85676e+10 | 237.889 |
    | 2025-03-11 | 2025-03-06   |      4 | 234.44 | 241.37 | 217.45 |  220.84 | 239652600 |     5.49015e+10 | 229.088 |
    | 2025-03-18 | 2025-03-12   |      5 | 220.14 | 221.75 | 208.42 |  212.69 | 274529200 |     5.85843e+10 | 213.399 |
    | 2025-03-24 | 2025-03-19   |      4 | 214.22 | 221.48 | 211.28 |  220.73 | 241675600 |     5.2491e+10  | 217.196 |
    """
    if bar_type not in BAR_TYPES:
        raise ValueError("Bar type must be volume or dollar.")
    if not threshold > 0:
        raise ValueError("The threshold must be positive.")

    data = prices[prices["Close"].notna() & prices["Volume"].notna()]
    dollar_volume = data["Close"] * data["Volume"]
    measure = (data["Volume"] if bar_type == "volume" else dollar_volume).to_numpy()

    # The bar each day belongs to: the count restarts after the day that reaches the threshold.  # noqa: E501
    bar_numbers = np.empty(len(measure), dtype=int)
    number, total = 0, 0.0
    for day, value in enumerate(measure):
        bar_numbers[day] = number
        total += value
        if total >= threshold:
            number, total = number + 1, 0.0

    days = pd.DataFrame(
        {
            "Start Date": data.index,
            "End Date": data.index,
            "Days": 1,
            "Open": data["Open"] if "Open" in data else data["Close"],
            "High": data["High"] if "High" in data else data["Close"],
            "Low": data["Low"] if "Low" in data else data["Close"],
            "Close": data["Close"],
            "Volume": data["Volume"],
            "Dollar Volume": dollar_volume,
        },
        index=data.index,
    )
    # The days after the last full bar form no bar.
    days = days[bar_numbers < number]

    bars = days.groupby(bar_numbers[bar_numbers < number]).agg(
        {
            "Start Date": "first",
            "End Date": "last",
            "Days": "sum",
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
            "Dollar Volume": "sum",
        }
    )
    bars["VWAP"] = bars["Dollar Volume"] / bars["Volume"]

    return bars.set_index("End Date").rename_axis(prices.index.name or "Date")
