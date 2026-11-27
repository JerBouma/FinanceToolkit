"""Portfolio Benchmark Model"""

__docformat__ = "google"

import pandas as pd


def get_benchmark_historical_data(benchmark_toolkit, period: str) -> pd.DataFrame:
    """
    Retrieve the historical data of a portfolio's benchmark with one column level.

    Args:
        benchmark_toolkit (Toolkit): The Toolkit of the benchmark ticker.
        period (str): The period of the data, such as "daily" or "weekly".

    Returns:
        pd.DataFrame: The historical data of the benchmark, one column per field.
    """
    data = benchmark_toolkit.get_historical_data(period=period)
    # A single benchmark can come back with the ticker as a second column level.
    if data.columns.nlevels > 1:
        data = data.droplevel(1, axis=1)
    return data
