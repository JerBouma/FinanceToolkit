"""Recorded outputs of the factor analysis details, groups, demeaning and binnings"""

import numpy as np
import pandas as pd

from financetoolkit.utilities import factor_model

TICKERS = [f"T{number:02d}" for number in range(12)]
GROUPS = pd.Series(
    {
        ticker: ("Alpha", "Beta", "Gamma")[number % 3]
        for number, ticker in enumerate(TICKERS)
    }
)


def monthly_data():
    """Twelve tickers over four years with a smooth factor that partly leads the returns."""
    dates = pd.bdate_range("2019-01-01", "2022-12-30")
    time = np.arange(len(dates))[:, None]
    ticker = np.arange(len(TICKERS))[None, :]
    log_returns = 0.012 * np.sin(0.013 * time * (ticker + 1) + ticker) + 0.0002 * ticker
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(log_returns, axis=0)), index=dates, columns=TICKERS
    )
    factor = pd.DataFrame(
        np.cos(0.004 * time * (ticker + 2) + 0.7 * ticker)
        + 0.3 * np.sin(0.02 * time + ticker),
        index=dates,
        columns=TICKERS,
    )

    return {"factor": factor}, prices


def analyse(**options):
    factors, prices = monthly_data()
    return factor_model.get_factor_analysis(
        factors, prices, horizons=[1, 3], quantiles=4, **options
    ).round(8)


def test_summary_with_ic_moments_and_alpha_beta(recorder):
    recorder.capture(analyse())
    recorder.capture(analyse(demeaned=True, groups=GROUPS))


def test_information_coefficient_by_calendar_and_group(recorder):
    recorder.capture(analyse(detail="information_coefficient_by_year"))
    recorder.capture(
        analyse(detail="information_coefficient_by_month", demeaned=True, groups=GROUPS)
    )
    recorder.capture(analyse(detail="group_information_coefficient", groups=GROUPS))


def test_quantile_returns_by_date_and_statistics(recorder):
    recorder.capture(analyse(detail="quantile_returns_by_date").iloc[:12])
    recorder.capture(analyse(detail="quantile_return_statistics"))
    recorder.capture(analyse(detail="quantile_return_statistics", demeaned=True))
    recorder.capture(analyse(detail="quantile_statistics"))


def test_binnings(recorder):
    recorder.capture(analyse(detail="quantile_statistics", binning="equal_width"))
    recorder.capture(analyse(detail="quantile_statistics", binning="zero_aware"))
    recorder.capture(
        analyse(detail="quantile_statistics", binning="group", groups=GROUPS)
    )


def test_factor_weights_returns_and_cumulative_returns(recorder):
    recorder.capture(analyse(detail="factor_weights").iloc[:12])
    recorder.capture(analyse(detail="factor_returns"))
    recorder.capture(analyse(detail="factor_returns", demeaned=True, groups=GROUPS))
    recorder.capture(analyse(detail="cumulative_returns"))


def test_event_returns(recorder):
    recorder.capture(analyse(detail="event_returns"))
    recorder.capture(analyse(detail="event_returns", demeaned=True, groups=GROUPS))
