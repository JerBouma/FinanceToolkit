"""Campbell, Hilscher and Szilagyi Module"""

__docformat__ = "google"

import numpy as np
import pandas as pd

from financetoolkit.models.helpers import values_on_or_before

# pylint: disable=too-many-arguments,too-many-positional-arguments

# Campbell, Hilscher and Szilagyi (2008), Table IV, column (3): the logit of failure 12
# months ahead (lag of 12 months), estimated on US companies from 1963 to 2003.
COEFFICIENTS = {
    "Constant": -9.164,
    "NIMTAAVG": -20.264,
    "TLMTA": 1.416,
    "EXRETAVG": -7.129,
    "SIGMA": 1.411,
    "RSIZE": -0.045,
    "CASHMTA": -2.132,
    "MB": 0.075,
    "PRICE": -0.058,
}

# The paper winsorizes every variable at the 5th and 95th percentiles of its monthly
# cross-section of all US companies, which a calculation per company cannot repeat. The
# variables are clipped instead to the range the winsorized variables span in the paper's
# full sample (Table II, "Entire dataset", minimum and maximum). NIMTA is a quarterly
# figure and EXRET a monthly one, as in the paper.
WINSORIZATION_BOUNDS = {
    "NIMTA": (-0.069, 0.028),
    "TLMTA": (0.036, 0.923),
    "EXRET": (-0.243, 0.218),
    "RSIZE": (-13.568, -6.773),
    "SIGMA": (0.153, 1.353),
    "CASHMTA": (0.002, 0.358),
    "MB": (0.358, 6.471),
    "PRICE": (-0.065, 2.708),
}

# The weight of a lag halves every quarter: phi = 2^(-1/3) per month.
MONTHLY_DECAY = 2 ** (-1 / 3)

# The price above which the price per share no longer predicts failure.
PRICE_CAP = 15.0

# The share of the gap between market and book equity added to book equity (Cohen, Polk
# and Vuolteenaho, 2003), and the value a book equity that is still not positive becomes.
BOOK_EQUITY_ADJUSTMENT = 0.1
SMALL_BOOK_EQUITY = 1.0


def get_winsorized_variable(values: pd.DataFrame, variable: str) -> pd.DataFrame:
    """
    Clips a Campbell, Hilscher and Szilagyi (2008) variable to the range the winsorized
    variable spans in the paper's sample (Table II, minimum and maximum of the entire
    dataset).

    Args:
        values (pd.DataFrame): The variable per ticker (rows) and period (columns).
        variable (str): The variable: NIMTA, TLMTA, EXRET, RSIZE, SIGMA, CASHMTA, MB or PRICE.

    Returns:
        pd.DataFrame: The clipped variable; missing values stay NaN.
    """
    lower, upper = WINSORIZATION_BOUNDS[variable]

    return values.clip(lower=lower, upper=upper)


def get_market_total_assets(
    market_equity: pd.DataFrame, total_liabilities: pd.DataFrame
) -> pd.DataFrame:
    """
    The market value of total assets (MTA): the market value of the equity plus the book
    value of the liabilities.

    Args:
        market_equity (pd.DataFrame): The market capitalization.
        total_liabilities (pd.DataFrame): The total liabilities.

    Returns:
        pd.DataFrame: The market value of total assets.
    """
    return market_equity + total_liabilities


def get_adjusted_book_equity(
    book_equity: pd.DataFrame, market_equity: pd.DataFrame
) -> pd.DataFrame:
    """
    Book equity moved 10% of the way toward the market value of the equity, following
    Cohen, Polk and Vuolteenaho (2003), which Campbell, Hilscher and Szilagyi (2008) use so
    that very small, probably mismeasured book values do not create outliers. A book equity
    that is still not positive becomes a small positive value (1).

    The formula is as follows:

        - Adjusted Book Equity = Book Equity + 0.1 * (Market Equity - Book Equity)

    Args:
        book_equity (pd.DataFrame): The book value of the equity.
        market_equity (pd.DataFrame): The market capitalization.

    Returns:
        pd.DataFrame: The adjusted book equity.
    """
    adjusted = book_equity + BOOK_EQUITY_ADJUSTMENT * (market_equity - book_equity)

    return adjusted.mask(adjusted <= 0, SMALL_BOOK_EQUITY)


def get_geometric_average(values: list[pd.DataFrame], decay: float) -> pd.DataFrame:
    """
    The geometrically weighted average of a variable and its lags, with weights that
    decline by a factor decay per lag and add up to one:

        - Average = (1 - decay) / (1 - decay^n) * (X(t) + decay * X(t-1) + ... + decay^(n-1) * X(t-n+1))

    Campbell, Hilscher and Szilagyi (2008) average four quarters of NIMTA with
    decay = phi^3 and twelve months of EXRET with decay = phi, phi = 2^(-1/3), so the weight
    halves every quarter. The average is NaN when any of the lags is missing.

    Args:
        values (list[pd.DataFrame]): The variable, the latest first and each next one a
            period earlier.
        decay (float): The factor by which the weight declines per lag.

    Returns:
        pd.DataFrame: The weighted average.
    """
    lags = len(values)
    scale = (1 - decay) / (1 - decay**lags)
    total = values[0] * scale

    for lag, value in enumerate(values[1:], start=1):
        total = total + value * scale * decay**lag

    return total


def get_monthly_log_excess_returns(
    prices: pd.DataFrame,
    index_levels: pd.Series,
    dates: pd.DataFrame,
    months: int = 12,
) -> list[pd.DataFrame]:
    """
    The log excess returns of every month before a set of dates over a market index:

        - EXRET = ln(1 + R(company)) - ln(1 + R(index))

    where month k is the month that ends k months before the date (k = 0 being the month
    that ends on the date itself). A price is the last one on or before a day, at most ten
    days earlier.

    Args:
        prices (pd.DataFrame): The daily total return prices (Adj Close) with a DatetimeIndex
            and the tickers as columns.
        index_levels (pd.Series): The daily levels of the total return index with a
            DatetimeIndex.
        dates (pd.DataFrame): The dates per ticker (rows) and period (columns).
        months (int, optional): The number of months. Defaults to 12.

    Returns:
        list[pd.DataFrame]: The log excess return of each month per ticker and period, the
        latest month first.
    """
    index_frame = pd.DataFrame(
        {ticker: index_levels for ticker in dates.index}, index=index_levels.index
    )
    excess_returns = []

    for month in range(months):
        end = dates.apply(lambda column, k=month: column - pd.DateOffset(months=k))
        begin = dates.apply(
            lambda column, k=month: column - pd.DateOffset(months=k + 1)
        )

        company = np.log(
            values_on_or_before(prices, end) / values_on_or_before(prices, begin)
        )
        index = np.log(
            values_on_or_before(index_frame, end)
            / values_on_or_before(index_frame, begin)
        )
        excess_returns.append(company - index)

    return excess_returns


def get_sigma(
    returns: pd.DataFrame,
    dates: pd.DataFrame,
    months: int = 3,
    minimum_observations: int = 5,
) -> pd.DataFrame:
    """
    The annualized volatility of the daily returns over the months up to a date, measured
    around zero rather than around the mean, as Campbell, Hilscher and Szilagyi (2008) do:

        - SIGMA = sqrt(252 / (N - 1) * sum(r^2))

    over the N daily returns of the three months up to the date. With fewer than five
    non-zero returns the volatility is NaN.

    Args:
        returns (pd.DataFrame): The daily returns with a DatetimeIndex and the tickers as
            columns.
        dates (pd.DataFrame): The dates per ticker (rows) and period (columns).
        months (int, optional): The number of months. Defaults to 3.
        minimum_observations (int, optional): The minimum number of non-zero returns.
            Defaults to 5.

    Returns:
        pd.DataFrame: The volatility per ticker and period.
    """
    sigma = pd.DataFrame(np.nan, index=dates.index, columns=dates.columns)

    for ticker in dates.index:
        if ticker not in returns.columns:
            continue
        ticker_returns = returns[ticker].dropna()

        for period in dates.columns:
            date = dates.loc[ticker, period]
            if pd.isna(date):
                continue
            window = ticker_returns.loc[
                (ticker_returns.index > date - pd.DateOffset(months=months))
                & (ticker_returns.index <= date)
            ]
            if (window != 0).sum() < minimum_observations:
                continue
            sigma.loc[ticker, period] = np.sqrt(
                252 / (len(window) - 1) * (window**2).sum()
            )

    return sigma


def get_relative_size(
    market_equity: pd.DataFrame, index_market_value: pd.DataFrame
) -> pd.DataFrame:
    """
    The relative size (RSIZE): the log of the company's market capitalization relative to
    the total market value of the S&P 500 companies.

    Args:
        market_equity (pd.DataFrame): The market capitalization.
        index_market_value (pd.DataFrame): The total market value of the S&P 500 at the
            same dates.

    Returns:
        pd.DataFrame: The relative size.
    """
    return np.log(market_equity / index_market_value)


def get_log_price(price: pd.DataFrame) -> pd.DataFrame:
    """
    The log of the price per share capped at $15 (PRICE), as distressed companies tend to
    trade at low prices and the price no longer predicts failure above it.

    Args:
        price (pd.DataFrame): The price per share.

    Returns:
        pd.DataFrame: ln(min(price, 15)).
    """
    return np.log(price.clip(upper=PRICE_CAP))


def get_chs_score(
    net_income_to_market_total_assets_average: pd.DataFrame,
    total_liabilities_to_market_total_assets: pd.DataFrame,
    excess_return_average: pd.DataFrame,
    sigma: pd.DataFrame,
    relative_size: pd.DataFrame,
    cash_to_market_total_assets: pd.DataFrame,
    market_to_book: pd.DataFrame,
    log_price: pd.DataFrame,
) -> pd.DataFrame:
    """
    The logit of the probability of failure 12 months ahead of Campbell, Hilscher and
    Szilagyi (2008), Table IV, column (3):

        - LPFD = -9.164 - 20.264 NIMTAAVG + 1.416 TLMTA - 7.129 EXRETAVG + 1.411 SIGMA
          - 0.045 RSIZE - 2.132 CASHMTA + 0.075 MB - 0.058 PRICE

    Args:
        net_income_to_market_total_assets_average (pd.DataFrame): NIMTAAVG.
        total_liabilities_to_market_total_assets (pd.DataFrame): TLMTA.
        excess_return_average (pd.DataFrame): EXRETAVG.
        sigma (pd.DataFrame): SIGMA.
        relative_size (pd.DataFrame): RSIZE.
        cash_to_market_total_assets (pd.DataFrame): CASHMTA.
        market_to_book (pd.DataFrame): MB.
        log_price (pd.DataFrame): PRICE.

    Returns:
        pd.DataFrame: The logit (LPFD).
    """
    return (
        COEFFICIENTS["Constant"]
        + COEFFICIENTS["NIMTAAVG"] * net_income_to_market_total_assets_average
        + COEFFICIENTS["TLMTA"] * total_liabilities_to_market_total_assets
        + COEFFICIENTS["EXRETAVG"] * excess_return_average
        + COEFFICIENTS["SIGMA"] * sigma
        + COEFFICIENTS["RSIZE"] * relative_size
        + COEFFICIENTS["CASHMTA"] * cash_to_market_total_assets
        + COEFFICIENTS["MB"] * market_to_book
        + COEFFICIENTS["PRICE"] * log_price
    )


def get_chs_failure_probability(chs_score: pd.DataFrame) -> pd.DataFrame:
    """
    The probability of failure from the logit: 1 / (1 + e^(-LPFD)). With the 12-month
    coefficients it is the probability of failing in the month 12 months ahead, given
    that the company survives until then; it is a monthly probability, not the
    probability of failing within the next 12 months.

    Args:
        chs_score (pd.DataFrame): The logit (LPFD).

    Returns:
        pd.DataFrame: The failure probability as a decimal.
    """
    return 1 / (1 + np.exp(-chs_score))


def get_index_market_value(
    price_index: pd.Series, latest_market_value: pd.Series
) -> pd.Series:
    """
    The total market value of the S&P 500 companies on every day, scaled back from the
    latest total with the S&P 500 price index:

        - Market Value(t) = Market Value(latest) * Price Index(t) / Price Index(latest)

    The price index moves with the float-adjusted market value of its companies, apart from
    the changes S&P offsets with its divisor (companies entering and leaving the index,
    share issuance and buybacks), so the scaled value holds the companies and share counts
    of the latest date.

    Args:
        price_index (pd.Series): The daily S&P 500 price index with a DatetimeIndex.
        latest_market_value (pd.Series): The total market value of the S&P 500 companies,
            indexed by its date (one value).

    Returns:
        pd.Series: The total market value per day; NaN when either input is missing.
    """
    price_index = price_index.dropna()

    if latest_market_value.dropna().empty or price_index.empty:
        return pd.Series(np.nan, index=price_index.index, dtype=float)

    date = latest_market_value.dropna().index[-1]
    before = price_index.loc[:date]

    if before.empty:
        return pd.Series(np.nan, index=price_index.index, dtype=float)

    return price_index / before.iloc[-1] * float(latest_market_value.loc[date])
