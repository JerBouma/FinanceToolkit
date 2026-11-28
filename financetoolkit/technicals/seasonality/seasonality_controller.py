"""Seasonality Module"""

__docformat__ = "google"

import calendar

import pandas as pd

from financetoolkit.helpers import get_non_trading_days
from financetoolkit.technicals.helpers import (
    calculate_per_ticker,
    get_observations_per_year,
    report_missing_values,
)
from financetoolkit.technicals.seasonality import seasonality_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import (
    InvalidArgumentError,
    check_option,
    handle_errors,
    validate_window,
)
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import (
    apply_rounding,
)
from financetoolkit.utilities.transform_model import transform_result

# The examples' output tables are wider than the line length.
# ruff: noqa: E501


@report_missing_values
@filter_shown_columns
@track_sources
class SeasonalityTechnicals:
    """
    Seasonality and relative strength: the average return per calendar month or weekday, the day-of-week,
    turn-of-the-month and pre-holiday effects with Newey-West tests, and the price relative to the benchmark with the Mansfield relative strength and the relative strength rating.
    """

    @handle_errors
    def get_seasonality(
        self,
        frequency: str = "month",
        close_column: str = "Adj Close",
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the seasonality of every ticker's returns: the average return per calendar month or per day of the
        week.

        Monthly returns give the months and daily returns the days of the week, each with the median, the hit rate
        (the share of positive returns), the t-statistic of the average and the number of observations. Well-known
        examples are the January effect and the "sell in May" pattern. A pattern needs many years and a t-statistic
        well above 2 before it says much, and even then it can fade once it is widely traded on.

        A monthly return runs from one month's last close to the next, so a final month that is
        not complete yet (an end date before that month's last business day, as with the
        default end date of today) is left out rather than averaged as a full month. The
        Benchmark is included as a ticker, so its seasonality is there to compare with; a
        Portfolio is left out.

        For more information, see the following paper:

        - Heston, S.L. and Sadka, R. (2008). "Seasonality in the Cross-Section of Stock
          Returns." Journal of Financial Economics, 87(2), 418-445.

        Also known as: seasonality, monthly seasonality, day of the week effect, January
        effect, calendar anomalies.

        Args:
            frequency (str, optional): Options are: "month" (per calendar month), "weekday" (per day of the
                week). Defaults to "month".
            close_column (str, optional): The name of the column containing the close prices.
                Defaults to "Adj Close".
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the Toolkit.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Average Return, Median Return, Hit Rate, t-Statistic and
            Observations per month or weekday, with a column per statistic and ticker.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2000-01-01",
            end_date="2025-12-31",
        )

        seasonality = toolkit.technicals.get_seasonality()

        seasonality.xs("AAPL", level=1, axis="columns")
        ```

        Which returns:

        | Month     |   Average Return |   Median Return |   Hit Rate |   t-Statistic |   Observations |
        |:----------|-----------------:|----------------:|-----------:|--------------:|---------------:|
        | January   |           0.0239 |          0.0477 |     0.6    |        0.8733 |             25 |
        | February  |           0.0139 |          0.0245 |     0.5385 |        0.7962 |             26 |
        | March     |           0.0486 |          0.0529 |     0.6538 |        2.6472 |             26 |
        | April     |           0.028  |          0.0057 |     0.5769 |        1.5009 |             26 |
        | May       |           0.018  |          0.0457 |     0.5769 |        0.7223 |             26 |
        | June      |           0.0171 |          0.0096 |     0.5385 |        0.8119 |             26 |
        | July      |           0.0523 |          0.0501 |     0.7692 |        2.7965 |             26 |
        | August    |           0.0501 |          0.0466 |     0.6538 |        3.2948 |             26 |
        | September |          -0.0252 |         -0.0128 |     0.4231 |       -0.8129 |             26 |
        | October   |           0.0529 |          0.0616 |     0.7308 |        2.4423 |             26 |
        | November  |           0.0306 |          0.0334 |     0.5769 |        1.4177 |             26 |
        | December  |          -0.0061 |          0.0112 |     0.5385 |       -0.4179 |             26 |
        """
        self._check_arguments("daily", close_column)
        check_option("frequency", frequency, ("month", "weekday"))
        # Per ticker over the days it traded, so a stock in a Toolkit with a cryptocurrency
        # gets no weekend days without a move.
        returns = calculate_per_ticker(
            self._historical_data["monthly" if frequency == "month" else "daily"],
            [close_column],
            lambda prices: prices.pct_change(fill_method=None),
        ).loc[self._start_date : self._end_date]

        seasonality = pd.concat(
            {
                ticker: seasonality_model.get_seasonality(
                    returns[ticker], frequency, end_date=self._end_date
                )
                for ticker in returns.columns
                if ticker != "Portfolio"
            },
            axis=1,
        ).swaplevel(0, 1, axis=1)

        return apply_rounding(
            seasonality, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_calendar_effects(
        self,
        effect: str = "day_of_week",
        close_column: str = "Adj Close",
        days_before: int = 1,
        days_after: int = 3,
        lags: int | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the calendar effects in every ticker's daily returns: whether returns differ by weekday, around
        the turn of the month or before holidays, tested with standard errors that allow for volatility clustering
        and autocorrelation.

        Each effect is a regression of the daily returns on calendar dummies, with Newey-West (1987) standard
        errors, as the plain t-statistics of `get_seasonality` assume independent returns of equal variance.

        - Day-of-week effect: the average return per weekday from a regression on a dummy per weekday, and the
          joint test of whether all weekdays share one average (French, 1980; Gibbons and Hess, 1981). The
          Monday effect, a lower return over the weekend, was strong in US stocks until the 1980s.
        - Turn-of-the-month effect: the average return of the last trading day of a month and the first three of
          the next against the other days (Ariel, 1987; Lakonishok and Smidt, 1988), which once held most of the
          month's return.
        - Pre-holiday effect: the average return of the trading day before an exchange holiday against the other
          days (Lakonishok and Smidt, 1988; Ariel, 1990). The holidays are read from each ticker's own dates: a
          day of the week it regularly trades on without trading counts as a holiday.

        The formula for the day-of-week effect is as follows:

        - Return_t = b_Monday * D_Monday,t + ... + b_Friday * D_Friday,t + e_t, with b_d the Average Return of weekday d
        - Equal Means = W / q, the Wald statistic of b_Monday = ... = b_Friday under the Newey-West covariance over
          the number of restrictions q, against the F(q, n - k) distribution

        The formula for the turn-of-the-month and the pre-holiday effect is as follows:

        - Return_t = a + b * D_t + e_t, with D_t = 1 on the turn-of-the-month (or pre-holiday) days and 0 on the others
        - Other Days = a, Turn of Month (or Pre-Holiday) = a + b and Difference = b

        Every Test Statistic is the estimate over its Newey-West standard error with a two-sided p-value from the
        standard normal, apart from the F-statistic of the Equal Means row. The Newey-West covariance uses
        Bartlett weights over `lags` lags, by default floor(4 * (n / 100)^(2/9)) for n returns (Newey and West,
        1994).

        For more information, see the following papers:

        - French, K.R. (1980). "Stock Returns and the Weekend Effect." Journal of Financial Economics, 8(1),
          55-69.
        - Gibbons, M.R. and Hess, P. (1981). "Day of the Week Effects and Asset Returns." The Journal of
          Business, 54(4), 579-596.
        - Ariel, R.A. (1987). "A Monthly Effect in Stock Returns." Journal of Financial Economics, 18(1),
          161-174.
        - Lakonishok, J. and Smidt, S. (1988). "Are Seasonal Anomalies Real? A Ninety-Year Perspective." The
          Review of Financial Studies, 1(4), 403-425.
        - Ariel, R.A. (1990). "High Stock Returns before Holidays: Existence and Evidence on Possible Causes."
          The Journal of Finance, 45(5), 1611-1626.
        - Newey, W.K. and West, K.D. (1987). "A Simple, Positive Semi-Definite, Heteroskedasticity and
          Autocorrelation Consistent Covariance Matrix." Econometrica, 55(3), 703-708.

        Referenced against:
        - statsmodels, OLS (HAC covariance and f_test): https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html

        Also known as: calendar anomalies, day of the week effect, weekend effect, Monday effect, turn of the
        month effect, pre-holiday effect, holiday effect.

        Args:
            effect (str, optional): The calendar effect to test. Options are: "day_of_week" (the day-of-week
                effect), "turn_of_month" (the turn-of-the-month effect), "pre_holiday" (the pre-holiday effect).
                Defaults to "day_of_week".
            close_column (str, optional): The name of the column containing the close prices.
                Defaults to "Adj Close".
            days_before (int, optional): The last trading days of a month in the turn of the month. Defaults to 1.
            days_after (int, optional): The first trading days of a month in the turn of the month. Defaults to 3.
            lags (int | None, optional): The number of lags of the Newey-West covariance. Defaults to None, which
                uses floor(4 * (n / 100)^(2/9)) for n returns.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the Toolkit.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Average Return, Standard Error, Test Statistic, p-Value and Observations, with a
            column per statistic and ticker: per weekday and an Equal Means row for the day-of-week effect, and
            for the Turn of Month (or Pre-Holiday) days, the Other Days and the Difference for the other effects.

        Notes:
        - The returns are the daily returns of each ticker on the days it traded. The Benchmark is included as a
          ticker; a Portfolio is left out.
        - The turn of the month needs whole months: a first month whose returns start after its first business
          day, or a last month that ends before its last business day, is left out of that effect.
        - The pre-holiday effect takes every missing regular trading day as a holiday, so a gap in a ticker's
          data counts as one too. A ticker that trades on weekends, such as a cryptocurrency, has no holidays
          and no pre-holiday effect.
        - Calendar effects are found by searching many calendars in the same data, so a p-value just below 0.05
          says little; an effect should hold in other periods and markets too.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2000-01-01",
            end_date="2025-12-31",
        )

        calendar_effects = toolkit.technicals.get_calendar_effects(effect="turn_of_month")

        calendar_effects.xs("AAPL", level=1, axis="columns")
        ```

        Which returns:

        |               |   Average Return |   Standard Error |   Test Statistic |   p-Value |   Observations |
        |:--------------|-----------------:|-----------------:|-----------------:|----------:|---------------:|
        | Turn of Month |           0.0011 |           0.0008 |           1.4347 |    0.1514 |           1248 |
        | Other Days    |           0.0012 |           0.0003 |           4.0196 |    0.0001 |           5291 |
        | Difference    |          -0.0001 |           0.0008 |          -0.108  |    0.914  |           6539 |
        """
        self._check_arguments("daily", close_column)
        check_option("effect", effect, ("day_of_week", "turn_of_month", "pre_holiday"))
        validate_window("days_before", days_before, 0)
        validate_window("days_after", days_after, 0)
        validate_window("lags", lags, 0)
        # Per ticker over the days it traded, so the calendar is each ticker's own.
        returns = calculate_per_ticker(
            self._historical_data["daily"],
            [close_column],
            lambda prices: prices.pct_change(fill_method=None),
        ).loc[self._start_date : self._end_date]

        effects = {}
        for ticker in returns.columns:
            if ticker == "Portfolio":
                continue
            values = returns[ticker].dropna()
            if effect == "day_of_week":
                effects[ticker] = seasonality_model.get_day_of_week_effect(values, lags)
            elif effect == "turn_of_month":
                effects[ticker] = seasonality_model.get_calendar_effect(
                    values,
                    seasonality_model.get_turn_of_month_days(
                        values.index, days_before, days_after
                    ),
                    "Turn of Month",
                    lags,
                )
            else:
                effects[ticker] = seasonality_model.get_calendar_effect(
                    values,
                    seasonality_model.get_pre_holiday_days(values.index),
                    "Pre-Holiday",
                    lags,
                )

        calendar_effects = pd.concat(effects, axis=1).swaplevel(0, 1, axis=1)

        if effect == "day_of_week":
            # The weekdays in calendar order, then the joint test, whichever ticker trades on which days.
            calendar_effects = calendar_effects.reindex(
                [day for day in calendar.day_name if day in calendar_effects.index]
                + ["Equal Means"]
            )

        return apply_rounding(
            calendar_effects, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    @transform_result
    def get_relative_strength(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int | None = None,
        rounding: int | None = None,
        transform: str | list[str] | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the relative strength of every ticker against the Benchmark and against the other tickers: which
        tickers outperform and by how much.

        Relative strength investing buys what has been outperforming, the momentum effect that has held across
        markets and decades.

        - Relative Strength: the price divided by the Benchmark's, set to 1 on the first date.
          It rises while the ticker outperforms the Benchmark.
        - Mansfield Relative Strength: the relative strength against its own moving average
          (of `window` periods, a year by default) minus 1. Above zero the ticker is gaining on
          the Benchmark.
        - Relative Strength Rating: the ticker's performance over the last year, the last quarter
          weighing most (the 3, 6, 9 and 12-month returns weighted 0.4, 0.2, 0.2 and
          0.2), ranked against the other tickers from 1 (weakest) to 99 (strongest), from daily
          prices. Investor's Business Daily publishes the original rating with a proprietary
          formula; this weighting is its common public approximation. Tickers with the same
          score share the highest rank among them, so the rating reads as the share of the
          other tickers outperformed or matched and tied leaders both rate 99. It needs many
          tickers to be meaningful, such as the constituents of an index (see
          `Discovery.get_index_constituents`, whose date argument gives the members on a past date;
          today's members leave out the companies removed from the index since).

        For more information, see the following papers and books:

        - Jegadeesh, N. and Titman, S. (1993). "Returns to Buying Winners and Selling Losers:
          Implications for Stock Market Efficiency." The Journal of Finance, 48(1), 65-91.
        - Weinstein, S. (1988). "Stan Weinstein's Secrets for Profiting in Bull and Bear
          Markets." Dow Jones-Irwin.
        - O'Neil, W.J. (1988). "How to Make Money in Stocks." McGraw-Hill.

        Also known as: relative strength, price relative, Mansfield relative strength, RS
        rating, relative performance.

        Args:
            period (str, optional): The time period to consider for historical data. Options are: "daily" (daily
                prices), "weekly" (weekly prices), "monthly" (monthly prices), "quarterly" (quarterly prices),
                "yearly" (yearly prices). Defaults to "daily".
            close_column (str, optional): The name of the column containing the close prices.
                Defaults to "Adj Close".
            window (int | None, optional): The moving average window of the Mansfield relative
                strength, in periods. Defaults to None, which uses a year of periods: 252 daily
                (365 for a ticker that also trades on weekends, such as a cryptocurrency), 52 weekly, 12 monthly, 4 quarterly and 1 yearly (a single period, which makes the
                measure zero, so pass a longer window for yearly data).
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the Toolkit.
            transform (str | list[str] | None, optional): Time series steps applied to the result in order,
                such as "sma:12" (moving average), "ema:12", "std:12", "zscore:60", "percentile:120",
                "yoy", "pct_change:3,annualize:3", "log", "diff", "cumsum" or "hp_trend" and
                "hp_cycle" (Hodrick-Prescott), along the dates of each series. A window needs that many
                periods of the result, so its first periods are empty. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Relative Strength, Mansfield Relative Strength and Relative Strength
            Rating per date, with a column per measure and ticker.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2022-01-01",
            end_date="2025-12-31",
        )

        relative_strength = toolkit.technicals.get_relative_strength()

        relative_strength["Relative Strength Rating"].tail()
        ```

        Which returns:

        | Date       |   AAPL |   MSFT |   NVDA |   AMZN |   GOOGL |
        |:-----------|-------:|-------:|-------:|-------:|--------:|
        | 2025-12-24 |     50 |      1 |     74 |     26 |      99 |
        | 2025-12-26 |     50 |      1 |     74 |     26 |      99 |
        | 2025-12-29 |     50 |      1 |     74 |     26 |      99 |
        | 2025-12-30 |     50 |      1 |     74 |     26 |      99 |
        | 2025-12-31 |     50 |      1 |     74 |     26 |      99 |
        """
        self._check_arguments(period, close_column, window=window)
        if period == "intraday":
            raise InvalidArgumentError(
                "The relative strength needs daily or longer periods."
            )
        prices = self._close_prices(period, close_column)
        if "Benchmark" not in prices:
            raise ValueError(
                "The relative strength needs the Benchmark of the Toolkit."
            )
        tickers = [
            ticker
            for ticker in prices.columns
            if ticker not in ("Portfolio", "Benchmark")
        ]
        benchmark = prices["Benchmark"]

        strength = calculate_per_ticker(
            self._historical_data[period],
            [close_column],
            seasonality_model.get_period_relative_strength,
            benchmark_prices=benchmark,
            period=period,
            window=window,
        )

        # Each ticker's score over the days it traded, its quarters counted in its own days
        # (63 trading days, or 91 calendar days for a ticker that trades on weekends). On a
        # day a ticker did not trade its latest score still ranks the others, but it gets
        # no rating itself.
        daily_data = self._historical_data["daily"]
        scores = calculate_per_ticker(
            daily_data,
            [close_column],
            lambda ticker_prices: seasonality_model.get_relative_strength_score(
                ticker_prices, get_observations_per_year(ticker_prices)
            ),
        )[tickers]
        rating = seasonality_model.get_relative_strength_rating(
            scores.ffill(limit_area="inside")
        ).where(~get_non_trading_days(daily_data, close_column)[tickers])
        if period != "daily":
            rating = rating.groupby(rating.index.asfreq(prices.index.freqstr)).last()
            rating = rating.reindex(prices.index)

        relative_strength = pd.concat(
            {
                "Relative Strength": strength["Relative Strength"][tickers],
                "Mansfield Relative Strength": strength["Mansfield Relative Strength"][
                    tickers
                ],
                "Relative Strength Rating": rating,
            },
            axis=1,
        ).loc[self._start_date : self._end_date]

        return apply_rounding(
            relative_strength, rounding if rounding is not None else self._rounding
        )
