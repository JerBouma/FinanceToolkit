"""Factors Module"""

__docformat__ = "google"

import inspect
from collections.abc import Callable

import pandas as pd

from financetoolkit.factors import factors_model
from financetoolkit.ratios.helpers import map_period_data_to_daily_data
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.statistics_model import apply_rounding

# The examples' output tables are wider than the line length.
# ruff: noqa: E501

# The forward return horizons per period frequency when none are given: about a period, a
# quarter and a year ahead.
DEFAULT_HORIZONS = {
    "daily": [1, 5, 21],
    "weekly": [1, 4, 13],
    "monthly": [1, 3, 12],
    "quarterly": [1, 2, 4],
    "yearly": [1, 2, 3],
}


class Factors:
    """
    The Factors module tests a metric as a cross-sectional signal: whether the tickers that
    score high on it go on to return more than those that score low. Any ratio or technical
    indicator of the Toolkit, or a DataFrame of your own, can serve as the factor. It gives
    the information coefficient, the forward return per factor quantile and the top minus
    bottom spread, the turnover of the quantiles and the rank autocorrelation of the factor,
    following alphalens.
    """

    def __init__(
        self,
        tickers: list[str] | Callable[[], list[str]],
        historical_data: dict[str, pd.DataFrame],
        factor_modules: Callable[[], dict],
        period_end_dates: Callable[[], pd.DataFrame],
        quarterly: bool | None = None,
        rounding: int | None = 4,
        start_date: str | None = None,
        end_date: str | None = None,
    ):
        """
        Initializes the Factors Controller Class.

        Args:
            tickers (list[str] | Callable[[], list[str]]): The tickers of the cross-section.
            historical_data (dict[str, pd.DataFrame]): The historical data per period frequency.
            factor_modules (Callable[[], dict]): Returns the Toolkit modules (ratios,
                technicals) whose methods can be named as the factor.
            period_end_dates (Callable[[], pd.DataFrame]): Returns the day from which each
                statement period's figures apply per ticker (the fiscal period end, or the
                filing day with point_in_time).
            quarterly (bool | None, optional): Whether the statements are quarterly. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            start_date (str | None, optional): The start date of the evaluation. Defaults to None.
            end_date (str | None, optional): The end date of the evaluation. Defaults to None.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "JNJ", "XOM", "PG", "KO", "WMT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2016-01-01",
            end_date="2025-12-31",
        )

        toolkit.factors.get_information_coefficient("get_price_to_earnings_ratio", summary=True)
        ```

        Which returns:
        |                       |   1 Month |   3 Months |   12 Months |
        |:----------------------|----------:|-----------:|------------:|
        | Mean IC               |    0.0109 |     0.0129 |      0.0107 |
        | IC Standard Deviation |    0.3955 |     0.3738 |      0.3327 |
        | IC Information Ratio  |    0.0276 |     0.0345 |      0.0322 |
        | t-Statistic           |    0.2903 |     0.3602 |      0.3215 |
        | p-Value               |    0.7721 |     0.7194 |      0.7485 |
        | Positive Share        |    0.5315 |     0.5138 |      0.53   |
        | Observations          |  111      |   109      |    100      |
        """
        self._tickers_source = tickers
        self._historical_data = historical_data
        self._factor_modules = factor_modules
        self._period_end_dates = period_end_dates
        self._quarterly = quarterly
        self._rounding = rounding
        self._start_date = start_date
        self._end_date = end_date

    @property
    def _tickers(self) -> list[str]:
        """The tickers of the cross-section, without the benchmark and the portfolio."""
        tickers: list[str] = (
            self._tickers_source()
            if callable(self._tickers_source)
            else self._tickers_source
        )

        return [
            ticker for ticker in tickers if ticker not in ("Benchmark", "Portfolio")
        ]

    def _get_daily_factor(self, factor: str | pd.DataFrame) -> pd.DataFrame:
        """
        Returns the factor per day (index) and ticker (columns).

        A name is looked up among the ratios and the technical indicators. A ratio is
        calculated daily where it can be (show_daily) and otherwise per statement period and
        applied from the day each period's figures are known, so no day uses figures of a
        period that had not ended (or, with point_in_time, not been filed) yet.

        Args:
            factor (str | pd.DataFrame): The name of a ratio or technical indicator method, or
                the factor itself, either per date (index) and ticker (columns) or per ticker
                (index) and statement period (columns).

        Returns:
            pd.DataFrame: The daily factor of the tickers of the Toolkit.
        """
        if isinstance(factor, str):
            for module in self._factor_modules().values():
                method = getattr(module, factor, None)
                if method is not None:
                    break
            else:
                raise ValueError(
                    f"{factor} is not a ratio or technical indicator of the Toolkit. "
                    "Pass the factor as a DataFrame instead."
                )

            parameters = inspect.signature(method).parameters
            factor = method(show_daily=True) if "show_daily" in parameters else method()

        if not isinstance(factor, pd.DataFrame):
            raise TypeError("The factor must be a method name or a DataFrame.")

        if factor.columns.nlevels > 1:
            raise ValueError(
                "The factor must have one value per ticker and date; select one column level "
                "of a factor with several values per ticker."
            )

        tickers = self._tickers

        # A factor with the tickers as rows holds statement periods, which apply from the day
        # each period's figures are known.
        if factor.index.isin(tickers).any() and not factor.columns.isin(tickers).any():
            prices = self._historical_data["daily"]["Adj Close"]
            factor = map_period_data_to_daily_data(
                period_data=factor.loc[factor.index.isin(tickers)],
                daily_dates=prices.index,
                quarterly=bool(self._quarterly),
                period_end_dates=self._period_end_dates(),
            )

        factor = factor.loc[:, factor.columns.isin(tickers)]

        if factor.shape[1] < 3:  # noqa: PLR2004
            raise ValueError(
                "A factor needs at least three tickers to compare; twenty or more give "
                "meaningful quantiles."
            )

        dates = factor.index
        factor.index = (
            dates.to_timestamp()
            if isinstance(dates, pd.PeriodIndex)
            else pd.DatetimeIndex(dates)
        )

        return factor.loc[self._start_date : self._end_date].astype(float)

    def _get_factor_and_prices(
        self, factor: str | pd.DataFrame, period: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns the factor and the adjusted closing prices at the end of each period, on the
        same periods and tickers.

        Args:
            factor (str | pd.DataFrame): The factor, see _get_daily_factor.
            period (str): The period frequency: daily, weekly, monthly, quarterly or yearly.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: The factor and the prices per period.
        """
        if period not in factors_model.PERIOD_FREQUENCIES:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly or yearly."
            )

        daily_factor = self._get_daily_factor(factor)
        prices = self._historical_data["daily"]["Adj Close"].loc[
            :, daily_factor.columns
        ]

        sampled_prices = factors_model.sample_by_period(prices, period)
        sampled_factor = factors_model.sample_by_period(daily_factor, period).reindex(
            sampled_prices.index
        )

        # Periods before the start date only serve as prices, never as factor dates.
        sampled_factor = sampled_factor.loc[sampled_factor.notna().any(axis=1)]

        return sampled_factor, sampled_prices

    @handle_errors
    def get_information_coefficient(
        self,
        factor: str | pd.DataFrame,
        period: str = "monthly",
        horizons: list[int] | int | None = None,
        summary: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculates the information coefficient (IC) of a factor: per period, the Spearman rank
        correlation between the factor of the tickers at the end of the period and their
        return over the following horizon. An IC of 1 means the factor ranks the tickers'
        later returns perfectly, -1 that it ranks them in reverse and 0 that it says nothing
        about them. The IC measures skill in Grinold and Kahn's fundamental law of active
        management, IR ≈ IC × √breadth: a small but consistent IC across many tickers and
        periods gives a high information ratio.

        The returns use the dividend and split adjusted close and start at the close of the
        period the factor belongs to, so a factor is only compared with returns that came
        after it. A ratio from the financial statements applies from the end of each fiscal
        period (or, with point_in_time, from its filing day), never earlier. Each ticker keeps
        its own history: a ticker without a factor or a return in a period is left out of
        that period only.

        With summary=True it returns, per horizon, the mean IC, its standard deviation, the
        IC information ratio (mean divided by standard deviation), the t-statistic of the mean
        with its p-value, the share of periods with a positive IC and the number of periods.
        A horizon longer than one period makes consecutive returns overlap, which inflates
        the t-statistic; judge those horizons by their mean IC and information ratio.

        Args:
            factor (str | pd.DataFrame): The name of a ratio or technical indicator method,
                e.g. "get_price_to_earnings_ratio", "get_return_on_equity" or
                "get_rate_of_change", or a DataFrame of your own with the dates as index and
                the tickers as columns.
            period (str, optional): The frequency at which the factor is evaluated: daily,
                weekly, monthly, quarterly or yearly. Defaults to "monthly".
            horizons (list[int] | int | None, optional): The number of periods the forward
                returns cover. Defaults to None, which uses about a period, a quarter and a
                year ahead (1, 3 and 12 months for monthly).
            summary (bool, optional): Whether to return the summary statistics instead of the
                IC per period. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.

        Returns:
            pd.DataFrame: The IC per period (index) and horizon (columns), or with summary
            the statistics (index) per horizon (columns).

        Notes:
            - The cross-section is the tickers of the Toolkit, so use at least ten and
              preferably twenty or more for a meaningful rank correlation.
            - A low P/E stock that goes on to outperform gives a negative IC for the P/E:
              read the sign as the direction of the relationship.
            - Based on alphalens' factor_information_coefficient and factor_information
              table (Quantopian).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "JNJ", "XOM", "PG", "KO", "WMT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2016-01-01",
            end_date="2025-12-31",
        )

        # Twelve-month momentum: the return over the last 252 trading days.
        momentum = toolkit.technicals.get_rate_of_change(window=252)

        toolkit.factors.get_information_coefficient(momentum, summary=True)
        ```

        Which returns:
        |                       |   1 Month |   3 Months |   12 Months |
        |:----------------------|----------:|-----------:|------------:|
        | Mean IC               |    0.0414 |     0.0968 |      0.0969 |
        | IC Standard Deviation |    0.3903 |     0.4009 |      0.4562 |
        | IC Information Ratio  |    0.1061 |     0.2415 |      0.2125 |
        | t-Statistic           |    1.0976 |     2.4744 |      2.0818 |
        | p-Value               |    0.2749 |     0.015  |      0.0401 |
        | Positive Share        |    0.5514 |     0.6476 |      0.5833 |
        | Observations          |  107      |   105      |     96      |
        """
        horizon_list = self._get_horizons(period, horizons)
        sampled_factor, sampled_prices = self._get_factor_and_prices(factor, period)

        information_coefficient = pd.DataFrame(
            {
                factors_model.get_horizon_label(
                    horizon, period
                ): factors_model.get_information_coefficient(
                    sampled_factor,
                    factors_model.get_forward_returns(sampled_prices, horizon).reindex(
                        sampled_factor.index
                    ),
                )
                for horizon in horizon_list
            },
            index=sampled_factor.index,
        )

        if summary:
            result = factors_model.summarize_information_coefficient(
                information_coefficient
            )
        else:
            result = information_coefficient.dropna(how="all")
            result.index.name = "Date"

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_quantile_returns(
        self,
        factor: str | pd.DataFrame,
        quantiles: int = 5,
        period: str = "monthly",
        horizons: list[int] | int | None = None,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculates the mean forward return of each factor quantile and the top minus bottom
        spread. At the end of each period the tickers are sorted into equally sized quantiles
        of the factor, Quantile 1 holding the lowest values; the return of a quantile is the
        equally weighted return of its tickers over the following horizon, averaged over the
        periods. The top minus bottom spread is the average of the highest quantile's return
        minus the lowest quantile's return, the return of a portfolio long the top and short
        the bottom quantile. Returns rising steadily from the first to the last quantile point
        to a factor that orders returns across the whole cross-section, not only at the
        extremes.

        The returns use the dividend and split adjusted close and start at the close of the
        period the factor belongs to; they cover the whole horizon and are not converted to a
        return per period. A ratio from the financial statements applies from the end of each
        fiscal period (or, with point_in_time, from its filing day).

        Args:
            factor (str | pd.DataFrame): The name of a ratio or technical indicator method,
                e.g. "get_price_to_earnings_ratio", "get_return_on_equity" or
                "get_rate_of_change", or a DataFrame of your own with the dates as index and
                the tickers as columns.
            quantiles (int, optional): The number of quantiles. Defaults to 5.
            period (str, optional): The frequency at which the quantiles are formed: daily,
                weekly, monthly, quarterly or yearly. Defaults to "monthly".
            horizons (list[int] | int | None, optional): The number of periods the forward
                returns cover. Defaults to None, which uses about a period, a quarter and a
                year ahead (1, 3 and 12 months for monthly).
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.

        Returns:
            pd.DataFrame: The mean forward return of each quantile and the "Top Minus Bottom"
            spread (index) per horizon (columns).

        Notes:
            - A period with fewer tickers with a factor than quantiles is left out. Twenty or
              more tickers keep at least four in each of five quantiles; with ten, use two or
              three quantiles.
            - Equal factor values are split over adjacent quantiles by the order of the
              tickers, so every quantile holds about the same number of tickers.
            - Based on alphalens' mean_return_by_quantile (Quantopian).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "JNJ", "XOM", "PG", "KO", "WMT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2016-01-01",
            end_date="2025-12-31",
        )

        # Twelve-month momentum: the return over the last 252 trading days.
        momentum = toolkit.technicals.get_rate_of_change(window=252)

        toolkit.factors.get_quantile_returns(momentum, quantiles=3)
        ```

        Which returns:
        |                  |   1 Month |   3 Months |   12 Months |
        |:-----------------|----------:|-----------:|------------:|
        | Quantile 1       |    0.0166 |     0.0522 |      0.2559 |
        | Quantile 2       |    0.017  |     0.0514 |      0.2163 |
        | Quantile 3       |    0.0241 |     0.0735 |      0.2986 |
        | Top Minus Bottom |    0.0075 |     0.0214 |      0.0427 |
        """
        horizon_list = self._get_horizons(period, horizons)
        sampled_factor, sampled_prices = self._get_factor_and_prices(factor, period)
        quantile_assignment = factors_model.assign_quantiles(sampled_factor, quantiles)

        result = pd.DataFrame(
            {
                factors_model.get_horizon_label(
                    horizon, period
                ): factors_model.get_quantile_returns(
                    quantile_assignment,
                    factors_model.get_forward_returns(sampled_prices, horizon).reindex(
                        sampled_factor.index
                    ),
                )
                for horizon in horizon_list
            }
        )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_quantile_turnover(
        self,
        factor: str | pd.DataFrame,
        quantiles: int = 5,
        period: str = "monthly",
        periods_apart: int = 1,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculates the turnover of each factor quantile per period: the share of the tickers
        in the quantile that were not in it periods_apart periods before. A factor whose quantiles turn
        over quickly needs frequent trading to follow, so its returns cost more to capture
        than a slow-moving factor's.

        Args:
            factor (str | pd.DataFrame): The name of a ratio or technical indicator method,
                e.g. "get_price_to_earnings_ratio", "get_return_on_equity" or
                "get_rate_of_change", or a DataFrame of your own with the dates as index and
                the tickers as columns.
            quantiles (int, optional): The number of quantiles. Defaults to 5.
            period (str, optional): The frequency at which the quantiles are formed: daily,
                weekly, monthly, quarterly or yearly. Defaults to "monthly".
            periods_apart (int, optional): The number of periods between the two dates
                compared. Defaults to 1.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.

        Returns:
            pd.DataFrame: The turnover per period (index) and quantile (columns).

        Notes:
            - The first periods_apart periods have no turnover, nor has a period whose earlier period
              had too few tickers to form quantiles.
            - Based on alphalens' quantile_turnover (Quantopian).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "JNJ", "XOM", "PG", "KO", "WMT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2016-01-01",
            end_date="2025-12-31",
        )

        # Twelve-month momentum: the return over the last 252 trading days.
        momentum = toolkit.technicals.get_rate_of_change(window=252)

        toolkit.factors.get_quantile_turnover(momentum, quantiles=3).tail()
        ```

        Which returns:
        | Date    |   Quantile 1 |   Quantile 2 |   Quantile 3 |
        |:--------|-------------:|-------------:|-------------:|
        | 2025-08 |         0    |         0.25 |         0.25 |
        | 2025-09 |         0    |         0.5  |         0.5  |
        | 2025-10 |         0.25 |         0.5  |         0.25 |
        | 2025-11 |         0.25 |         0.25 |         0.25 |
        | 2025-12 |         0.25 |         0.25 |         0    |
        """
        sampled_factor, _ = self._get_factor_and_prices(factor, period)
        quantile_assignment = factors_model.assign_quantiles(sampled_factor, quantiles)

        result = factors_model.get_quantile_turnover(
            quantile_assignment, periods_apart
        ).dropna(how="all")
        result.index.name = "Date"

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_factor_rank_autocorrelation(
        self,
        factor: str | pd.DataFrame,
        period: str = "monthly",
        periods_apart: int = 1,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculates the rank autocorrelation of a factor per period: the correlation between
        the ranks of the tickers' factor at the end of the period and periods_apart periods before,
        over the tickers with a factor on both. A value close to one means the factor ranks
        the tickers much as it did before, so a portfolio built on it changes little; a low
        value means a fast-moving factor with a high turnover.

        Args:
            factor (str | pd.DataFrame): The name of a ratio or technical indicator method,
                e.g. "get_price_to_earnings_ratio", "get_return_on_equity" or
                "get_rate_of_change", or a DataFrame of your own with the dates as index and
                the tickers as columns.
            period (str, optional): The frequency at which the factor is ranked: daily,
                weekly, monthly, quarterly or yearly. Defaults to "monthly".
            periods_apart (int, optional): The number of periods between the two dates
                compared. Defaults to 1.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.

        Returns:
            pd.DataFrame: The "Rank Autocorrelation" per period.

        Notes:
            - Based on alphalens' factor_rank_autocorrelation (Quantopian).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "JNJ", "XOM", "PG", "KO", "WMT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2016-01-01",
            end_date="2025-12-31",
        )

        toolkit.factors.get_factor_rank_autocorrelation("get_price_to_earnings_ratio").tail()
        ```

        Which returns:
        | Date    |   Rank Autocorrelation |
        |:--------|-----------------------:|
        | 2025-08 |                 0.993  |
        | 2025-09 |                 0.972  |
        | 2025-10 |                 0.958  |
        | 2025-11 |                 0.965  |
        | 2025-12 |                 0.8881 |
        """
        sampled_factor, _ = self._get_factor_and_prices(factor, period)

        result = (
            factors_model.get_factor_rank_autocorrelation(sampled_factor, periods_apart)
            .dropna()
            .to_frame("Rank Autocorrelation")
        )
        result.index.name = "Date"

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @staticmethod
    def _get_horizons(period: str, horizons: list[int] | int | None) -> list[int]:
        """
        Returns the forward return horizons as a list of positive numbers of periods.

        Args:
            period (str): The period frequency, which sets the default horizons.
            horizons (list[int] | int | None): The horizons given, if any.

        Returns:
            list[int]: The horizons.
        """
        if horizons is None:
            if period not in DEFAULT_HORIZONS:
                raise ValueError(
                    "Period must be daily, weekly, monthly, quarterly or yearly."
                )
            return DEFAULT_HORIZONS[period]

        horizon_list = [horizons] if isinstance(horizons, int) else list(horizons)

        if not horizon_list or any(int(horizon) < 1 for horizon in horizon_list):
            raise ValueError("Horizons must be whole numbers of periods of at least 1.")

        return [int(horizon) for horizon in horizon_list]
