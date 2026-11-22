"""Distribution Risk Module"""

__docformat__ = "google"

from typing import Literal

import numpy as np
import pandas as pd

from financetoolkit.helpers import handle_portfolio
from financetoolkit.risk import risk_model
from financetoolkit.risk.distribution import dispersion_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import (
    PERIODS_PER_YEAR,
    apply_rounding,
    finalize_dataset,
)

# ruff: noqa: E501


@filter_shown_columns
@track_sources
class DistributionRisk:
    """
    The shape of the return distribution: variance, skewness, kurtosis, central and partial moments, L-moments, the Gini mean difference, the tail Gini, range measures, the worst realization, the Hill estimator, the tail ratio, the mean absolute deviation, the coefficient of variation, autocorrelation, the Hurst exponent and the significance of a move.
    """

    @handle_portfolio
    @handle_errors
    def get_tail_ratio(
        self,
        period: str | None = None,
        alpha: float = 0.05,
        within_period: bool = False,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Tail Ratio of an investment portfolio or asset's returns.

        The Tail Ratio compares the size of the right (gain) tail to the left (loss) tail of the
        return distribution. A Tail Ratio above 1 indicates that best-case gains outsize worst-case
        losses.

        The formula is as follows (with the default alpha=0.05):

        - Tail Ratio = |Quantile(Returns, 1 - alpha)| / |Quantile(Returns, alpha)|

        The quantiles interpolate linearly, and a left-tail quantile of 0 gives NaN.

        Also known as: gain-to-pain tail ratio.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            alpha (float, optional): The percentile used to define each tail (e.g., 0.05 uses the 5th and
            95th percentile). Defaults to 0.05.
            within_period (bool, optional): Whether to calculate the Tail Ratio within the specified period or
            for the entire period. Thus whether to look at the Tail Ratio within a specific year (if period =
            'yearly') or look at the entirety of all years. Defaults to False.
            Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rolling (int, optional): The rolling window size to use for the calculation. If set, the Tail
            Ratio is calculated over a rolling window of this many periods across the full return history
            instead of per `period`. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Tail Ratio values over time.
            Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: Tail Ratio values with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates the Tail
        Ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of Tail Ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["QQQ", "TLT", "GLD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_tail_ratio(period="yearly", within_period=True)
        ```

        Which returns:

        |      |    QQQ |    TLT |    GLD |   Benchmark |
        |:-----|-------:|-------:|-------:|------------:|
        | 2021 | 0.9646 | 1.0836 | 0.7734 |      1.1304 |
        | 2022 | 0.9109 | 0.8904 | 1.0385 |      0.9524 |
        | 2023 | 1.2227 | 1.1695 | 1.1281 |      1.0466 |
        | 2024 | 0.9029 | 0.8128 | 1.072  |      0.9091 |
        | 2025 | 0.8622 | 1.104  | 1.2495 |      0.93   |
        """
        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if (
            period == "daily"
            and within_period
            and self._historical_data["intraday"].empty
        ):
            raise ValueError(
                "Intraday data is required for within-period daily calculations. Either set "
                "within_period=False or initialise the Toolkit with an intraday_period."
            )
        if growth and not within_period and not rolling:
            raise ValueError(
                "Growth calculations need one value per period, while "
                "within_period=False produces a single value for the entire period. "
                "Set within_period=True to calculate growth."
            )

        if rolling:
            returns = self._historical_data[period]["Return"]
            tail_ratio = risk_model.get_rolling_tail_ratio(returns, alpha, rolling)
        else:
            returns = self._get_returns(period, within_period)

            tail_ratio = risk_model.get_tail_ratio(returns, alpha)

        if rolling or within_period:
            tail_ratio = tail_ratio.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=tail_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            apply_slice=False,
        )

    @handle_portfolio
    @handle_errors
    def get_skewness(
        self,
        period: str | None = None,
        within_period: bool = False,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Skewness of an investment portfolio or asset's returns.

        Skewness is a statistical measure used in finance to assess the asymmetry in the distribution of
        returns for an investment portfolio or asset over a defined period. It offers valuable insights
        into the shape of the return distribution, indicating whether returns are skewed towards the
        positive or negative side of the mean. Skewness is a crucial tool for investors and analysts seeking
        to understand the potential risk and return characteristics of an investment, aiding in the assessment
        of the distribution's tails and potential outliers. It provides a means to gauge the level of
        skew in returns, enabling more informed investment decisions and risk management strategies.

        The formula is as follows:

        - Skewness = n / ((n - 1) * (n - 2)) * Σ ((Return - Mean) / Std)^3

        Where n is the number of returns and Std the sample standard deviation (ddof=1), which is the
        bias-adjusted sample skewness (G1) of pandas and Excel's SKEW.

        Also known as: return distribution asymmetry, tail skew.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            within_period (bool, optional): Whether to calculate the Skewness within the specified period or for the
            entire period. Thus whether to look at the Skewness within a specific year (if period = 'yearly') or look
            at the entirety of all years. Defaults to False.
            Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rolling (int, optional): The rolling window size to use for the calculation. If set, Skewness is
            calculated over a rolling window of this many periods across the full return history instead of
            per `period`. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Skewness values over time.
            Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: Skewness values with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates Skew for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of Skewness values using the specified `lag`.
        - Returns that do not vary (a constant price, such as a suspended stock) have no Skewness and give NaN.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NFLX", "DIS", "WBD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_skewness(period="yearly", within_period=True)
        ```

        Which returns:

        |      |    NFLX |     DIS |     WBD |   Benchmark |
        |:-----|--------:|--------:|--------:|------------:|
        | 2021 |  1.8966 |  0.2848 | -2.1536 |     -0.3334 |
        | 2022 | -2.1029 | -0.5086 | -0.0126 |      0.0454 |
        | 2023 |  1.5636 | -0.2155 | -0.42   |     -0.0206 |
        | 2024 |  0.872  |  0.7012 |  0.6324 |     -0.5242 |
        | 2025 | -0.2855 |  0.6672 |  1.9711 |      1.5069 |
        """
        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if (
            period == "daily"
            and within_period
            and self._historical_data["intraday"].empty
        ):
            raise ValueError(
                "Intraday data is required for within-period daily calculations. Either set "
                "within_period=False or initialise the Toolkit with an intraday_period."
            )
        if growth and not within_period and not rolling:
            raise ValueError(
                "Growth calculations need one value per period, while "
                "within_period=False produces a single value for the entire period. "
                "Set within_period=True to calculate growth."
            )

        if rolling:
            returns = self._historical_data[period]["Return"]
            skewness = risk_model.get_rolling_skewness(returns, rolling)
        else:
            returns = self._get_returns(period, within_period)

            skewness = risk_model.get_skewness(returns)

        if rolling or within_period:
            skewness = skewness.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=skewness,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            apply_slice=False,
        )

    @handle_portfolio
    @handle_errors
    def get_kurtosis(
        self,
        period: str | None = None,
        within_period: bool = False,
        fisher: bool = False,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Kurtosis of an investment portfolio or asset's returns.

        Kurtosis is a statistical measure used in finance to evaluate the shape of the probability
        distribution of returns for an investment portfolio or asset over a defined time period.
        It assesses the "tailedness" of the return distribution, indicating whether returns have
        fatter or thinner tails compared to a normal distribution. Kurtosis plays a critical role
        in risk assessment by revealing the potential presence of extreme outliers or the likelihood
        of heavy tails in the return data. This information aids investors and analysts in
        understanding the degree of risk associated with an investment and assists in making
        more informed decisions regarding risk tolerance. In essence, kurtosis serves as a valuable
        tool for comprehending the distribution characteristics of returns, offering insights
        into the potential for rare but significant events in the financial markets.

        By default this is Pearson's kurtosis, which is 3 for a normal distribution; with `fisher=True`
        it is the excess kurtosis, which is 0. Daily stock returns typically show an excess kurtosis of
        several points, and within a single year one exceptional day (an earnings shock, a market-wide
        gap) can push it above 10, so a high value often reflects one event rather than a persistently
        fat-tailed distribution.

        The formula is as follows (with the default fisher=False):

        - Kurtosis = Excess Kurtosis + 3
        - Excess Kurtosis = n(n + 1) / ((n - 1)(n - 2)(n - 3)) * Σ z^4 - 3(n - 1)^2 / ((n - 2)(n - 3))
        - z = (Return - Mean) / Std

        Where n is the number of returns and Std the sample standard deviation (ddof=1), which is the
        bias-adjusted sample excess kurtosis (G2) of pandas and Excel's KURT. With fisher=True the Excess
        Kurtosis is returned.

        Also known as: tail heaviness, fat tails, leptokurtosis.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            within_period (bool, optional): Whether to calculate the Kurtosis within the specified period or for the
            entire period. Thus whether to look at the Kurtosis within a specific year (if period = 'yearly') or look
            at the entirety of all years. Defaults to False.
            Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            fisher (bool, optional): Whether to use Fisher's definition of kurtosis (kurtosis = 0.0
            for a normal distribution). Defaults to False.
            rolling (int, optional): The rolling window size to use for the calculation. If set, Kurtosis is
            calculated over a rolling window of this many periods across the full return history instead of
            per `period`. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Kurtosis values over time.
            Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: Kurtosis values with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates Kurtosis for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of Kurtosis values using the specified `lag`.
        - Returns that do not vary (a constant price, such as a suspended stock) have no Kurtosis and give NaN.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["BTCUSD", "ETHUSD", "SOLUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_kurtosis(period="yearly", within_period=True, fisher=True)
        ```

        Which returns:

        |      |   BTCUSD |   ETHUSD |   SOLUSD |   Benchmark |
        |:-----|---------:|---------:|---------:|------------:|
        | 2021 |   1.6215 |   3.5555 |   2.46   |      2.5374 |
        | 2022 |   4.0666 |   2.545  |   6.8965 |      2.359  |
        | 2023 |   3.2756 |   2.4865 |   5.3817 |      1.416  |
        | 2024 |   1.9522 |   4.0366 |   0.2918 |      3.8134 |
        | 2025 |   2.3709 |   4.0428 |   4.1957 |     45.0722 |
        """
        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if (
            period == "daily"
            and within_period
            and self._historical_data["intraday"].empty
        ):
            raise ValueError(
                "Intraday data is required for within-period daily calculations. Either set "
                "within_period=False or initialise the Toolkit with an intraday_period."
            )
        if growth and not within_period and not rolling:
            raise ValueError(
                "Growth calculations need one value per period, while "
                "within_period=False produces a single value for the entire period. "
                "Set within_period=True to calculate growth."
            )

        if rolling:
            returns = self._historical_data[period]["Return"]
            kurtosis = risk_model.get_rolling_kurtosis(returns, rolling, fisher=fisher)
        else:
            returns = self._get_returns(period, within_period)

            kurtosis = risk_model.get_kurtosis(returns, fisher=fisher)

        if rolling or within_period:
            kurtosis = kurtosis.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=kurtosis,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            apply_slice=False,
        )

    @handle_portfolio
    @handle_errors
    def get_hill_estimator(
        self,
        period: str | None = None,
        within_period: bool = False,
        k: int | float = 0.1,
        tail: str = "left",
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Hill Estimator of the tail index of returns, per asset.

        Unlike the (finite-sample) Skewness and Kurtosis above, the Hill Estimator is
        a semi-parametric estimate of how heavy the tail of the return distribution
        actually is, under the assumption that the tail follows a Pareto-type power
        law. Smaller values of the tail index indicate a heavier tail (more extreme
        outliers are likely) -- as a rule of thumb, a tail index below 4 implies the
        Kurtosis is theoretically infinite, and below 2 implies the Variance itself is
        theoretically infinite.

        For daily stock returns the tail index typically lies between about 2 and 5, while
        a normal distribution has an infinite tail index. The estimate depends on `k`:
        practitioners plot it against a range of `k` (a Hill plot) and read it where it is
        stable, since a small `k` is noisy and a large `k` pulls in the center of the
        distribution.

        The formula is as follows (with the default tail="left" and k=0.1):

        - Hill Shape (xi) = (1 / k) * Σ (ln X_(i) - ln X_(k+1)) for i = 1, ..., k
        - Hill Tail Index = 1 / xi
        - Standard Error = Hill Tail Index / √k

        Where X_(1) >= X_(2) >= ... are the strictly positive losses (the negated returns, or the returns
        themselves for tail="right") sorted from large to small, and k is the number of order statistics used:
        int(0.1 * the number of positive losses) by default, or `k` itself when it is an integer.

        For more information about the method, see the following paper:

        - Hill, B.M. (1975). "A Simple General Approach to Inference About the Tail of
        a Distribution." The Annals of Statistics, 3(5), 1163-1174.

        Also known as: Hill tail index estimator, Hill's estimator.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            within_period (bool, optional): Whether to calculate the estimator within the specified period
            or for the entire period. Defaults to False.
            Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            k (int | float, optional): The number of upper order statistics to use. If a float in (0, 1)
            it is interpreted as the fraction of the strictly positive observations to use. A period
            without such observations reports NaN with 0 observations used. Defaults to 0.1 (the top 10%).
            tail (str, optional): Which tail to estimate, one of "left" (the loss tail) or "right" (the
            gain tail). Defaults to "left".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Hill tail index, the Hill shape parameter, its Standard Error and the
            number of order statistics used, per asset.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates the
        Hill Estimator for each asset in the Toolkit instance.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["BTCUSD", "^GSPC", "GCUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_hill_estimator(k=60)
        ```

        Which returns:

        |                       |   BTCUSD |   ^GSPC |   GCUSD |   Benchmark |
        |:----------------------|---------:|--------:|--------:|------------:|
        | Hill Tail Index       |   3.4392 |  2.9068 |  3.0565 |      2.7613 |
        | Hill Shape (xi)       |   0.2908 |  0.344  |  0.3272 |      0.3621 |
        | Standard Error        |   0.444  |  0.3753 |  0.3946 |      0.3565 |
        | Observations Used (k) |  60      | 60      | 60      |     60      |
        """
        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if (
            period == "daily"
            and within_period
            and self._historical_data["intraday"].empty
        ):
            raise ValueError(
                "Intraday data is required for within-period daily calculations. Either set "
                "within_period=False or initialise the Toolkit with an intraday_period."
            )

        returns = self._get_returns(period, within_period)

        result = risk_model.get_hill_estimator(returns, k=k, tail=tail)

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_variance(
        self,
        period: str | None = None,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Variance of an investment portfolio or asset's returns for a
        given period based on the daily historical returns.

        Variance measures the spread or dispersion of returns around the mean. A higher
        Variance indicates more variability in the returns, while a lower Variance suggests
        that the returns are closer to the mean.

        The formula is as follows:

        - Variance = Var(Daily Returns within the period) * Trading Days per Period

        Where Var is the sample variance (ddof=1) and the trading days per period are 252 / 52 (weekly),
        252 / 12 (monthly), 252 / 4 (quarterly) or 252 (yearly). A ticker with returns on weekends, such as a
        cryptocurrency, has a return on every calendar day and uses 365 days a year instead (365 / 52 weekly).
        With `rolling`, the result is the sample variance of the last `rolling` returns of the chosen period
        itself, without scaling.

        Also known as: dispersion, spread.

        Args:
            period (str, optional): The data frequency for returns (weekly, monthly, quarterly, or yearly). Defaults
                to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
            rolling (int, optional): The rolling window size to use for the calculation. If set,
            Variance is calculated over a rolling window of this many periods (e.g. period='monthly'
            and rolling=6 gives the rolling 6-month Variance) instead of one value per `period`.
            Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Variance values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: Variance values with time as the index.

        Notes:
        - The method retrieves the daily historical return data and calculates the Variance for
        the specified `period` for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of Variance values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["WMT", "TGT", "COST"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_variance(period="yearly")
        ```

        Which returns:

        | Date   |    WMT |    TGT |   COST |   Benchmark |
        |:-------|-------:|-------:|-------:|------------:|
        | 2021   | 0.0278 | 0.0559 | 0.0359 |      0.0167 |
        | 2022   | 0.0714 | 0.2131 | 0.0984 |      0.0588 |
        | 2023   | 0.0245 | 0.0965 | 0.0365 |      0.0171 |
        | 2024   | 0.0313 | 0.1377 | 0.0355 |      0.0158 |
        | 2025   | 0.063  | 0.1166 | 0.0467 |      0.038  |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError("Period must be weekly, monthly, quarterly, or yearly.")

        if rolling:
            period_returns = self._historical_data[period]["Return"]
            variance = risk_model.get_rolling_variance(period_returns, period, rolling)
        else:
            returns = self._historical_data["daily"]["Return"]
            variance = risk_model.get_variance(returns, period)

        variance = variance.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=variance,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            apply_slice=False,
        )

    @handle_portfolio
    @handle_errors
    def get_mean_absolute_deviation(
        self,
        period: str | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Mean Absolute Deviation (MAD) of an investment portfolio or asset's
        returns for a given period based on the daily historical returns.

        MAD measures the average absolute distance of each return from the mean return. Unlike
        Variance and Volatility, it does not square the deviations, making it less sensitive to
        outliers.

        The formula is as follows:

        - Mean Absolute Deviation = Σ |Return_t - Mean| / n

        Where the returns are the n daily returns within each period, without scaling to the period length.

        Args:
            period (str, optional): The data frequency for returns (weekly, monthly, quarterly, or yearly). Defaults
                to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the MAD values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: Mean Absolute Deviation values with time as the index.

        Notes:
        - The method retrieves the daily historical return data and calculates the MAD for
        the specified `period` for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of MAD values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["KO", "PEP", "MDLZ"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_mean_absolute_deviation(period="yearly")
        ```

        Which returns:

        | Date   |     KO |    PEP |   MDLZ |   Benchmark |
        |:-------|-------:|-------:|-------:|------------:|
        | 2021   | 0.007  | 0.0069 | 0.0073 |      0.0061 |
        | 2022   | 0.0091 | 0.009  | 0.0098 |      0.0119 |
        | 2023   | 0.0063 | 0.007  | 0.008  |      0.0064 |
        | 2024   | 0.0061 | 0.0076 | 0.0082 |      0.0058 |
        | 2025   | 0.0083 | 0.0104 | 0.0104 |      0.0074 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError("Period must be weekly, monthly, quarterly, or yearly.")

        returns = self._historical_data["daily"]["Return"]
        mean_absolute_deviation = risk_model.get_mean_absolute_deviation(
            returns, period
        )

        mean_absolute_deviation = mean_absolute_deviation.loc[
            self._start_date : self._end_date
        ]
        mean_absolute_deviation = mean_absolute_deviation.dropna(how="all", axis=0)

        return finalize_dataset(
            dataset=mean_absolute_deviation,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            apply_slice=False,
        )

    @handle_portfolio
    @handle_errors
    def get_coefficient_of_variation(
        self,
        period: str | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Coefficient of Variation (CV) of an investment portfolio or asset's
        returns for a given period based on the daily historical returns.

        The Coefficient of Variation is the ratio of the standard deviation to the mean of
        returns, which normalizes dispersion relative to the average return. This makes it
        useful for comparing the relative volatility of assets with different average returns,
        which a raw standard deviation cannot do.

        The Coefficient of Variation is only meaningful when the mean return is clearly positive. In a
        period with an average daily return close to zero it explodes, and in a down period it turns
        negative, so for returns it is best read over periods with a positive mean, or replaced by a
        Sharpe ratio, which is its inverse with the risk-free rate subtracted.

        The formula is as follows:

        - Coefficient of Variation = Std(Daily Returns) / Mean(Daily Returns)

        Where both are taken over the daily returns within each period, with Std the sample standard deviation
        (ddof=1), without scaling to the period length.

        Also known as: relative standard deviation.

        Args:
            period (str, optional): The data frequency for returns (weekly, monthly, quarterly, or yearly). Defaults
                to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the CV values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: Coefficient of Variation values with time as the index.

        Notes:
        - The method retrieves the daily historical return data and calculates the CV for
        the specified `period` for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of CV values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["MSFT", "ORCL", "IBM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2023-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_coefficient_of_variation(period="yearly")
        ```

        Which returns:

        | Date   |    MSFT |    ORCL |     IBM |   Benchmark |
        |:-------|--------:|--------:|--------:|------------:|
        | 2023   |  8.0237 | 16.2298 | 12.0942 |      8.3745 |
        | 2024   | 22.4294 |  9.8256 | 10.3503 |      8.6564 |
        | 2025   | 21.9925 | 28.2432 | 12.934  |     16.8499 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError("Period must be weekly, monthly, quarterly, or yearly.")

        returns = self._historical_data["daily"]["Return"]
        coefficient_of_variation = risk_model.get_coefficient_of_variation(
            returns, period
        )

        coefficient_of_variation = coefficient_of_variation.loc[
            self._start_date : self._end_date
        ]
        coefficient_of_variation = coefficient_of_variation.dropna(how="all", axis=0)

        return finalize_dataset(
            dataset=coefficient_of_variation,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            apply_slice=False,
        )

    @handle_portfolio
    @handle_errors
    def get_move_significance(
        self,
        period: str = "daily",
        window: int | None = None,
        alpha: float = 0.05,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate how unusual each move of an investment portfolio or asset is: the return in
        units of its trailing volatility and of its trailing Value at Risk, its percentile among
        the earlier moves and the date of the last move at least as large in the same direction.

        A return only says how far a price moved, not whether that is a lot for the asset: a 3%
        day is ordinary for a volatile growth stock and exceptional for a utility or an index.
        Dividing the return by the volatility of the returns before it (the "sigma" of a move)
        makes moves comparable across assets and calm or turbulent markets, the percentile says
        how many earlier moves were smaller, and "largest since" gives the familiar headline
        ("the biggest one-day fall since March 2020"). The VaR Multiple measures the move against
        the historic Value at Risk of the same earlier returns instead of their standard deviation,
        so it does not assume normal returns: a fall with a VaR Multiple below -1 is a VaR
        exceedance, worse than the loss the trailing VaR expected on all but a share alpha of days. When the
        trailing quantile is not a loss (every earlier return was a gain or nearly so), the VaR Multiple is NaN.
        Ties count half, as in `get_level_significance`.

        The formula is as follows:

        - Volatility Multiple(t) = Return(t) / σ(Return(t - window), ..., Return(t - 1))
        - VaR Multiple(t) = Return(t) / |Quantile(Return(t - window), ..., Return(t - 1); alpha)|
        - Percentile(t) = (Count(|Return(s)| < |Return(t)|, s < t) + 0.5 · Count(|Return(s)| = |Return(t)|, s < t)) / Count(s < t)
        - Largest Since(t) = max(s < t : Return(s) <= Return(t)) if Return(t) < 0
        - Largest Since(t) = max(s < t : Return(s) >= Return(t)) if Return(t) > 0

        The volatility σ uses ddof=1 and the quantile is the historic Value at Risk, interpolated linearly as
        in `get_value_at_risk`.

        Each measure only uses the returns before the period itself, so a move is never part of
        the volatility or the history it is measured against (no look-ahead). Largest Since
        compares falls with falls and rises with rises, as a headline does: a 6% fall is the
        largest fall since the last fall of 6% or more, whatever the rises in between.

        Also known as: sigma move, standardized return, biggest move since, return z-score, worst day
        since, VaR exceedance, VaR breach ratio.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly,
                quarterly, or yearly). Defaults to "daily".
            window (int | None, optional): The number of earlier returns the volatility is
                measured over, which the percentile also needs at least. Defaults to None, which
                uses a year of observations with a minimum of 20 (252 daily, 52 weekly and 20
                monthly, quarterly or yearly returns).
            alpha (float, optional): The tail probability of the trailing historic Value at Risk
                the VaR Multiple divides by (0.05 for the 95% VaR). Defaults to 0.05.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Return, Volatility Multiple, VaR Multiple, Percentile and Largest Since
            (the date as text) per asset, with time as the index and the measures and tickers as the column
            levels.

        Notes:
        - The returns are total returns (Adj Close), compounded within each period for a period
          other than daily.
        - `get_value_at_risk` reports the VaR as a return (negative for a loss); the VaR Multiple
          divides by its size, so its sign is the move's, like the Volatility Multiple.
        - The history a move is compared with starts at the Toolkit's start_date: the volatility,
          the VaR and the percentile are NaN until `window` returns are available, and Largest Since is
          NaN when no earlier move in the data was as large, which makes the move the largest
          since the start_date. Choose an early start_date for a long history.
        - Each ticker uses its own history, so a ticker that listed later starts later.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NVDA", "KO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2015-01-01",
            end_date="2025-12-31",
        )

        significance = toolkit.risk.get_move_significance()

        significance.loc["2025-01-27"].unstack(level=0)
        ```

        Which returns:

        |           |   Return |   Volatility Multiple |   VaR Multiple |   Percentile | Largest Since   |
        |:----------|---------:|----------------------:|---------------:|-------------:|:----------------|
        | NVDA      |  -0.1697 |               -5.0682 |        -3.4613 |       0.9976 | 2020-03-16      |
        | KO        |   0.0315 |                3.8515 |         2.3126 |       0.981  | 2022-11-10      |
        | Benchmark |  -0.0141 |               -1.76   |        -1.026  |       0.8632 | 2025-01-10      |
        """
        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )

        window = window if window else max(PERIODS_PER_YEAR[period], 20)

        if window < 2:  # noqa: PLR2004
            raise ValueError(
                "The window needs at least 2 returns to measure a volatility."
            )
        if not 0 < alpha < 1:
            raise ValueError(
                "alpha must be between 0 and 1, e.g. 0.05 for the 95% VaR."
            )

        returns = self._historical_data[period]["Return"]

        move_significance = pd.concat(
            {
                ticker: risk_model.get_move_significance(returns[ticker], window, alpha)
                for ticker in returns.columns
            },
            axis=1,
        ).swaplevel(1, 0, axis=1)

        # The measures in reading order, each with the tickers in the Toolkit's order.
        move_significance = move_significance[
            pd.MultiIndex.from_product(
                [
                    [
                        "Return",
                        "Volatility Multiple",
                        "VaR Multiple",
                        "Percentile",
                        "Largest Since",
                    ],
                    returns.columns,
                ]
            )
        ]

        move_significance = move_significance.loc[
            self._start_date : self._end_date
        ].dropna(how="all", axis=0)

        return finalize_dataset(
            dataset=move_significance,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rounding=rounding,
            axis="rows",
            apply_slice=False,
        )

    @handle_errors
    def get_autocorrelation(
        self,
        lags: int = 10,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Autocorrelation Function (ACF) of each asset's daily returns for a
        range of lags.

        The ACF measures the correlation between a return series and a lagged version of
        itself. A significant ACF at a given lag indicates that returns are not fully
        independent over time, which is relevant for assessing return predictability and
        volatility clustering (as opposed to a trading-signal use case, which is why this
        lives in the Risk module rather than Technicals).

        The formula is as follows:

        - ACF(k) = Σ (r_t - Mean) * (r_(t+k) - Mean) / Σ (r_t - Mean)^2

        Where r_t are the n daily returns (missing values dropped), the numerator runs over t = 1, ..., n - k,
        the denominator over all n returns, and k = 1, ..., `lags` (10 by default). A lag of n or more has
        no pairs of returns and is NaN.

        Args:
            lags (int, optional): The number of lags to calculate the ACF for. Defaults to 10.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The ACF value for each lag (rows) and each asset (columns).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["SPY", "QQQ", "IWM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
            benchmark_ticker=None,
        )

        toolkit.risk.get_autocorrelation()
        ```

        Which returns:

        |   Lag |     SPY |     QQQ |     IWM |
        |------:|--------:|--------:|--------:|
        |     1 | -0.031  | -0.0387 | -0.0092 |
        |     2 | -0.0069 | -0.001  | -0.0254 |
        |     3 | -0.0626 | -0.0629 | -0.0055 |
        |     4 | -0.0344 | -0.0216 | -0.0437 |
        |     5 | -0.0075 | -0.0071 | -0.016  |
        |     6 | -0.0104 |  0      | -0.0112 |
        |     7 | -0.0109 |  0.0031 | -0.013  |
        |     8 | -0.0063 | -0.0118 | -0.0168 |
        |     9 |  0.0399 |  0.0505 |  0.0285 |
        |    10 | -0.0111 | -0.0214 | -0.0059 |
        """
        returns = self._historical_data["daily"]["Return"]

        autocorrelation = returns.apply(
            lambda column: risk_model.get_autocorrelation(column, lags=lags)
        )
        autocorrelation.index.name = "Lag"

        return apply_rounding(
            autocorrelation, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_hurst_exponent(
        self,
        max_lag: int = 20,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Hurst Exponent of each asset's daily (adjusted close) log price series, a
        measure of long-term memory that indicates whether the price is mean-reverting,
        trending, or a random walk.

        The Hurst Exponent (H) is interpreted as follows:

        - H < 0.5: the series is mean-reverting (anti-persistent).
        - H = 0.5: the series is a random walk (no memory).
        - H > 0.5: the series is trending (persistent).

        For liquid stocks the estimate is usually close to 0.5, and with lags of at most
        `max_lag` days it describes short-term behaviour only; a value of 0.45 or 0.55 over a
        few years is within estimation noise rather than evidence of a tradable pattern.

        The formula is as follows (with the default max_lag=20):

        - sigma(tau) = Std(ln P_(t+tau) - ln P_t)
        - ln sigma(tau) = c + H * ln tau, fitted by least squares over tau = 2, ..., 20

        Where P is the daily adjusted close, Std the population standard deviation (ddof=0) of the lagged log
        price differences and H, the slope, the Hurst Exponent.

        Args:
            max_lag (int, optional): The maximum lag, included, to use when estimating the exponent:
                the regression runs over the lags 2 up to and including `max_lag`. Defaults to 20.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: The estimated Hurst Exponent for each asset.

        Notes:
        - The exponent is estimated with the generalized Hurst (structure function) method of
        Di Matteo, Aste & Dacorogna (2005) on the logarithm of the adjusted close, as in that paper,
        see `risk_model.get_hurst_exponent`. The lagged differences of log prices are log returns,
        so the estimate does not depend on the price level.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["EURUSD", "GCUSD", "BTCUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_hurst_exponent()
        ```

        Which returns:

        |           |   Value |
        |:----------|--------:|
        | EURUSD    |  0.4229 |
        | GCUSD     |  0.4886 |
        | BTCUSD    |  0.5057 |
        | Benchmark |  0.4621 |
        """
        # The estimator regresses the dispersion of lagged differences on the lag, which only identifies self-affinity on a level series. Feeding it returns, which are already differenced, collapses the exponent towards zero.  # noqa: E501
        # Log prices, as in Di Matteo et al. (2005): their lagged differences are log returns over the lag, whose dispersion does not depend on the price level, whereas price differences grow with the price and tilt the slope on a series that trends over the sample.  # noqa: E501
        log_prices = np.log(self._get_prices("Adj Close"))

        hurst_exponent = log_prices.apply(
            lambda column: risk_model.get_hurst_exponent(column, max_lag=max_lag)
        )

        return apply_rounding(
            hurst_exponent, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_partial_moment(
        self,
        period: str | None = None,
        order: float = 1,
        minimum_acceptable_return: float | str = 0.0,
        side: str = "lower",
        root: bool = False,
        within_period: bool = False,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Lower (or Upper) Partial Moment of an investment portfolio or asset's returns.

        A partial moment measures only one side of the return distribution: the lower partial moment
        averages the shortfalls below a minimum acceptable return (MAR) raised to a power, the order, and
        the upper partial moment does the same for the returns above it. The order sets what is measured:
        order 0 is how often the returns fall below the MAR (the downside frequency), order 1 how far they
        fall below it on average (the downside potential, the expected shortfall below the target), order 2
        the downside variance behind the Sortino Ratio and higher orders the weight of the extreme losses.
        Investors who care about losses rather than about deviations in either direction use them in place
        of the variance, and the ratio of an upper to a lower partial moment is the Omega Ratio (order 1)
        and the Upside Potential Ratio's building block.

        The formula is as follows (with the default order=1, side="lower" and minimum_acceptable_return=0):

        - Lower Partial Moment = Σ max(MAR - Return_t, 0)^order / n
        - Upper Partial Moment = Σ max(Return_t - MAR, 0)^order / n

        Where n is the number of all returns, so the returns on the other side of the MAR count as zero
        rather than being left out. With order=0 the moment counts the returns strictly below (or above)
        the MAR, as a share of all returns. With root=True the moment is raised to the power 1 / order,
        which turns order 2 into the Downside Deviation (side="lower") or the Upside Deviation
        (side="upper"). With minimum_acceptable_return="mean" the MAR is the mean of each sample, which
        gives the semi-moments around the mean (order 2: the semi-variance with n in the denominator).

        For more information, see the following sources:

        - Bawa, V.S. (1975). "Optimal Rules for Ordering Uncertain Prospects." Journal of Financial
        Economics, 2(1), 95-121.
        - Fishburn, P.C. (1977). "Mean-Risk Analysis with Risk Associated with Below-Target Returns."
        The American Economic Review, 67(2), 116-126.
        - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution." John Wiley &
        Sons, chapter 4 (Downside Potential, Downside and Upside Frequency, Upside Potential).
        - Riskfolio-Lib: RiskFunctions.LPM (https://github.com/dcajasn/Riskfolio-Lib).
        - skfolio: measures.first_lower_partial_moment and measures.fourth_lower_partial_moment
        (https://skfolio.org/api.html#module-skfolio.measures).
        - PerformanceAnalytics (R): DownsidePotential, DownsideFrequency, UpsideFrequency, UpsideRisk and
        lpm (https://github.com/braverock/PerformanceAnalytics).

        Also known as: LPM, UPM, downside potential, downside frequency, upside frequency, upside potential,
        shortfall probability, semi-moment.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            order (float, optional): The order of the moment, 0 or higher: 0 for the frequency, 1 for the
            potential, 2 for the variance and higher orders for the extremes. Defaults to 1.
            minimum_acceptable_return (float | str, optional): The minimum acceptable return (MAR) per
            observation of the chosen `period` (e.g. per day), or "mean" for the mean of each sample.
            Defaults to 0.0.
            side (str, optional): "lower" for the returns below the MAR, "upper" for those above it.
            Defaults to "lower".
            root (bool, optional): Whether to raise the moment to the power 1 / order, which needs an
            order above 0. Defaults to False.
            within_period (bool, optional): Whether to calculate the Partial Moment within the specified period or
            for the entire period. Thus whether to look at the Partial Moment within a specific year (if period =
            'yearly') or look at the entirety of all years. Defaults to False.
            Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rolling (int, optional): The rolling window size to use for the calculation. If set, the Partial Moment
            is calculated over a rolling window of this many periods across the full return history
            instead of per `period`. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Partial Moment values over time.
            Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series | pd.DataFrame: Partial Moment values, per ticker, or per period (or date) and ticker.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates the
        Partial Moment for each asset in the Toolkit instance, each on its own returns (a ticker that listed later
        or misses a date another ticker traded is measured on the returns it has).
        - If `growth` is set to True, the method calculates the growth of the Partial Moment values using the
        specified `lag`.
        - The lower partial moment of order 1 agrees with PerformanceAnalytics' DownsidePotential and
        Riskfolio-Lib's LPM(p=1), order 0 equals DownsideFrequency (UpsideFrequency with side="upper") and
        side="upper" with order 1 and 2 equals UpsideRisk(stat="potential") and UpsideRisk(stat="variance").
        - With minimum_acceptable_return="mean" order 1 and 4 agree with skfolio's first_lower_partial_moment
        and fourth_lower_partial_moment. Riskfolio-Lib's LPM(p=2) divides by n - 1 instead of n and
        takes the root, and PerformanceAnalytics' lpm divides by the number of returns below the
        threshold. Riskfolio-Lib's SemiKurtosis is the square of order 4 with root=True and
        minimum_acceptable_return="mean", and EvenSemiMoment(p) the (order 2p) moment to the power 1 / p.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["JPM", "GS", "MS"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_partial_moment(period="yearly", within_period=True)
        ```

        Which returns:

        |      |    JPM |     GS |     MS |   Benchmark |
        |:-----|-------:|-------:|-------:|------------:|
        | 2021 | 0.0048 | 0.0055 | 0.0057 |      0.0026 |
        | 2022 | 0.0074 | 0.0072 | 0.0081 |      0.0063 |
        | 2023 | 0.0041 | 0.0053 | 0.0056 |      0.0028 |
        | 2024 | 0.0041 | 0.0046 | 0.0048 |      0.0025 |
        | 2025 | 0.0048 | 0.0059 | 0.0057 |      0.0034 |
        """
        period = period if period else "daily"
        returns = self._get_measure_returns(period, within_period, rolling, growth)

        result = risk_model.get_partial_moment(
            returns,
            order=order,
            minimum_acceptable_return=minimum_acceptable_return,
            side=side,
            root=root,
            window_size=rolling,
        )

        return self._finalize_measure(
            result,
            bool(rolling or within_period),
            rounding,
            growth,
            lag,
            standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_central_moment(
        self,
        period: str | None = None,
        order: int = 3,
        within_period: bool = False,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate a Central Moment of an investment portfolio or asset's returns.

        A central moment is the average deviation from the mean return raised to a power, the order. The
        second central moment is the (population) variance, the third measures the asymmetry of the return
        distribution in squared-and-cubed return units (negative when the large deviations are losses) and
        the fourth the weight of the tails. Unlike the Skewness and Kurtosis, which divide by a power of the
        standard deviation, they keep the scale of the returns, so a volatile asset has larger moments, and
        they are the building blocks of the moment-based risk measures of portfolio optimization, such as
        the square-root kurtosis.

        The formula is as follows (with the default order=3):

        - Central Moment = Σ (Return_t - Mean)^order / n

        Where Mean is the average return and n the number of returns of the sample (the population moment,
        without a bias adjustment).

        For more information, see the following sources:

        - Cajas, D. (2021). "Riskfolio-Lib Documentation: Risk Functions." (the square-root kurtosis and
        even moments, https://riskfolio-lib.readthedocs.io/en/latest/risk.html).
        - skfolio: measures.third_central_moment and measures.fourth_central_moment
        (https://skfolio.org/api.html#module-skfolio.measures).
        - Riskfolio-Lib: RiskFunctions.Kurtosis and RiskFunctions.EvenMoment
        (https://github.com/dcajasn/Riskfolio-Lib).
        - PerformanceAnalytics (R): centeredmoment (https://github.com/braverock/PerformanceAnalytics).

        Also known as: moment about the mean, third central moment, fourth central moment, square-root
        kurtosis (its square root for order 4).

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            order (int, optional): The order of the moment, an integer of 2 or higher. Defaults to 3.
            within_period (bool, optional): Whether to calculate the Central Moment within the specified period or
            for the entire period. Thus whether to look at the Central Moment within a specific year (if period =
            'yearly') or look at the entirety of all years. Defaults to False.
            Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rolling (int, optional): The rolling window size to use for the calculation. If set, the Central Moment
            is calculated over a rolling window of this many periods across the full return history
            instead of per `period`. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Central Moment values over time.
            Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series | pd.DataFrame: Central Moment values, per ticker, or per period (or date) and ticker.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates the
        Central Moment for each asset in the Toolkit instance, each on its own returns (a ticker that listed later
        or misses a date another ticker traded is measured on the returns it has).
        - If `growth` is set to True, the method calculates the growth of the Central Moment values using the
        specified `lag`.
        - Order 3 and 4 agree with skfolio's third_central_moment and fourth_central_moment and
        PerformanceAnalytics' centeredmoment. Riskfolio-Lib's Kurtosis is the square root of order 4 and
        its EvenMoment(p) the (order 2p) moment to the power 1 / p.
        - The moments of daily returns are small numbers (order 4 of a stock is about 1e-7), so use the
        rounding parameter to show them.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["XOM", "CVX", "COP"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_central_moment(period="yearly", within_period=True, order=4, rounding=10)
        ```

        Which returns:

        |      |       XOM |       CVX |        COP |   Benchmark |
        |:-----|----------:|----------:|-----------:|------------:|
        | 2021 | 3.194e-07 | 1.706e-07 | 7.526e-07  |   1.61e-08  |
        | 2022 | 8.624e-07 | 8.274e-07 | 2.0632e-06 |   1.797e-07 |
        | 2023 | 2.417e-07 | 2.375e-07 | 6.268e-07  |   1.29e-08  |
        | 2024 | 6.71e-08  | 8.64e-08  | 2.004e-07  |   1.85e-08  |
        | 2025 | 2.85e-07  | 5.392e-07 | 1.8409e-06 |   5.877e-07 |
        """
        period = period if period else "daily"
        returns = self._get_measure_returns(period, within_period, rolling, growth)

        result = dispersion_model.get_central_moment(
            returns, order=order, window_size=rolling
        )

        return self._finalize_measure(
            result,
            bool(rolling or within_period),
            rounding,
            growth,
            lag,
            standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_worst_realization(
        self,
        period: str | None = None,
        within_period: bool = False,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Worst Realization of an investment portfolio or asset's returns, the lowest return.

        The Worst Realization is the most conservative measure of the risk in the left tail: the loss on
        the worst day (or week, month) of the sample. It is the limit of the Value at Risk and Conditional
        Value at Risk as their tail probability goes to 0, and the risk measure of Young's (1998) minimax
        portfolio, which minimizes the worst loss rather than the variance. As it rests on a single
        observation it moves in jumps and depends on the length of the sample: a longer history can only
        add worse days.

        The formula is as follows:

        - Worst Realization = min(Return_t)

        The result is a return, negative for a loss, as the Value at Risk and Conditional Value at Risk of
        the Toolkit are.

        For more information, see the following sources:

        - Young, M.R. (1998). "A Minimax Portfolio Selection Rule with Linear Programming Solution."
        Management Science, 44(5), 673-683.
        - Riskfolio-Lib: RiskFunctions.WR (https://github.com/dcajasn/Riskfolio-Lib).
        - skfolio: measures.worst_realization (https://skfolio.org/api.html#module-skfolio.measures).

        Also known as: worst return, minimax risk, maximum loss.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            within_period (bool, optional): Whether to calculate the Worst Realization within the specified period or
            for the entire period. Thus whether to look at the Worst Realization within a specific year (if period =
            'yearly') or look at the entirety of all years. Defaults to False.
            Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rolling (int, optional): The rolling window size to use for the calculation. If set, the Worst Realization
            is calculated over a rolling window of this many periods across the full return history
            instead of per `period`. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Worst Realization values over time.
            Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series | pd.DataFrame: Worst Realization values, per ticker, or per period (or date) and ticker.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates the
        Worst Realization for each asset in the Toolkit instance, each on its own returns (a ticker that listed later
        or misses a date another ticker traded is measured on the returns it has).
        - If `growth` is set to True, the method calculates the growth of the Worst Realization values using the
        specified `lag`.
        - Riskfolio-Lib's WR and skfolio's worst_realization report the loss as a positive number
        (-min(Return)); the Toolkit keeps the sign of the return.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NVDA", "AMD", "INTC"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_worst_realization(period="yearly", within_period=True)
        ```

        Which returns:

        |      |    NVDA |     AMD |    INTC |   Benchmark |
        |:-----|--------:|--------:|--------:|------------:|
        | 2021 | -0.0821 | -0.062  | -0.1168 |     -0.0244 |
        | 2022 | -0.0951 | -0.1387 | -0.0856 |     -0.0435 |
        | 2023 | -0.0589 | -0.0922 | -0.0641 |     -0.0201 |
        | 2024 | -0.1    | -0.1062 | -0.2606 |     -0.0298 |
        | 2025 | -0.1697 | -0.089  | -0.115  |     -0.0585 |
        """
        period = period if period else "daily"
        returns = self._get_measure_returns(period, within_period, rolling, growth)

        result = dispersion_model.get_worst_realization(returns, window_size=rolling)

        return self._finalize_measure(
            result,
            bool(rolling or within_period),
            rounding,
            growth,
            lag,
            standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_gini_mean_difference(
        self,
        period: str | None = None,
        within_period: bool = False,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Gini Mean Difference (GMD) of an investment portfolio or asset's returns.

        The Gini Mean Difference is the average absolute difference between two returns drawn from the
        sample, a measure of dispersion that, unlike the standard deviation, does not square the
        deviations and needs no mean to measure them from. Yitzhaki (1982) showed that, unlike the variance,
        it is consistent with second-degree stochastic dominance, which makes it a sound risk measure for
        returns that are not normally distributed. For a normal distribution it equals 2σ / √π, about 1.13
        times the standard deviation.

        The formula is as follows:

        - Gini Mean Difference = Σ_i Σ_j |Return_i - Return_j| / (n * (n - 1))

        Where the double sum runs over every pair of the n returns. It is calculated as an ordered weighted
        average (OWA) of the returns sorted from low to high, with weight 2 * (2i - 1 - n) / (n * (n - 1))
        on the i-th lowest (Cajas 2021), and equals twice the second L-moment (`get_l_moment`).

        For more information, see the following sources:

        - Yitzhaki, S. (1982). "Stochastic Dominance, Mean Variance, and Gini's Mean Difference." The
        American Economic Review, 72(1), 178-185.
        - Cajas, D. (2021). "OWA Portfolio Optimization: A Disciplined Convex Programming Framework."
        SSRN Electronic Journal.
        - Riskfolio-Lib: RiskFunctions.GMD and OwaWeights.owa_gmd (https://github.com/dcajasn/Riskfolio-Lib).
        - skfolio: measures.gini_mean_difference (https://skfolio.org/api.html#module-skfolio.measures).

        Also known as: GMD, Gini's mean difference, mean absolute difference.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            within_period (bool, optional): Whether to calculate the Gini Mean Difference within the specified period or
            for the entire period. Thus whether to look at the Gini Mean Difference within a specific year (if period =
            'yearly') or look at the entirety of all years. Defaults to False.
            Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rolling (int, optional): The rolling window size to use for the calculation. If set, the Gini Mean Difference
            is calculated over a rolling window of this many periods across the full return history
            instead of per `period`. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Gini Mean Difference values over time.
            Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series | pd.DataFrame: Gini Mean Difference values, per ticker, or per period (or date) and ticker.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates the
        Gini Mean Difference for each asset in the Toolkit instance, each on its own returns (a ticker that listed later
        or misses a date another ticker traded is measured on the returns it has).
        - If `growth` is set to True, the method calculates the growth of the Gini Mean Difference values using the
        specified `lag`.
        - The result agrees with Riskfolio-Lib's GMD and skfolio's gini_mean_difference.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["MCD", "SBUX", "CMG"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_gini_mean_difference(period="yearly", within_period=True)
        ```

        Which returns:

        |      |    MCD |   SBUX |    CMG |   Benchmark |
        |:-----|-------:|-------:|-------:|------------:|
        | 2021 | 0.0104 | 0.0149 | 0.0174 |      0.009  |
        | 2022 | 0.0136 | 0.0244 | 0.0274 |      0.0171 |
        | 2023 | 0.0098 | 0.0142 | 0.0158 |      0.0093 |
        | 2024 | 0.012  | 0.0189 | 0.0194 |      0.0086 |
        | 2025 | 0.0126 | 0.022  | 0.0237 |      0.0112 |
        """
        period = period if period else "daily"
        returns = self._get_measure_returns(period, within_period, rolling, growth)

        result = dispersion_model.get_gini_mean_difference(returns, window_size=rolling)

        return self._finalize_measure(
            result,
            bool(rolling or within_period),
            rounding,
            growth,
            lag,
            standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_tail_gini(
        self,
        period: str | None = None,
        alpha: float = 0.05,
        simulations: int = 100,
        within_period: bool = False,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Tail Gini of an investment portfolio or asset's returns.

        The Tail Gini of Furman, Wang and Zitikis (2017) measures the variability within the loss tail,
        the alpha share of the worst returns. Cajas (2021) writes it as a weighted average of the
        Conditional Value at Risk at tail probabilities from 0.0001 up to alpha, which puts more weight on
        the worst returns within the tail than the Conditional Value at Risk at alpha, which weighs every
        return in the tail equally. It is therefore always at or below the Conditional Value at Risk and
        above the Worst Realization.

        The formula is as follows (with the default alpha=0.05 and simulations=100):

        - Tail Gini = Σ_j c_j * CVaR(alpha_j)
        - c_1 = alpha_2 * alpha_1 / alpha^2, c_j = (alpha_(j+1) - alpha_(j-1)) * alpha_j / alpha^2 and
          c_m = (alpha_m - alpha_(m-1)) / alpha

        Where alpha_1 < ... < alpha_m = alpha are m = `simulations` evenly spaced tail probabilities from
        0.0001 to alpha and CVaR(alpha_j) is the average of the alpha_j share of the lowest returns, the
        return at the boundary counting with its fraction (the ordered weighted average of Cajas 2021).
        The result is a return, negative for a loss, as the Conditional Value at Risk of the Toolkit is.

        For more information, see the following sources:

        - Furman, E., Wang, R. and Zitikis, R. (2017). "Gini-type Measures of Risk and Variability: Gini
        Shortfall, Capital Allocations, and Heavy-tailed Risks." Journal of Banking & Finance, 83, 70-84.
        - Cajas, D. (2021). "OWA Portfolio Optimization: A Disciplined Convex Programming Framework."
        SSRN Electronic Journal.
        - Riskfolio-Lib: RiskFunctions.TG and OwaWeights.owa_tg (https://github.com/dcajasn/Riskfolio-Lib).

        Also known as: TG, Gini shortfall, tail Gini risk.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            alpha (float, optional): The tail probability, between 0.0001 and 1. Defaults to 0.05.
            simulations (int, optional): The number of tail probabilities the Conditional Value at Risk
            is averaged over. Defaults to 100.
            within_period (bool, optional): Whether to calculate the Tail Gini within the specified period or
            for the entire period. Thus whether to look at the Tail Gini within a specific year (if period =
            'yearly') or look at the entirety of all years. Defaults to False.
            Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rolling (int, optional): The rolling window size to use for the calculation. If set, the Tail Gini
            is calculated over a rolling window of this many periods across the full return history
            instead of per `period`. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Tail Gini values over time.
            Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series | pd.DataFrame: Tail Gini values, per ticker, or per period (or date) and ticker.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates the
        Tail Gini for each asset in the Toolkit instance, each on its own returns (a ticker that listed later
        or misses a date another ticker traded is measured on the returns it has).
        - If `growth` is set to True, the method calculates the growth of the Tail Gini values using the
        specified `lag`.
        - The result agrees with Riskfolio-Lib's TG with the opposite sign: Riskfolio-Lib reports the loss as
        a positive number. Its Conditional Value at Risk is the exact one of Rockafellar and Uryasev
        (2002), which can differ slightly from the average of the returns below the interpolated quantile
        that `get_conditional_value_at_risk` takes.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["TSLA", "F", "GM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_tail_gini(period="yearly", within_period=True)
        ```

        Which returns:

        |      |    TSLA |       F |      GM |   Benchmark |
        |:-----|--------:|--------:|--------:|------------:|
        | 2021 | -0.0785 | -0.0558 | -0.056  |     -0.0207 |
        | 2022 | -0.1    | -0.0823 | -0.0642 |     -0.0365 |
        | 2023 | -0.0832 | -0.0642 | -0.0462 |     -0.0168 |
        | 2024 | -0.0902 | -0.0719 | -0.0542 |     -0.0218 |
        | 2025 | -0.099  | -0.0526 | -0.0582 |     -0.0348 |
        """
        period = period if period else "daily"
        returns = self._get_measure_returns(period, within_period, rolling, growth)

        result = dispersion_model.get_tail_gini(
            returns, alpha=alpha, simulations=simulations, window_size=rolling
        )

        return self._finalize_measure(
            result,
            bool(rolling or within_period),
            rounding,
            growth,
            lag,
            standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_risk_range(
        self,
        period: str | None = None,
        measure: str = "range",
        alpha: float = 0.05,
        beta: float | None = None,
        simulations: int = 100,
        within_period: bool = False,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate a Range measure of an investment portfolio or asset's returns: the distance between the
        gain tail and the loss tail of the return distribution.

        The Range is the distance between the best and the worst return. Cajas (2021) generalizes it to
        the tails: the VaR Range is the distance between the 1 - beta and the alpha quantile, the CVaR
        Range between the average return in the best beta share and in the worst alpha share of the
        returns and the Tail Gini Range between the Tail Gini of the gains and of the losses. They measure
        the width of the distribution where it matters most, without the assumption of symmetry behind the
        standard deviation, and with beta different from alpha they weigh the gains and the losses
        differently. The VaR Range at alpha = beta = 0.25 is the interquartile range. The EVaR Range does the
        same with the historic Entropic Value at Risk, the most severe of these tail measures.

        The formula is as follows (with the default alpha=0.05 and beta=alpha):

        - Range = max(Return) - min(Return)
        - VaR Range = Quantile(Return, 1 - beta) - Quantile(Return, alpha)
        - CVaR Range = mean(Return >= Quantile(Return, 1 - beta)) - mean(Return <= Quantile(Return, alpha))
        - Tail Gini Range = Tail Gini of the gains (beta) - Tail Gini of the losses (alpha)
        - EVaR Range = EVaR of the gains (beta) - EVaR of the losses (alpha)

        Where the quantiles are interpolated linearly and the CVaR is the average of the returns beyond the
        quantile, as in the historic `get_value_at_risk` and `get_conditional_value_at_risk`, the Tail
        Gini is that of `get_tail_gini` and the EVaR that of `get_entropic_value_at_risk` with
        distribution="historic", each measured on the negated returns for the gains. The result is a
        positive decimal.

        For more information, see the following sources:

        - Cajas, D. (2021). "OWA Portfolio Optimization: A Disciplined Convex Programming Framework."
        SSRN Electronic Journal.
        - Riskfolio-Lib: RiskFunctions.RG, RiskFunctions.VRG, RiskFunctions.CVRG, RiskFunctions.TGRG and
        RiskFunctions.EVRG (https://github.com/dcajasn/Riskfolio-Lib).

        Also known as: return range, VaR range, CVaR range, Tail Gini range, EVaR range, tail spread.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            measure (str, optional): The range to calculate: "range", "var", "cvar", "tail_gini" or "evar".
            Defaults to "range".
            alpha (float, optional): The tail probability of the losses, between 0 and 1 (between 0.0001
            and 1 for the Tail Gini). Defaults to 0.05.
            beta (float | None, optional): The tail probability of the gains. Defaults to None, which uses
            alpha.
            simulations (int, optional): The number of tail probabilities of the Tail Gini Range.
            Defaults to 100.
            within_period (bool, optional): Whether to calculate the Risk Range within the specified period or
            for the entire period. Thus whether to look at the Risk Range within a specific year (if period =
            'yearly') or look at the entirety of all years. Defaults to False.
            Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rolling (int, optional): The rolling window size to use for the calculation. If set, the Risk Range
            is calculated over a rolling window of this many periods across the full return history
            instead of per `period`. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Risk Range values over time.
            Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series | pd.DataFrame: Range values, per ticker, or per period (or date) and ticker.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates the
        Risk Range for each asset in the Toolkit instance, each on its own returns (a ticker that listed later
        or misses a date another ticker traded is measured on the returns it has).
        - If `growth` is set to True, the method calculates the growth of the Risk Range values using the
        specified `lag`.
        - The Range, the Tail Gini Range and the EVaR Range agree with Riskfolio-Lib's RG, TGRG and EVRG.
        Riskfolio-Lib's VRG and CVRG take the order statistic at ceil(alpha * n) instead of the interpolated quantile and the exact
        Conditional Value at Risk of Rockafellar and Uryasev (2002), so the VaR Range and CVaR Range differ
        slightly from them and equal the difference of the Toolkit's own historic Value at Risk and
        Conditional Value at Risk of the gains and the losses instead.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "WMT", "COST"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_risk_range(period="yearly", within_period=True, measure="cvar")
        ```

        Which returns:

        |      |   AMZN |    WMT |   COST |   Benchmark |
        |:-----|-------:|-------:|-------:|------------:|
        | 2021 | 0.065  | 0.0457 | 0.05   |      0.0352 |
        | 2022 | 0.1352 | 0.0736 | 0.0879 |      0.0634 |
        | 2023 | 0.0913 | 0.0393 | 0.0523 |      0.0332 |
        | 2024 | 0.0772 | 0.0485 | 0.0499 |      0.0352 |
        | 2025 | 0.0975 | 0.0728 | 0.0631 |      0.0538 |
        """
        period = period if period else "daily"
        returns = self._get_measure_returns(period, within_period, rolling, growth)

        result = dispersion_model.get_risk_range(
            returns,
            measure=measure,
            alpha=alpha,
            beta=beta,
            simulations=simulations,
            window_size=rolling,
        )

        return self._finalize_measure(
            result,
            bool(rolling or within_period),
            rounding,
            growth,
            lag,
            standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_l_moment(
        self,
        period: str | None = None,
        order: int = 2,
        ratio: bool = False,
        within_period: bool = False,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate an L-moment, or an L-moment ratio, of an investment portfolio or asset's returns.

        L-moments (Hosking 1990) describe the location, scale and shape of a distribution as the ordinary
        moments do, but as linear combinations of the sorted returns rather than powers of the deviations.
        This makes them far less sensitive to a few extreme returns and well defined whenever the mean is:
        the L-skewness and L-kurtosis of a sample are bounded and nearly unbiased, where the sample skewness
        and kurtosis of fat-tailed returns are dominated by the largest move. The first L-moment is the
        mean, the second a measure of scale (half the Gini Mean Difference), and the ratios of the third and
        fourth to the second, the L-skewness and L-kurtosis, measure the asymmetry and the tails. For a
        normal distribution the L-skewness is 0 and the L-kurtosis 0.1226.

        The formula is as follows (with the default order=2):

        - L-moment λ_k = Σ_i w_i * Return_(i)
        - w_i = Σ_j (-1)^j * C(k - 1, j) * C(i - 1, k - 1 - j) * C(n - i, j) / (k * C(n, k))
        - L-moment ratio τ_k = λ_k / λ_2 for k >= 3 and τ_2 = λ_2 / λ_1 (the L-CV)

        Where Return_(i) is the i-th lowest of the n returns, k the order, C the binomial coefficient and j
        runs from 0 to k - 1. These are Hosking's unbiased sample L-moments, written as an ordered weighted
        average as in Cajas (2021).

        For more information, see the following sources:

        - Hosking, J.R.M. (1990). "L-moments: Analysis and Estimation of Distributions using Linear
        Combinations of Order Statistics." Journal of the Royal Statistical Society: Series B, 52(1),
        105-124.
        - Cajas, D. (2021). "OWA Portfolio Optimization: A Disciplined Convex Programming Framework."
        SSRN Electronic Journal.
        - Riskfolio-Lib: RiskFunctions.L_Moment and OwaWeights.owa_l_moment
        (https://github.com/dcajasn/Riskfolio-Lib).

        Also known as: L-moments, L-scale, L-skewness, L-kurtosis, L-CV, linear moments.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            order (int, optional): The order k, an integer of 1 or higher (1 is the mean, 2 the L-scale,
            3 with ratio=True the L-skewness and 4 the L-kurtosis). Defaults to 2.
            ratio (bool, optional): Whether to return the L-moment ratio, which needs an order of 2 or
            higher. Defaults to False.
            within_period (bool, optional): Whether to calculate the L-moment within the specified period or
            for the entire period. Thus whether to look at the L-moment within a specific year (if period =
            'yearly') or look at the entirety of all years. Defaults to False.
            Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rolling (int, optional): The rolling window size to use for the calculation. If set, the L-moment
            is calculated over a rolling window of this many periods across the full return history
            instead of per `period`. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the L-moment values over time.
            Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series | pd.DataFrame: L-moment values, per ticker, or per period (or date) and ticker.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates the
        L-moment for each asset in the Toolkit instance, each on its own returns (a ticker that listed later
        or misses a date another ticker traded is measured on the returns it has).
        - If `growth` is set to True, the method calculates the growth of the L-moment values using the
        specified `lag`.
        - The L-moments agree with Riskfolio-Lib's L_Moment and scipy's stats.lmoment(standardize=False), the
        ratios scipy's stats.lmoment with standardize=True.
        - A sample needs at least `order` returns, fewer give NaN.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["PFE", "MRK", "JNJ"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_l_moment(period="yearly", within_period=True, order=3, ratio=True)
        ```

        Which returns:

        |      |     PFE |     MRK |     JNJ |   Benchmark |
        |:-----|--------:|--------:|--------:|------------:|
        | 2021 |  0.1252 | -0.0172 | -0.0248 |     -0.0386 |
        | 2022 |  0.0344 |  0.0321 |  0.0518 |      0.0138 |
        | 2023 | -0.0124 | -0.0261 | -0.0389 |     -0.0039 |
        | 2024 |  0.0588 | -0.0149 |  0.0183 |     -0.0586 |
        | 2025 | -0.0016 | -0.0154 | -0.0189 |     -0.0597 |
        """
        period = period if period else "daily"
        returns = self._get_measure_returns(period, within_period, rolling, growth)

        result = dispersion_model.get_l_moment(
            returns, order=order, ratio=ratio, window_size=rolling
        )

        return self._finalize_measure(
            result,
            bool(rolling or within_period),
            rounding,
            growth,
            lag,
            standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_l_moment_risk_measure(
        self,
        period: str | None = None,
        order: int = 4,
        method: Literal["msd", "mss", "crra"] = "msd",
        risk_aversion: float = 0.5,
        max_weight: float = 0.5,
        within_period: bool = False,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the L-moment Convex Risk Measure of an investment portfolio or asset's returns.

        Cajas (2023) combines the L-moments of order 2 up to `order` into one convex risk measure: a
        weighted sum of the L-scale, minus the L-moment of order 3 (so a negative skew adds risk), plus that
        of order 4 and so on, with weights that are non-negative, sum to 1 and decrease with the order. The
        weights are chosen so that the ordered weighted average they imply puts at least as much weight on
        every worse return, which keeps the measure convex and therefore usable in portfolio optimization.
        It captures the dispersion, the skew and the tails of the returns in one figure that is as robust to
        outliers as the L-moments themselves.

        The formula is as follows (with the default order=4 and method="msd"):

        - L-moment Convex Risk Measure = Σ_k φ_k * (-1)^k * λ_k for k = 2, ..., order
        - Σ φ_k = 1, 0 <= φ_k <= max_weight and φ_k >= φ_(k+1)

        Where λ_k is the L-moment of order k (`get_l_moment`) and the φ_k follow from `method`: "msd"
        minimizes the squared differences between the neighbouring weights of the ordered weighted average,
        "mss" the sum of its squared weights and "crra" sets φ_k in proportion to
        Π_(i=1..k-1) (risk_aversion + i - 1) / k!, constant relative risk aversion, after which the ordered
        weights are made non-decreasing. The result is a positive decimal.

        For more information, see the following sources:

        - Cajas, D. (2023). "Higher Order Moment Portfolio Optimization with L-Moments." SSRN 4393155.
        - Hosking, J.R.M. (1990). "L-moments: Analysis and Estimation of Distributions using Linear
        Combinations of Order Statistics." Journal of the Royal Statistical Society: Series B, 52(1),
        105-124.
        - Riskfolio-Lib: RiskFunctions.L_Moment_CRM and OwaWeights.owa_l_moment_crm
        (https://github.com/dcajasn/Riskfolio-Lib).

        Also known as: L-moment CRM, L-moment based convex risk measure.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            order (int, optional): The highest L-moment order, an integer of 2 or higher. Defaults to 4.
            method (str, optional): How the L-moments are weighted: "msd", "mss" or "crra". Defaults to
            "msd".
            risk_aversion (float, optional): The risk aversion of method="crra", between 0 and 1.
            Defaults to 0.5.
            max_weight (float, optional): The largest weight of one L-moment for "msd" and "mss", between
            0 and 1. Defaults to 0.5.
            within_period (bool, optional): Whether to calculate the L-moment Convex Risk Measure within the specified
            period or for the entire period. Thus whether to look at the L-moment Convex Risk Measure within a specific
            year (if period = 'yearly') or look at the entirety of all years. Defaults to False.
            Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rolling (int, optional): The rolling window size to use for the calculation. If set, the
            L-moment Convex Risk Measure is calculated over a rolling window of this many periods across the full return
            history instead of per `period`. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the L-moment Convex Risk Measure values over time.
            Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series | pd.DataFrame: L-moment Convex Risk Measure values, per ticker, or per period (or date) and ticker.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates the
        L-moment Convex Risk Measure for each asset in the Toolkit instance, each on its own returns (a ticker that
        listed later or misses a date another ticker traded is measured on the returns it has).
        - If `growth` is set to True, the method calculates the growth of the L-moment Convex Risk Measure values using
        the specified `lag`.
        - With method="crra" the result agrees with Riskfolio-Lib's L_Moment_CRM. For "msd" and "mss" the weights
        solve a small convex program over the order - 1 weights with scipy (SLSQP) where Riskfolio-Lib uses
        cvxpy with the Clarabel solver; the Toolkit's weights meet the constraints exactly and match a
        tightly solved cvxpy program, while Clarabel's default tolerances leave Riskfolio-Lib's weights
        slightly off, so the results differ in the fourth significant digit for "msd" and the second for
        "mss". Riskfolio-Lib's maximum entropy method ("ME") is left out: the entropy it maximizes is
        that of a bound which is the same for every set of weights, so its result depends on the solver.
        - With order=3 and max_weight=0.5 no weights meet the constraints, which gives an error.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["V", "MA", "PYPL"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_l_moment_risk_measure(period="yearly", within_period=True)
        ```

        Which returns:

        |      |      V |     MA |   PYPL |   Benchmark |
        |:-----|-------:|-------:|-------:|------------:|
        | 2021 | 0.0044 | 0.0052 | 0.0068 |      0.0024 |
        | 2022 | 0.0054 | 0.0058 | 0.0098 |      0.0044 |
        | 2023 | 0.0028 | 0.0031 | 0.0067 |      0.0024 |
        | 2024 | 0.0031 | 0.0029 | 0.0061 |      0.0024 |
        | 2025 | 0.0039 | 0.004  | 0.0069 |      0.0032 |
        """
        period = period if period else "daily"
        returns = self._get_measure_returns(period, within_period, rolling, growth)

        result = dispersion_model.get_l_moment_risk_measure(
            returns,
            order=order,
            method=method,
            risk_aversion=risk_aversion,
            max_weight=max_weight,
            window_size=rolling,
        )

        return self._finalize_measure(
            result,
            bool(rolling or within_period),
            rounding,
            growth,
            lag,
            standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_skewness_kurtosis_ratio(
        self,
        period: str | None = None,
        within_period: bool = False,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Skewness-Kurtosis Ratio of an investment portfolio or asset's returns.

        The Skewness-Kurtosis Ratio (Bacon 2008) divides the skewness by the kurtosis, combining the two
        shape statistics an investor in fat-tailed assets cares about into one number: a positive skew is
        attractive and fat tails are not, so a higher ratio is better. It is used to rank funds whose
        returns are far from normal, where the Sharpe Ratio's reliance on the standard deviation misleads.

        The formula is as follows:

        - Skewness-Kurtosis Ratio = Skewness / Kurtosis

        Where the Skewness is that of `get_skewness` and the Kurtosis that of `get_kurtosis` with the
        default fisher=False, Pearson's kurtosis (3 for a normal distribution), both the bias-adjusted
        sample estimates.

        For more information, see the following sources:

        - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution." John Wiley &
        Sons, chapter 4.
        - PerformanceAnalytics (R): SkewnessKurtosisRatio (https://github.com/braverock/PerformanceAnalytics).

        Also known as: skewness to kurtosis ratio.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            within_period (bool, optional): Whether to calculate the Skewness-Kurtosis Ratio within the specified
            period or for the entire period. Thus whether to look at the Skewness-Kurtosis Ratio within a specific
            year (if period = 'yearly') or look at the entirety of all years. Defaults to False.
            Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rolling (int, optional): The rolling window size to use for the calculation. If set, the
            Skewness-Kurtosis Ratio is calculated over a rolling window of this many periods across the full return
            history instead of per `period`. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Skewness-Kurtosis Ratio values over time.
            Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series | pd.DataFrame: Skewness-Kurtosis Ratio values, per ticker, or per period (or date) and ticker.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates the
        Skewness-Kurtosis Ratio for each asset in the Toolkit instance, each on its own returns (a ticker that
        listed later or misses a date another ticker traded is measured on the returns it has).
        - If `growth` is set to True, the method calculates the growth of the Skewness-Kurtosis Ratio values using
        the specified `lag`.
        - PerformanceAnalytics' SkewnessKurtosisRatio divides the population (moment) skewness by the
        population kurtosis; the Toolkit uses its bias-adjusted sample estimates of both, as in
        `get_skewness` and `get_kurtosis`, which differ by about 0.4% of the ratio for three years of daily
        returns and more for short samples.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["META", "GOOGL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_skewness_kurtosis_ratio(period="yearly", within_period=True)
        ```

        Which returns:

        |      |    META |   GOOGL |    MSFT |   Benchmark |
        |:-----|--------:|--------:|--------:|------------:|
        | 2021 |  0.0058 |  0.0708 | -0.0057 |     -0.0897 |
        | 2022 | -0.0997 |  0.0468 |  0.038  |      0.0132 |
        | 2023 |  0.1119 | -0.0405 |  0.0846 |     -0.007  |
        | 2024 |  0.0804 |  0.0277 | -0.1715 |     -0.1095 |
        | 2025 |  0.0563 |  0.0465 |  0.1    |      0.0567 |
        """
        period = period if period else "daily"
        returns = self._get_measure_returns(period, within_period, rolling, growth)

        result = dispersion_model.get_skewness_kurtosis_ratio(
            returns, window_size=rolling
        )

        return self._finalize_measure(
            result,
            bool(rolling or within_period),
            rounding,
            growth,
            lag,
            standardize,
        )
