"""Distribution Risk Module"""

__docformat__ = "google"

import numpy as np
import pandas as pd

from financetoolkit.helpers import handle_portfolio
from financetoolkit.risk import risk_model
from financetoolkit.risk.base.base_controller import RiskBase
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
class DistributionRisk(RiskBase):
    """
    The shape of the return distribution: variance, skewness, kurtosis, the Hill estimator, the tail ratio, the mean absolute deviation, the coefficient of variation, autocorrelation, the Hurst exponent and the significance of a move.
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
        252 / 12 (monthly), 252 / 4 (quarterly) or 252 (yearly). With `rolling`, the result is the sample
        variance of the last `rolling` returns of the chosen period itself, without scaling.

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
        exceedance, worse than the loss the trailing VaR expected on all but a share alpha of days.

        The formula is as follows:

        - Volatility Multiple(t) = Return(t) / σ(Return(t - window), ..., Return(t - 1))
        - VaR Multiple(t) = Return(t) / |Quantile(Return(t - window), ..., Return(t - 1); alpha)|
        - Percentile(t) = Count(|Return(s)| < |Return(t)|, s < t) / Count(s < t)
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
        | NVDA      |  -0.1696 |               -5.0657 |        -3.4612 |       0.9976 | 2020-03-16      |
        | KO        |   0.0319 |                3.9007 |         2.343  |       0.9818 | 2022-11-10      |
        | Benchmark |  -0.0141 |               -1.7546 |        -1.0217 |       0.8613 | 2025-01-10      |
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
        the denominator over all n returns, and k = 1, ..., `lags` (10 by default).

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
