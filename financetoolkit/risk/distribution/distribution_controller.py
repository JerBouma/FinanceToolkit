"""Distribution Risk Module"""

__docformat__ = "google"

from typing import Literal

import numpy as np
import pandas as pd

from financetoolkit.helpers import calculate_per_window, handle_portfolio
from financetoolkit.risk import risk_model
from financetoolkit.risk.distribution import dispersion_model, fractal_model
from financetoolkit.risk.helpers import check_rolling
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import (
    InvalidArgumentError,
    check_option,
    check_period,
    handle_errors,
    validate_alpha,
)
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import (
    apply_rounding,
    finalize_dataset,
)

# ruff: noqa: E501


@filter_shown_columns
@track_sources
class DistributionRisk:
    """
    The shape of the return distribution: variance, skewness, kurtosis, central and partial moments, L-moments, the Gini mean difference, the tail Gini, range measures, the worst realization, the Hill estimator, the tail ratio, the mean absolute deviation, the coefficient of variation, autocorrelation, the significance of a move, and memory, fractal and chaos measures: the Hurst exponent (generalized, rescaled range, detrended fluctuation), the fractal dimension (Higuchi, Katz, Petrosian), the correlation dimension and the Lyapunov exponent.
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
        Calculate the Tail Ratio: how large the best returns of an investment portfolio or asset are
        compared with its worst returns.

        The Tail Ratio compares the size of the right (gain) tail to the left (loss) tail of the
        return distribution. A Tail Ratio above 1 indicates that best-case gains outsize worst-case
        losses.

        The formula is as follows:

        - Tail Ratio = |Quantile(Returns, 1 - alpha)| / |Quantile(Returns, alpha)|

        The quantiles interpolate linearly, and a left-tail quantile of 0 gives NaN.

        Also known as: gain-to-pain tail ratio.

        Shares similar logic:
        - empyrical, tail_ratio: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L1501
        - quantstats, tail_ratio: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L2192

        Args:
            period (str, optional): The data frequency for returns. Options are: "daily", "weekly", "monthly",
                "quarterly", "yearly". Defaults to "daily".
            alpha (float, optional): The percentile used to define each tail (e.g., 0.05 uses the 5th and
            95th percentile). Defaults to 0.05.
            within_period (bool, optional): Whether to calculate the Tail Ratio within the specified period or
            for the entire period. Thus whether to look at the Tail Ratio within a specific year (if period =
            'yearly') or look at the entirety of all years. Defaults to False.
            Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rolling (int, optional): The rolling window size to use for the calculation. If set, the Tail
            Ratio is calculated over a rolling window of this many periods across the full return history
            instead of per `period`. At least 2. Defaults to None.
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
        - With growth, the method calculates the growth of Tail Ratio values using the specified `lag`.

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
        validate_alpha(alpha)
        period = period if period else "daily"

        check_period(period, ["daily", "weekly", "monthly", "quarterly", "yearly"])
        check_rolling(rolling, self._historical_data, period)
        if (
            period == "daily"
            and within_period
            and self._historical_data["intraday"].empty
        ):
            raise InvalidArgumentError(
                "Intraday data is required for within-period daily calculations. Either set "
                "within_period=False or initialise the Toolkit with an intraday_period."
            )
        if growth and not within_period and not rolling:
            raise InvalidArgumentError(
                "Growth calculations need one value per period, while "
                "within_period=False produces a single value for the entire period. "
                "Set within_period=True to calculate growth."
            )

        if rolling:
            returns = self._historical_data[period]["Return"]
            tail_ratio = risk_model.get_tail_ratio(returns, alpha, window=rolling)
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
        Calculate the Skewness: how asymmetric the returns of an investment portfolio or asset are,
        negative when the large deviations from the mean are losses.

        Skewness is a statistical measure used in finance to assess the asymmetry in the distribution of
        returns for an investment portfolio or asset over a defined period. It offers valuable insights
        into the shape of the return distribution, indicating whether returns are skewed towards the
        positive or negative side of the mean. Skewness is a crucial tool for investors and analysts seeking
        to understand the potential risk and return characteristics of an investment, aiding in the assessment
        of the distribution's tails and potential outliers. It provides a means to gauge the level of
        skew in returns, enabling more informed investment decisions and risk management strategies.

        The formula is as follows:

        - Skewness = n / ((n - 1) * (n - 2)) * Σ ((Return - Mean) / Std)^3

        Where n is the number of returns and Std the sample standard deviation (with n - 1 in the
        denominator), which is the bias-adjusted sample skewness (G1) of pandas and Excel's SKEW.

        Also known as: return distribution asymmetry, tail skew.

        Shares similar logic:
        - scipy, skew: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html

        Args:
            period (str, optional): The data frequency for returns. Options are: "daily", "weekly", "monthly",
                "quarterly", "yearly". Defaults to "daily".
            within_period (bool, optional): Whether to calculate the Skewness within the specified period or for the
            entire period. Thus whether to look at the Skewness within a specific year (if period = 'yearly') or look
            at the entirety of all years. Defaults to False.
            Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rolling (int, optional): The rolling window size to use for the calculation. If set, Skewness is
            calculated over a rolling window of this many periods across the full return history instead of
            per `period`. At least 3. Defaults to None.
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
        - With growth, the method calculates the growth of Skewness values using the specified `lag`.
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

        check_period(period, ["daily", "weekly", "monthly", "quarterly", "yearly"])
        check_rolling(rolling, self._historical_data, period, 3)
        if (
            period == "daily"
            and within_period
            and self._historical_data["intraday"].empty
        ):
            raise InvalidArgumentError(
                "Intraday data is required for within-period daily calculations. Either set "
                "within_period=False or initialise the Toolkit with an intraday_period."
            )
        if growth and not within_period and not rolling:
            raise InvalidArgumentError(
                "Growth calculations need one value per period, while "
                "within_period=False produces a single value for the entire period. "
                "Set within_period=True to calculate growth."
            )

        if rolling:
            returns = self._historical_data[period]["Return"]
            skewness = risk_model.get_skewness(returns, window=rolling)
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
        Calculate the Kurtosis: how heavy the tails of the returns of an investment portfolio or asset
        are, how likely extreme returns are compared with a normal distribution.

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

        By default this is Pearson's kurtosis, which is 3 for a normal distribution; Fisher's
        definition gives the excess kurtosis, which is 0. Daily stock returns typically show an excess kurtosis of
        several points, and within a single year one exceptional day (an earnings shock, a market-wide
        gap) can push it above 10, so a high value often reflects one event rather than a persistently
        fat-tailed distribution.

        The formula is as follows:

        - Kurtosis = Excess Kurtosis + 3
        - Excess Kurtosis = n(n + 1) / ((n - 1)(n - 2)(n - 3)) * Σ z^4 - 3(n - 1)^2 / ((n - 2)(n - 3))
        - z = (Return - Mean) / Std

        Where n is the number of returns and Std the sample standard deviation (with n - 1 in the
        denominator), which is the bias-adjusted sample excess kurtosis (G2) of pandas and Excel's KURT.
        Fisher's definition returns the Excess Kurtosis.

        Also known as: tail heaviness, fat tails, leptokurtosis.

        Shares similar logic:
        - scipy, kurtosis: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosis.html

        Args:
            period (str, optional): The data frequency for returns. Options are: "daily", "weekly", "monthly",
                "quarterly", "yearly". Defaults to "daily".
            within_period (bool, optional): Whether to calculate the Kurtosis within the specified period or for the
            entire period. Thus whether to look at the Kurtosis within a specific year (if period = 'yearly') or look
            at the entirety of all years. Defaults to False.
            Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            fisher (bool, optional): Whether to use Fisher's definition of kurtosis (kurtosis = 0.0
            for a normal distribution). Defaults to False.
            rolling (int, optional): The rolling window size to use for the calculation. If set, Kurtosis is
            calculated over a rolling window of this many periods across the full return history instead of
            per `period`. At least 4. Defaults to None.
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
        - With growth, the method calculates the growth of Kurtosis values using the specified `lag`.
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

        check_period(period, ["daily", "weekly", "monthly", "quarterly", "yearly"])
        check_rolling(rolling, self._historical_data, period, 4)
        if (
            period == "daily"
            and within_period
            and self._historical_data["intraday"].empty
        ):
            raise InvalidArgumentError(
                "Intraday data is required for within-period daily calculations. Either set "
                "within_period=False or initialise the Toolkit with an intraday_period."
            )
        if growth and not within_period and not rolling:
            raise InvalidArgumentError(
                "Growth calculations need one value per period, while "
                "within_period=False produces a single value for the entire period. "
                "Set within_period=True to calculate growth."
            )

        if rolling:
            returns = self._historical_data[period]["Return"]
            kurtosis = risk_model.get_kurtosis(returns, fisher=fisher, window=rolling)
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
        Calculate the Hill Estimator of the tail index of returns: how heavy the tail of each asset's
        return distribution is, smaller for a heavier tail.

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

        The formula is as follows:

        - Hill Shape (xi) = (1 / k) * Σ (ln X_(i) - ln X_(k+1)) for i = 1, ..., k
        - Hill Tail Index = 1 / xi
        - Standard Error = Hill Tail Index / √k

        Where X_(1) >= X_(2) >= ... are the strictly positive losses (the negated returns, or the returns
        themselves for the right tail) sorted from large to small, and k is the number of order statistics used:
        int(0.1 * the number of positive losses) by default, or `k` itself when it is an integer.

        For more information about the method, see the following paper:

        - Hill, B.M. (1975). "A Simple General Approach to Inference About the Tail of
        a Distribution." The Annals of Statistics, 3(5), 1163-1174.

        Also known as: Hill tail index estimator, Hill's estimator.

        Args:
            period (str, optional): The data frequency for returns. Options are: "daily", "weekly", "monthly",
                "quarterly", "yearly". Defaults to "daily".
            within_period (bool, optional): Whether to calculate the estimator within the specified period
            or for the entire period. Defaults to False.
            Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            k (int | float, optional): The number of upper order statistics to use. If a float in (0, 1)
            it is interpreted as the fraction of the strictly positive observations to use. A period
            without such observations reports NaN with 0 observations used. Defaults to 0.1 (the top 10%).
            tail (str, optional): Which tail to estimate. Options are: "left" (the loss tail), "right" (the
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

        check_period(period, ["daily", "weekly", "monthly", "quarterly", "yearly"])
        if (
            period == "daily"
            and within_period
            and self._historical_data["intraday"].empty
        ):
            raise InvalidArgumentError(
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
        Calculate the Variance: how widely the daily returns of an investment portfolio or asset spread
        around their mean in each period.

        Variance measures the spread or dispersion of returns around the mean. A higher
        Variance indicates more variability in the returns, while a lower Variance suggests
        that the returns are closer to the mean.

        The formula is as follows:

        - Variance = Var(Daily Returns within the period) * Trading Days per Period

        Where Var is the sample variance (with n - 1 in the denominator) and the trading days per period are 252 / 52
        (weekly), 252 / 12 (monthly), 252 / 4 (quarterly) or 252 (yearly). A ticker with returns on weekends, such as
        a cryptocurrency, has a return on every calendar day and uses 365 days a year instead (365 / 52 weekly). With
        `rolling`, the result is the sample variance of the last `rolling` returns of the chosen period itself,
        without scaling.

        Also known as: dispersion, spread.

        Args:
            period (str, optional): The data frequency for returns. Options are: "weekly", "monthly", "quarterly",
                "yearly". Defaults to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise
                "yearly".
            rolling (int, optional): The rolling window size to use for the calculation. If set,
            Variance is calculated over a rolling window of this many periods (e.g. period='monthly'
            and rolling=6 gives the rolling 6-month Variance) instead of one value per `period`.
            At least 2. Defaults to None.
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
        - With growth, the method calculates the growth of Variance values using the specified `lag`.

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

        check_period(period, ["weekly", "monthly", "quarterly", "yearly"])
        check_rolling(rolling, self._historical_data, period)

        if rolling:
            period_returns = self._historical_data[period]["Return"]
            variance = risk_model.get_variance(period_returns, period, window=rolling)
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
        Calculate the Mean Absolute Deviation (MAD): how far the daily returns of an investment
        portfolio or asset lie from their mean on average in each period.

        MAD measures the average absolute distance of each return from the mean return. Unlike
        Variance and Volatility, it does not square the deviations, making it less sensitive to
        outliers.

        The formula is as follows:

        - Mean Absolute Deviation = Σ |Return_t - Mean| / n

        Where the returns are the n daily returns within each period, without scaling to the period length.

        Args:
            period (str, optional): The data frequency for returns. Options are: "weekly", "monthly", "quarterly",
                "yearly". Defaults to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise
                "yearly".
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
        - With growth, the method calculates the growth of MAD values using the specified `lag`.

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

        check_period(period, ["weekly", "monthly", "quarterly", "yearly"])

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
        Calculate the Coefficient of Variation (CV): how large the spread of the daily returns of an
        investment portfolio or asset is relative to their mean in each period.

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
        (with n - 1 in the denominator), without scaling to the period length.

        Also known as: relative standard deviation.

        Args:
            period (str, optional): The data frequency for returns. Options are: "weekly", "monthly", "quarterly",
                "yearly". Defaults to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise
                "yearly".
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
        - With growth, the method calculates the growth of CV values using the specified `lag`.

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

        check_period(period, ["weekly", "monthly", "quarterly", "yearly"])

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

    @handle_errors
    def get_autocorrelation(
        self,
        lags: int = 10,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Autocorrelation Function (ACF) of each asset's daily returns: how strongly a
        return is related to the returns before it, for a range of lags.

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

        Referenced against:
        - statsmodels, acf: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html

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
        method: Literal[
            "generalized", "rescaled_range", "detrended_fluctuation"
        ] = "generalized",
        period: str | None = None,
        rolling: int | None = None,
        q: float | list[float] = 2.0,
        window_sizes: list[int] | None = None,
        corrected: bool = True,
        order: int = 1,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Hurst Exponent of each asset: its long-term memory, which shows whether the price
        is mean-reverting, trending, or a random walk.

        The Hurst Exponent (H) is interpreted as follows:

        - H < 0.5: the series is mean-reverting (anti-persistent).
        - H = 0.5: the series is a random walk (no memory).
        - H > 0.5: the series is trending (persistent).

        For liquid stocks the estimate is usually close to 0.5, and with lags of at most
        `max_lag` days it describes short-term behaviour only; a value of 0.45 or 0.55 over a
        few years is within estimation noise rather than evidence of a tradable pattern, so the
        exponent is mostly of use to compare assets or periods (with `rolling`) with each other.

        The methods are as follows:

        - Generalized (Di Matteo, Aste and Dacorogna, 2005), on the log prices X = ln P:
          K_q(tau) = Mean(abs(d - Mean(d))^q) of the differences d = X_(t+tau) - X_t, and
          ln K_q(tau) = c + q * H(q) * ln tau, fitted by least squares over tau = 2, ..., 20.
          With q = 2 this is the slope of the log standard deviation (with n in the denominator) of the lagged
          differences on the log lag. A list of q gives H(q) per q: for fat-tailed returns H(1)
          above H(2) points to multifractality.
        - Rescaled range (Hurst, 1951; Anis and Lloyd, 1976; Peters, 1994), on the log returns
          x: within windows of n returns, R/S = (Max(Y) - Min(Y)) / Std(x) with Y the cumulative
          deviations from the window mean and Std the population standard deviation; (R/S)(n)
          its average over the floor(N / n) windows, and H = 0.5 + slope of
          ln (R/S)(n) - ln E[R/S](n) on ln n, with E[R/S](n) = ((n - 0.5) / n) *
          Gamma((n - 1) / 2) / (Sqrt(pi) * Gamma(n / 2)) * Sum(Sqrt((n - i) / i), i = 1, ..., n - 1)
          the expected rescaled range of independent returns. Without the correction, H is the
          slope of ln (R/S)(n) on ln n.
        - Detrended fluctuation (Peng et al., 1994), on the log returns x: the profile
          Y_k = Sum(x_i - Mean(x)) is cut into windows of n, a straight line (a polynomial of
          degree `order`) is fitted and removed in every window, F(n) = Sqrt(Mean(residual^2))
          and H (alpha) = slope of ln F(n) on ln n.

        The window sizes n of the rescaled range and the detrended fluctuation grow by 20% from
        10 up to a tenth of the observations, so every size has at least ten windows.

        For more information, see the following sources:

        - Di Matteo, T., Aste, T., & Dacorogna, M.M. (2005). "Long-Term Memories of Developed and
          Emerging Markets: Using the Scaling Analysis to Characterize Their Stage of
          Development." Journal of Banking & Finance, 29(4), 827-851.
        - Di Matteo, T. (2007). "Multi-scaling in finance." Quantitative Finance, 7(1), 21-36.
        - Hurst, H.E. (1951). "Long-term storage capacity of reservoirs." Transactions of the
          American Society of Civil Engineers, 116, 770-799.
        - Anis, A.A. & Lloyd, E.H. (1976). "The expected value of the adjusted rescaled Hurst
          range of independent normal summands." Biometrika, 63(1), 111-116.
        - Peters, E.E. (1994). "Fractal Market Analysis." Wiley.
        - Peng, C.-K., Buldyrev, S.V., Havlin, S., Simons, M., Stanley, H.E. & Goldberger, A.L.
          (1994). "Mosaic organization of DNA nucleotides." Physical Review E, 49(2), 1685-1689.

        Also known as: generalized Hurst exponent, GHE, R/S analysis, rescaled range, DFA,
        detrended fluctuation analysis, multifractal Hurst exponent, trending or mean-reverting,
        mean reversion, persistence, long memory.

        Referenced against:
        - nolds, hurst_rs (rescaled range): https://github.com/CSchoel/nolds/blob/0.6.3/nolds/measures.py#L1203
        - nolds, dfa (detrended fluctuation): https://github.com/CSchoel/nolds/blob/0.6.3/nolds/measures.py#L1997

        Shares similar logic:
        - nolds, mfhurst_dm (generalized): https://github.com/CSchoel/nolds/blob/0.6.3/nolds/measures.py#L1695
        - antropy, detrended_fluctuation (detrended fluctuation): https://github.com/raphaelvallat/antropy/blob/v0.2.3/src/antropy/fractal.py#L369

        Args:
            max_lag (int, optional): The maximum lag of the generalized method, included: the
                regression runs over the lags 2 up to and including `max_lag`. Defaults to 20.
            method (str, optional): The estimation method. Options are: "generalized" (generalized Hurst
                exponent), "rescaled_range" (rescaled range), "detrended_fluctuation" (detrended fluctuation
                analysis). Defaults to "generalized".
            period (str | None, optional): The data frequency. Options are: "daily", "weekly", "monthly", "quarterly",
                "yearly". Defaults to None, which uses "daily".
            rolling (int | None, optional): The number of periods of a rolling window, at least the
                shortest window with a value: 5 for "generalized", and for the other methods the second
                smallest window size, 120 with the default window sizes. Defaults to None, which
                calculates one value over the full history of every asset.
            q (float | list[float], optional): The order of the moment of the generalized method,
                or a list of orders for a column per order. Defaults to 2.0.
            window_sizes (list[int] | None, optional): The window sizes of the rescaled range and
                the detrended fluctuation. Defaults to None, which uses 10 up to a tenth of the
                observations, growing by 20%.
            corrected (bool, optional): Whether to apply the Anis-Lloyd-Peters correction to the
                rescaled range. Defaults to True.
            order (int, optional): The degree of the polynomial removed from every window of the
                detrended fluctuation. Defaults to 1.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series | pd.DataFrame: The Hurst Exponent per asset, with `rolling` per date (a
            column per asset), and with a list of q a column "H(q)" per order.

        Notes:
        - The generalized method works on the log prices: their lagged differences are log returns
          over the lag, whose dispersion does not depend on the price level. The rescaled range
          and the detrended fluctuation work on the log returns, the differences of the log prices.
        - The rescaled range equals nolds' hurst_rs without its unbiased standard deviation and with a polynomial fit,
          and the detrended fluctuation nolds' dfa without overlapping windows and with a polynomial fit, for the same
          window sizes. nolds' mfhurst_dm samples the differences without overlap, detrends the series and averages
          over the largest lags 5 to 19, so it differs by about 0.01 to 0.03.

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
        | EURUSD    |  0.4718 |
        | GCUSD     |  0.4617 |
        | BTCUSD    |  0.5057 |
        | Benchmark |  0.4396 |
        """
        # The generalized estimator regresses the dispersion of lagged differences on the lag, which only identifies self-affinity on a level series: on log prices, as in Di Matteo et al. (2005), whose lagged differences are log returns over the lag. The rescaled range and the detrended fluctuation sum the changes themselves, so they take the log returns.  # noqa: E501
        # The smallest rolling window with a value: the generalized estimator needs two lags
        # (of 2 and 3 periods) with two differences each, five prices; the rescaled range and
        # the detrended fluctuation need two window sizes, which by default (10 and 12, at most
        # a tenth of the observations) takes 120 returns.
        if method == "generalized":
            minimum = 5
        elif window_sizes is None:
            minimum = 120
        else:
            sizes = sorted(
                size
                for size in window_sizes
                if size > (1 if method == "rescaled_range" else order + 1)
            )
            minimum = sizes[1] if len(sizes) > 1 else 2
        check_rolling(rolling, self._historical_data, period or "daily", minimum)
        log_prices = self._get_log_prices(period)

        if method == "generalized":
            orders = q if isinstance(q, list) else [q]
            hurst_exponents = {
                f"H({order_q:g})": calculate_per_window(
                    log_prices,
                    fractal_model.get_generalized_hurst_exponent,
                    rolling,
                    max_lag=max_lag,
                    q=order_q,
                )
                for order_q in orders
            }
            if not isinstance(q, list):
                hurst_exponent = next(iter(hurst_exponents.values()))
            elif rolling:
                hurst_exponent = pd.concat(hurst_exponents, axis=1)
            else:
                hurst_exponent = pd.DataFrame(hurst_exponents)
        elif method in ["rescaled_range", "detrended_fluctuation"]:
            hurst_exponent = calculate_per_window(
                self._get_log_prices(period, returns=True),
                (
                    fractal_model.get_rescaled_range_hurst_exponent
                    if method == "rescaled_range"
                    else fractal_model.get_detrended_fluctuation
                ),
                rolling,
                window_sizes=window_sizes,
                **(
                    {"corrected": corrected}
                    if method == "rescaled_range"
                    else {"order": order}
                ),
            )
        else:
            raise InvalidArgumentError(
                f"method must be one of 'generalized', 'rescaled_range' or 'detrended_fluctuation', "
                f"not {method!r}. Please pass one of these as method."
            )

        return apply_rounding(
            hurst_exponent, rounding if rounding is not None else self._rounding
        )

    def _get_log_prices(
        self, period: str | None, returns: bool = False
    ) -> pd.DataFrame:
        """
        The log of the adjusted close of `period` on the dates each ticker traded, or with
        `returns` their differences, the log returns, each ticker over its own trading days.
        """
        period = period if period else "daily"
        check_period(period, ["daily", "weekly", "monthly", "quarterly", "yearly"])

        log_prices = np.log(self._get_prices("Adj Close", period))
        if not returns:
            return log_prices

        # A day another ticker traded on is NaN here, so each ticker is differenced over its own trading days rather than losing the return around the gap.  # noqa: E501
        return log_prices.apply(lambda prices: prices.dropna().diff())

    @handle_errors
    def get_fractal_dimension(
        self,
        method: Literal["higuchi", "katz", "petrosian"] = "higuchi",
        period: str | None = None,
        rolling: int | None = None,
        kmax: int = 10,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Fractal Dimension of each asset's log price path: how rough the path is,
        between 1 for a straight line and 2 for a path that fills the plane.

        A random walk has a Higuchi dimension of 1.5, which is 2 - H for a self-affine path
        with Hurst exponent H: a trending path has a lower and a mean-reverting path a higher
        dimension. The Katz and Petrosian dimensions are quick measures of the same roughness
        on their own scales: Katz from how far the path gets for the distance it travels and
        Petrosian from how often it turns. Price paths are close to random walks, so the
        dimensions are mostly of use to compare assets or periods (with `rolling`).

        The methods are as follows (on the log prices x of N periods):

        - Higuchi (1988): L_m(k) = Sum(abs(x_(m + i*k) - x_(m + (i - 1)*k))) * (N - 1) /
          (floor((N - m) / k) * k) / k, L(k) = Mean(L_m(k)) over m = 1, ..., k, and
          FD = -slope of ln L(k) on ln k over k = 1, ..., kmax.
        - Katz (1988): FD = log10(n) / (log10(n) + log10(d / L)), with L = Sum(abs(x_(i+1) - x_i))
          the length of the path, d = Max(abs(x_i - x_1)) its largest distance from the start
          and n = N - 1 the number of steps.
        - Petrosian (1995): FD = log10(N) / (log10(N) + log10(N / (N + 0.4 * N_delta))), with
          N_delta the number of sign changes of the returns (a return of 0 interrupts one).

        For more information, see the following sources:

        - Higuchi, T. (1988). "Approach to an irregular time series on the basis of the fractal
          theory." Physica D, 31(2), 277-283.
        - Katz, M.J. (1988). "Fractals and the analysis of waveforms." Computers in Biology and
          Medicine, 18(3), 145-156.
        - Petrosian, A. (1995). "Kolmogorov complexity of finite sequences and recognition of
          different preictal EEG patterns." Proceedings of the Eighth IEEE Symposium on
          Computer-Based Medical Systems, 212-217.

        Also known as: HFD, KFD, PFD, Higuchi fractal dimension, fractal roughness.

        Referenced against:
        - antropy, higuchi_fd: https://github.com/raphaelvallat/antropy/blob/v0.2.3/src/antropy/fractal.py#L232
        - antropy, katz_fd: https://github.com/raphaelvallat/antropy/blob/v0.2.3/src/antropy/fractal.py#L104
        - antropy, petrosian_fd: https://github.com/raphaelvallat/antropy/blob/v0.2.3/src/antropy/fractal.py#L14

        Args:
            method (str, optional): The fractal dimension. Options are: "higuchi" (Higuchi), "katz" (Katz),
                "petrosian" (Petrosian). Defaults to "higuchi".
            period (str | None, optional): The data frequency. Options are: "daily", "weekly", "monthly", "quarterly",
                "yearly". Defaults to None, which uses "daily".
            rolling (int | None, optional): The number of periods of a rolling window, at least kmax + 2
                for "higuchi" (12 by default) and 3 for the others. Defaults to None, which calculates
                one value over the full history of every asset.
            kmax (int, optional): The largest step k of the Higuchi dimension. Defaults to 10.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series | pd.DataFrame: The fractal dimension per asset, or with `rolling` per date
            (a column per asset).

        Notes:
        - The dimensions equal antropy's higuchi_fd, katz_fd and petrosian_fd of the log prices.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["EURUSD", "GCUSD", "BTCUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_fractal_dimension()
        ```

        Which returns:

        |           |   Value |
        |:----------|--------:|
        | EURUSD    |  1.515  |
        | GCUSD     |  1.4903 |
        | BTCUSD    |  1.4827 |
        | Benchmark |  1.5085 |
        """
        # The Higuchi dimension needs more observations than its largest step plus one, the
        # Katz and Petrosian dimensions three.
        check_rolling(
            rolling,
            self._historical_data,
            period or "daily",
            kmax + 2 if method == "higuchi" else 3,
        )
        functions = {
            "higuchi": fractal_model.get_higuchi_fractal_dimension,
            "katz": fractal_model.get_katz_fractal_dimension,
            "petrosian": fractal_model.get_petrosian_fractal_dimension,
        }
        check_option("method", method, functions)

        fractal_dimension = calculate_per_window(
            self._get_log_prices(period),
            functions[method],
            rolling,
            **({"kmax": kmax} if method == "higuchi" else {}),
        )

        return apply_rounding(
            fractal_dimension, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_correlation_dimension(
        self,
        period: str | None = None,
        embedding: int = 10,
        delay: int = 1,
        radii: list[float] | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Correlation Dimension of each asset's log returns by the Grassberger-Procaccia
        algorithm: the dimension of the set that stretches of consecutive returns fill, low for a
        deterministic process.

        A deterministic, chaotic process fills a set of a low dimension however many returns a
        stretch holds; for noise the dimension grows with the embedding dimension (close to it
        for independent returns). Daily returns behave mostly like noise, so a value well below
        the embedding dimension is the exception, and comparing several embedding dimensions
        shows whether the dimension levels off.

        The formula is as follows:

        - X_i = [x_i, x_(i + tau), ..., x_(i + (m - 1) * tau)], the delay embedding of the log returns
        - C(r) = 2 / (M * (M - 1)) * Sum(1[||X_i - X_j|| < r], i < j), the share of the M * (M - 1) / 2
          pairs of vectors closer than r
        - D2 = slope of ln C(r) on ln r

        with ||.|| the Euclidean norm and, by default, the radii (3% apart) at which C(r) lies
        between 0.0001 and 0.01.

        For more information, see the following sources:

        - Grassberger, P. & Procaccia, I. (1983). "Characterization of strange attractors."
          Physical Review Letters, 50(5), 346-349.

        Also known as: D2, Grassberger-Procaccia dimension, correlation integral.

        Shares similar logic:
        - nolds, corr_dim: https://github.com/CSchoel/nolds/blob/0.6.3/nolds/measures.py#L1848

        Args:
            period (str | None, optional): The data frequency. Options are: "daily", "weekly", "monthly", "quarterly",
                "yearly". Defaults to None, which uses "daily".
            embedding (int, optional): The embedding dimension m. Defaults to 10.
            delay (int, optional): The number of periods between the returns of a vector.
                Defaults to 1.
            radii (list[float] | None, optional): The radii as multiples of the standard deviation
                (with n - 1 in the denominator) of the returns. Defaults to None, which uses the radii at which C(r) lies
                between 0.0001 and 0.01.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: The correlation dimension per asset.

        Notes:
        - Every pair of vectors is compared, so the time grows with the square of the number of
          returns: about 0.6 seconds for ten years of daily returns.
        - With the radii given, it equals a direct count of the pairs closer than every radius. nolds'
          corr_dim also counts every vector as a pair with itself, which pulls its slope towards
          0 where few pairs are close (0 for ten daily returns at its default radii of 0.1 to 0.5
          standard deviations, at which there are hardly any pairs).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["EURUSD", "GCUSD", "BTCUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_correlation_dimension(embedding=4)
        ```

        Which returns:

        |           |   Value |
        |:----------|--------:|
        | EURUSD    |  4.0576 |
        | GCUSD     |  3.6576 |
        | BTCUSD    |  3.8436 |
        | Benchmark |  3.8736 |
        """
        correlation_dimension = calculate_per_window(
            self._get_log_prices(period, returns=True),
            fractal_model.get_correlation_dimension,
            embedding=embedding,
            delay=delay,
            radii=radii,
        )

        return apply_rounding(
            correlation_dimension, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_lyapunov_exponent(
        self,
        method: Literal["rosenstein", "eckmann"] = "rosenstein",
        period: str | None = None,
        embedding: int = 10,
        delay: int = 1,
        min_separation: int | None = None,
        trajectory_length: int = 20,
        matrix_dimension: int = 4,
        min_neighbors: int | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Lyapunov Exponent of each asset's log returns: the rate, per period, at which
        nearby paths of the reconstructed dynamics move apart, positive for a chaotic process.

        A positive largest exponent points to chaos, a sensitive dependence on the starting
        point that makes the series unpredictable beyond a few periods even if it were
        deterministic; a value around 0 or below to a regular process. For noise, such as
        daily returns, the estimate is not 0 but depends on the embedding, so the exponent is
        mostly of use to compare assets or periods with each other.

        The methods are as follows (X_i the delay embedding of the log returns x):

        - Rosenstein, Collins and De Luca (1993): every vector X_i is paired with its nearest
          neighbour X_j (Euclidean) at least `min_separation` periods away, y(k) = Mean(ln
          ||X_(i+k) - X_(j+k)||) for k = 0, ..., trajectory_length - 1, and lambda_1 = slope of
          y(k) on k, the largest exponent.
        - Eckmann, Kamphorst, Ruelle and Ciliberto (1986): with m = (embedding - 1) /
          (matrix_dimension - 1), a linear map T_i from X_i = [x_i, x_(i + m), ...] to X_(i+m)
          is fitted by least squares on the min_neighbors nearest neighbours (Chebyshev); the
          maps are multiplied along the path with a QR decomposition T_i Q_(i-m) = Q_i R_i and
          lambda_k = Mean(ln abs(R_i,kk)) / m, the `matrix_dimension` largest exponents.

        For more information, see the following sources:

        - Rosenstein, M.T., Collins, J.J. & De Luca, C.J. (1993). "A practical method for
          calculating largest Lyapunov exponents from small data sets." Physica D, 65(1-2),
          117-134.
        - Eckmann, J.-P., Kamphorst, S.O., Ruelle, D. & Ciliberto, S. (1986). "Liapunov exponents
          from time series." Physical Review A, 34(6), 4971-4979.

        Also known as: largest Lyapunov exponent, LLE, Lyapunov spectrum, chaos exponent.

        Referenced against:
        - nolds, lyap_r (Rosenstein): https://github.com/CSchoel/nolds/blob/0.6.3/nolds/measures.py#L117
        - nolds, lyap_e (Eckmann): https://github.com/CSchoel/nolds/blob/0.6.3/nolds/measures.py#L400

        Args:
            method (str, optional): The algorithm. Options are: "rosenstein" (Rosenstein, Collins and De
                Luca, the largest exponent), "eckmann" (Eckmann, Kamphorst, Ruelle and Ciliberto, the
                spectrum). Defaults to "rosenstein".
            period (str | None, optional): The data frequency. Options are: "daily", "weekly", "monthly", "quarterly",
                "yearly". Defaults to None, which uses "daily".
            embedding (int, optional): The embedding dimension; for "eckmann", embedding - 1 must be
                a multiple of matrix_dimension - 1. Defaults to 10.
            delay (int, optional): The number of periods between the returns of a vector of the
                Rosenstein method. Defaults to 1.
            min_separation (int | None, optional): The smallest number of periods between a vector
                and its neighbour. Defaults to None, which uses the mean period of the returns
                (one over the power-weighted mean frequency of the periodogram, rounded up) for
                "rosenstein" and 0 for "eckmann".
            trajectory_length (int, optional): The number of periods the pairs of the Rosenstein
                method are followed. Defaults to 20.
            matrix_dimension (int, optional): The number of exponents of the Eckmann method.
                Defaults to 4.
            min_neighbors (int | None, optional): The number of neighbours of every vector of the
                Eckmann method. Defaults to None, which uses min(2 * matrix_dimension,
                matrix_dimension + 4).
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series | pd.DataFrame: The largest exponent per asset, or for "eckmann" the
            exponents "Exponent 1" to "Exponent d" per asset (a row per asset), per period.

        Notes:
        - Every vector is compared with every other one, so the time grows with the square of the
          number of returns: about 1 second for the Rosenstein and 2 seconds for the Eckmann
          method on ten years of daily returns.
        - The Rosenstein method equals nolds' lyap_r with the same delay and minimum separation and a
          polynomial fit.
          The Eckmann method multiplies the maps along each path i, i + m, i + 2m, ..., where
          nolds' lyap_e multiplies the maps of consecutive vectors; with m = 1 both are equal, and
          on the Hénon map (largest exponent 0.42) the paths here give 0.42 for m = 2 and 3, where
          nolds gives 0.45 and 0.47.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["EURUSD", "GCUSD", "BTCUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_lyapunov_exponent()
        ```

        Which returns:

        |           |   Value |
        |:----------|--------:|
        | EURUSD    |  0.0388 |
        | GCUSD     |  0.0398 |
        | BTCUSD    |  0.0395 |
        | Benchmark |  0.0409 |
        """
        log_returns = self._get_log_prices(period, returns=True)

        if method == "rosenstein":
            lyapunov_exponent = calculate_per_window(
                log_returns,
                fractal_model.get_largest_lyapunov_exponent,
                embedding=embedding,
                delay=delay,
                min_separation=min_separation,
                trajectory_length=trajectory_length,
            )
        elif method == "eckmann":
            lyapunov_exponent = calculate_per_window(
                log_returns,
                fractal_model.get_lyapunov_spectrum,
                embedding=embedding,
                matrix_dimension=matrix_dimension,
                min_neighbors=min_neighbors,
                min_separation=min_separation if min_separation is not None else 0,
            )
        else:
            raise InvalidArgumentError(
                f"method must be one of 'rosenstein' or 'eckmann', "
                f"not {method!r}. Please pass one of these as method."
            )

        return apply_rounding(
            lyapunov_exponent, rounding if rounding is not None else self._rounding
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
        Calculate the Lower (or Upper) Partial Moment: how often and how far the returns of an
        investment portfolio or asset fall below (or rise above) a minimum acceptable return.

        A partial moment measures only one side of the return distribution: the lower partial moment
        averages the shortfalls below a minimum acceptable return (MAR) raised to a power, the order, and
        the upper partial moment does the same for the returns above it. The order sets what is measured:
        order 0 is how often the returns fall below the MAR (the downside frequency), order 1 how far they
        fall below it on average (the downside potential, the expected shortfall below the target), order 2
        the downside variance behind the Sortino Ratio and higher orders the weight of the extreme losses.
        Investors who care about losses rather than about deviations in either direction use them in place
        of the variance, and the ratio of an upper to a lower partial moment is the Omega Ratio (order 1)
        and the Upside Potential Ratio's building block.

        The formula is as follows:

        - Lower Partial Moment = Σ max(MAR - Return_t, 0)^order / n
        - Upper Partial Moment = Σ max(Return_t - MAR, 0)^order / n

        Where n is the number of all returns, so the returns on the other side of the MAR count as zero
        rather than being left out. Of order 0 the moment counts the returns strictly below (or above)
        the MAR, as a share of all returns. Raised to the power 1 / order (its root), order 2 becomes the
        Downside Deviation (the lower side) or the Upside Deviation (the upper side). With the mean of
        each sample as the MAR it gives the semi-moments around the mean (order 2: the semi-variance with
        n in the denominator).

        For more information, see the following sources:

        - Bawa, V.S. (1975). "Optimal Rules for Ordering Uncertain Prospects." Journal of Financial
        Economics, 2(1), 95-121.
        - Fishburn, P.C. (1977). "Mean-Risk Analysis with Risk Associated with Below-Target Returns."
        The American Economic Review, 67(2), 116-126.
        - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution." John Wiley &
        Sons, chapter 4 (Downside Potential, Downside and Upside Frequency, Upside Potential).

        Also known as: LPM, UPM, downside potential, downside frequency, upside frequency, upside potential,
        shortfall probability, semi-moment.

        Referenced against:
        - Riskfolio-Lib, LPM: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.LPM
        - Riskfolio-Lib, SemiKurtosis: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.SemiKurtosis
        - skfolio, first_lower_partial_moment: https://skfolio.org/generated/skfolio.measures.first_lower_partial_moment.html
        - skfolio, fourth_lower_partial_moment: https://skfolio.org/generated/skfolio.measures.fourth_lower_partial_moment.html

        Shares similar logic:
        - PerformanceAnalytics (R), DownsideFrequency: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/DownsideFrequency.R#L35
        - PerformanceAnalytics (R), UpsideFrequency: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/UpsideFrequency.R#L35
        - PerformanceAnalytics (R), UpsideRisk: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/UpsideRisk.R#L54
        - PerformanceAnalytics (R), DownsidePotential: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/DownsideDeviation.R#L166
        - PerformanceAnalytics (R), lpm: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/lpm.R#L23

        Args:
            period (str, optional): The data frequency for returns. Options are: "daily", "weekly", "monthly",
                "quarterly", "yearly". Defaults to "daily".
            order (float, optional): The order of the moment, 0 or higher: 0 for the frequency, 1 for the
            potential, 2 for the variance and higher orders for the extremes. Defaults to 1.
            minimum_acceptable_return (float | str, optional): The minimum acceptable return (MAR) per
            observation of the chosen `period` (e.g. per day), or "mean" for the mean of each sample.
            Defaults to 0.0.
            side (str, optional): The side of the MAR. Options are: "lower" (the returns below the MAR),
            "upper" (the returns above it). Defaults to "lower".
            root (bool, optional): Whether to raise the moment to the power 1 / order, which needs an
            order above 0. Defaults to False.
            within_period (bool, optional): Whether to calculate the Partial Moment within the specified period or
            for the entire period. Thus whether to look at the Partial Moment within a specific year (if period =
            'yearly') or look at the entirety of all years. Defaults to False.
            Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rolling (int, optional): The rolling window size to use for the calculation. If set, the Partial Moment
            is calculated over a rolling window of this many periods across the full return history
            instead of per `period`. At least 2. Defaults to None.
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
        - With growth, the method calculates the growth of the Partial Moment values using the
        specified `lag`.
        - The lower partial moment of order 1 agrees with PerformanceAnalytics' DownsidePotential and
        Riskfolio-Lib's LPM of order 1, order 0 equals DownsideFrequency (UpsideFrequency on the upper side)
        and the upper side of order 1 and 2 equals the potential and the variance of UpsideRisk.
        - With the mean as the MAR, order 1 and 4 agree with skfolio's first_lower_partial_moment and
        fourth_lower_partial_moment. Riskfolio-Lib's LPM of order 2 divides by n - 1 instead of n and
        takes the root, and PerformanceAnalytics' lpm divides by the number of returns below the
        threshold. Riskfolio-Lib's SemiKurtosis is the square of the root of order 4 around the mean, and
        its EvenSemiMoment of p the (order 2p) moment to the power 1 / p.

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
        Calculate a Central Moment: the average deviation of the returns of an investment portfolio or
        asset from their mean raised to a power, the spread, asymmetry or tails in the returns' own units.

        A central moment is the average deviation from the mean return raised to a power, the order. The
        second central moment is the (population) variance, the third measures the asymmetry of the return
        distribution in squared-and-cubed return units (negative when the large deviations are losses) and
        the fourth the weight of the tails. Unlike the Skewness and Kurtosis, which divide by a power of the
        standard deviation, they keep the scale of the returns, so a volatile asset has larger moments, and
        they are the building blocks of the moment-based risk measures of portfolio optimization, such as
        the square-root kurtosis.

        The formula is as follows:

        - Central Moment = Σ (Return_t - Mean)^order / n

        Where Mean is the average return and n the number of returns of the sample (the population moment,
        without a bias adjustment).

        Also known as: moment about the mean, third central moment, fourth central moment, square-root
        kurtosis (its square root for order 4).

        Referenced against:
        - skfolio, third_central_moment: https://skfolio.org/generated/skfolio.measures.third_central_moment.html
        - skfolio, fourth_central_moment: https://skfolio.org/generated/skfolio.measures.fourth_central_moment.html
        - Riskfolio-Lib, Kurtosis: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.Kurtosis

        Shares similar logic:
        - Riskfolio-Lib, EvenMoment: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.EvenMoment
        - PerformanceAnalytics (R), centeredmoment: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/CoMoments.R#L127

        Args:
            period (str, optional): The data frequency for returns. Options are: "daily", "weekly", "monthly",
                "quarterly", "yearly". Defaults to "daily".
            order (int, optional): The order of the moment, an integer of 2 or higher. Defaults to 3.
            within_period (bool, optional): Whether to calculate the Central Moment within the specified period or
            for the entire period. Thus whether to look at the Central Moment within a specific year (if period =
            'yearly') or look at the entirety of all years. Defaults to False.
            Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rolling (int, optional): The rolling window size to use for the calculation. If set, the Central Moment
            is calculated over a rolling window of this many periods across the full return history
            instead of per `period`. At least 2. Defaults to None.
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
        - With growth, the method calculates the growth of the Central Moment values using the
        specified `lag`.
        - Order 3 and 4 agree with skfolio's third_central_moment and fourth_central_moment and
        PerformanceAnalytics' centeredmoment. Riskfolio-Lib's Kurtosis is the square root of order 4 and
        its EvenMoment of p the (order 2p) moment to the power 1 / p.
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
        Calculate the Worst Realization: the lowest return of an investment portfolio or asset, the loss
        on the worst day (or week, month) of the sample.

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

        Also known as: worst return, minimax risk, maximum loss.

        Referenced against:
        - Riskfolio-Lib, WR: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.WR
        - skfolio, worst_realization: https://skfolio.org/generated/skfolio.measures.worst_realization.html

        Args:
            period (str, optional): The data frequency for returns. Options are: "daily", "weekly", "monthly",
                "quarterly", "yearly". Defaults to "daily".
            within_period (bool, optional): Whether to calculate the Worst Realization within the specified period or
            for the entire period. Thus whether to look at the Worst Realization within a specific year (if period =
            'yearly') or look at the entirety of all years. Defaults to False.
            Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rolling (int, optional): The rolling window size to use for the calculation. If set, the Worst Realization
            is calculated over a rolling window of this many periods across the full return history
            instead of per `period`. At least 2. Defaults to None.
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
        - With growth, the method calculates the growth of the Worst Realization values using the
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
        Calculate the Gini Mean Difference (GMD): how far apart two returns of an investment portfolio or
        asset lie on average, a measure of dispersion that does not square the deviations.

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

        Also known as: GMD, Gini's mean difference, mean absolute difference.

        Referenced against:
        - Riskfolio-Lib, GMD: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.GMD
        - skfolio, gini_mean_difference: https://skfolio.org/generated/skfolio.measures.gini_mean_difference.html

        Shares similar logic:
        - Riskfolio-Lib, owa_gmd: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/auxiliary.html#OwaWeights.owa_gmd

        Args:
            period (str, optional): The data frequency for returns. Options are: "daily", "weekly", "monthly",
                "quarterly", "yearly". Defaults to "daily".
            within_period (bool, optional): Whether to calculate the Gini Mean Difference within the specified period or
            for the entire period. Thus whether to look at the Gini Mean Difference within a specific year (if period =
            'yearly') or look at the entirety of all years. Defaults to False.
            Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rolling (int, optional): The rolling window size to use for the calculation. If set, the Gini Mean Difference
            is calculated over a rolling window of this many periods across the full return history
            instead of per `period`. At least 2. Defaults to None.
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
        - With growth, the method calculates the growth of the Gini Mean Difference values using the
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
        points: int = 100,
        within_period: bool = False,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Tail Gini: the loss in the worst tail of the returns of an investment portfolio or
        asset, weighing the worst returns within that tail most.

        The Tail Gini of Furman, Wang and Zitikis (2017) measures the variability within the loss tail,
        the alpha share of the worst returns. Cajas (2021) writes it as a weighted average of the
        Conditional Value at Risk at tail probabilities from 0.0001 up to alpha, which puts more weight on
        the worst returns within the tail than the Conditional Value at Risk at alpha, which weighs every
        return in the tail equally. It is therefore always at or below the Conditional Value at Risk and
        above the Worst Realization.

        The formula is as follows (with the default alpha of 0.05 and 100 points):

        - Tail Gini = Σ_j c_j * CVaR(alpha_j)
        - c_1 = alpha_2 * alpha_1 / alpha^2, c_j = (alpha_(j+1) - alpha_(j-1)) * alpha_j / alpha^2 and
          c_m = (alpha_m - alpha_(m-1)) / alpha

        Where alpha_1 < ... < alpha_m = alpha are m = `points` evenly spaced tail probabilities from
        0.0001 to alpha and CVaR(alpha_j) is the average of the alpha_j share of the lowest returns, the
        return at the boundary counting with its fraction (the ordered weighted average of Cajas 2021).
        The result is a return, negative for a loss, as the Conditional Value at Risk of the Toolkit is.

        For more information, see the following sources:

        - Furman, E., Wang, R. and Zitikis, R. (2017). "Gini-type Measures of Risk and Variability: Gini
        Shortfall, Capital Allocations, and Heavy-tailed Risks." Journal of Banking & Finance, 83, 70-84.
        - Cajas, D. (2021). "OWA Portfolio Optimization: A Disciplined Convex Programming Framework."
        SSRN Electronic Journal.

        Also known as: TG, Gini shortfall, tail Gini risk.

        Referenced against:
        - Riskfolio-Lib, TG: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.TG

        Shares similar logic:
        - Riskfolio-Lib, owa_tg: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/auxiliary.html#OwaWeights.owa_tg

        Args:
            period (str, optional): The data frequency for returns. Options are: "daily", "weekly", "monthly",
                "quarterly", "yearly". Defaults to "daily".
            alpha (float, optional): The tail probability, between 0.0001 and 1. Defaults to 0.05.
            points (int, optional): The number of tail probabilities the Conditional Value at Risk
            is averaged over. Defaults to 100.
            within_period (bool, optional): Whether to calculate the Tail Gini within the specified period or
            for the entire period. Thus whether to look at the Tail Gini within a specific year (if period =
            'yearly') or look at the entirety of all years. Defaults to False.
            Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rolling (int, optional): The rolling window size to use for the calculation. If set, the Tail Gini
            is calculated over a rolling window of this many periods across the full return history
            instead of per `period`. At least 2. Defaults to None.
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
        - With growth, the method calculates the growth of the Tail Gini values using the
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
        validate_alpha(alpha)
        period = period if period else "daily"
        returns = self._get_measure_returns(period, within_period, rolling, growth)

        result = dispersion_model.get_tail_gini(
            returns, alpha=alpha, points=points, window_size=rolling
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
        points: int = 100,
        within_period: bool = False,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate a Range measure: how far the gain tail of the returns of an investment portfolio or
        asset lies from its loss tail.

        The Range is the distance between the best and the worst return. Cajas (2021) generalizes it to
        the tails: the VaR Range is the distance between the 1 - beta and the alpha quantile, the CVaR
        Range between the average return in the best beta share and in the worst alpha share of the
        returns and the Tail Gini Range between the Tail Gini of the gains and of the losses. They measure
        the width of the distribution where it matters most, without the assumption of symmetry behind the
        standard deviation, and with beta different from alpha they weigh the gains and the losses
        differently. The VaR Range at alpha = beta = 0.25 is the interquartile range. The EVaR Range does the
        same with the historic Entropic Value at Risk, the most severe of these tail measures.

        The formula is as follows:

        - Range = max(Return) - min(Return)
        - VaR Range = Quantile(Return, 1 - beta) - Quantile(Return, alpha)
        - CVaR Range = mean(Return >= Quantile(Return, 1 - beta)) - mean(Return <= Quantile(Return, alpha))
        - Tail Gini Range = Tail Gini of the gains (beta) - Tail Gini of the losses (alpha)
        - EVaR Range = EVaR of the gains (beta) - EVaR of the losses (alpha)

        Where the quantiles are interpolated linearly and the CVaR is the average of the returns beyond the
        quantile, as in the historic `get_value_at_risk` and `get_conditional_value_at_risk`, the Tail
        Gini is that of `get_tail_gini` and the EVaR the historic one of `get_entropic_value_at_risk`,
        each measured on the negated returns for the gains. The result is a
        positive decimal.

        For more information, see the following sources:

        - Cajas, D. (2021). "OWA Portfolio Optimization: A Disciplined Convex Programming Framework."
        SSRN Electronic Journal.

        Also known as: return range, VaR range, CVaR range, Tail Gini range, EVaR range, tail spread.

        Referenced against:
        - Riskfolio-Lib, RG (Range): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.RG
        - Riskfolio-Lib, TGRG (Tail Gini Range): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.TGRG
        - Riskfolio-Lib, EVRG (EVaR Range): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.EVRG

        Shares similar logic:
        - Riskfolio-Lib, VRG (VaR Range): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.VRG
        - Riskfolio-Lib, CVRG (CVaR Range): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.CVRG

        Args:
            period (str, optional): The data frequency for returns. Options are: "daily", "weekly", "monthly",
                "quarterly", "yearly". Defaults to "daily".
            measure (str, optional): The range to calculate. Options are: "range" (Range), "var" (VaR
            Range), "cvar" (CVaR Range), "tail_gini" (Tail Gini Range), "evar" (EVaR Range). Defaults to
            "range".
            alpha (float, optional): The tail probability of the losses, between 0 and 1 (between 0.0001
            and 1 for the Tail Gini). Defaults to 0.05.
            beta (float | None, optional): The tail probability of the gains. Defaults to None, which uses
            alpha.
            points (int, optional): The number of tail probabilities of the Tail Gini Range.
            Defaults to 100.
            within_period (bool, optional): Whether to calculate the Risk Range within the specified period or
            for the entire period. Thus whether to look at the Risk Range within a specific year (if period =
            'yearly') or look at the entirety of all years. Defaults to False.
            Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rolling (int, optional): The rolling window size to use for the calculation. If set, the Risk Range
            is calculated over a rolling window of this many periods across the full return history
            instead of per `period`. At least 2. Defaults to None.
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
        - With growth, the method calculates the growth of the Risk Range values using the
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
        validate_alpha(alpha)
        period = period if period else "daily"
        returns = self._get_measure_returns(period, within_period, rolling, growth)

        result = dispersion_model.get_risk_range(
            returns,
            measure=measure,
            alpha=alpha,
            beta=beta,
            points=points,
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
        Calculate an L-moment, or an L-moment ratio: the location, spread, asymmetry or tails of the
        returns of an investment portfolio or asset, robust to a few extreme returns.

        L-moments (Hosking 1990) describe the location, scale and shape of a distribution as the ordinary
        moments do, but as linear combinations of the sorted returns rather than powers of the deviations.
        This makes them far less sensitive to a few extreme returns and well defined whenever the mean is:
        the L-skewness and L-kurtosis of a sample are bounded and nearly unbiased, where the sample skewness
        and kurtosis of fat-tailed returns are dominated by the largest move. The first L-moment is the
        mean, the second a measure of scale (half the Gini Mean Difference), and the ratios of the third and
        fourth to the second, the L-skewness and L-kurtosis, measure the asymmetry and the tails. For a
        normal distribution the L-skewness is 0 and the L-kurtosis 0.1226.

        The formula is as follows:

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

        Also known as: L-moments, L-scale, L-skewness, L-kurtosis, L-CV, linear moments.

        Referenced against:
        - Riskfolio-Lib, L_Moment: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.L_Moment
        - scipy, lmoment: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lmoment.html

        Shares similar logic:
        - Riskfolio-Lib, owa_l_moment: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/auxiliary.html#OwaWeights.owa_l_moment

        Args:
            period (str, optional): The data frequency for returns. Options are: "daily", "weekly", "monthly",
                "quarterly", "yearly". Defaults to "daily".
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
            instead of per `period`. At least 2. Defaults to None.
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
        - With growth, the method calculates the growth of the L-moment values using the
        specified `lag`.
        - The L-moments agree with Riskfolio-Lib's L_Moment and scipy's lmoment, and the ratios with the
        standardized L-moments of scipy's lmoment.
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
        Calculate the L-moment Convex Risk Measure: one risk figure for the spread, skew and tails of the
        returns of an investment portfolio or asset, robust to outliers.

        Cajas (2023) combines the L-moments of order 2 up to `order` into one convex risk measure: a
        weighted sum of the L-scale, minus the L-moment of order 3 (so a negative skew adds risk), plus that
        of order 4 and so on, with weights that are non-negative, sum to 1 and decrease with the order. The
        weights are chosen so that the ordered weighted average they imply puts at least as much weight on
        every worse return, which keeps the measure convex and therefore usable in portfolio optimization.
        It captures the dispersion, the skew and the tails of the returns in one figure that is as robust to
        outliers as the L-moments themselves.

        The formula is as follows:

        - L-moment Convex Risk Measure = Σ_k φ_k * (-1)^k * λ_k for k = 2, ..., order
        - Σ φ_k = 1, 0 <= φ_k <= max_weight and φ_k >= φ_(k+1)

        Where λ_k is the L-moment of order k (`get_l_moment`) and the φ_k follow from the method: the minimum squared
        differences method minimizes the squared differences between the neighbouring weights of the ordered weighted
        average, the minimum sum of squares method the sum of its squared weights and the constant relative risk
        aversion method sets φ_k in proportion to Π_(i=1..k-1) (risk_aversion + i - 1) / k!, after which the ordered
        weights are made non-decreasing. The result is a positive decimal.

        For more information, see the following sources:

        - Cajas, D. (2023). "Higher Order Moment Portfolio Optimization with L-Moments." SSRN 4393155.
        - Hosking, J.R.M. (1990). "L-moments: Analysis and Estimation of Distributions using Linear
        Combinations of Order Statistics." Journal of the Royal Statistical Society: Series B, 52(1),
        105-124.

        Also known as: L-moment CRM, L-moment based convex risk measure.

        Referenced against:
        - Riskfolio-Lib, L_Moment_CRM (constant relative risk aversion): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.L_Moment_CRM

        Shares similar logic:
        - Riskfolio-Lib, owa_l_moment_crm: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/auxiliary.html#OwaWeights.owa_l_moment_crm

        Args:
            period (str, optional): The data frequency for returns. Options are: "daily", "weekly", "monthly",
                "quarterly", "yearly". Defaults to "daily".
            order (int, optional): The highest L-moment order, an integer of 2 or higher. Defaults to 4.
            method (str, optional): How the L-moments are weighted. Options are: "msd" (minimum squared
            differences), "mss" (minimum sum of squares), "crra" (constant relative risk aversion).
            Defaults to "msd".
            risk_aversion (float, optional): The risk aversion of the constant relative risk aversion
            method, between 0 and 1. Defaults to 0.5.
            max_weight (float, optional): The largest weight of one L-moment for "msd" and "mss", between
            0 and 1. Defaults to 0.5.
            within_period (bool, optional): Whether to calculate the L-moment Convex Risk Measure within the specified
            period or for the entire period. Thus whether to look at the L-moment Convex Risk Measure within a specific
            year (if period = 'yearly') or look at the entirety of all years. Defaults to False.
            Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rolling (int, optional): The rolling window size to use for the calculation. If set, the
            L-moment Convex Risk Measure is calculated over a rolling window of this many periods across the full return
            history instead of per `period`. At least 2. Defaults to None.
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
        - With growth, the method calculates the growth of the L-moment Convex Risk Measure values using
        the specified `lag`.
        - With constant relative risk aversion the result agrees with Riskfolio-Lib's L_Moment_CRM. For the
        minimum squared differences and minimum sum of squares methods the weights
        solve a small convex program over the order - 1 weights with scipy (SLSQP) where Riskfolio-Lib uses
        cvxpy with the Clarabel solver; the Toolkit's weights meet the constraints exactly and match a
        tightly solved cvxpy program, while Clarabel's default tolerances leave Riskfolio-Lib's weights
        slightly off, so the results differ in the fourth significant digit for the minimum squared
        differences and the second for the minimum sum of squares. Riskfolio-Lib's maximum entropy method is
        left out: the entropy it maximizes is
        that of a bound which is the same for every set of weights, so its result depends on the solver.
        - With an order of 3 and a maximum weight of 0.5 no weights meet the constraints, which gives an error.

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
        Calculate the Skewness-Kurtosis Ratio: how lopsided the returns of an investment portfolio or
        asset are relative to how heavy their tails are, higher being better.

        The Skewness-Kurtosis Ratio (Bacon 2008) divides the skewness by the kurtosis, combining the two
        shape statistics an investor in fat-tailed assets cares about into one number: a positive skew is
        attractive and fat tails are not, so a higher ratio is better. It is used to rank funds whose
        returns are far from normal, where the Sharpe Ratio's reliance on the standard deviation misleads.

        The formula is as follows:

        - Skewness-Kurtosis Ratio = Skewness / Kurtosis

        Where the Skewness is that of `get_skewness` and the Kurtosis that of `get_kurtosis`, Pearson's
        kurtosis (3 for a normal distribution), both the bias-adjusted
        sample estimates.

        For more information, see the following sources:

        - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution." John Wiley &
        Sons, chapter 4.

        Also known as: skewness to kurtosis ratio.

        Shares similar logic:
        - PerformanceAnalytics (R), SkewnessKurtosisRatio: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/SkewnessKurtosisRatio.R#L31

        Args:
            period (str, optional): The data frequency for returns. Options are: "daily", "weekly", "monthly",
                "quarterly", "yearly". Defaults to "daily".
            within_period (bool, optional): Whether to calculate the Skewness-Kurtosis Ratio within the specified
            period or for the entire period. Thus whether to look at the Skewness-Kurtosis Ratio within a specific
            year (if period = 'yearly') or look at the entirety of all years. Defaults to False.
            Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rolling (int, optional): The rolling window size to use for the calculation. If set, the
            Skewness-Kurtosis Ratio is calculated over a rolling window of this many periods across the full return
            history instead of per `period`. At least 4. Defaults to None.
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
        - With growth, the method calculates the growth of the Skewness-Kurtosis Ratio values using
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
        returns = self._get_measure_returns(period, within_period, rolling, growth, 4)

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
