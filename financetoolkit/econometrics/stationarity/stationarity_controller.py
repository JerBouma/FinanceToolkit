"""Stationarity and Cointegration Module"""

__docformat__ = "google"

import itertools

import pandas as pd

from financetoolkit.econometrics.base.base_controller import EconometricsBase
from financetoolkit.econometrics.stationarity import (
    causality_model,
    cointegration_model,
    unitroot_model,
)
from financetoolkit.helpers import handle_portfolio
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import apply_rounding

# ruff: noqa: E501

logger = get_logger()


@filter_shown_columns
@track_sources
class StationarityEconometrics(EconometricsBase):
    """
    Unit root tests (ADF, KPSS, Phillips-Perron, Zivot-Andrews), Engle-Granger and Johansen cointegration and Granger causality.
    """

    def _all_ordered_pairs(
        self, include_benchmark: bool = False
    ) -> list[tuple[str, str]]:
        """
        Every ordered pair `(ticker_a, ticker_b)`, `ticker_a != ticker_b`, of
        `self._tickers` -- used by the directional pairwise methods below (Granger
        causality, Engle-Granger cointegration), where swapping the pair changes
        which ticker is treated as dependent.
        """
        return list(
            itertools.permutations(
                self._get_tickers(include_benchmark=include_benchmark), 2
            )
        )

    @handle_portfolio
    @handle_errors
    def get_augmented_dickey_fuller(
        self,
        period: str | None = None,
        column: str = "Adj Close",
        max_lag: int | None = None,
        regression: str = "c",
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Augmented Dickey-Fuller (ADF) test for a unit root, per asset.

        The test regresses the first difference of the series on its own lagged level and
        `p` lags of its own first difference. The null hypothesis is that the series has a
        unit root (is a random walk, not mean-reverting); the alternative is that it is
        stationary. This is a standard first step before modeling a price series or spread
        with mean-reverting methods, since those methods assume stationarity.

        The formula is as follows:

        - ΔYₜ = α + γ·Yₜ₋₁ + Σ(i=1..p) δᵢ·ΔYₜ₋ᵢ + εₜ, with Yₜ the `column` level (Adj Close by default)
          and regression="c" (the default); "ct" adds a trend β·t and "n" drops α
        - ADF = γ̂ / SE(γ̂), compared with MacKinnon (1994, 2010) critical values and p-values
        - p = the number of lags that minimises the AIC, searched from 0 up to `max_lag`, which defaults to
          min(ceil(12·(n / 100)^(1/4)), (n - 5) // 3)

        For more information about the method, see the following paper:

        - Dickey, D.A. and Fuller, W.A. (1979). "Distribution of the Estimators for
        Autoregressive Time Series with a Unit Root." Journal of the American Statistical
        Association, 74(366a), 427-431.

        Also known as: ADF test, unit root test, stationarity test.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to test. Defaults to "Adj Close".
            max_lag (int, optional): The maximum number of lagged differences to consider. Defaults to
            the Schwert (1989) rule of thumb, capped at (n - 5) // 3 for short samples.
            regression (str, optional): Which deterministic terms to include, one of "n" (none), "c"
            (constant) or "ct" (constant and trend). Defaults to "c".
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
            assets tested. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The ADF statistic, its p-value, lags used, observations used, 1%/5%/10%
            critical values, and whether the unit root is rejected at the 5% level, per asset.
            An asset with fewer than 10 observations has NaN values.

        Notes:
        - The method retrieves historical price data based on the specified `period` and runs the ADF
        test for each asset in the Toolkit instance.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["EURUSD", "USDJPY", "GBPUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_augmented_dickey_fuller()
        ```

        Which returns:

        |                       |      EURUSD |     USDJPY |      GBPUSD |
        |:----------------------|------------:|-----------:|------------:|
        | ADF Statistic         |   -1.83568  |   -1.68762 |   -1.81162  |
        | P-Value               |    0.362896 |    0.4375  |    0.374708 |
        | Lags Used             |    5        |    0       |    2        |
        | Observations          | 1345        | 1351       | 1349        |
        | Critical Value 1%     |   -3.43522  |   -3.4352  |   -3.43521  |
        | Critical Value 5%     |   -2.86369  |   -2.86368 |   -2.86368  |
        | Critical Value 10%    |   -2.56792  |   -2.56791 |   -2.56791  |
        | Reject Unit Root (5%) |    0        |    0       |    0        |
        """
        period = period if period else "daily"
        prices = self._filter_benchmark(
            self._get_series_column(period, column),
            include_benchmark=include_benchmark,
        )

        result = pd.DataFrame(
            {
                ticker: unitroot_model.get_augmented_dickey_fuller(
                    prices[ticker], max_lag=max_lag, regression=regression
                )
                for ticker in prices.columns
            }
        )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_kpss_test(
        self,
        period: str | None = None,
        column: str = "Adj Close",
        regression: str = "c",
        lags: int | None = None,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test for stationarity, per asset.

        KPSS is the natural complement to the Augmented Dickey-Fuller test: where the ADF null
        hypothesis is that the series HAS a unit root, the KPSS null hypothesis is that the
        series IS stationary, with a unit root as the alternative. Running both together is
        standard practice to triangulate a confident conclusion -- ADF rejecting a unit root
        and KPSS failing to reject stationarity together give a confident stationarity
        conclusion, while the two tests disagreeing flags an ambiguous case.

        The formula is as follows:

        - Yₜ = α + eₜ fitted by OLS on the `column` level (Adj Close by default), with regression="c"
          (the default); "ct" adds a trend β·t
        - Sₜ = Σ(i=1..t) êᵢ
        - KPSS = Σ(t=1..n) Sₜ² / (n²·λ²)
        - λ² = (1 / n)·Σêₜ² + (2 / n)·Σ(s=1..l) (1 - s / (l + 1))·Σ(t=s+1..n) êₜ·êₜ₋ₛ, with l = `lags` or,
          by default, the Hobijn, Franses and Ooms (2004) automatic bandwidth
        - The p-value is interpolated from the KPSS (1992) critical value table, so it is bounded
          between 0.01 and 0.10

        For more information about the method, see the following paper:

        - Kwiatkowski, D., Phillips, P.C.B., Schmidt, P., & Shin, Y. (1992). "Testing the Null
        Hypothesis of Stationarity against the Alternative of a Unit Root." Journal of
        Econometrics, 54(1-3), 159-178.

        Also known as: KPSS test, stationarity test.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to test. Defaults to "Adj Close".
            regression (str, optional): Which deterministic term to remove before testing, one of "c"
            (constant, level-stationarity) or "ct" (constant and trend, trend-stationarity).
            Defaults to "c".
            lags (int, optional): The truncation lag for the long-run variance estimate. Defaults
            to `statsmodels`' automatic (Hobijn, Franses & Ooms, 2004) bandwidth selection.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
            assets tested. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The KPSS statistic, its p-value, truncation lag used, observations used,
            1%/2.5%/5%/10% critical values, and whether stationarity is rejected at the 5% level,
            per asset. An asset with fewer than 10 observations has NaN values.

        Notes:
        - The method retrieves historical price data based on the specified `period` and runs the
        KPSS test for each asset in the Toolkit instance.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["^GSPC", "^N225"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_kpss_test()
        ```

        Which returns:

        |                          |      ^GSPC |      ^N225 |
        |:-------------------------|-----------:|-----------:|
        | KPSS Statistic           |    4.73448 |    4.79639 |
        | P-Value                  |    0.01    |    0.01    |
        | Lags Used                |   21       |   21       |
        | Observations             | 1299       | 1298       |
        | Critical Value 1%        |    0.739   |    0.739   |
        | Critical Value 2.5%      |    0.574   |    0.574   |
        | Critical Value 5%        |    0.463   |    0.463   |
        | Critical Value 10%       |    0.347   |    0.347   |
        | Reject Stationarity (5%) |    1       |    1       |
        """
        period = period if period else "daily"
        prices = self._filter_benchmark(
            self._get_series_column(period, column),
            include_benchmark=include_benchmark,
        )

        result = pd.DataFrame(
            {
                ticker: unitroot_model.get_kpss_test(
                    prices[ticker], regression=regression, lags=lags
                )
                for ticker in prices.columns
            }
        )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_phillips_perron_test(
        self,
        period: str | None = None,
        column: str = "Adj Close",
        regression: str = "c",
        lags: int | None = None,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Phillips-Perron (PP) test for a unit root, per asset.

        Phillips-Perron tests the same null hypothesis as the Augmented Dickey-Fuller test (a
        unit root), but corrects for heteroskedasticity and serial correlation in the errors
        nonparametrically via a Newey-West long-run variance estimate, rather than by adding
        lagged-difference terms to the regression as ADF does. PP and ADF should broadly agree
        on the same series since they test the same null with different correction methods.

        The formula is as follows:

        - Yₜ = α + ρ·Yₜ₋₁ + uₜ fitted by OLS on the `column` level (Adj Close by default), with
          regression="c" (the default); "ct" adds a trend β·t
        - t_ρ = (ρ̂ - 1) / SE(ρ̂), γ₀ = Σûₜ² / n and s² = Σûₜ² / (n - k), with k the number of coefficients
        - λ² = γ₀ + 2·Σ(j=1..l) (1 - j / (l + 1))·γⱼ, with γⱼ = (1 / n)·Σ(t=j+1..n) ûₜ·ûₜ₋ⱼ and l = `lags`,
          which defaults to int(12·(n / 100)^(1/4))
        - Z_t = √(γ₀ / λ²)·t_ρ - (λ² - γ₀) / (2·λ·s)·n·SE(ρ̂)
        - The unit root is rejected at 5% when Z_t is below the MacKinnon (1996) asymptotic critical value
          (-2.86 for "c", -3.41 for "ct"); no p-value is reported

        For more information about the method, see the following paper:

        - Phillips, P.C.B., & Perron, P. (1988). "Testing for a Unit Root in Time Series
        Regression." Biometrika, 75(2), 335-346.

        Also known as: PP test, Z_t test.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to test. Defaults to "Adj Close".
            regression (str, optional): Which deterministic term to include, one of "c" (constant) or
            "ct" (constant and trend). Defaults to "c". Note "n" (no constant) is not supported, see
            `econometrics.unitroot_model.get_phillips_perron_test` for why.
            lags (int, optional): The truncation lag for the Newey-West long-run variance estimate.
            Defaults to None, meaning the Schwert (1989) rule ceil(12 * (n / 100) ** 0.25), capped
            for short samples as the ADF lag search is.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
            assets tested. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Phillips-Perron Z_t statistic, its MacKinnon (1994) p-value, truncation
            lag used, observations used, the finite-sample MacKinnon (2010) 1%/5%/10% critical values,
            and whether the unit root is rejected at the 5% level, per asset.

        Notes:
        - The method retrieves historical price data based on the specified `period` and runs the
        Phillips-Perron test for each asset in the Toolkit instance.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["GCUSD", "SIUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_phillips_perron_test()
        ```

        Which returns:

        |                           |      GCUSD |      SIUSD |
        |:--------------------------|-----------:|-----------:|
        | Phillips-Perron Statistic |    3.39009 |    4.5285  |
        | P-Value                   |    1       |    1       |
        | Lags Used                 |   23       |   23       |
        | Observations              | 1300       | 1300       |
        | Critical Value 1%         |   -3.43539 |   -3.43539 |
        | Critical Value 5%         |   -2.86377 |   -2.86377 |
        | Critical Value 10%        |   -2.56796 |   -2.56796 |
        | Reject Unit Root (5%)     |    0       |    0       |
        """
        period = period if period else "daily"
        prices = self._filter_benchmark(
            self._get_series_column(period, column),
            include_benchmark=include_benchmark,
        )

        result = pd.DataFrame(
            {
                ticker: unitroot_model.get_phillips_perron_test(
                    prices[ticker], regression=regression, lags=lags
                )
                for ticker in prices.columns
            }
        )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_zivot_andrews_test(
        self,
        period: str | None = None,
        column: str = "Adj Close",
        max_lag: int | None = None,
        regression: str = "c",
        trim: float = 0.15,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Zivot-Andrews test for a unit root, per asset, allowing for a
        single structural break at an unknown (endogenously estimated) date.

        The (A)DF/KPSS/PP tests above all assume the deterministic component of the
        series (its constant and/or trend) is stable throughout the sample. If a
        series instead has a single one-time break -- e.g. a permanent level shift or
        a change in trend slope, such as a stock split, spin-off, or a structural
        shift in the business -- the ordinary ADF test is biased towards not
        rejecting the unit root even for a genuinely (trend-)stationary series with a
        break. The Zivot-Andrews test corrects for this by adding a break dummy to
        the ADF regression and choosing, for each asset, the break date that is most
        favorable to the stationary alternative -- which is why it needs its own
        (more negative) critical values rather than the ordinary ADF ones.

        The formula is as follows:

        - ΔYₜ = μ + θ·DUₜ + β·t + α·Yₜ₋₁ + Σ(i=1..p) cᵢ·ΔYₜ₋ᵢ + εₜ, with Yₜ the `column` level (Adj Close
          by default) and DUₜ = 1 after the break date T_B and 0 before it (regression="c", the default)
        - regression="t" replaces θ·DUₜ by a trend break γ·DTₜ, with DTₜ = t - T_B after the break and 0
          before it, and regression="ct" includes both
        - ZA = min over T_B of α̂ / SE(α̂), searching every break date outside the first and last `trim`
          (15% by default) of the sample
        - p = the number of lags that minimises the AIC, up to `max_lag` (same default as
          `get_augmented_dickey_fuller`)

        For more information about the method, see the following papers:

        - Zivot, E., & Andrews, D.W.K. (1992). "Further Evidence on the Great Crash,
        the Oil-Price Shock, and the Unit-Root Hypothesis." Journal of Business &
        Economic Statistics, 10(3), 251-270.
        - Perron, P. (1989). "The Great Crash, the Oil Price Shock, and the Unit Root
        Hypothesis." Econometrica, 57(6), 1361-1401.

        Also known as: ZA test, structural break unit root test.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to test. Defaults to "Adj Close".
            max_lag (int, optional): The maximum number of lagged differences to consider when
            selecting the (single, reused) lag length. Defaults to the Schwert (1989) rule of thumb,
            capped at (n - 5) // 3 for short samples.
            regression (str, optional): Which break to allow for, one of "c" (a break in the
            level/intercept), "t" (a break in the trend slope) or "ct" (both). Defaults to "c".
            trim (float, optional): The fraction of observations excluded from the candidate break
            date search at the start and end of the sample. Must be in [0, 1/3). Defaults to 0.15.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
            assets tested. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Zivot-Andrews statistic, its p-value, the (0-indexed) position of the
            selected break date within the used sample, the number of lags and observations used,
            the 1%/5%/10% critical values, and whether the unit root is rejected at the 5% level, per
            asset.

        Notes:
        - The method retrieves historical price data based on the specified `period` and runs the
        Zivot-Andrews test for each asset in the Toolkit instance.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["CLUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_zivot_andrews_test()
        ```

        Which returns:

        |                         |       CLUSD |
        |:------------------------|------------:|
        | Zivot-Andrews Statistic |   -4.1369   |
        | P-Value                 |    0.271885 |
        | Break Index             |  243        |
        | Observations            | 1280        |
        | Lags Used               |   10        |
        | Critical Value 1%       |   -5.27644  |
        | Critical Value 5%       |   -4.81067  |
        | Critical Value 10%      |   -4.56618  |
        | Reject Unit Root (5%)   |    0        |
        """
        period = period if period else "daily"
        prices = self._filter_benchmark(
            self._get_series_column(period, column),
            include_benchmark=include_benchmark,
        )

        result = pd.DataFrame(
            {
                ticker: unitroot_model.get_zivot_andrews_test(
                    prices[ticker], max_lag=max_lag, regression=regression, trim=trim
                )
                for ticker in prices.columns
            }
        )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_engle_granger_cointegration(
        self,
        period: str | None = None,
        column: str = "Adj Close",
        max_lag: int | None = None,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Engle-Granger test for cointegration between every ordered
        pair of tickers in the Toolkit instance.

        Two individually non-stationary series (e.g. two stock prices, each following a
        random walk) are cointegrated if some linear combination of them is stationary,
        i.e. they share a long-run equilibrium relationship even though each wanders on
        its own in the short run. This is the classic statistical foundation for
        pairs-trading: if two assets are cointegrated, deviations of the spread from its
        equilibrium level tend to revert, making the spread itself tradeable.

        In pairs trading the slope of the regression of the dependent on the independent
        ticker is the hedge ratio, and its residual is the spread that is traded.
        Cointegration found in one window often breaks down in the next, so the relationship
        is usually tested again on a later sample before it is relied on. The result depends
        on which ticker is the dependent one, so the two orderings can disagree;
        `get_johansen_cointegration` tests the pair without that choice.

        The formula is as follows:

        - Step 1: Aₜ = α + β·Bₜ + uₜ, the OLS regression of the dependent on the independent `column`
          level (Adj Close by default)
        - Step 2: Δûₜ = γ·ûₜ₋₁ + Σ(i=1..p) δᵢ·Δûₜ₋ᵢ + εₜ, an ADF regression without constant on the residuals
        - EG = γ̂ / SE(γ̂), compared with the MacKinnon (1994, 2010) cointegration critical values and
          p-values for two variables
        - p = the number of lags that minimises the AIC, up to `max_lag` (same default as
          `get_augmented_dickey_fuller`)

        For more information about the method, see the following paper:

        - Engle, R.F. and Granger, C.W.J. (1987). "Co-integration and Error Correction:
        Representation, Estimation, and Testing." Econometrica, 55(2), 251-276.

        Also known as: EG test, residual-based cointegration test, pairs-trading test.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to test. Defaults to "Adj Close".
            max_lag (int, optional): The maximum number of lagged differences to consider in the
            underlying ADF test on the residuals. Defaults to the Schwert (1989) rule of thumb,
            capped at (n - 5) // 3 for short samples.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
            tickers paired up. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: One row per ordered `(Dependent, Independent)` ticker pair, with the
            Engle-Granger statistic, its p-value, 1%/5%/10% critical values, and whether
            cointegration is found at the 5% level -- the test is not symmetric (normalizing on
            the dependent ticker matters), so both orderings of every pair are included. A pair
            with fewer than 10 common observations has NaN values.

        Notes:
        - The method retrieves historical price data based on the specified `period` for every
        ticker in the Toolkit instance and runs the Engle-Granger test on every ordered pair.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["KO", "PEP"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        engle_granger = toolkit.econometrics.get_engle_granger_cointegration()

        # The 1% and 10% critical values follow the same pattern as the 5% one.
        engle_granger[["Engle-Granger Statistic", "P-Value", "Critical Value 5%", "Cointegrated (5%)"]]
        ```

        Which returns:

        |               |   Engle-Granger Statistic |   P-Value |   Critical Value 5% | Cointegrated (5%)   |
        |:--------------|--------------------------:|----------:|--------------------:|:--------------------|
        | ('KO', 'PEP') |                   -1.0266 |    0.8977 |              -3.341 | False               |
        | ('PEP', 'KO') |                   -2.0565 |    0.4991 |              -3.341 | False               |
        """
        period = period if period else "daily"
        prices = self._get_series_column(period, column)

        pairs = self._all_ordered_pairs(include_benchmark=include_benchmark)
        logger.info("Computing %d dependent/independent ticker pairs", len(pairs))

        rows = {
            (ticker_a, ticker_b): cointegration_model.get_engle_granger_cointegration(
                prices[ticker_a], prices[ticker_b], max_lag=max_lag
            )
            for ticker_a, ticker_b in pairs
        }

        # Transposing rows that mix numbers and a boolean leaves object columns, which
        # pandas does not round, so the numeric columns are restored first.
        result = pd.DataFrame(rows).T.infer_objects()
        result.index = result.index.set_names(["Dependent", "Independent"])

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_johansen_cointegration(
        self,
        period: str | None = None,
        column: str = "Adj Close",
        det_order: int = 0,
        k_ar_diff: int = 1,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Johansen test for cointegration among every ticker in the
        Toolkit instance.

        The Engle-Granger test only handles two assets and imposes an arbitrary
        normalization (which asset is "dependent"). Johansen's test generalizes this to
        `N >= 2` assets at once by testing the rank of the long-run coefficient matrix in a
        Vector Error Correction Model (VECM) fit to all assets jointly. The estimated rank
        equals the number of independent cointegrating (long-run equilibrium) relationships
        among the assets: rank 0 means none of them are cointegrated, rank `N` means the
        whole system is already stationary in levels, and a rank in between means that many
        independent stationary combinations exist among the `N` individually non-stationary
        price series.

        The formula is as follows:

        - ΔYₜ = Π·Yₜ₋₁ + Σ(i=1..k) Γᵢ·ΔYₜ₋ᵢ + μ + εₜ, with Yₜ the vector of `column` levels (Adj Close by
          default), k = `k_ar_diff` (1 by default) and the deterministic term μ set by `det_order`: none
          (-1, Johansen's case 1), an unrestricted constant μ₀ (0, the default, case 3) or a linear trend
          (1: the levels are linearly detrended before the test, evaluated against the case 5 critical
          values)
        - λ₁ ≥ ... ≥ λ_N: the eigenvalues of the reduced-rank regression of ΔYₜ on Yₜ₋₁, both corrected for
          the lagged differences and μ
        - Trace(r) = -T·Σ(i=r+1..N) ln(1 - λᵢ)
        - Max-Eigenvalue(r) = -T·ln(1 - λᵣ₊₁)
        - Each statistic rejects "rank ≤ r" at the 5% level when it exceeds its 95% critical value

        For more information about the method, see the following papers:

        - Johansen, S. (1988). "Statistical Analysis of Cointegration Vectors." Journal of
        Economic Dynamics and Control, 12(2-3), 231-254.
        - Johansen, S. (1991). "Estimation and Hypothesis Testing of Cointegration Vectors in
        Gaussian Vector Autoregressive Models." Econometrica, 59(6), 1551-1580.
        - MacKinnon, J.G., Haug, A.A., & Michelis, L. (1999). "Numerical Distribution
        Functions of Likelihood Ratio Tests for Cointegration." Journal of Applied
        Econometrics, 14(5), 563-577.

        Also known as: Johansen test, Johansen procedure, VECM rank test.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to test. Defaults to "Adj Close".
            det_order (int, optional): Which deterministic term to include: -1 (none,
            Johansen's case 1), 0 (an unrestricted constant, case 3) or 1 (a linear trend,
            evaluated against the case 5 critical values). The critical values are those of
            MacKinnon, Haug & Michelis (1999). Defaults to 0.
            k_ar_diff (int, optional): The number of lagged first differences to include as
            short-run dynamics. Defaults to 1.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
            tickers tested jointly. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: One row per candidate cointegration rank `r = 0, ..., N-1`, with the
            corresponding eigenvalue, trace statistic, max-eigenvalue statistic, their 90%/95%/99%
            critical values, and whether each is rejected at the 5% level. The estimated rank is the
            first row (in rank order) that is NOT rejected. With fewer observations than the
            VECM needs (the tickers times k_ar_diff + 1, plus the tickers, plus 3) every row
            is NaN.

        Notes:
        - The method retrieves historical price data based on the specified `period` for every ticker
        in the Toolkit instance and runs the Johansen test jointly across all of them.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["V", "MA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        johansen = toolkit.econometrics.get_johansen_cointegration()

        # The maximum eigenvalue statistic columns follow the same pattern as the trace ones.
        johansen[["Eigenvalue", "Trace Statistic", "Trace Critical Value 95%", "Reject (Trace, 5%)"]]
        ```

        Which returns:

        |        |   Eigenvalue |   Trace Statistic |   Trace Critical Value 95% | Reject (Trace, 5%)   |
        |:-------|-------------:|------------------:|---------------------------:|:---------------------|
        | r <= 0 |       0.0107 |           13.9015 |                    15.4943 | False                |
        | r <= 1 |       0.0003 |            0.3612 |                     3.8415 | False                |
        """
        period = period if period else "daily"
        tickers = self._get_tickers(include_benchmark=include_benchmark)
        prices = self._get_series_column(period, column, tickers)

        result = cointegration_model.get_johansen_cointegration(
            prices[tickers], det_order=det_order, k_ar_diff=k_ar_diff
        )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_granger_causality(
        self,
        period: str | None = None,
        column: str = "Return",
        max_lag: int = 5,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Granger causality test, for every ordered pair of tickers in
        the Toolkit instance, of whether the second helps predict the first.

        "Granger causality" is a statement about predictive power, not true causation:
        one asset is said to Granger-cause another if past values of the first,
        combined with past values of the second itself, predict the second
        significantly better than past values of the second alone.

        On daily data a significant result can reflect trading hours rather than
        information: of two markets that close at different times, the one that closes
        later already contains news the other only prices the next day, so it appears to
        predict it. The result also depends on `max_lag`, so it is usually checked at a few
        lag lengths, and across many pairs some significant results arise by chance alone.

        The formula is as follows:

        - Restricted: Aₜ = c + Σ(i=1..p) aᵢ·Aₜ₋ᵢ + eₜ
        - Unrestricted: Aₜ = c + Σ(i=1..p) aᵢ·Aₜ₋ᵢ + Σ(i=1..p) bᵢ·Bₜ₋ᵢ + uₜ
        - F = ((RSS_r - RSS_u) / p) / (RSS_u / (T - 2p - 1)), F-distributed with (p, T - 2p - 1) degrees
          of freedom
        - A is the dependent and B the independent ticker's `column` (Return by default), p = `max_lag`
          (5 by default) and T the number of observations after the lags

        For more information about the method, see the following paper:

        - Granger, C.W.J. (1969). "Investigating Causal Relations by Econometric Models
        and Cross-Spectral Methods." Econometrica, 37(3), 424-438.

        Also known as: Granger causality test, predictive causality, lead-lag test.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to test. Defaults to "Return", since
            Granger causality assumes a stationary series (unlike the ADF/Engle-Granger tests, which
            operate on price levels on purpose).
            max_lag (int, optional): The number of lags of both assets to include in the regressions.
            Defaults to 5.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
            tickers paired up. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: One row per ordered `(Dependent, Independent)` ticker pair, with the
            F-statistic, its p-value, and whether the independent ticker is found to
            Granger-cause the dependent ticker at the 5% level -- Granger causality is inherently
            directional, so both orderings of every pair are included.

        Notes:
        - The method retrieves historical return data based on the specified `period` for every
        ticker in the Toolkit instance and runs the Granger causality test on every ordered pair.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["CLUSD", "XLE"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_granger_causality()
        ```

        Which returns:

        |                  |   F-Statistic |   P-Value | Granger-Causes (5%)   |
        |:-----------------|--------------:|----------:|:----------------------|
        | ('CLUSD', 'XLE') |        4.2749 |    0.0007 | True                  |
        | ('XLE', 'CLUSD') |        1.5533 |    0.1705 | False                 |
        """
        period = period if period else "daily"
        returns = self._get_series_column(period, column)

        pairs = self._all_ordered_pairs(include_benchmark=include_benchmark)
        logger.info("Computing %d dependent/independent ticker pairs", len(pairs))

        rows = {
            (ticker_a, ticker_b): causality_model.get_granger_causality(
                returns[ticker_a], returns[ticker_b], max_lag=max_lag
            )
            for ticker_a, ticker_b in pairs
        }

        # Transposing rows that mix numbers and a boolean leaves object columns, which
        # pandas does not round, so the numeric columns are restored first.
        result = pd.DataFrame(rows).T.infer_objects()
        result.index = result.index.set_names(["Dependent", "Independent"])

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )
