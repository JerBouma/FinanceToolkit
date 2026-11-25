"""Stationarity and Cointegration Module"""

__docformat__ = "google"

import itertools
from typing import Literal

import numpy as np
import pandas as pd

from financetoolkit.econometrics.stationarity import (
    bubble_model,
    causality_model,
    cointegration_model,
    fractional_model,
    unitroot_model,
)
from financetoolkit.helpers import handle_portfolio
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import check_option, handle_errors
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import apply_rounding

# ruff: noqa: E501

logger = get_logger()


@filter_shown_columns
@track_sources
class StationarityEconometrics:
    """
    Unit root tests (ADF, KPSS, Phillips-Perron, Zivot-Andrews), fractional differentiation, bubble tests (SADF, GSADF, Chow-type Dickey-Fuller), Engle-Granger and Johansen cointegration and Granger causality.
    """

    def _get_levels(
        self, period: str, column: str, log_prices: bool, include_benchmark: bool
    ) -> pd.DataFrame:
        """
        The `column` series of `period` per asset, as natural logarithms when `log_prices` is
        True -- shared by the fractional differentiation and bubble methods, which work on the
        levels of a series and, for prices, on their logarithms (López de Prado, 2018).
        """
        levels = self._filter_benchmark(
            self._get_series_column(period, column),
            include_benchmark=include_benchmark,
        )

        if not log_prices:
            return levels

        if (levels <= 0).any().any():
            raise ValueError(
                "The logarithm needs positive values; pass log_prices=False for a series that "
                "can be zero or negative, such as a rate or a spread."
            )

        return np.log(levels)

    def _all_ordered_pairs(
        self, include_benchmark: bool = False
    ) -> list[tuple[str, str]]:
        """
        Every ordered pair `(ticker_a, ticker_b)`, `ticker_a != ticker_b`, of
        `self._tickers` -- used by the directional pairwise methods below (Granger
        causality, Engle-Granger cointegration), where swapping the pair changes
        which ticker is treated as dependent.
        """
        tickers = self._get_tickers(include_benchmark=include_benchmark)
        if len(tickers) < 2:  # noqa: PLR2004
            raise ValueError(
                f"A test between two series needs at least two series, not only "
                f"{', '.join(tickers) or 'none'}. Please add a ticker, or pass "
                "include_benchmark=True to test against the Benchmark."
            )

        return list(itertools.permutations(tickers, 2))

    @handle_portfolio
    @handle_errors
    def get_unit_root_test(
        self,
        period: str | None = None,
        column: str = "Adj Close",
        method: Literal[
            "augmented_dickey_fuller", "phillips_perron", "zivot_andrews"
        ] = "augmented_dickey_fuller",
        regression: str = "c",
        max_lag: int | None = None,
        lags: int | None = None,
        trim: float = 0.15,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate a unit root test per asset: whether the prices wander like a random walk or are
        stationary and revert to their mean.

        Three tests are available: the Augmented Dickey-Fuller (ADF) test, the Phillips-Perron
        (PP) test and the Zivot-Andrews test, which allows for a single structural break at an
        unknown (endogenously estimated) date.

        All three test the same null hypothesis: the series has a unit root (is a random walk,
        not mean-reverting); the alternative is that it is stationary. This is a standard first
        step before modeling a price series or spread with mean-reverting methods, since those
        methods assume stationarity. `get_kpss_test` is the complement with the null and the
        alternative swapped.

        The ADF test regresses the first difference of the series on its own lagged level and
        `p` lags of its own first difference. The Phillips-Perron test instead corrects for
        heteroskedasticity and serial correlation in the errors nonparametrically via a
        Newey-West long-run variance estimate, rather than by adding lagged-difference terms to
        the regression, so PP and ADF should broadly agree on the same series.

        The ADF and PP tests assume the deterministic component of the series (its constant
        and/or trend) is stable throughout the sample. If a series instead has a single one-time
        break -- e.g. a permanent level shift or a change in trend slope, such as a stock split,
        spin-off, or a structural shift in the business -- the ordinary ADF test is biased
        towards not rejecting the unit root even for a genuinely (trend-)stationary series with
        a break. The Zivot-Andrews test corrects for this by adding a break dummy to the ADF
        regression and choosing, for each asset, the break date that is most favorable to the
        stationary alternative -- which is why it needs its own (more negative) critical values
        rather than the ordinary ADF ones.

        The formula for the Augmented Dickey-Fuller test is as follows:

        - ΔYₜ = α + γ·Yₜ₋₁ + Σ(i=1..p) δᵢ·ΔYₜ₋ᵢ + εₜ, with Yₜ the `column` level (Adj Close by default)
          and a constant (the default); the trend version adds a trend β·t and the version without
          deterministic terms drops α
        - ADF = γ̂ / SE(γ̂), compared with MacKinnon (1994, 2010) critical values and p-values
        - p = the number of lags that minimises the AIC, searched from 0 up to `max_lag`, which defaults to
          min(ceil(12·(n / 100)^(1/4)), (n - 5) // 3)

        The formula for the Phillips-Perron test is as follows:

        - Yₜ = α + ρ·Yₜ₋₁ + uₜ fitted by OLS on the `column` level (Adj Close by default), with a
          constant (the default); the trend version adds a trend β·t
        - t_ρ = (ρ̂ - 1) / SE(ρ̂), γ₀ = Σûₜ² / n and s² = Σûₜ² / (n - k), with k the number of coefficients
        - λ² = γ₀ + 2·Σ(j=1..l) (1 - j / (l + 1))·γⱼ, with γⱼ = (1 / n)·Σ(t=j+1..n) ûₜ·ûₜ₋ⱼ and l = `lags`,
          which defaults to int(12·(n / 100)^(1/4))
        - Z_t = √(γ₀ / λ²)·t_ρ - (λ² - γ₀) / (2·λ·s)·n·SE(ρ̂)
        - The p-value is the MacKinnon (1994) approximation of the Dickey-Fuller distribution and the unit
          root is rejected at 5% when it is below 0.05, as in the ADF test; the critical values are the
          finite-sample MacKinnon (2010) ones for the number of observations

        The formula for the Zivot-Andrews test is as follows:

        - ΔYₜ = μ + θ·DUₜ + β·t + α·Yₜ₋₁ + Σ(i=1..p) cᵢ·ΔYₜ₋ᵢ + εₜ, with Yₜ the `column` level (Adj Close
          by default) and DUₜ = 1 after the break date T_B and 0 before it (a level break, the default)
        - A trend break replaces θ·DUₜ by γ·DTₜ, with DTₜ = t - T_B after the break and 0 before it, and
          a break in both includes both
        - ZA = min over T_B of α̂ / SE(α̂), searching every break date outside the first and last `trim`
          (15% by default) of the sample
        - p = the number of lags that minimises the AIC, up to `max_lag` (same default as the ADF test)

        For more information about the methods, see the following papers:

        - Dickey, D.A. and Fuller, W.A. (1979). "Distribution of the Estimators for
        Autoregressive Time Series with a Unit Root." Journal of the American Statistical
        Association, 74(366a), 427-431.
        - Phillips, P.C.B., & Perron, P. (1988). "Testing for a Unit Root in Time Series
        Regression." Biometrika, 75(2), 335-346.
        - Zivot, E., & Andrews, D.W.K. (1992). "Further Evidence on the Great Crash,
        the Oil-Price Shock, and the Unit-Root Hypothesis." Journal of Business &
        Economic Statistics, 10(3), 251-270.
        - Perron, P. (1989). "The Great Crash, the Oil Price Shock, and the Unit Root
        Hypothesis." Econometrica, 57(6), 1361-1401.

        Also known as: ADF test, augmented Dickey-Fuller test, PP test, Z_t test, ZA test,
        structural break unit root test, stationarity test, mean reversion test, random walk test,
        trending or mean-reverting.

        Referenced against:
        - statsmodels, adfuller (Augmented Dickey-Fuller test): https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html
        - arch, PhillipsPerron (Phillips-Perron test): https://bashtage.github.io/arch/unitroot/generated/arch.unitroot.PhillipsPerron.html

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to test. Defaults to "Adj Close".
            method (str, optional): The test. Options are: "augmented_dickey_fuller" (Augmented
            Dickey-Fuller test), "phillips_perron" (Phillips-Perron test), "zivot_andrews"
            (Zivot-Andrews test with a structural break). Defaults to "augmented_dickey_fuller".
            regression (str, optional): Which deterministic terms to include. For
            "augmented_dickey_fuller" one of "n" (none), "c" (constant) or "ct" (constant and trend);
            for "phillips_perron" one of "c" or "ct" ("n" is not supported, see
            `econometrics.unitroot_model.get_phillips_perron_test` for why); for "zivot_andrews"
            which break to allow for, one of "c" (a break in the level/intercept), "t" (a break in
            the trend slope) or "ct" (both). Defaults to "c".
            max_lag (int, optional): The maximum number of lagged differences to consider in the AIC
            lag search of "augmented_dickey_fuller" and "zivot_andrews" (which selects a single lag
            length and reuses it for every break date). Not used with "phillips_perron". Defaults to
            the Schwert (1989) rule of thumb, capped at (n - 5) // 3 for short samples.
            lags (int, optional): The truncation lag for the Newey-West long-run variance estimate of
            "phillips_perron"; not used by the other methods. Defaults to None, meaning the Schwert
            (1989) rule ceil(12 * (n / 100) ** 0.25), capped for short samples as the ADF lag search is.
            trim (float, optional): The fraction of observations excluded from the candidate break
            date search at the start and end of the sample, for "zivot_andrews" only. Must be in
            [0, 1/3). Defaults to 0.15.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
            assets tested. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The test statistic ("ADF Statistic", "Phillips-Perron Statistic" or
            "Zivot-Andrews Statistic"), its p-value, the lags used (the Newey-West truncation lag for
            Phillips-Perron), the observations used, the 1%/5%/10% critical values, and whether the
            unit root is rejected at the 5% level, per asset. The Zivot-Andrews test adds the
            (0-indexed) position of the selected break date within the used sample ("Break Index").
            With the ADF test an asset with fewer than 10 observations has NaN values.

        Notes:
        - The method retrieves historical price data based on the specified `period` and runs the
        chosen test for each asset in the Toolkit instance.
        - The Phillips-Perron p-value is MacKinnon's (1994) approximation and its critical values the
        finite-sample MacKinnon (2010) ones.
        - Requires statsmodels (`pip install financetoolkit[econometrics]`).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["EURUSD", "USDJPY", "GBPUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_unit_root_test()
        ```

        Which returns:

        |                       |    EURUSD |    USDJPY |    GBPUSD |
        |:----------------------|----------:|----------:|----------:|
        | ADF Statistic         |   -2.0711 |   -1.6878 |   -1.8861 |
        | P-Value               |    0.2563 |    0.4374 |    0.3386 |
        | Lags Used             |    2      |    0      |    0      |
        | Observations          | 1336      | 1351      | 1343      |
        | Critical Value 1%     |   -3.4353 |   -3.4352 |   -3.4352 |
        | Critical Value 5%     |   -2.8637 |   -2.8637 |   -2.8637 |
        | Critical Value 10%    |   -2.5679 |   -2.5679 |   -2.5679 |
        | Reject Unit Root (5%) |    0      |    0      |    0      |
        """
        check_option(
            "method",
            method,
            (
                "augmented_dickey_fuller",
                "phillips_perron",
                "zivot_andrews",
            ),
        )

        period = period if period else "daily"
        prices = self._filter_benchmark(
            self._get_series_column(period, column),
            include_benchmark=include_benchmark,
        )

        # Each test keeps its own lag parameter: the ADF and Zivot-Andrews tests search the
        # lag length by AIC up to max_lag, Phillips-Perron takes the Newey-West truncation lag.
        if method == "phillips_perron":
            values = {
                ticker: unitroot_model.get_phillips_perron_test(
                    prices[ticker], regression=regression, lags=lags
                )
                for ticker in prices.columns
            }
        elif method == "zivot_andrews":
            values = {
                ticker: unitroot_model.get_zivot_andrews_test(
                    prices[ticker], max_lag=max_lag, regression=regression, trim=trim
                )
                for ticker in prices.columns
            }
        else:
            values = {
                ticker: unitroot_model.get_augmented_dickey_fuller(
                    prices[ticker], max_lag=max_lag, regression=regression
                )
                for ticker in prices.columns
            }

        return apply_rounding(
            pd.DataFrame(values), rounding if rounding is not None else self._rounding
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
        Calculate the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test per asset: whether the prices
        are stationary, around a constant or a trend, rather than having a unit root.

        KPSS is the natural complement to the Augmented Dickey-Fuller test (`get_unit_root_test`):
        where the ADF null hypothesis is that the series HAS a unit root, the KPSS null hypothesis
        is that the series IS stationary, with a unit root as the alternative. Running both together is
        standard practice to triangulate a confident conclusion -- ADF rejecting a unit root
        and KPSS failing to reject stationarity together give a confident stationarity
        conclusion, while the two tests disagreeing flags an ambiguous case.

        The formula is as follows:

        - Yₜ = α + eₜ fitted by OLS on the `column` level (Adj Close by default), with a constant
          (the default); the trend version adds a trend β·t
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

        The test runs statsmodels' kpss.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to test. Defaults to "Adj Close".
            regression (str, optional): Which deterministic term to remove before testing. Options
            are: "c" (constant, level-stationarity), "ct" (constant and trend, trend-stationarity).
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
        - Requires statsmodels (`pip install financetoolkit[econometrics]`).

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
        Calculate the Engle-Granger test for every ordered pair of tickers: whether the two prices
        are cointegrated, held together by a long-run equilibrium so that the spread reverts.

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
          `get_unit_root_test`)

        For more information about the method, see the following paper:

        - Engle, R.F. and Granger, C.W.J. (1987). "Co-integration and Error Correction:
        Representation, Estimation, and Testing." Econometrica, 55(2), 251-276.

        Also known as: EG test, residual-based cointegration test, pairs-trading test.

        Referenced against:
        - statsmodels, coint: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html

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
        - Requires statsmodels (`pip install financetoolkit[econometrics]`).

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
        | ('KO', 'PEP') |                   -1.0287 |    0.8973 |              -3.341 | False               |
        | ('PEP', 'KO') |                   -2.0562 |    0.4993 |              -3.341 | False               |
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
        lags: int = 1,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Johansen test across every ticker: how many long-run equilibrium
        relationships (cointegrating relations) hold the prices together.

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
          default), k = `lags` (1 by default) and the deterministic term μ: none (Johansen's case 1),
          an unrestricted constant μ₀ (the default, case 3) or a linear trend (the levels are linearly
          detrended before the test, evaluated against the case 5 critical values)
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

        The test runs statsmodels' coint_johansen.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to test. Defaults to "Adj Close".
            det_order (int, optional): Which deterministic term to include: -1 (none,
            Johansen's case 1), 0 (an unrestricted constant, case 3) or 1 (a linear trend,
            evaluated against the case 5 critical values). The critical values are those of
            MacKinnon, Haug & Michelis (1999). Defaults to 0.
            lags (int, optional): The number of lagged first differences to include as
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
            VECM needs (the tickers times lags + 1, plus the tickers, plus 3) every row
            is NaN.

        Notes:
        - The method retrieves historical price data based on the specified `period` for every ticker
        in the Toolkit instance and runs the Johansen test jointly across all of them.
        - Requires statsmodels (`pip install financetoolkit[econometrics]`).

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
        johansen[["Eigenvalue", "Trace Statistic", "Trace Critical Value 5%", "Reject (Trace, 5%)"]]
        ```

        Which returns:

        |        |   Eigenvalue |   Trace Statistic |   Trace Critical Value 5% | Reject (Trace, 5%)   |
        |:-------|-------------:|------------------:|--------------------------:|:---------------------|
        | r <= 0 |       0.0107 |           13.8983 |                   15.4943 | False                |
        | r <= 1 |       0.0003 |            0.3614 |                    3.8415 | False                |
        """
        period = period if period else "daily"
        tickers = self._get_tickers(include_benchmark=include_benchmark)
        prices = self._get_series_column(period, column, tickers)

        result = cointegration_model.get_johansen_cointegration(
            prices[tickers], det_order=det_order, lags=lags
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
        Calculate the Granger causality test for every ordered pair of tickers: whether the past
        returns of the second help predict the first beyond its own past.

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

        Referenced against:
        - statsmodels, grangercausalitytests: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.grangercausalitytests.html

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

    @handle_portfolio
    @handle_errors
    def get_fractional_difference(
        self,
        period: str | None = None,
        column: str = "Adj Close",
        d: float = 0.4,
        method: Literal["fixed_width", "expanding"] = "fixed_width",
        threshold: float | None = None,
        log_prices: bool = True,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the fractional difference of the log prices per asset: a stationary series that
        keeps most of the memory of the prices, (1 - L)^d with a d between 0 and 1.

        A price series has a unit root, so models that need a stationary input usually take
        returns, the first difference (d = 1), which throws away the memory of the series: the
        return of today says nothing about where the price stands. López de Prado (2018) shows
        that a fractional d, often between 0.2 and 0.6 for prices, already gives a stationary
        series that keeps most of that memory (a correlation with the prices well above that of
        the returns). Find the smallest such d with `get_minimum_fractional_difference`.

        The formula is as follows, with Xₜ the log of the `column` (Adj Close by default):

        - ω₀ = 1, ωₖ = -ωₖ₋₁·(d - k + 1) / k, the binomial weights of (1 - L)^d (Hosking, 1981)
        - Fixed-width window (FFD): X̃ₜ = Σ(k=0..l*) ωₖ·Xₜ₋ₖ, with l* the last
          lag whose weight is at least the threshold (1e-5 by default) in absolute value, so every
          date has the same weights
        - Expanding window: X̃ₜ = Σ(k=0..t) ωₖ·Xₜ₋ₖ over the whole history up
          to t, kept from the first date at which the share of the weight that falls before the
          start of the series, Σ(j>t) |ωⱼ| / Σ(j=0..n-1) |ωⱼ|, is at most the threshold (0.01 by
          default)
        - d = 0 returns the log prices, d = 1 the log returns

        For more information, see the following sources:

        - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, chapter 5
        (Snippets 5.1 to 5.3).
        - Hosking, J.R.M. (1981). "Fractional Differencing." Biometrika, 68(1), 165-176.

        Also known as: fractional differentiation, fractionally differenced prices, FFD, fixed-width
        window fracdiff, memory-preserving stationarity transform.

        Referenced against:
        - RiskLabAI, fractional_difference_std: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/data/differentiation/differentiation.py#L85
        - RiskLabAI, fractional_difference_fixed: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/data/differentiation/differentiation.py#L142

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to differentiate. Defaults to "Adj Close".
            d (float, optional): The order of differentiation, at least 0. Defaults to 0.4.
            method (Literal["fixed_width", "expanding"], optional): The window. Options are:
                "fixed_width" (fixed-width window, FFD), "expanding" (expanding window). Defaults to
                "fixed_width".
            threshold (float | None, optional): The smallest absolute weight kept by the fixed-width
                window, or the largest share of weight left out by the expanding window. Defaults to
                None, which is 1e-5 for "fixed_width" and 0.01 for "expanding".
            log_prices (bool, optional): Whether to take the natural logarithm of the series first.
                Pass False for a series that can be zero or negative, such as a rate. Defaults to True.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
                assets. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
                None, which uses the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The fractionally differentiated series per asset and date, NaN for the
            dates before the first full window.

        Notes:
        - Each asset is differentiated over the dates it has a value. With a threshold of 1e-5 the
          fixed-width window spans about 3,400 observations for d = 0.2, 1,460 for d = 0.4 and 230
          for d = 0.8, so a small d needs a long history: on a shorter one the result is NaN throughout. A
          larger threshold shortens the window (1e-4: about 500, 280 and 64; 0.01: about 10).
        - The weights, the skipped values and the series agree with RiskLabAI to 1e-14.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2016-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_fractional_difference(d=0.4).tail(5)
        ```

        Which returns:

        | Date       |   AAPL |   MSFT |
        |:-----------|-------:|-------:|
        | 2025-12-24 | 0.237  | 0.2441 |
        | 2025-12-26 | 0.2332 | 0.2418 |
        | 2025-12-29 | 0.2345 | 0.2401 |
        | 2025-12-30 | 0.2313 | 0.241  |
        | 2025-12-31 | 0.2274 | 0.2327 |
        """
        period = period if period else "daily"
        levels = self._get_levels(period, column, log_prices, include_benchmark)

        result = pd.DataFrame(
            {
                ticker: fractional_model.get_fractional_difference(
                    levels[ticker], d=d, method=method, threshold=threshold
                )
                for ticker in levels.columns
            },
            index=levels.index,
        )

        # The fixed-width window of a small d spans thousands of observations; a series shorter
        # than the window has no values at all, which would otherwise pass without a reason.
        if method == "fixed_width":
            observations = levels.count()
            width = len(
                fractional_model.get_fractional_weights(
                    d,
                    size=int(observations.max()) + 1,
                    threshold=(
                        fractional_model.FIXED_WIDTH_THRESHOLD
                        if threshold is None
                        else threshold
                    ),
                )
            )
            short = list(observations.index[observations < width])
            if short:
                logger.warning(
                    "The fixed-width window of d = %s needs at least %s observations, more than %s %s, "
                    "so the result is NaN there. Please pass a larger threshold (such as 1e-4) "
                    "or a longer date range.",
                    d,
                    width,
                    ", ".join(map(str, short)),
                    "has" if len(short) == 1 else "have",
                )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_minimum_fractional_difference(
        self,
        period: str | None = None,
        column: str = "Adj Close",
        confidence: float = 0.95,
        step: float = 0.1,
        threshold: float = 0.01,
        max_lag: int | None = 1,
        log_prices: bool = True,
        grid: bool = False,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the minimum order of fractional differentiation d that makes the log prices
        stationary, per asset: the smallest d at which the fixed-width fractional difference
        passes the Augmented Dickey-Fuller test, the stationary series that keeps the most memory
        of the prices (López de Prado, 2018, section 5.6).

        The formula is as follows, with Xₜ the log of the `column` (Adj Close by default):

        - For d = 0, 0.1, ..., 1 (`step`): X̃ = FFD(X, d) of `get_fractional_difference` with the
          weight threshold (0.01 by default) and the ADF test on X̃ with `get_unit_root_test`
          (constant, lags by AIC up to `max_lag`, 1 by default), as in López de Prado's Snippet 5.4
        - d* = the smallest d whose ADF p-value is below 1 - confidence (0.05 by default), refined
          in steps of step / 10 between the last d that fails and the first that passes
        - Correlation = the correlation of X̃ with X over the dates of X̃, the memory kept

        For more information, see the following sources:

        - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, section 5.6
        (Snippet 5.4).
        - Dickey, D.A. and Fuller, W.A. (1979). "Distribution of the Estimators for Autoregressive
        Time Series with a Unit Root." Journal of the American Statistical Association, 74(366a),
        427-431.

        Also known as: minimum d, optimal fractional differentiation, minimum FFD, memory-preserving
        stationarity.

        Referenced against:
        - RiskLabAI, find_optimal_ffd_simple: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/data/differentiation/differentiation.py#L273

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to test. Defaults to "Adj Close".
            confidence (float, optional): The confidence of the ADF test: the p-value has to be below
                1 - confidence. Defaults to 0.95.
            step (float, optional): The step of the grid of d from 0 to 1. Defaults to 0.1.
            threshold (float, optional): The smallest absolute weight of the fixed-width window. At
                1e-5 the window of d = 0.1 spans about 4,000 observations and at 1e-4 about 500, so
                on a few years of data a small d leaves too few values to test. Defaults to 0.01.
            max_lag (int | None, optional): The largest lag of the ADF lag search. Defaults to 1, as
                López de Prado (2018) uses. None searches up to min(ceil(12·(n / 100)^(1/4)),
                (n - 5) // 3) lags by AIC, which on daily data picks many lags, weakens the test and
                pushes the minimum d towards 1.
            log_prices (bool, optional): Whether to take the natural logarithm of the series first.
                Pass False for a series that can be zero or negative, such as a rate. Defaults to True.
            grid (bool, optional): Whether to return the ADF test, the correlation and the window
                width for every d tried instead of the minimum d per asset. Defaults to False.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
                assets. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
                None, which uses the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per asset, the Minimum d and, at that d, the ADF Statistic, P-Value, 5%
            Critical Value, Correlation with the log prices, Window Width and Observations (NaN
            when no d up to 1 passes); with grid=True the ADF Statistic, P-Value, Lags Used,
            Observations, Critical Value 5%, Correlation, Window Width and Stationary per d.

        Notes:
        - Requires statsmodels (`pip install financetoolkit[econometrics]`) for the ADF test.
        - RiskLabAI and López de Prado's Snippet 5.4 run the ADF test with a single lag and a weight
          threshold of 0.01, the defaults here.
        - A d whose window is longer than the history leaves fewer than 10 values to test and
          shows NaN; pass a larger threshold or a longer history.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2016-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_minimum_fractional_difference()
        ```

        Which returns:

        |                   |      AAPL |      MSFT |
        |:------------------|----------:|----------:|
        | Minimum d         |    0.53   |    0.56   |
        | ADF Statistic     |   -2.9108 |   -2.9734 |
        | P-Value           |    0.0441 |    0.0375 |
        | Critical Value 5% |   -2.8627 |   -2.8627 |
        | Correlation       |    0.9917 |    0.9919 |
        | Window Width      |   10      |    9      |
        | Observations      | 2503      | 2504      |
        """
        period = period if period else "daily"
        levels = self._get_levels(period, column, log_prices, include_benchmark)

        fits = {
            ticker: fractional_model.get_minimum_fractional_difference(
                levels[ticker],
                confidence=confidence,
                step=step,
                threshold=threshold,
                max_lag=max_lag,
            )
            for ticker in levels.columns
        }

        if grid:
            result = pd.concat(
                {ticker: table for ticker, (table, _) in fits.items()}, axis=1
            )
            result = result.swaplevel(0, 1, axis=1).sort_index(axis=1)
        else:
            summary = {}
            for ticker, (table, minimum_d) in fits.items():
                row = (
                    table.loc[minimum_d]
                    if not np.isnan(minimum_d)
                    else table.iloc[0] * np.nan
                )
                summary[ticker] = pd.Series(
                    {
                        "Minimum d": minimum_d,
                        "ADF Statistic": row["ADF Statistic"],
                        "P-Value": row["P-Value"],
                        "Critical Value 5%": row["Critical Value 5%"],
                        "Correlation": row["Correlation"],
                        "Window Width": row["Window Width"],
                        "Observations": row["Observations"],
                    },
                    dtype=float,
                )
            result = pd.DataFrame(summary)

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_bubble_test(
        self,
        period: str | None = None,
        column: str = "Adj Close",
        method: Literal["gsadf", "sadf"] = "gsadf",
        lag_order: int = 0,
        minimum_window: int | None = None,
        simulations: int = 2000,
        seed: int = 0,
        log_prices: bool = True,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the GSADF test of Phillips, Shi and Yu (2015) or the SADF test of Phillips, Wu
        and Yu (2011) for explosive behaviour, a bubble, in the log prices, per asset.

        The ADF test looks for mean reversion (a left tail); these tests look at the right tail:
        a price that grows faster than a random walk, as in a bubble, gives a large positive ADF
        statistic over the window of the bubble. The SADF test runs the ADF regression on every
        window from the first observation that is at least the minimum window long and takes the
        largest statistic; the GSADF test also moves the start of the window, so it finds a
        bubble that came after an earlier one collapsed. A statistic above its critical value
        rejects the random walk in favour of explosive behaviour somewhere in the sample; which
        dates are explosive follows from `get_bubble_episodes`.

        The formula is as follows, with yₜ the log of the `column` (Adj Close by default) and T
        observations:

        - ADF(r₁, r₂) = β̂ / SE(β̂) of Δyₜ = α + β·yₜ₋₁ + Σ(i=1..k) γᵢ·Δyₜ₋ᵢ + εₜ on the observations r₁
          to r₂, with k = `lag_order`
        - w = floor((0.01 + 1.8 / √T)·T), the minimum window of PSY (2015)
        - SADF = max over r₂ ≥ w - 1 of ADF(0, r₂)
        - GSADF = max over r₂ of BSADF(r₂), with BSADF(r₂) = max over r₁ ≤ r₂ - w + 1 of ADF(r₁, r₂)
        - Critical values and the p-value from `simulations` random walks yₜ = yₜ₋₁ + 1 / T + εₜ,
          εₜ ~ N(0, 1), with the same T, w and k: the 90%, 95% and 99% quantiles of the simulated
          statistic and the share of simulated statistics at least as large

        For more information, see the following papers:

        - Phillips, P.C.B., Shi, S. and Yu, J. (2015). "Testing for Multiple Bubbles: Historical
        Episodes of Exuberance and Collapse in the S&P 500." International Economic Review, 56(4),
        1043-1078.
        - Phillips, P.C.B., Wu, Y. and Yu, J. (2011). "Explosive Behavior in the 1990s Nasdaq: When
        Did Exuberance Escalate Asset Values?" International Economic Review, 52(1), 201-226.
        - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, section
        17.4.2 (Snippets 17.1 to 17.4).

        Also known as: GSADF test, SADF test, sup ADF, generalized sup ADF, PSY test, PWY test,
        explosive root test, right-tailed unit root test, bubble detection.

        Referenced against:
        - RiskLabAI, get_gsadf_statistic: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/features/structural_breaks/structural_breaks.py#L486
        - RiskLabAI, simulate_psy_critical_values: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/features/structural_breaks/structural_breaks.py#L607
        - statsmodels, adfuller: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to test. Defaults to "Adj Close".
            method (Literal["gsadf", "sadf"], optional): The test. Options are: "gsadf" (generalized
                sup ADF test, GSADF), "sadf" (forward sup ADF test, SADF). Defaults to "gsadf".
            lag_order (int, optional): The number of lagged differences k in every ADF regression.
                Defaults to 0, as PSY (2015) use.
            minimum_window (int | None, optional): The smallest window in observations. Defaults to
                None, floor((0.01 + 1.8 / √T)·T).
            simulations (int, optional): The number of simulated random walks for the critical
                values. Defaults to 2,000, as in PSY (2015): the critical values serve every series of
                the same length, so with 500 random walks the size of the test at 5% ranges from about
                3.5% to 7% depending on the seed.
            seed (int, optional): The seed of the simulation. Defaults to 0.
            log_prices (bool, optional): Whether to take the natural logarithm of the series first.
                Pass False for a series that can be zero or negative, such as a rate. Defaults to True.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
                assets. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
                None, which uses the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The GSADF (or SADF) statistic, its simulated p-value, the 10%, 5% and 1%
            critical values (the 90%, 95% and 99% quantiles), the observations, minimum window,
            lag order and replications (the number of simulations), and whether the series is explosive at the 5% level, per
            asset.

        Notes:
        - The test fits about (T - w)² / 2 regressions per series and as many per simulated path,
          each from window sums of prefix sums: about 0.14 seconds per path for ten years of daily
          data and 0.04 for five, so 2,000 simulations take about 5 minutes and 80 seconds, and
          with two lagged differences about 60% longer. Ten years of weekly data take about 16 seconds;
          monthly data, as in PSY (2015), a few. The critical values depend
          only on T, w, the lag order, the simulations and the seed, so assets with the same
          number of observations share one simulation, which is kept for the session.
        - Each asset is tested over the dates it has a value.
        - Every window statistic agrees with statsmodels' adfuller to 1e-10, and the simulated
          critical values with those of PSY (2015, Table 1).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NVDA", "KO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2016-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_bubble_test(period="weekly")
        ```

        Which returns:

        |                    |      NVDA |        KO |
        |:-------------------|----------:|----------:|
        | GSADF Statistic    |    1.5086 |    1.056  |
        | P-Value            |    0.3165 |    0.6405 |
        | Critical Value 10% |    1.9834 |    1.9834 |
        | Critical Value 5%  |    2.2071 |    2.2071 |
        | Critical Value 1%  |    2.8067 |    2.8067 |
        | Observations       |  522      |  522      |
        | Minimum Window     |   46      |   46      |
        | Lags               |    0      |    0      |
        | Replications       | 2000      | 2000      |
        | Explosive (5%)     |    0      |    0      |
        """
        period = period if period else "daily"
        levels = self._get_levels(period, column, log_prices, include_benchmark)

        result = pd.DataFrame(
            {
                ticker: bubble_model.get_bubble_test(
                    levels[ticker],
                    method=method,
                    lags=lag_order,
                    minimum_window=minimum_window,
                    simulations=simulations,
                    seed=seed,
                )
                for ticker in levels.columns
            }
        )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_bubble_episodes(
        self,
        period: str | None = None,
        column: str = "Adj Close",
        method: Literal["gsadf", "sadf"] = "gsadf",
        lag_order: int = 0,
        minimum_window: int | None = None,
        confidence: float = 0.95,
        minimum_duration: int | None = None,
        simulations: int = 2000,
        seed: int = 0,
        log_prices: bool = True,
        time_series: bool = False,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Date-stamp the explosive episodes (bubbles) of the log prices, per asset, with the
        backward SADF (BSADF) sequence of Phillips, Shi and Yu (2015) or the forward ADF sequence
        of Phillips, Wu and Yu (2011).

        For every date, the BSADF statistic is the largest ADF statistic of the windows that end
        on that date, so it only uses data up to the date: it rises once the prices start to grow
        explosively and falls back when the bubble collapses. A bubble starts on the first date
        its statistic crosses above the critical value of that date and ends on the last date
        before it falls back. PSY (2015) show that the backward windows date-stamp a bubble that
        follows an earlier one, which the forward sequence of PWY (2011) misses.

        The formula is as follows, with yₜ the log of the `column` (Adj Close by default) and the
        ADF(r₁, r₂) statistics and minimum window w of `get_bubble_test`:

        - BSADF(r₂) = max over r₁ ≤ r₂ - w + 1 of ADF(r₁, r₂) (the backward sequence), or ADF(0, r₂)
          (the forward sequence)
        - cv(r₂) = the `confidence` quantile (95% by default) of the same statistic at r₂ over
          `simulations` random walks yₜ = yₜ₋₁ + 1 / T + εₜ
        - Bubble on r₂ when the statistic is above cv(r₂); an episode is a run of such dates of at
          least `minimum_duration` observations, by default round(log(T)) (δ·log(T) with δ = 1 in
          PSY, 2015)

        For more information, see the following papers:

        - Phillips, P.C.B., Shi, S. and Yu, J. (2015). "Testing for Multiple Bubbles: Historical
        Episodes of Exuberance and Collapse in the S&P 500." International Economic Review, 56(4),
        1043-1078.
        - Phillips, P.C.B., Wu, Y. and Yu, J. (2011). "Explosive Behavior in the 1990s Nasdaq: When
        Did Exuberance Escalate Asset Values?" International Economic Review, 52(1), 201-226.

        Also known as: BSADF, backward sup ADF, PSY date-stamping, bubble dating, explosive episodes,
        exuberance periods.

        Referenced against:
        - RiskLabAI, get_bsadf_sequence: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/features/structural_breaks/structural_breaks.py#L434
        - RiskLabAI, get_bubble_episodes: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/features/structural_breaks/structural_breaks.py#L544

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to test. Defaults to "Adj Close".
            method (Literal["gsadf", "sadf"], optional): The sequence. Options are: "gsadf" (backward
                sequence, BSADF of PSY), "sadf" (forward ADF sequence of PWY). Defaults to "gsadf".
            lag_order (int, optional): The number of lagged differences k in every ADF regression.
                Defaults to 0.
            minimum_window (int | None, optional): The smallest window in observations. Defaults to
                None, floor((0.01 + 1.8 / √T)·T).
            confidence (float, optional): The confidence of the critical value per date. Defaults to
                0.95.
            minimum_duration (int | None, optional): The fewest observations an episode lasts.
                Defaults to None, round(log(T)).
            simulations (int, optional): The number of simulated random walks for the critical
                values. Defaults to 2,000, as in PSY (2015).
            seed (int, optional): The seed of the simulation. Defaults to 0.
            log_prices (bool, optional): Whether to take the natural logarithm of the series first.
                Pass False for a series that can be zero or negative, such as a rate. Defaults to True.
            time_series (bool, optional): Whether to return the statistic, its critical value and
                whether it is above it per date instead of the episodes. Defaults to False.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
                assets. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
                None, which uses the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per asset and episode, the Start and End date, the Duration in
            observations, and the Peak Date and Peak Statistic (the largest statistic of the
            episode); with time_series=True the BSADF (or ADF) statistic, Critical Value and Bubble
            per date and asset.

        Notes:
        - The computation and its time are those of `get_bubble_test`, which shares the
          simulation when called with the same settings.
        - The critical value of a date is simulated for the observations up to it, so a date's
          verdict only uses data up to that date.
        - The critical value of a single date is lower than that of the GSADF test, which is the
          largest statistic over all dates, so episodes can be dated while `get_bubble_test` does
          not reject the random walk for the whole sample (as for NVDA below). PSY (2015) read
          the dating as a monitoring device and the GSADF test as the verdict on the sample.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NVDA", "KO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2016-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_bubble_episodes(period="weekly")
        ```

        Which returns:

        |             | Start                 | End                   |   Duration | Peak Date             |   Peak Statistic |
        |:------------|:----------------------|:----------------------|-----------:|:----------------------|-----------------:|
        | ('NVDA', 1) | 2020-08-03/2020-08-09 | 2020-09-07/2020-09-13 |          6 | 2020-08-24/2020-08-30 |           1.0849 |
        | ('NVDA', 2) | 2021-10-25/2021-10-31 | 2021-12-27/2022-01-02 |         10 | 2021-11-15/2021-11-21 |           1.409  |
        | ('NVDA', 3) | 2023-05-22/2023-05-28 | 2023-07-31/2023-08-06 |         11 | 2023-06-12/2023-06-18 |           1.0915 |
        | ('NVDA', 4) | 2024-01-29/2024-02-04 | 2025-01-20/2025-01-26 |         52 | 2024-06-10/2024-06-16 |           1.5086 |
        """
        period = period if period else "daily"
        levels = self._get_levels(period, column, log_prices, include_benchmark)

        fits = {
            ticker: bubble_model.get_bubble_episodes(
                levels[ticker],
                method=method,
                lags=lag_order,
                minimum_window=minimum_window,
                confidence=confidence,
                minimum_duration=minimum_duration,
                simulations=simulations,
                seed=seed,
            )
            for ticker in levels.columns
        }

        if time_series:
            result = pd.concat(
                {ticker: sequence for ticker, (_, sequence) in fits.items()}, axis=1
            ).reindex(levels.index)
            result = result.swaplevel(0, 1, axis=1).sort_index(axis=1)
        else:
            result = pd.concat(
                {ticker: episodes for ticker, (episodes, _) in fits.items()}
            )
            result.index = result.index.set_names(["Ticker", "Episode"])
            # Episodes are numbered from 1 per asset.
            result.index = result.index.set_levels(
                result.index.levels[1] + 1, level="Episode"
            )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_chow_dickey_fuller_test(
        self,
        period: str | None = None,
        column: str = "Adj Close",
        trim: float = 0.15,
        simulations: int = 1000,
        seed: int = 0,
        log_prices: bool = True,
        time_series: bool = False,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Chow-type Dickey-Fuller test for a switch from a random walk to explosive
        behaviour at an unknown date, per asset (López de Prado, 2018, section 17.3.3; Homm and
        Breitung, 2012).

        The regression lets the log prices follow a random walk up to a break date and an
        explosive process after it, and the statistic is the t-statistic of the explosive
        coefficient. Every date in the middle of the sample is tried as the break; the largest
        statistic (SDFC) is compared with critical values simulated under a random walk, and its
        date is the most likely start of the explosive regime. Unlike the Chow test of
        `get_chow_test`, which tests a known break in the coefficients of a regression, the break
        date is unknown and the alternative is a bubble.

        The formula is as follows, with yₜ the log of the `column` (Adj Close by default) minus its
        first value, so the test does not depend on the level of the prices:

        - Δyₜ = δ·yₜ₋₁·Dₜ(τ) + εₜ, Dₜ(τ) = 1 after the break date τ and 0 up to it
        - DFC(τ) = δ̂ / SE(δ̂), with SE from the residuals of all T - 1 changes
        - SDFC = max over τ from floor(trim·T) to T - 1 - floor(trim·T) of DFC(τ), the supremum over
          the break dates of Andrews (1993)
        - Critical values and the p-value from `simulations` Gaussian random walks of the same
          length started at zero

        For more information, see the following sources:

        - Homm, U. and Breitung, J. (2012). "Testing for Speculative Bubbles in Stock Markets: A
        Comparison of Alternative Methods." Journal of Financial Econometrics, 10(1), 198-231.
        - Andrews, D.W.K. (1993). "Tests for Parameter Instability and Structural Change With
        Unknown Change Point." Econometrica, 61(4), 821-856.
        - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, section 17.4.1.

        The book writes the regression on the log prices as they are; here their first value is
        subtracted, so the statistic does not change with the price level.

        Also known as: DFC, SDFC, Chow-type Dickey-Fuller, sup DFC, Homm-Breitung test, explosive
        break test.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to test. Defaults to "Adj Close".
            trim (float, optional): The share of observations at either end that is not tried as the
                break date. Defaults to 0.15.
            simulations (int, optional): The number of simulated random walks for the critical
                values. Defaults to 1000.
            seed (int, optional): The seed of the simulation. Defaults to 0.
            log_prices (bool, optional): Whether to take the natural logarithm of the series first.
                Pass False for a series that can be zero or negative, such as a rate. Defaults to True.
            time_series (bool, optional): Whether to return DFC(τ) per break date instead of the test.
                Defaults to False.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
                assets. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
                None, which uses the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The SDFC statistic, the break date, the simulated p-value, the 10%, 5% and 1%
            critical values (the 90%, 95% and 99% quantiles), the observations and replications (the number of simulations), and
            whether the series turns explosive at the 5% level, per asset; with time_series=True
            DFC(τ) per date and asset.

        Notes:
        - The test is right-tailed: a large positive statistic points to explosive behaviour after
          the break date. A falling or mean-reverting series gives negative statistics.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NVDA", "KO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2016-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_chow_dickey_fuller_test()
        ```

        Which returns:

        |                    | NVDA       | KO         |
        |:-------------------|:-----------|:-----------|
        | SDFC Statistic     | 2.7139     | 1.0338     |
        | Break Date         | 2019-06-03 | 2023-10-05 |
        | P-Value            | 0.004      | 0.21       |
        | Critical Value 10% | 1.4845     | 1.4845     |
        | Critical Value 5%  | 1.8067     | 1.8067     |
        | Critical Value 1%  | 2.4087     | 2.4087     |
        | Observations       | 2514       | 2514       |
        | Replications       | 1000       | 1000       |
        | Explosive (5%)     | True       | False      |
        """
        period = period if period else "daily"
        levels = self._get_levels(period, column, log_prices, include_benchmark)

        if time_series:
            result = pd.DataFrame(
                {
                    ticker: bubble_model.get_chow_dickey_fuller_statistics(
                        levels[ticker], trim=trim
                    )
                    for ticker in levels.columns
                },
                index=levels.index,
            )
        else:
            result = pd.DataFrame(
                {
                    ticker: bubble_model.get_chow_dickey_fuller_test(
                        levels[ticker],
                        trim=trim,
                        simulations=simulations,
                        seed=seed,
                    )
                    for ticker in levels.columns
                }
            )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )
