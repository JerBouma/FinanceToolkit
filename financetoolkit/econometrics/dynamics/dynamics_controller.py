"""Regimes and Dynamics Module"""

__docformat__ = "google"

import numpy as np
import pandas as pd

from financetoolkit.econometrics.dynamics import (
    pairs_trading_model,
    regime_model,
    state_space_model,
)
from financetoolkit.econometrics.helpers import BENCHMARK_TICKER
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import (
    InvalidArgumentError,
    handle_errors,
    validate_window,
)
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import apply_rounding

logger = get_logger()

# ruff: noqa: E501


@filter_shown_columns
@track_sources
class DynamicsEconometrics:
    """
    Regimes and changing relationships: Markov regime switching, structural breaks, PELT change points, the time-varying beta, DCC-GARCH, pairs trading and optimal trading rules on an Ornstein-Uhlenbeck process.
    """

    @handle_errors
    def get_markov_regime_switching(
        self,
        period: str | None = None,
        column: str = "Return",
        regimes: int = 2,
        switching_variance: bool = True,
        probabilities: bool = False,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Fit a Markov regime switching model to every asset: the hidden regimes, such as calm and
        turbulent markets, that set the mean and volatility of its returns.

        The regime is not observed and switches from one period to the next with fixed
        probabilities (Hamilton, 1989), and the model is fitted with statsmodels'
        MarkovRegression. With two regimes the model typically
        separates calm markets (a higher mean and a low volatility) from turbulent ones (a
        lower mean and a high volatility), and it says which regime the market is most
        likely in today, how persistent each regime is and how long it lasts on average.

        The regimes are ordered from the lowest to the highest volatility, so Regime 1 is
        the calmest. Without a switching variance all regimes share one volatility, and
        they are ordered from the lowest to the highest mean instead, so Regime 1 is the
        one with the lowest mean. The expected duration of a regime is 1 / (1 - p), with p
        the probability of staying in it, in periods. The share is the part of the periods
        in which the regime was the most likely one.

        The likelihood of a mixture of regimes has local optima, so the model is fitted from
        its default starting values and from the best of 20 random perturbations of them,
        and the fit with the highest likelihood is kept. The random perturbations come from
        NumPy's global random state, so when competing optima are close, repeated calls can
        differ slightly; call np.random.seed beforehand for results that can be reproduced.

        The formula is as follows:

        - yₜ = μ_Sₜ + σ_Sₜ·εₜ, with yₜ the `column` series (Return by default), εₜ standard normal and
          Sₜ ∈ {1, ..., K} the regime, K = `regimes` (2 by default); σ_Sₜ = σ for every regime without a
          switching variance
        - P(Sₜ = j | Sₜ₋₁ = i) = pᵢⱼ, a first-order Markov chain with constant transition probabilities
        - μ, σ and pᵢⱼ are fitted by Maximum Likelihood with the Hamilton filter (statsmodels
          MarkovRegression), keeping the best of the default start and 20 random perturbations
        - Smoothed probabilities P(Sₜ = j | y₁..y_T) per date; Expected Duration = 1 / (1 - pⱼⱼ) periods;
          Share = the fraction of dates where regime j has the highest smoothed probability; Current
          Regime = that regime on the last date, with its smoothed probability

        For more information, see the following paper:

        - Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of Nonstationary
          Time Series and the Business Cycle." Econometrica, 57(2), 357-384.

        Also known as: Markov switching model, hidden Markov model, regime detection,
        bull and bear regimes.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or
                yearly). Defaults to "daily".
            column (str, optional): The historical data column to model. Defaults to "Return".
            regimes (int, optional): The number of regimes. Defaults to 2.
            switching_variance (bool, optional): Whether the volatility switches with the
                regime as well as the mean. Defaults to True.
            probabilities (bool, optional): Whether to return the smoothed probability of
                every regime per date instead of the summary. Defaults to False.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
                assets. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per asset the Mean, Volatility, Probability of Staying, Expected
            Duration and Share of every regime, and the Current Regime with its probability;
            or, with probabilities=True, the probability of every regime per date.

        Notes:
        - Requires statsmodels (`pip install financetoolkit[econometrics]`).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2015-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_markov_regime_switching(period="weekly")
        ```

        Which returns:

        |                                 |    AAPL |    MSFT |
        |:--------------------------------|--------:|--------:|
        | Regime 1 Mean                   |  0.009  |  0.0065 |
        | Regime 1 Volatility             |  0.0236 |  0.0217 |
        | Regime 1 Probability of Staying |  0.917  |  0.9528 |
        | Regime 1 Expected Duration      | 12.0479 | 21.194  |
        | Regime 1 Share                  |  0.5436 |  0.6411 |
        | Regime 2 Mean                   |  0.0006 |  0.0021 |
        | Regime 2 Volatility             |  0.0488 |  0.0455 |
        | Regime 2 Probability of Staying |  0.91   |  0.9209 |
        | Regime 2 Expected Duration      | 11.1105 | 12.6346 |
        | Regime 2 Share                  |  0.4564 |  0.3589 |
        | Current Regime                  |  1      |  1      |
        | Current Regime Probability      |  0.8877 |  0.8087 |
        """
        period = period if period else "daily"
        data = self._filter_benchmark(
            self._get_series_column(period, column), include_benchmark=include_benchmark
        )

        # A ticker the model cannot be fitted to (too short, constant or without convergence)
        # is left out instead of failing the other tickers.
        fits = {}
        for ticker in data.columns:
            try:
                fits[ticker] = regime_model.get_markov_regime_switching(
                    data[ticker], regimes=regimes, switching_variance=switching_variance
                )
            except InvalidArgumentError:
                # A mistake in the call is not a reason to skip a ticker.
                raise
            except ValueError as error:
                logger.warning(
                    "No Markov regime switching fit for %s: %s", ticker, error
                )

        if not fits:
            raise ValueError("None of the tickers has data to fit the regimes to.")

        if probabilities:
            result = pd.concat(
                {ticker: smoothed for ticker, (_, smoothed) in fits.items()}, axis=1
            )
        else:
            result = pd.DataFrame(
                {ticker: summary for ticker, (summary, _) in fits.items()}
            )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_structural_breaks(
        self,
        period: str | None = None,
        column: str = "Return",
        max_breaks: int = 5,
        minimum_segment: float = 0.15,
        breaks: int | None = None,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Find the structural breaks of every asset's series: the dates at which its mean shifts.

        The breaks follow the Bai and Perron (1998, 2003) method for multiple structural
        breaks. For every number of breaks
        the break dates that minimize the sum of squared residuals around a mean per segment
        are found exactly (by dynamic programming), and the number of breaks is chosen by the
        Bayesian information criterion. Every segment covers at least a share of the
        observations (the integer part of that share times the number of observations, as
        in Bai and Perron), so a break is not fitted to a few outliers.

        The BIC works well for series without serial correlation, such as returns, but tends
        to choose too many breaks when the series is serially correlated (Bai and Perron,
        2003), as levels such as inflation or interest rates are: a persistent swing then
        looks like a shift in the mean. For such series, compare the segments for a few
        values of breaks rather than relying on the chosen number alone.

        Breaks show when a series changed character, such as the regimes of inflation or of
        interest rates, or a change in the average return of an asset. Unlike the Chow test,
        the break dates do not have to be known in advance.

        The formula is as follows:

        - yₜ = μⱼ + εₜ for the dates of segment j = 1, ..., m + 1, with yₜ the `column` series (Return by
          default) and m breaks
        - SSR(m) = min over the break dates of Σⱼ Σ(t in segment j) (yₜ - ȳⱼ)², found exactly by dynamic
          programming, with every segment at least h = max(⌊`minimum_segment`·T⌋, 2) observations
          (`minimum_segment` = 0.15 by default)
        - BIC(m) = T·ln(SSR(m) / T) + (2m + 1)·ln(T), for m = 0, ..., `max_breaks` (5 by default); the m
          with the lowest BIC is chosen unless `breaks` is given
        - Per segment: its Start, End, Observations, Mean ȳⱼ and Standard Deviation (ddof=1)

        For more information, see the following papers:

        - Bai, J. and Perron, P. (1998). "Estimating and Testing Linear Models with Multiple
          Structural Changes." Econometrica, 66(1), 47-78.
        - Bai, J. and Perron, P. (2003). "Computation and Analysis of Multiple Structural
          Change Models." Journal of Applied Econometrics, 18(1), 1-22.

        Also known as: Bai-Perron test, multiple structural breaks, change point detection,
        regime shifts.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or
                yearly). Defaults to "daily".
            column (str, optional): The historical data column to analyze. Defaults to
                "Return"; use "Adj Close" (or "Level" with standalone data) for the series
                itself, such as an inflation rate.
            max_breaks (int, optional): The most breaks considered. Defaults to 5.
            minimum_segment (float, optional): The smallest segment as a share of the
                observations (the trimming). Defaults to 0.15.
            breaks (int | None, optional): The number of breaks to find instead of choosing
                it by the BIC. Defaults to None.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
                assets. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per asset and segment the Start, End, Observations, Mean and
            Standard Deviation, with the BIC of the chosen number of breaks.

        As an example:

        ```python
        from financetoolkit import Economics, Econometrics

        economics = Economics(start_date="1960-01-01", end_date="2025-12-31")

        inflation = economics.get_inflation_rate(countries="United States", period="yearly")

        econometrics = Econometrics(data=inflation)

        econometrics.get_structural_breaks(column="Level", max_breaks=3)
        ```

        Which returns:

        |                      | Start      | End        |   Observations |   Mean |   Standard Deviation |      BIC |
        |:---------------------|:-----------|:-----------|---------------:|-------:|---------------------:|---------:|
        | ('United States', 1) | 1960-12-31 | 1972-12-31 |             13 | 0.0282 |               0.0169 | -519.037 |
        | ('United States', 2) | 1973-12-31 | 1982-12-31 |             10 | 0.0876 |               0.0269 | -519.037 |
        | ('United States', 3) | 1983-12-31 | 2025-12-31 |             43 | 0.0285 |               0.0143 | -519.037 |
        """
        period = period if period else "daily"
        data = self._filter_benchmark(
            self._get_series_column(period, column), include_benchmark=include_benchmark
        )

        # A ticker without breaks to find (a constant or short series) is left out instead of
        # failing the other tickers.
        segments = {}
        for ticker in data.columns:
            try:
                segments[ticker] = regime_model.get_structural_breaks(
                    data[ticker],
                    max_breaks=max_breaks,
                    minimum_segment=minimum_segment,
                    breaks=breaks,
                )
            except InvalidArgumentError:
                # A mistake in the call is not a reason to skip a ticker.
                raise
            except ValueError as error:
                logger.warning("No structural breaks for %s: %s", ticker, error)

        if not segments:
            raise ValueError(
                "None of the tickers has data to find structural breaks in."
            )

        result = pd.concat(segments)
        result.index = result.index.set_names(["Asset", "Segment"])

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_change_points(
        self,
        period: str | None = None,
        column: str = "Return",
        cost: str = "mean_variance",
        penalty: float | None = None,
        minimum_segment: int = 5,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Find the change points of every asset's series: the dates at which its mean, its
        volatility or both change.

        The change points follow the PELT (Pruned Exact Linear Time) method of Killick,
        Fearnhead and Eckley (2012). It finds the exact segmentation that minimizes the Gaussian cost of
        every segment plus a penalty per change, with no upper limit on the number of
        changes, and prunes the candidate dates that can no longer be the last change, so it
        runs in about linear time.

        Where `get_structural_breaks` (Bai and Perron) looks for shifts in the mean with a
        common variance and at most a few breaks, PELT with the default mean and variance cost
        also splits the series where the volatility changes, which for returns separates calm
        and turbulent periods; the variance cost keeps the mean fixed and only lets the
        volatility change, the mean cost only the mean. Every segment is dated after the fact
        with the whole series, as a description of the past rather than a signal on the day.

        The formula is as follows, with yₜ the `column` series (Return by default), n observations
        and a segment s of nₛ observations:

        - Mean and variance cost: C(s) = nₛ·ln(σ̂ₛ²), with σ̂ₛ² = Σ(t in s) (yₜ - ȳₛ)² / nₛ
        - Variance cost: C(s) = nₛ·ln(Σ(t in s) (yₜ - ȳ)² / nₛ), with ȳ the mean of the whole series
        - Mean cost: C(s) = Σ(t in s) (yₜ - ȳₛ)² / σ̂², with σ̂ = MAD(Δy) / (0.6745·√2) the robust
          standard deviation from the changes of the series
        - The segmentation minimizes Σ C(s) + β·(number of changes), every segment at least
          `minimum_segment` observations (5 by default)
        - β = (p + 1)·ln(n), the BIC, with p = 2 parameters per segment for the mean and variance cost and 1
          for the mean or the variance cost plus 1 for the date of the change, unless `penalty` is given
        - Per segment: its Start, End, Observations, Mean and Standard Deviation (ddof=1)

        For more information, see the following papers:

        - Killick, R., Fearnhead, P. and Eckley, I.A. (2012). "Optimal Detection of
        Changepoints With a Linear Computational Cost." Journal of the American Statistical
        Association, 107(500), 1590-1598.
        - Jackson, B. et al. (2005). "An Algorithm for Optimal Partitioning of Data on an
        Interval." IEEE Signal Processing Letters, 12(2), 105-108.

        Also known as: PELT, change point detection, changepoints, variance change points,
        volatility regimes, segmentation, multiple structural breaks.

        Referenced against:
        - ruptures, Pelt: https://github.com/deepcharles/ruptures/blob/v1.1.10/src/ruptures/detection/pelt.py#L11
        - changepoint (R), penalty_decision: https://github.com/cran/changepoint/blob/2.3/R/penalty_decision.R#L1

        Shares similar logic:
        - RiskLabAI, pelt_change_points: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/features/structural_breaks/pelt_change_points.py#L34

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or
                yearly). Defaults to "daily".
            column (str, optional): The historical data column to analyze. Defaults to
                "Return"; use "Adj Close" (or "Level" with standalone data) for the series
                itself, such as an inflation rate.
            cost (str, optional): What may change between segments. Options are: "mean_variance"
                (both), "variance" (the volatility around the mean of the series), "mean" (the
                mean).
                Defaults to "mean_variance".
            penalty (float | None, optional): The penalty β per change; a larger one finds
                fewer changes. Defaults to None, the BIC (p + 1)·ln(n).
            minimum_segment (int, optional): The fewest observations of a segment. A segment of
                two nearly equal values has a variance close to zero, which the Gaussian cost
                rewards without bound, so the variance costs need a few more. Defaults to 5.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
                assets. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per asset and segment the Start, End, Observations, Mean and
            Standard Deviation, with the Penalty used.

        Notes:
        - Daily returns have fat tails, so the Gaussian cost splits them more often than a
          volatility model would: weekly or monthly returns, or a larger penalty, give fewer
          and longer segments.
        - A variance is floored at 1e-6 times that of the whole series, so a segment of
          equal values has a finite cost, unlike ruptures, which adds 1e-6 to it instead.
        - The change points equal those of an exhaustive search; RiskLabAI's
          pelt_change_points calls ruptures.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["QQQ", "TLT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2016-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_change_points(period="monthly", cost="variance")
        ```

        Which returns:

        |            | Start   | End     |   Observations |    Mean |   Standard Deviation |   Penalty |
        |:-----------|:--------|:--------|---------------:|--------:|---------------------:|----------:|
        | ('QQQ', 1) | 2016-02 | 2018-09 |             32 |  0.0196 |               0.0319 |    9.5582 |
        | ('QQQ', 2) | 2018-10 | 2025-12 |             87 |  0.0161 |               0.0587 |    9.5582 |
        | ('TLT', 1) | 2016-02 | 2025-12 |            119 | -0.0002 |               0.0393 |    9.5582 |
        """
        period = period if period else "daily"
        data = self._filter_benchmark(
            self._get_series_column(period, column), include_benchmark=include_benchmark
        )

        # A ticker without change points to find (a constant or short series) is left out
        # instead of failing the other tickers.
        segments = {}
        for ticker in data.columns:
            try:
                segments[ticker] = regime_model.get_change_points(
                    data[ticker],
                    cost=cost,
                    penalty=penalty,
                    minimum_segment=minimum_segment,
                )
            except InvalidArgumentError:
                # A mistake in the call is not a reason to skip a ticker.
                raise
            except ValueError as error:
                logger.warning("No change points for %s: %s", ticker, error)

        if not segments:
            raise ValueError("None of the tickers has data to find change points in.")

        result = pd.concat(segments)
        result.index = result.index.set_names(["Asset", "Segment"])

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_time_varying_beta(
        self,
        dependent_tickers: str | list[str] | None = None,
        independent_ticker: str | None = None,
        period: str | None = None,
        column: str = "Return",
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Estimate a time-varying regression of every asset with the Kalman filter: its alpha and
        beta as they change over time.

        The model is a statsmodels MLEModel state space model. The regression
        y_t = α_t + β_t x_t + ε_t lets both coefficients take a random step every period; how large
        those steps are, and how much noise surrounds the regression, is estimated by
        maximum likelihood. A beta that drifts, such as a stock becoming more sensitive to
        the market, shows up without choosing the window of a rolling regression, while a
        beta that is constant stays flat.

        Every alpha and beta is the filtered estimate, updated with the observations up to
        its date only. The three variances that set how fast the estimates adapt are
        estimated on the whole sample, however, so the path is not entirely what would have
        been known at the time. The filter starts without prior knowledge of the
        coefficients, so the first date (or the first few, when the independent series
        starts with zeros) cannot separate the alpha from the beta and is NaN.

        By default every asset is regressed on the Benchmark of the Toolkit (or, with
        standalone data, on the second series).

        The formula is as follows:

        - yₜ = αₜ + βₜ·xₜ + εₜ, εₜ ~ N(0, σ²_ε), with yₜ the asset's and xₜ the independent asset's `column`
          series (Return by default), on the dates both have
        - αₜ = αₜ₋₁ + ηₜ and βₜ = βₜ₋₁ + ζₜ, ηₜ ~ N(0, σ²_α) and ζₜ ~ N(0, σ²_β): random walks
        - σ²_ε, σ²_α and σ²_β are fitted by Maximum Likelihood with the Kalman filter (diffusely
          initialized) on the series divided by their standard deviations, and scaled back
        - Alpha and Beta are the filtered states E[αₜ, βₜ | y₁..yₜ]; the Beta Standard Error is the square
          root of the filtered variance of βₜ

        For more information, see the following book:

        - Durbin, J. and Koopman, S.J. (2012). "Time Series Analysis by State Space
          Methods." Oxford University Press, 2nd edition.

        Also known as: Kalman filter beta, rolling beta, dynamic beta, state space
        regression.

        Args:
            dependent_tickers (str | list[str] | None, optional): The assets whose beta is
                estimated. Defaults to None, which uses every asset except the independent one.
            independent_ticker (str | None, optional): The asset the beta is measured
                against. Defaults to None, which uses the Benchmark or the second series.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or
                yearly). Defaults to "daily".
            column (str, optional): The historical data column to use. Defaults to "Return".
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per date and asset the Alpha, the Beta and the standard error of
            the beta, NaN on the first date(s) where they are not yet identified.

        Notes:
        - Requires statsmodels (`pip install financetoolkit[econometrics]`).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2015-01-01",
            end_date="2025-12-31",
        )

        beta = toolkit.econometrics.get_time_varying_beta(period="monthly")

        beta.xs("Beta", axis=1, level=1).iloc[12::24]
        ```

        Which returns:

        | Date    |   AAPL |   MSFT |
        |:--------|-------:|-------:|
        | 2016-02 | 1.2708 | 1.5813 |
        | 2018-02 | 1.1427 | 1.3183 |
        | 2020-02 | 1.3197 | 1.1153 |
        | 2022-02 | 1.2496 | 0.9853 |
        | 2024-02 | 1.2526 | 0.9525 |
        """
        period = period if period else "daily"
        # Every pair is aligned on its own common dates by the model.
        data = self._get_series_column(period, column)

        tickers = list(data.columns)
        if independent_ticker is None:
            independent_ticker = (
                BENCHMARK_TICKER
                if BENCHMARK_TICKER in tickers
                else tickers[1] if len(tickers) > 1 else None
            )
        if independent_ticker is None or independent_ticker not in tickers:
            raise InvalidArgumentError(
                f"The independent ticker {independent_ticker} is not available; choose from "
                f"{', '.join(tickers)}."
            )
        dependent = (
            [ticker for ticker in tickers if ticker != independent_ticker]
            if dependent_tickers is None
            else (
                [dependent_tickers]
                if isinstance(dependent_tickers, str)
                else list(dependent_tickers)
            )
        )

        # A ticker without a beta to estimate (too short or constant) is left out instead of
        # failing the other tickers.
        betas = {}
        for ticker in dependent:
            try:
                betas[ticker] = state_space_model.get_time_varying_beta(
                    data[ticker], data[independent_ticker]
                )[0]
            except InvalidArgumentError:
                # A mistake in the call is not a reason to skip a ticker.
                raise
            except ValueError as error:
                logger.warning("No time-varying beta for %s: %s", ticker, error)

        if not betas:
            raise ValueError("None of the tickers has data for a time-varying beta.")

        result = pd.concat(betas, axis=1)

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_dcc_garch(
        self,
        period: str | None = None,
        column: str = "Return",
        parameters: bool = False,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Estimate the dynamic conditional correlation between every pair of assets: how it changes
        over time, with the DCC-GARCH(1,1) model of Engle (2002).

        Every
        asset gets its own GARCH(1,1) volatility, and the correlation of the standardized
        returns follows a GARCH-like recursion around their long-run correlation. Correlations
        between assets tend to rise in crises, exactly when diversification is needed most.
        A rolling correlation sees that late and depends on its window; the DCC model
        updates the correlation every period, with a (how strongly it reacts to news) and b
        (how persistent it is) estimated by maximum likelihood.

        The formula is as follows:

        - zᵢ,ₜ = (rᵢ,ₜ - mean(rᵢ)) / σᵢ,ₜ, with rᵢ,ₜ the `column` series (Return by default) on the dates all
          assets have and σ²ᵢ,ₜ = ω + α·uᵢ,ₜ₋₁² + β·σ²ᵢ,ₜ₋₁ each asset's GARCH(1,1) variance of the demeaned
          returns u (GARCH Persistence = α + β)
        - Qₜ = (1 - a - b)·Q̄ + a·zₜ₋₁·zₜ₋₁' + b·Qₜ₋₁, with Q̄ the sample correlation of z and Q₀ = Q̄
        - Rₜ = diag(Qₜ)^(-1/2)·Qₜ·diag(Qₜ)^(-1/2), the conditional correlations
        - a and b maximize the correlation log-likelihood -½·Σₜ (ln|Rₜ| + zₜ'·Rₜ⁻¹·zₜ), with a, b ≥ 0 and
          a + b < 1

        For more information, see the following paper:

        - Engle, R. (2002). "Dynamic Conditional Correlation: A Simple Class of
          Multivariate Generalized Autoregressive Conditional Heteroskedasticity Models."
          Journal of Business & Economic Statistics, 20(3), 339-350.

        Also known as: DCC-GARCH, dynamic conditional correlation, time-varying
        correlation, multivariate GARCH.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or
                yearly). Defaults to "daily".
            column (str, optional): The historical data column to use. Defaults to "Return".
            parameters (bool, optional): Whether to return the estimated parameters (a, b and
                the GARCH persistence of every asset) instead of the correlations. Defaults
                to False.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
                assets. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The conditional correlation of every pair of assets per date, or
            the parameters of the model.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2018-01-01",
            end_date="2025-12-31",
        )

        correlations = toolkit.econometrics.get_dcc_garch()

        correlations.resample("Y").mean().round(4)
        ```

        Which returns:

        | Date   |   AAPL / MSFT |   AAPL / XOM |   MSFT / XOM |
        |:-------|--------------:|-------------:|-------------:|
        | 2018   |        0.6047 |       0.2794 |       0.2729 |
        | 2019   |        0.578  |       0.2982 |       0.2641 |
        | 2020   |        0.6665 |       0.2879 |       0.226  |
        | 2021   |        0.6199 |       0.1384 |       0.0786 |
        | 2022   |        0.7084 |       0.2283 |       0.1692 |
        | 2023   |        0.5802 |       0.1706 |       0.0386 |
        | 2024   |        0.5378 |       0.1018 |       0.0237 |
        | 2025   |        0.4636 |       0.1991 |       0.114  |
        """
        period = period if period else "daily"
        # The model aligns the assets on the dates they share, after the Benchmark is left
        # out, so the Benchmark's own gaps do not shorten the sample when it is excluded.
        data = self._filter_benchmark(
            self._get_series_column(period, column), include_benchmark=include_benchmark
        )

        correlations, estimates = state_space_model.get_dcc_garch(data)
        result = (
            pd.Series(estimates, name="Value").to_frame()
            if parameters
            else correlations
        )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_pairs_trading(
        self,
        period: str | None = None,
        column: str = "Adj Close",
        rolling: int | None = None,
        entry_z_score: float = 2.0,
        exit_z_score: float = 0.5,
        log_prices: bool = True,
        time_series: bool = False,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the statistics of trading the spread between every pair of assets: how far the
        spread is from its mean and the position that gives.

        The hedge ratio follows from regressing one log price on the other; what remains is the
        spread, and when the two are cointegrated the spread reverts to its mean. Its
        z-score says how far it is from that mean in standard deviations, and threshold
        rules trade on it: short the spread (sell the first asset, buy the hedge ratio of
        the second) once the z-score rises above the entry threshold, buy it once it falls
        below minus the threshold, and close the position once the z-score crosses the exit
        threshold on its way back (below it for a short, above minus it for a long). A
        z-score that jumps past the band in one period still closes the position, and one
        that lands beyond the entry threshold on the other side reverses it. The half-life
        says how many periods the spread takes to close half its gap to the mean.

        Gatev, Goetzmann and Rouwenhorst (2006) open a position when the normalized prices
        of a pair drift two standard deviations (of their formation period) apart and close
        it when the prices cross, which is an exit threshold of 0; closing at 0.5 is a
        common variant that takes profits before the spread is fully back at its mean.

        The hedge ratio and intercept are estimated on the whole period, so the spread, and
        with it every z-score and signal, uses information from after its date. A rolling
        window only makes the mean and standard deviation of the z-score backward-looking;
        a backtest without look-ahead estimates the hedge ratio on a formation period and
        trades in the period after it. The Engle-Granger p-value says whether there is a
        mean-reverting relationship to trade at all (below 0.05).

        The formula is as follows:

        - yₜ = c + h·xₜ + sₜ, fitted by OLS on the dates both have, with yₜ and xₜ the (log, by
          default) `column` levels (Adj Close by default): h the Hedge Ratio, c the Intercept and
          sₜ the Spread
        - Z-Scoreₜ = (sₜ - mean(s)) / std(s) (ddof=1), over the full sample or a rolling window
        - Signal: -1 from Z > `entry_z_score` until Z < `exit_z_score`, +1 from Z < -`entry_z_score`
          until Z > -`exit_z_score`, 0 otherwise (2.0 and 0.5 by default)
        - Half-Life = -ln(2) / ln(1 + λ), with λ from Δsₜ = a + λ·sₜ₋₁ + εₜ (infinite when λ is (nearly)
          zero or positive)
        - Cointegration P-Value: the Engle-Granger test of yₜ on xₜ

        For more information, see the following papers:

        - Gatev, E., Goetzmann, W.N. and Rouwenhorst, K.G. (2006). "Pairs Trading:
          Performance of a Relative-Value Arbitrage Rule." The Review of Financial Studies,
          19(3), 797-827.
        - Engle, R.F. and Granger, C.W.J. (1987). "Co-Integration and Error Correction:
          Representation, Estimation, and Testing." Econometrica, 55(2), 251-276.

        Also known as: statistical arbitrage, spread trading, z-score strategy, half-life
        of mean reversion.

        Referenced against:
        - statsmodels, coint: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or
                yearly). Defaults to "daily".
            column (str, optional): The price column. Defaults to "Adj Close".
            rolling (int | None, optional): The number of periods of a rolling window for the
                mean and standard deviation of the z-score, at least 2. Defaults to None, which
                uses the full sample.
            entry_z_score (float, optional): The z-score at which a position is opened.
                Defaults to 2.0.
            exit_z_score (float, optional): The z-score whose crossing on the way back closes
                a position (0 closes when the spread is back at its mean). Defaults to 0.5.
            log_prices (bool, optional): Whether to use log prices, so the hedge ratio is an
                elasticity. Defaults to True.
            time_series (bool, optional): Whether to return the Spread, Z-Score and Signal
                per date instead of the summary. Defaults to False.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
                assets. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per pair the Hedge Ratio, Intercept, Half-Life (in periods),
            Cointegration P-Value, the latest Z-Score and Signal (1 long, -1 short, 0 flat)
            and the Number of Trades (positions opened, a reversal counting as a new one);
            or, with time_series=True, the Spread, Z-Score and Signal per date.

        Notes:
        - Requires statsmodels (`pip install financetoolkit[econometrics]`).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["KO", "PEP"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2015-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_pairs_trading(rolling=60)
        ```

        Which returns:

        |               |   Hedge Ratio |   Intercept |   Half-Life |   Cointegration P-Value |   Z-Score |   Signal |   Number of Trades |
        |:--------------|--------------:|------------:|------------:|------------------------:|----------:|---------:|-------------------:|
        | ('KO', 'PEP') |        0.8685 |     -0.3098 |     290.448 |                  0.8345 |     0.684 |        0 |                 42 |
        """
        validate_window("rolling", rolling, 2)
        period = period if period else "daily"
        # Every pair is aligned on its own common dates by the model.
        prices = self._get_series_column(period, column)

        fits = {
            (ticker_a, ticker_b): pairs_trading_model.get_pairs_trading(
                prices[ticker_a],
                prices[ticker_b],
                window=rolling,
                entry=entry_z_score,
                exit_=exit_z_score,
                log_prices=log_prices,
            )
            for ticker_a, ticker_b in self._all_unordered_pairs(
                include_benchmark=include_benchmark
            )
        }

        if time_series:
            result = pd.concat(
                {f"{a} / {b}": series for (a, b), (series, _) in fits.items()}, axis=1
            )
        else:
            result = pd.DataFrame(
                {pair: summary for pair, (_, summary) in fits.items()}
            ).T
            result.index = result.index.set_names(["Dependent", "Independent"])

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_optimal_trading_rule(
        self,
        period: str | None = None,
        column: str = "Adj Close",
        maximum_threshold: float = 10.0,
        grid_size: int = 20,
        maximum_holding_period: int = 100,
        simulations: int = 10000,
        seed: int = 0,
        log_prices: bool = True,
        grid: bool = False,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Find the optimal trading rule of every asset: the profit-taking and stop-loss levels with
        the highest Sharpe ratio for a long position.

        The log price follows the mean-reverting Ornstein-Uhlenbeck process fitted to it, with
        the Monte Carlo method of López de Prado (2018, chapter 13).

        The process is fitted to the log prices by a regression of each on the one before it,
        which gives how persistent the price is (Phi), the level it reverts to (the Long-Run
        Mean) and the size of its shocks (Sigma). Thousands of paths of that process are then
        simulated from the last price, and every pair of a profit-taking and a stop-loss level
        on a mesh, in units of Sigma, is tried on the same paths: a position is closed once
        the price rises above the profit-taking level, falls below the stop-loss level or has
        been held for the maximum holding period. The rule with the highest Sharpe ratio of
        the gains is the optimal one. A price far below its long-run mean (a positive
        Forecast) favours a wide profit-taking level, one above it a tight stop-loss; for a
        price that hardly reverts (Phi close to 1, a long Half-Life) the Sharpe ratios are
        close to zero and the rule matters little.

        The formula is as follows, with pₜ the log of the `column` (Adj Close by default):

        - pₜ = c + φ·pₜ₋₁ + σ·εₜ by OLS, with σ the standard deviation of the residuals (n - 2 degrees of
          freedom), Long-Run Mean = c / (1 - φ), Half-Life = -ln(2) / ln(φ) and
          Forecast = (Long-Run Mean - p₀) / σ, with p₀ the last log price
        - yₜ = (pₜ - p₀) / σ = (1 - φ)·Forecast + φ·yₜ₋₁ + εₜ, y₀ = 0, εₜ ~ N(0, 1), for `simulations` paths
          (10,000 by default) of at most `maximum_holding_period` steps (100 by default)
        - For every profit-taking π̄ and stop-loss π̲ in {h, 2h, ..., `maximum_threshold`},
          h = `maximum_threshold` / `grid_size` (0.5 to 10 by default): the gain of a path is yₜ at the
          first t with yₜ > π̄ or yₜ < -π̲, or at the maximum holding period
        - Sharpe Ratio = mean / standard deviation (ddof=1) of the gains; the rule with the highest
          is reported with its Mean and Standard Deviation (in σ) and Average Holding Period

        For more information, see the following book:

        - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, chapter
        13, "Backtesting on Synthetic Data".

        Also known as: optimal trading rule, OTR, profit-taking and stop-loss, Ornstein-Uhlenbeck
        trading rule, mean reversion trading rule, synthetic backtest.

        Referenced against:
        - RiskLabAI, ou_rule_metrics: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/backtest/ou_trading_rules.py#L148

        Shares similar logic:
        - RiskLabAI, fit_ornstein_uhlenbeck: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/backtest/ou_trading_rules.py#L269
        - RiskLabAI, optimal_ou_trading_rule: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/backtest/ou_trading_rules.py#L198

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or
                yearly). Defaults to "daily".
            column (str, optional): The price column. Defaults to "Adj Close".
            maximum_threshold (float, optional): The widest profit-taking and stop-loss level,
                in units of Sigma. Defaults to 10.0.
            grid_size (int, optional): The number of levels on each side of the mesh, so the
                mesh has grid_size² rules. Defaults to 20.
            maximum_holding_period (int, optional): The most periods a position is held.
                Defaults to 100.
            simulations (int, optional): The number of simulated paths. Defaults to 10000.
            seed (int, optional): The seed of the simulation. Defaults to 0.
            log_prices (bool, optional): Whether to fit the process to the log prices. Pass False
                for a series that can be zero or negative, such as a spread. Defaults to True.
            grid (bool, optional): Whether to return every rule of the mesh instead of the
                optimal one. Defaults to False.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
                assets. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per asset the Phi, Half-Life (in periods), Sigma, Long-Run Mean (of the
            log price), Forecast, the optimal Profit-Taking and Stop-Loss (in Sigma), and the
            Mean and Standard Deviation of the gain (in Sigma), its Sharpe Ratio and the Average
            Holding Period; with grid=True the Mean, Standard Deviation, Sharpe Ratio and
            Average Holding Period of every rule, per asset, Profit-Taking and Stop-Loss.

        Notes:
        - Every rule is tried on the same paths, so the differences between neighbouring rules
          are not simulation noise; the paths follow from the seed, so the result is the same
          on every run. With a wide stop-loss a few large losses decide the standard deviation,
          so the Sharpe ratio of such a rule can move by 0.1 or more with another seed at
          10,000 paths; more paths narrow it.
        - RiskLabAI's ou_rule_metrics gives the closed-form gain and exit time of the continuous
          process, which the simulation agrees with once the levels are moved out by 0.5826 Sigma
          for the discrete monitoring (Broadie, Glasserman and Kou, 1997).
        - Unlike here, López de Prado draws new paths for every rule, closes a position one period after the
          maximum holding period and takes the standard deviation with ddof=0.
        - The rule is for a long position. For a short position, analyze the negative of the
          log prices (`Econometrics(data=...)` without log prices), whose rise is a fall
          in the price.
        - The simulation takes about a second per asset with the defaults.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["PFE", "KO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_optimal_trading_rule()
        ```

        Which returns:

        |                        |      PFE |       KO |
        |:-----------------------|---------:|---------:|
        | Phi                    |   0.9977 |   0.9969 |
        | Half-Life              | 303.568  | 222.457  |
        | Sigma                  |   0.0158 |   0.01   |
        | Long-Run Mean          |   3.3005 |   4.1335 |
        | Forecast               |   9.5111 |  -9.4587 |
        | Profit-Taking          |   6      |   0.5    |
        | Stop-Loss              |  10      |   0.5    |
        | Mean                   |   1.3971 |  -0.0498 |
        | Standard Deviation     |   7.018  |   1.2679 |
        | Sharpe Ratio           |   0.1991 |  -0.0393 |
        | Average Holding Period |  55.5542 |   1.5936 |
        """
        period = period if period else "daily"
        prices = self._filter_benchmark(
            self._get_series_column(period, column), include_benchmark=include_benchmark
        )

        if log_prices:
            if (prices <= 0).any().any():
                raise ValueError(
                    "The logarithm needs positive values; pass log_prices=False for a series "
                    "that can be zero or negative, such as a spread."
                )
            prices = np.log(prices)

        # A ticker without an Ornstein-Uhlenbeck fit (too short or without noise) is left out
        # instead of failing the other tickers.
        fits = {}
        for ticker in prices.columns:
            try:
                fits[ticker] = pairs_trading_model.get_optimal_trading_rule(
                    prices[ticker],
                    maximum_threshold=maximum_threshold,
                    grid_size=grid_size,
                    maximum_holding_period=maximum_holding_period,
                    simulations=simulations,
                    seed=seed,
                )
            except InvalidArgumentError:
                # A mistake in the call is not a reason to skip a ticker.
                raise
            except ValueError as error:
                logger.warning("No optimal trading rule for %s: %s", ticker, error)

        if not fits:
            raise ValueError(
                "None of the tickers has data to fit an Ornstein-Uhlenbeck process to."
            )

        if grid:
            result = pd.concat({ticker: mesh for ticker, (_, mesh) in fits.items()})
            result.index = result.index.set_names(
                ["Asset", "Profit-Taking", "Stop-Loss"]
            )
        else:
            result = pd.DataFrame(
                {ticker: summary for ticker, (summary, _) in fits.items()}
            )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )
