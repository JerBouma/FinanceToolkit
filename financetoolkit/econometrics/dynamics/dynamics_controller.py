"""Regimes and Dynamics Module"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.econometrics.base.base_controller import (
    BENCHMARK_TICKER,
    EconometricsBase,
)
from financetoolkit.econometrics.dynamics import (
    pairs_trading_model,
    regime_model,
    state_space_model,
)
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import apply_rounding

# ruff: noqa: E501


@filter_shown_columns
@track_sources
class DynamicsEconometrics(EconometricsBase):
    """
    Regimes and changing relationships: Markov regime switching, structural breaks, the time-varying beta, DCC-GARCH and pairs trading.
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
        Fit a Markov regime switching model to every asset: the mean and volatility of its
        returns depend on a regime that is not observed and switches from one period to the
        next with fixed probabilities (Hamilton, 1989). With two regimes the model typically
        separates calm markets (a higher mean and a low volatility) from turbulent ones (a
        lower mean and a high volatility), and it says which regime the market is most
        likely in today, how persistent each regime is and how long it lasts on average.

        The regimes are ordered from the lowest to the highest volatility, so Regime 1 is
        the calmest. With switching_variance=False all regimes share one volatility, and
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
          Sₜ ∈ {1, ..., K} the regime, K = `regimes` (2 by default); σ_Sₜ = σ for every regime when
          switching_variance=False
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

        fits = {
            ticker: regime_model.get_markov_regime_switching(
                data[ticker], regimes=regimes, switching_variance=switching_variance
            )
            for ticker in data.columns
        }

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
        Find the dates at which the mean of every asset's series shifts, with the Bai and
        Perron (1998, 2003) method for multiple structural breaks. For every number of breaks
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

        result = pd.concat(
            {
                ticker: regime_model.get_structural_breaks(
                    data[ticker],
                    max_breaks=max_breaks,
                    minimum_segment=minimum_segment,
                    breaks=breaks,
                )
                for ticker in data.columns
            }
        )
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
        Estimate the alpha and beta of every asset as they change over time, with the Kalman
        filter. The regression y_t = α_t + β_t x_t + ε_t lets both coefficients take a random
        step every period; how large those steps are, and how much noise surrounds the
        regression, is estimated by maximum likelihood. A beta that drifts, such as a stock
        becoming more sensitive to the market, shows up without choosing the window of a
        rolling regression, while a beta that is constant stays flat.

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
            raise ValueError(
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

        result = pd.concat(
            {
                ticker: state_space_model.get_time_varying_beta(
                    data[ticker], data[independent_ticker]
                )[0]
                for ticker in dependent
            },
            axis=1,
        )

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
        Estimate the correlation between every pair of assets as it changes over time, with
        the dynamic conditional correlation (DCC) GARCH(1,1) model of Engle (2002). Every
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
        window: int | None = None,
        entry_z_score: float = 2.0,
        exit_z_score: float = 0.5,
        log_prices: bool = True,
        time_series: bool = False,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the statistics of trading the spread between every pair of assets. The
        hedge ratio follows from regressing one log price on the other; what remains is the
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

        - yₜ = c + h·xₜ + sₜ, fitted by OLS on the dates both have, with yₜ and xₜ the (log, when
          log_prices=True) `column` levels (Adj Close by default): h the Hedge Ratio, c the Intercept and
          sₜ the Spread
        - Z-Scoreₜ = (sₜ - mean(s)) / std(s) (ddof=1), over the full sample or a rolling `window`
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

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or
                yearly). Defaults to "daily".
            column (str, optional): The price column. Defaults to "Adj Close".
            window (int | None, optional): The rolling window of the z-score. Defaults to
                None, which uses the full sample.
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

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["KO", "PEP"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2015-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_pairs_trading(window=60)
        ```

        Which returns:

        |               |   Hedge Ratio |   Intercept |   Half-Life |   Cointegration P-Value |   Z-Score |   Signal |   Number of Trades |
        |:--------------|--------------:|------------:|------------:|------------------------:|----------:|---------:|-------------------:|
        | ('KO', 'PEP') |        0.8685 |     -0.3098 |     290.448 |                  0.8345 |     0.684 |        0 |                 42 |
        """
        period = period if period else "daily"
        # Every pair is aligned on its own common dates by the model.
        prices = self._get_series_column(period, column)

        fits = {
            (ticker_a, ticker_b): pairs_trading_model.get_pairs_trading(
                prices[ticker_a],
                prices[ticker_b],
                window=window,
                entry=entry_z_score,
                exit_=exit_z_score,
                log_prices=log_prices,
            )
            for ticker_a, ticker_b in self._all_unordered_pairs(
                include_benchmark=include_benchmark
            )
        }
        if not fits:
            raise ValueError("Pairs trading needs at least two assets.")

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
