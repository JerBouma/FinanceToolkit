"""Value at Risk Module"""

__docformat__ = "google"

import functools

import numpy as np
import pandas as pd

from financetoolkit.helpers import handle_portfolio
from financetoolkit.risk.helpers import as_float, compound_into_shared_dates
from financetoolkit.risk.value_at_risk import (
    backtesting_model,
    cvar_model,
    evar_model,
    var_model,
)
from financetoolkit.risk.volatility import garch_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import (
    PERIODS_PER_YEAR,
    apply_rounding,
    finalize_dataset,
)

# ruff: noqa: E501

logger = get_logger()

# The tests of get_var_backtest with the name of their P-Value row when several are run.
VAR_BACKTESTS = {
    "kupiec": "Kupiec",
    "christoffersen": "Christoffersen",
    "conditional_coverage": "Conditional Coverage",
    "duration": "Duration",
    "mcneil_frey": "McNeil-Frey",
    "tick_loss": "Tick Loss",
}


def _default_window_size(period: str, distribution: str) -> int:
    """
    The rolling window of the backtests: a year of observations with a minimum of 20, as
    get_move_significance uses (252 is a year of daily returns but more than the history
    holds of quarterly or yearly ones), and two years with a minimum of 100 for a GARCH
    fit, which needs a longer sample (var_model.MINIMUM_GARCH_OBSERVATIONS).
    """
    if distribution == "garch":
        return max(2 * PERIODS_PER_YEAR[period], var_model.MINIMUM_GARCH_OBSERVATIONS)

    return max(PERIODS_PER_YEAR[period], 20)


def _get_out_of_sample_estimates(
    returns: pd.DataFrame,
    measure: str,
    distribution: str,
    alpha: float,
    window_size: int,
    refit_every: int = 21,
    garch_method: str = "garch",
    garch_distribution: str = "normal",
) -> pd.DataFrame:
    """
    Builds the rolling, out-of-sample Value at Risk, Conditional Value at Risk or volatility
    path of every ticker, as used by the VaR and Expected Shortfall backtests.

    Each ticker is rolled over its own non-missing returns, so a ticker with a shorter
    history (or a missing day) does not shorten the sample of the others, and the window
    always holds `window_size` actual observations. The path is shifted by one
    observation so that every estimate only uses the returns before the one it is
    compared against.

    Args:
        returns (pd.DataFrame): The returns, one column per ticker.
        measure (str): "var", "cvar" or "volatility" (the sample standard deviation of the
            window, ddof=1, or the GARCH volatility forecast).
        distribution (str): One of "historic", "gaussian", "studentt", "evt" or "garch".
        alpha (float): The tail probability (e.g., 0.05 for the 95% confidence level).
        window_size (int): The number of observations in each rolling window.
        refit_every (int, optional): The number of returns between two GARCH fits. Defaults
            to 21.
        garch_method (str, optional): The GARCH variance model. Defaults to "garch".
        garch_distribution (str, optional): The GARCH innovation distribution. Defaults to
            "normal".

    Returns:
        pd.DataFrame: The out-of-sample estimates, on the index of `returns`.
    """
    parametric_estimators = {
        ("var", "gaussian"): var_model.get_var_gaussian,
        ("var", "studentt"): var_model.get_var_studentt,
        ("var", "evt"): var_model.get_var_evt,
        ("cvar", "gaussian"): cvar_model.get_cvar_gaussian,
        ("cvar", "studentt"): cvar_model.get_cvar_studentt,
        ("cvar", "evt"): cvar_model.get_cvar_evt,
    }

    if distribution == "evt" and measure != "volatility":
        # The fit needs MINIMUM_NUMBER_OF_EXCEEDANCES losses above the 95th percentile in a window, see var_model.fit_gpd_tail. A shorter window gives NaN everywhere, so that is reported once instead of fitting every window.  # noqa: E501
        exceedances_per_window = var_model.get_number_of_exceedances(
            pd.DataFrame(np.arange(window_size, dtype=float)), 0.95
        ).iloc[0]

        if exceedances_per_window < var_model.MINIMUM_NUMBER_OF_EXCEEDANCES:
            logger.warning(
                "A window of %s returns holds %s losses above the 95th percentile, fewer than the %s the "
                "Extreme Value Theory fit needs, so the estimates are NaN. Use a window_size of at least %s.",
                window_size,
                int(exceedances_per_window),
                var_model.MINIMUM_NUMBER_OF_EXCEEDANCES,
                int(np.ceil(var_model.MINIMUM_NUMBER_OF_EXCEEDANCES / 0.05)),
            )

            return pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)

        parametric_estimators[(measure, distribution)] = functools.partial(
            parametric_estimators[(measure, distribution)], warn=False
        )

    estimates = {}

    for column in returns.columns:
        column_returns = returns[column].dropna()

        if distribution == "garch":
            # One rolling fit serves the VaR, the CVaR and the volatility, see garch_model.get_garch_family_rolling_forecast.  # noqa: E501
            if len(column_returns) < window_size:
                estimates[column] = pd.Series(np.nan, index=returns.index)
                continue
            if measure == "volatility":
                estimate = garch_model.get_garch_family_rolling_forecast(
                    column_returns,
                    window_size,
                    refit_every,
                    garch_method,
                    garch_distribution,
                    "constant",
                )["Volatility"]
            else:
                estimate = var_model.get_var_garch(
                    column_returns,
                    alpha,
                    window_size,
                    refit_every,
                    garch_method,
                    garch_distribution,
                    expected_shortfall=measure == "cvar",
                )
        elif measure == "volatility":
            estimate = column_returns.rolling(window=window_size).std(ddof=1)
        elif distribution == "historic":
            estimate = (
                var_model.get_rolling_var_historic(column_returns, alpha, window_size)
                if measure == "var"
                else cvar_model.get_rolling_cvar_historic(
                    column_returns, alpha, window_size
                )
            )
        elif distribution == "gaussian":
            # The gaussian estimate only needs the mean and the sample standard deviation, which
            # a rolling window gives in one pass, so the same formula is applied to it directly.
            estimate = parametric_estimators[(measure, distribution)](
                column_returns.rolling(window=window_size), alpha
            )
        else:
            # The Student-t and EVT estimates fit a distribution to every window, which takes
            # seconds per thousand windows.
            estimator = parametric_estimators[(measure, distribution)]
            estimate = column_returns.rolling(window=window_size).apply(
                lambda window, estimator=estimator: as_float(estimator(window, alpha)),
                raw=False,
            )

        # A trailing rolling window includes the current observation (and a GARCH forecast is dated by the last return it uses), so without the shift each estimate would be compared against a return it was computed from.  # noqa: E501
        estimates[column] = estimate.shift(1).reindex(returns.index)

    return pd.DataFrame(estimates, index=returns.index, columns=returns.columns)


@filter_shown_columns
@track_sources
class ValueAtRiskRisk:
    """
    Value at Risk and its relatives: conditional and entropic VaR, marginal and component VaR, and the VaR and expected shortfall backtests.
    """

    @handle_portfolio
    @handle_errors
    def get_value_at_risk(
        self,
        period: str | None = None,
        alpha: float = 0.05,
        within_period: bool = False,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        distribution: str = "historic",
        threshold_percentile: float = 0.95,
        refit_every: int = 21,
        garch_method: str = "garch",
        garch_distribution: str = "normal",
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Value at Risk (VaR) of an investment portfolio or asset's returns.

        Value at Risk (VaR) is a risk management metric that quantifies the maximum potential loss
        an investment portfolio or asset may experience over a specified time horizon and confidence level.
        It provides insights into the downside risk associated with an investment and helps investors make
        informed decisions about risk tolerance.

        The VaR is calculated as the quantile of the return distribution, representing the loss threshold
        that is only exceeded with tail probability alpha (e.g., in 5% of the periods for alpha=0.05, the
        95% confidence level).

        Value at Risk is a quantile: it says how bad a period gets at the chosen confidence level, not how
        bad the periods beyond it are, so two assets with the same VaR can have very different tails (see
        the Conditional Value at Risk for that). Historical VaR depends heavily on the window: one calm year
        gives a much smaller figure than a window that includes a crisis, so compare assets over the same
        period and confidence level. Scaling a daily VaR to a longer horizon with the square root of time
        assumes independent returns, which tends to understate risk when volatility clusters.

        The formula is as follows (with the default distribution="historic" and alpha=0.05):

        - VaR = Quantile(Returns, alpha)

        The quantile interpolates linearly between the two nearest returns and the VaR is reported as a return
        (negative for a loss). With `rolling`, the same quantile is taken over each trailing window of `rolling`
        returns. The other distributions replace the empirical quantile with:

        - Gaussian VaR = Mean + z * Std
        - Cornish-Fisher VaR = Mean + z_cf * Std
        - z_cf = z + (z^2 - 1) * S / 6 + (z^3 - 3 * z) * K / 24 - (2 * z^3 - 5 * z) * S^2 / 36
        - Student-T VaR = Location + Scale * t_v^-1(alpha)
        - EVT VaR = -(u + (Scale_GPD / xi) * ((alpha / p)^(-xi) - 1))
        - Kernel VaR solves mean(Φ((VaR - Return_i) / h)) = alpha, with h = Std * (4 / (3 * n))^(1/5)
        - GARCH VaR = Mu + Volatility_(t+1) * F^-1(alpha)

        Where z is the standard normal quantile at alpha, Std the sample standard deviation (ddof=1), S the
        sample skewness and K the sample excess kurtosis. The Student-T degrees of freedom v, Location and Scale
        are fitted by maximum likelihood. For "evt", u is the loss at `threshold_percentile`, xi and Scale_GPD the
        shape and scale of the Generalized Pareto Distribution fitted to the losses above it and p the fraction of
        losses above it. The kernel VaR is the alpha quantile of a Gaussian kernel density of the n returns with
        Silverman's bandwidth h, a smoothed historical quantile (Gourieroux, Laurent and Scaillet 2000), and Φ the
        standard normal distribution function.

        The GARCH VaR is the conditional VaR of the next period: Mu is the constant mean and Volatility_(t+1) the
        one-step volatility forecast of the GARCH-family model `garch_method` with innovations `garch_distribution`,
        fitted by maximum likelihood (see `get_garch`), and F^-1 the quantile of the standardized innovations with
        their fitted Shape and Skew. It responds to the latest volatility, where the other distributions weigh every
        return of the period equally. With `rolling`, the model is refitted every `refit_every` returns on the
        `rolling` returns before it, and the value at each date is the VaR of the next period given the returns up
        to that date (McNeil and Frey 2000).

        For more information about the kernel and GARCH VaR, see the following sources:

        - Gourieroux, C., Laurent, J.P. and Scaillet, O. (2000). "Sensitivity Analysis of Values at Risk."
        Journal of Empirical Finance, 7(3-4), 225-245.
        - Silverman, B.W. (1986). "Density Estimation for Statistics and Data Analysis." Chapman and Hall.
        - McNeil, A.J. and Frey, R. (2000). "Estimation of Tail-Related Risk Measures for Heteroscedastic
        Financial Time Series: An Extreme Value Approach." Journal of Empirical Finance, 7(3-4), 271-300.
        - rugarch: ugarchroll and its VaR (the rolling GARCH VaR, https://cran.r-project.org/package=rugarch).

        Also known as: VaR, maximum expected loss, portfolio loss risk.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            alpha (float, optional): The tail probability for the VaR calculation (e.g., 0.05 for the 95%
            confidence level). Defaults to 0.05.
            within_period (bool, optional): Whether to calculate VaR within the specified period or for the entire
            period. Thus whether to look at the VaR within a specific year (if period = 'yearly') or look at the entirety
            of all years. Defaults to False. Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rolling (int, optional): The rolling window size to use for the calculation. If set, VaR is
            calculated over a rolling window of this many periods across the full return history instead
            of per `period` (e.g. a rolling 60-day VaR). Only available for
            `distribution="historic"` and `distribution="garch"` (the number of returns of every fit, at
            least 100); see `get_var_backtest` for a rolling, out-of-sample VaR path under the other
            parametric distributions. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the VaR values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            distribution (str): The distribution to use for the VaR calculations (historic, gaussian,
            cornish-fisher, studentt, evt, kernel or garch). Defaults to "historic". The parametric distributions use the
            sample standard deviation; "cornish-fisher" adjusts the gaussian quantile for the skewness and
            excess kurtosis of the returns (see `var_model.get_var_cornish_fisher`), "studentt" uses the
            quantile of a Student-T distribution fitted by maximum likelihood and "kernel" the quantile of a
            Gaussian kernel density of the returns. "garch" fits a GARCH-family model per ticker (and per period with
            `within_period`), which takes one to five seconds per fit and needs at least 100 returns; a ticker
            or period with fewer is NaN.
            threshold_percentile (float, optional): Only used when `distribution` is "evt". The percentile
            of losses above which the Generalized Pareto Distribution is fitted; `alpha` can be at most
            1 - `threshold_percentile`, and a larger `alpha` gives an empty result (with a logged error).
            The fit needs at least 10 losses above the threshold (about 200 returns at 0.95); a period
            with fewer is NaN, with a logged warning. Defaults to 0.95.
            refit_every (int, optional): Only used with `distribution="garch"` and `rolling`. The number of
            returns between two fits; the fitted model filters the returns in between. Every refit after the
            first starts from the previous fit and takes a fraction of a second. Defaults to 21.
            garch_method (str, optional): Only used with `distribution="garch"`. The variance model, one of the
            methods of `get_garch` (garch, gjr, egarch, igarch, aparch, tgarch, ngarch, avgarch, nagarch or
            component). Defaults to "garch".
            garch_distribution (str, optional): Only used with `distribution="garch"`. The innovation
            distribution (normal, t, skewt, skewt_fernandez_steel, ged or johnson_su). Defaults to "normal".
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: VaR values with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates VaR for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of VaR values using the specified `lag`.
        - The kernel VaR equals the quantile of scipy's gaussian_kde with bw_method="silverman". It differs
        from PerformanceAnalytics' VaR(method="kernel"), which is the kernel estimator of the component VaR of
        Epperlein and Smillie (2006) and is not defined for a single asset.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["BTCUSD", "ETHUSD", "^GSPC"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_value_at_risk(period="yearly", within_period=True)
        ```

        Which returns:

        |      |   BTCUSD |   ETHUSD |   ^GSPC |   Benchmark |
        |:-----|---------:|---------:|--------:|------------:|
        | 2021 |  -0.0626 |  -0.0838 | -0.013  |     -0.0126 |
        | 2022 |  -0.0574 |  -0.0742 | -0.0264 |     -0.0263 |
        | 2023 |  -0.0312 |  -0.0374 | -0.0138 |     -0.0138 |
        | 2024 |  -0.0407 |  -0.0533 | -0.0134 |     -0.0132 |
        | 2025 |  -0.0358 |  -0.0569 | -0.0161 |     -0.0165 |
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

        if rolling and distribution not in ("historic", "garch"):
            raise ValueError(
                "The rolling parameter is only available for distribution='historic' and "
                "distribution='garch'. Use get_var_backtest to obtain a rolling, out-of-sample "
                "VaR path under one of the other parametric distributions instead."
            )

        if distribution == "garch":
            value_at_risk = var_model.get_var_garch(
                (
                    self._historical_data[period]["Return"]
                    if rolling
                    else self._get_returns(period, within_period)
                ),
                alpha,
                rolling,
                refit_every,
                garch_method,
                garch_distribution,
            )
        elif rolling:
            returns = self._historical_data[period]["Return"]
            value_at_risk = var_model.get_rolling_var_historic(returns, alpha, rolling)
        else:
            returns = self._get_returns(period, within_period)

            if distribution == "historic":
                value_at_risk = var_model.get_var_historic(returns, alpha)
            elif distribution == "gaussian":
                value_at_risk = var_model.get_var_gaussian(returns, alpha)
            elif distribution == "cornish-fisher":
                value_at_risk = var_model.get_var_cornish_fisher(returns, alpha)
            elif distribution == "studentt":
                value_at_risk = var_model.get_var_studentt(returns, alpha)
            elif distribution == "evt":
                value_at_risk = var_model.get_var_evt(
                    returns, alpha, threshold_percentile
                )
            elif distribution == "kernel":
                value_at_risk = var_model.get_var_kernel(returns, alpha)
            else:
                raise ValueError(
                    "Distribution must be historic, gaussian, cornish-fisher, studentt, "
                    "evt, kernel or garch."
                )

        if rolling or within_period:
            value_at_risk = value_at_risk.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=value_at_risk,
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
    def get_conditional_value_at_risk(
        self,
        period: str | None = None,
        alpha: float | list[float] = 0.05,
        within_period: bool = False,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        distribution: str = "historic",
        threshold_percentile: float = 0.95,
        alpha_weights: list[float] | None = None,
        refit_every: int = 21,
        garch_method: str = "garch",
        garch_distribution: str = "normal",
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Conditional Value at Risk (CVaR) of an investment portfolio or asset's returns.

        Conditional Value at Risk (CVaR) is a risk management metric that quantifies the loss in the worst % of cases of
        an investment portfolio or asset may experience over a specified time horizon and confidence level.
        It provides insights into the downside risk associated with an investment and helps investors make
        informed decisions about risk tolerance.

        The CVaR is calculated as the expected loss given that the loss threshold (VaR) with a given confidence
        level (e.g., 5% for alpha=0.05) is exceeded.

        The Conditional Value at Risk (expected shortfall) is the measure banks report for market risk under
        the Basel framework, at 97.5% rather than 95% confidence, because unlike VaR it accounts for the size
        of the losses in the tail and never penalises diversification (it is subadditive). With daily data
        the estimate rests on few observations (one year at 95% leaves about 12 days in the tail), so a
        single extreme day can move it a lot; a longer window or a parametric distribution gives a steadier
        figure.

        The formula is as follows (with the default distribution="historic" and alpha=0.05):

        - CVaR = Mean(Returns | Returns <= VaR)
        - VaR = Quantile(Returns, alpha)

        The quantile interpolates linearly and the CVaR is reported as a return (negative for a loss). The other
        distributions use closed forms, with Std the sample standard deviation (ddof=1) and z_alpha the standard
        normal quantile at alpha:

        - Gaussian CVaR = Mean - Std * pdf(z_alpha) / alpha
        - Student-T CVaR = Location - Scale * (v + t^2) / (v - 1) * pdf_v(t) / alpha, with t = t_v^-1(1 - alpha)
        - Laplace CVaR = Mean - b * (1 - ln(2 * alpha)), with b = Std / √2
        - Logistic CVaR = Mean - s * ln((1 - alpha)^(1 - 1 / alpha) / alpha), with s = Std * √3 / π
        - Cornish-Fisher CVaR = Mean of (Mean + z_cf(p) * Std) over 1,000 evenly spaced p in (0, alpha)
        - EVT CVaR = -(VaR_loss + Scale_GPD - xi * u) / (1 - xi)
        - Kernel CVaR = mean(Return_i * Φ(u_i) - h * φ(u_i)) / alpha, with u_i = (VaR_kernel - Return_i) / h
        - GARCH CVaR = Mu + Volatility_(t+1) * E[z | z <= F^-1(alpha)]

        Where the Student-T v, Location and Scale are fitted by maximum likelihood (NaN for v <= 1), z_cf is the
        Cornish-Fisher quantile of `get_value_at_risk`, and u, xi, Scale_GPD and VaR_loss are the loss threshold,
        the fitted Generalized Pareto shape and scale and the EVT VaR as a positive loss (NaN for xi >= 1). The
        kernel CVaR is the mean return below the kernel VaR of `get_value_at_risk` under a Gaussian kernel
        density of the returns with Silverman's bandwidth h = Std * (4 / (3 * n))^(1/5) (Scaillet 2004), with Φ
        and φ the standard normal distribution and density. The GARCH CVaR is the conditional CVaR of the next
        period, with Mu, Volatility_(t+1) and the standardized innovations z with distribution function F of the
        GARCH VaR of `get_value_at_risk` (McNeil and Frey 2000).

        With a list of tail probabilities in `alpha`, the result is the Weighted Conditional Value at Risk,
        the weighted sum of the CVaR at each of them:

        - Weighted CVaR = Σ_k Weight_k * CVaR(alpha_k)

        With non-negative weights that sum to 1 this is a spectral risk measure (Acerbi 2002): it weighs the
        worst returns more than the CVaR at the largest alpha does, and any weighting of the tail can be built
        from enough tail probabilities. Each CVaR uses the chosen `distribution`.

        For more information, see the following sources:

        - Acerbi, C. (2002). "Spectral Measures of Risk: A Coherent Representation of Subjective Risk
        Aversion." Journal of Banking & Finance, 26(7), 1505-1518.
        - Cajas, D. (2021). "OWA Portfolio Optimization: A Disciplined Convex Programming Framework."
        SSRN Electronic Journal.
        - McNeil, A.J. and Frey, R. (2000). "Estimation of Tail-Related Risk Measures for Heteroscedastic
        Financial Time Series: An Extreme Value Approach." Journal of Empirical Finance, 7(3-4), 271-300.
        - Scaillet, O. (2004). "Nonparametric Estimation and Sensitivity Analysis of Expected Shortfall."
        Mathematical Finance, 14(1), 115-129.
        - Riskfolio-Lib: OwaWeights.owa_wcvar (the Weighted CVaR, https://github.com/dcajasn/Riskfolio-Lib).

        Also known as: CVaR, expected shortfall, ES, tail risk, weighted CVaR, WCVaR, spectral risk measure.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            alpha (float | list[float], optional): The tail probability for the CVaR calculation (e.g., 0.05 for
            the 95% confidence level), or a list of them for the Weighted CVaR. Defaults to 0.05.
            within_period (bool, optional): Whether to calculate CVaR within the specified period or for the entire
            period. Thus whether to look at the CVaR within a specific year (if period = 'yearly') or look at the entirety
            of all years. Defaults to False. Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rolling (int, optional): The rolling window size to use for the calculation. If set, CVaR is
            calculated over a rolling window of this many periods across the full return history instead
            of per `period` (e.g. a rolling 60-day CVaR). Only available for
            `distribution="historic"` and `distribution="garch"` (the number of returns of every fit, at
            least 100); see `get_acerbi_szekely_test` for a rolling, out-of-sample CVaR path under the
            other parametric distributions. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the CVaR values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            distribution (str): The distribution to use for the CVaR calculations (historic, gaussian, studentt, laplace,
            logistic, cornish-fisher, evt, kernel or garch). Defaults to "historic". "garch" fits a GARCH-family
            model per ticker (and per period with `within_period`), see `get_value_at_risk`.
            threshold_percentile (float, optional): Only used when `distribution` is "evt". The percentile
            of losses above which the Generalized Pareto Distribution is fitted; `alpha` can be at most
            1 - `threshold_percentile`, and a larger `alpha` gives an empty result (with a logged error).
            The fit needs at least 10 losses above the threshold (about 200 returns at 0.95); a period
            with fewer is NaN, with a logged warning. Defaults to 0.95.
            alpha_weights (list[float] | None, optional): The weight of the CVaR at each tail probability in `alpha`,
            non-negative and summing to 1. Defaults to None, which weighs every tail probability equally.
            refit_every (int, optional): Only used with `distribution="garch"` and `rolling`. The number of
            returns between two fits. Defaults to 21.
            garch_method (str, optional): Only used with `distribution="garch"`. The variance model, one of the
            methods of `get_garch`. Defaults to "garch".
            garch_distribution (str, optional): Only used with `distribution="garch"`. The innovation
            distribution (normal, t, skewt, skewt_fernandez_steel, ged or johnson_su). Defaults to "normal".
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: CVaR values with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates CVaR for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of CVaR values using the specified `lag`.
        - Riskfolio-Lib's Weighted CVaR applies the ordered weighted average weights of its CVaR, the exact
        tail average of Rockafellar and Uryasev (2002) that counts the return at the boundary with a fraction,
        where the historic CVaR here averages the returns up to the interpolated quantile; the weighted sum is
        the same, so the two differ by the difference of the CVaR estimators. Riskfolio-Lib reports the loss
        as a positive number, the Toolkit as a return.
        - The kernel CVaR equals the mean below the quantile of scipy's gaussian_kde with
        bw_method="silverman". PerformanceAnalytics' ES(method="kernel") is defined for the component
        Expected Shortfall of a portfolio only.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["XOM", "CVX", "COP"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_conditional_value_at_risk()
        ```

        Which returns:

        |           |   Value |
        |:----------|--------:|
        | XOM       | -0.0359 |
        | CVX       | -0.0383 |
        | COP       | -0.0532 |
        | Benchmark | -0.0275 |
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

        if rolling and distribution not in ("historic", "garch"):
            raise ValueError(
                "The rolling parameter is only available for distribution='historic' and "
                "distribution='garch'. Use get_acerbi_szekely_test to obtain a rolling, "
                "out-of-sample CVaR path under one of the other parametric distributions instead."
            )

        estimators = {
            "historic": cvar_model.get_cvar_historic,
            "gaussian": cvar_model.get_cvar_gaussian,
            "studentt": cvar_model.get_cvar_studentt,
            "laplace": cvar_model.get_cvar_laplace,
            "logistic": cvar_model.get_cvar_logistic,
            "cornish-fisher": cvar_model.get_cvar_cornish_fisher,
            "evt": functools.partial(
                cvar_model.get_cvar_evt, threshold_percentile=threshold_percentile
            ),
            "kernel": cvar_model.get_cvar_kernel,
            "garch": functools.partial(
                cvar_model.get_cvar_garch,
                window_size=rolling,
                refit_every=refit_every,
                method=garch_method,
                distribution=garch_distribution,
            ),
        }

        if distribution not in estimators:
            raise ValueError(
                "Distribution must be historic, gaussian, studentt, laplace, "
                "logistic, cornish-fisher, evt, kernel or garch."
            )

        alphas = alpha if isinstance(alpha, list | tuple) else [alpha]

        if rolling:
            returns = self._historical_data[period]["Return"]
            conditional_values_at_risk = [
                (
                    estimators["garch"](returns, tail)
                    if distribution == "garch"
                    else cvar_model.get_rolling_cvar_historic(returns, tail, rolling)
                )
                for tail in alphas
            ]
        else:
            returns = self._get_returns(period, within_period)
            conditional_values_at_risk = [
                estimators[distribution](returns, tail) for tail in alphas
            ]

        conditional_value_at_risk = (
            conditional_values_at_risk[0]
            if not isinstance(alpha, list | tuple) and alpha_weights is None
            else cvar_model.get_weighted_cvar(
                conditional_values_at_risk,
                (
                    alpha_weights
                    if alpha_weights is not None
                    else [1 / len(alphas)] * len(alphas)
                ),
            )
        )

        if rolling or within_period:
            conditional_value_at_risk = conditional_value_at_risk.loc[
                self._start_date : self._end_date
            ]

        return finalize_dataset(
            dataset=conditional_value_at_risk,
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
    def get_entropic_value_at_risk(
        self,
        period: str | None = None,
        alpha: float = 0.05,
        within_period: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        distribution: str = "gaussian",
        rolling: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Entropic Value at Risk (EVaR) of an investment portfolio or asset's returns.

        Entropic Value at Risk (EVaR) is a risk management metric that quantifies upper bound for the value
        at risk (VaR) and the conditional value at risk (CVaR) over a specified time horizon and confidence
        level. EVaR is obtained from the Chernoff inequality. It provides insights into the downside risk
        associated with an investment and helps investors make informed decisions about risk tolerance.

        By default the EVaR is calculated under the assumption that returns are normally distributed, for
        which the Chernoff bound has a closed form. With distribution="historic" it uses the empirical moment
        generating function of the returns instead, so it reflects their skewness and fat tails and always
        lies between the historical CVaR and the worst return.

        The formula is as follows (with the default distribution="gaussian" and alpha=0.05):

        - EVaR = Mean - Std * √(-2 * ln(alpha))

        Where alpha is the tail probability (0.05 for the 95% confidence level) and Std the sample standard
        deviation (ddof=1). It is an upper bound of the gaussian VaR and CVaR at the same confidence level.
        Since it only uses the mean and the standard deviation of the returns, it does not reflect their
        skewness or fat tails, and at high confidence levels it can be less severe than the historical CVaR
        of an asset with heavy tails. With distribution="historic":

        - EVaR = -inf over z > 0 of z * ln(mean(exp(-Return_i / z)) / alpha)

        Where z is the risk aversion of the entropic risk measure (see `get_entropic_risk_measure`), the
        bound is minimized over it numerically (on the logarithm of 1 / z, with scipy's brentq) and the mean
        runs over the n returns. When the worst return holds a share of the returns of at least alpha (for
        instance fewer than 20 returns at alpha=0.05), the EVaR is that worst return.

        For more information about the method, see the following sources:

        - Ahmadi-Javid, A. (2012). "Entropic Value-at-Risk: A New Coherent Risk Measure." Journal of
        Optimization Theory and Applications, 155(3), 1105-1123.
        - Riskfolio-Lib: RiskFunctions.EVaR_Hist (https://github.com/dcajasn/Riskfolio-Lib).
        - skfolio: measures.evar (https://skfolio.org/api.html#module-skfolio.measures).

        Also known as: EVaR, historical EVaR, entropic VaR, Chernoff bound.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            alpha (float, optional): The tail probability for the EVaR calculation (e.g., 0.05 for the 95%
            confidence level). Defaults to 0.05.
            within_period (bool, optional): Whether to calculate EVaR within the specified period or for the entire
            period. Thus whether to look at the EVaR within a specific year (if period = 'yearly') or look at the entirety
            of all years. Defaults to False. Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the CVaR values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            distribution (str, optional): "gaussian" for the closed form of normally distributed returns or
            "historic" for the empirical moment generating function of the returns. Defaults to "gaussian".
            rolling (int, optional): The rolling window size to use for the calculation. If set, EVaR is
            calculated over a rolling window of this many periods across the full return history instead
            of per `period`. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: EVaR values with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates EVaR for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of EVaR values using the specified `lag`.
        - The historic EVaR agrees with Riskfolio-Lib's EVaR_Hist and skfolio's evar, which report the loss as a
        positive number; the Toolkit keeps the sign of a return, as for the VaR and CVaR.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["CLUSD", "NGUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_entropic_value_at_risk(period="yearly", within_period=True)
        ```

        Which returns:

        |      |   CLUSD |   NGUSD |   Benchmark |
        |:-----|--------:|--------:|------------:|
        | 2021 | -0.0496 | -0.0837 |     -0.0185 |
        | 2022 | -0.0718 | -0.1505 |     -0.0373 |
        | 2023 | -0.0509 | -0.1079 |     -0.0187 |
        | 2024 | -0.0436 | -0.1183 |     -0.018  |
        | 2025 | -0.0474 | -0.1001 |     -0.0279 |
        """
        period = period if period else "daily"

        if distribution not in ["gaussian", "historic"]:
            raise ValueError("Distribution must be gaussian or historic.")

        returns = self._get_measure_returns(period, within_period, rolling, growth)

        entropic_value_at_risk = (
            evar_model.get_evar_historic(returns, alpha, window_size=rolling)
            if distribution == "historic"
            else evar_model.get_evar_gaussian(returns, alpha, window_size=rolling)
        )

        return self._finalize_measure(
            entropic_value_at_risk,
            bool(rolling or within_period),
            rounding,
            growth,
            lag,
            standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_entropic_risk_measure(
        self,
        period: str | None = None,
        risk_aversion: float = 1.0,
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
        Calculate the Entropic Risk Measure (ERM) of an investment portfolio or asset's returns at a given
        risk aversion.

        The Entropic Risk Measure is the certainty equivalent loss of an investor with exponential utility,
        shifted by the confidence level: by the Chernoff inequality the loss exceeds it with a probability of
        at most alpha, whatever the risk aversion z. The Entropic Value at Risk is its smallest value over z
        (see `get_entropic_value_at_risk` with distribution="historic"), so the ERM at any risk aversion is
        at least as severe as the historic EVaR, and equal to it at the z that minimizes it. A z close to
        the size of the returns (about 0.01 for daily returns) gives a value near the EVaR; a large z gives
        a value dominated by z * ln(1 / alpha).

        The formula is as follows (with the default risk_aversion=1 and alpha=0.05):

        - ERM = -z * ln(mean(exp(-Return_i / z)) / alpha)

        Where z is the risk aversion and the mean runs over the returns. The result is a return, negative
        for a loss, as the VaR, CVaR and EVaR of the Toolkit are.

        For more information, see the following sources:

        - Föllmer, H. and Schied, A. (2002). "Convex Measures of Risk and Trading Constraints." Finance and
        Stochastics, 6(4), 429-447.
        - Ahmadi-Javid, A. (2012). "Entropic Value-at-Risk: A New Coherent Risk Measure." Journal of
        Optimization Theory and Applications, 155(3), 1105-1123.
        - Riskfolio-Lib: RiskFunctions.Entropic_RM (https://github.com/dcajasn/Riskfolio-Lib).
        - skfolio: measures.entropic_risk_measure (https://skfolio.org/api.html#module-skfolio.measures).

        Also known as: ERM, entropic risk, exponential utility risk measure.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            risk_aversion (float, optional): The risk aversion z, larger than 0, in the units of the returns.
                Defaults to 1, the default of Riskfolio-Lib and skfolio.
            alpha (float, optional): The tail probability (e.g., 0.05 for the 95% confidence level).
                Defaults to 0.05.
            within_period (bool, optional): Whether to calculate the ERM within the specified period or for the
            entire period. Thus whether to look at the ERM within a specific year (if period = 'yearly') or look at
            the entirety of all years. Defaults to False. Note that this requires intraday data when period =
            'daily', since a single day only nests observations when intraday data was fetched.
            rolling (int, optional): The rolling window size to use for the calculation. If set, the ERM is
            calculated over a rolling window of this many periods across the full return history instead
            of per `period`. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ERM values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series | pd.DataFrame: ERM values, per ticker, or per period (or date) and ticker.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates the ERM
        for each asset in the Toolkit instance, each on its own returns.
        - If `growth` is set to True, the method calculates the growth of ERM values using the specified `lag`.
        - Equals Riskfolio-Lib's Entropic_RM and skfolio's entropic_risk_measure, which report the loss as a
        positive number; the Toolkit keeps the sign of a return. The exponential average is taken in log
        space (logsumexp), so a small risk aversion does not overflow.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["GLD", "SLV", "USO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_entropic_risk_measure(period="yearly", within_period=True, risk_aversion=0.01)
        ```

        Which returns:

        |      |     GLD |     SLV |     USO |   Benchmark |
        |:-----|--------:|--------:|--------:|------------:|
        | 2021 | -0.0349 | -0.0614 | -0.0871 |     -0.0326 |
        | 2022 | -0.0345 | -0.0434 | -0.0922 |     -0.0423 |
        | 2023 | -0.0329 | -0.0438 | -0.0502 |     -0.0323 |
        | 2024 | -0.0346 | -0.0496 | -0.0448 |     -0.0329 |
        | 2025 | -0.0444 | -0.0638 | -0.0612 |     -0.0429 |
        """
        period = period if period else "daily"
        returns = self._get_measure_returns(period, within_period, rolling, growth)

        entropic_risk_measure = evar_model.get_entropic_risk_measure(
            returns, risk_aversion=risk_aversion, alpha=alpha, window_size=rolling
        )

        return self._finalize_measure(
            entropic_risk_measure,
            bool(rolling or within_period),
            rounding,
            growth,
            lag,
            standardize,
        )

    def _get_portfolio_weights(
        self, returns: pd.DataFrame, weights: dict[str, float] | pd.Series | None
    ) -> pd.Series:
        if weights is None:
            constituent_tickers = [
                ticker
                for ticker in returns.columns
                if ticker not in ("Portfolio", "Benchmark")
            ]
            if not constituent_tickers:
                raise ValueError(
                    "No tickers available to build equal weights from. Please "
                    "provide the `weights` argument explicitly."
                )
            return pd.Series(1 / len(constituent_tickers), index=constituent_tickers)

        weights_series = (
            pd.Series(weights) if isinstance(weights, dict) else weights.copy()
        )
        missing_tickers = [
            ticker for ticker in weights_series.index if ticker not in returns.columns
        ]
        if missing_tickers:
            raise ValueError(
                f"weights contains tickers not present in the Toolkit: {missing_tickers}"
            )

        return weights_series

    @staticmethod
    def _get_shared_returns(returns: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        The returns of the portfolio constituents on the dates all of them have one. The returns of a
        date only some of them traded on (a holiday on one exchange, a weekend for a cryptocurrency) are
        compounded into the next shared date (see `helpers.compound_into_shared_dates`), so the portfolio
        return of a Monday combines each constituent's return since the Friday. A column other than a
        return (a price) only keeps the shared dates.
        """
        if column.endswith("Return"):
            return compound_into_shared_dates(returns)

        return returns.dropna()

    @handle_errors
    def get_marginal_value_at_risk(
        self,
        weights: dict[str, float] | None = None,
        period: str | None = None,
        column: str = "Return",
        alpha: float = 0.05,
        distribution: str = "historic",
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series:
        """
        Calculate the Marginal Value at Risk (Marginal VaR) of each asset in a
        portfolio.

        Ordinary VaR (see `get_value_at_risk`) treats each asset in isolation.
        Marginal VaR instead measures how much the *portfolio's* VaR would change for
        an infinitesimal increase in a given asset's portfolio weight — i.e. the
        sensitivity of portfolio risk to each holding, not the risk of the holding on
        its own.

        The formula is as follows (with the default distribution="historic"):

        - Portfolio Return = SUM(weight_i * Return_i)
        - Beta_i = Cov(Return_i, Portfolio Return) / Var(Portfolio Return)
        - Marginal VaR_i = Beta_i * Portfolio VaR

        Where the weights are normalized to sum to 1, the covariance and variance are sample estimates
        (ddof=1) on the dates all assets have a return (the returns of a date only some of them traded on are
        compounded into the next date all of them traded on), and Portfolio VaR is the VaR of the portfolio return
        under `distribution` (see `get_value_at_risk`). The beta-based sensitivity is exact for a zero-mean
        elliptical distribution and an approximation otherwise; distribution="gaussian" uses the exact
        derivative instead, with z_alpha the standard normal quantile at alpha:

        - Gaussian Marginal VaR_i = Mean_i + z_alpha * Cov(Return_i, Portfolio Return) / Std(Portfolio Return)

        An asset with Beta_i > 1 contributes disproportionately to portfolio risk,
        while Beta_i < 1 (and especially Beta_i < 0) indicates a diversifying holding.

        For more information about the method, see the following sources:

        - Garman, M.B. (1997). "Taking VaR to Pieces." Risk, 10(10), 70-71.
        - Litterman, R. (1996). "Hot Spots and Hedges." Goldman Sachs Risk Management
        Series.
        - Jorion, P. (2006). "Value at Risk: The New Benchmark for Managing Financial
        Risk." 3rd ed., McGraw-Hill, Chapter 7.

        Also known as: Marginal VaR, MVaR.

        Args:
            weights (dict[str, float] | None, optional): Portfolio weights
            keyed by ticker. Normalized internally to sum to 1. Defaults to None, which
            uses equal weights across every ticker in the Toolkit instance (excluding
            the "Portfolio" and "Benchmark" pseudo-tickers, if present).
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily", since a whole-series estimate needs far more observations than a single
                quarter or year holds.
            column (str, optional): The historical data column to use. Defaults to "Return".
            alpha (float, optional): The tail probability (e.g., 0.05 for the 95% confidence level).
            Defaults to 0.05.
            distribution (str, optional): The distribution to use for the underlying portfolio
            VaR calculation (historic, gaussian, cornish-fisher or studentt). Defaults to
            "historic".
            rounding (int | None, optional): The number of decimals to round the results to.
            Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: The Marginal VaR of each asset.

        Notes:
        - The method retrieves historical return data based on the specified `period` for
        every ticker in the Toolkit instance and computes each asset's Marginal VaR against
        the weighted portfolio built from `weights`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["VTI", "TLT", "GLD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_marginal_value_at_risk(weights={"VTI": 0.6, "TLT": 0.3, "GLD": 0.1})
        ```

        Which returns:

        |     |   Marginal VaR |
        |:----|---------------:|
        | VTI |        -0.0159 |
        | TLT |        -0.0078 |
        | GLD |        -0.0053 |
        """
        period = period if period else "daily"
        returns = self._get_price_column(period, column)
        weights_series = self._get_portfolio_weights(returns, weights)

        marginal_var = var_model.get_marginal_var(
            self._get_shared_returns(returns[weights_series.index], column),
            weights_series,
            alpha,
            distribution,
        )
        marginal_var.name = "Marginal VaR"

        return apply_rounding(
            marginal_var, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_component_value_at_risk(
        self,
        weights: dict[str, float] | None = None,
        period: str | None = None,
        column: str = "Return",
        alpha: float = 0.05,
        distribution: str = "historic",
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series:
        """
        Calculate the Component Value at Risk (Component VaR) of each asset in a
        portfolio.

        Component VaR allocates total portfolio VaR across its constituent assets,
        such that the allocations sum exactly back to the portfolio VaR — an "Euler"
        (or "fully consistent") risk decomposition.

        The formula is as follows (with the default distribution="historic"):

        - Component VaR_i = weight_i * Marginal VaR_i
        - SUM(Component VaR_i) = Portfolio VaR

        Where the weights are normalized to sum to 1 and Marginal VaR_i is that of
        `get_marginal_value_at_risk`: Beta_i * Portfolio VaR by default, or the exact derivative with
        distribution="gaussian".

        Where Marginal VaR measures each asset's risk *sensitivity*, Component VaR
        measures its actual *contribution* in the same units as portfolio VaR, making
        it directly usable to identify which holdings account for the largest share of
        portfolio risk.

        Component VaR is the basis of risk budgeting: a holding with a 20% weight that
        accounts for 45% of the portfolio VaR stands out immediately, and a component with
        the opposite sign of the portfolio VaR marks a holding that hedges the rest. The
        components always add up to the portfolio VaR, but the beta-based allocation is
        exact only for normally (elliptically) distributed returns and an approximation for
        the historical distribution used by default.

        For more information about the method, see the following sources:

        - Garman, M.B. (1997). "Taking VaR to Pieces." Risk, 10(10), 70-71.
        - Litterman, R. (1996). "Hot Spots and Hedges." Goldman Sachs Risk Management
        Series.
        - Jorion, P. (2006). "Value at Risk: The New Benchmark for Managing Financial
        Risk." 3rd ed., McGraw-Hill, Chapter 7.

        Also known as: Component VaR, risk contribution.

        Args:
            weights (dict[str, float] | None, optional): Portfolio weights
            keyed by ticker. Normalized internally to sum to 1. Defaults to None, which
            uses equal weights across every ticker in the Toolkit instance (excluding
            the "Portfolio" and "Benchmark" pseudo-tickers, if present).
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily", since a whole-series estimate needs far more observations than a single
                quarter or year holds.
            column (str, optional): The historical data column to use. Defaults to "Return".
            alpha (float, optional): The tail probability (e.g., 0.05 for the 95% confidence level).
            Defaults to 0.05.
            distribution (str, optional): The distribution to use for the underlying portfolio
            VaR calculation (historic, gaussian, cornish-fisher or studentt). Defaults to
            "historic".
            rounding (int | None, optional): The number of decimals to round the results to.
            Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: The Component VaR of each asset, plus a "Portfolio" entry equal to
            their sum (the total portfolio VaR), for validation.

        Notes:
        - The method retrieves historical return data based on the specified `period` for
        every ticker in the Toolkit instance and computes each asset's Component VaR against
        the weighted portfolio built from `weights`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["VTI", "AGG", "GLD", "VNQ"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_component_value_at_risk(
            weights={"VTI": 0.5, "AGG": 0.3, "GLD": 0.1, "VNQ": 0.1}
        )
        ```

        Which returns:

        |           |   Component VaR |
        |:----------|----------------:|
        | VTI       |         -0.0081 |
        | AGG       |         -0.0008 |
        | GLD       |         -0.0005 |
        | VNQ       |         -0.0015 |
        | Portfolio |         -0.0109 |
        """
        period = period if period else "daily"
        returns = self._get_price_column(period, column)
        weights_series = self._get_portfolio_weights(returns, weights)

        component_var = var_model.get_component_var(
            self._get_shared_returns(returns[weights_series.index], column),
            weights_series,
            alpha,
            distribution,
        )
        component_var.loc["Portfolio"] = component_var.sum()
        component_var.name = "Component VaR"

        return apply_rounding(
            component_var, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_var_backtest(
        self,
        period: str | None = None,
        distribution: str = "historic",
        alpha: float = 0.05,
        window_size: int | None = None,
        test: str | list[str] = "both",
        refit_every: int = 21,
        garch_method: str = "garch",
        garch_distribution: str = "normal",
        n_bootstrap: int = 1000,
        random_state: int = 42,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Backtest a Value at Risk model against realized returns.

        A Value at Risk estimate is only useful if it is actually well-calibrated
        against reality. This method builds a rolling, out-of-sample VaR path (each
        VaR estimate uses only the `window_size` returns preceding it, never the
        return it is compared against) and tests it with Kupiec's Proportion of
        Failures test (is the overall breach rate consistent with `alpha`?),
        Christoffersen's independence and conditional coverage tests (are breaches spread
        out over time, or do they cluster together?), Christoffersen and Pelletier's
        duration test (does the time between breaches have no memory?), McNeil and Frey's
        exceedance residual test of the Expected Shortfall (are the losses beyond the VaR
        as large as the CVaR says?) and the tick loss, which ranks VaR models.

        Regulators backtest VaR on the number of exceptions over the last 250 trading days
        (the Basel traffic-light approach at 99% confidence), and Kupiec's test formalises
        the same question. With few exceptions the test has little power, so a model that
        passes is not necessarily well calibrated. Clustered exceptions, which
        Christoffersen's and the duration test detect, are the usual failure of historical
        VaR when volatility rises; distribution="garch" follows the volatility from day to day.

        The formula is as follows (with the default distribution="historic", alpha=0.05 and daily returns, for which the window is 252):

        - VaR_t = Quantile(Return_(t-252), ..., Return_(t-1); alpha)
        - Breach_t = 1 if Return_t < VaR_t, else 0
        - Kupiec Statistic = -2 * ln[(1 - alpha)^(n - x) * alpha^x / ((1 - x / n)^(n - x) * (x / n)^x)]
        - Christoffersen Statistic = -2 * ln[L(pi) / L(pi_01, pi_11)]
        - Conditional Coverage Statistic = Kupiec Statistic + Christoffersen Statistic
        - Duration Statistic = 2 * (ln L(a_hat, b_hat) - ln L(a_tilde, 1))
        - McNeil-Frey Statistic = mean(e) / (std(e) / √x), with e_t = (CVaR_t - Return_t) / Volatility_t if Breach_t = 1
        - Tick Loss = (1 / n) * Σ (alpha - Breach_t) * (Return_t - VaR_t)

        Where n is the number of out-of-sample VaR estimates and x the number of breaches. L(pi) is the
        likelihood of the breach sequence with a single breach probability pi, and L(pi_01, pi_11) that of a
        first-order Markov chain in which the breach probability depends on whether the previous period was a
        breach. L(a, b) is the Weibull likelihood of the number of periods between breaches, with the periods
        before the first and after the last breach as censored durations, and b = 1 the exponential distribution
        of a model without memory (b_hat below 1 means breaches cluster). CVaR_t is the Conditional Value at Risk of
        the same window and distribution and Volatility_t the standard deviation (ddof=1) of the window (the
        volatility forecast with distribution="garch"); the McNeil-Frey p-value is the share of `n_bootstrap`
        resamples of the residuals less their mean whose statistic is at least as large (one-sided). The Kupiec,
        Christoffersen and duration statistics are compared with a χ² distribution with one degree of freedom
        and the conditional coverage statistic with two.

        With distribution="garch", VaR_t = Mu + Volatility_t * F^-1(alpha) and CVaR_t = Mu + Volatility_t *
        E[z | z <= F^-1(alpha)] come from the GARCH-family model `garch_method` with innovations
        `garch_distribution`, refitted every `refit_every` returns on the `window_size` returns before it (see
        `get_value_at_risk`).

        For more information about the methods, see the following papers:

        - Kupiec, P.H. (1995). "Techniques for Verifying the Accuracy of Risk
        Measurement Models." The Journal of Derivatives, 3(2), 73-84.
        - Christoffersen, P.F. (1998). "Evaluating Interval Forecasts." International
        Economic Review, 39(4), 841-862.
        - Christoffersen, P.F. and Pelletier, D. (2004). "Backtesting Value-at-Risk: A Duration-Based
        Approach." Journal of Financial Econometrics, 2(1), 84-108.
        - McNeil, A.J. and Frey, R. (2000). "Estimation of Tail-Related Risk Measures for Heteroscedastic
        Financial Time Series: An Extreme Value Approach." Journal of Empirical Finance, 7(3-4), 271-300.
        - González-Rivera, G., Lee, T.H. and Mishra, S. (2004). "Forecasting Volatility: A Reality Check Based
        on Option Pricing, Utility Function, Value-at-Risk, and Predictive Likelihood." International Journal
        of Forecasting, 20(4), 629-645.
        - rugarch: VaRTest, VaRDurTest, ESTest and VaRloss (https://cran.r-project.org/package=rugarch).

        Also known as: VaR validation, VaR backtest, POF test, conditional coverage test, duration test,
        exceedance residual test, quantile loss.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly or yearly).
            Defaults to "daily".
            distribution (str, optional): The distribution to use for the rolling VaR estimates, one of
            "historic", "gaussian", "studentt", "evt" or "garch". Defaults to "historic".
            "studentt" and "evt" fit their distribution to every window, which takes minutes for
            years of daily data, where "historic" and "gaussian" take well under a second. "garch" takes
            one to five seconds for its first fit per ticker and a fraction of a second for every refit.
            alpha (float, optional): The tail probability for the VaR estimates (e.g., 0.05 for the 95%
            confidence level). Defaults to 0.05. With "evt" it can be at most 0.05, since the Generalized
            Pareto Distribution is fitted above the 95th percentile of losses; a larger `alpha` gives an
            empty result (with a logged error).
            window_size (int | None, optional): The rolling window size (in number of return observations of
            `period`) used to estimate each VaR value. Defaults to None, which uses a year of observations
            with a minimum of 20 (252 daily, 52 weekly and 20 monthly, quarterly or yearly returns), and two
            years with a minimum of 100 for "garch" (504 daily, 104 weekly and 100 otherwise).
            test (str | list[str], optional): Which test(s) to run: "kupiec", "christoffersen",
            "conditional_coverage", "duration", "mcneil_frey" or "tick_loss", a list of them, "both" (Kupiec and
            Christoffersen) or "all". Defaults to "both".
            refit_every (int, optional): Only used with `distribution="garch"`. The number of returns between
            two fits. Defaults to 21.
            garch_method (str, optional): Only used with `distribution="garch"`. The variance model, one of the
            methods of `get_garch`. Defaults to "garch".
            garch_distribution (str, optional): Only used with `distribution="garch"`. The innovation
            distribution (normal, t, skewt, skewt_fernandez_steel, ged or johnson_su). Defaults to "normal".
            n_bootstrap (int, optional): The number of bootstrap resamples of the McNeil-Frey test. Defaults
            to 1000.
            random_state (int, optional): The seed of the McNeil-Frey bootstrap. Defaults to 42.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The statistic and p-value of every requested test (named after the test), the
            Weibull Shape of the duration test, the number of breaches of the McNeil-Frey test and the Tick
            Loss, per asset.

        Notes:
        - The rolling VaR path is calculated over the full return history, not the `within_period` slices
        used elsewhere in this module, since a meaningful rolling window generally needs more history than
        a single sub-period provides. Each asset is rolled over its own available returns, so a shorter
        history of one asset does not shorten the others.
        - The conditional coverage statistic adds Kupiec's statistic over all n estimates and
        Christoffersen's over the n - 1 transitions, as rugarch's VaRTest does.
        - rugarch's ESTest tests the unscaled gaps CVaR_t - Return_t with a normal p-value; the McNeil-Frey
        test here scales them by the volatility forecast and bootstraps the p-value, as in the paper. rugarch's
        VaRloss smooths the breach indicator and reports the loss per period in percentage points; the Tick
        Loss here is the exact average in decimals.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["EURUSD", "USDJPY", "GBPUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_var_backtest(test="kupiec")
        ```

        Which returns:

        |                  |   EURUSD |   USDJPY |   GBPUSD |   Benchmark |
        |:-----------------|---------:|---------:|---------:|------------:|
        | Kupiec Statistic |   8.3686 |   1.8323 |   2.2039 |      0.0836 |
        | P-Value          |   0.0038 |   0.1759 |   0.1377 |      0.7725 |
        """
        period = period if period else "daily"
        tests = (
            {
                "both": ["kupiec", "christoffersen"],
                "all": list(VAR_BACKTESTS),
            }.get(test, test)
            if isinstance(test, str)
            else test
        )
        tests = [tests] if isinstance(tests, str) else list(tests)

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if distribution not in ["historic", "gaussian", "studentt", "evt", "garch"]:
            raise ValueError(
                "Distribution must be historic, gaussian, studentt, evt or garch."
            )
        if not tests or any(name not in VAR_BACKTESTS for name in tests):
            raise ValueError(
                f"Test must be {', '.join(VAR_BACKTESTS)}, a list of them, both or all."
            )

        window_size = (
            window_size if window_size else _default_window_size(period, distribution)
        )
        returns = self._historical_data[period]["Return"].dropna(how="all")
        estimates = functools.partial(
            _get_out_of_sample_estimates,
            returns,
            distribution=distribution,
            alpha=alpha,
            window_size=window_size,
            refit_every=refit_every,
            garch_method=garch_method,
            garch_distribution=garch_distribution,
        )
        rolling_var = estimates("var")

        results = []
        for name in tests:
            if name == "kupiec":
                result = backtesting_model.get_kupiec_test(returns, rolling_var, alpha)
            elif name == "christoffersen":
                result = backtesting_model.get_christoffersen_test(returns, rolling_var)
            elif name == "conditional_coverage":
                result = backtesting_model.get_conditional_coverage_test(
                    returns, rolling_var, alpha
                )
            elif name == "duration":
                result = backtesting_model.get_duration_test(returns, rolling_var)
            elif name == "mcneil_frey":
                result = backtesting_model.get_mcneil_frey_test(
                    returns,
                    rolling_var,
                    estimates("cvar"),
                    estimates("volatility"),
                    n_bootstrap,
                    random_state,
                )
            else:
                result = backtesting_model.get_tick_loss(returns, rolling_var, alpha)

            # Every test has a P-Value, so with more than one test each is named after its test.
            results.append(
                result.rename(index={"P-Value": f"{VAR_BACKTESTS[name]} P-Value"})
                if len(tests) > 1
                else result
            )

        result = pd.concat(results, axis=0) if len(results) > 1 else results[0]

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_acerbi_szekely_test(
        self,
        period: str | None = None,
        distribution: str = "historic",
        alpha: float = 0.05,
        window_size: int | None = None,
        refit_every: int = 21,
        garch_method: str = "garch",
        garch_distribution: str = "normal",
        n_bootstrap: int = 1000,
        random_state: int = 42,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Backtest a Conditional Value at Risk (Expected Shortfall) model against
        realized returns, via the Acerbi-Szekely (2014) Z2 statistic.

        `get_var_backtest` above only backtests the VaR estimate itself -- it checks
        how often (and how independently) the VaR threshold is breached, but says
        nothing about the *severity* of the losses on those breach days, which is
        exactly the extra information a CVaR (Expected Shortfall) estimate is
        supposed to add over VaR. This method builds a rolling, out-of-sample CVaR
        (and VaR) path and compares the actual loss on each breach day to the CVaR
        that was supposed to describe the average loss on such days.

        The formula is as follows (with the default distribution="historic", alpha=0.05 and daily returns, for which the window is 252):

        - Z2 = Σ (Return_t * Breach_t / CVaR_t) / (n * alpha) - 1
        - P-Value = 1 - Φ(Z2 / Standard Error)

        Where VaR_t and CVaR_t are estimated from the window_size returns before t (see `get_var_backtest`), Breach_t is
        1 if Return_t < VaR_t and 0 otherwise, n is the number of estimates and the Standard Error is the sample
        standard deviation (ddof=1) of Z2 over `n_bootstrap` paired bootstrap resamples. Returns and CVaR are
        both negative, so Z2 is about 0 for a well-calibrated model and positive when the realized tail losses
        exceed the CVaR (the opposite sign of the paper, which counts losses as positive).

        For more information about the method, see the following paper:

        - Acerbi, C., & Szekely, B. (2014). "Back-Testing Expected Shortfall." RISK
        Magazine, 27(11), 76-81.

        Also known as: Acerbi-Szekely test, ES backtest, Z2 test.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly or yearly).
            Defaults to "daily".
            distribution (str, optional): The distribution to use for the rolling VaR/CVaR estimates,
            one of "historic", "gaussian", "studentt", "evt" or "garch". Defaults to "historic".
            "studentt" and "evt" fit their distribution to every window, which takes minutes for
            years of daily data, where "historic" and "gaussian" take well under a second. "garch" refits
            a GARCH-family model every `refit_every` returns (see `get_var_backtest`).
            alpha (float, optional): The tail probability for the VaR/CVaR estimates (e.g., 0.05 for the 95%
            confidence level). Defaults to 0.05. With "evt" it can be at most 0.05, since the Generalized
            Pareto Distribution is fitted above the 95th percentile of losses; a larger `alpha` gives an
            empty result (with a logged error).
            window_size (int | None, optional): The rolling window size (in number of return observations
            of `period`) used to estimate each VaR/CVaR value. Defaults to None, which uses a year of
            observations with a minimum of 20 (252 daily, 52 weekly and 20 monthly, quarterly or yearly
            returns), and two years with a minimum of 100 for "garch".
            refit_every (int, optional): Only used with `distribution="garch"`. The number of returns between
            two fits. Defaults to 21.
            garch_method (str, optional): Only used with `distribution="garch"`. The variance model, one of the
            methods of `get_garch`. Defaults to "garch".
            garch_distribution (str, optional): Only used with `distribution="garch"`. The innovation
            distribution (normal, t, skewt, skewt_fernandez_steel, ged or johnson_su). Defaults to "normal".
            n_bootstrap (int, optional): The number of bootstrap resamples used to estimate the Standard
            Error of Z2. Defaults to 1000.
            random_state (int, optional): The seed for the bootstrap random number generator. Defaults to
            42.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Z2 statistic, its bootstrap Standard Error, its one-sided p-value (a low
            value means the model underestimates the tail losses), and the number of breaches observed
            (returns strictly below the VaR), per asset.

        Notes:
        - The rolling VaR/CVaR path is calculated over the full return history, not the `within_period`
        slices used elsewhere in this module, since a meaningful rolling window generally needs more
        history than a single sub-period provides. Each asset is rolled over its own available returns.
        - The paper simulates the null distribution of Z2 from the model's predictive distributions; a
        rolling path does not carry those, so the p-value is approximated with a paired bootstrap of the
        observed (return, VaR, CVaR) triples and a normal approximation (see
        `backtesting_model.get_acerbi_szekely_test`).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["BA", "DAL", "CCL"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_acerbi_szekely_test().loc[["Acerbi-Szekely Statistic", "P-Value"]]
        ```

        Which returns:

        |                          |     BA |    DAL |    CCL |   Benchmark |
        |:-------------------------|-------:|-------:|-------:|------------:|
        | Acerbi-Szekely Statistic | 0.1548 | 0.2473 | 0.1402 |      0.2394 |
        | P-Value                  | 0.176  | 0.0762 | 0.1963 |      0.0785 |
        """
        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if distribution not in ["historic", "gaussian", "studentt", "evt", "garch"]:
            raise ValueError(
                "Distribution must be historic, gaussian, studentt, evt or garch."
            )

        window_size = (
            window_size if window_size else _default_window_size(period, distribution)
        )
        returns = self._historical_data[period]["Return"].dropna(how="all")
        rolling_var, rolling_cvar = (
            _get_out_of_sample_estimates(
                returns,
                measure,
                distribution,
                alpha,
                window_size,
                refit_every,
                garch_method,
                garch_distribution,
            )
            for measure in ("var", "cvar")
        )

        result = backtesting_model.get_acerbi_szekely_test(
            returns, rolling_var, rolling_cvar, alpha, n_bootstrap, random_state
        )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )
