"""Volatility Risk Module"""

__docformat__ = "google"

from typing import Literal

import numpy as np
import pandas as pd

from financetoolkit.helpers import handle_portfolio
from financetoolkit.risk import risk_model
from financetoolkit.risk.helpers import check_rolling
from financetoolkit.risk.volatility import garch_model, realized_volatility_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import (
    InvalidArgumentError,
    check_option,
    check_period,
    handle_errors,
)
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import (
    apply_rounding,
    finalize_dataset,
)

# ruff: noqa: E501


def _forecast_index(
    index: pd.PeriodIndex, period: str, time_steps: int
) -> pd.PeriodIndex:
    """The periods after the last return, continuing its period (business days for daily)."""
    period_symbol = {
        "daily": "B",
        "weekly": "W",
        "monthly": "ME",
        "quarterly": "QE",
    }.get(period, "YE")
    # Business days are kept as daily periods, as business-day periods are deprecated.
    period_index = pd.PeriodIndex(
        pd.date_range(
            start=index[-1].to_timestamp(),
            periods=time_steps + 1,
            freq=period_symbol,
        ),
        freq="D" if period_symbol == "B" else None,
    )

    return period_index[1:]


@filter_shown_columns
@track_sources
class VolatilityRisk:
    """
    Volatility: GARCH-family fits (GARCH, GJR-GARCH, EGARCH, IGARCH, APARCH, the family GARCH sub-models, component GARCH,
    FIGARCH and realized GARCH, with Student-t, skewed-t, GED and Johnson SU innovations and constant, ARMA, in-mean and ARFIMA
    mean models), their forecasts, diagnostics, news impact curve, simulation and bootstrap forecast density, realized and EWMA volatility, the HAR-RV forecast, excess volatility, the downside and upside deviation and the volatility skewness.
    """

    def _get_realized_measure(
        self, method: str, period: str, within_period: bool = False
    ) -> pd.DataFrame | None:
        """
        The realized measure of the realized GARCH (method="realized"): the daily Parkinson
        (1980) variance (ln(High_t / Low_t))^2 / (4 ln 2) of every ticker, None for the other
        models. A day without a positive range (a missing or flat bar) has no measure.
        """
        if method != "realized":
            return None
        if period != "daily" or within_period:
            raise InvalidArgumentError(
                "method='realized' takes the daily Parkinson variance of the High and Low "
                "prices as its realized measure, so it needs period='daily' and "
                "within_period=False."
            )

        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(self._get_prices("High") / self._get_prices("Low")) ** 2 / (
                4 * np.log(2)
            )

    @handle_portfolio
    @handle_errors
    def get_garch(
        self,
        period: str | None = None,
        time_steps: int | None = None,
        optimization_t: int | None = None,
        within_period: bool = False,
        method: Literal[
            "garch",
            "gjr",
            "egarch",
            "igarch",
            "aparch",
            "tgarch",
            "ngarch",
            "avgarch",
            "nagarch",
            "component",
            "figarch",
            "realized",
        ] = "garch",
        distribution: str = "normal",
        mean: str = "zero",
        output: str = "variance",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the conditional variance (sigma_t^2) of the returns with a GARCH model: how volatile each
        period is expected to be given the shocks before it, as volatility clusters.

        The model is a GARCH(1, 1) fitted to the returns, or one of its variants: the asymmetric GJR-GARCH(1, 1, 1)
        and EGARCH(1, 1), the integrated IGARCH(1, 1), the asymmetric power APARCH(1, 1) of Ding, Granger and Engle,
        the TGARCH, NGARCH, AVGARCH and NAGARCH sub-models of Hentschel's family GARCH, the component GARCH of Engle
        and Lee, the long-memory FIGARCH(1, d, 1) of Baillie, Bollerslev and Mikkelsen and the realized GARCH of
        Hansen, Huang and Shek. The innovations can follow a normal, Student-t, skewed Student-t, GED or Johnson SU
        distribution, and the returns can have a zero, constant, ARMA(1, 1), GARCH-in-mean or ARFIMA(1, d, 1) mean.
        Take the square root for the conditional volatility, see `get_garch_forecast` for the variance forecast beyond
        the last observation and `get_garch_parameters` for the fitted parameters and the fit diagnostics.

        GARCH (Generalized autoregressive conditional heteroskedasticity) is stochastic model for time series, which is
        for instance used to model volatility clusters, stock return and inflation. It is a generalisation of the ARCH
        models.

        GJR-GARCH extends GARCH with a leverage term that lets negative shocks (bad news)
        raise volatility by more than positive shocks of the same size, a well documented asymmetry in
        equity returns that symmetric GARCH cannot capture. EGARCH models the log of the
        conditional variance, which avoids having to constrain the parameters to keep the variance
        positive and, like GJR-GARCH, lets negative and positive shocks of the same size have a different
        impact on volatility.

        IGARCH restricts Alpha + Beta to 1, so a shock never fades and the variance has no
        long-run level, as in the RiskMetrics EWMA (which also sets Omega to 0). APARCH models a
        power Delta of the volatility, estimated from the data, with a leverage rotation Gamma; it nests GARCH
        (Delta = 2, Gamma = 0), TGARCH (Delta = 1) and NGARCH (Gamma = 0). Hentschel's family GARCH measures the
        shock in units of the volatility, z_t = e_t / sigma_t, and allows it to be shifted as well as rotated:
        AVGARCH is the absolute value model with both, and NAGARCH the
        variance model with a shift only. The component GARCH splits the variance into a
        long-run component q_t that fades slowly (with Rho) and a short-run component that fades quickly (with
        Alpha + Beta).

        FIGARCH lets a shock fade hyperbolically rather than by a fixed factor per period: the
        fractional difference D between 0 (GARCH) and 1 (IGARCH) sets how long the volatility remembers it, the
        long memory of the squared returns of most assets. The realized GARCH adds a second
        source of information on the volatility, a realized measure x_t of every day, and models how it relates
        to the conditional variance; the Toolkit takes the daily Parkinson variance of the High and Low prices as
        that measure, in place of the intraday realized variance of the paper. The ARFIMA mean
        lets the returns themselves have a long memory, with a fractional difference between -0.5 and 0.5.

        For more information about the methods, see the following papers:

        - Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity." Journal of
        Econometrics, 31(3), 307-327.
        - Engle, R.F. and Bollerslev, T. (1986). "Modelling the Persistence of Conditional Variances."
        Econometric Reviews, 5(1), 1-50.
        - Ding, Z., Granger, C.W.J. and Engle, R.F. (1993). "A Long Memory Property of Stock Market Returns and
        a New Model." Journal of Empirical Finance, 1(1), 83-106.
        - Hentschel, L. (1995). "All in the Family: Nesting Symmetric and Asymmetric GARCH Models." Journal of
        Financial Economics, 39(1), 71-104.
        - Zakoian, J.M. (1994). "Threshold Heteroskedastic Models." Journal of Economic Dynamics and Control,
        18(5), 931-955.
        - Higgins, M.L. and Bera, A.K. (1992). "A Class of Nonlinear ARCH Models." International Economic
        Review, 33(1), 137-158.
        - Engle, R.F. and Ng, V.K. (1993). "Measuring and Testing the Impact of News on Volatility." The
        Journal of Finance, 48(5), 1749-1778.
        - Engle, R.F. and Lee, G.G.J. (1999). "A Long-Run and Short-Run Component Model of Stock Return
        Volatility." In Cointegration, Causality and Forecasting, Oxford University Press, 475-497.
        - Engle, R.F., Lilien, D.M. and Robins, R.P. (1987). "Estimating Time Varying Risk Premia in the Term
        Structure: The ARCH-M Model." Econometrica, 55(2), 391-407.
        - Baillie, R.T., Bollerslev, T. and Mikkelsen, H.O. (1996). "Fractionally Integrated Generalized
        Autoregressive Conditional Heteroskedasticity." Journal of Econometrics, 74(1), 3-30.
        - Conrad, C. and Haag, B.R. (2006). "Inequality Constraints in the Fractionally Integrated GARCH Model."
        Journal of Financial Econometrics, 4(3), 413-449.
        - Hansen, P.R., Huang, Z. and Shek, H.H. (2012). "Realized GARCH: A Joint Model for Returns and Realized
        Measures of Volatility." Journal of Applied Econometrics, 27(6), 877-906.
        - Parkinson, M. (1980). "The Extreme Value Method for Estimating the Variance of the Rate of Return."
        Journal of Business, 53(1), 61-65.
        - Granger, C.W.J. and Joyeux, R. (1980). "An Introduction to Long-Memory Time Series Models and
        Fractional Differencing." Journal of Time Series Analysis, 1(1), 15-29.
        - Hosking, J.R.M. (1981). "Fractional Differencing." Biometrika, 68(1), 165-176.

        - Glosten, L.R., Jagannathan, R., and Runkle, D.E. (1993). "On the Relation
        between the Expected Value and the Volatility of the Nominal Excess Return on
        Stocks." The Journal of Finance, 48(5), 1779-1801.
        - Nelson, D.B. (1991). "Conditional Heteroskedasticity in Asset Returns: A New
        Approach." Econometrica, 59(2), 347-370.

        The formula is as follows:

        - GARCH: sigma_t^2 = Omega + Alpha * r_(t-1)^2 + Beta * sigma_(t-1)^2, with
        sigma_1^2 = Omega + (Alpha + Beta) * Backcast
        - GJR-GARCH: sigma_t^2 = Omega + (Alpha + Gamma * I_(t-1)) * r_(t-1)^2 + Beta * sigma_(t-1)^2, with
        sigma_1^2 = Omega + (Alpha + Gamma / 2 + Beta) * Backcast
        - EGARCH: ln sigma_t^2 = Omega + Alpha * (|z_(t-1)| - √(2 / π)) + Gamma * z_(t-1) + Beta * ln sigma_(t-1)^2,
        with z_t = r_t / sigma_t and ln sigma_1^2 = Omega + Beta * ln(Backcast)
        - IGARCH: sigma_t^2 = Omega + Alpha * e_(t-1)^2 + (1 - Alpha) * sigma_(t-1)^2, with
        sigma_1^2 = Omega + Backcast
        - APARCH: sigma_t^Delta = Omega + Alpha * (|e_(t-1)| - Gamma * e_(t-1))^Delta + Beta * sigma_(t-1)^Delta,
        with sigma_1^Delta = Omega + (Alpha + Beta) * Backcast^(Delta / 2)
        - TGARCH: sigma_t = Omega + Alpha * (|e_(t-1)| - Gamma * e_(t-1)) + Beta * sigma_(t-1), APARCH with Delta = 1
        - NGARCH: sigma_t^Delta = Omega + Alpha * |e_(t-1)|^Delta + Beta * sigma_(t-1)^Delta, APARCH with Gamma = 0
        - AVGARCH: sigma_t = Omega + Alpha * sigma_(t-1) * (|z_(t-1) - Shift| - Gamma * (z_(t-1) - Shift))
        + Beta * sigma_(t-1)
        - NAGARCH: sigma_t^2 = Omega + Alpha * sigma_(t-1)^2 * (z_(t-1) - Shift)^2 + Beta * sigma_(t-1)^2
        - Component GARCH: sigma_t^2 = q_t + Alpha * (e_(t-1)^2 - q_(t-1)) + Beta * (sigma_(t-1)^2 - q_(t-1)), with
        the long-run component q_t = Omega + Rho * q_(t-1) + Phi * (e_(t-1)^2 - sigma_(t-1)^2) and q_1 = sigma_1^2 =
        Omega + Rho * Backcast
        - FIGARCH: sigma_t^2 = Omega / (1 - Beta) + Σ_(k=1..1000) lambda_k * e_(t-k)^2, the ARCH(∞) form of
        (1 - Beta L) sigma_t^2 = Omega + [1 - Beta L - (1 - Phi L)(1 - L)^D] e_t^2 truncated at 1,000 lags, with
        lambda_1 = D + Phi - Beta, the lambda_k from the expansion of (1 - Phi L)(1 - L)^D / (1 - Beta L) and the
        Backcast for every e_(t-k)^2 before the first return
        - Realized GARCH: ln sigma_t^2 = Omega + Beta * ln sigma_(t-1)^2 + Gamma * ln x_(t-1) and the measurement
        equation ln x_t = Xi + Phi * ln sigma_t^2 + Tau 1 * z_t + Tau 2 * (z_t^2 - 1) + u_t, u_t ~ N(0, Sigma U^2), with
        x_t = (ln(High_t / Low_t))^2 / (4 ln 2) the daily Parkinson variance
        - Zero mean: e_t = r_t; constant mean: e_t = r_t - Mu; ARMA mean: e_t = r_t - Mu - AR(1) * r_(t-1) -
        MA(1) * e_(t-1) from the second return, with e_1 = 0 before it; GARCH-in-mean: e_t = r_t - Mu - Lambda *
        sigma_t; ARFIMA mean: e_t = w_t - AR(1) * w_(t-1) - MA(1) * e_(t-1), with w_t = (1 - L)^D (r_t - Mu) over the
        returns from the first (Fractional D the D of the mean)

        Where r_t are the returns (not demeaned), e_t the residuals of the mean model (the returns themselves with
        the default zero mean, so r_t = e_t in the first three lines), z_t = e_t / sigma_t, I_(t-1) is 1 when
        r_(t-1) < 0 and 0 otherwise, √(2 / π) is the expected |z| of a standard normal shock and the Backcast is an
        exponentially weighted average (decay 0.94) of the first 75 squared residuals of the mean model at its
        starting values (the returns less their average for a constant or in-mean mean, the residuals of the
        least-squares AR(1) for the ARMA and ARFIMA means). The parameters maximize the log-likelihood of the innovation
        distribution, which for the default normal distribution means minimizing Σ (ln sigma_t^2 + e_t^2 / sigma_t^2),
        under the constraints listed in `get_garch_parameters`; the log variance of EGARCH is kept within [-20, 20].

        Also known as: GARCH, volatility clustering, conditional heteroscedasticity, GJR-GARCH, gjr garch,
        threshold GARCH, TGARCH, EGARCH, exponential GARCH, log-GARCH, integrated GARCH, IGARCH, asymmetric power
        ARCH, APARCH, power GARCH, family GARCH, fGARCH, NGARCH, nonlinear GARCH, AVGARCH, absolute value GARCH,
        NAGARCH, nonlinear asymmetric GARCH, component GARCH, CGARCH, GARCH-in-mean, ARCH-M, AR-GARCH, ARMA-GARCH,
        Student-t GARCH, skewed-t GARCH, standardized residuals, FIGARCH, fractionally integrated GARCH, long memory
        volatility, realized GARCH, RealGARCH, ARFIMA-GARCH, fractionally integrated ARMA.

        Referenced against:
        - arch, GARCH: https://bashtage.github.io/arch/univariate/generated/arch.univariate.GARCH.html
        - arch, GARCH (GJR-GARCH): https://bashtage.github.io/arch/univariate/generated/arch.univariate.GARCH.html
        - arch, EGARCH: https://bashtage.github.io/arch/univariate/generated/arch.univariate.EGARCH.html
        - arch, APARCH (APARCH, TGARCH, NGARCH): https://bashtage.github.io/arch/univariate/generated/arch.univariate.APARCH.html
        - arch, FIGARCH: https://bashtage.github.io/arch/univariate/generated/arch.univariate.FIGARCH.html
        - arch, ARCHInMean (GARCH-in-mean): https://bashtage.github.io/arch/univariate/generated/arch.univariate.ARCHInMean.html

        Shares similar logic:
        - rugarch (R), ugarchspec (sGARCH, iGARCH, apARCH, fGARCH, csGARCH, realGARCH, ARFIMA mean): https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-methods.R#L21
        - rugarch (R), ugarchfit: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-methods.R#L1059

        Args:
            period (str, optional): The data frequency for returns. Options are: "daily", "weekly", "monthly",
                "quarterly", "yearly". Defaults to "daily".
            time_steps (int, optional): Time steps to calculate GARCH for. Defaults to None, which uses the length of the
                returns.
            optimization_t (int, optional): Time steps to optimize GARCH for. It is only used if no weights are given.
                Defaults to None, which uses the length of the returns.
            within_period (bool, optional): Whether to calculate GARCH within the specified period or for the entire
            period. Thus whether to look at the GARCH within a specific year (if period = 'yearly') or look at the
            entirety of all years. Defaults to False.
            method (str, optional): The model. Options are: "garch" (GARCH(1, 1)), "gjr" (GJR-GARCH(1, 1, 1)),
                "egarch" (EGARCH(1, 1)), "igarch" (IGARCH(1, 1)), "aparch" (APARCH(1, 1)), "tgarch" (threshold GARCH),
                "ngarch" (nonlinear GARCH), "avgarch" (absolute value GARCH), "nagarch" (nonlinear asymmetric GARCH),
                "component" (component GARCH), "figarch" (FIGARCH(1, d, 1)), "realized" (realized GARCH(1, 1) on the
                daily Parkinson variance). Defaults to "garch".
            distribution (str, optional): The distribution of the standardized innovations z_t. Options are: "normal"
                (normal), "t" (Student-t, with a Shape v > 2), "skewt" (Hansen's skewed Student-t, with a Shape and a
                Skew λ in (-1, 1)), "skewt_fernandez_steel" (Fernández and Steel's skewed Student-t, the same
                distribution with a Skew ξ = √((1 + λ) / (1 - λ)), as in R's rugarch), "ged" (generalized error
                distribution, with a Shape v, 2 for the normal), "johnson_su" (Johnson's SU, with a Shape and a Skew).
                Defaults to "normal".
            mean (str, optional): The mean model. Options are: "zero" (zero mean, the returns themselves are the
                shocks), "constant" (constant mean Mu), "arma" (ARMA(1, 1)), "in_mean" (GARCH-in-mean, Mu + Lambda *
                sigma_t), "arfima" (ARFIMA(1, d, 1)). Defaults to "zero".
            output (str, optional): What to return. Options are: "variance" (the conditional variance sigma_t^2),
                "standardized_residuals" (z_t = e_t / sigma_t, which should look like independent draws from the
                chosen distribution when the model fits), "long_run_variance" (the long-run component q_t of the
                component GARCH). Defaults to "variance".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the GARCH values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame | pd.Series: GARCH values

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates GARCH for each
        asset in the Toolkit instance.
        - Every asset is fitted on its own available returns, so an asset with a shorter history (or a gap)
        does not shorten the sample of the others; dates without a return for an asset are NaN.
        - The EGARCH fit is a deterministic multi-start search that takes far longer than the GARCH and
        GJR-GARCH fits, about 45 seconds per ticker of daily returns. The AVGARCH and NAGARCH recursions and the
        GARCH-in-mean filter also run step by step and take several times longer than the other models; the
        realized GARCH (eight parameters) and the ARFIMA mean (a fractional difference over the whole sample in
        every step of the search) take 10 to 15 seconds per ticker of ten years of daily returns, FIGARCH 2 to 5;
        the others take one to a few seconds.
        - The realized GARCH takes daily returns over the entire period (not within periods): its realized measure is
          the daily Parkinson variance, which leaves out the overnight move and so sits below the close-to-close
          variance; Xi and Phi absorb that level. A day without a positive High-Low range takes the expected measure
          given its return.
        - FIGARCH keeps every ARCH(∞) weight non-negative, the condition of Conrad and Haag (2006) for a
        non-negative variance, a wider region than the arch package's bounds (0 <= Phi <= (1 - D) / 2 and
        Beta <= D + Phi); where the fit stays inside the arch bounds both agree.
        - Every model other than GARCH, GJR-GARCH and EGARCH with normal innovations and a zero mean is fitted on
        the returns divided by their standard deviation, which puts every parameter on a scale of about 1, and the
        parameters are converted back (see `get_garch_parameters`).
        - With growth, the method calculates the growth of GARCH values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["BTCUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_garch().tail(5)
        ```

        Which returns:

        | Date       |   BTCUSD |   Benchmark |
        |:-----------|---------:|------------:|
        | 2025-12-27 |   0.0004 |    nan      |
        | 2025-12-28 |   0.0004 |    nan      |
        | 2025-12-29 |   0.0004 |      0.0001 |
        | 2025-12-30 |   0.0004 |      0      |
        | 2025-12-31 |   0.0004 |      0      |
        """
        garch_model.validate_garch_family(method, distribution, mean)

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

        # Only rows without any return are dropped: every column is fitted on its own non-missing returns, so a ticker that listed later (or has a gap) does not cut the others down to its history.  # noqa: E501
        returns = self._get_returns(period, within_period).dropna(how="all")

        garch_sigma_2 = garch_model.get_garch_family(
            returns=returns,
            method=method,
            distribution=distribution,
            mean=mean,
            output=output,
            time_steps=time_steps,
            optimization_t=optimization_t,
            realized_measure=self._get_realized_measure(method, period, within_period),
        )

        garch_sigma_2 = garch_sigma_2.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=garch_sigma_2,
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
    def get_garch_forecast(
        self,
        period: str | None = None,
        forecast_steps: int = 10,
        within_period: bool = False,
        method: Literal[
            "garch",
            "gjr",
            "egarch",
            "igarch",
            "aparch",
            "tgarch",
            "ngarch",
            "avgarch",
            "nagarch",
            "component",
            "figarch",
            "realized",
        ] = "garch",
        distribution: str = "normal",
        mean: str = "zero",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the GARCH variance forecast (sigma_2): the conditional variance a GARCH model expects for each
        period after the last return, as a volatility shock fades towards its long-run level.

        The model is a GARCH(1, 1), or one of its variants (see `get_garch`): the
        asymmetric GJR-GARCH(1, 1, 1) and EGARCH(1, 1), IGARCH(1, 1), APARCH(1, 1), the TGARCH, NGARCH, AVGARCH and
        NAGARCH sub-models of family GARCH, the component GARCH, FIGARCH(1, d, 1) and the realized GARCH, with any of
        its innovation distributions and mean models.

        GARCH (Generalized autoregressive conditional heteroskedasticity) is stochastic model for time series, which is
        for instance used to model volatility clusters, stock return and inflation. It is a generalisation of the ARCH
        models.

        The formula is as follows:

        - GARCH: sigma_(n+h)^2 = sigma_L^2 + (sigma_(n+1)^2 - sigma_L^2) * (Alpha + Beta)^(h - 1), with
        sigma_L^2 = Omega / (1 - Alpha - Beta)
        - GJR-GARCH: sigma_(n+h)^2 = sigma_L^2 + (sigma_(n+1)^2 - sigma_L^2) * (Alpha + Gamma / 2 + Beta)^(h - 1),
        with sigma_L^2 = Omega / (1 - Alpha - Gamma / 2 - Beta)
        - EGARCH: ln sigma_(n+h)^2 = L + Beta^(h - 1) * (ln sigma_(n+1)^2 - L), with L = Omega / (1 - Beta)
        and sigma_(n+h)^2 = exp(ln sigma_(n+h)^2)
        - IGARCH: sigma_(n+h)^2 = sigma_(n+1)^2 + (h - 1) * Omega
        - APARCH, TGARCH, NGARCH, AVGARCH and NAGARCH: s_(n+h) = Omega + P * s_(n+h-1), with
        s = sigma^d the power of the model (Delta, 1 for TGARCH and AVGARCH, 2 for NAGARCH), P the persistence (see
        `get_garch_parameters`) and sigma_(n+h)^2 = s_(n+h)^(2 / d)
        - Component GARCH: sigma_(n+h)^2 = q_(n+h) + (Alpha + Beta)^(h - 1) * (sigma_(n+1)^2 - q_(n+1)), with
        q_(n+h) = Omega + Rho * q_(n+h-1)
        - FIGARCH: sigma_(n+h)^2 = Omega / (1 - Beta) + Σ_(k=1..1000) lambda_k * E[e_(n+h-k)^2], with the squared
        residuals up to the last return and the forecast variance sigma_(n+j)^2 in place of e_(n+j)^2 after it
        - Realized GARCH: ln sigma_(n+h)^2 = Omega + Gamma * Xi + (Beta + Gamma * Phi) * ln sigma_(n+h-1)^2 and
        sigma_(n+h)^2 = exp(ln sigma_(n+h)^2), the expected realized measure in place of the measure

        Where n is the last observation, h = 1, ..., `forecast_steps` (10 by default) the forecast step,
        sigma_(n+1)^2 the one-step-ahead variance of the fitted recursion (see `get_garch`) and sigma_L^2 the
        long-run variance. The leverage term of GJR-GARCH enters at its expectation, Gamma / 2, as a negative
        shock is as likely as a positive one; the shock terms of EGARCH drop out at their expectation of zero,
        so its forecast is the exponential of the expected log variance. The other models also take the shock
        terms at their expectation over the innovation distribution, which makes the recursion of the power of the
        volatility linear; transforming its forecast back, like the exponential of EGARCH, understates the expected
        variance somewhat (Jensen's inequality) when the power is not 2. With a skewed distribution the leverage
        term of GJR-GARCH enters at Gamma * E[z^2 1(z < 0)] instead of Gamma / 2, and EGARCH's shock term at
        Alpha * (E|z| - √(2 / π)). The mean model only changes the residuals the recursion runs on.

        For more information about the methods, see the following sources:

        - Finance Compact Plus Band 1, by Yvonne Seler Zimmerman and Heinz Zimmerman; ISBN: 978-3-907291-31-1
        - Glosten, L.R., Jagannathan, R., and Runkle, D.E. (1993). "On the Relation
        between the Expected Value and the Volatility of the Nominal Excess Return on
        Stocks." The Journal of Finance, 48(5), 1779-1801.
        - Nelson, D.B. (1991). "Conditional Heteroskedasticity in Asset Returns: A New
        Approach." Econometrica, 59(2), 347-370.

        - Ding, Z., Granger, C.W.J. and Engle, R.F. (1993). "A Long Memory Property of Stock Market Returns and
        a New Model." Journal of Empirical Finance, 1(1), 83-106.
        - Engle, R.F. and Lee, G.G.J. (1999). "A Long-Run and Short-Run Component Model of Stock Return
        Volatility." In Cointegration, Causality and Forecasting, Oxford University Press, 475-497.

        Also known as: volatility forecast, predicted volatility, GJR-GARCH forecast, EGARCH forecast, APARCH
        forecast, IGARCH forecast, component GARCH forecast.

        Referenced against:
        - arch, GARCH (GJR-GARCH forecast): https://bashtage.github.io/arch/univariate/generated/arch.univariate.GARCH.html
        - arch, GARCH: https://bashtage.github.io/arch/univariate/generated/arch.univariate.GARCH.html
        - arch, GARCH (GJR-GARCH): https://bashtage.github.io/arch/univariate/generated/arch.univariate.GARCH.html
        - arch, EGARCH: https://bashtage.github.io/arch/univariate/generated/arch.univariate.EGARCH.html
        - arch, APARCH (APARCH, TGARCH, NGARCH): https://bashtage.github.io/arch/univariate/generated/arch.univariate.APARCH.html
        - arch, FIGARCH: https://bashtage.github.io/arch/univariate/generated/arch.univariate.FIGARCH.html
        - arch, ARCHInMean (GARCH-in-mean): https://bashtage.github.io/arch/univariate/generated/arch.univariate.ARCHInMean.html

        Shares similar logic:
        - rugarch (R), ugarchforecast: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-methods.R#L1071

        Args:
            period (str, optional): The data frequency for returns. Options are: "daily", "weekly", "monthly",
                "quarterly", "yearly". Defaults to "daily".
            forecast_steps (int, optional): Time steps to calculate GARCH and to forecast sigma_2 values for. Defaults to 10.
            within_period (bool, optional): Whether to calculate GARCH within each specified period or all
            at once. Thus whether to look at the GARCH within each specific year (if period = 'yearly') or
            look at the entirety of all years. Defaults to False.
            method (str, optional): The model. Options are: "garch" (GARCH(1, 1)), "gjr" (GJR-GARCH(1, 1, 1)),
                "egarch" (EGARCH(1, 1)), "igarch" (IGARCH(1, 1)), "aparch" (APARCH(1, 1)), "tgarch" (threshold GARCH),
                "ngarch" (nonlinear GARCH), "avgarch" (absolute value GARCH), "nagarch" (nonlinear asymmetric GARCH),
                "component" (component GARCH), "figarch" (FIGARCH(1, d, 1)), "realized" (realized GARCH(1, 1) on the
                daily Parkinson variance). Defaults to "garch".
            distribution (str, optional): The distribution of the standardized innovations z_t. Options are: "normal"
                (normal), "t" (Student-t, with a Shape v > 2), "skewt" (Hansen's skewed Student-t, with a Shape and a
                Skew λ in (-1, 1)), "skewt_fernandez_steel" (Fernández and Steel's skewed Student-t, the same
                distribution with a Skew ξ = √((1 + λ) / (1 - λ)), as in R's rugarch), "ged" (generalized error
                distribution, with a Shape v, 2 for the normal), "johnson_su" (Johnson's SU, with a Shape and a Skew).
                Defaults to "normal".
            mean (str, optional): The mean model. Options are: "zero" (zero mean, the returns themselves are the
                shocks), "constant" (constant mean Mu), "arma" (ARMA(1, 1)), "in_mean" (GARCH-in-mean, Mu + Lambda *
                sigma_t), "arfima" (ARFIMA(1, d, 1)). Defaults to "zero".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to calculate the growth of the GARCH values over time. Defaults to
            False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame | pd.Series: sigma_2 forecast values

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates the sigma_2
        forecast for each asset in the Toolkit instance.
        - Every asset is fitted on its own available returns, so an asset with a shorter history (or a gap)
        does not shorten the sample of the others.
        - The EGARCH fit is a deterministic multi-start search that takes far longer than the GARCH and
        GJR-GARCH fits, about 45 seconds per ticker of daily returns. The AVGARCH and NAGARCH recursions and the
        GARCH-in-mean filter also run step by step and take several times longer than the other models; the
        others take one to a few seconds per ticker of ten years of daily returns.
        - Every model other than GARCH, GJR-GARCH and EGARCH with normal innovations and a zero mean is fitted on
        the returns divided by their standard deviation, which puts every parameter on a scale of about 1, and the
        parameters are converted back (see `get_garch_parameters`).
        - With growth, the method calculates the growth of the forecasted sigma_2 values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["^GSPC"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_garch_forecast()
        ```

        Which returns:

        |            |   ^GSPC |   Benchmark |
        |:-----------|--------:|------------:|
        | 2026-01-01 |  0.0001 |      0      |
        | 2026-01-02 |  0.0001 |      0.0001 |
        | 2026-01-05 |  0.0001 |      0.0001 |
        | 2026-01-06 |  0.0001 |      0.0001 |
        | 2026-01-07 |  0.0001 |      0.0001 |
        | 2026-01-08 |  0.0001 |      0.0001 |
        | 2026-01-09 |  0.0001 |      0.0001 |
        | 2026-01-12 |  0.0001 |      0.0001 |
        | 2026-01-13 |  0.0001 |      0.0001 |
        | 2026-01-14 |  0.0001 |      0.0001 |
        """
        garch_model.validate_garch_family(method, distribution, mean)

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

        # Only rows without any return are dropped: every column is fitted on its own non-missing returns, so a ticker that listed later (or has a gap) does not cut the others down to its history.  # noqa: E501
        returns = self._get_returns(period, within_period).dropna(how="all")

        sigma_2_forecast = garch_model.get_garch_family_forecast(
            returns,
            method=method,
            distribution=distribution,
            mean=mean,
            forecast_steps=forecast_steps,
            realized_measure=self._get_realized_measure(method, period, within_period),
        )
        sigma_2_forecast.index = _forecast_index(returns.index, period, forecast_steps)

        return finalize_dataset(
            dataset=sigma_2_forecast,
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
    def get_garch_parameters(
        self,
        period: str | None = None,
        optimization_t: int | None = None,
        method: Literal[
            "garch",
            "gjr",
            "egarch",
            "igarch",
            "aparch",
            "tgarch",
            "ngarch",
            "avgarch",
            "nagarch",
            "component",
            "figarch",
            "realized",
        ] = "garch",
        distribution: str = "normal",
        mean: str = "zero",
        diagnostics: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the parameters of a GARCH model fitted to the returns: how strongly volatility reacts to a
        shock, how long it lasts and the level it reverts to.

        The model is a GARCH(1, 1), or one of its variants (see `get_garch`), fitted to the historical returns,
        with the estimated Omega, Alpha and Beta parameters for each asset, with the Gamma of the leverage effect for
        GJR-GARCH, EGARCH, APARCH, TGARCH and AVGARCH, the power Delta of APARCH and NGARCH, the Shift of AVGARCH and
        NAGARCH, the Rho and Phi of the long-run component of component GARCH, the Phi, D and Beta of FIGARCH, the
        Beta, Gamma and measurement equation (Xi, Phi, Tau 1, Tau 2, Sigma U) of the realized GARCH, the parameters
        of the mean model (Mu, AR(1), MA(1), Lambda, Fractional D) and the Shape and Skew of the innovation
        distribution. With the diagnostics,
        it adds the persistence, half-life and unconditional variance of the model, the log-likelihood and
        information criteria, and the standard and robust standard error of every parameter.

        GARCH (Generalized autoregressive conditional heteroskedasticity) is a stochastic model
        for time series, used to model volatility clustering. A positive Gamma of GJR-GARCH, or a negative
        Gamma of EGARCH, indicates the presence of a leverage effect (negative shocks raise volatility by
        more than positive ones of the same size), which symmetric GARCH cannot represent.

        The formula is as follows:

        - GARCH: sigma_t^2 = Omega + Alpha * r_(t-1)^2 + Beta * sigma_(t-1)^2
        - GJR-GARCH: sigma_t^2 = Omega + (Alpha + Gamma * I_(t-1)) * r_(t-1)^2 + Beta * sigma_(t-1)^2
        - EGARCH: ln sigma_t^2 = Omega + Alpha * (|z_(t-1)| - √(2 / π)) + Gamma * z_(t-1) + Beta * ln sigma_(t-1)^2
        - Log-Likelihood = -Σ (ln sigma_t^2 + r_t^2 / sigma_t^2) (up to constants)
        - The other models: see `get_garch`, with Log-Likelihood = Σ (ln f(z_t) - ln(sigma_t^2) / 2), f the density
        of the standardized innovation distribution
        - Persistence = Alpha + Beta (GARCH), Alpha + Gamma * E[z^2 1(z < 0)] + Beta (GJR-GARCH), Beta (EGARCH), 1 (IGARCH),
        Alpha * E[(|z| - Gamma * z)^Delta] + Beta (APARCH, TGARCH, NGARCH), Alpha * E[(|z - Shift| - Gamma *
        (z - Shift))^d] + Beta (AVGARCH, NAGARCH), Rho (component GARCH), 1 for FIGARCH with D > 0 (Phi with D = 0)
        or Beta + Gamma * Phi (realized GARCH, the persistence of the log variance)
        - Half-Life = ln(0.5) / ln(Persistence)
        - Unconditional Variance = (Omega / (1 - Persistence))^(2 / d), Omega / (1 - Rho) for component GARCH,
        exp(Omega / (1 - Beta)) for EGARCH, exp((Omega + Gamma * Xi) / (1 - Beta - Gamma * Phi)) for the realized
        GARCH and none (NaN) for IGARCH and FIGARCH with D > 0
        - AIC = (-2 LL + 2 k) / n, BIC = (-2 LL + k ln n) / n, Shibata = -2 LL / n + ln((n + 2 k) / n) and
        Hannan-Quinn = (-2 LL + 2 k ln ln n) / n
        - Standard Error = √diag(H^-1) and Robust Standard Error = √diag(H^-1 J H^-1)

        Where r_t are the returns (not demeaned), I_(t-1) is 1 when r_(t-1) < 0 and 0 otherwise and
        z_t = r_t / sigma_t. The constraints are Omega, Alpha, Beta > 0 and Alpha + Beta < 1 for GARCH,
        Alpha + Gamma / 2 + Beta < 1 and Alpha + Gamma >= 0 for GJR-GARCH, and Omega in [-15, 5], Alpha and
        Gamma in [-1, 1] and |Beta| < 1 for EGARCH. The GARCH parameters are estimated by maximizing the
        Gaussian log-likelihood with L-BFGS-B, started from a domain-informed initial guess and from 32 points
        of a Sobol sequence, and the best fit is finished with Nelder-Mead. The search runs on the parameters
        scaled to their bounds, as Omega (of the order of a daily variance) and Alpha and Beta differ by orders
        of magnitude.

        The variance recursion starts from a backcast, an exponentially weighted average
        (decay 0.94) of the first 75 squared returns, which stands in for both the squared
        return and the variance before the first observation (with the leverage term of GJR-GARCH at its
        expectation, Gamma / 2, and ln sigma_1^2 = Omega + Beta * ln(Backcast) for EGARCH), and every
        observation enters the likelihood. This is the start the arch package uses, so the estimates agree
        with it.

        The other models (and GARCH, GJR-GARCH and EGARCH with another distribution or mean) are fitted the same way
        on the returns divided by their standard deviation, with bounds for every parameter on that scale, and the
        parameters are converted back: Omega is multiplied by the scale to the power of the model (2, Delta, or 1
        for TGARCH and AVGARCH; EGARCH's Omega gains 2 * (1 - Beta) * ln(scale)) and Mu by the scale. The constraints
        are a persistence below 1 (IGARCH sets Beta = 1 - Alpha), |Gamma| < 1 for the rotation of APARCH and family
        GARCH, Delta in [0.05, 4], 0 < Alpha + Beta < Rho < 1 and Phi < Beta for component GARCH, D in [0, 1], Beta
        in [0, 1) and every ARCH(∞) weight non-negative for FIGARCH (Conrad and Haag, 2006), Beta + Gamma * Phi < 1
        for the realized GARCH, |AR(1)| and |MA(1)| < 1, a Fractional D in (-0.5, 0.5), a Shape in [2.05, 100] for the Student-t distributions, in [1.01, 50] for the GED and in
        [0.2, 20] for the Johnson SU, and a Skew in (-1, 1) for Hansen's, [0.1, 10] for Fernández and Steel's and
        [-10, 10] for the Johnson SU. The power models (APARCH and its sub-models) start from sigma_1^Delta = Omega +
        (Alpha + Beta) * Backcast^(Delta / 2), as the arch package does, and the component model from
        q_1 = sigma_1^2 = Omega + Rho * Backcast. The realized GARCH divides its realized measure by the square of
        the scale, so Omega gains 2 * (1 - Beta - Gamma) * ln(scale) and Xi 2 * (1 - Phi) * ln(scale), and starts
        from ln sigma_1^2 = Omega + Beta * ln(Backcast) + Gamma * ln(Measure Backcast), the measure averaged like the
        squared returns. Its log-likelihood is the joint one of the returns and the measure (Hansen, Huang and
        Shek, 2012), so its information criteria do not compare with those of the other models.

        The persistence is the factor by which a shock carries over to the next period on average, and the
        half-life the number of periods until half of it has faded. The expectations over the innovation
        distribution are evaluated numerically. The information criteria are per observation, as in R's rugarch, so
        the lowest value marks the preferred model among fits on the same returns. The standard errors come from
        the Hessian H of the log-likelihood and the robust ones (Bollerslev and Wooldridge, 1992) also from the
        outer product J of the scores of every observation, both by central finite differences; the robust ones
        remain valid when the innovation distribution is misspecified (quasi maximum likelihood).

        Unlike `get_garch` and `get_garch_forecast`, which return a (forecasted) conditional variance
        path, this method returns the fitted parameters themselves. This is useful when the parameters
        are needed directly, for example to seed a separate volatility simulation.

        For daily equity returns, the GARCH Alpha is typically between 0.05 and 0.15 and Alpha + Beta (the
        persistence) between 0.95 and 0.99, meaning a volatility shock fades over weeks rather than
        days. The long-run variance the model reverts to is Omega / (1 - Alpha - Beta). A fit far
        outside these ranges is a reason to check the data and the window before using the parameters.

        For more information about the methods, see the following sources:

        - Finance Compact Plus Band 1, by Yvonne Seler Zimmerman and Heinz Zimmerman; ISBN: 978-3-907291-31-1
        - Glosten, L.R., Jagannathan, R., and Runkle, D.E. (1993). "On the Relation
        between the Expected Value and the Volatility of the Nominal Excess Return on
        Stocks." The Journal of Finance, 48(5), 1779-1801.
        - Nelson, D.B. (1991). "Conditional Heteroskedasticity in Asset Returns: A New
        Approach." Econometrica, 59(2), 347-370.

        - Ding, Z., Granger, C.W.J. and Engle, R.F. (1993). "A Long Memory Property of Stock Market Returns and
        a New Model." Journal of Empirical Finance, 1(1), 83-106.
        - Hentschel, L. (1995). "All in the Family: Nesting Symmetric and Asymmetric GARCH Models." Journal of
        Financial Economics, 39(1), 71-104.
        - Bollerslev, T. (1987). "A Conditionally Heteroskedastic Time Series Model for Speculative Prices and
        Rates of Return." The Review of Economics and Statistics, 69(3), 542-547.
        - Hansen, B.E. (1994). "Autoregressive Conditional Density Estimation." International Economic Review,
        35(3), 705-730.
        - Fernández, C. and Steel, M.F.J. (1998). "On Bayesian Modeling of Fat Tails and Skewness." Journal of
        the American Statistical Association, 93(441), 359-371.
        - Johnson, N.L. (1949). "Systems of Frequency Curves Generated by Methods of Translation." Biometrika,
        36(1/2), 149-176.
        - Bollerslev, T. and Wooldridge, J.M. (1992). "Quasi-Maximum Likelihood Estimation and Inference in
        Dynamic Models with Time-Varying Covariances." Econometric Reviews, 11(2), 143-172.


        Also known as: GARCH weights, GARCH coefficients, conditional variance parameters, GJR-GARCH
        parameters, EGARCH parameters, leverage parameters, APARCH parameters, degrees of freedom, persistence,
        half-life, unconditional variance, long-run variance, information criteria, GARCH standard errors.

        Referenced against:
        - arch, GARCH: https://bashtage.github.io/arch/univariate/generated/arch.univariate.GARCH.html
        - arch, GARCH (GJR-GARCH): https://bashtage.github.io/arch/univariate/generated/arch.univariate.GARCH.html
        - arch, EGARCH: https://bashtage.github.io/arch/univariate/generated/arch.univariate.EGARCH.html
        - arch, APARCH (APARCH, TGARCH, NGARCH): https://bashtage.github.io/arch/univariate/generated/arch.univariate.APARCH.html
        - arch, FIGARCH: https://bashtage.github.io/arch/univariate/generated/arch.univariate.FIGARCH.html
        - arch, ARCHInMean (GARCH-in-mean): https://bashtage.github.io/arch/univariate/generated/arch.univariate.ARCHInMean.html
        - arch, GARCH (classic and robust standard errors, information criteria): https://bashtage.github.io/arch/univariate/generated/arch.univariate.GARCH.html

        Shares similar logic:
        - rugarch (R), ugarchspec (sGARCH, iGARCH, apARCH, fGARCH, csGARCH, realGARCH, ARFIMA mean): https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-methods.R#L21
        - rugarch (R), ugarchfit: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-methods.R#L1059
        - rugarch (R), persistence: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-methods.R#L2529
        - rugarch (R), halflife: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-methods.R#L2490
        - rugarch (R), uncvariance: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-methods.R#L3036
        - rugarch (R), infocriteria: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-methods.R#L2063

        Args:
            period (str, optional): The data frequency for returns. Options are: "daily", "weekly", "monthly",
                "quarterly", "yearly". Defaults to "daily".
            optimization_t (int, optional): Time steps of the returns series to use for the optimization.
            Defaults to the full length of the returns series.
            method (str, optional): The model. Options are: "garch" (GARCH(1, 1)), "gjr" (GJR-GARCH(1, 1, 1)),
                "egarch" (EGARCH(1, 1)), "igarch" (IGARCH(1, 1)), "aparch" (APARCH(1, 1)), "tgarch" (threshold GARCH),
                "ngarch" (nonlinear GARCH), "avgarch" (absolute value GARCH), "nagarch" (nonlinear asymmetric GARCH),
                "component" (component GARCH), "figarch" (FIGARCH(1, d, 1)), "realized" (realized GARCH(1, 1) on the
                daily Parkinson variance). Defaults to "garch".
            distribution (str, optional): The distribution of the standardized innovations z_t. Options are: "normal"
                (normal), "t" (Student-t, with a Shape v > 2), "skewt" (Hansen's skewed Student-t, with a Shape and a
                Skew λ in (-1, 1)), "skewt_fernandez_steel" (Fernández and Steel's skewed Student-t, the same
                distribution with a Skew ξ = √((1 + λ) / (1 - λ)), as in R's rugarch), "ged" (generalized error
                distribution, with a Shape v, 2 for the normal), "johnson_su" (Johnson's SU, with a Shape and a Skew).
                Defaults to "normal".
            mean (str, optional): The mean model. Options are: "zero" (zero mean, the returns themselves are the
                shocks), "constant" (constant mean Mu), "arma" (ARMA(1, 1)), "in_mean" (GARCH-in-mean, Mu + Lambda *
                sigma_t), "arfima" (ARFIMA(1, d, 1)). Defaults to "zero".
            diagnostics (bool, optional): Whether to add the Persistence, Half-Life (in periods), Unconditional
                Variance, Log-Likelihood, AIC, BIC, Shibata and Hannan-Quinn criteria and the Standard Error and Robust
                Standard Error of every parameter. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Omega, Alpha and Beta values per asset, with Gamma between Alpha and Beta for
            GJR-GARCH and EGARCH, the other parameters of the model as listed above (Beta = 1 - Alpha for
            IGARCH) and, with the diagnostics, the diagnostics below them.

        Notes:
        - The method retrieves historical return data based on the specified `period` and fits the model
        for each asset in the Toolkit instance.
        - Every asset is fitted on its own available returns, so an asset with a shorter history (or a gap)
        does not shorten the sample of the others.
        - GJR-GARCH nests GARCH (gamma = 0), so its fit never ends at a lower log-likelihood than the
        GARCH(1, 1) fit: if the search does, the GARCH weights with a Gamma of 0 are returned.
        - EGARCH's log-variance parameterization is less constrained than plain GARCH, which makes it
        prone to unstable, boundary-hugging fits on short return histories, which include quarterly
        data over a few years. Prefer daily or weekly returns. Its fit is a deterministic multi-start search
        that takes far longer than the others, about 45 seconds per ticker of daily returns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["LOW", "SBUX", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_garch_parameters(rounding=6)
        ```

        Which returns:

        |       |      LOW |     SBUX |     TSLA |   Benchmark |
        |:------|---------:|---------:|---------:|------------:|
        | Omega | 2.1e-05  | 0.000102 | 1.6e-05  |    5e-06    |
        | Alpha | 0.035851 | 0.057577 | 0.048624 |    0.153925 |
        | Beta  | 0.87526  | 0.716127 | 0.940179 |    0.811124 |
        """
        garch_model.validate_garch_family(method, distribution, mean)

        period = period if period else "daily"

        check_period(period, ["daily", "weekly", "monthly", "quarterly", "yearly"])

        returns = self._historical_data[period]["Return"]
        names = garch_model.get_garch_family_parameter_names(method, distribution, mean)
        measure = self._get_realized_measure(method, period)

        columns = {}
        for ticker in returns.columns:
            # Each ticker is fitted on its own non-missing returns, see get_garch.
            ticker_series = returns[ticker].dropna()
            ticker_returns = ticker_series.to_numpy()
            ticker_measure = (
                measure[ticker].reindex(ticker_series.index).to_numpy(dtype=float)
                if measure is not None
                else None
            )
            weights = garch_model.get_garch_family_weights(
                ticker_returns,
                method,
                distribution,
                mean,
                optimization_t,
                realized_measure=ticker_measure,
            )
            values = dict(zip(names, weights, strict=True))
            if method == "igarch":
                # IGARCH's Beta is not estimated but follows from Alpha + Beta = 1.
                values = {
                    "Omega": values.pop("Omega"),
                    "Alpha": values.pop("Alpha"),
                    "Beta": 1 - weights[1],
                    **values,
                }
            if diagnostics:
                fitted = (
                    ticker_returns[:optimization_t]
                    if optimization_t is not None
                    else ticker_returns
                )
                values.update(
                    garch_model.get_garch_family_diagnostics(
                        fitted,
                        weights,
                        method,
                        distribution,
                        mean,
                        realized_measure=(
                            ticker_measure[: len(fitted)]
                            if ticker_measure is not None
                            else None
                        ),
                    )
                    if np.all(np.isfinite(weights))
                    else {}
                )
            columns[ticker] = pd.Series(values, dtype=float)

        parameters = pd.DataFrame(columns)

        return apply_rounding(
            parameters, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_news_impact_curve(
        self,
        period: str | None = None,
        method: Literal[
            "garch",
            "gjr",
            "egarch",
            "igarch",
            "aparch",
            "tgarch",
            "ngarch",
            "avgarch",
            "nagarch",
            "component",
            "figarch",
            "realized",
        ] = "garch",
        distribution: str = "normal",
        mean: str = "zero",
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the news impact curve of Engle and Ng (1993) of a GARCH-family model fitted to the returns of
        each asset: how a shock of each size moves the next period's volatility, the conditional variance of the next
        period as a function of this period's shock, with this period's variance at the unconditional (long-run)
        variance of the model.

        The curve shows how a model lets news move volatility. For GARCH it is a parabola that is symmetric around
        zero: a fall raises the next variance by as much as a rise of the same size. The asymmetric models
        (GJR-GARCH, EGARCH, APARCH, TGARCH, AVGARCH and NAGARCH) let it rise more steeply to the left when they find
        a leverage effect, so bad news raises volatility by more than good news, and the power models bend it into
        a different shape than a parabola.

        The formula is as follows:

        - News Impact(z) = sigma_(t+1)^2 given e_t = z * sigma and sigma_t^2 = sigma^2

        Where z is the shock in unconditional standard deviations (-5 to 5 in steps of 0.5), sigma^2 the
        unconditional variance of the model (see `get_garch_parameters`) and sigma_(t+1)^2 follows from the
        variance recursion of the model (see `get_garch`). The long-run component of component GARCH is also set
        to sigma^2. IGARCH and FIGARCH (with D > 0) have no unconditional variance, so their curve is NaN; the curve
        of EGARCH is taken at exp(Omega / (1 - Beta)), the exponential of its long-run log variance. The shock of the
        realized GARCH moves the next log variance through its measurement equation (Hansen, Huang and Shek,
        2012): News Impact(z) = exp(ln sigma^2 + Gamma * (Tau 1 * z + Tau 2 * (z^2 - 1))), with sigma^2 its
        unconditional variance exp((Omega + Gamma * Xi) / (1 - Beta - Gamma * Phi)).

        For more information, see the following paper:

        - Engle, R.F. and Ng, V.K. (1993). "Measuring and Testing the Impact of News on Volatility." The Journal
        of Finance, 48(5), 1749-1778.

        Also known as: news impact, leverage effect curve, asymmetric volatility response.

        Shares similar logic:
        - rugarch (R), newsimpact: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-methods.R#L2287

        Args:
            period (str, optional): The data frequency for returns. Options are: "daily", "weekly", "monthly",
                "quarterly", "yearly". Defaults to "daily".
            method (str, optional): The model. Options are: "garch" (GARCH(1, 1)), "gjr" (GJR-GARCH(1, 1, 1)),
                "egarch" (EGARCH(1, 1)), "igarch" (IGARCH(1, 1)), "aparch" (APARCH(1, 1)), "tgarch" (threshold GARCH),
                "ngarch" (nonlinear GARCH), "avgarch" (absolute value GARCH), "nagarch" (nonlinear asymmetric GARCH),
                "component" (component GARCH), "figarch" (FIGARCH(1, d, 1)), "realized" (realized GARCH(1, 1) on the
                daily Parkinson variance). Defaults to "garch".
            distribution (str, optional): The distribution of the standardized innovations z_t. Options are: "normal"
                (normal), "t" (Student-t, with a Shape v > 2), "skewt" (Hansen's skewed Student-t, with a Shape and a
                Skew λ in (-1, 1)), "skewt_fernandez_steel" (Fernández and Steel's skewed Student-t, the same
                distribution with a Skew ξ = √((1 + λ) / (1 - λ)), as in R's rugarch), "ged" (generalized error
                distribution, with a Shape v, 2 for the normal), "johnson_su" (Johnson's SU, with a Shape and a Skew).
                Defaults to "normal".
            mean (str, optional): The mean model. Options are: "zero" (zero mean, the returns themselves are the
                shocks), "constant" (constant mean Mu), "arma" (ARMA(1, 1)), "in_mean" (GARCH-in-mean, Mu + Lambda *
                sigma_t), "arfima" (ARFIMA(1, d, 1)). Defaults to "zero".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The conditional variance of the next period per asset, indexed by the shock z.

        Notes:
        - Every asset is fitted on its own available returns, see `get_garch_parameters`.
        - The variance is that of a single period of the chosen `period`; its square root is the volatility.
        - A fit with a persistence of at least 1 - 10^-6 is treated as integrated, as IGARCH, so its curve is NaN:
          its unconditional variance Omega / (1 - P) is unbounded as the persistence P approaches 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2016-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_news_impact_curve(method="gjr", rounding=6).iloc[::4]
        ```

        Which returns:

        |   Shock |     AAPL |     MSFT |   Benchmark |
        |--------:|---------:|---------:|------------:|
        |      -5 | 0.002689 | 0.001411 |    0.001215 |
        |      -3 | 0.001218 | 0.000689 |    0.000523 |
        |      -1 | 0.000483 | 0.000328 |    0.000176 |
        |       1 | 0.000398 | 0.000288 |    0.000139 |
        |       3 | 0.000452 | 0.000327 |    0.00019  |
        |       5 | 0.000559 | 0.000404 |    0.000292 |
        """
        garch_model.validate_garch_family(method, distribution, mean)

        period = period if period else "daily"

        check_period(period, ["daily", "weekly", "monthly", "quarterly", "yearly"])

        returns = self._historical_data[period]["Return"]
        measure = self._get_realized_measure(method, period)

        news_impact = pd.DataFrame(
            {
                # Each ticker is fitted on its own non-missing returns, see get_garch.
                ticker: garch_model.get_news_impact_curve(
                    garch_model.get_garch_family_weights(
                        returns[ticker].dropna().to_numpy(),
                        method,
                        distribution,
                        mean,
                        realized_measure=(
                            measure[ticker]
                            .reindex(returns[ticker].dropna().index)
                            .to_numpy(dtype=float)
                            if measure is not None
                            else None
                        ),
                    ),
                    method,
                    distribution,
                    mean,
                )
                for ticker in returns.columns
            }
        )

        return apply_rounding(
            news_impact, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_garch_simulation(
        self,
        period: str | None = None,
        forecast_steps: int = 10,
        simulations: int = 10000,
        seed: int | None = 0,
        output: str = "returns",
        method: Literal[
            "garch",
            "gjr",
            "egarch",
            "igarch",
            "aparch",
            "tgarch",
            "ngarch",
            "avgarch",
            "nagarch",
            "component",
            "figarch",
            "realized",
        ] = "garch",
        distribution: str = "normal",
        mean: str = "zero",
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
        bootstrap_fits: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate a GARCH simulation of each asset: the spread of the returns (or the conditional volatility)
        a GARCH-family model fitted to its returns expects for the periods after the last return.

        The paths are summarized per period: the mean and the 5th, 25th, 50th, 75th and 95th percentiles.

        Unlike `get_garch_forecast`, which gives the expected variance, the simulation shows the whole spread of
        outcomes the model implies, including the fat tails and skewness of the innovation distribution and the
        way a large shock raises the volatility of the periods after it. The percentiles of the cumulative return
        give a GARCH-based Value at Risk over the horizon.

        The formula is as follows, for h = 1, ..., `forecast_steps`:

        - e_(n+h) = sigma_(n+h) * z_(n+h)
        - r_(n+h) = Mean_(n+h) + e_(n+h)
        - sigma_(n+h+1)^2 from the variance recursion of the model (see `get_garch`)

        Where n is the last return, sigma_(n+1)^2 the one-step-ahead variance of the fitted model, z drawn from
        the fitted innovation distribution and Mean_(n+h) = 0, Mu, Mu + AR(1) * r_(n+h-1) + MA(1) * e_(n+h-1) or
        Mu + Lambda * sigma_(n+h) by the mean model (the last return and residual start the ARMA mean; the ARFIMA
        mean continues its fractional difference over every return). FIGARCH continues its sum over the squared
        residuals of the returns, and the realized GARCH draws its realized measure from the measurement equation
        with a normal error u ~ N(0, Sigma U^2). The cumulative return compounds the simulated returns,
        (1 + r_(n+1)) * ... * (1 + r_(n+h)) - 1, treating them as simple returns.

        With bootstrap fits, the bootstrap forecast density of Pascual, Romo and Ruiz (2006) replaces the fitted
        innovation distribution with the model's own standardized residuals and adds the uncertainty of the
        estimated parameters:

        - z*_t drawn with replacement from the standardized residuals z_t of the fit (and, for the realized
          GARCH, the measurement error of the same day)
        - a series r*_1, ..., r*_n as long as the returns simulated from the fitted parameters with those draws,
          starting after the last return, and the model fitted to it again, `bootstrap_fits` times
        - every refitted model filters the actual returns to its own sigma_(n+1)^2 and simulates `simulations` /
          `bootstrap_fits` of the paths from there with the resampled z*
        - no bootstrap fits (0) keeps the fitted parameters and only resamples the residuals (the partial bootstrap)

        The spread of the first period's volatility then shows the parameter uncertainty, which a simulation
        from the fitted parameters does not have.

        For more information, see the following paper:

        - Pascual, L., Romo, J. and Ruiz, E. (2006). "Bootstrap Prediction for Returns and Volatilities in GARCH
        Models." Computational Statistics & Data Analysis, 50(9), 2293-2312.

        Also known as: GARCH Monte Carlo, volatility simulation, GARCH scenarios, simulated return paths, GARCH
        bootstrap, bootstrap forecast density, ugarchboot, parameter uncertainty.

        Shares similar logic:
        - rugarch (R), ugarchsim: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-methods.R#L1090
        - rugarch (R), ugarchpath: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-methods.R#L1105
        - rugarch (R), ugarchboot (bootstrap forecast density): https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-methods.R#L1210

        Args:
            period (str, optional): The data frequency for returns. Options are: "daily", "weekly", "monthly",
                "quarterly", "yearly". Defaults to "daily".
            forecast_steps (int, optional): The number of periods to simulate. Defaults to 10.
            simulations (int, optional): The number of simulated paths per asset. Defaults to 10,000.
            seed (int | None, optional): The seed of the random draws, None for a different simulation every
                time. Defaults to 0.
            output (str, optional): What to simulate. Options are: "returns" (the return of each period),
                "cumulative_returns" (the compounded return from the last return to each period), "volatility" (the
                conditional volatility sigma of each period). Defaults to "returns".
            method (str, optional): The model. Options are: "garch" (GARCH(1, 1)), "gjr" (GJR-GARCH(1, 1, 1)),
                "egarch" (EGARCH(1, 1)), "igarch" (IGARCH(1, 1)), "aparch" (APARCH(1, 1)), "tgarch" (threshold GARCH),
                "ngarch" (nonlinear GARCH), "avgarch" (absolute value GARCH), "nagarch" (nonlinear asymmetric GARCH),
                "component" (component GARCH), "figarch" (FIGARCH(1, d, 1)), "realized" (realized GARCH(1, 1) on the
                daily Parkinson variance). Defaults to "garch".
            distribution (str, optional): The distribution of the standardized innovations z_t. Options are: "normal"
                (normal), "t" (Student-t, with a Shape v > 2), "skewt" (Hansen's skewed Student-t, with a Shape and a
                Skew λ in (-1, 1)), "skewt_fernandez_steel" (Fernández and Steel's skewed Student-t, the same
                distribution with a Skew ξ = √((1 + λ) / (1 - λ)), as in R's rugarch), "ged" (generalized error
                distribution, with a Shape v, 2 for the normal), "johnson_su" (Johnson's SU, with a Shape and a Skew).
                Defaults to "normal".
            mean (str, optional): The mean model. Options are: "zero" (zero mean, the returns themselves are the
                shocks), "constant" (constant mean Mu), "arma" (ARMA(1, 1)), "in_mean" (GARCH-in-mean, Mu + Lambda *
                sigma_t), "arfima" (ARFIMA(1, d, 1)). Defaults to "zero".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.
            bootstrap_fits (int | None, optional): The number of bootstrap series the model is fitted to again
                for the bootstrap forecast density of Pascual, Romo and Ruiz, 0 for the partial bootstrap that only
                resamples the residuals. Defaults to None, which draws from the fitted innovation distribution.

        Returns:
            pd.DataFrame: The Mean, 5%, 25%, 50%, 75% and 95% of the simulated values per asset, indexed by the
            periods after the last return.

        Notes:
        - Every asset is fitted on its own available returns, see `get_garch_parameters`.
        - The paths of every asset are drawn independently of the other assets.
        - Every bootstrap refit is a local search from the fitted parameters (with the multi-start search when it
          starts or ends on a bound), about 0.5 seconds for GARCH(1, 1) on ten years of daily returns, several
          seconds for FIGARCH, the realized GARCH and the ARFIMA mean.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2016-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_garch_simulation(
            output="cumulative_returns", distribution="t", show_columns="AAPL"
        )
        ```

        Which returns:

        |            |    Mean |      5% |     25% |     50% |    75% |    95% |
        |:-----------|--------:|--------:|--------:|--------:|-------:|-------:|
        | 2026-01-01 |  0      | -0.0155 | -0.0054 | -0.0002 | 0.0053 | 0.0151 |
        | 2026-01-02 | -0.0001 | -0.0227 | -0.0084 | -0.0001 | 0.0079 | 0.0225 |
        | 2026-01-05 | -0.0002 | -0.0286 | -0.0106 | -0.0003 | 0.0102 | 0.0279 |
        | 2026-01-06 | -0.0003 | -0.033  | -0.0126 | -0.0004 | 0.012  | 0.0326 |
        | 2026-01-07 | -0.0003 | -0.038  | -0.0144 | -0.0002 | 0.0138 | 0.0372 |
        | 2026-01-08 | -0.0002 | -0.0419 | -0.0159 | -0.0005 | 0.0152 | 0.0418 |
        | 2026-01-09 | -0.0002 | -0.045  | -0.0176 | -0.0009 | 0.0167 | 0.0456 |
        | 2026-01-12 | -0.0001 | -0.0487 | -0.0188 | -0.0004 | 0.0181 | 0.0497 |
        | 2026-01-13 | -0.0001 | -0.0532 | -0.0201 | -0.0006 | 0.0193 | 0.0528 |
        | 2026-01-14 | -0.0003 | -0.0561 | -0.0214 | -0.0007 | 0.0208 | 0.0561 |
        """
        garch_model.validate_garch_family(method, distribution, mean)

        period = period if period else "daily"

        check_period(period, ["daily", "weekly", "monthly", "quarterly", "yearly"])

        returns = self._historical_data[period]["Return"].dropna(how="all")
        measure = self._get_realized_measure(method, period)

        simulation = pd.concat(
            {
                # Each ticker is fitted on its own non-missing returns, see get_garch.
                ticker: garch_model.get_garch_simulation(
                    returns[ticker].dropna().to_numpy(),
                    method=method,
                    distribution=distribution,
                    mean=mean,
                    forecast_steps=forecast_steps,
                    simulations=simulations,
                    seed=seed,
                    output=output,
                    bootstrap_fits=bootstrap_fits,
                    realized_measure=(
                        measure[ticker]
                        .reindex(returns[ticker].dropna().index)
                        .to_numpy(dtype=float)
                        if measure is not None
                        else None
                    ),
                )
                for ticker in returns.columns
            },
            axis=1,
        )
        simulation.index = _forecast_index(returns.index, period, forecast_steps)

        return apply_rounding(
            simulation, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_volatility(
        self,
        period: str | None = None,
        rolling: int | None = None,
        method: Literal[
            "close_to_close",
            "parkinson",
            "garman_klass",
            "rogers_satchell",
            "yang_zhang",
            "garman_klass_yang_zhang",
            "compounded",
        ] = "close_to_close",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Volatility: how much the returns of an investment portfolio or asset fluctuate in each
        period, from the daily prices.

        Volatility measures the amount of dispersion or variability in returns. It is the
        square root of the Variance. A higher Volatility indicates greater variability, while
        a lower Volatility suggests that returns are closer to the mean.

        By default this is the close-to-close Volatility, i.e. the standard deviation of the
        daily returns. One of five range-based estimators can be used
        instead, each of which uses more of the day's price action than just the close and is
        therefore more statistically efficient (i.e. needs fewer observations to reach the
        same precision), at the cost of additional assumptions about how prices move:

        - Parkinson — uses the daily trading range (High vs Low) rather than the
        close-to-close return, assuming prices follow a continuous geometric Brownian motion
        with no drift and no overnight jumps.
        - Garman-Klass — extends Parkinson by also incorporating the Open and Close, which
        allows it to account for the opening jump and makes it more efficient still (assuming,
        as Parkinson does, no drift and no overnight jumps beyond the modeled open).
        - Rogers-Satchell — drift-independent, meaning it remains unbiased even when the
        underlying asset has a non-zero expected return over the period, at the cost of still
        assuming no overnight jumps.
        - Yang-Zhang — a weighted combination of the overnight (close-to-open) Variance,
        the open-to-close Variance and the Rogers-Satchell Variance. It is both
        drift-independent and accounts for overnight jumps, which makes it the most
        statistically efficient of the range-based estimators implemented here.
        - Garman-Klass Yang-Zhang — Garman-Klass with the squared overnight (close-to-open)
        return of Yang and Zhang added, so it accounts for overnight jumps while keeping the
        Garman-Klass term for the session (still assuming no drift).

        The compounded Volatility keeps the close-to-close returns but scales them differently: rather than
        the square root of time, it gives the standard deviation of the period's return when the
        daily returns compound, the exact value for independent daily returns with the measured
        mean and variance. It exceeds the close-to-close Volatility when the returns are large
        (a cryptocurrency's yearly Volatility) and barely differs for small ones.

        The formula for the close-to-close Volatility is as follows:

        - Volatility = Std(Daily Returns within the period) * √(Trading Days per Period)

        Where Std is the sample standard deviation (with n - 1 in the denominator) and the trading days per period are
        252 / 52 (weekly), 252 / 12 (monthly), 252 / 4 (quarterly) or 252 (yearly). A ticker with returns on weekends,
        such as a cryptocurrency, has a return on every calendar day and uses 365 days a year instead (365 / 52
        weekly), so its Volatility is not understated against the stocks. The range-based estimators take the square
        root of a daily variance and scale it the same way, with H, L, O and C the daily High, Low, Open and Close,
        C_(-1) the previous Close, Mean the average and Var the sample variance (with n - 1 in the denominator) over
        the days in the period, and the compounded Volatility uses the daily returns directly:

        - Parkinson Variance = Mean(ln(H / L)^2) / (4 * ln 2)
        - Garman-Klass Variance = Mean(0.5 * ln(H / L)^2 - (2 * ln 2 - 1) * ln(C / O)^2)
        - Rogers-Satchell Variance = Mean(ln(H / C) * ln(H / O) + ln(L / C) * ln(L / O))
        - Yang-Zhang Variance = Var(ln(O / C_(-1))) + k * Var(ln(C / O)) + (1 - k) * Rogers-Satchell Variance
        - k = 0.34 / (1.34 + (n + 1) / (n - 1)), with n the number of days in the period
        - Garman-Klass Yang-Zhang Variance = Mean(ln(O / C_(-1))^2 + 0.5 * ln(H / L)^2 - (2 * ln 2 - 1) * ln(C / O)^2)
        - Compounded Volatility = SQRT((Var(r) + (1 + g)^2)^P - (1 + g)^(2P)), with r the daily returns, g their
          geometric mean (PRODUCT(1 + r)^(1 / n) - 1 over the n days) and P the trading days per period

        With `rolling`, the result is the sample standard deviation of the last `rolling` returns of the
        chosen period itself, without scaling.

        Also known as: standard deviation of returns. The range-based estimators are also known
        as Parkinson's range-based or high-low Volatility, Garman-Klass range-based Volatility,
        Rogers-Satchell drift-independent Volatility, Yang-Zhang drift-independent
        overnight-aware Volatility and GK-YZ Volatility.

        For more information about the range-based estimators, see the following papers:

        - Parkinson, M. (1980). "The Extreme Value Method for Estimating the Variance of the
        Rate of Return." Journal of Business, 53(1), 61-65.
        - Garman, M.B., & Klass, M.J. (1980). "On the Estimation of Security Price
        Volatilities from Historical Data." Journal of Business, 53(1), 67-78.
        - Rogers, L.C.G., & Satchell, S.E. (1991). "Estimating Variance from High, Low and
        Closing Prices." Annals of Applied Probability, 1(4), 504-512.
        - Yang, D., & Zhang, Q. (2000). "Drift-Independent Volatility Estimation Based on
        High, Low, Open, and Close Prices." Journal of Business, 73(3), 477-492.

        Shares similar logic:
        - TTR (R), volatility (Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang, Garman-Klass Yang-Zhang): https://github.com/joshuaulrich/TTR/blob/v0.24.4/R/volatility.R#L139
        - backtesting.py, compute_stats (compounded, Volatility (Ann.) [%]): https://github.com/kernc/backtesting.py/blob/0.6.4/backtesting/_stats.py#L140

        Args:
            period (str, optional): The data frequency for returns. Options are: "weekly", "monthly", "quarterly",
                "yearly". Defaults to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise
                "yearly".
            rolling (int, optional): The rolling window size to use for the calculation. If set,
            Volatility is calculated over a rolling window of this many periods (e.g. period='monthly'
            and rolling=6 gives the rolling 6-month Volatility) instead of one value per `period`.
            Only available for method="close_to_close". At least 2. Defaults to None.
            method (str, optional): The Volatility estimator, as described above. Options are: "close_to_close"
                (close-to-close), "parkinson" (Parkinson), "garman_klass" (Garman-Klass), "rogers_satchell"
                (Rogers-Satchell), "yang_zhang" (Yang-Zhang), "garman_klass_yang_zhang" (Garman-Klass Yang-Zhang),
                "compounded" (compounded). Defaults to "close_to_close".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Volatility values over time.
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
            pd.Series: Volatility values with time as the index.

        Notes:
        - The method retrieves the daily historical price data and calculates the Volatility for
        the specified `period` for each asset in the Toolkit instance. The close-to-close estimator
        only needs the daily returns, Parkinson needs the High and Low, and the remaining
        estimators need the Open, High, Low and Close.
        - A range-based estimator leaves out each day on which a price it uses is 0 or below (a bad bar, as some
        feeds report for illiquid days), since its logarithm is undefined; the period is measured over the
        remaining days. The Yang-Zhang overnight return also needs the previous Close.
        - With growth, the method calculates the growth of Volatility values using the specified `lag`.
        - backtesting.py (Zach Lûster) applies the compounded Volatility to the daily returns of a whole backtest
        with 252 (or 365) days, and TTR scales the range-based estimators by a fixed number of days per year.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["EURUSD", "BTCUSD", "^GSPC", "GCUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_volatility(period="yearly")
        ```

        Which returns:

        | Date   |   EURUSD |   BTCUSD |   ^GSPC |   GCUSD |   Benchmark |
        |:-------|---------:|---------:|--------:|--------:|------------:|
        | 2021   |   0.0679 |   0.8133 |  0.1303 |  0.1752 |      0.1293 |
        | 2022   |   0.1199 |   0.639  |  0.2417 |  0.1881 |      0.2424 |
        | 2023   |   0.0895 |   0.4408 |  0.1309 |  0.1652 |      0.1308 |
        | 2024   |   0.0708 |   0.5345 |  0.1265 |  0.185  |      0.1257 |
        | 2025   |   0.089  |   0.4194 |  0.1875 |  0.2421 |      0.1949 |
        """
        check_option(
            "method",
            method,
            (
                "close_to_close",
                "parkinson",
                "garman_klass",
                "rogers_satchell",
                "yang_zhang",
                "garman_klass_yang_zhang",
                "compounded",
            ),
        )

        period = period if period else "quarterly" if self._quarterly else "yearly"

        check_period(period, ["weekly", "monthly", "quarterly", "yearly"])
        check_rolling(rolling, self._historical_data, period)

        if rolling and method != "close_to_close":
            raise InvalidArgumentError(
                "The rolling parameter is only available for method='close_to_close'."
            )

        if method == "close_to_close":
            if rolling:
                period_returns = self._historical_data[period]["Return"]
                volatility = risk_model.get_volatility(
                    period_returns, period, window=rolling
                )
            else:
                returns = self._historical_data["daily"]["Return"]
                volatility = risk_model.get_volatility(returns, period)
        elif method == "compounded":
            volatility = risk_model.get_compounded_volatility(
                self._historical_data["daily"]["Return"], period
            )
        elif method == "parkinson":
            volatility = realized_volatility_model.get_parkinson_volatility(
                self._get_prices("High"),
                self._get_prices("Low"),
                period,
            )
        else:
            estimators = {
                "garman_klass": realized_volatility_model.get_garman_klass_volatility,
                "rogers_satchell": realized_volatility_model.get_rogers_satchell_volatility,
                "yang_zhang": realized_volatility_model.get_yang_zhang_volatility,
                "garman_klass_yang_zhang": realized_volatility_model.get_garman_klass_yang_zhang_volatility,
            }

            volatility = estimators[method](
                self._get_prices("Open"),
                self._get_prices("High"),
                self._get_prices("Low"),
                self._get_prices("Close"),
                period,
            )

        volatility = volatility.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=volatility,
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
    def get_har_rv_forecast(
        self,
        estimator: str = "squared_return",
        weekly_window: int = 5,
        monthly_window: int = 22,
        horizon: int = 1,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Corsi (2009) Heterogeneous Autoregressive Realized Volatility (HAR-RV) forecast: the daily
        Realized Variance each asset is expected to have next, from its daily, weekly and monthly averages.

        Volatility clustering happens across multiple, overlapping time horizons at
        once. HAR-RV captures this cheaply -- without the numerical optimization a
        GARCH-family fit requires (see `get_garch`) -- by regressing future daily
        Realized Variance on trailing daily, weekly and monthly average Realized
        Variance components. The daily Realized Variance itself can be constructed in
        several ways: the simplest is the squared daily return, while
        the OHLC range-based estimators (see `get_volatility`, which offers the same
        estimators) use the daily
        (pre-period-aggregation) term behind each of those estimators instead,
        which is more statistically efficient since it uses the daily trading range
        rather than only the close-to-close move.

        The formula is as follows (with the squared return, a weekly window of 5 days, a monthly window of 22 days
        and a horizon of 1 day):

        - RV_t = Return_t^2
        - RV^w_t = Mean(RV_(t-4), ..., RV_t) and RV^m_t = Mean(RV_(t-21), ..., RV_t)
        - Forecast of RV_(t+1) = b_0 + b_d * RV_t + b_w * RV^w_t + b_m * RV^m_t

        Where b_0, b_d, b_w and b_m are estimated once by ordinary least squares over each asset's full
        history, and the forecast of RV_(t+1) is shown at date t. The range-based estimators replace RV_t with
        the daily Parkinson, Garman-Klass or Rogers-Satchell variance term (see `get_volatility`).

        For more information about the method, see the following paper:

        - Corsi, F. (2009). "A Simple Approximate Long-Memory Model of Realized
        Volatility." Journal of Financial Econometrics, 7(2), 174-196.

        Also known as: HAR-RV model, Corsi's HAR model, Heterogeneous Autoregressive model.

        Args:
            estimator (str, optional): How to construct the daily Realized Variance input. Options are:
                "squared_return" (the squared daily return), "parkinson" (Parkinson), "garman_klass" (Garman-Klass),
                "rogers_satchell" (Rogers-Satchell). Defaults to "squared_return".
            weekly_window (int, optional): The trailing window (in trading days) for the weekly RV
            component. Defaults to 5.
            monthly_window (int, optional): The trailing window (in trading days) for the monthly RV
            component. Defaults to 22.
            horizon (int, optional): The number of days ahead to forecast. Defaults to 1.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the HAR-RV forecast values over
            time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: The HAR-RV forecast of Realized Variance with time as the index.

        Notes:
        - The method retrieves daily historical price data and calculates the HAR-RV forecast for each
        asset in the Toolkit instance.
        - With growth, the method calculates the growth of the HAR-RV forecast values
        using the specified `lag`.

        In the example below, the last row is the forecast for the first trading day after the end date,
        which has not been observed yet.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["SPY", "QQQ"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
            benchmark_ticker=None,
        )

        toolkit.risk.get_har_rv_forecast(estimator="squared_return").tail(5)
        ```

        Which returns:

        | Date       |    SPY |    QQQ |
        |:-----------|-------:|-------:|
        | 2025-12-24 | 0.0001 | 0.0001 |
        | 2025-12-26 | 0.0001 | 0.0001 |
        | 2025-12-29 | 0.0001 | 0.0001 |
        | 2025-12-30 | 0.0001 | 0.0001 |
        | 2025-12-31 | 0.0001 | 0.0001 |
        """
        check_option(
            "estimator",
            estimator,
            (
                "squared_return",
                "parkinson",
                "garman_klass",
                "rogers_satchell",
            ),
        )

        if estimator == "squared_return":
            # The forecast drops missing values per ticker, so only rows without any return go.
            realized_variance = (
                self._historical_data["daily"]["Return"].dropna(how="all") ** 2
            )
        else:
            # A bad bar (a price of 0 or below) has no logarithm and is left out, as in
            # get_volatility, instead of putting an infinite variance into the regression.
            open_prices, high_prices, low_prices, close_prices = (
                prices.where(prices > 0)
                for prices in (
                    self._get_prices(column)
                    for column in ("Open", "High", "Low", "Close")
                )
            )

            if estimator == "parkinson":
                realized_variance = np.log(high_prices / low_prices) ** 2 / (
                    4 * np.log(2)
                )
            elif estimator == "garman_klass":
                realized_variance = (
                    0.5 * np.log(high_prices / low_prices) ** 2
                    - (2 * np.log(2) - 1) * np.log(close_prices / open_prices) ** 2
                )
            else:
                realized_variance = np.log(high_prices / close_prices) * np.log(
                    high_prices / open_prices
                ) + np.log(low_prices / close_prices) * np.log(low_prices / open_prices)

        forecast = realized_volatility_model.get_har_rv_forecast(
            realized_variance, weekly_window, monthly_window, horizon
        )

        forecast = forecast.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=forecast,
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
    def get_excess_volatility(
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
        Calculate the Excess Volatility: how much the returns of an investment portfolio or asset above the risk
        free rate fluctuate in each period.

        Excess Volatility is the Volatility of the Excess Return, i.e. the daily return minus
        the risk free rate, scaled to the given period in the same way as the Volatility.

        The formula is as follows:

        - Excess Volatility = Std(Daily Return - Daily Risk Free Rate) * √(Trading Days per Period)

        Where Std is the sample standard deviation (with n - 1 in the denominator) over the days within the period,
        the annualized risk free rate is converted to the rate earned with each return (a trading day of it, 1 / 252
        of a year, or for a ticker that trades on weekends the calendar days since its previous return, on 365 days a
        year), and the trading days per period are 252 / 52 (weekly), 252 / 12 (monthly), 252 / 4 (quarterly) or 252
        (yearly), or 365 days a year for a ticker with returns on weekends. With `rolling`, the result is the sample
        standard deviation of the last `rolling` excess returns of the chosen period itself, without scaling, with the
        risk free rate compounded over the dates in each period on which the ticker has a return.

        Also known as: standard deviation of excess returns.

        Args:
            period (str, optional): The data frequency for returns. Options are: "weekly", "monthly", "quarterly",
                "yearly". Defaults to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise
                "yearly".
            rolling (int, optional): The rolling window size to use for the calculation. If set,
            Excess Volatility is calculated over a rolling window of this many periods (e.g.
            period='monthly' and rolling=6 gives the rolling 6-month Excess Volatility) instead of
            one value per `period`. At least 2. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Excess Volatility values
            over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: Excess Volatility values with time as the index.

        Notes:
        - The method retrieves the daily historical return data and calculates the Excess Volatility for
        the specified `period` for each asset in the Toolkit instance.
        - The risk-free rate is often represented by the return of a risk-free investment, such as a Treasury bond.
        - With growth, the method calculates the growth of Excess Volatility values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["ADBE", "CRM", "ORCL"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_excess_volatility(period="yearly")
        ```

        Which returns:

        | Date   |   ADBE |    CRM |   ORCL |   Benchmark |
        |:-------|-------:|-------:|-------:|------------:|
        | 2021   | 0.2924 | 0.282  | 0.2927 |      0.1293 |
        | 2022   | 0.4506 | 0.4722 | 0.3053 |      0.2425 |
        | 2023   | 0.3178 | 0.2958 | 0.302  |      0.1307 |
        | 2024   | 0.3663 | 0.3594 | 0.3226 |      0.1257 |
        | 2025   | 0.3062 | 0.319  | 0.6169 |      0.1948 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        check_period(period, ["weekly", "monthly", "quarterly", "yearly"])
        check_rolling(rolling, self._historical_data, period)

        if rolling:
            period_returns = self._historical_data[period]["Return"]
            # Each ticker earns the risk free rate compounded over its own dates in each period, so a weekend at the turn of a month that only a cryptocurrency trades on stays in the month of a stock's next trading day.  # noqa: E501
            risk_free_rate = self._risk_free_rate_data[period].reindex(
                index=period_returns.index, columns=period_returns.columns
            )
            excess_volatility = risk_model.get_excess_volatility(
                period_returns, risk_free_rate, period, window=rolling
            )
        else:
            returns = self._historical_data["daily"]["Return"]
            # Each ticker earns the risk free rate per observation of its own calendar: a trading day of it per return, or the calendar days since its previous return for a ticker that trades on weekends.  # noqa: E501
            risk_free_rate = self._risk_free_rate_data["daily"][returns.columns]

            excess_volatility = risk_model.get_excess_volatility(
                returns, risk_free_rate, period
            )

        excess_volatility = excess_volatility.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=excess_volatility,
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
    def get_downside_deviation(
        self,
        period: str | None = None,
        minimum_acceptable_return: float | str = 0.0,
        within_period: bool = False,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Downside Deviation: how widely the returns of an investment portfolio or asset fall below a
        minimum acceptable return, the volatility of the losses.

        The Downside Deviation, also known as semi-deviation, measures the dispersion of the returns
        below a minimum acceptable return (MAR), isolating the volatility of negative outcomes from the
        volatility of the overall return distribution. It underlies risk-adjusted
        return measures such as the Sortino Ratio and the Omega Ratio.

        The formula is as follows:

        - Downside Deviation = √(Σ min(Return_t - MAR, 0)^2 / n)

        Where n is the number of all returns, so returns above the MAR count as a shortfall of zero rather than
        being left out.

        With the mean as the minimum acceptable return it is the Semi-Deviation of Markowitz (1959), the dispersion
        below each sample's own mean:

        - Semi-Deviation = √(Σ min(Return_t - Mean, 0)^2 / (n - 1))

        Divided by n - 1, as the sample variance is, since the mean is estimated from the same returns. Its
        square is the semi-variance.

        For more information, see the following sources:

        - Sortino, F.A. and Price, L.N. (1994). "Performance Measurement in a Downside Risk Framework." The
        Journal of Investing, 3(3), 59-64.
        - Markowitz, H.M. (1959). "Portfolio Selection: Efficient Diversification of Investments." John
        Wiley & Sons, chapter 9 (the semi-variance).

        Also known as: semi-deviation, semi-variance (its square), downside risk, downside volatility.

        Referenced against:
        - Riskfolio-Lib, SemiDeviation: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.SemiDeviation
        - skfolio, semi_deviation: https://skfolio.org/generated/skfolio.measures.semi_deviation.html

        Shares similar logic:
        - PerformanceAnalytics (R), DownsideDeviation: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/DownsideDeviation.R#L104
        - PerformanceAnalytics (R), SemiDeviation: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/SemiDeviation.R#L4
        - skfolio, semi_variance: https://skfolio.org/generated/skfolio.measures.semi_variance.html

        Args:
            period (str, optional): The data frequency for returns. Options are: "daily", "weekly", "monthly",
                "quarterly", "yearly". Defaults to "daily".
            minimum_acceptable_return (float | str, optional): The minimum acceptable return (MAR) used as the
            threshold below which returns are considered downside, per observation of the chosen `period`, or
            "mean" for the Semi-Deviation below the mean. Defaults to 0.0.
            within_period (bool, optional): Whether to calculate the Downside Deviation within the specified
            period or for the entire period. Thus whether to look at the Downside Deviation within a specific
            year (if period = 'yearly') or look at the entirety of all years. Defaults to False.
            Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rolling (int, optional): The rolling window size to use for the calculation. If set, the Downside
            Deviation is calculated over a rolling window of this many periods across the full return history
            instead of per `period`. At least 2. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Downside Deviation values over
            time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: Downside Deviation values with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates the
        Downside Deviation for each asset in the Toolkit instance.
        - With growth, the method calculates the growth of the Downside Deviation values using
        the specified `lag`.
        - With the default MAR of 0 the result agrees with PerformanceAnalytics' DownsideDeviation over all
        returns. With the mean as the MAR it agrees with Riskfolio-Lib's SemiDeviation and skfolio's
        semi_deviation (skfolio's semi_variance is its square), while PerformanceAnalytics' SemiDeviation
        divides by n.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NEE", "DUK", "SO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_downside_deviation()
        ```

        Which returns:

        |           |   Value |
        |:----------|--------:|
        | NEE       |  0.0124 |
        | DUK       |  0.0074 |
        | SO        |  0.0079 |
        | Benchmark |  0.0081 |
        """
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
            downside_deviation = risk_model.get_downside_deviation(
                returns, minimum_acceptable_return, window=rolling
            )
        else:
            returns = self._get_returns(period, within_period)

            downside_deviation = risk_model.get_downside_deviation(
                returns, minimum_acceptable_return
            )

        if rolling or within_period:
            downside_deviation = downside_deviation.loc[
                self._start_date : self._end_date
            ]

        return finalize_dataset(
            dataset=downside_deviation,
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
    def get_upside_deviation(
        self,
        period: str | None = None,
        minimum_acceptable_return: float = 0.0,
        within_period: bool = False,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Upside Deviation: how widely the returns of an investment portfolio or asset rise above a
        minimum acceptable return, the volatility an investor welcomes.

        The Upside Deviation is the mirror image of the Downside Deviation: it measures the dispersion of
        the returns above a minimum acceptable return (MAR), the volatility an investor welcomes. Next to the
        Downside Deviation it shows whether an asset's volatility comes from its gains or from its losses,
        and their ratio is the Variability Skewness (`get_volatility_skewness`).

        The formula is as follows:

        - Upside Deviation = √(Σ max(Return_t - MAR, 0)^2 / n)

        Where n is the number of all returns, so returns below the MAR count as zero rather than being left
        out. Its square is the upside variance and the upper partial moment of order 1 the upside potential
        (`get_partial_moment` on the upper side).

        For more information, see the following sources:

        - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution." John Wiley &
        Sons, chapter 4 (upside risk).

        Also known as: upside risk, upside volatility, upside semi-deviation.

        Shares similar logic:
        - PerformanceAnalytics (R), UpsideRisk: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/UpsideRisk.R#L54

        Args:
            period (str, optional): The data frequency for returns. Options are: "daily", "weekly", "monthly",
                "quarterly", "yearly". Defaults to "daily".
            minimum_acceptable_return (float, optional): The minimum acceptable return (MAR) per
            observation of the chosen `period` (e.g. per day), above which returns count as upside.
            Defaults to 0.0.
            within_period (bool, optional): Whether to calculate the Upside Deviation within the specified period or
            for the entire period. Thus whether to look at the Upside Deviation within a specific year (if period =
            'yearly') or look at the entirety of all years. Defaults to False.
            Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rolling (int, optional): The rolling window size to use for the calculation. If set, the Upside Deviation
            is calculated over a rolling window of this many periods across the full return history
            instead of per `period`. At least 2. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Upside Deviation values over time.
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
            pd.Series | pd.DataFrame: Upside Deviation values, per ticker, or per period (or date) and ticker.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates the
        Upside Deviation for each asset in the Toolkit instance, each on its own returns (a ticker that listed later
        or misses a date another ticker traded is measured on the returns it has).
        - With growth, the method calculates the growth of the Upside Deviation values using the
        specified `lag`.
        - The result agrees with the upside risk of PerformanceAnalytics' UpsideRisk over all returns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NEE", "DUK", "SO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_upside_deviation(period="yearly", within_period=True)
        ```

        Which returns:

        |      |    NEE |    DUK |     SO |   Benchmark |
        |:-----|-------:|-------:|-------:|------------:|
        | 2021 | 0.0098 | 0.0077 | 0.0072 |      0.0062 |
        | 2022 | 0.014  | 0.0091 | 0.0102 |      0.0105 |
        | 2023 | 0.0112 | 0.0081 | 0.0085 |      0.0063 |
        | 2024 | 0.0109 | 0.0076 | 0.0081 |      0.0058 |
        | 2025 | 0.0124 | 0.0076 | 0.0078 |      0.0092 |
        """
        period = period if period else "daily"
        returns = self._get_measure_returns(period, within_period, rolling, growth)

        result = risk_model.get_partial_moment(
            returns,
            order=2,
            minimum_acceptable_return=minimum_acceptable_return,
            side="upper",
            root=True,
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
    def get_volatility_skewness(
        self,
        period: str | None = None,
        minimum_acceptable_return: float = 0.0,
        method: Literal["volatility", "variability"] = "volatility",
        within_period: bool = False,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Volatility Skewness or Variability Skewness: whether the returns of an investment portfolio
        or asset spread out more on the gain side than on the loss side of a minimum acceptable return.

        The Volatility Skewness (Bacon 2008) divides the upside variance by the downside variance around a
        minimum acceptable return (MAR): above 1 the returns spread out more on the gain side than on the
        loss side, below 1 the volatility comes mostly from losses. The Variability Skewness is its square
        root, the Upside Deviation over the Downside Deviation, on the scale of the returns. Both describe
        the asymmetry of the return distribution without the third moment, which a single extreme return
        can dominate.

        The formula is as follows:

        - Volatility Skewness = Σ max(Return_t - MAR, 0)^2 / Σ max(MAR - Return_t, 0)^2
        - Variability Skewness = √(Volatility Skewness) = Upside Deviation / Downside Deviation

        Where the sums run over all returns of the sample, so it is the upside variance over the downside
        variance, both divided by the number of all returns. A sample without a return below the MAR gives
        NaN.

        For more information, see the following sources:

        - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution." John Wiley &
        Sons, chapter 4.

        Also known as: variability skewness, upside to downside variance ratio.

        Shares similar logic:
        - PerformanceAnalytics (R), VolatilitySkewness: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/VolatilitySkewness.R#L41

        Args:
            period (str, optional): The data frequency for returns. Options are: "daily", "weekly", "monthly",
                "quarterly", "yearly". Defaults to "daily".
            minimum_acceptable_return (float, optional): The minimum acceptable return (MAR) per
            observation of the chosen `period` (e.g. per day). Defaults to 0.0.
            method (str, optional): The ratio. Options are: "volatility" (Volatility Skewness, the ratio of
            the variances), "variability" (Variability Skewness, the ratio of the deviations). Defaults to
            "volatility".
            within_period (bool, optional): Whether to calculate the Volatility Skewness within the specified period or
            for the entire period. Thus whether to look at the Volatility Skewness within a specific year (if period =
            'yearly') or look at the entirety of all years. Defaults to False.
            Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rolling (int, optional): The rolling window size to use for the calculation. If set, the Volatility Skewness
            is calculated over a rolling window of this many periods across the full return history
            instead of per `period`. At least 2. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Volatility Skewness values over time.
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
            pd.Series | pd.DataFrame: Volatility Skewness values, per ticker, or per period (or date) and ticker.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates the
        Volatility Skewness for each asset in the Toolkit instance, each on its own returns (a ticker that listed later
        or misses a date another ticker traded is measured on the returns it has).
        - With growth, the method calculates the growth of the Volatility Skewness values using the
        specified `lag`.
        - The result agrees with the volatility and variability statistics of PerformanceAnalytics'
        VolatilitySkewness.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["SPY", "QQQ", "IWM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_volatility_skewness(period="yearly", within_period=True)
        ```

        Which returns:

        |      |    QQQ |    IWM |   Benchmark |
        |:-----|-------:|-------:|------------:|
        | 2021 | 1.1361 | 1.1843 |      1.2921 |
        | 2022 | 0.8549 | 0.9106 |      0.9091 |
        | 2023 | 1.7396 | 1.3442 |      1.4283 |
        | 2024 | 1.0406 | 1.1335 |      1.1207 |
        | 2025 | 1.2522 | 1.1954 |      1.289  |
        """
        period = period if period else "daily"
        returns = self._get_measure_returns(period, within_period, rolling, growth)

        result = risk_model.get_volatility_skewness(
            returns,
            minimum_acceptable_return=minimum_acceptable_return,
            method=method,
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
    def get_ewma_volatility(
        self,
        lambda_: float = 0.94,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the exponentially weighted moving average (EWMA) Volatility: how volatile the daily returns of
        an investment portfolio or asset are now, weighing recent returns most, following RiskMetrics.

        Unlike a fixed-window rolling Volatility, EWMA Volatility weights recent
        observations more heavily than older ones, so it reacts faster to changes in the
        underlying volatility regime. It is a simpler, more interpretable alternative to a
        full GARCH fit.

        The formula is as follows:

        - EWMA Variance(t) = lambda * EWMA Variance(t-1) + (1 - lambda) * Return(t-1) ** 2
        - EWMA Volatility(t) = √(EWMA Variance(t))

        Where lambda is 0.94 by default and Return(t-1) the previous daily return, so the value on a day only
        uses the returns before it. The recursion starts from the first squared return.

        With the RiskMetrics decay factor of 0.94, about 70% of the weight sits on the last 20 trading
        days, so EWMA Volatility reacts within days to a market shock and falls back almost as quickly.
        The values are daily; multiply them by the square root of 252 (365 for a ticker that trades on
        weekends) for an annualized figure comparable with the yearly `get_volatility`.

        Also known as: RiskMetrics volatility, exponentially weighted volatility.

        Args:
            lambda_ (float, optional): The decay factor. Higher values weight the past
            more heavily (slower to react), lower values weight recent returns more
            heavily (faster to react). RiskMetrics uses 0.94 for daily data. Defaults to 0.94.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the EWMA Volatility values over
            time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: Daily EWMA Volatility values with time as the index.

        Notes:
        - The method retrieves the daily historical return data and calculates the EWMA Volatility for
        each asset in the Toolkit instance.
        - With growth, the method calculates the growth of the EWMA Volatility values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["USDJPY", "EURUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_ewma_volatility().tail(5)
        ```

        Which returns:

        | Date       |   USDJPY |   EURUSD |   Benchmark |
        |:-----------|---------:|---------:|------------:|
        | 2025-12-26 |   0.0046 |   0.0046 |      0.0057 |
        | 2025-12-28 |   0.0045 |   0.0044 |      0.0055 |
        | 2025-12-29 |   0.0043 |   0.0043 |      0.0054 |
        | 2025-12-30 |   0.0042 |   0.0042 |      0.0052 |
        | 2025-12-31 |   0.0041 |   0.0045 |      0.0051 |
        """
        returns = self._historical_data["daily"]["Return"]
        ewma_volatility = risk_model.get_ewma_volatility(returns, lambda_)

        ewma_volatility = ewma_volatility.loc[self._start_date : self._end_date]
        ewma_volatility = ewma_volatility.dropna(how="all", axis=0)

        return finalize_dataset(
            dataset=ewma_volatility,
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
