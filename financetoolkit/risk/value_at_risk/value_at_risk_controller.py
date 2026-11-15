"""Value at Risk Module"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.helpers import handle_portfolio
from financetoolkit.risk.base.base_controller import RiskBase
from financetoolkit.risk.helpers import as_float
from financetoolkit.risk.value_at_risk import (
    backtesting_model,
    cvar_model,
    evar_model,
    var_model,
)
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import (
    apply_rounding,
    finalize_dataset,
)

# ruff: noqa: E501


def _get_out_of_sample_estimates(
    returns: pd.DataFrame,
    measure: str,
    distribution: str,
    alpha: float,
    window_size: int,
) -> pd.DataFrame:
    """
    Builds the rolling, out-of-sample Value at Risk or Conditional Value at Risk path
    of every ticker, as used by the VaR and Expected Shortfall backtests.

    Each ticker is rolled over its own non-missing returns, so a ticker with a shorter
    history (or a missing day) does not shorten the sample of the others, and the window
    always holds `window_size` actual observations. The path is shifted by one
    observation so that every estimate only uses the returns before the one it is
    compared against.

    Args:
        returns (pd.DataFrame): The returns, one column per ticker.
        measure (str): Either "var" or "cvar".
        distribution (str): One of "historic", "gaussian", "studentt" or "evt".
        alpha (float): The tail probability (e.g., 0.05 for the 95% confidence level).
        window_size (int): The number of observations in each rolling window.

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

    estimates = {}

    for column in returns.columns:
        column_returns = returns[column].dropna()

        if distribution == "historic":
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

        # A trailing rolling window includes the current observation, so without the shift each estimate would be compared against a return it was computed from.  # noqa: E501
        estimates[column] = estimate.shift(1).reindex(returns.index)

    return pd.DataFrame(estimates, index=returns.index, columns=returns.columns)


@filter_shown_columns
@track_sources
class ValueAtRiskRisk(RiskBase):
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

        Where z is the standard normal quantile at alpha, Std the sample standard deviation (ddof=1), S the
        sample skewness and K the sample excess kurtosis. The Student-T degrees of freedom v, Location and Scale
        are fitted by maximum likelihood. For "evt", u is the loss at `threshold_percentile`, xi and Scale_GPD the
        shape and scale of the Generalized Pareto Distribution fitted to the losses above it and p the fraction of
        losses above it.

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
            `distribution="historic"`; see `get_var_backtest` for a rolling, out-of-sample VaR path
            under the parametric distributions. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the VaR values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            distribution (str): The distribution to use for the VaR calculations (historic, gaussian,
            cornish-fisher, studentt or evt). Defaults to "historic". The parametric distributions use the
            sample standard deviation; "cornish-fisher" adjusts the gaussian quantile for the skewness and
            excess kurtosis of the returns (see `var_model.get_var_cornish_fisher`) and "studentt" uses the
            quantile of a Student-T distribution fitted by maximum likelihood.
            threshold_percentile (float, optional): Only used when `distribution` is "evt". The percentile
            of losses above which the Generalized Pareto Distribution is fitted; `alpha` can be at most
            1 - `threshold_percentile`, and a larger `alpha` gives an empty result (with a logged error).
            Defaults to 0.95.
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
        | 2021 |  -0.0626 |  -0.0838 | -0.0087 |     -0.0089 |
        | 2022 |  -0.0574 |  -0.0742 | -0.02   |     -0.0203 |
        | 2023 |  -0.0313 |  -0.0374 | -0.0116 |     -0.0117 |
        | 2024 |  -0.0407 |  -0.0533 | -0.0094 |     -0.0096 |
        | 2025 |  -0.0358 |  -0.0569 | -0.0113 |     -0.0113 |
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

        if rolling and distribution != "historic":
            raise ValueError(
                "The rolling parameter is only available for distribution='historic'. "
                "Use get_var_backtest to obtain a rolling, out-of-sample VaR path "
                "under one of the parametric distributions instead."
            )

        if rolling:
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
            else:
                raise ValueError(
                    "Distribution must be historic, gaussian, cornish-fisher, studentt "
                    "or evt."
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
        alpha: float = 0.05,
        within_period: bool = False,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        distribution: str = "historic",
        threshold_percentile: float = 0.95,
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

        Where the Student-T v, Location and Scale are fitted by maximum likelihood (NaN for v <= 1), z_cf is the
        Cornish-Fisher quantile of `get_value_at_risk`, and u, xi, Scale_GPD and VaR_loss are the loss threshold,
        the fitted Generalized Pareto shape and scale and the EVT VaR as a positive loss (NaN for xi >= 1).

        Also known as: CVaR, expected shortfall, ES, tail risk.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            alpha (float, optional): The tail probability for the CVaR calculation (e.g., 0.05 for the 95%
            confidence level). Defaults to 0.05.
            within_period (bool, optional): Whether to calculate CVaR within the specified period or for the entire
            period. Thus whether to look at the CVaR within a specific year (if period = 'yearly') or look at the entirety
            of all years. Defaults to False. Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rolling (int, optional): The rolling window size to use for the calculation. If set, CVaR is
            calculated over a rolling window of this many periods across the full return history instead
            of per `period` (e.g. a rolling 60-day CVaR). Only available for
            `distribution="historic"`; see `get_acerbi_szekely_test` for a rolling, out-of-sample CVaR
            path under the parametric distributions. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the CVaR values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            distribution (str): The distribution to use for the CVaR calculations (historic, gaussian, studentt, laplace,
            logistic, cornish-fisher or evt). Defaults to "historic".
            threshold_percentile (float, optional): Only used when `distribution` is "evt". The percentile
            of losses above which the Generalized Pareto Distribution is fitted; `alpha` can be at most
            1 - `threshold_percentile`, and a larger `alpha` gives an empty result (with a logged error).
            Defaults to 0.95.
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

        if rolling and distribution != "historic":
            raise ValueError(
                "The rolling parameter is only available for distribution='historic'. "
                "Use get_acerbi_szekely_test to obtain a rolling, out-of-sample CVaR "
                "path under one of the parametric distributions instead."
            )

        if rolling:
            returns = self._historical_data[period]["Return"]
            conditional_value_at_risk = cvar_model.get_rolling_cvar_historic(
                returns, alpha, rolling
            )
        else:
            returns = self._get_returns(period, within_period)

            if distribution == "historic":
                conditional_value_at_risk = cvar_model.get_cvar_historic(returns, alpha)
            elif distribution == "gaussian":
                conditional_value_at_risk = cvar_model.get_cvar_gaussian(returns, alpha)
            elif distribution == "studentt":
                conditional_value_at_risk = cvar_model.get_cvar_studentt(returns, alpha)
            elif distribution == "laplace":
                conditional_value_at_risk = cvar_model.get_cvar_laplace(returns, alpha)
            elif distribution == "logistic":
                conditional_value_at_risk = cvar_model.get_cvar_logistic(returns, alpha)
            elif distribution == "cornish-fisher":
                conditional_value_at_risk = cvar_model.get_cvar_cornish_fisher(
                    returns, alpha
                )
            elif distribution == "evt":
                conditional_value_at_risk = cvar_model.get_cvar_evt(
                    returns, alpha, threshold_percentile
                )
            else:
                raise ValueError(
                    "Distribution must be historic, gaussian, studentt, laplace, "
                    "logistic, cornish-fisher or evt."
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
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Entropic Value at Risk (EVaR) of an investment portfolio or asset's returns.

        Entropic Value at Risk (EVaR) is a risk management metric that quantifies upper bound for the value
        at risk (VaR) and the conditional value at risk (CVaR) over a specified time horizon and confidence
        level. EVaR is obtained from the Chernoff inequality. It provides insights into the downside risk
        associated with an investment and helps investors make informed decisions about risk tolerance.

        The EVaR is calculated under the assumption that returns are normally distributed, for which the
        Chernoff bound has a closed form.

        The formula is as follows (with the default alpha=0.05):

        - EVaR = Mean - Std * √(-2 * ln(alpha))

        Where alpha is the tail probability (0.05 for the 95% confidence level) and Std the sample standard
        deviation (ddof=1). It is an upper bound of the gaussian VaR and CVaR at the same confidence level.
        Since it only uses the mean and the standard deviation of the returns, it does not reflect their
        skewness or fat tails, and at high confidence levels it can be less severe than the historical CVaR
        of an asset with heavy tails.

        For more information about the method, see the following paper:

        - Ahmadi-Javid, A. (2012). "Entropic Value-at-Risk: A New Coherent Risk Measure." Journal of
        Optimization Theory and Applications, 155(3), 1105-1123.

        Also known as: EVaR.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            alpha (float, optional): The tail probability for the EVaR calculation (e.g., 0.05 for the 95%
            confidence level). Defaults to 0.05.
            within_period (bool, optional): Whether to calculate EVaR within the specified period or for the entire
            period. Thus whether to look at the CVaR within a specific year (if period = 'yearly') or look at the entirety
            of all years. Defaults to False. Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the CVaR values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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
        if growth and not within_period:
            raise ValueError(
                "Growth calculations need one value per period, while "
                "within_period=False produces a single value for the entire period. "
                "Set within_period=True to calculate growth."
            )

        returns = self._get_returns(period, within_period)

        entropic_value_at_risk = evar_model.get_evar_gaussian(returns, alpha)

        if within_period:
            entropic_value_at_risk = entropic_value_at_risk.loc[
                self._start_date : self._end_date
            ]

        return finalize_dataset(
            dataset=entropic_value_at_risk,
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
        (ddof=1) on the dates all assets have a return, and Portfolio VaR is the VaR of the portfolio return
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

        # The portfolio return needs every constituent on the same dates, so only here are the rows with any missing return dropped.  # noqa: E501
        marginal_var = var_model.get_marginal_var(
            returns[weights_series.index].dropna(),
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

        # The portfolio return needs every constituent on the same dates, so only here are the rows with any missing return dropped.  # noqa: E501
        component_var = var_model.get_component_var(
            returns[weights_series.index].dropna(),
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
        window_size: int = 252,
        test: str = "both",
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Backtest a Value at Risk model against realized returns.

        A Value at Risk estimate is only useful if it is actually well-calibrated
        against reality. This method builds a rolling, out-of-sample VaR path (each
        VaR estimate uses only the `window_size` returns preceding it, never the
        return it is compared against) and tests it with Kupiec's Proportion of
        Failures test (is the overall breach rate consistent with `alpha`?) and/or
        Christoffersen's independence test (are breaches spread out over time, or do
        they cluster together?).

        Regulators backtest VaR on the number of exceptions over the last 250 trading days
        (the Basel traffic-light approach at 99% confidence), and Kupiec's test formalises
        the same question. With few exceptions the test has little power, so a model that
        passes is not necessarily well calibrated. Clustered exceptions, which
        Christoffersen's test detects, are the usual failure of historical VaR when
        volatility rises.

        The formula is as follows (with the default distribution="historic", alpha=0.05 and window_size=252):

        - VaR_t = Quantile(Return_(t-252), ..., Return_(t-1); alpha)
        - Breach_t = 1 if Return_t < VaR_t, else 0
        - Kupiec Statistic = -2 * ln[(1 - alpha)^(n - x) * alpha^x / ((1 - x / n)^(n - x) * (x / n)^x)]
        - Christoffersen Statistic = -2 * ln[L(pi) / L(pi_01, pi_11)]

        Where n is the number of out-of-sample VaR estimates and x the number of breaches. L(pi) is the
        likelihood of the breach sequence with a single breach probability pi, and L(pi_01, pi_11) that of a
        first-order Markov chain in which the breach probability depends on whether the previous period was a
        breach. Both statistics are compared with a χ² distribution with one degree of freedom for the p-value.

        For more information about the methods, see the following papers:

        - Kupiec, P.H. (1995). "Techniques for Verifying the Accuracy of Risk
        Measurement Models." The Journal of Derivatives, 3(2), 73-84.
        - Christoffersen, P.F. (1998). "Evaluating Interval Forecasts." International
        Economic Review, 39(4), 841-862.

        Also known as: VaR validation, VaR backtest, POF test.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly or yearly).
            Defaults to "daily", since `window_size` is expressed in return observations of this frequency
            (252 only means "about one year" when `period` is daily).
            distribution (str, optional): The distribution to use for the rolling VaR estimates, one of
            "historic", "gaussian", "studentt" or "evt". Defaults to "historic".
            "studentt" and "evt" fit their distribution to every window, which takes minutes for
            years of daily data, where "historic" and "gaussian" take well under a second.
            alpha (float, optional): The tail probability for the VaR estimates (e.g., 0.05 for the 95%
            confidence level). Defaults to 0.05. With "evt" it can be at most 0.05, since the Generalized
            Pareto Distribution is fitted above the 95th percentile of losses; a larger `alpha` gives an
            empty result (with a logged error).
            window_size (int, optional): The rolling window size (in number of return observations) used to
            estimate each VaR value. Defaults to 252 (approximately one trading year of daily returns).
            test (str, optional): Which test(s) to run, one of "kupiec", "christoffersen" or "both". Defaults
            to "both".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The requested test statistic(s) and their p-value(s) per asset.

        Notes:
        - The rolling VaR path is calculated over the full return history, not the `within_period` slices
        used elsewhere in this module, since a meaningful rolling window generally needs more history than
        a single sub-period provides. Each asset is rolled over its own available returns, so a shorter
        history of one asset does not shorten the others.

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

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if distribution not in ["historic", "gaussian", "studentt", "evt"]:
            raise ValueError(
                "Distribution must be historic, gaussian, studentt, or evt."
            )
        if test not in ["kupiec", "christoffersen", "both"]:
            raise ValueError("Test must be kupiec, christoffersen, or both.")

        returns = self._historical_data[period]["Return"].dropna(how="all")
        rolling_var = _get_out_of_sample_estimates(
            returns, "var", distribution, alpha, window_size
        )

        results = []
        if test in ("kupiec", "both"):
            results.append(
                backtesting_model.get_kupiec_test(returns, rolling_var, alpha)
            )
        if test in ("christoffersen", "both"):
            results.append(
                backtesting_model.get_christoffersen_test(returns, rolling_var)
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
        window_size: int = 252,
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

        The formula is as follows (with the default distribution="historic", alpha=0.05 and window_size=252):

        - Z2 = Σ (Return_t * Breach_t / CVaR_t) / (n * alpha) - 1
        - P-Value = 1 - Φ(Z2 / Standard Error)

        Where VaR_t and CVaR_t are estimated from the 252 returns before t (see `get_var_backtest`), Breach_t is
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
            Defaults to "daily", since `window_size` is expressed in return observations of this frequency.
            distribution (str, optional): The distribution to use for the rolling VaR/CVaR estimates,
            one of "historic", "gaussian", "studentt" or "evt". Defaults to "historic".
            "studentt" and "evt" fit their distribution to every window, which takes minutes for
            years of daily data, where "historic" and "gaussian" take well under a second.
            alpha (float, optional): The tail probability for the VaR/CVaR estimates (e.g., 0.05 for the 95%
            confidence level). Defaults to 0.05. With "evt" it can be at most 0.05, since the Generalized
            Pareto Distribution is fitted above the 95th percentile of losses; a larger `alpha` gives an
            empty result (with a logged error).
            window_size (int, optional): The rolling window size (in number of return observations) used
            to estimate each VaR/CVaR value. Defaults to 252 (approximately one trading year of daily
            returns).
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
        if distribution not in ["historic", "gaussian", "studentt", "evt"]:
            raise ValueError(
                "Distribution must be historic, gaussian, studentt, or evt."
            )

        returns = self._historical_data[period]["Return"].dropna(how="all")
        rolling_var = _get_out_of_sample_estimates(
            returns, "var", distribution, alpha, window_size
        )
        rolling_cvar = _get_out_of_sample_estimates(
            returns, "cvar", distribution, alpha, window_size
        )

        result = backtesting_model.get_acerbi_szekely_test(
            returns, rolling_var, rolling_cvar, alpha, n_bootstrap, random_state
        )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )
