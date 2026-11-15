"""Volatility Risk Module"""

__docformat__ = "google"

import numpy as np
import pandas as pd

from financetoolkit.helpers import handle_portfolio
from financetoolkit.risk import risk_model
from financetoolkit.risk.base.base_controller import RiskBase
from financetoolkit.risk.volatility import garch_model, realized_volatility_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import (
    apply_rounding,
    finalize_dataset,
)

# ruff: noqa: E501


@filter_shown_columns
@track_sources
class VolatilityRisk(RiskBase):
    """
    Volatility: GARCH, GJR-GARCH and EGARCH fits and forecasts, realized and EWMA volatility, the HAR-RV forecast, excess volatility and the downside deviation.
    """

    @handle_portfolio
    @handle_errors
    def get_garch(
        self,
        period: str | None = None,
        time_steps: int | None = None,
        optimization_t: int | None = None,
        within_period: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculates the conditional variance (sigma_t^2) of the returns with a GARCH(1, 1) model fitted to
        them. Take the square root for the conditional volatility, and see `get_garch_forecast` for the
        variance forecast beyond the last observation.

        GARCH (Generalized autoregressive conditional heteroskedasticity) is stochastic model for time series, which is
        for instance used to model volatility clusters, stock return and inflation. It is a generalisation of the ARCH
        models.

        The formula is as follows:

        - sigma_t^2 = Omega + Alpha * r_(t-1)^2 + Beta * sigma_(t-1)^2
        - sigma_1^2 = Omega + (Alpha + Beta) * Backcast

        Where r_t are the returns (not demeaned) and the Backcast is an exponentially weighted average (decay
        0.94) of the first 75 squared returns. Omega, Alpha and Beta maximize the Gaussian log-likelihood, i.e.
        minimize Σ (ln sigma_t^2 + r_t^2 / sigma_t^2), under Omega, Alpha, Beta > 0 and Alpha + Beta < 1 (see
        `get_garch_parameters`).

        Also known as: GARCH, volatility clustering, conditional heteroscedasticity.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            time_steps (int, optional): Time steps to calculate GARCH for. Defaults to None, which uses the length of the
                returns.
            optimization_t (int, optional): Time steps to optimize GARCH for. It is only used if no weights are given.
                Defaults to None, which uses the length of the returns.
            within_period (bool, optional): Whether to calculate GARCH within the specified period or for the entire
            period. Thus whether to look at the GARCH within a specific year (if period = 'yearly') or look at the
            entirety of all years. Defaults to False.
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
        - If `growth` is set to True, the method calculates the growth of GARCH values using the specified `lag`.

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
        | 2025-12-27 |   0.0004 |           0 |
        | 2025-12-28 |   0.0004 |           0 |
        | 2025-12-29 |   0.0004 |           0 |
        | 2025-12-30 |   0.0004 |           0 |
        | 2025-12-31 |   0.0004 |           0 |
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

        # Only rows without any return are dropped: every column is fitted on its own non-missing returns, so a ticker that listed later (or has a gap) does not cut the others down to its history.  # noqa: E501
        returns = self._get_returns(period, within_period).dropna(how="all")

        garch_sigma_2 = garch_model.get_garch(
            returns=returns,
            weights=None,
            time_steps=time_steps,
            optimization_t=optimization_t,
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
        time_steps: int = 10,
        within_period: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculates sigma_2 forecasts.

        GARCH (Generalized autoregressive conditional heteroskedasticity) is stochastic model for time series, which is
        for instance used to model volatility clusters, stock return and inflation. It is a generalisation of the ARCH
        models.

        The formula is as follows:

        - sigma_(n+h)^2 = sigma_L^2 + (sigma_(n+1)^2 - sigma_L^2) * (Alpha + Beta)^(h - 1)
        - sigma_L^2 = Omega / (1 - Alpha - Beta)

        Where n is the last observation, h = 1, ..., `time_steps` (10 by default) the forecast step,
        sigma_(n+1)^2 the one-step-ahead variance of the fitted GARCH(1, 1) recursion (see `get_garch`) and
        sigma_L^2 the long-run variance.

        For more information about the method, see the following book:

        - Finance Compact Plus Band 1, by Yvonne Seler Zimmerman and Heinz Zimmerman; ISBN: 978-3-907291-31-1

        Also known as: volatility forecast, predicted volatility.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            time_steps (int, optional): Time steps to calculate GARCH and to forecast sigma_2 values for. Defaults to 10.
            within_period (bool, optional): Whether to calculate GARCH within each specified period or all
            at once. Thus whether to look at the GARCH within each specific year (if period = 'yearly') or
            look at the entirety of all years. Defaults to False.
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
        - If `growth` is set to True, the method calculates the growth of the forecasted sigma_2 values using
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

        # Only rows without any return are dropped: every column is fitted on its own non-missing returns, so a ticker that listed later (or has a gap) does not cut the others down to its history.  # noqa: E501
        returns = self._get_returns(period, within_period).dropna(how="all")

        sigma_2_forecast = garch_model.get_garch_forecast(returns, None, time_steps)

        # The forecast steps continue the period of the returns (business days for daily).
        period_symbol = {
            "daily": "B",
            "weekly": "W",
            "monthly": "ME",
            "quarterly": "QE",
        }.get(period, "YE")
        # Business days are kept as daily periods, as business-day periods are deprecated.
        period_index = pd.PeriodIndex(
            pd.date_range(
                start=returns.index[-1].to_timestamp(),
                periods=time_steps + 1,
                freq=period_symbol,
            ),
            freq="D" if period_symbol == "B" else None,
        )

        sigma_2_forecast.index = period_index[1:]

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
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Fits a GARCH(1, 1) model to the historical returns and returns the estimated
        Omega, Alpha and Beta parameters for each asset.

        GARCH (Generalized autoregressive conditional heteroskedasticity) is a stochastic model
        for time series, used to model volatility clustering.

        The formula is as follows:

        - sigma_t^2 = Omega + Alpha * r_(t-1)^2 + Beta * sigma_(t-1)^2
        - Log-Likelihood = -Σ (ln sigma_t^2 + r_t^2 / sigma_t^2) (up to constants)

        Where r_t are the returns (not demeaned), with the constraints Omega, Alpha, Beta > 0 and
        Alpha + Beta < 1. The parameters are estimated by maximizing the Gaussian log-likelihood with
        L-BFGS-B, started from a
        domain-informed initial guess and from 32 points of a Sobol sequence, and the best fit is
        finished with Nelder-Mead. The search runs on the parameters scaled to their bounds, as
        Omega (of the order of a daily variance) and Alpha and Beta differ by orders of magnitude.

        The variance recursion starts from a backcast, an exponentially weighted average
        (decay 0.94) of the first 75 squared returns, which stands in for both the squared
        return and the variance before the first observation, and every observation enters the
        likelihood. This is the start the arch package uses, so the estimates agree with it.

        Unlike `get_garch` and `get_garch_forecast`, which return a (forecasted) conditional variance
        path, this method returns the fitted parameters themselves. This is useful when the parameters
        are needed directly, for example to seed a separate volatility simulation.

        For daily equity returns, Alpha is typically between 0.05 and 0.15 and Alpha + Beta (the
        persistence) between 0.95 and 0.99, meaning a volatility shock fades over weeks rather than
        days. The long-run variance the model reverts to is Omega / (1 - Alpha - Beta). A fit far
        outside these ranges is a reason to check the data and the window before using the parameters.

        For more information about the method, see the following book:

        - Finance Compact Plus Band 1, by Yvonne Seler Zimmerman and Heinz Zimmerman; ISBN: 978-3-907291-31-1

        Also known as: GARCH weights, GARCH coefficients, conditional variance parameters.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            optimization_t (int, optional): Time steps of the returns series to use for the optimization.
            Defaults to the full length of the returns series.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Omega, Alpha and Beta values per asset.

        Notes:
        - The method retrieves historical return data based on the specified `period` and fits a
        GARCH(1, 1) model for each asset in the Toolkit instance.
        - Every asset is fitted on its own available returns, so an asset with a shorter history (or a gap)
        does not shorten the sample of the others.

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
        | Alpha | 0.035763 | 0.057692 | 0.048602 |    0.154079 |
        | Beta  | 0.875036 | 0.715864 | 0.940197 |    0.811018 |
        """
        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )

        returns = self._historical_data[period]["Return"]

        parameters = pd.DataFrame(
            {
                # Each ticker is fitted on its own non-missing returns, see get_garch.
                ticker: garch_model.get_garch_weights(
                    returns[ticker].dropna().to_numpy(), t=optimization_t
                )
                for ticker in returns.columns
            },
            index=["Omega", "Alpha", "Beta"],
        )

        return apply_rounding(
            parameters, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_gjr_garch(
        self,
        period: str | None = None,
        time_steps: int | None = None,
        optimization_t: int | None = None,
        within_period: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculates the conditional variance (sigma_t^2) of the returns with a GJR-GARCH(1, 1, 1) model
        fitted to them. Take the square root for the conditional volatility, and see
        `get_gjr_garch_forecast` for the variance forecast.

        GJR-GARCH extends GARCH with a leverage term that lets negative shocks (bad
        news) raise volatility by more than positive shocks of the same size, a well
        documented asymmetry in equity returns that symmetric GARCH cannot capture.

        For more information about the method, see the following paper:

        - Glosten, L.R., Jagannathan, R., and Runkle, D.E. (1993). "On the Relation
        between the Expected Value and the Volatility of the Nominal Excess Return on
        Stocks." The Journal of Finance, 48(5), 1779-1801.

        The formula is as follows:

        - sigma_t^2 = Omega + (Alpha + Gamma * I_(t-1)) * r_(t-1)^2 + Beta * sigma_(t-1)^2
        - sigma_1^2 = Omega + (Alpha + Gamma / 2 + Beta) * Backcast

        Where r_t are the returns (not demeaned), I_(t-1) is 1 when r_(t-1) < 0 and 0 otherwise, and the Backcast
        is the same as in `get_garch`. The parameters maximize the Gaussian log-likelihood under
        Alpha + Gamma / 2 + Beta < 1 and Alpha + Gamma >= 0 (see `get_gjr_garch_parameters`).

        Also known as: GJR-GARCH, threshold GARCH, TGARCH.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            time_steps (int, optional): Time steps to calculate GJR-GARCH for. Defaults to None, which uses the length of
                the returns.
            optimization_t (int, optional): Time steps to optimize GJR-GARCH for. It is only used if no
            weights are given. Defaults to None, which uses the length of the returns.
            within_period (bool, optional): Whether to calculate GJR-GARCH within the specified period or
            for the entire period. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the GJR-GARCH values over time.
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
            pd.DataFrame | pd.Series: GJR-GARCH values

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates
        GJR-GARCH for each asset in the Toolkit instance.
        - Every asset is fitted on its own available returns, so an asset with a shorter history (or a gap)
        does not shorten the sample of the others; dates without a return for an asset are NaN.
        - If `growth` is set to True, the method calculates the growth of GJR-GARCH values using the
        specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NVDA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_gjr_garch().tail(5)
        ```

        Which returns:

        | Date       |   NVDA |   Benchmark |
        |:-----------|-------:|------------:|
        | 2025-12-24 | 0.0005 |      0.0001 |
        | 2025-12-26 | 0.0005 |      0.0001 |
        | 2025-12-29 | 0.0005 |      0      |
        | 2025-12-30 | 0.0005 |      0      |
        | 2025-12-31 | 0.0005 |      0      |
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

        # Only rows without any return are dropped: every column is fitted on its own non-missing returns, so a ticker that listed later (or has a gap) does not cut the others down to its history.  # noqa: E501
        returns = self._get_returns(period, within_period).dropna(how="all")

        gjr_garch_sigma_2 = garch_model.get_gjr_garch(
            returns=returns,
            weights=None,
            time_steps=time_steps,
            optimization_t=optimization_t,
        )

        gjr_garch_sigma_2 = gjr_garch_sigma_2.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=gjr_garch_sigma_2,
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
    def get_gjr_garch_forecast(
        self,
        period: str | None = None,
        time_steps: int = 10,
        within_period: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculates sigma_2 forecasts based on the GJR-GARCH model.

        The formula is as follows:

        - sigma_(n+h)^2 = sigma_L^2 + (sigma_(n+1)^2 - sigma_L^2) * (Alpha + Gamma / 2 + Beta)^(h - 1)
        - sigma_L^2 = Omega / (1 - Alpha - Gamma / 2 - Beta)

        Where n is the last observation, h = 1, ..., `time_steps` (10 by default) the forecast step and
        sigma_(n+1)^2 the one-step-ahead variance of the fitted GJR-GARCH(1, 1, 1) recursion (see
        `get_gjr_garch`). The leverage term enters at its expectation, Gamma / 2, as a negative shock is as
        likely as a positive one.

        For more information about the method, see the following paper:

        - Glosten, L.R., Jagannathan, R., and Runkle, D.E. (1993). "On the Relation
        between the Expected Value and the Volatility of the Nominal Excess Return on
        Stocks." The Journal of Finance, 48(5), 1779-1801.

        Also known as: volatility forecast, predicted volatility.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            time_steps (int, optional): Time steps to calculate GJR-GARCH and to forecast sigma_2 values for. Defaults to
                10.
            within_period (bool, optional): Whether to calculate GJR-GARCH within each specified period or
            all at once. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            growth (bool, optional): Whether to calculate the growth of the GJR-GARCH values over time.
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
            pd.DataFrame | pd.Series: sigma_2 forecast values

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates the
        sigma_2 forecast for each asset in the Toolkit instance.
        - Every asset is fitted on its own available returns, so an asset with a shorter history (or a gap)
        does not shorten the sample of the others.
        - If `growth` is set to True, the method calculates the growth of the forecasted sigma_2 values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["CLUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_gjr_garch_forecast()
        ```

        Which returns:

        |            |   CLUSD |   Benchmark |
        |:-----------|--------:|------------:|
        | 2026-01-01 |  0.0003 |      0      |
        | 2026-01-02 |  0.0003 |      0      |
        | 2026-01-05 |  0.0003 |      0.0001 |
        | 2026-01-06 |  0.0003 |      0.0001 |
        | 2026-01-07 |  0.0003 |      0.0001 |
        | 2026-01-08 |  0.0003 |      0.0001 |
        | 2026-01-09 |  0.0003 |      0.0001 |
        | 2026-01-12 |  0.0003 |      0.0001 |
        | 2026-01-13 |  0.0003 |      0.0001 |
        | 2026-01-14 |  0.0004 |      0.0001 |
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

        # Only rows without any return are dropped: every column is fitted on its own non-missing returns, so a ticker that listed later (or has a gap) does not cut the others down to its history.  # noqa: E501
        returns = self._get_returns(period, within_period).dropna(how="all")

        sigma_2_forecast = garch_model.get_gjr_garch_forecast(returns, None, time_steps)

        # The forecast steps continue the period of the returns (business days for daily).
        period_symbol = {
            "daily": "B",
            "weekly": "W",
            "monthly": "ME",
            "quarterly": "QE",
        }.get(period, "YE")
        # Business days are kept as daily periods, as business-day periods are deprecated.
        period_index = pd.PeriodIndex(
            pd.date_range(
                start=returns.index[-1].to_timestamp(),
                periods=time_steps + 1,
                freq=period_symbol,
            ),
            freq="D" if period_symbol == "B" else None,
        )

        sigma_2_forecast.index = period_index[1:]

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
    def get_gjr_garch_parameters(
        self,
        period: str | None = None,
        optimization_t: int | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Fits a GJR-GARCH(1, 1, 1) model to the historical returns and returns the
        estimated Omega, Alpha, Gamma and Beta parameters for each asset.

        A positive Gamma indicates the presence of a leverage effect (negative shocks
        raise volatility by more than positive ones of the same size), which symmetric
        GARCH cannot represent.

        The formula is as follows:

        - sigma_t^2 = Omega + (Alpha + Gamma * I_(t-1)) * r_(t-1)^2 + Beta * sigma_(t-1)^2
        - Log-Likelihood = -Σ (ln sigma_t^2 + r_t^2 / sigma_t^2) (up to constants)

        Where r_t are the returns (not demeaned) and I_(t-1) is 1 when r_(t-1) < 0 and 0 otherwise. The
        log-likelihood is maximized under Alpha + Gamma / 2 + Beta < 1 and Alpha + Gamma >= 0.

        The variance recursion starts from a backcast, an exponentially weighted average
        (decay 0.94) of the first 75 squared returns, with the leverage term at its expectation
        (Gamma / 2) for the period before the first observation, and every observation enters
        the likelihood. This is the start the arch package uses, so the estimates agree with it.

        For more information about the method, see the following paper:

        - Glosten, L.R., Jagannathan, R., and Runkle, D.E. (1993). "On the Relation
        between the Expected Value and the Volatility of the Nominal Excess Return on
        Stocks." The Journal of Finance, 48(5), 1779-1801.

        Also known as: GJR-GARCH weights, GJR-GARCH coefficients, leverage parameters.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            optimization_t (int, optional): Time steps of the returns series to use for the optimization.
            Defaults to the full length of the returns series.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Omega, Alpha, Gamma and Beta values per asset.

        Notes:
        - The method retrieves historical return data based on the specified `period` and fits a
        GJR-GARCH(1, 1, 1) model for each asset in the Toolkit instance.
        - Every asset is fitted on its own available returns, so an asset with a shorter history (or a gap)
        does not shorten the sample of the others.
        - GJR-GARCH nests GARCH (gamma = 0), so the fit never ends at a lower log-likelihood than the
        GARCH(1, 1) fit of `get_garch_parameters`: if the search does, the GARCH weights with a Gamma of 0
        are returned.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["^GSPC", "^STOXX50E"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_gjr_garch_parameters(rounding=6)
        ```

        Which returns:

        |       |    ^GSPC |   ^STOXX50E |   Benchmark |
        |:------|---------:|------------:|------------:|
        | Omega | 3e-06    |    1e-05    |    3e-06    |
        | Alpha | 0        |    0        |    0        |
        | Gamma | 0.210531 |    0.308907 |    0.221968 |
        | Beta  | 0.870765 |    0.764269 |    0.865708 |
        """
        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )

        returns = self._historical_data[period]["Return"]

        parameters = pd.DataFrame(
            {
                # Each ticker is fitted on its own non-missing returns, see get_garch.
                ticker: garch_model.get_gjr_garch_weights(
                    returns[ticker].dropna().to_numpy(), t=optimization_t
                )
                for ticker in returns.columns
            },
            index=["Omega", "Alpha", "Gamma", "Beta"],
        )

        return apply_rounding(
            parameters, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_egarch(
        self,
        period: str | None = None,
        time_steps: int | None = None,
        optimization_t: int | None = None,
        within_period: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculates the conditional variance (sigma_t^2) of the returns with an EGARCH(1, 1) model fitted
        to them. Take the square root for the conditional volatility, and see `get_egarch_forecast` for
        the variance forecast.

        EGARCH models the log of the conditional variance, which avoids having to
        constrain the parameters to keep the variance positive and, like GJR-GARCH,
        lets negative and positive shocks of the same size have a different impact on
        volatility (the leverage effect).

        For more information about the method, see the following paper:

        - Nelson, D.B. (1991). "Conditional Heteroskedasticity in Asset Returns: A New
        Approach." Econometrica, 59(2), 347-370.

        The formula is as follows:

        - ln sigma_t^2 = Omega + Alpha * (|z_(t-1)| - √(2 / π)) + Gamma * z_(t-1) + Beta * ln sigma_(t-1)^2
        - z_t = r_t / sigma_t
        - ln sigma_1^2 = Omega + Beta * ln(Backcast)

        Where r_t are the returns (not demeaned), √(2 / π) is the expected |z| of a standard normal shock, the
        log variance is kept within [-20, 20] and the Backcast is the same as in `get_garch`. The parameters
        maximize the Gaussian log-likelihood with |Beta| < 1 (see `get_egarch_parameters`).

        Also known as: exponential GARCH, log-GARCH.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            time_steps (int, optional): Time steps to calculate EGARCH for. Defaults to None, which uses the length of the
                returns.
            optimization_t (int, optional): Time steps to optimize EGARCH for. It is only used if no
            weights are given. Defaults to None, which uses the length of the returns.
            within_period (bool, optional): Whether to calculate EGARCH within the specified period or
            for the entire period. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the EGARCH values over time.
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
            pd.DataFrame | pd.Series: EGARCH values

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates
        EGARCH for each asset in the Toolkit instance.
        - Every asset is fitted on its own available returns, so an asset with a shorter history (or a gap)
        does not shorten the sample of the others; dates without a return for an asset are NaN.
        - If `growth` is set to True, the method calculates the growth of EGARCH values using the
        specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_egarch(period="quarterly")
        ```

        Which returns:

        | Date   |   AMZN |   TSLA |   Benchmark |
        |:-------|-------:|-------:|------------:|
        | 2021Q2 | 0.0229 | 0.0923 |      0.0077 |
        | 2021Q3 | 0.0158 | 0.1671 |      0.0049 |
        | 2021Q4 | 0.0244 | 0.1704 |      0.0131 |
        | 2022Q1 | 0.0411 | 0.0597 |      0.0062 |
        | 2022Q2 | 0.0737 | 0.1035 |      0.0116 |
        | 2022Q3 | 0.0642 | 0.2104 |      0.009  |
        | 2022Q4 | 0.0868 | 0.1936 |      0.0145 |
        | 2023Q1 | 0.0975 | 0.3935 |      0.0105 |
        | 2023Q2 | 0.0675 | 0.0902 |      0.0078 |
        | 2023Q3 | 0.0321 | 0.0321 |      0.0048 |
        | 2023Q4 | 0.0565 | 0.0654 |      0.0108 |
        | 2024Q1 | 0.0348 | 0.133  |      0.0046 |
        | 2024Q2 | 0.0167 | 0.2705 |      0.0017 |
        | 2024Q3 | 0.0149 | 0.3389 |      0.002  |
        | 2024Q4 | 0.0239 | 0.2276 |      0.0015 |
        | 2025Q1 | 0.0094 | 0.0473 |      0.0033 |
        | 2025Q2 | 0.0077 | 0.0964 |      0.0076 |
        | 2025Q3 | 0.0013 | 0.0458 |      0.0033 |
        | 2025Q4 | 0.0025 | 0.0022 |      0.0017 |
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

        # Only rows without any return are dropped: every column is fitted on its own non-missing returns, so a ticker that listed later (or has a gap) does not cut the others down to its history.  # noqa: E501
        returns = self._get_returns(period, within_period).dropna(how="all")

        egarch_sigma_2 = garch_model.get_egarch(
            returns=returns,
            weights=None,
            time_steps=time_steps,
            optimization_t=optimization_t,
        )

        egarch_sigma_2 = egarch_sigma_2.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=egarch_sigma_2,
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
    def get_egarch_forecast(
        self,
        period: str | None = None,
        time_steps: int = 10,
        within_period: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculates sigma_2 forecasts based on the EGARCH model.

        The formula is as follows:

        - ln sigma_(n+h)^2 = L + Beta^(h - 1) * (ln sigma_(n+1)^2 - L)
        - L = Omega / (1 - Beta)
        - sigma_(n+h)^2 = exp(ln sigma_(n+h)^2)

        Where n is the last observation, h = 1, ..., `time_steps` (10 by default) the forecast step and
        sigma_(n+1)^2 the one-step-ahead variance of the fitted EGARCH(1, 1) recursion (see `get_egarch`). The
        shock terms drop out at their expectation of zero, so the forecast is the exponential of the expected
        log variance.

        For more information about the method, see the following paper:

        - Nelson, D.B. (1991). "Conditional Heteroskedasticity in Asset Returns: A New
        Approach." Econometrica, 59(2), 347-370.

        Also known as: volatility forecast, predicted volatility.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            time_steps (int, optional): Time steps to calculate EGARCH and to forecast sigma_2 values for. Defaults to 10.
            within_period (bool, optional): Whether to calculate EGARCH within each specified period or
            all at once. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            growth (bool, optional): Whether to calculate the growth of the EGARCH values over time.
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
            pd.DataFrame | pd.Series: sigma_2 forecast values

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates the
        sigma_2 forecast for each asset in the Toolkit instance.
        - Every asset is fitted on its own available returns, so an asset with a shorter history (or a gap)
        does not shorten the sample of the others.
        - If `growth` is set to True, the method calculates the growth of the forecasted sigma_2 values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_egarch_forecast(period="quarterly")
        ```

        Which returns:

        |        |   AMZN |   TSLA |   Benchmark |
        |:-------|-------:|-------:|------------:|
        | 2026Q1 | 0.0012 | 0.0028 |      0.0034 |
        | 2026Q2 | 0.001  | 0.0026 |      0.0055 |
        | 2026Q3 | 0.0009 | 0.0024 |      0.007  |
        | 2026Q4 | 0.0008 | 0.0022 |      0.0079 |
        | 2027Q1 | 0.0007 | 0.002  |      0.0084 |
        | 2027Q2 | 0.0006 | 0.0018 |      0.0086 |
        | 2027Q3 | 0.0005 | 0.0017 |      0.0087 |
        | 2027Q4 | 0.0004 | 0.0016 |      0.0088 |
        | 2028Q1 | 0.0004 | 0.0014 |      0.0089 |
        | 2028Q2 | 0.0003 | 0.0013 |      0.0089 |
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

        # Only rows without any return are dropped: every column is fitted on its own non-missing returns, so a ticker that listed later (or has a gap) does not cut the others down to its history.  # noqa: E501
        returns = self._get_returns(period, within_period).dropna(how="all")

        sigma_2_forecast = garch_model.get_egarch_forecast(returns, None, time_steps)

        # The forecast steps continue the period of the returns (business days for daily).
        period_symbol = {
            "daily": "B",
            "weekly": "W",
            "monthly": "ME",
            "quarterly": "QE",
        }.get(period, "YE")
        # Business days are kept as daily periods, as business-day periods are deprecated.
        period_index = pd.PeriodIndex(
            pd.date_range(
                start=returns.index[-1].to_timestamp(),
                periods=time_steps + 1,
                freq=period_symbol,
            ),
            freq="D" if period_symbol == "B" else None,
        )

        sigma_2_forecast.index = period_index[1:]

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
    def get_egarch_parameters(
        self,
        period: str | None = None,
        optimization_t: int | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Fits an EGARCH(1, 1) model to the historical returns and returns the estimated
        Omega, Alpha, Gamma and Beta parameters for each asset.

        A negative Gamma indicates the presence of a leverage effect (negative shocks
        raise volatility by more than positive ones of the same size).

        The formula is as follows:

        - ln sigma_t^2 = Omega + Alpha * (|z_(t-1)| - √(2 / π)) + Gamma * z_(t-1) + Beta * ln sigma_(t-1)^2
        - Log-Likelihood = -Σ (ln sigma_t^2 + r_t^2 / sigma_t^2) (up to constants)

        Where r_t are the returns (not demeaned) and z_t = r_t / sigma_t. The log-likelihood is maximized with
        Omega in [-15, 5], Alpha and Gamma in [-1, 1] and |Beta| < 1, starting the recursion from
        ln sigma_1^2 = Omega + Beta * ln(Backcast), with the Backcast of `get_garch_parameters`.

        For more information about the method, see the following paper:

        - Nelson, D.B. (1991). "Conditional Heteroskedasticity in Asset Returns: A New
        Approach." Econometrica, 59(2), 347-370.

        Also known as: EGARCH weights, EGARCH coefficients, leverage parameters.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            optimization_t (int, optional): Time steps of the returns series to use for the optimization.
            Defaults to the full length of the returns series.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Omega, Alpha, Gamma and Beta values per asset.

        Notes:
        - The method retrieves historical return data based on the specified `period` and fits an
        EGARCH(1, 1) model for each asset in the Toolkit instance.
        - Every asset is fitted on its own available returns, so an asset with a shorter history (or a gap)
        does not shorten the sample of the others.
        - EGARCH's log-variance parameterization is less constrained than plain GARCH, which makes it
        prone to unstable, boundary-hugging fits on short return histories, which include quarterly
        data over a few years. Prefer daily or weekly returns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_egarch_parameters(period="daily", rounding=6)
        ```

        Which returns:

        |       |      AMZN |      TSLA |   Benchmark |
        |:------|----------:|----------:|------------:|
        | Omega | -0.382664 | -0.225452 |   -0.309737 |
        | Alpha |  0.193573 |  0.081906 |    0.059855 |
        | Gamma | -0.080739 | -0.03535  |   -0.181222 |
        | Beta  |  0.949074 |  0.964858 |    0.96578  |
        """
        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )

        returns = self._historical_data[period]["Return"]

        parameters = pd.DataFrame(
            {
                # Each ticker is fitted on its own non-missing returns, see get_garch.
                ticker: garch_model.get_egarch_weights(
                    returns[ticker].dropna().to_numpy(), t=optimization_t
                )
                for ticker in returns.columns
            },
            index=["Omega", "Alpha", "Gamma", "Beta"],
        )

        return apply_rounding(
            parameters, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_volatility(
        self,
        period: str | None = None,
        rolling: int | None = None,
        method: str = "close_to_close",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Volatility of an investment portfolio or asset's returns for a
        given period based on the daily historical prices.

        Volatility measures the amount of dispersion or variability in returns. It is the
        square root of the Variance. A higher Volatility indicates greater variability, while
        a lower Volatility suggests that returns are closer to the mean.

        By default this is the close-to-close Volatility, i.e. the standard deviation of the
        daily returns. The `method` parameter selects one of four range-based estimators
        instead, each of which uses more of the day's price action than just the close and is
        therefore more statistically efficient (i.e. needs fewer observations to reach the
        same precision), at the cost of additional assumptions about how prices move:

        - `"parkinson"` — uses the daily trading range (High vs Low) rather than the
        close-to-close return, assuming prices follow a continuous geometric Brownian motion
        with no drift and no overnight jumps.
        - `"garman_klass"` — extends Parkinson by also incorporating the Open and Close, which
        allows it to account for the opening jump and makes it more efficient still (assuming,
        as Parkinson does, no drift and no overnight jumps beyond the modeled open).
        - `"rogers_satchell"` — drift-independent, meaning it remains unbiased even when the
        underlying asset has a non-zero expected return over the period, at the cost of still
        assuming no overnight jumps.
        - `"yang_zhang"` — a weighted combination of the overnight (close-to-open) Variance,
        the open-to-close Variance and the Rogers-Satchell Variance. It is both
        drift-independent and accounts for overnight jumps, which makes it the most
        statistically efficient of the range-based estimators implemented here.

        The formula is as follows (with the default method="close_to_close"):

        - Volatility = Std(Daily Returns within the period) * √(Trading Days per Period)

        Where Std is the sample standard deviation (ddof=1) and the trading days per period are 252 / 52
        (weekly), 252 / 12 (monthly), 252 / 4 (quarterly) or 252 (yearly). The range-based estimators
        take the square root of a daily variance and scale it the same way, with H, L, O and C the daily
        High, Low, Open and Close, C_(-1) the previous Close, Mean the average and Var the sample variance
        (ddof=1) over the days in the period:

        - Parkinson Variance = Mean(ln(H / L)^2) / (4 * ln 2)
        - Garman-Klass Variance = Mean(0.5 * ln(H / L)^2 - (2 * ln 2 - 1) * ln(C / O)^2)
        - Rogers-Satchell Variance = Mean(ln(H / C) * ln(H / O) + ln(L / C) * ln(L / O))
        - Yang-Zhang Variance = Var(ln(O / C_(-1))) + k * Var(ln(C / O)) + (1 - k) * Rogers-Satchell Variance
        - k = 0.34 / (1.34 + (n + 1) / (n - 1)), with n the number of days in the period

        With `rolling`, the result is the sample standard deviation of the last `rolling` returns of the
        chosen period itself, without scaling.

        Also known as: standard deviation of returns. The range-based estimators are also known
        as Parkinson's range-based or high-low Volatility, Garman-Klass range-based Volatility,
        Rogers-Satchell drift-independent Volatility and Yang-Zhang drift-independent
        overnight-aware Volatility.

        For more information about the range-based estimators, see the following papers:

        - Parkinson, M. (1980). "The Extreme Value Method for Estimating the Variance of the
        Rate of Return." Journal of Business, 53(1), 61-65.
        - Garman, M.B., & Klass, M.J. (1980). "On the Estimation of Security Price
        Volatilities from Historical Data." Journal of Business, 53(1), 67-78.
        - Rogers, L.C.G., & Satchell, S.E. (1991). "Estimating Variance from High, Low and
        Close Prices." Annals of Applied Probability, 1(4), 504-512.
        - Yang, D., & Zhang, Q. (2000). "Drift-Independent Volatility Estimation Based on
        High, Low, Open, and Close Prices." Journal of Business, 73(3), 477-491.

        Args:
            period (str, optional): The data frequency for returns (weekly, monthly, quarterly, or yearly). Defaults
                to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
            rolling (int, optional): The rolling window size to use for the calculation. If set,
            Volatility is calculated over a rolling window of this many periods (e.g. period='monthly'
            and rolling=6 gives the rolling 6-month Volatility) instead of one value per `period`.
            Only available for method="close_to_close". Defaults to None.
            method (str, optional): Which Volatility estimator to use, one of "close_to_close",
            "parkinson", "garman_klass", "rogers_satchell" or "yang_zhang", as described above.
            Defaults to "close_to_close".
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
        - If `growth` is set to True, the method calculates the growth of Volatility values using the specified `lag`.

        Passing `method="parkinson"` (or another range-based estimator) uses the daily trading range instead of
        only the closes.

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
        | 2021   |   0.0574 |   0.6758 |  0.1    |  0.1156 |      0.0993 |
        | 2022   |   0.091  |   0.531  |  0.189  |  0.1216 |      0.1897 |
        | 2023   |   0.0764 |   0.3662 |  0.1042 |  0.1088 |      0.104  |
        | 2024   |   0.0665 |   0.4441 |  0.098  |  0.1201 |      0.0975 |
        | 2025   |   0.0831 |   0.3485 |  0.1457 |  0.1554 |      0.152  |
        """
        if method not in (
            "close_to_close",
            "parkinson",
            "garman_klass",
            "rogers_satchell",
            "yang_zhang",
        ):
            raise ValueError(
                "Method must be close_to_close, parkinson, garman_klass, "
                "rogers_satchell, or yang_zhang."
            )

        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError("Period must be weekly, monthly, quarterly, or yearly.")

        if rolling and method != "close_to_close":
            raise ValueError(
                "The rolling parameter is only available for method='close_to_close'."
            )

        if method == "close_to_close":
            if rolling:
                period_returns = self._historical_data[period]["Return"]
                volatility = risk_model.get_rolling_volatility(
                    period_returns, period, rolling
                )
            else:
                returns = self._historical_data["daily"]["Return"]
                volatility = risk_model.get_volatility(returns, period)
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
        Calculate the Corsi (2009) Heterogeneous Autoregressive Realized Volatility
        (HAR-RV) forecast of future daily Realized Variance, per asset.

        Volatility clustering happens across multiple, overlapping time horizons at
        once. HAR-RV captures this cheaply -- without the numerical optimization a
        GARCH-family fit requires (see `get_garch`) -- by regressing future daily
        Realized Variance on trailing daily, weekly and monthly average Realized
        Variance components. The daily Realized Variance itself can be constructed in
        several ways via `estimator`: the simplest is the squared daily return, while
        the OHLC range-based estimators (see `get_volatility`, which exposes the same
        estimators via its own `method` parameter) use the daily
        (pre-period-aggregation) term behind each of those estimators instead,
        which is more statistically efficient since it uses the daily trading range
        rather than only the close-to-close move.

        The formula is as follows (with the default estimator="squared_return", weekly_window=5,
        monthly_window=22 and horizon=1):

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
            estimator (str, optional): How to construct the daily Realized Variance input, one of
            "squared_return", "parkinson", "garman_klass" or "rogers_satchell". Defaults to
            "squared_return".
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
        - If `growth` is set to True, the method calculates the growth of the HAR-RV forecast values
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
        if estimator not in (
            "squared_return",
            "parkinson",
            "garman_klass",
            "rogers_satchell",
        ):
            raise ValueError(
                "estimator must be 'squared_return', 'parkinson', 'garman_klass', "
                "or 'rogers_satchell'."
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
        Calculate the Excess Volatility of an investment portfolio or asset's returns for a
        given period based on the daily historical returns.

        Excess Volatility is the Volatility of the Excess Return, i.e. the daily return minus
        the risk free rate, scaled to the given period in the same way as the Volatility.

        The formula is as follows:

        - Excess Volatility = Std(Daily Return - Daily Risk Free Rate) * √(Trading Days per Period)

        Where Std is the sample standard deviation (ddof=1) over the days within the period, the annualized risk
        free rate is converted to a daily rate, and the trading days per period are 252 / 52 (weekly), 252 / 12
        (monthly), 252 / 4 (quarterly) or 252 (yearly). With `rolling`, the result is the sample standard
        deviation of the last `rolling` excess returns of the chosen period itself, without scaling.

        Also known as: standard deviation of excess returns.

        Args:
            period (str, optional): The data frequency for returns (weekly, monthly, quarterly, or yearly). Defaults
                to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
            rolling (int, optional): The rolling window size to use for the calculation. If set,
            Excess Volatility is calculated over a rolling window of this many periods (e.g.
            period='monthly' and rolling=6 gives the rolling 6-month Excess Volatility) instead of
            one value per `period`. Defaults to None.
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
        - If `growth` is set to True, the method calculates the growth of Excess Volatility values using
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

        if period not in ["weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError("Period must be weekly, monthly, quarterly, or yearly.")

        if rolling:
            period_returns = self._historical_data[period]["Return"]
            risk_free_rate = self._risk_free_rate_data[period]
            excess_volatility = risk_model.get_rolling_excess_volatility(
                period_returns, risk_free_rate, period, rolling
            )
        else:
            returns = self._historical_data["daily"]["Return"]
            risk_free_rate = self._risk_free_rate_data["daily"]

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
        Calculate the Downside Deviation of an investment portfolio or asset's returns.

        The Downside Deviation, also known as semi-deviation, measures the dispersion of the returns
        below a minimum acceptable return (MAR), isolating the volatility of negative outcomes from the
        volatility of the overall return distribution. It underlies risk-adjusted
        return measures such as the Sortino Ratio and the Omega Ratio.

        The formula is as follows (with the default minimum_acceptable_return of 0):

        - Downside Deviation = √(Σ min(Return_t - MAR, 0)^2 / n)

        Where n is the number of all returns, so returns above the MAR count as a shortfall of zero rather than
        being left out.

        Also known as: semi-deviation, downside risk, downside volatility.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            minimum_acceptable_return (float, optional): The minimum acceptable return (MAR) used as the
            threshold below which returns are considered downside. Defaults to 0.0.
            within_period (bool, optional): Whether to calculate the Downside Deviation within the specified
            period or for the entire period. Thus whether to look at the Downside Deviation within a specific
            year (if period = 'yearly') or look at the entirety of all years. Defaults to False.
            Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rolling (int, optional): The rolling window size to use for the calculation. If set, the Downside
            Deviation is calculated over a rolling window of this many periods across the full return history
            instead of per `period`. Defaults to None.
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
        - If `growth` is set to True, the method calculates the growth of the Downside Deviation values using
        the specified `lag`.

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
            downside_deviation = risk_model.get_rolling_downside_deviation(
                returns, rolling, minimum_acceptable_return
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
        Calculate the exponentially weighted moving average (EWMA) Volatility of an
        investment portfolio or asset's daily returns, following the RiskMetrics
        methodology.

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
        The values are daily; multiply them by the square root of 252 for an annualized figure comparable
        with `get_volatility(period="yearly")`.

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
        - If `growth` is set to True, the method calculates the growth of the EWMA Volatility values using
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
