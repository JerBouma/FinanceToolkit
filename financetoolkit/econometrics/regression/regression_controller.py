"""Regression Module"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.econometrics.base.base_controller import (
    BENCHMARK_TICKER,
    EconometricsBase,
)
from financetoolkit.econometrics.regression import (
    fama_macbeth_model,
    panel_data_model,
    regression_model,
)
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import apply_rounding

# ruff: noqa: E501


@filter_shown_columns
@track_sources
class RegressionEconometrics(EconometricsBase):
    """
    Regressions: OLS, WLS, GLS, logistic, probit and quantile regressions, Fama-MacBeth, and fixed and random effects panels with the Hausman test.
    """

    @handle_errors
    def get_ols(
        self,
        dependent_ticker: str | None = None,
        independent_tickers: str | list[str] | None = None,
        include_benchmark: bool = False,
        period: str | None = None,
        column: str = "Return",
        add_constant: bool = True,
        cov_type: str = "nonrobust",
        clusters: pd.Series | None = None,
        maxlags: int | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Fit an Ordinary Least Squares (OLS) regression of `dependent_ticker` on
        `independent_tickers`, using the selected `column` (returns by default) of each ticker.

        OLS estimates the coefficients that minimise the sum of squared residuals, i.e. the
        line (or plane) that best explains the dependent ticker from the independent tickers.

        The formula is as follows:

        - β̂ = (X'X)⁻¹X'y, with y the dependent ticker's `column` (Return by default) and X the
          independent tickers' `column` plus a constant column (add_constant=True)
        - Cov(β̂) = σ̂²·(X'X)⁻¹, with σ̂² = e'e / (n - k) (cov_type="nonrobust"); "HC0" to "HC3",
          "cluster" and "HAC" (Newey-West with `maxlags`) use statsmodels' sandwich estimators
        - t = β̂ⱼ / SE(β̂ⱼ), with p-values from a Student-t distribution with n - k degrees of freedom
        - R² = 1 - e'e / Σ(yₜ - mean(y))²

        Each coefficient is the expected change in `dependent_ticker` for a one-unit change in
        that independent ticker, holding the others fixed, and the intercept is what remains
        when all of them are zero. With returns this is the familiar way to estimate how
        strongly one asset moves with others (e.g. a beta against an index).

        Under the classical assumptions (linearity, no perfect multicollinearity, homoskedastic
        and uncorrelated errors) OLS is the Best Linear Unbiased Estimator (Gauss-Markov). The
        coefficients stay unbiased when the errors are heteroskedastic, but the classical
        standard errors do not: check with `get_breusch_pagan_test` or `get_white_test` and,
        if needed, refit with a robust `cov_type` such as "HC1" (or "HC3" for small samples),
        or "HAC" when the errors are also autocorrelated.

        For more information about the HAC covariance, see the following paper:

        - Newey, W.K. & West, K.D. (1987). "A Simple, Positive Semi-Definite,
        Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." Econometrica,
        55(3), 703-708.

        Also known as: linear regression, least squares regression.

        Args:
            dependent_ticker (str | None, optional): The dependent (predicted) asset.
            Defaults to None, meaning the Toolkit instance's first ticker.
            independent_tickers (str | list[str] | None, optional): The independent
            (predictor) asset(s). Defaults to None, meaning every other ticker in
            the Toolkit instance besides `dependent_ticker`.
            include_benchmark (bool, optional): Whether to include "Benchmark" in the
            default independent ticker(s) (has no effect when independent_tickers is given
            explicitly). Defaults to False.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to regress on. Defaults to "Return".
            add_constant (bool, optional): Whether to include an intercept. Defaults to True.
            cov_type (str, optional): Which covariance estimator to use for the standard errors --
            one of "nonrobust" (classical, assumes homoskedastic errors), "HC0"/"HC1"/"HC2"/"HC3"
            (heteroskedasticity-robust), "cluster" (cluster-robust, requires `clusters`) or "HAC"
            (Newey-West, heteroskedasticity-and-autocorrelation-consistent, uses `maxlags`).
            Use `get_breusch_pagan_test`/`get_white_test` to check for heteroskedasticity and
            `get_ljung_box_test` to check for autocorrelation first. Defaults to "nonrobust".
            clusters (pd.Series | None, optional): The cluster label for each observation
            (e.g. a coarser time bucket derived from the return index, to correct for
            within-period correlation), required when `cov_type="cluster"`. Aligned to the
            regression's own index before use, so it may be indexed by the full period
            index even though the regression drops periods with missing data. Defaults
            to None.
            maxlags (int | None, optional): The maximum lag to include when estimating the
            HAC (Newey-West) covariance matrix, only used when `cov_type="HAC"`. Defaults to
            None, meaning the Newey & West (1994) rule of thumb `floor(4 * (n / 100)^(2/9))`
            with `n` the number of observations.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A coefficient table (Coefficient, Std. Error, t-Statistic, P-Value),
            indexed by regressor name.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["XOM", "USO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        # XOM (the first ticker) is dependent; USO and the Benchmark are independent.
        # dependent_ticker and independent_tickers choose others, cov_type='HC1' gives
        # heteroskedasticity-robust and cov_type='HAC' (with maxlags) Newey-West standard errors.
        toolkit.econometrics.get_ols(include_benchmark=True)
        ```

        Which returns:

        |           |   Coefficient |   Std. Error |   t-Statistic |   P-Value |
        |:----------|--------------:|-------------:|--------------:|----------:|
        | Intercept |        0.0005 |       0.0004 |        1.5047 |    0.1327 |
        | USO       |        0.4878 |       0.0172 |       28.4214 |    0      |
        | Benchmark |        0.366  |       0.0333 |       10.9991 |    0      |
        """
        period = period if period else "daily"
        dependent_ticker, independent_tickers = self._resolve_dependent_independent(
            dependent_ticker=dependent_ticker,
            independent_tickers=independent_tickers,
            include_benchmark=include_benchmark,
        )
        returns = self._get_series_column(
            period, column, [dependent_ticker, *independent_tickers]
        )

        # Positionally aligned by statsmodels, so it has to be reindexed onto the regression's own index first -- `returns` has already dropped any period with missing data, which a caller-supplied cluster label will not have.  # noqa: E501
        aligned_clusters = (
            clusters.reindex(returns.index) if clusters is not None else None
        )

        result = regression_model.get_ols(
            returns[dependent_ticker],
            returns[independent_tickers],
            add_constant=add_constant,
            cov_type=cov_type,
            clusters=aligned_clusters,
            maxlags=maxlags,
        )

        return apply_rounding(
            regression_model.regression_summary_table(result),
            rounding if rounding is not None else self._rounding,
        )

    @handle_errors
    def get_wls(
        self,
        weights: pd.Series | str = "feasible",
        dependent_ticker: str | None = None,
        independent_tickers: str | list[str] | None = None,
        include_benchmark: bool = False,
        period: str | None = None,
        column: str = "Return",
        add_constant: bool = True,
        cov_type: str = "nonrobust",
        clusters: pd.Series | None = None,
        maxlags: int | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Fit a Weighted Least Squares (WLS) regression of `dependent_ticker` on
        `independent_tickers`.

        When the error variance differs across observations (heteroskedasticity) with a known
        or estimable structure, `Var(e_i) = sigma^2 / w_i`, OLS remains unbiased but is no
        longer the most efficient estimator. WLS restores efficiency by minimising the
        weighted sum of squared residuals, so noisier observations (low weight) count less
        than precise ones (high weight).

        The formula is as follows:

        - β̂ = (X'WX)⁻¹X'Wy, with W = diag(weights), the same as OLS of √wₜ·yₜ on √wₜ·xₜ
        - Cov(β̂) = σ̂²·(X'WX)⁻¹, with σ̂² = Σwₜ·eₜ² / (n - k) (cov_type="nonrobust")
        - y is the dependent ticker's `column` (Return by default) and X the independent tickers'
          `column` plus a constant, on the dates where the returns and `weights` all have data

        With equal weights WLS reduces to OLS. A typical use is down-weighting periods of
        high volatility, so that a few turbulent periods do not dominate the estimated
        relationship. By default (weights="feasible") the weights are estimated from the
        regression itself with the feasible GLS procedure of Wooldridge (2016): the
        logarithm of the squared OLS residuals is regressed on the independent tickers, and
        each period's weight is 1 / exp(fitted value), the inverse of its estimated error
        variance. Use `get_breusch_pagan_test` or `get_white_test` to establish whether
        heteroskedasticity is present before choosing WLS over OLS.

        For more information about the feasible weights, see the following book:

        - Wooldridge, J.M. (2016). "Introductory Econometrics: A Modern Approach", 6th
        edition, Section 8.4. Cengage Learning.

        Also known as: weighted regression, feasible WLS.

        Args:
            weights (pd.Series | str, optional): The (positive) weight of each observation,
            aligned to the same period index as the return data (e.g. `1 / rolling_variance`),
            or "feasible" to estimate them from the regression's own residuals with the
            feasible GLS procedure of Wooldridge (2016). Defaults to "feasible".
            dependent_ticker (str | None, optional): The dependent (predicted) asset.
            Defaults to None, meaning the Toolkit instance's first ticker.
            independent_tickers (str | list[str] | None, optional): The independent
            (predictor) asset(s). Defaults to None, meaning every other ticker in
            the Toolkit instance besides `dependent_ticker`.
            include_benchmark (bool, optional): Whether to include "Benchmark" in the
            default independent ticker(s) (has no effect when independent_tickers is given
            explicitly). Defaults to False.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to regress on. Defaults to "Return".
            add_constant (bool, optional): Whether to include an intercept. Defaults to True.
            cov_type (str, optional): Which covariance estimator to use, applied to the weighted/
            transformed problem -- see `get_ols`'s `cov_type` for the full list of options.
            Defaults to "nonrobust".
            clusters (pd.Series | None, optional): The cluster label for each observation,
            required when `cov_type="cluster"`. Defaults to None.
            maxlags (int | None, optional): The maximum lag to include when estimating the
            HAC (Newey-West) covariance matrix, only used when `cov_type="HAC"`. Defaults to
            None, meaning the Newey & West (1994) rule of thumb (see `get_ols`).
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A coefficient table (Coefficient, Std. Error, t-Statistic, P-Value),
            indexed by regressor name.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        # AAPL (the first ticker) is dependent, MSFT is independent; the weights are estimated
        toolkit.econometrics.get_wls(period="weekly")
        ```

        Which returns:

        |           |   Coefficient |   Std. Error |   t-Statistic |   P-Value |
        |:----------|--------------:|-------------:|--------------:|----------:|
        | Intercept |        0.0012 |       0.0019 |        0.6263 |    0.5317 |
        | MSFT      |        0.6356 |       0.0596 |       10.6643 |    0      |
        """
        period = period if period else "daily"
        dependent_ticker, independent_tickers = self._resolve_dependent_independent(
            dependent_ticker=dependent_ticker,
            independent_tickers=independent_tickers,
            include_benchmark=include_benchmark,
        )
        returns = self._get_series_column(
            period, column, [dependent_ticker, *independent_tickers]
        )

        estimate_weights = isinstance(weights, str)
        if estimate_weights and weights != "feasible":
            raise ValueError(
                f"weights must be a Series of weights or 'feasible', received '{weights}'."
            )

        to_align = [returns[dependent_ticker], returns[independent_tickers]]
        if not estimate_weights:
            to_align.append(weights.rename("__weights__"))
        if clusters is not None:
            to_align.append(clusters.rename("__clusters__"))

        aligned = pd.concat(to_align, axis=1, join="inner").dropna()
        aligned_clusters = aligned.pop("__clusters__") if clusters is not None else None
        # Estimated weights come from the same observations as the regression they weight.
        aligned_weights = (
            regression_model.get_feasible_weights(
                aligned.iloc[:, 0], aligned[independent_tickers], add_constant
            )
            if estimate_weights
            else aligned.pop("__weights__")
        )

        result = regression_model.get_wls(
            aligned.iloc[:, 0],
            aligned[independent_tickers],
            aligned_weights,
            add_constant=add_constant,
            cov_type=cov_type,
            clusters=aligned_clusters,
            maxlags=maxlags,
        )

        return apply_rounding(
            regression_model.regression_summary_table(result),
            rounding if rounding is not None else self._rounding,
        )

    @handle_errors
    def get_gls(
        self,
        omega: pd.DataFrame | str = "ar1",
        dependent_ticker: str | None = None,
        independent_tickers: str | list[str] | None = None,
        include_benchmark: bool = False,
        period: str | None = None,
        column: str = "Return",
        add_constant: bool = True,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Fit a Generalized Least Squares (GLS) regression of `dependent_ticker` on
        `independent_tickers`, given a known error covariance structure `omega`.

        GLS generalises WLS to an arbitrary error covariance matrix `Var(e) = sigma^2 * Omega`.
        Where WLS only corrects for heteroskedasticity (a diagonal `Omega`), GLS also
        corrects for correlation between the errors of different observations, such as the
        autocorrelation that is common in financial time series (a non-diagonal `Omega`).

        The formula is as follows:

        - β̂ = (X'Ω⁻¹X)⁻¹X'Ω⁻¹y
        - Cov(β̂) = σ̂²·(X'Ω⁻¹X)⁻¹, with σ̂² = e'Ω⁻¹e / (n - k)
        - y is the dependent ticker's `column` (Return by default) and X the independent tickers'
          `column` plus a constant (add_constant=True)

        Under a correctly specified `Omega` this is the most efficient linear unbiased
        estimator. By default (omega="ar1") `Omega` is estimated from the regression itself
        as an AR(1) error structure, eₜ = ρ·eₜ₋₁ + uₜ: ρ is the slope of the OLS residuals on
        their own lag and Ωᵢⱼ = ρ^|i - j|, which makes this the Prais-Winsten estimator. Any
        other structure can be supplied as `omega`. `get_durbin_watson_test` helps establish
        whether the residuals are autocorrelated in the first place.

        For more information about the method, see the following paper:

        - Aitken, A.C. (1935). "On Least Squares and Linear Combinations of Observations."
        Proceedings of the Royal Society of Edinburgh, 55, 42-48.
        - Prais, S.J. and Winsten, C.B. (1954). "Trend Estimators and Serial Correlation."
        Cowles Commission Discussion Paper No. 383.

        Also known as: GLS, Prais-Winsten.

        Args:
            omega (pd.DataFrame | str, optional): The (symmetric, positive-definite) error
            covariance structure, up to a scalar, shape `(n, n)`, or "ar1" to estimate an
            AR(1) structure from the regression's own residuals (Prais-Winsten). Defaults
            to "ar1".
            dependent_ticker (str | None, optional): The dependent (predicted) asset.
            Defaults to None, meaning the Toolkit instance's first ticker.
            independent_tickers (str | list[str] | None, optional): The independent
            (predictor) asset(s). Defaults to None, meaning every other ticker in
            the Toolkit instance besides `dependent_ticker`.
            include_benchmark (bool, optional): Whether to include "Benchmark" in the
            default independent ticker(s) (has no effect when independent_tickers is given
            explicitly). Defaults to False.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to regress on. Defaults to "Return".
            add_constant (bool, optional): Whether to include an intercept. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A coefficient table (Coefficient, Std. Error, t-Statistic, P-Value),
            indexed by regressor name.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        # AAPL (the first ticker) is dependent, MSFT is independent; Omega is estimated (AR(1))
        toolkit.econometrics.get_gls(period="weekly")
        ```

        Which returns:

        |           |   Coefficient |   Std. Error |   t-Statistic |   P-Value |
        |:----------|--------------:|-------------:|--------------:|----------:|
        | Intercept |        0.0012 |       0.0017 |        0.7282 |    0.4671 |
        | MSFT      |        0.6312 |       0.0576 |       10.9557 |    0      |
        """
        period = period if period else "daily"
        dependent_ticker, independent_tickers = self._resolve_dependent_independent(
            dependent_ticker=dependent_ticker,
            independent_tickers=independent_tickers,
            include_benchmark=include_benchmark,
        )
        returns = self._get_series_column(
            period, column, [dependent_ticker, *independent_tickers]
        )

        if isinstance(omega, str):
            if omega != "ar1":
                raise ValueError(
                    f"omega must be a DataFrame or 'ar1', received '{omega}'."
                )
            omega_values = regression_model.get_ar1_covariance(
                returns[dependent_ticker], returns[independent_tickers], add_constant
            )
        else:
            omega_values = omega.to_numpy()

        result = regression_model.get_gls(
            returns[dependent_ticker],
            returns[independent_tickers],
            omega_values,
            add_constant=add_constant,
        )

        return apply_rounding(
            regression_model.regression_summary_table(result),
            rounding if rounding is not None else self._rounding,
        )

    @handle_errors
    def get_logistic_regression(
        self,
        dependent_ticker: str | None = None,
        independent_tickers: str | list[str] | None = None,
        include_benchmark: bool = False,
        period: str | None = None,
        column: str = "Return",
        add_constant: bool = True,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Fit a Logistic Regression (Logit model) of whether `dependent_ticker`'s return is
        positive on `independent_tickers`.

        The dependent variable is turned into a binary outcome (1 when the return is
        positive, 0 otherwise) and the model estimates the probability of that outcome
        through the logistic (sigmoid) function, which maps any value onto the 0 to 1 range.

        The formula is as follows:

        - yₜ = 1 if the dependent ticker's `column` (Return by default) is above 0, else 0
        - P(yₜ = 1 | xₜ) = 1 / (1 + exp(-xₜ'β)), with xₜ the independent tickers' `column` plus a
          constant
        - β̂ = argmax Σ[yₜ·ln(pₜ) + (1 - yₜ)·ln(1 - pₜ)], solved by Newton-Raphson (statsmodels Logit)
        - Pseudo R² = 1 - ln L(β̂) / ln L₀ (McFadden), with L₀ the likelihood of a constant-only model

        The model is fitted by Maximum Likelihood. Coefficients are log-odds: a one-unit
        increase in an independent ticker changes the log-odds of a positive return by that
        coefficient, holding the others fixed, so a positive coefficient means a higher
        probability of a positive return. This answers "does this asset tend to be up when
        the others move this way" rather than "by how much", which is what `get_ols`
        estimates. `get_probit_regression` is the closely related alternative with a normal
        instead of a logistic link.

        The coefficients are per unit of the independent ticker's return, and a unit is a
        return of 100%: a coefficient of 90 on the benchmark means the log-odds of an up
        period rise by about 0.9 for every 1% the benchmark moves. With returns of the same
        period as regressors the model describes co-movement and predicts nothing; a
        forecast needs regressors known in advance, such as the returns of the previous
        period.

        For more information about the method, see the following paper:

        - Berkson, J. (1944). "Application of the Logistic Function to Bio-Assay." Journal of
        the American Statistical Association, 39(227), 357-365.

        Also known as: logit model, logit regression.

        Args:
            dependent_ticker (str | None, optional): The dependent asset (whose
            up/down direction is predicted). Defaults to None, meaning the Toolkit
            instance's first ticker.
            independent_tickers (str | list[str] | None, optional): The independent
            (predictor) asset(s). Defaults to None, meaning every other ticker in
            the Toolkit instance besides `dependent_ticker`.
            include_benchmark (bool, optional): Whether to include "Benchmark" in the
            default independent ticker(s) (has no effect when independent_tickers is given
            explicitly). Defaults to False.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to derive returns from. Defaults to
            "Return".
            add_constant (bool, optional): Whether to include an intercept. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A coefficient table (Coefficient, Std. Error, z-Statistic, P-Value),
            indexed by regressor name.

        Notes:
        - The dependent variable is derived as `1` if the dependent ticker's return in the given
        period is positive, `0` otherwise -- this method predicts the *direction*, not the
        magnitude, of the dependent asset's return.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        # AAPL (the first ticker) is dependent; the Benchmark is independent
        toolkit.econometrics.get_logistic_regression(include_benchmark=True)
        ```

        Which returns:

        |           |   Coefficient |   Std. Error |   z-Statistic |   P-Value |
        |:----------|--------------:|-------------:|--------------:|----------:|
        | Intercept |       -0.0275 |       0.15   |       -0.1836 |    0.8543 |
        | Benchmark |      164.846  |      25.4127 |        6.4868 |    0      |
        """
        period = period if period else "daily"
        dependent_ticker, independent_tickers = self._resolve_dependent_independent(
            dependent_ticker=dependent_ticker,
            independent_tickers=independent_tickers,
            include_benchmark=include_benchmark,
        )
        returns = self._get_series_column(
            period, column, [dependent_ticker, *independent_tickers]
        )

        direction = (returns[dependent_ticker] > 0).astype(float)

        result = regression_model.get_logistic_regression(
            direction, returns[independent_tickers], add_constant=add_constant
        )

        return apply_rounding(
            regression_model.binary_regression_summary_table(result),
            rounding if rounding is not None else self._rounding,
        )

    @handle_errors
    def get_probit_regression(
        self,
        dependent_ticker: str | None = None,
        independent_tickers: str | list[str] | None = None,
        include_benchmark: bool = False,
        period: str | None = None,
        column: str = "Return",
        add_constant: bool = True,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Fit a Probit Regression of whether `dependent_ticker`'s return is positive on
        `independent_tickers`.

        As with `get_logistic_regression`, the dependent variable is turned into a binary
        outcome (1 when the return is positive, 0 otherwise). The probability of that outcome
        is modelled through the standard normal cumulative distribution function instead of
        the logistic function.

        The formula is as follows:

        - yₜ = 1 if the dependent ticker's `column` (Return by default) is above 0, else 0
        - P(yₜ = 1 | xₜ) = Φ(xₜ'β), with xₜ the independent tickers' `column` plus a constant
        - β̂ = argmax Σ[yₜ·ln(pₜ) + (1 - yₜ)·ln(1 - pₜ)], solved by Newton-Raphson (statsmodels Probit)
        - Pseudo R² = 1 - ln L(β̂) / ln L₀ (McFadden), with L₀ the likelihood of a constant-only model

        Where `Phi` is the standard normal CDF. The model is fitted by Maximum Likelihood. A
        positive coefficient means a higher probability of a positive return, but the size of
        a coefficient is not a log-odds ratio as in the Logit model. In practice Probit and
        Logit give very similar fitted probabilities; Probit is the convention in parts of the
        econometrics literature and underlies two-step estimators such as the Heckman
        selection correction.

        For more information about the method, see the following paper:

        - Bliss, C.I. (1934). "The Method of Probits." Science, 79(2037), 38-39.

        Also known as: probit model.

        Args:
            dependent_ticker (str | None, optional): The dependent asset (whose
            up/down direction is predicted). Defaults to None, meaning the Toolkit
            instance's first ticker.
            independent_tickers (str | list[str] | None, optional): The independent
            (predictor) asset(s). Defaults to None, meaning every other ticker in
            the Toolkit instance besides `dependent_ticker`.
            include_benchmark (bool, optional): Whether to include "Benchmark" in the
            default independent ticker(s) (has no effect when independent_tickers is given
            explicitly). Defaults to False.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to derive returns from. Defaults to
            "Return".
            add_constant (bool, optional): Whether to include an intercept. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A coefficient table (Coefficient, Std. Error, z-Statistic, P-Value),
            indexed by regressor name.

        Notes:
        - The dependent variable is derived as `1` if the dependent ticker's return in the given
        period is positive, `0` otherwise -- this method predicts the *direction*, not the
        magnitude, of the dependent asset's return.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["TSLA", "QQQ"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        # TSLA (the first ticker) is dependent; QQQ is independent
        toolkit.econometrics.get_probit_regression()
        ```

        Which returns:

        |           |   Coefficient |   Std. Error |   z-Statistic |   P-Value |
        |:----------|--------------:|-------------:|--------------:|----------:|
        | Intercept |       -0.0848 |       0.0913 |       -0.9294 |    0.3527 |
        | QQQ       |       78.1324 |      11.1444 |        7.0109 |    0      |
        """
        period = period if period else "daily"
        dependent_ticker, independent_tickers = self._resolve_dependent_independent(
            dependent_ticker=dependent_ticker,
            independent_tickers=independent_tickers,
            include_benchmark=include_benchmark,
        )
        returns = self._get_series_column(
            period, column, [dependent_ticker, *independent_tickers]
        )

        direction = (returns[dependent_ticker] > 0).astype(float)

        result = regression_model.get_probit_regression(
            direction, returns[independent_tickers], add_constant=add_constant
        )

        return apply_rounding(
            regression_model.binary_regression_summary_table(result),
            rounding if rounding is not None else self._rounding,
        )

    @handle_errors
    def get_quantile_regression(
        self,
        dependent_ticker: str | None = None,
        independent_tickers: str | list[str] | None = None,
        include_benchmark: bool = False,
        tau: float = 0.5,
        period: str | None = None,
        column: str = "Return",
        add_constant: bool = True,
        n_bootstrap: int = 0,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Fit a Quantile Regression of `dependent_ticker` on `independent_tickers` at quantile
        `tau`.

        Where OLS estimates how the average (conditional mean) of the dependent ticker depends
        on the independent tickers, Quantile Regression estimates how a chosen quantile of it
        does. It minimises an asymmetric ("pinball") loss that is smallest at the conditional
        `tau`-quantile.

        The formula is as follows:

        - β̂(τ) = argmin Σ(τ·max(uₜ, 0) + (1 - τ)·max(-uₜ, 0)), with uₜ = yₜ - xₜ'β and τ = `tau` (0.5
          by default)
        - y is the dependent ticker's `column` (Return by default) and X the independent tickers'
          `column` plus a constant (add_constant=True)
        - SE(β̂): statsmodels' kernel sandwich estimator (Epanechnikov kernel, Hall-Sheather bandwidth),
          or, when `n_bootstrap` > 0, the standard deviation (ddof=1) of β̂ over `n_bootstrap`
          resampled-pairs fits

        With `tau = 0.5` this is the median regression, which is far less sensitive to
        outliers than OLS. Fitting at several quantiles (e.g. 0.05, 0.5 and 0.95) shows how
        the whole distribution of returns responds to the independent tickers, for example
        whether an asset's sensitivity to the market is larger in sell-offs (low quantiles)
        than in normal periods, which is directly relevant for downside risk.

        Comparing the coefficients at a low `tau` with those at `tau=0.5`, or with the
        betas from `get_ols`, shows whether the sensitivity to the market rises in
        sell-offs. The table has no p-values: a coefficient more than about two standard
        errors from zero is significant at the 5% level.

        For more information about the method, see the following paper:

        - Koenker, R., & Bassett, G. (1978). "Regression Quantiles." Econometrica, 46(1),
        33-50.

        Also known as: QR.

        Args:
            dependent_ticker (str | None, optional): The dependent (predicted) asset.
            Defaults to None, meaning the Toolkit instance's first ticker.
            independent_tickers (str | list[str] | None, optional): The independent
            (predictor) asset(s). Defaults to None, meaning every other ticker in
            the Toolkit instance besides `dependent_ticker`.
            include_benchmark (bool, optional): Whether to include "Benchmark" in the
            default independent ticker(s) (has no effect when independent_tickers is given
            explicitly). Defaults to False.
            tau (float, optional): The quantile to fit, in (0, 1). Defaults to 0.5 (the median).
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to regress on. Defaults to "Return".
            add_constant (bool, optional): Whether to include an intercept. Defaults to True.
            n_bootstrap (int, optional): The number of bootstrap resamples used for coefficient
            standard errors, overriding `statsmodels`' default analytic (kernel density-based)
            standard errors. Defaults to 0 (use the analytic standard errors).
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A coefficient table (Coefficient, Std. Error), indexed by regressor name.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["BAC", "KRE"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        # BAC (the first ticker) is dependent; KRE and the Benchmark are independent.
        # tau=0.05 describes how the worst 5% of days respond.
        toolkit.econometrics.get_quantile_regression(include_benchmark=True, tau=0.05)
        ```

        Which returns:

        |           |   Coefficient |   Std. Error |
        |:----------|--------------:|-------------:|
        | Intercept |       -0.0146 |       0.0006 |
        | KRE       |        0.5271 |       0.0484 |
        | Benchmark |        0.3965 |       0.1133 |
        """
        period = period if period else "daily"
        dependent_ticker, independent_tickers = self._resolve_dependent_independent(
            dependent_ticker=dependent_ticker,
            independent_tickers=independent_tickers,
            include_benchmark=include_benchmark,
        )
        returns = self._get_series_column(
            period, column, [dependent_ticker, *independent_tickers]
        )

        result = regression_model.get_quantile_regression(
            returns[dependent_ticker],
            returns[independent_tickers],
            tau=tau,
            add_constant=add_constant,
            n_bootstrap=n_bootstrap,
        )

        return apply_rounding(
            regression_model.quantile_regression_summary_table(result),
            rounding if rounding is not None else self._rounding,
        )

    @handle_errors
    def get_fama_macbeth_regression(
        self,
        factor_tickers: str | list[str] | None = None,
        asset_tickers: str | list[str] | None = None,
        period: str | None = None,
        column: str = "Return",
        add_constant: bool = True,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Fit a Fama-MacBeth (1973) two-pass cross-sectional regression: `asset_tickers` is
        treated as the cross-section of test assets, `factor_tickers` as the risk factor(s)
        whose risk premia are estimated -- the standard procedure for testing whether a
        proposed risk factor is actually priced.

        The method runs in two passes:

        1. Time-series pass (per asset): regress each asset's returns on the factor(s) to
        estimate its factor loadings (betas).
        2. Cross-sectional pass (per period): regress that period's returns of all assets on
        those betas, which gives one estimate of each factor's risk premium per period.

        The formula is as follows:

        - First pass: R_i,t = alpha_i + beta_i' * F_t + e_i,t, for each asset i, over the full sample
        - Second pass: R_i,t = gamma_0,t + gamma_t' * beta_i + u_i,t, for each period t (gamma_0,t only
          with add_constant=True)
        - Risk premium: gamma_hat = mean_t(gamma_t)
        - Standard error: SE(gamma_hat) = std_t(gamma_t) / sqrt(T), with ddof=1
        - t = gamma_hat / SE(gamma_hat), with p-values from a Student-t distribution with T - 1 degrees
          of freedom
        - R are the asset tickers' `column` (Return by default) and F the factor tickers' `column`
          (the Benchmark by default)

        The risk premium is the average of the per-period estimates, and its standard error
        comes from how much those estimates vary over time. That is the method's key
        contribution: it accounts for correlation between the assets' residuals without
        having to model it. A significant positive premium means investors are compensated
        for exposure to the factor. This uses a single full-sample set of betas (the
        textbook variant), not the original rolling-window betas.

        The first-pass betas are estimates, which makes the second-pass standard errors too
        small; the Shanken (1992) correction, not applied here, widens them. The returns are
        not in excess of the risk-free rate, so the intercept is the return of a zero-beta
        asset, which should be close to the risk-free rate if the factors price the assets.

        For more information about the method, see the following papers:

        - Fama, E.F. & MacBeth, J.D. (1973). "Risk, Return, and Equilibrium: Empirical Tests."
        Journal of Political Economy, 81(3), 607-636.
        - Shanken, J. (1992). "On the Estimation of Beta-Pricing Models." Review of Financial
        Studies, 5(1), 1-33.
        - Cochrane, J.H. (2005). "Asset Pricing." Princeton University Press, Chapter 12.

        Also known as: two-pass regression, Fama-MacBeth procedure.

        Args:
            factor_tickers (str | list[str] | None, optional): The ticker(s) whose
            returns are used as the risk factor(s) (e.g. "Benchmark" for a
            single-factor/CAPM-style test). Defaults to None, meaning `["Benchmark"]`.
            asset_tickers (str | list[str] | None, optional): The ticker(s) forming the
            cross-section of test assets. Defaults to None, meaning every Toolkit
            ticker (including "Benchmark") not already used as a factor.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to regress on. Defaults to "Return".
            add_constant (bool, optional): Whether to include an intercept in the
            second-pass cross-sectional regression. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A coefficient table (Risk Premium, Std. Error, t-Statistic,
            P-Value), indexed by factor name (plus "Intercept" if present).

        Notes:
        - The cross-sectional second pass needs strictly more assets than factors
        (plus an intercept, if `add_constant=True`) to be identified -- with only a
        couple of Toolkit tickers loaded, either pass `add_constant=False` or (better)
        construct the Toolkit with many tickers, since Fama-MacBeth is fundamentally a
        many-asset cross-sectional technique (dozens of assets is typical in practice),
        not a limitation of this method's implementation.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "PG", "KO", "WMT", "CAT", "NEE"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        # The Benchmark (the default factor) is the single risk factor and the ten
        # stocks are the test assets.
        toolkit.econometrics.get_fama_macbeth_regression(period="monthly")
        ```

        Which returns:

        |           |   Risk Premium |   Std. Error |   t-Statistic |   P-Value |
        |:----------|---------------:|-------------:|--------------:|----------:|
        | Intercept |         0.0066 |       0.0058 |        1.1505 |    0.2547 |
        | Benchmark |         0.0101 |       0.008  |        1.2668 |    0.2103 |
        """
        period = period if period else "daily"
        factor_tickers = (
            [factor_tickers]
            if isinstance(factor_tickers, str)
            else (
                list(factor_tickers)
                if factor_tickers is not None
                else [BENCHMARK_TICKER]
            )
        )

        if asset_tickers is None:
            asset_tickers = [
                ticker
                for ticker in self._get_tickers(include_benchmark=True)
                if ticker not in factor_tickers
            ]
        else:
            asset_tickers = (
                [asset_tickers]
                if isinstance(asset_tickers, str)
                else list(asset_tickers)
            )

        returns = self._get_series_column(
            period, column, [*asset_tickers, *factor_tickers]
        )

        result = fama_macbeth_model.get_fama_macbeth_regression(
            returns[asset_tickers],
            returns[factor_tickers],
            add_constant=add_constant,
        )

        return apply_rounding(
            fama_macbeth_model.fama_macbeth_summary_table(result),
            rounding if rounding is not None else self._rounding,
        )

    def _get_panel_data(
        self,
        independent_tickers: str | list[str] | None,
        independent_column: str | None,
        dependent_tickers: str | list[str] | None,
        period: str | None,
        column: str,
        include_benchmark: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        """
        Shared input-shaping step for `get_fixed_effects`/`get_random_effects`/
        `get_hausman_test`: unlike every other regression method in this module
        (which pairs one dependent ticker with one or more independent tickers,
        all on equal footing), panel data methods need a genuine entity dimension
        that a handful of single ticker return series don't have on their own.

        This module treats `dependent_tickers` (defaulting to every ticker in the
        `Toolkit` instance, without "Benchmark" unless `include_benchmark` is True, minus
        `independent_tickers` if given) as the panel of ENTITIES -- e.g. a panel of stocks -- and supports
        two, mutually exclusive ways of building the regressor `x`:

        - `independent_tickers`: a COMMON, time-varying factor (or factors),
        broadcast identically to every entity at each point in time -- e.g. a
        market benchmark, mirroring a Fama-French-style factor regression.
        - `independent_column`: a PER-ENTITY regressor -- each entity's own value
        of a different historical data column (e.g. `"Volume"` explaining
        `"Return"`), rather than an external, shared series.

        Both describe "does X explain Y across a panel of stocks, controlling for
        stock fixed effects," but they are not interchangeable for every estimator
        here: `independent_tickers` carries no BETWEEN-entity variation once
        broadcast to every entity over the same dates (this `Toolkit`'s historical
        data is always aligned onto a common calendar across tickers), which makes
        Random Effects'/the Hausman test's between-entity regression step exactly
        rank-deficient (the entity-mean of a broadcast series is identical for
        every entity, perfectly collinear with the intercept). `independent_column`
        does not have this problem, since different entities genuinely have
        different average Volume (or whichever column), and so is the more
        reliable choice for `get_random_effects`/`get_hausman_test`, while
        `independent_tickers` remains natural (and works for every estimator here,
        including these two, granted enough cross-entity variation) for
        `get_fixed_effects`'s canonical Fama-French-style use case.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, list[str]]: `y` shaped entity (ticker)
            x time (a wide `pd.DataFrame`, accepted as-is by
            `panel_data_model.get_fixed_effects`/`get_random_effects`), `x` in the
            same shape (single regressor) or `(entity, time)`-MultiIndexed (multiple
            `independent_tickers`), and the resolved list of dependent (entity)
            tickers.

        Raises:
            ValueError: If both or neither of `independent_tickers`/
            `independent_column` are given, the period is invalid, or no dependent
            (entity) tickers remain.
        """
        if (independent_tickers is None) == (independent_column is None):
            raise ValueError(
                "Provide exactly one of independent_tickers (a common factor "
                "broadcast to every entity) or independent_column (a per-entity "
                "regressor taken from each dependent ticker's own data)."
            )

        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )

        # The raw slice: panel methods need each entity's own NaN pattern on the full date index.
        prices = self._historical_data[period][column].loc[
            self._start_date : self._end_date
        ]

        if independent_tickers is not None:
            independent_tickers = (
                [independent_tickers]
                if isinstance(independent_tickers, str)
                else list(independent_tickers)
            )

            # The factor is read from the loaded data, so it has to be one of its tickers.
            missing = [ticker for ticker in independent_tickers if ticker not in prices]
            if missing:
                raise ValueError(
                    f"independent_tickers {', '.join(missing)} are not among the loaded "
                    "tickers: add them to the tickers, or use 'Benchmark' for the "
                    "benchmark."
                )

            if dependent_tickers is None:
                dependent_tickers = [
                    ticker
                    for ticker in prices.columns
                    if ticker not in independent_tickers
                    and (include_benchmark or ticker != BENCHMARK_TICKER)
                ]
            else:
                dependent_tickers = (
                    [dependent_tickers]
                    if isinstance(dependent_tickers, str)
                    else list(dependent_tickers)
                )

            if not dependent_tickers:
                raise ValueError(
                    "No dependent (entity) tickers remain after excluding "
                    "independent_tickers -- provide dependent_tickers explicitly."
                )

            y_panel = prices[dependent_tickers].T
            factor_values = prices[independent_tickers]
            x_panel = pd.concat(
                {entity: factor_values for entity in dependent_tickers},
                names=["entity", "time"],
            )
        else:
            if dependent_tickers is None:
                dependent_tickers = [
                    ticker
                    for ticker in prices.columns
                    if include_benchmark or ticker != BENCHMARK_TICKER
                ]
            else:
                dependent_tickers = (
                    [dependent_tickers]
                    if isinstance(dependent_tickers, str)
                    else list(dependent_tickers)
                )

            if not dependent_tickers:
                raise ValueError(
                    "No dependent (entity) tickers -- provide dependent_tickers "
                    "explicitly."
                )

            independent_prices = self._historical_data[period][independent_column].loc[
                self._start_date : self._end_date
            ]

            y_panel = prices[dependent_tickers].T
            x_series = independent_prices[dependent_tickers].T.stack()
            x_series.name = independent_column
            x_panel = x_series.to_frame()

        return y_panel, x_panel, dependent_tickers

    @handle_errors
    def get_fixed_effects(
        self,
        independent_tickers: str | list[str] | None = None,
        independent_column: str | None = None,
        dependent_tickers: str | list[str] | None = None,
        period: str | None = None,
        column: str = "Return",
        include_benchmark: bool = False,
        entity_effects: bool = True,
        time_effects: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Fit a Fixed Effects ("within") estimator explaining a panel of
        `dependent_tickers` (the entities) by a regressor `x`, controlling for
        entity-specific (and/or time-specific) fixed effects.

        Also known as: within estimator, FE, least squares dummy variable (LSDV)
        estimator.

        Unlike this module's other regression methods (`get_ols`, `get_wls`, ...),
        which each compare a handful of individual ticker return series on equal
        footing, this treats `dependent_tickers` as a genuine panel of entities --
        e.g. every stock in the `Toolkit` instance -- observed over time. The
        regressor `x` is built in exactly one of two ways (provide exactly one of
        the two arguments below):

        - `independent_tickers`: a COMMON factor (or factors) applied identically
        to every entity at each date (e.g. a market benchmark, mirroring a
        Fama-French-style factor regression).
        - `independent_column`: a PER-ENTITY regressor -- each entity's own value
        of a *different* historical data column (e.g. does `"Volume"` explain
        `"Return"`, across the panel).

        Fixed Effects removes any purely entity-specific, time-invariant
        characteristic (e.g. a stock's typical risk premium) before estimating the
        regressor's coefficient(s), by demeaning every variable by its entity's
        mean. See `panel_data_model.get_fixed_effects` for the full formula and
        references.

        The formula is as follows:

        - ỹᵢₜ = yᵢₜ - mean_t(yᵢₜ) and x̃ᵢₜ = xᵢₜ - mean_t(xᵢₜ) (entity_effects=True, the default);
          time_effects=True also removes each date's mean (two-way fixed effects)
        - β̂_FE = (X̃'X̃)⁻¹X̃'ỹ, without a separate constant (linearmodels PanelOLS, unadjusted covariance)
        - Entity Effect: αᵢ = mean_t(yᵢₜ - xᵢₜ'β̂_FE); Time Effect: λₜ = mean_i(yᵢₜ - xᵢₜ'β̂_FE)
        - yᵢₜ is each dependent ticker's `column` (Return by default)

        Args:
            independent_tickers (str | list[str] | None, optional): The factor
            ticker(s), whose `column` values are broadcast identically to every
            entity at each date. Mutually exclusive with `independent_column`.
            independent_column (str | None, optional): A different historical data
            column, taken per-entity from each of `dependent_tickers`' own data, to
            use as the regressor. Mutually exclusive with `independent_tickers`.
            dependent_tickers (str | list[str] | None, optional): The panel of
            entity tickers to explain. Defaults to None, meaning every ticker in
            the `Toolkit` instance (other than `independent_tickers`, if given).
            period (str, optional): The data frequency (daily, weekly, monthly,
            quarterly, or yearly). Defaults to "daily".
            column (str, optional): The dependent variable's historical data
            column. Defaults to "Return".
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
            entities. Defaults to False.
            entity_effects (bool, optional): Whether to control for time-invariant
            entity-specific characteristics. Defaults to True.
            time_effects (bool, optional): Whether to control for entity-invariant,
            time-specific shocks. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the
            results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A coefficient table (Coefficient, Std. Error,
            t-Statistic, P-Value) for the regressor(s), indexed by name, with the
            recovered entity (and/or time) fixed-effect intercepts appended as
            additional rows (only the Coefficient column is populated for those).

        Raises:
            ValueError: If both or neither of `independent_tickers`/
            `independent_column` are given, or (see `panel_data_model.
            get_fixed_effects`) if there are not enough observations for the
            requested fixed effects, or the fixed effects absorb the regressor: the
            common factor of `independent_tickers` is identical for every entity at
            each date, so it cannot be combined with `time_effects=True`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["XOM", "CVX", "SHEL", "BP", "TTE"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_fixed_effects(independent_tickers="Benchmark")
        ```

        Which returns:

        |                     |   Coefficient |   Std. Error |   t-Statistic |   P-Value |
        |:--------------------|--------------:|-------------:|--------------:|----------:|
        | Benchmark           |        0.5795 |       0.0183 |       31.7272 |         0 |
        | Entity Effect: BP   |        0.0004 |     nan      |      nan      |       nan |
        | Entity Effect: CVX  |        0.0004 |     nan      |      nan      |       nan |
        | Entity Effect: SHEL |        0.0005 |     nan      |      nan      |       nan |
        | Entity Effect: TTE  |        0.0004 |     nan      |      nan      |       nan |
        | Entity Effect: XOM  |        0.0008 |     nan      |      nan      |       nan |
        """
        y_panel, x_panel, dependent_tickers = self._get_panel_data(
            independent_tickers,
            independent_column,
            dependent_tickers,
            period,
            column,
            include_benchmark=include_benchmark,
        )

        result = panel_data_model.get_fixed_effects(
            y_panel, x_panel, entity_effects=entity_effects, time_effects=time_effects
        )

        rounding = rounding if rounding is not None else self._rounding
        summary = regression_model.regression_summary_table(result["regression"])

        if result["entity_effects"] is not None:
            entity_rows = pd.DataFrame({"Coefficient": result["entity_effects"]})
            entity_rows.index = [
                f"Entity Effect: {entity}" for entity in entity_rows.index
            ]
            summary = pd.concat([summary, entity_rows])

        if result["time_effects"] is not None:
            time_rows = pd.DataFrame({"Coefficient": result["time_effects"]})
            time_rows.index = [f"Time Effect: {time}" for time in time_rows.index]
            summary = pd.concat([summary, time_rows])

        return apply_rounding(summary, rounding)

    @handle_errors
    def get_random_effects(
        self,
        independent_tickers: str | list[str] | None = None,
        independent_column: str | None = None,
        dependent_tickers: str | list[str] | None = None,
        period: str | None = None,
        column: str = "Return",
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Fit a Random Effects estimator explaining a panel of `dependent_tickers`
        (the entities) by a regressor `x`, via Swamy-Arora feasible Generalized
        Least Squares.

        Also known as: RE, GLS panel estimator, Swamy-Arora estimator.

        See `get_fixed_effects` for how `dependent_tickers`/`independent_tickers`/
        `independent_column` are shaped into a panel, and
        `panel_data_model.get_random_effects` for the full estimator formula,
        references, and how it compares to Fixed Effects. Unlike Fixed Effects,
        Random Effects retains and estimates a single, population-average
        intercept rather than one intercept per entity -- more efficient than
        Fixed Effects if entity effects are indeed uncorrelated with the
        regressor(s) (see `get_hausman_test` to check that assumption).

        The formula is as follows:

        - θᵢ = 1 - √(σ²_ε / (σ²_ε + Tᵢ·σ²_u)), with the Swamy-Arora estimates of the idiosyncratic (σ²_ε)
          and entity (σ²_u) variance components
        - y*ᵢₜ = yᵢₜ - θᵢ·mean_t(yᵢₜ) and x*ᵢₜ = xᵢₜ - θᵢ·mean_t(xᵢₜ), the constant becoming 1 - θᵢ
        - β̂_RE = OLS of y* on x* (linearmodels RandomEffects), with yᵢₜ each dependent ticker's `column`
          (Return by default)

        Args:
            independent_tickers (str | list[str] | None, optional): The factor
                ticker(s), whose `column` values are broadcast identically to every
                entity at each date. Mutually exclusive with `independent_column`.
                Since this `Toolkit` instance's historical data is aligned onto a
                common calendar across tickers, a broadcast factor's entity mean is
                identical for every entity, leaving no between-entity variation to
                identify Random Effects' between-regression step -- prefer
                `independent_column` here unless `dependent_tickers` genuinely differ
                in their date coverage (e.g. different listing histories).
            independent_column (str | None, optional): A different historical data
                column, taken per-entity from each of `dependent_tickers`' own data, to
                use as the regressor. Mutually exclusive with `independent_tickers`.
            dependent_tickers (str | list[str] | None, optional): The panel of
                entity tickers to explain. Defaults to None, meaning every ticker in
                the `Toolkit` instance (other than `independent_tickers`, if given).
            period (str, optional): The data frequency (daily, weekly, monthly,
                quarterly, or yearly). Defaults to "daily".
            column (str, optional): The dependent variable's historical data
                column. Defaults to "Return".
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
                entities. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the
                results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A coefficient table (Coefficient, Std. Error,
            t-Statistic, P-Value), indexed by regressor name plus "Intercept".

        Raises:
            ValueError: If both or neither of `independent_tickers`/
            `independent_column` are given, or (see `panel_data_model.
            get_random_effects`) if there are not enough entities or observations
            to estimate the model.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["JPM", "BAC", "C", "WFC"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_random_effects(independent_tickers="Benchmark")
        ```

        Which returns:

        |           |   Coefficient |   Std. Error |   t-Statistic |   P-Value |
        |:----------|--------------:|-------------:|--------------:|----------:|
        | Intercept |        0.0003 |       0.0002 |        1.7192 |    0.0856 |
        | Benchmark |        0.9788 |       0.0183 |       53.4931 |    0      |
        """
        y_panel, x_panel, dependent_tickers = self._get_panel_data(
            independent_tickers,
            independent_column,
            dependent_tickers,
            period,
            column,
            include_benchmark=include_benchmark,
        )

        result = panel_data_model.get_random_effects(y_panel, x_panel)

        return apply_rounding(
            regression_model.regression_summary_table(result),
            rounding if rounding is not None else self._rounding,
        )

    @handle_errors
    def get_hausman_test(
        self,
        independent_tickers: str | list[str] | None = None,
        independent_column: str | None = None,
        dependent_tickers: str | list[str] | None = None,
        period: str | None = None,
        column: str = "Return",
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series:
        """
        Calculate the Hausman specification test comparing a Fixed Effects and a
        Random Effects fit of `dependent_tickers` (the panel of entities) on a
        regressor `x`.

        Also known as: Hausman specification test, fixed versus random effects test.

        Random Effects is more efficient than Fixed Effects but relies on entity
        effects being uncorrelated with the regressor(s) -- if that assumption is
        violated, Random Effects is inconsistent while Fixed Effects remains
        consistent regardless. See `panel_data_model.get_hausman_test` for the full
        formula and references, and `get_fixed_effects`/`get_random_effects` for
        how the two models being compared are estimated (including how
        `independent_tickers`/`independent_column` shape the panel, and why
        `independent_column` is generally the more reliable choice here).

        The formula is as follows:

        - H = (β̂_FE - β̂_RE)'·[Var(β̂_FE) - Var(β̂_RE)]⁻¹·(β̂_FE - β̂_RE), over the k slope coefficients
          the two models share (a pseudo-inverse when the difference is singular, and H floored at 0)
        - H is chi-squared with k degrees of freedom; Random Effects is rejected when the p-value is
          below 0.05
        - β̂_FE comes from `get_fixed_effects` with entity effects only and β̂_RE from `get_random_effects`

        Args:
            independent_tickers (str | list[str] | None, optional): The factor
            ticker(s), whose `column` values are broadcast identically to every
            entity at each date. Mutually exclusive with `independent_column`.
            independent_column (str | None, optional): A different historical data
            column, taken per-entity from each of `dependent_tickers`' own data, to
            use as the regressor. Mutually exclusive with `independent_tickers`.
            dependent_tickers (str | list[str] | None, optional): The panel of
            entity tickers to explain. Defaults to None, meaning every ticker in
            the `Toolkit` instance (other than `independent_tickers`, if given).
            period (str, optional): The data frequency (daily, weekly, monthly,
            quarterly, or yearly). Defaults to "daily".
            column (str, optional): The dependent variable's historical data
            column. Defaults to "Return".
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
            entities. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the
            results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: The Hausman statistic, its degrees of freedom, its p-value,
            and whether Fixed Effects is preferred over Random Effects at the 5%
            level.

        Raises:
            ValueError: If both or neither of `independent_tickers`/
            `independent_column` are given, or (see `panel_data_model.
            get_hausman_test`) if there are not enough observations or entities to
            fit either model, or the two estimators share no common regressors to
            compare.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "AMZN"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_hausman_test(independent_column="Volume", period="weekly")
        ```

        Which returns:

        |                           |   Value |
        |:--------------------------|--------:|
        | Hausman Statistic         |  0.5866 |
        | Degrees of Freedom        |  1      |
        | P-Value                   |  0.4438 |
        | Prefer Fixed Effects (5%) |  0      |
        """
        y_panel, x_panel, dependent_tickers = self._get_panel_data(
            independent_tickers,
            independent_column,
            dependent_tickers,
            period,
            column,
            include_benchmark=include_benchmark,
        )

        result = panel_data_model.get_hausman_test(y_panel, x_panel)

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )
