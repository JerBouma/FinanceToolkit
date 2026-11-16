"""Hypothesis and Specification Tests Module"""

__docformat__ = "google"

import numpy as np
import pandas as pd

from financetoolkit.econometrics.base.base_controller import EconometricsBase
from financetoolkit.econometrics.regression import regression_model
from financetoolkit.econometrics.testing import (
    hypothesis_testing_model,
    specification_tests_model,
)
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import apply_rounding

# ruff: noqa: E501


@filter_shown_columns
@track_sources
class TestingEconometrics(EconometricsBase):
    """
    Hypothesis tests (two-sample t, F, likelihood ratio, Wald, Hausman-Wu) and specification tests (Breusch-Pagan, White, Durbin-Watson, VIF, Ramsey RESET, Chow).
    """

    def _fit_ols_result(
        self,
        dependent_ticker: str,
        independent_tickers: str | list[str],
        period: str | None,
        column: str,
        add_constant: bool,
    ) -> dict:
        """
        Shared helper for the hypothesis-testing methods below -- fits an OLS model
        via `regression_model.get_ols` and returns the raw (unrounded) regression
        result dict, rather than the rounded coefficient table `get_ols` returns,
        since the F-test, Wald test and Likelihood Ratio test all operate directly
        on the result dict's residuals/coefficients/covariance matrix.
        """
        period = period if period else "daily"
        independent_tickers = (
            [independent_tickers]
            if isinstance(independent_tickers, str)
            else independent_tickers
        )
        returns = self._get_series_column(
            period, column, [dependent_ticker, *independent_tickers]
        )

        return regression_model.get_ols(
            returns[dependent_ticker],
            returns[independent_tickers],
            add_constant=add_constant,
        )

    @handle_errors
    def get_two_sample_t_test(
        self,
        period: str | None = None,
        column: str = "Return",
        equal_variance: bool = False,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate a two-sample t-test for a difference in mean `column` between every
        unordered pair of tickers in the Toolkit instance.

        The test checks the null hypothesis that two samples come from distributions with
        the same mean, against the two-sided alternative that the means differ. Applied to
        returns it answers whether one asset has had a genuinely different average return
        than another, or whether the difference could be down to chance. By default this is
        Welch's t-test, which does not assume the two tickers have the same variance.

        The formula is as follows:

        - Welch (the default): t = (mean_a - mean_b) / sqrt(s_a^2 / n_a + s_b^2 / n_b)
        - Welch-Satterthwaite: df = (s_a^2 / n_a + s_b^2 / n_b)^2 / ((s_a^2 / n_a)^2 / (n_a - 1)
          + (s_b^2 / n_b)^2 / (n_b - 1))
        - Student (equal_variance=True): t = (mean_a - mean_b) / (s_p·sqrt(1 / n_a + 1 / n_b)), with
          s_p^2 = ((n_a - 1)·s_a^2 + (n_b - 1)·s_b^2) / (n_a + n_b - 2) and df = n_a + n_b - 2
        - s_a^2 and s_b^2 are sample variances (ddof=1) of the two tickers' `column` (Return by
          default), with a two-sided p-value

        Setting `equal_variance=True` pools the two variances instead (Student's t-test), which is
        slightly more powerful but only valid when the variances are truly equal. Welch's
        version is the safer default since assets rarely share the same volatility. A low
        p-value (e.g. below 0.05) means the difference in means is statistically
        significant.

        For more information about the method, see the following papers:

        - Student (1908). "The Probable Error of a Mean." Biometrika, 6(1), 1-25.
        - Welch, B.L. (1947). "The Generalization of 'Student's' Problem when Several
        Different Population Variances are Involved." Biometrika, 34(1/2), 28-35.

        Also known as: independent samples t-test, Welch's t-test (default), Student's
        t-test (`equal_variance=True`).

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to compare. Defaults to "Return".
            equal_variance (bool, optional): Whether to assume the two samples share a common
            variance (Student's pooled t-test) instead of Welch's (unequal-variance) t-test.
            Defaults to False.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
            tickers paired up. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: One row per unordered `(Ticker A, Ticker B)` pair, with the t-statistic,
            degrees of freedom, its p-value, and the two sample means.

        Notes:
        - The two return series are compared as independent samples (not paired by date), which is
        the standard use of a two-sample t-test -- if a date-by-date, paired comparison is needed
        instead, take the difference of the two return series and run a one-sample test on it.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["MSFT", "ORCL", "SAP"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_two_sample_t_test()
        ```

        Which returns:

        |                  |   T-Statistic |   Degrees of Freedom |   P-Value |   Mean A |   Mean B |
        |:-----------------|--------------:|---------------------:|----------:|---------:|---------:|
        | ('MSFT', 'ORCL') |       -0.528  |              2176.24 |    0.5975 |   0.0008 |   0.0012 |
        | ('MSFT', 'SAP')  |        0.1389 |              2504.05 |    0.8895 |   0.0008 |   0.0007 |
        | ('ORCL', 'SAP')  |        0.6326 |              2210.92 |    0.5271 |   0.0012 |   0.0007 |
        """
        period = period if period else "daily"
        returns = self._get_series_column(period, column)

        rows = {
            (ticker_a, ticker_b): hypothesis_testing_model.get_two_sample_t_test(
                returns[ticker_a], returns[ticker_b], equal_variance=equal_variance
            )
            for ticker_a, ticker_b in self._all_unordered_pairs(
                include_benchmark=include_benchmark
            )
        }

        # Transposing rows that mix numbers and a boolean leaves object columns, which
        # pandas does not round, so the numeric columns are restored first.
        result = pd.DataFrame(rows).T.infer_objects()
        result.index = result.index.set_names(["Ticker A", "Ticker B"])

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_f_test(
        self,
        dependent_ticker: str,
        restricted_independent_tickers: str | list[str],
        unrestricted_independent_tickers: str | list[str],
        period: str | None = None,
        column: str = "Return",
        add_constant: bool = True,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series:
        """
        Calculate a nested-model F-test for the joint significance of the regressors in
        `unrestricted_independent_tickers` that are not already in
        `restricted_independent_tickers`.

        Two OLS regressions of `dependent_ticker` are fitted internally: a "restricted" one on
        `restricted_independent_tickers` and an "unrestricted" one that adds the extra
        tickers. The test checks the null hypothesis that the `q` extra tickers are jointly
        zero, i.e. that adding them does not explain the dependent ticker any better, by
        comparing how much the Residual Sum of Squares (RSS) drops when they are added.

        The formula is as follows:

        - F = ((RSS_restricted - RSS_unrestricted) / q) / (RSS_unrestricted / (n - k_unrestricted))
        - Both models are OLS regressions of the dependent ticker's `column` (Return by default),
          with a constant when add_constant=True

        Where `q = k_unrestricted - k_restricted` is the number of added regressors, `n` the
        number of observations and `k_unrestricted` the number of parameters of the larger
        model. Under the null, `F` follows an F(q, n - k_unrestricted) distribution. A low
        p-value means the extra tickers meaningfully improve the fit as a group, even when
        none of them is significant individually. `get_likelihood_ratio_test` is the
        asymptotic Maximum Likelihood counterpart.

        Also known as: nested F-test, restricted vs. unrestricted F-test, partial F-test.

        Args:
            dependent_ticker (str): The dependent (predicted) asset.
            restricted_independent_tickers (str | list[str]): The independent asset(s) in the
            restricted (smaller) model.
            unrestricted_independent_tickers (str | list[str]): The independent asset(s) in the
            unrestricted (larger) model -- must be a superset of `restricted_independent_tickers`.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly).
            Defaults to "daily".
            column (str, optional): The historical data column to regress on. Defaults to "Return".
            add_constant (bool, optional): Whether to include an intercept in both models. Defaults
            to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: The F-statistic, its numerator/denominator degrees of freedom, its p-value,
            and whether the added regressors are jointly significant at the 5% level.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_f_test(
            "AAPL", "MSFT", ["MSFT", "Benchmark"], period="weekly"
        )
        ```

        Which returns:

        |                          |       0 |
        |:-------------------------|--------:|
        | F-Statistic              | 104.748 |
        | Df Numerator             |   1     |
        | Df Denominator           | 257     |
        | P-Value                  |   0     |
        | Reject Restrictions (5%) |   1     |
        """
        restricted_result = self._fit_ols_result(
            dependent_ticker,
            restricted_independent_tickers,
            period=period,
            column=column,
            add_constant=add_constant,
        )
        unrestricted_result = self._fit_ols_result(
            dependent_ticker,
            unrestricted_independent_tickers,
            period=period,
            column=column,
            add_constant=add_constant,
        )

        result = hypothesis_testing_model.get_f_test(
            restricted_result, unrestricted_result
        )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_likelihood_ratio_test(
        self,
        dependent_ticker: str,
        restricted_independent_tickers: str | list[str],
        unrestricted_independent_tickers: str | list[str],
        period: str | None = None,
        column: str = "Return",
        add_constant: bool = True,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series:
        """
        Calculate a nested-model Likelihood Ratio (LR) test, the Maximum Likelihood analogue
        of `get_f_test`, for the joint significance of the regressors in
        `unrestricted_independent_tickers` that are not already in
        `restricted_independent_tickers`.

        As with `get_f_test`, a "restricted" and an "unrestricted" OLS regression of
        `dependent_ticker` are fitted internally and the null hypothesis is that the `q` extra
        tickers are jointly zero. Instead of an F-statistic, the two models are compared
        through the ratio of their likelihoods, which for linear models with normal errors
        reduces to a function of the two Residual Sums of Squares (RSS).

        The formula is as follows:

        - LR = -2·(ln L_restricted - ln L_unrestricted) = n * ln(RSS_restricted / RSS_unrestricted)
        - Both models are OLS regressions of the dependent ticker's `column` (Return by default),
          with a constant when add_constant=True

        Under the null this is asymptotically chi-squared distributed with `q` degrees of
        freedom. Where the F-test is exact in small samples under the classical assumptions,
        the LR test relies on a large-sample approximation: both agree for long return
        histories and can differ somewhat for short ones. A low p-value means the extra
        tickers meaningfully improve the model.

        For more information about the method, see the following paper:

        - Wilks, S.S. (1938). "The Large-Sample Distribution of the Likelihood Ratio for
        Testing Composite Hypotheses." Annals of Mathematical Statistics, 9(1), 60-62.

        Also known as: LR test, Wilks' likelihood ratio test.

        Args:
            dependent_ticker (str): The dependent (predicted) asset.
            restricted_independent_tickers (str | list[str]): The independent asset(s) in the
            restricted (smaller) model.
            unrestricted_independent_tickers (str | list[str]): The independent asset(s) in the
            unrestricted (larger) model -- must be a superset of `restricted_independent_tickers`.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly).
            Defaults to "daily".
            column (str, optional): The historical data column to regress on. Defaults to "Return".
            add_constant (bool, optional): Whether to include an intercept in both models. Defaults
            to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: The LR statistic, its degrees of freedom, its p-value, and whether the added
            regressors are jointly significant at the 5% level.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_likelihood_ratio_test(
            "AAPL", "MSFT", ["MSFT", "Benchmark"], period="weekly"
        )
        ```

        Which returns:

        |                          |       0 |
        |:-------------------------|--------:|
        | LR Statistic             | 88.8863 |
        | Degrees of Freedom       |  1      |
        | P-Value                  |  0      |
        | Reject Restrictions (5%) |  1      |
        """
        restricted_result = self._fit_ols_result(
            dependent_ticker,
            restricted_independent_tickers,
            period=period,
            column=column,
            add_constant=add_constant,
        )
        unrestricted_result = self._fit_ols_result(
            dependent_ticker,
            unrestricted_independent_tickers,
            period=period,
            column=column,
            add_constant=add_constant,
        )

        result = hypothesis_testing_model.get_likelihood_ratio_test(
            restricted_result, unrestricted_result
        )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_wald_test(
        self,
        restriction_matrix: pd.DataFrame | np.ndarray | list[list[float]],
        dependent_ticker: str | None = None,
        independent_tickers: str | list[str] | None = None,
        include_benchmark: bool = False,
        restriction_values: pd.Series | np.ndarray | list[float] | None = None,
        period: str | None = None,
        column: str = "Return",
        add_constant: bool = True,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series:
        """
        Calculate a Wald test of `q` general linear restriction(s) on the coefficients of an
        OLS regression of `dependent_ticker` on `independent_tickers`.

        A single OLS regression is fitted internally and the test checks the null hypothesis
        `H0: R @ beta = r`, where `R` (`restriction_matrix`) encodes `q` linear restrictions
        on the coefficients and `r` (`restriction_values`) holds their hypothesised values
        (zero by default). This generalises the t-test reported for each coefficient to
        combinations of coefficients, for example that two coefficients are jointly zero,
        that a coefficient equals a specific value (such as a beta of exactly 1), or that two
        tickers have the same coefficient.

        The formula is as follows:

        - W = (R @ beta_hat - r)' @ (R @ Cov(beta_hat) @ R')^-1 @ (R @ beta_hat - r)
        - F = W / q, F-distributed with (q, n - k) degrees of freedom
        - beta_hat and the classical Cov(beta_hat) = σ̂²·(X'X)⁻¹ come from the OLS regression of the
          dependent ticker's `column` (Return by default), with a constant when add_constant=True

        Which is asymptotically chi-squared distributed with `q` degrees of freedom. `W / q`
        is reported alongside it as an F-statistic for small samples; for a single
        restriction it equals the squared t-statistic of that coefficient. A low p-value
        means the data reject the restriction(s).

        For more information about the method, see the following paper:

        - Wald, A. (1943). "Tests of Statistical Hypotheses Concerning Several Parameters When
        the Number of Observations is Large." Transactions of the American Mathematical
        Society, 54(3), 426-482.

        Also known as: Wald chi-squared test.

        Args:
            restriction_matrix (pd.DataFrame | np.ndarray | list[list[float]]): The `(q, k)`
            restriction matrix `R`, one row per restriction, one column per coefficient in the
            same order as `add_constant` (if True, "Intercept" first) followed by the
            independent ticker(s), e.g. [[0, 1, 0], [0, 0, 1]].
            dependent_ticker (str | None, optional): The dependent (predicted) asset.
            Defaults to None, meaning the Toolkit instance's first ticker.
            independent_tickers (str | list[str] | None, optional): The independent
            (predictor) asset(s), in the order the restriction matrix's columns
            (after "Intercept", if `add_constant`) refer to them. Defaults to None,
            meaning every other ticker in the Toolkit instance besides
            `dependent_ticker`.
            include_benchmark (bool, optional): Whether to include "Benchmark" in the
            default independent ticker(s) (has no effect when independent_tickers is given
            explicitly). Defaults to False.
            restriction_values (pd.Series | np.ndarray | list[float] | None, optional): The
            length-`q` vector of hypothesized values. Defaults to None, i.e. all restrictions
            equal zero.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly).
            Defaults to "daily".
            column (str, optional): The historical data column to regress on. Defaults to "Return".
            add_constant (bool, optional): Whether to include an intercept. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: The Wald (chi-squared) statistic and its p-value, the small-sample F-statistic
            and its p-value, the number of restrictions, and whether they are rejected at the 5% level.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        # H0: the coefficients on both MSFT and Benchmark are jointly zero.
        toolkit.econometrics.get_wald_test(
            restriction_matrix=[[0, 1, 0], [0, 0, 1]],
            independent_tickers=["MSFT", "Benchmark"],
            period="weekly",
        )
        ```

        Which returns:

        |                          |       0 |
        |:-------------------------|--------:|
        | Wald Statistic (Chi2)    | 269.191 |
        | Chi2 P-Value             |   0     |
        | F-Statistic              | 134.595 |
        | F P-Value                |   0     |
        | Restrictions (q)         |   2     |
        | Reject Restrictions (5%) |   1     |
        """
        dependent_ticker, independent_tickers = self._resolve_dependent_independent(
            dependent_ticker=dependent_ticker,
            independent_tickers=independent_tickers,
            include_benchmark=include_benchmark,
        )

        result = self._fit_ols_result(
            dependent_ticker,
            independent_tickers,
            period=period,
            column=column,
            add_constant=add_constant,
        )

        restriction_matrix_values = (
            restriction_matrix.to_numpy()
            if isinstance(restriction_matrix, pd.DataFrame)
            else np.asarray(restriction_matrix)
        )
        restriction_values_array = (
            restriction_values.to_numpy()
            if isinstance(restriction_values, pd.Series)
            else restriction_values
        )

        wald_result = hypothesis_testing_model.get_wald_test(
            result, restriction_matrix_values, restriction_values_array
        )

        return apply_rounding(
            wald_result, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_hausman_wu_test(
        self,
        dependent_ticker: str,
        suspect_ticker: str,
        instrument_tickers: str | list[str],
        other_independent_tickers: str | list[str] | None = None,
        period: str | None = None,
        column: str = "Return",
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series:
        """
        Calculate a regression-based Hausman-Wu test for the endogeneity of `suspect_ticker`
        in a regression of `dependent_ticker` on `suspect_ticker` (and, optionally,
        `other_independent_tickers`), using `instrument_tickers` as instruments for
        `suspect_ticker`.

        OLS requires every regressor to be uncorrelated with the error term. When that fails,
        for example because of an omitted common driver, reverse causality or measurement
        error, the regressor is "endogenous" and OLS is biased; an instrumental-variables
        approach such as `get_iv_2sls` is then needed. This test checks whether that is the
        case before committing to a full IV fit, using two auxiliary OLS regressions:

        1. Regress `suspect_ticker` on the instruments (and the other tickers, if given) and
        keep the residuals: the part of `suspect_ticker` that the instruments do not explain.
        2. Regress `dependent_ticker` on `suspect_ticker`, those residuals (and the other
        tickers) and test whether the coefficient on the residuals is zero.

        If `suspect_ticker` is exogenous the residuals add no information, so their coefficient
        should be statistically zero. A low p-value is evidence that `suspect_ticker` is
        endogenous and that the OLS estimate is unreliable. The test is only as good as the
        instruments: they must be correlated with `suspect_ticker` but not with the error term.

        The formula is as follows:

        - Stage 1: Xₜ = π₀ + π'Zₜ + δ'Wₜ + vₜ, the OLS regression of `suspect_ticker` (X) on a constant,
          the instruments (Z) and `other_independent_tickers` (W), keeping the residuals v̂ₜ
        - Stage 2: yₜ = β₀ + β₁·Xₜ + ρ·v̂ₜ + γ'Wₜ + εₜ, the OLS regression of `dependent_ticker` (y)
        - t = ρ̂ / SE(ρ̂), with the classical OLS standard error and a Student-t distribution with n - k
          degrees of freedom; endogeneity is flagged when the p-value is below 0.05
        - All variables are the tickers' `column` (Return by default)

        For more information about the method, see the following papers:

        - Wu, D-M. (1973). "Alternative Tests of Independence Between Stochastic Regressors
        and Disturbances." Econometrica, 41(4), 733-750.
        - Hausman, J.A. (1978). "Specification Tests in Econometrics." Econometrica, 46(6),
        1251-1271.

        Also known as: Hausman test, Durbin-Wu-Hausman test, regression test for
        endogeneity.

        Args:
            dependent_ticker (str): The dependent (predicted) asset.
            suspect_ticker (str): The (possibly endogenous) asset being tested.
            instrument_tickers (str | list[str]): One or more instrument asset(s) for
            `suspect_ticker` -- assets correlated with `suspect_ticker` but assumed uncorrelated
            with `dependent_ticker`'s error term.
            other_independent_tickers (str | list[str] | None, optional): Any other (assumed
            exogenous) independent asset(s) to include. Defaults to None.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to regress on. Defaults to "Return".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: The coefficient on the first-stage residuals, its t-statistic and p-value, and
            whether `suspect_ticker` is flagged as endogenous at the 5% level.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["SMCI", "NVDA", "TSM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_hausman_wu_test(
            dependent_ticker="SMCI", suspect_ticker="NVDA", instrument_tickers="TSM"
        )
        ```

        Which returns:

        |                    |    Value |
        |:-------------------|---------:|
        | V-Hat Coefficient  |  -0.4021 |
        | T-Statistic        |  -2.3377 |
        | Degrees of Freedom | 246      |
        | P-Value            |   0.0202 |
        | Endogenous (5%)    |   1      |
        """
        period = period if period else "daily"
        instrument_tickers = (
            [instrument_tickers]
            if isinstance(instrument_tickers, str)
            else instrument_tickers
        )
        other_independent_tickers = (
            [other_independent_tickers]
            if isinstance(other_independent_tickers, str)
            else other_independent_tickers
        )

        returns = self._get_series_column(
            period,
            column,
            [
                dependent_ticker,
                suspect_ticker,
                *instrument_tickers,
                *(other_independent_tickers or []),
            ],
        )

        result = hypothesis_testing_model.get_hausman_wu_test(
            returns[dependent_ticker],
            returns[suspect_ticker],
            returns[instrument_tickers],
            (
                returns[other_independent_tickers]
                if other_independent_tickers is not None
                else None
            ),
        )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    def _get_ols_result(
        self,
        dependent_ticker: str,
        independent_tickers: str | list[str],
        period: str | None,
        column: str,
        add_constant: bool,
    ) -> dict:
        """
        Shared helper that fits the OLS regression of `dependent_ticker` on
        `independent_tickers` used internally by every post-estimation diagnostic
        below (`get_breusch_pagan_test`, `get_white_test`, `get_durbin_watson_test`,
        `get_ramsey_reset_test`, `get_chow_test`) -- each of these tests operates on
        an already-fitted regression result dict (see `specification_tests_model`),
        so the controller's job is simply to assemble that same fit `get_ols` itself
        produces before handing it off to the requested diagnostic.
        """
        period = period if period else "daily"
        independent_tickers = (
            [independent_tickers]
            if isinstance(independent_tickers, str)
            else independent_tickers
        )
        returns = self._get_series_column(
            period, column, [dependent_ticker, *independent_tickers]
        )

        return regression_model.get_ols(
            returns[dependent_ticker],
            returns[independent_tickers],
            add_constant=add_constant,
        )

    @handle_errors
    def get_breusch_pagan_test(
        self,
        dependent_ticker: str | None = None,
        independent_tickers: str | list[str] | None = None,
        include_benchmark: bool = False,
        period: str | None = None,
        column: str = "Return",
        add_constant: bool = True,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series:
        """
        Calculate the Breusch-Pagan test for heteroskedasticity of a regression of
        `dependent_ticker` on `independent_tickers`.

        OLS standard errors assume that the residual variance is constant
        (homoskedasticity). The test regresses the squared residuals of the regression on the
        original independent tickers and checks whether they explain any of the residual
        variance, by testing whether the R-squared of that auxiliary regression is
        significantly different from zero.

        The formula is as follows:

        - Auxiliary regression: e_t^2 = a_0 + a' * x_t + v_t, with e_t the OLS residuals and x_t the
          independent tickers' `column` (Return by default)
        - LM = n * R^2_auxiliary (Koenker's studentized form, the statsmodels default)

        This is the studentized statistic of Koenker (1981), which does not assume normally
        distributed errors. It is asymptotically chi-squared distributed with `k - 1` degrees
        of freedom under the null hypothesis of homoskedasticity, where `k` is the number of parameters of the
        auxiliary regression. A low p-value means the residual variance moves with the
        independent tickers (for example, errors grow when the market moves more), so the
        regression's standard errors, t-statistics and p-values are unreliable even though
        the coefficients themselves remain unbiased. Remedies are a robust `cov_type` in
        `get_ols` or `get_wls`. `get_white_test` is a more general alternative that also
        catches non-linear forms of heteroskedasticity.

        For more information about the method, see the following paper:

        - Breusch, T.S., & Pagan, A.R. (1979). "A Simple Test for Heteroscedasticity and
        Random Coefficient Variation." Econometrica, 47(5), 1287-1294.
        - Koenker, R. (1981). "A Note on Studentizing a Test for Heteroscedasticity."
        Journal of Econometrics, 17(1), 107-112.

        Also known as: BP test, Breusch-Pagan-Godfrey test.

        Args:
            dependent_ticker (str | None, optional): The dependent (predicted) asset.
            Defaults to None, meaning the Toolkit instance's first ticker.
            independent_tickers (str | list[str] | None, optional): The independent
            (predictor) asset(s). Defaults to None, meaning every other ticker in
            the Toolkit instance besides `dependent_ticker`.
            include_benchmark (bool, optional): Whether to include "Benchmark" in the
            default independent ticker(s) (has no effect when independent_tickers is given
            explicitly). Defaults to False.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly).
            Defaults to "daily".
            column (str, optional): The historical data column to regress on. Defaults to "Return".
            add_constant (bool, optional): Whether to include an intercept in the underlying
            regression. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: The Breusch-Pagan LM statistic, its p-value, and whether homoskedasticity is
            rejected at the 5% level.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["CAT", "DE"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        # CAT (the first ticker) is dependent; DE is independent
        toolkit.econometrics.get_breusch_pagan_test()
        ```

        Which returns:

        |                              |   Value |
        |:-----------------------------|--------:|
        | Breusch-Pagan Statistic      |  0.0129 |
        | P-Value                      |  0.9096 |
        | Reject Homoskedasticity (5%) |  0      |
        """
        dependent_ticker, independent_tickers = self._resolve_dependent_independent(
            dependent_ticker=dependent_ticker,
            independent_tickers=independent_tickers,
            include_benchmark=include_benchmark,
        )

        result = self._get_ols_result(
            dependent_ticker, independent_tickers, period, column, add_constant
        )

        test_result = specification_tests_model.get_breusch_pagan_test(result)

        return apply_rounding(
            test_result, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_white_test(
        self,
        dependent_ticker: str | None = None,
        independent_tickers: str | list[str] | None = None,
        include_benchmark: bool = False,
        period: str | None = None,
        column: str = "Return",
        add_constant: bool = True,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series:
        """
        Calculate White's test for heteroskedasticity of a regression of `dependent_ticker` on
        `independent_tickers`.

        A more general version of `get_breusch_pagan_test`. Instead of regressing the squared
        residuals on only the original independent tickers, the auxiliary regression uses
        each ticker, its square and every pairwise cross-product.

        The formula is as follows:

        - e_t^2 = a_0 + SUM_i(a_i * x_it) + SUM_i(b_i * x_it^2) + SUM_{i<j}(c_ij * x_it * x_jt) + v_t
        - LM = n * R^2_auxiliary, chi-squared with as many degrees of freedom as the auxiliary
          regression has terms besides a_0
        - e_t are the OLS residuals and x_it the independent tickers' `column` (Return by default)

        The null hypothesis is homoskedasticity (constant residual variance). A low p-value means the
        regression's standard errors are unreliable and a robust `cov_type` should be used.
        Because it also detects non-linear patterns in the residual variance, White's test
        makes fewer assumptions than Breusch-Pagan, at the cost of power when there are many
        independent tickers, since the number of auxiliary terms grows quadratically.

        For more information about the method, see the following paper:

        - White, H. (1980). "A Heteroskedasticity-Consistent Covariance Matrix Estimator and a
        Direct Test for Heteroskedasticity." Econometrica, 48(4), 817-838.

        Also known as: White's general test.

        Args:
            dependent_ticker (str | None, optional): The dependent (predicted) asset.
            Defaults to None, meaning the Toolkit instance's first ticker.
            independent_tickers (str | list[str] | None, optional): The independent
            (predictor) asset(s). Defaults to None, meaning every other ticker in
            the Toolkit instance besides `dependent_ticker`.
            include_benchmark (bool, optional): Whether to include "Benchmark" in the
            default independent ticker(s) (has no effect when independent_tickers is given
            explicitly). Defaults to False.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly).
            Defaults to "daily".
            column (str, optional): The historical data column to regress on. Defaults to "Return".
            add_constant (bool, optional): Whether to include an intercept in the underlying
            regression. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: White's LM statistic, its p-value, and whether homoskedasticity is rejected at
            the 5% level.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["GS", "MS"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        # GS (the first ticker) is dependent; MS is independent
        toolkit.econometrics.get_white_test()
        ```

        Which returns:

        |                              |   Value |
        |:-----------------------------|--------:|
        | White Statistic              | 11.5102 |
        | P-Value                      |  0.0032 |
        | Reject Homoskedasticity (5%) |  1      |
        """
        dependent_ticker, independent_tickers = self._resolve_dependent_independent(
            dependent_ticker=dependent_ticker,
            independent_tickers=independent_tickers,
            include_benchmark=include_benchmark,
        )

        result = self._get_ols_result(
            dependent_ticker, independent_tickers, period, column, add_constant
        )

        test_result = specification_tests_model.get_white_test(result)

        return apply_rounding(
            test_result, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_durbin_watson_test(
        self,
        dependent_ticker: str | None = None,
        independent_tickers: str | list[str] | None = None,
        include_benchmark: bool = False,
        period: str | None = None,
        column: str = "Return",
        add_constant: bool = True,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series:
        """
        Calculate the Durbin-Watson statistic for first-order autocorrelation in the
        residuals of a regression of `dependent_ticker` on `independent_tickers`.

        OLS standard errors assume that the residuals of consecutive periods are
        uncorrelated. The Durbin-Watson statistic compares each residual with the one before
        it.

        The formula is as follows:

        - DW = SUM_{t=2}^{n}((e_t - e_(t-1))^2) / SUM_{t=1}^{n}(e_t^2)
        - e_t are the residuals of the OLS regression of the dependent ticker's `column` (Return by
          default) on the independent tickers' `column`
        - Interpretation: DW < 1.5 positive autocorrelation likely, DW > 2.5 negative autocorrelation
          likely, otherwise no strong evidence

        The statistic ranges from 0 to 4. A value near 2 means no first-order
        autocorrelation; values below 2 point to positive autocorrelation (consecutive
        residuals tend to share the same sign, often a sign of an omitted trend or lagged
        effect) and values above 2 to negative autocorrelation (residuals tend to alternate in
        sign). Like heteroskedasticity, autocorrelation leaves the coefficients unbiased but
        makes the standard errors unreliable; a "HAC" `cov_type` in `get_ols`, or `get_gls`
        with an AR(1) `omega`, are common remedies. There is no exact p-value: the returned
        interpretation is a rule of thumb (below 1.5 is likely positive, above 2.5 likely
        negative autocorrelation).

        For more information about the method, see the following paper:

        - Durbin, J., & Watson, G.S. (1950). "Testing for Serial Correlation in Least Squares
        Regression I." Biometrika, 37(3/4), 409-428.

        Also known as: DW statistic.

        Args:
            dependent_ticker (str | None, optional): The dependent (predicted) asset.
            Defaults to None, meaning the Toolkit instance's first ticker.
            independent_tickers (str | list[str] | None, optional): The independent
            (predictor) asset(s). Defaults to None, meaning every other ticker in
            the Toolkit instance besides `dependent_ticker`.
            include_benchmark (bool, optional): Whether to include "Benchmark" in the
            default independent ticker(s) (has no effect when independent_tickers is given
            explicitly). Defaults to False.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly).
            Defaults to "daily".
            column (str, optional): The historical data column to regress on. Defaults to "Return".
            add_constant (bool, optional): Whether to include an intercept in the underlying
            regression. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: The Durbin-Watson statistic and an approximate interpretation flag.

        Notes:
        - Unlike the other tests in this module, this does not carry a formal p-value/reject flag at a
        stated significance level -- see `specification_tests_model.get_durbin_watson_test` for why.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["HD", "LOW"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        # HD (the first ticker) is dependent; LOW is independent
        toolkit.econometrics.get_durbin_watson_test()
        ```

        Which returns:

        |                         | Value              |
        |:------------------------|:-------------------|
        | Durbin-Watson Statistic | 1.8584             |
        | Interpretation          | No Strong Evidence |
        """
        dependent_ticker, independent_tickers = self._resolve_dependent_independent(
            dependent_ticker=dependent_ticker,
            independent_tickers=independent_tickers,
            include_benchmark=include_benchmark,
        )

        result = self._get_ols_result(
            dependent_ticker, independent_tickers, period, column, add_constant
        )

        test_result = specification_tests_model.get_durbin_watson_test(result)

        # The 'Interpretation' entry is a string, so only the statistic is rounded.
        rounding_value = rounding if rounding is not None else self._rounding
        if rounding_value is not None:
            test_result["Durbin-Watson Statistic"] = round(
                test_result["Durbin-Watson Statistic"], rounding_value
            )

        return test_result

    @handle_errors
    def get_vif(
        self,
        period: str | None = None,
        column: str = "Return",
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series:
        """
        Calculate the Variance Inflation Factor (VIF) of every ticker in the Toolkit
        instance, treated as regressors against one another.

        Multicollinearity, where one regressor can (nearly) be predicted from the others, does
        not bias regression coefficients but inflates their standard errors. Individual
        coefficients can then look insignificant, and swing wildly between samples, even when
        the regressors jointly explain the dependent variable well. VIF measures this for
        each ticker by regressing it on all the other tickers.

        The formula is as follows:

        - VIF_i = 1 / (1 - R^2_i)
        - R^2_i is the R-squared of the OLS regression of ticker i's `column` (Return by default) on a
          constant and every other ticker's `column`

        Where `R^2_i` is the R-squared of regressing ticker `i` on all other tickers. A VIF of
        1 means the ticker is uncorrelated with the others; a VIF of `V` means the variance of
        its coefficient is `V` times larger than it would be without multicollinearity. As a
        rule of thumb, a VIF above 10 (or above 5 for a stricter cutoff) signals problematic
        multicollinearity. This is common with assets that move closely together, such as
        several stocks from the same sector, and is worth checking before using them together
        as independent tickers in `get_ols` or the other regression methods.

        Also known as: VIF.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to use. Defaults to "Return".
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
            regressors tested. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: The VIF of each asset, indexed by ticker.

        Notes:
        - Unlike the other methods in this module, `get_vif` has no `dependent_ticker` -- VIF is a
        property of the regressors alone, independent of any particular dependent variable.
        - `VIF > 10` is the conventional rule-of-thumb threshold for concerning multicollinearity.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["GOOGL", "META", "AMZN"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_vif()
        ```

        Which returns:

        |       |    VIF |
        |:------|-------:|
        | GOOGL | 1.4431 |
        | META  | 1.8234 |
        | AMZN  | 2.0804 |
        """
        period = period if period else "daily"
        tickers = self._get_tickers(include_benchmark=include_benchmark)
        returns = self._get_series_column(period, column, tickers)

        test_result = specification_tests_model.get_vif(returns[tickers])

        return apply_rounding(
            test_result, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_ramsey_reset_test(
        self,
        dependent_ticker: str | None = None,
        independent_tickers: str | list[str] | None = None,
        include_benchmark: bool = False,
        period: str | None = None,
        column: str = "Return",
        add_constant: bool = True,
        power: int = 3,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series:
        """
        Calculate Ramsey's RESET test for functional form misspecification of a regression of
        `dependent_ticker` on `independent_tickers`.

        If the true relationship is non-linear, or a relevant transformation or variable is
        missing, the fitted values of a linear regression still carry information about that
        missing piece. RESET tests for this by adding powers of the fitted values (`fitted^2`,
        ..., `fitted^power`) to the regression and checking whether they are jointly
        significant with an F-test that compares the original (restricted) and augmented
        (unrestricted) model.

        The formula is as follows:

        - Augmented model: y_t = x_t' * beta + SUM_{j=2}^{power}(gamma_j * yhat_t^j) + u_t, with yhat_t
          the fitted values of the original OLS regression and `power` = 3 by default
        - F = ((RSS_r - RSS_u) / q) / (RSS_u / (n - k_u))

        Where `RSS_r` and `RSS_u` are the residual sums of squares of both models, `q` is the
        number of added terms (`power - 1`), `n` the number of observations and `k_u` the
        number of parameters of the augmented model. Under the null hypothesis that the linear
        form is correct, `F` follows an F(q, n - k_u) distribution. A low p-value suggests the
        model is missing something, for example a squared or interaction term or a log
        transformation, without saying what; it is a signal to reconsider the specification,
        not a prescription.

        For more information about the method, see the following paper:

        - Ramsey, J.B. (1969). "Tests for Specification Errors in Classical Linear
        Least-Squares Regression Analysis." Journal of the Royal Statistical Society, Series
        B, 31(2), 350-371.

        Also known as: RESET test, Ramsey RESET.

        Args:
            dependent_ticker (str | None, optional): The dependent (predicted) asset.
            Defaults to None, meaning the Toolkit instance's first ticker.
            independent_tickers (str | list[str] | None, optional): The independent
            (predictor) asset(s). Defaults to None, meaning every other ticker in
            the Toolkit instance besides `dependent_ticker`.
            include_benchmark (bool, optional): Whether to include "Benchmark" in the
            default independent ticker(s) (has no effect when independent_tickers is given
            explicitly). Defaults to False.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly).
            Defaults to "daily".
            column (str, optional): The historical data column to regress on. Defaults to "Return".
            add_constant (bool, optional): Whether to include an intercept in the underlying
            regression. Defaults to True.
            power (int, optional): The highest power of the fitted values to add. Defaults to 3.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: The RESET F-statistic, its p-value, and whether correct specification is
            rejected at the 5% level.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NKE", "ADS.DE"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        # NKE (the first ticker) is dependent; ADS.DE is independent
        toolkit.econometrics.get_ramsey_reset_test()
        ```

        Which returns:

        |                                   |   Value |
        |:----------------------------------|--------:|
        | RESET F-Statistic                 |  1.4847 |
        | P-Value                           |  0.2285 |
        | Reject Correct Specification (5%) |  0      |
        """
        dependent_ticker, independent_tickers = self._resolve_dependent_independent(
            dependent_ticker=dependent_ticker,
            independent_tickers=independent_tickers,
            include_benchmark=include_benchmark,
        )

        result = self._get_ols_result(
            dependent_ticker, independent_tickers, period, column, add_constant
        )

        test_result = specification_tests_model.get_ramsey_reset_test(
            result, power=power
        )

        return apply_rounding(
            test_result, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_chow_test(
        self,
        break_date: str,
        dependent_ticker: str | None = None,
        independent_tickers: str | list[str] | None = None,
        include_benchmark: bool = False,
        period: str | None = None,
        column: str = "Return",
        add_constant: bool = True,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series:
        """
        Calculate the Chow test for a structural break at `break_date` in a regression of
        `dependent_ticker` on `independent_tickers`.

        The sample is split at `break_date` into a "before" and "after" period, a regression
        is fitted on each part separately and on the full sample, and an F-test checks whether
        allowing the coefficients to differ between the two periods significantly reduces the
        residual sum of squares.

        The formula is as follows:

        - F = ((RSS_pooled - (RSS_1 + RSS_2)) / k) / ((RSS_1 + RSS_2) / (n - 2k))
        - The second sub-period starts at the first date on or after `break_date`, and `k` includes
          the constant when add_constant=True

        Where `RSS_pooled` is the residual sum of squares of the full-sample regression,
        `RSS_1` and `RSS_2` those of the two sub-periods, `k` the number of parameters and `n`
        the total number of observations. Under the null hypothesis of no structural break
        (the same coefficients apply throughout), `F` follows an F(k, n - 2k) distribution. A
        low p-value means the relationship changed at `break_date`, for example an asset's
        beta shifting after a merger, a regulatory change or a market regime change, so a
        single regression over the full sample is misspecified. The break date must be chosen
        in advance; testing many candidate dates and keeping the most significant overstates
        the evidence.

        For more information about the method, see the following paper:

        - Chow, G.C. (1960). "Tests of Equality Between Sets of Coefficients in Two Linear
        Regressions." Econometrica, 28(3), 591-605.

        Also known as: Chow breakpoint test.

        Args:
            break_date (str): The date (e.g. "2021-06-30") at which to split the sample -- all
            periods starting on or after this date form the "after" sub-sample, everything
            before it forms the "before" sub-sample.
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
            add_constant (bool, optional): Whether to include an intercept in the underlying
            regression(s). Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: The Chow F-statistic, its p-value, and whether no structural break is rejected
            at the 5% level.

        Raises:
            ValueError: If `break_date` does not leave enough observations on either side of the
            split to estimate the regression's parameters.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NVDA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        # NVDA is dependent and the Benchmark independent, with a break at the launch of ChatGPT
        toolkit.econometrics.get_chow_test(break_date="2022-11-30", include_benchmark=True)
        ```

        Which returns:

        |                                 |   Value |
        |:--------------------------------|--------:|
        | Chow F-Statistic                |  0.811  |
        | P-Value                         |  0.4447 |
        | Reject No Structural Break (5%) |  0      |
        """
        dependent_ticker, independent_tickers = self._resolve_dependent_independent(
            dependent_ticker=dependent_ticker,
            independent_tickers=independent_tickers,
            include_benchmark=include_benchmark,
        )

        period = period if period else "daily"
        returns = self._get_series_column(
            period, column, [dependent_ticker, *independent_tickers]
        )

        result_full = regression_model.get_ols(
            returns[dependent_ticker],
            returns[independent_tickers],
            add_constant=add_constant,
        )

        index = returns.index
        timestamps = (
            index.to_timestamp()
            if isinstance(index, pd.PeriodIndex)
            else pd.DatetimeIndex(index)
        )
        break_index = int(
            timestamps.searchsorted(pd.Timestamp(break_date).to_datetime64())
        )

        test_result = specification_tests_model.get_chow_test(
            result_full,
            returns[independent_tickers],
            returns[dependent_ticker],
            break_index=break_index,
            add_constant=add_constant,
        )

        return apply_rounding(
            test_result, rounding if rounding is not None else self._rounding
        )
