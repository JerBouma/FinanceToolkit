"""Causal Inference Module"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.econometrics.base.base_controller import (
    BENCHMARK_TICKER,
    EconometricsBase,
)
from financetoolkit.econometrics.causal import causal_inference_model, event_study_model
from financetoolkit.econometrics.regression import regression_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import apply_rounding

# ruff: noqa: E501


@filter_shown_columns
@track_sources
class CausalEconometrics(EconometricsBase):
    """
    Causal designs: instrumental variables, difference-in-differences, regression discontinuity, propensity score matching, synthetic control and the event study.
    """

    @handle_errors
    def get_iv_2sls(
        self,
        dependent_ticker: str,
        endogenous_ticker: str | list[str],
        instrument_tickers: str | list[str],
        exogenous_tickers: str | list[str] | None = None,
        period: str | None = None,
        column: str = "Return",
        add_constant: bool = True,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Fit an Instrumental Variables regression via Two-Stage Least Squares (2SLS) of
        `dependent_ticker` on `endogenous_ticker`, instrumented by `instrument_tickers`.

        Also known as: IV, 2SLS, IV-2SLS.

        For more information about the method, and why plain OLS on an asset suspected of
        reverse-causality/omitted-confounder bias against another asset is unreliable, see
        `causal_inference_model.get_iv_2sls`.

        A natural use case in a multi-asset return panel: suppose `endogenous_ticker`'s
        return is suspected to be simultaneously determined together with
        `dependent_ticker`'s return (e.g. two closely related assets that react to each
        other intraday, or one asset's return partly reflects news about the other) --
        plain OLS of one on the other is then biased. An `instrument_tickers` asset that
        moves `endogenous_ticker` for reasons unrelated to `dependent_ticker`'s own error
        term (e.g. a supplier/peer whose moves affect `endogenous_ticker` but only reach
        `dependent_ticker`, if at all, THROUGH `endogenous_ticker`) allows recovering a
        cleaner estimate of the causal pass-through.

        The formula is as follows:

        - Z = [1, exogenous tickers, instruments] and X = [1, exogenous tickers, endogenous tickers], all
          in `column` (Return by default), the constant only with add_constant=True
        - Stage 1: X̂ = Z(Z'Z)⁻¹Z'X, the fitted values of each endogenous ticker regressed on Z
        - Stage 2: β̂ = (X̂'X)⁻¹X̂'y
        - Cov(β̂) = σ̂²·(X̂'X̂)⁻¹, with σ̂² = Σ(yₜ - xₜ'β̂)² / (n - k) (linearmodels IV2SLS with
          cov_type="unadjusted" and debiased=True)

        Args:
            dependent_ticker (str): The dependent (predicted) asset.
            endogenous_ticker (str | list[str]): The endogenous regressor asset(s) --
            suspected correlated with the error term.
            instrument_tickers (str | list[str]): The excluded instrument asset(s),
            correlated with `endogenous_ticker` but assumed uncorrelated with the error
            term. Must supply at least as many instruments as endogenous regressors.
            exogenous_tickers (str | list[str] | None, optional): Other, non-instrumented
            control asset(s) included as-is in both stages. Defaults to None.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to regress on. Defaults to
            "Return".
            add_constant (bool, optional): Whether to include an intercept. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to.
            Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A coefficient table (Coefficient, CORRECTED Std. Error, t-Statistic,
            P-Value), indexed by regressor name -- see `causal_inference_model.get_iv_2sls` for
            why these standard errors differ from (and correct) what a naive two-OLS-calls
            approach would report.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["SMCI", "NVDA", "TSM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_iv_2sls(
            dependent_ticker="SMCI", endogenous_ticker="NVDA", instrument_tickers="TSM"
        )
        ```

        Which returns:

        |           |   Coefficient |   Std. Error |   t-Statistic |   P-Value |
        |:----------|--------------:|-------------:|--------------:|----------:|
        | Intercept |       -0.0008 |       0.0027 |       -0.2931 |    0.7697 |
        | NVDA      |        1.2518 |       0.1094 |       11.441  |    0      |
        """
        period = period if period else "daily"
        endogenous_ticker = (
            [endogenous_ticker]
            if isinstance(endogenous_ticker, str)
            else endogenous_ticker
        )
        instrument_tickers = (
            [instrument_tickers]
            if isinstance(instrument_tickers, str)
            else instrument_tickers
        )
        exogenous_tickers = (
            [exogenous_tickers]
            if isinstance(exogenous_tickers, str)
            else exogenous_tickers
        )

        returns = self._get_series_column(
            period,
            column,
            [
                dependent_ticker,
                *endogenous_ticker,
                *instrument_tickers,
                *(exogenous_tickers or []),
            ],
        )

        result = causal_inference_model.get_iv_2sls(
            returns[dependent_ticker],
            returns[endogenous_ticker],
            returns[instrument_tickers],
            returns[exogenous_tickers] if exogenous_tickers else None,
            add_constant=add_constant,
        )

        return apply_rounding(
            regression_model.regression_summary_table(result),
            rounding if rounding is not None else self._rounding,
        )

    @handle_errors
    def get_difference_in_differences(
        self,
        treated_tickers: str | list[str],
        treatment_date: str,
        control_tickers: str | list[str] | None = None,
        period: str | None = None,
        column: str = "Return",
        add_constant: bool = True,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Fit a Difference-in-Differences (DiD) regression estimating the effect of some
        event on `treatment_date` for `treated_tickers`, relative to `control_tickers`.

        Also known as: DiD, DD, difference-in-differences estimator.

        For more information about the method, and why the `Treated x Post` interaction
        coefficient IS the DiD treatment-effect estimate, see
        `causal_inference_model.get_difference_in_differences`.

        Builds a stacked (ticker, date) panel from every ticker in `treated_tickers` and
        `control_tickers`: each observation's outcome is that ticker's return on that
        date, `Treated` is 1 for every observation belonging to a `treated_tickers`
        asset (regardless of date) and 0 for `control_tickers` assets, and `Post` is 1
        for observations on or after `treatment_date` (regardless of asset). This
        answers "did `treated_tickers` behave differently after `treatment_date`,
        beyond both their normal average gap versus `control_tickers` and the common
        market-wide move over that same before/after window?" -- e.g. isolating the
        effect of an event (an index-inclusion announcement, a regulatory change
        affecting only some tickers, an earnings surprise) that hits `treated_tickers`
        but not `control_tickers`, at a known date.

        With returns as the outcome, the `Treated x Post` coefficient is the change in the
        average return per period after the event, not the price impact around it, which
        is what `get_event_study` measures. The panel stacks tickers observed on the same
        dates, whose returns are correlated, while the standard errors treat every
        observation as independent, so the p-value tends to be too low for closely related
        tickers. The estimate rests on parallel trends, so the first check is whether both
        groups moved alike before the event.

        The formula is as follows:

        - yᵢₜ = β₀ + β₁·Treatedᵢ + β₂·Postₜ + δ·(Treatedᵢ × Postₜ) + εᵢₜ, fitted by OLS with classical
          standard errors on the stacked (ticker, date) panel of `column` (Return by default)
        - Treatedᵢ = 1 for `treated_tickers`, Postₜ = 1 on or after `treatment_date`
        - δ̂ = (mean(y_treated,post) - mean(y_treated,pre)) - (mean(y_control,post) - mean(y_control,pre))

        Args:
            treated_tickers (str | list[str]): The asset(s) subject to the event/treatment.
            treatment_date (str): The date the event/treatment occurs, in the same format
            accepted by `pd.Timestamp`. Observations on or after this date are `Post = 1`.
            control_tickers (str | list[str] | None, optional): The untreated comparison
            asset(s). Defaults to None, which uses every ticker NOT in `treated_tickers`,
            excluding "Benchmark" (pass `control_tickers="Benchmark"` to use the benchmark as
            the control group).
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to use as the outcome.
            Defaults to "Return".
            add_constant (bool, optional): Whether to include an intercept. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to.
            Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A coefficient table (Coefficient, Std. Error, t-Statistic, P-Value)
            with rows `Intercept`, `Treated`, `Post` and `Treated x Post` -- the last of
            which is the DiD treatment-effect estimate.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NVDA", "INTC"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_difference_in_differences(
            treated_tickers="NVDA",
            control_tickers="INTC",
            treatment_date="2023-06-01",
            period="monthly",
        )
        ```

        Which returns:

        |                |   Coefficient |   Std. Error |   t-Statistic |   P-Value |
        |:---------------|--------------:|-------------:|--------------:|----------:|
        | Intercept      |       -0.0126 |       0.0262 |       -0.4794 |    0.6325 |
        | Treated        |        0.0669 |       0.0371 |        1.8061 |    0.0735 |
        | Post           |        0.0291 |       0.0362 |        0.8044 |    0.4229 |
        | Treated x Post |       -0.0246 |       0.0511 |       -0.4817 |    0.6309 |
        """
        period = period if period else "daily"
        returns = self._get_series_column(period, column)

        treated_tickers = (
            [treated_tickers] if isinstance(treated_tickers, str) else treated_tickers
        )

        if control_tickers is None:
            control_tickers = [
                ticker
                for ticker in self._get_tickers()
                if ticker not in treated_tickers
            ]
        elif isinstance(control_tickers, str):
            control_tickers = [control_tickers]

        panel_tickers = [*treated_tickers, *control_tickers]

        index = returns.index
        timestamps = (
            index.to_timestamp()
            if isinstance(index, pd.PeriodIndex)
            else pd.DatetimeIndex(index)
        )
        post_flags = pd.Series(
            (timestamps >= pd.Timestamp(treatment_date)).astype(float),
            index=returns.index,
        )

        outcome_values = []
        treated_values = []
        post_values = []
        for ticker in panel_tickers:
            ticker_returns = returns[ticker].dropna()
            outcome_values.append(ticker_returns)
            treated_values.append(
                pd.Series(
                    1.0 if ticker in treated_tickers else 0.0,
                    index=ticker_returns.index,
                )
            )
            post_values.append(post_flags.loc[ticker_returns.index])

        outcome = pd.concat(outcome_values, ignore_index=True)
        treated = pd.concat(treated_values, ignore_index=True)

        # Without one of the two groups the Treated dummy is constant and the design
        # is rank deficient, which says nothing about the cause.
        for group, flag in (("treated", 1.0), ("control", 0.0)):
            if not treated.eq(flag).any():
                raise ValueError(
                    f"There are no {group} observations: the {group} tickers have no "
                    f"{column} data in this period."
                )

        post = pd.concat(post_values, ignore_index=True)

        result = causal_inference_model.get_difference_in_differences(
            outcome, treated, post, add_constant=add_constant
        )

        return apply_rounding(
            regression_model.regression_summary_table(result),
            rounding if rounding is not None else self._rounding,
        )

    @handle_errors
    def get_regression_discontinuity(
        self,
        dependent_ticker: str,
        running_variable_ticker: str,
        cutoff: float,
        period: str | None = None,
        column: str = "Return",
        bandwidth: float | None = None,
        kernel: str = "uniform",
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Estimate a Sharp Regression Discontinuity (RDD): the jump in `dependent_ticker`
        exactly where `running_variable_ticker` crosses `cutoff`.

        Also known as: RDD, sharp RD.

        For more information about the method, see
        `causal_inference_model.get_regression_discontinuity`.

        Treats `running_variable_ticker`'s value as the running variable that
        (hypothetically) triggers some discrete change once it crosses `cutoff` -- e.g.
        testing whether `dependent_ticker`'s return behaves discontinuously around a
        round-number/threshold level of another asset or indicator (a psychological
        price level, an index-inclusion market-cap threshold, a macro indicator's
        policy-relevant threshold) fed in as `running_variable_ticker`. Fits separate
        local linear regressions of `dependent_ticker`'s return on the (cutoff-centered)
        `running_variable_ticker` value, one on each side of `cutoff`, and reports the
        gap between the two fitted lines exactly at the cutoff.

        The formula is as follows:

        - x̃ₜ = xₜ - cutoff, keeping the observations with |x̃ₜ| < h, where xₜ is the running ticker's
          `column` (Return by default) and h = `bandwidth`, or half the range of xₜ by default
        - Each side (x̃ₜ < 0 and x̃ₜ ≥ 0): yₜ = a + b·x̃ₜ + eₜ, fitted by OLS (kernel="uniform", the
          default) or by WLS with weights 1 - |x̃ₜ| / h (kernel="triangular")
        - Discontinuity = a_right - a_left, with SE = √(SE(a_right)² + SE(a_left)²)
        - t = Discontinuity / SE, with a Student-t distribution on the two sides' residual degrees of
          freedom combined

        Args:
            dependent_ticker (str): The outcome asset.
            running_variable_ticker (str): The asset (or column) whose value determines
            which side of `cutoff` an observation falls on.
            cutoff (float): The threshold value of `running_variable_ticker` at which the
            discontinuity is estimated.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to use for both series.
            Defaults to "Return".
            bandwidth (float | None, optional): The maximum distance from `cutoff` an
            observation may be to be included in either local regression. Defaults to
            None, which uses half of the running variable's observed range -- see
            `causal_inference_model.get_regression_discontinuity` for why this is a
            deliberately naive default.
            kernel (str, optional): One of "uniform" or "triangular". Defaults to "uniform".
            rounding (int | None, optional): The number of decimals to round the results to.
            Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A one-column results table with the discontinuity estimate, its
            standard error/t-statistic/p-value, and the cutoff/bandwidth/sample sizes used.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["COIN", "BTCUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2022-01-01",
            end_date="2025-12-31",
        )

        # COIN's return on days Bitcoin rises versus days it falls
        toolkit.econometrics.get_regression_discontinuity(
            dependent_ticker="COIN", running_variable_ticker="BTCUSD", cutoff=0
        )
        ```

        Which returns:

        |               |    Value |
        |:--------------|---------:|
        | Discontinuity |   0.0031 |
        | Std. Error    |   0.0027 |
        | t-Statistic   |   1.1191 |
        | P-Value       |   0.2633 |
        | Cutoff        |   0      |
        | Bandwidth     |   0.1497 |
        | N Left        | 729      |
        | N Right       | 728      |
        """
        period = period if period else "daily"
        returns = self._get_series_column(
            period, column, [dependent_ticker, running_variable_ticker]
        )

        result = causal_inference_model.get_regression_discontinuity(
            returns[dependent_ticker],
            returns[running_variable_ticker],
            cutoff=cutoff,
            bandwidth=bandwidth,
            kernel=kernel,
        )

        return apply_rounding(
            causal_inference_model.regression_discontinuity_summary_table(result),
            rounding if rounding is not None else self._rounding,
        )

    @handle_errors
    def get_propensity_score_matching(
        self,
        dependent_ticker: str,
        treatment_ticker: str,
        covariate_tickers: str | list[str],
        treatment_threshold: float = 0.0,
        period: str | None = None,
        column: str = "Return",
        caliper: float | None = None,
        add_constant: bool = True,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series:
        """
        Estimate the Average Treatment effect on the Treated (ATT) of `treatment_ticker`
        exceeding `treatment_threshold` on `dependent_ticker`'s return, via Propensity
        Score Matching (PSM) on `covariate_tickers`.

        Also known as: PSM, nearest-neighbor propensity matching.

        For more information about the method, see
        `causal_inference_model.get_propensity_score_matching`.

        Derives a binary "treatment" indicator the same way `get_logistic_regression`
        derives its binary outcome: 1 if `treatment_ticker`'s return in a given period
        exceeds `treatment_threshold` (0.0, i.e. a positive return, by default), else 0.
        This lets PSM answer e.g. "on periods where `treatment_ticker` has an
        outsized/positive move, is `dependent_ticker`'s return different than it would
        otherwise be -- comparing only periods that LOOK similar on `covariate_tickers`
        (to control for the possibility that `treatment_ticker` tends to move on the
        same periods/regimes that also independently affect `dependent_ticker`)?"

        The formula is as follows:

        - Dₜ = 1 if the treatment ticker's `column` (Return by default) is above `treatment_threshold`
          (0 by default), else 0
        - e(xₜ) = 1 / (1 + exp(-xₜ'β)), a logit of Dₜ on the covariate tickers (plus a constant)
        - Matching on ℓₜ = ln(e / (1 - e)): each treated period, in order, takes the nearest unmatched
          control period (one-to-one, without replacement) if |ℓ_treated - ℓ_control| ≤ caliper, which
          defaults to 0.2 × the standard deviation (ddof=1) of ℓ
        - ATT = (1 / m)·Σ(y_treated - y_control) over the m matched pairs
        - SE = std(y_treated - y_control, ddof=1) / √m, with a Student-t distribution with m - 1 degrees
          of freedom

        Args:
            dependent_ticker (str): The outcome asset.
            treatment_ticker (str): The asset whose return, once it exceeds
            `treatment_threshold`, defines the treatment indicator.
            covariate_tickers (str | list[str]): The asset(s) used as covariates to
            estimate the propensity score -- should include asset(s) believed to drive
            selection into "treatment".
            treatment_threshold (float, optional): The return threshold defining
            treatment. Defaults to 0.0.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to use. Defaults to "Return".
            caliper (float | None, optional): The maximum allowed logit-propensity-score
            matching distance. Defaults to None, which uses Austin's (2011) rule of thumb
            -- see `causal_inference_model.get_propensity_score_matching`.
            add_constant (bool, optional): Whether to include an intercept in the
            propensity score model. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to.
            Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: The ATT estimate, its standard error/t-statistic/p-value, and the
            number of matched pairs/treated/control observations.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["XOM", "USO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        # XOM's return on days USO rises more than 2%, matched on the Benchmark's return
        toolkit.econometrics.get_propensity_score_matching(
            dependent_ticker="XOM",
            treatment_ticker="USO",
            covariate_tickers="Benchmark",
            treatment_threshold=0.02,
        )
        ```

        Which returns:

        |               |     Value |
        |:--------------|----------:|
        | ATT           |    0.0163 |
        | Std. Error    |    0.0016 |
        | t-Statistic   |    9.9366 |
        | P-Value       |    0      |
        | Matched Pairs |  181      |
        | N Treated     |  185      |
        | N Control     | 1069      |
        """
        period = period if period else "daily"
        covariate_tickers = (
            [covariate_tickers]
            if isinstance(covariate_tickers, str)
            else covariate_tickers
        )

        returns = self._get_series_column(
            period, column, [dependent_ticker, treatment_ticker, *covariate_tickers]
        )
        treatment = (returns[treatment_ticker] > treatment_threshold).astype(float)

        result = causal_inference_model.get_propensity_score_matching(
            treatment,
            returns[dependent_ticker],
            returns[covariate_tickers],
            caliper=caliper,
            add_constant=add_constant,
        )

        return apply_rounding(
            causal_inference_model.propensity_score_matching_summary(result),
            rounding if rounding is not None else self._rounding,
        )

    @handle_errors
    def get_synthetic_control(
        self,
        treated_ticker: str,
        treatment_period: str,
        donor_tickers: str | list[str] | None = None,
        period: str | None = None,
        column: str = "Return",
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series:
        """
        Construct a Synthetic Control for `treated_ticker` from a weighted combination of
        `donor_tickers` (Abadie, Diamond & Hainmueller, 2010), and estimate the effect of an
        event/intervention as the post-`treatment_period` gap between `treated_ticker`'s
        actual and synthetic counterfactual return path.

        When a single company or market experiences an event (an acquisition, a regulatory
        change, an index inclusion) and there is no single obvious comparison, the synthetic
        control builds one from the data: a weighted average of unaffected "donor" tickers,
        with weights chosen so it tracks the treated ticker as closely as possible before the
        event. If that pre-event fit is good, the synthetic control's path after the event
        estimates what would have happened without it, and the gap is the estimated effect.

        The formula is as follows:

        - Weights: w* = argmin_w ||treated_pre - donors_pre @ w||^2, subject to w >= 0 and
          sum(w) = 1 (SLSQP), with pre the dates before `treatment_period`
        - Synthetic control: synthetic_t = donors_t @ w*, for every period t
        - Treatment effect: gap_t = treated_t - synthetic_t, for every post-event period t
        - Average Treatment Effect = mean of gap_t over the post-event periods
        - RMSPE ratio = sqrt(mean(gap_post^2)) / sqrt(mean(gap_pre^2))
        - P-Value = the share of the treated ticker's and every placebo donor's RMSPE ratios that are at
          least the treated ticker's ratio
        - All series are the tickers' `column` (Return by default)

        Unlike difference-in-differences, this does not require the treated and comparison
        units to follow parallel trends, as the synthetic control is built to match the
        treated ticker's own pre-event path. Because there is only one treated unit, the
        p-value comes from placebo tests: the procedure is rerun with each donor as if it were
        treated, and the treated ticker's post/pre fit ratio is compared with theirs. Matching
        is done on the pre-event outcome path only, a common simplification of the original
        method.

        For more information about the method, see the following papers:

        - Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control Methods for
        Comparative Case Studies: Estimating the Effect of California's Tobacco Control
        Program." Journal of the American Statistical Association, 105(490), 493-505.
        - Abadie, A. (2021). "Using Synthetic Controls: Feasibility, Data Requirements, and
        Methodological Aspects." Journal of Economic Literature, 59(2), 391-425.

        Also known as: SCM, synthetic control method.

        Args:
            treated_ticker (str): The asset believed to be affected by an event/
            intervention starting at `treatment_period`.
            treatment_period (str): The first post-treatment period -- periods at or
            after this value (within `period`'s index) are treated as post-treatment,
            everything before as pre-treatment (used to fit the synthetic control's
            weights).
            donor_tickers (str | list[str] | None, optional): The ticker(s) forming
            the donor pool the synthetic control is built from. Defaults to None,
            meaning every other Toolkit ticker (subject to `include_benchmark`).
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to use. Defaults to "Return".
            include_benchmark (bool, optional): Whether to include "Benchmark" in the
            default donor pool (has no effect when donor_tickers is given explicitly).
            Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: The average post-treatment effect, pre/post RMSPE and their
            ratio, the placebo-based p-value, and the donor pool/period counts.

        Notes:
        - Needs at least 2 donor tickers -- with only 1-2 Toolkit tickers loaded,
        pass `donor_tickers` explicitly (e.g. including "Benchmark") or (better)
        construct the Toolkit with more tickers, since small donor pools give both a
        poorly-identified synthetic control and a very coarse placebo p-value (with
        `k` donors, the smallest achievable p-value is `1 / (k + 1)`).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["BA", "AIR.PA", "LMT", "GD", "RTX", "NOC"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2023-01-01",
            end_date="2024-12-31",
        )

        # Boeing after the 737 MAX 9 grounding, with the other aerospace companies as donors
        toolkit.econometrics.get_synthetic_control(treated_ticker="BA", treatment_period="2024-01-08")
        ```

        Which returns:

        |                          |    Value |
        |:-------------------------|---------:|
        | Average Treatment Effect |  -0.0017 |
        | Pre-Treatment RMSPE      |   0.0156 |
        | Post-Treatment RMSPE     |   0.0204 |
        | RMSPE Ratio              |   1.3094 |
        | P-Value                  |   0.5    |
        | N Donors                 |   5      |
        | N Pre-Periods            | 260      |
        | N Post-Periods           | 255      |
        """
        period = period if period else "daily"
        if donor_tickers is None:
            donor_tickers = [
                ticker
                for ticker in self._get_tickers(include_benchmark=include_benchmark)
                if ticker != treated_ticker
            ]
        else:
            donor_tickers = (
                [donor_tickers]
                if isinstance(donor_tickers, str)
                else list(donor_tickers)
            )

        returns = self._get_series_column(
            period, column, [treated_ticker, *donor_tickers]
        )

        result = causal_inference_model.get_synthetic_control(
            returns[treated_ticker],
            returns[donor_tickers],
            treatment_period=treatment_period,
        )

        return apply_rounding(
            causal_inference_model.synthetic_control_summary(result),
            rounding if rounding is not None else self._rounding,
        )

    @handle_errors
    def get_event_study(
        self,
        event_date: str,
        dependent_ticker: str | None = None,
        column: str = "Return",
        estimation_window: int = 250,
        gap_days: int = 30,
        pre_event_days: int = 10,
        post_event_days: int = 10,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series:
        """
        Perform a market-model event study around a single event date, following the
        methodology in MacKinlay, A.C. (1997), "Event Studies in Economics and
        Finance," Journal of Economic Literature, 35(1), 13-39 -- the canonical
        reference and still the dominant approach used to measure the stock-price
        impact of corporate events (earnings announcements, M&A deals, index
        additions/deletions, dividend changes, regulatory actions, etc.).

        A market model (`Return_t = alpha + beta * Benchmark_Return_t + e_t`) is fit
        via OLS over a clean "estimation window" ending `gap_days` before the event,
        then the Abnormal Return on each day of the "event window" around the event
        is the actual return minus the market-model-predicted expected return. These
        are cumulated into the Cumulative Abnormal Return (CAR) -- the stock-price
        impact attributable to the event, after stripping out what would have been
        expected from general market movements alone -- along with a t-test of
        whether CAR is significantly different from zero.

        The `event_date` is the first trading day on which the news can be priced, so for
        results released after the close it is the next trading day. The CAR runs from
        `pre_event_days` before to `post_event_days` after it (21 trading days by default),
        which also captures any leak ahead of the announcement. The t-test treats the
        abnormal returns as independent and uses the residual Variance of the estimation
        window, so a single event is weak evidence; studies average the CARs of many events
        of the same kind.

        The formula is as follows:

        - Market model: Rₜ = α + β·R_m,t + εₜ, fitted by OLS on the daily `column` (Return by default) of
          the ticker and the Benchmark over the `estimation_window` (250) trading days that end
          `gap_days` (30) trading days before the event
        - ARₜ = Rₜ - (α̂ + β̂·R_m,t), for t = -`pre_event_days` .. +`post_event_days` (10 and 10)
        - CAR = Σ ARₜ over the L = `pre_event_days` + `post_event_days` + 1 event-window days
        - t = CAR / √(L·σ̂²), with σ̂² = Σε̂ₜ² / (N - 2) over the N estimation-window days and a Student-t
          distribution with N - 2 degrees of freedom

        Also known as: CAR analysis, abnormal returns analysis, market model event
        study.

        Args:
            event_date (str): The date of the event (e.g. "2023-05-04"). Must fall
                within the Toolkit instance's daily historical data.
            dependent_ticker (str | None, optional): The ticker being studied.
                Defaults to the first ticker in the Toolkit instance.
            column (str, optional): The historical data column to use. Defaults to
                "Return".
            estimation_window (int, optional): Number of trading days used to
                estimate the market model. Defaults to 250 (~one trading year).
            gap_days (int, optional): Number of trading days between the end of the
                estimation window and the event date. Defaults to 30.
            pre_event_days (int, optional): Number of trading days before the event
                date included in the event window. Defaults to 10.
            post_event_days (int, optional): Number of trading days after the event
                date included in the event window. Defaults to 10.
            rounding (int | None, optional): The number of decimals to round the
                results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: The Cumulative Abnormal Return (CAR), its t-statistic and
            p-value, the market-model alpha and beta, and the number of estimation-
            window observations used.

        Notes:
        - Always uses daily data -- event windows are measured in trading days, so
          quarterly/yearly granularity would not be meaningful here.
        - The market/benchmark return series is the Toolkit instance's `"Benchmark"`
          column.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NVDA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2022-01-01",
            end_date="2023-12-31",
        )

        # NVDA's abnormal return around its May 2023 earnings release
        toolkit.econometrics.get_event_study(event_date="2023-05-25")
        ```

        Which returns:

        |                                |    Value |
        |:-------------------------------|---------:|
        | Cumulative Abnormal Return     |   0.2044 |
        | CAR t-statistic                |   1.9468 |
        | CAR p-value                    |   0.0527 |
        | Alpha                          |   0.0018 |
        | Beta                           |   2.0781 |
        | Estimation Window Observations | 250      |
        """
        dependent_ticker = dependent_ticker or self._get_tickers()[0]

        # The windows count trading days of the dependent ticker and the Benchmark
        # together, so only their common dates are kept: another ticker's missing
        # dates do not shorten or shift the estimation and event windows.
        returns = self._get_series_column(
            "daily", column, [dependent_ticker, BENCHMARK_TICKER]
        )

        result = event_study_model.get_event_study(
            returns=returns[dependent_ticker],
            market_returns=returns[BENCHMARK_TICKER],
            event_date=event_date,
            estimation_window=estimation_window,
            gap_days=gap_days,
            pre_event_days=pre_event_days,
            post_event_days=post_event_days,
        )

        summary = pd.Series(
            {
                "Cumulative Abnormal Return": result["cumulative_abnormal_return"],
                "CAR t-statistic": result["car_t_statistic"],
                "CAR p-value": result["car_p_value"],
                "Alpha": result["alpha"],
                "Beta": result["beta"],
                "Estimation Window Observations": result[
                    "estimation_window_observations"
                ],
            }
        )

        return apply_rounding(
            summary, rounding if rounding is not None else self._rounding
        )
