"""Dependence Risk Module"""

__docformat__ = "google"

import itertools
from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd

from financetoolkit.ratios import valuation_model
from financetoolkit.ratios.helpers import map_period_data_to_daily_data
from financetoolkit.risk.dependence import copula_model, covar_model
from financetoolkit.risk.helpers import check_rolling
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import (
    InvalidArgumentError,
    check_option,
    check_period,
    handle_errors,
    validate_alpha,
)
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import (
    PERIOD_TRANSLATION,
    apply_rounding,
    finalize_dataset,
)

# ruff: noqa: E501

MINIMUM_TICKERS_FOR_ALL_PAIRS = 2


@filter_shown_columns
@track_sources
class DependenceRisk:
    """
    Dependence between assets: copula fits, simulations and the best fitting copula, the tail dependence coefficient,
    and the systemic risk measures CoVaR, Marginal Expected Shortfall and SRISK.
    """

    # The Toolkit sets this to a function returning the Total Liabilities of the balance sheet
    # and the Weighted Average Shares Diluted of the income statement (tickers as index,
    # periods as columns), each with the day its periods' figures apply from, which SRISK
    # takes the debt and the market capitalization from.
    _capital_shortfall_function: (
        Callable[[], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]
        | None
    ) = None
    # The Toolkit sets these to the benchmark ticker (reported as "Benchmark") and to a function
    # returning the returns of tickers outside the Toolkit for a period, so a market or system
    # ticker can be named as it trades, also when it is not one of the tickers.
    _benchmark_name: str | None = None
    _other_returns_function: Callable[[list[str], str], pd.DataFrame] | None = None

    def _resolve_ticker_pairs(
        self, ticker_a: str | None, ticker_b: str | None
    ) -> list[tuple[str, str]]:
        if ticker_a is None and ticker_b is None:
            if len(self._tickers) < MINIMUM_TICKERS_FOR_ALL_PAIRS:
                raise ValueError(
                    "At least 2 tickers are required in the Toolkit instance to "
                    "calibrate a copula between every pair."
                )

            return list(itertools.combinations(self._tickers, 2))

        if ticker_a is None or ticker_b is None:
            raise InvalidArgumentError(
                "Either provide both ticker_a and ticker_b, or neither (to use "
                "every pair of tickers in the Toolkit instance)."
            )

        return [(ticker_a, ticker_b)]

    def _benchmark_pairs(
        self, varying: str | None, counterpart: str | None
    ) -> list[tuple[str, str]]:
        # The (varying, counterpart) pairs of a pairwise measure: the counterpart defaults to the benchmark and the varying side to every other ticker, so a given ticker keeps the role of its own parameter. The benchmark named as it trades (SPY) is the "Benchmark" column, unless it is also one of the tickers.  # noqa: E501
        varying, counterpart = (
            (
                "Benchmark"
                if name is not None
                and self._benchmark_name
                and name not in self._tickers
                and name.upper() == self._benchmark_name.upper()
                and "Benchmark" in self._tickers
                else name
            )
            for name in (varying, counterpart)
        )
        if counterpart is None:
            if "Benchmark" not in self._tickers:
                raise ValueError(
                    "Without a benchmark (benchmark_ticker=None) both tickers of the pair are required."
                )
            counterpart = "Benchmark"

        others = (
            [varying]
            if varying is not None
            else [
                ticker
                for ticker in self._tickers
                if ticker not in ("Portfolio", counterpart)
            ]
        )

        return [(ticker, counterpart) for ticker in others]

    def _add_other_returns(
        self, returns: pd.DataFrame, tickers: list[str], period: str
    ) -> pd.DataFrame:
        # The returns of tickers that are not in the Toolkit (a market index such as ^GSPC) are retrieved for the same dates, sorted as a PeriodIndex union keeps the order of the frames.  # noqa: E501
        missing = list(dict.fromkeys(t for t in tickers if t not in returns.columns))
        if not missing:
            return returns
        if self._other_returns_function is None:
            raise ValueError(
                f"{', '.join(missing)} is not part of the Toolkit; add it to the tickers or "
                "set it as the benchmark_ticker."
            )

        return pd.concat(
            [
                returns,
                self._other_returns_function(missing, period).reindex(columns=missing),
            ],
            axis=1,
        ).sort_index()

    @handle_errors
    def get_tail_dependence_coefficient(
        self,
        ticker_a: str | None = None,
        ticker_b: str | None = None,
        period: str | None = None,
        column: str = "Return",
        q: float = 0.95,
        method: Literal["empirical", "gaussian", "student-t"] = "empirical",
        dof: float = 4.0,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series:
        """
        Calculate the Upper and Lower Tail Dependence Coefficients between `ticker_a` and `ticker_b`: how
        likely one asset is in extreme distress given that the other one already is.

        Without a pair, every ticker is paired with the benchmark, which
        tells how likely each company is to be in distress when the market is.

        Correlation only captures the *average* co-movement between two assets -- it
        says nothing about whether they are more likely to crash together than an
        equivalent gaussian relationship would imply. The Tail Dependence Coefficient
        answers that specific question directly: the probability that one asset is in
        extreme distress, given that the other one already is.

        The empirical estimate at q = 0.95 rests on the few days on which both assets are
        in their worst (or best) 5% at the same time, so it needs several years of daily
        data to be stable. A gaussian dependence of any correlation below 1 implies zero
        tail dependence, so a Lower Tail Dependence Coefficient well above zero means that
        diversification between the two assets is weakest exactly when it is needed.

        For more information about the method, see the following papers:

        - Embrechts, P., McNeil, A., & Straumann, D. (1999). "Correlation: Pitfalls
        and Alternatives." RISK Magazine, 12, 69-71.
        - Poon, S.H., Rockinger, M., & Tawn, J. (2004). "Extreme Value Dependence in
        Financial Markets: Diagnostics, Models, and Financial Implications." Review of
        Financial Studies, 17(2), 581-610.

        The formula is as follows (with the default empirical estimate and a q of 0.95):

        - Upper Tail Dependence = #(A > Q_A(q) and B > Q_B(q)) / #(A > Q_A(q))
        - Lower Tail Dependence = #(A <= Q_A(1 - q) and B <= Q_B(1 - q)) / #(A <= Q_A(1 - q))

        Where A and B are the values of `ticker_a` and `ticker_b` on their common dates and Q_A and Q_B their
        empirical quantiles. With the gaussian copula both coefficients are 0 (1 only for a correlation of exactly
        1), and with the Student-T copula both are 2 * t_(v+1)(-√((v + 1) * (1 - rho) / (1 + rho))), with v = `dof`,
        rho the Pearson correlation and t_(v+1) the Student-T distribution function.

        Also known as: tail dependence, extremal dependence coefficient.

        Shares similar logic:
        - skfolio, StudentTCopula (Student-T): https://skfolio.org/generated/skfolio.distribution.StudentTCopula.html

        Args:
            ticker_a (str | None, optional): The first asset. Defaults to None, which pairs every other ticker
            in the Toolkit instance (the benchmark included when `ticker_b` is given) with `ticker_b`.
            ticker_b (str | None, optional): The second asset, the benchmark by its own ticker (SPY) or as
                "Benchmark", or a ticker outside the Toolkit, whose returns are retrieved for the same dates
                (with column="Return"). Defaults to None, which uses the benchmark.
            period (str, optional): The data frequency. Options are: "daily", "weekly", "monthly", "quarterly",
                "yearly". Defaults to "daily", since a dependence estimate needs far more observations than a
                lower frequency provides -- at "yearly" a decade of history is only ten
                observations.
            column (str, optional): The historical data column to use. Defaults to "Return", since
            tail dependence between return series is the standard risk management application.
            q (float, optional): The threshold quantile used for the "empirical" method, in (0.5, 1).
            Defaults to 0.95.
            method (str, optional): The estimation method. Options are: "empirical" (nonparametric plug-in
                estimate), "gaussian" (gaussian copula), "student-t" (Student-T copula). Defaults to "empirical".
            dof (float, optional): The degrees of freedom of the Student-T copula, only used when
            method is "student-t". Defaults to 4.0.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Raises:
            ValueError: If one of the two tickers is missing while the Toolkit has no benchmark.

        Returns:
            pd.Series | pd.DataFrame: The Lower and Upper Tail Dependence Coefficients, the linear (Pearson)
            correlation between the two assets, and the number of observations used -- a Series for a given
            pair, or a DataFrame with a row per ticker paired with `ticker_b` (or the benchmark) otherwise.

        Notes:
        - The method retrieves historical data based on the specified `period` for each pair of
        assets and estimates the tail dependence coefficient between them.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "JPM", "XOM", "^STOXX50E"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_tail_dependence_coefficient()
        ```

        Which returns:

        |           |   Lower Tail Dependence |   Upper Tail Dependence |   Correlation |   Observations |
        |:----------|------------------------:|------------------------:|--------------:|---------------:|
        | AAPL      |                  0.4603 |                  0.4921 |        0.7582 |           1254 |
        | JPM       |                  0.3968 |                  0.4127 |        0.6233 |           1254 |
        | XOM       |                  0.2063 |                  0.2222 |        0.3425 |           1254 |
        | ^STOXX50E |                  0.3387 |                  0.3226 |        0.4509 |           1240 |
        """
        period = period if period else "daily"
        returns = self._get_price_column(period, column)
        pairs = self._benchmark_pairs(ticker_a, ticker_b)
        if column == "Return":
            returns = self._add_other_returns(
                returns, [name for pair in pairs for name in pair], period
            )

        result = pd.DataFrame(
            {
                ticker: copula_model.get_tail_dependence_coefficient(
                    returns[ticker], returns[other], q=q, method=method, dof=dof
                )
                for ticker, other in pairs
            }
        ).T

        if ticker_a is not None and ticker_b is not None:
            result = result.iloc[0].rename(None)

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_copula_parameters(
        self,
        ticker_a: str | None = None,
        ticker_b: str | None = None,
        copula: str = "gaussian",
        period: str | None = None,
        column: str = "Return",
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calibrate a bivariate copula between `ticker_a` and `ticker_b`: how their returns move together
        apart from their own distributions, including how often their crashes or rallies coincide.

        It is fitted on the pseudo-observations (ranks) of their returns: the Clayton, Gumbel and Frank
        parameters by maximum likelihood, the gaussian correlation from the normal
        scores, and the Student-T correlation from Kendall's tau with the degrees of
        freedom by maximum likelihood (Demarta & McNeil, 2005).

        When `ticker_a`/`ticker_b` are not given, every unique pair among the
        Toolkit's tickers is calibrated instead.

        A copula separates the dependence structure between two assets from
        their individual (marginal) return distributions (Sklar's theorem),
        letting the two be modeled independently. Five families are supported,
        each capturing a different shape of dependence:

        - Gaussian: zero tail dependence -- crashes and rallies are no more
        likely to happen together than the correlation alone implies. Included
        mainly as a baseline, since real asset returns typically show more
        joint tail risk than this.
        - Student-T: symmetric, nonzero tail dependence in both tails.
        - Clayton: nonzero *lower* tail dependence only -- assets crash
        together more than they rally together. The most common choice for
        equity return pairs.
        - Gumbel: nonzero *upper* tail dependence only -- assets rally
        together more than they crash together.
        - Frank: zero tail dependence in both tails, but (unlike gaussian) can
        represent negative dependence and is symmetric around independence.

        See `get_best_fitting_copula` to compare all five by AIC on the same
        pair of assets, and see `get_copula_simulation` to draw joint scenarios
        from the fitted copula.

        The formula is as follows:

        - u_i = Rank(A_i) / (n + 1), v_i = Rank(B_i) / (n + 1)
        - Gaussian Rho = Correlation(Φ^-1(u), Φ^-1(v))
        - Student-T Rho = sin(π * Kendall's Tau / 2)
        - Theta (and the Student-T Degrees of Freedom) = argmax Σ ln c(u_i, v_i)
        - AIC = 2 * k - 2 * Log-Likelihood

        Where A and B are the values of the two assets on their n common dates, c is the copula density and k
        the number of fitted parameters (2 for the Student-T, 1 otherwise). The search bounds are [2.05, 60] for
        the Degrees of Freedom and, for Theta, [0.0001, 100] (Clayton), [1, 50] (Gumbel) and [-50, 50] (Frank).
        The implied tail dependence is 2^(-1 / Theta) in the lower tail for Clayton, 2 - 2^(1 / Theta) in the
        upper tail for Gumbel, 2 * t_(v+1)(-√((v + 1) * (1 - Rho) / (1 + Rho))) in both tails for the Student-T
        and 0 otherwise.

        For more information about the method, see:

        - Sklar, A. (1959). Publications de l'Institut de Statistique de
        l'Universite de Paris, 8, 229-231.
        - McNeil, A.J., Frey, R., & Embrechts, P. (2015). "Quantitative Risk
        Management: Concepts, Techniques and Tools." Princeton University Press.
        - Demarta, S., & McNeil, A.J. (2005). "The T Copula and Related
        Copulas." International Statistical Review, 73(1), 111-129.

        Also known as: copula calibration, copula fit, dependence modeling.

        Shares similar logic:
        - skfolio, GaussianCopula: https://skfolio.org/generated/skfolio.distribution.GaussianCopula.html
        - skfolio, StudentTCopula: https://skfolio.org/generated/skfolio.distribution.StudentTCopula.html
        - skfolio, ClaytonCopula: https://skfolio.org/generated/skfolio.distribution.ClaytonCopula.html
        - skfolio, GumbelCopula: https://skfolio.org/generated/skfolio.distribution.GumbelCopula.html
        - statsmodels, FrankCopula: https://www.statsmodels.org/stable/generated/statsmodels.distributions.copula.api.FrankCopula.html

        Args:
            ticker_a (str, optional): The first asset. Defaults to None, meaning every unique pair of
            tickers in the Toolkit instance is calibrated (requires `ticker_b` to also be None).
            ticker_b (str, optional): The second asset. Defaults to None, see `ticker_a`.
            copula (str, optional): The copula family to fit. Options are: "gaussian" (Gaussian copula),
                "student-t" (Student-T copula), "clayton" (Clayton copula), "gumbel" (Gumbel copula), "frank" (Frank
                copula). Defaults to "gaussian".
            period (str, optional): The data frequency. Options are: "daily", "weekly", "monthly", "quarterly",
                "yearly". Defaults to "daily", since a dependence estimate needs far more observations than a
                lower frequency provides -- at "yearly" a decade of history is only ten
                observations.
            column (str, optional): The historical data column to use. Defaults to "Return".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Raises:
            ValueError: If `copula` is not one of "gaussian", "student-t", "clayton", "gumbel" or "frank",
            if only one of `ticker_a`/`ticker_b` is given, if fewer than 2 tickers are available to pair
            up when neither is given, or if fewer than 10 paired, non-missing observations are available.

        Returns:
            pd.Series | pd.DataFrame: The fitted copula parameter(s), the Lower and Upper Tail
            Dependence implied by them, the Log-Likelihood, the AIC and the number of observations used
            -- a Series for a single given pair, or a DataFrame indexed by every (Ticker A, Ticker B)
            pair when neither ticker is given.

        Notes:
        - The method retrieves historical data based on the specified `period` and calibrates the
        requested copula between the given pair, or every pair, of tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["GCUSD", "SIUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_copula_parameters("GCUSD", "SIUSD", copula="student-t")
        ```

        Which returns:

        |                       |      Value |
        |:----------------------|-----------:|
        | Rho                   |     0.7498 |
        | Degrees of Freedom    |     8.1362 |
        | Lower Tail Dependence |     0.2821 |
        | Upper Tail Dependence |     0.2821 |
        | Log-Likelihood        |   532.413  |
        | AIC                   | -1060.83   |
        | Observations          |  1300      |
        """
        period = period if period else "daily"
        returns = self._get_price_column(period, column)

        fit_functions = {
            "gaussian": copula_model.get_gaussian_copula_parameters,
            "student-t": copula_model.get_student_t_copula_parameters,
            "clayton": copula_model.get_clayton_copula_parameters,
            "gumbel": copula_model.get_gumbel_copula_parameters,
            "frank": copula_model.get_frank_copula_parameters,
        }

        check_option("copula", copula, fit_functions)

        ticker_pairs = self._resolve_ticker_pairs(ticker_a, ticker_b)

        if len(ticker_pairs) == 1:
            ticker_a, ticker_b = ticker_pairs[0]
            result = pd.Series(
                fit_functions[copula](returns[ticker_a], returns[ticker_b])
            )

            return apply_rounding(
                result, rounding if rounding is not None else self._rounding
            )

        results = {
            (pair_a, pair_b): fit_functions[copula](returns[pair_a], returns[pair_b])
            for pair_a, pair_b in ticker_pairs
        }

        result_df = pd.DataFrame(results).T
        result_df.index.names = ["Ticker A", "Ticker B"]

        return apply_rounding(
            result_df, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_copula_simulation(
        self,
        ticker_a: str | None = None,
        ticker_b: str | None = None,
        copula: str = "gaussian",
        period: str | None = None,
        column: str = "Return",
        simulations: int = 10_000,
        seed: int = 0,
        empirical_margins: bool = True,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Simulate joint return scenarios of `ticker_a` and `ticker_b` from a bivariate copula: draws that keep
        how the two move together, including in their tails.

        The copula is calibrated as in `get_copula_parameters`.

        When `ticker_a`/`ticker_b` are not given, every unique pair among the
        Toolkit's tickers is calibrated and simulated instead, as columns
        grouped under a top-level (Ticker A, Ticker B) column per pair.

        This is what a copula-based portfolio Monte Carlo simulation needs: draws
        that preserve the calibrated dependence structure -- including tail
        dependence, for the Student-T, Clayton and Gumbel families -- rather
        than the (typically understated) joint crash risk a plain multivariate
        gaussian simulation would produce.

        The formula is as follows (with the default gaussian copula and empirical margins):

        - (Z1, Z2) ~ N(0, [[1, Rho], [Rho, 1]])
        - U = Φ(Z1), V = Φ(Z2)
        - Simulated Return A = Quantile(A, U), Simulated Return B = Quantile(B, V)

        Where Rho is fitted as in `get_copula_parameters` and Quantile is the empirical quantile (linear
        interpolation) of each asset's own history. The Student-T copula scales (Z1, Z2) by √(v / W), with
        W ~ χ²(v), and maps them with the Student-T distribution function; the Clayton, Gumbel and Frank copulas
        draw U and T uniformly and solve C(V | U) = T for V.

        Also known as: copula Monte Carlo, dependence simulation, scenario
        generation.

        Shares similar logic:
        - skfolio, GaussianCopula: https://skfolio.org/generated/skfolio.distribution.GaussianCopula.html
        - skfolio, StudentTCopula: https://skfolio.org/generated/skfolio.distribution.StudentTCopula.html
        - skfolio, ClaytonCopula: https://skfolio.org/generated/skfolio.distribution.ClaytonCopula.html
        - skfolio, GumbelCopula: https://skfolio.org/generated/skfolio.distribution.GumbelCopula.html
        - statsmodels, FrankCopula: https://www.statsmodels.org/stable/generated/statsmodels.distributions.copula.api.FrankCopula.html

        Args:
            ticker_a (str, optional): The first asset. Defaults to None, meaning every unique pair of
            tickers in the Toolkit instance is used (requires `ticker_b` to also be None).
            ticker_b (str, optional): The second asset. Defaults to None, see `ticker_a`.
            copula (str, optional): The copula family to fit and simulate from. Options are: "gaussian" (Gaussian copula),
                "student-t" (Student-T copula), "clayton" (Clayton copula), "gumbel" (Gumbel copula), "frank" (Frank
                copula). Defaults to "gaussian".
            period (str, optional): The data frequency. Options are: "daily", "weekly", "monthly", "quarterly",
                "yearly". Defaults to "daily", since a dependence estimate needs far more observations than a
                lower frequency provides -- at "yearly" a decade of history is only ten
                observations.
            column (str, optional): The historical data column to use. Defaults to "Return".
            simulations (int, optional): The number of joint draws to simulate. Defaults to 10,000.
            seed (int, optional): The seed for the random number generator. Defaults to 0.
            empirical_margins (bool, optional): Whether to map the simulated pseudo-observations back
            to realistic returns via each asset's own empirical (historical) quantile function.
            Defaults to True. When False, the raw pseudo-observations (each in (0, 1)) are returned
            instead.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Raises:
            ValueError: If `copula` is not one of "gaussian", "student-t", "clayton", "gumbel" or "frank",
            if only one of `ticker_a`/`ticker_b` is given, if fewer than 2 tickers are available to pair
            up when neither is given, or if fewer than 10 paired, non-missing observations are available.

        Returns:
            pd.DataFrame: `simulations` simulated joint draws, one column per asset (or "U"/"V" without
            empirical margins) for a single given pair, or with columns grouped by
            (Ticker A, Ticker B) pair when neither ticker is given.

        Notes:
        - The method retrieves historical data based on the specified `period`, calibrates the
        requested copula, and simulates from it, for the given pair or every pair of tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["HD", "LOW"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_copula_simulation("HD", "LOW", simulations=30).head(5)
        ```

        Which returns:

        |    |      HD |     LOW |
        |---:|--------:|--------:|
        |  0 |  0.0001 | -0.0073 |
        |  1 | -0.012  | -0.0059 |
        |  2 |  0.0332 |  0.0247 |
        |  3 |  0.0001 | -0.0019 |
        |  4 |  0.0038 | -0.0019 |
        """
        period = period if period else "daily"
        returns = self._get_price_column(period, column)

        fit_functions = {
            "gaussian": copula_model.get_gaussian_copula_parameters,
            "student-t": copula_model.get_student_t_copula_parameters,
            "clayton": copula_model.get_clayton_copula_parameters,
            "gumbel": copula_model.get_gumbel_copula_parameters,
            "frank": copula_model.get_frank_copula_parameters,
        }
        simulate_functions = {
            "gaussian": lambda parameters: copula_model.get_gaussian_copula_simulation(
                parameters["Rho"], simulations, seed
            ),
            "student-t": lambda parameters: copula_model.get_student_t_copula_simulation(
                parameters["Rho"],
                parameters["Degrees of Freedom"],
                simulations,
                seed,
            ),
            "clayton": lambda parameters: copula_model.get_clayton_copula_simulation(
                parameters["Theta"], simulations, seed
            ),
            "gumbel": lambda parameters: copula_model.get_gumbel_copula_simulation(
                parameters["Theta"], simulations, seed
            ),
            "frank": lambda parameters: copula_model.get_frank_copula_simulation(
                parameters["Theta"], simulations, seed
            ),
        }

        check_option("copula", copula, fit_functions)

        ticker_pairs = self._resolve_ticker_pairs(ticker_a, ticker_b)

        if len(ticker_pairs) == 1:
            ticker_a, ticker_b = ticker_pairs[0]
            simulation = self._simulate_copula_pair(
                ticker_a,
                ticker_b,
                returns,
                fit_functions[copula],
                simulate_functions[copula],
                empirical_margins,
            )

            return apply_rounding(
                simulation, rounding if rounding is not None else self._rounding
            )

        simulation = pd.concat(
            {
                (pair_a, pair_b): self._simulate_copula_pair(
                    pair_a,
                    pair_b,
                    returns,
                    fit_functions[copula],
                    simulate_functions[copula],
                    empirical_margins,
                )
                for pair_a, pair_b in ticker_pairs
            },
            axis=1,
        )
        simulation.columns.names = ["Ticker A", "Ticker B", None]

        return apply_rounding(
            simulation, rounding if rounding is not None else self._rounding
        )

    @staticmethod
    def _simulate_copula_pair(
        pair_a: str,
        pair_b: str,
        returns: pd.DataFrame,
        fit_function,
        simulate_function,
        empirical_margins: bool,
    ) -> pd.DataFrame:
        fitted_parameters = fit_function(returns[pair_a], returns[pair_b])
        simulation = simulate_function(fitted_parameters)

        if empirical_margins:
            simulation = pd.DataFrame(
                {
                    pair_a: np.quantile(returns[pair_a].dropna(), simulation["U"]),
                    pair_b: np.quantile(returns[pair_b].dropna(), simulation["V"]),
                }
            )

        return simulation

    @handle_errors
    def get_best_fitting_copula(
        self,
        ticker_a: str | None = None,
        ticker_b: str | None = None,
        period: str | None = None,
        column: str = "Return",
        show_full_results: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate which copula family best describes how `ticker_a` and `ticker_b` move together, by
        comparing all five supported families on their AIC (Akaike Information Criterion).

        Each family is calibrated as in `get_copula_parameters`. The lower the AIC, the better the fit relative
        to its number of parameters, so the top row is the best-fitting family.

        When `ticker_a`/`ticker_b` are not given, every unique pair among the
        Toolkit's tickers is compared instead.

        The formula is as follows:

        - AIC = 2 * k - 2 * Log-Likelihood

        Where the Log-Likelihood is that of each family fitted as in `get_copula_parameters` and k the number of
        fitted parameters (2 for the Student-T copula, 1 for the others).

        Also known as: copula selection, copula comparison.

        Shares similar logic:
        - skfolio, GaussianCopula: https://skfolio.org/generated/skfolio.distribution.GaussianCopula.html
        - skfolio, StudentTCopula: https://skfolio.org/generated/skfolio.distribution.StudentTCopula.html
        - skfolio, ClaytonCopula: https://skfolio.org/generated/skfolio.distribution.ClaytonCopula.html
        - skfolio, GumbelCopula: https://skfolio.org/generated/skfolio.distribution.GumbelCopula.html
        - statsmodels, FrankCopula: https://www.statsmodels.org/stable/generated/statsmodels.distributions.copula.api.FrankCopula.html

        Args:
            ticker_a (str, optional): The first asset. Defaults to None, meaning every unique pair of
            tickers in the Toolkit instance is compared (requires `ticker_b` to also be None).
            ticker_b (str, optional): The second asset. Defaults to None, see `ticker_a`.
            period (str, optional): The data frequency. Options are: "daily", "weekly", "monthly", "quarterly",
                "yearly". Defaults to "daily", since a dependence estimate needs far more observations than a
                lower frequency provides -- at "yearly" a decade of history is only ten
                observations.
            column (str, optional): The historical data column to use. Defaults to "Return".
            show_full_results (bool, optional): Only relevant when neither ticker is given. When False
            (the default), returns a square ticker-by-ticker grid of just the winning copula family
            per pair. When True, returns one row per pair instead, with the winning family's fitted
            parameter(s), Lower and Upper Tail Dependence, Log-Likelihood, AIC and the number of
            observations used. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Raises:
            ValueError: If only one of `ticker_a`/`ticker_b` is given, or if fewer than 2 tickers are
            available to pair up when neither is given.

        Returns:
            pd.DataFrame: For a single given pair, one row per copula family, sorted by AIC (best fit
            first), with each family's fitted parameter(s), Lower and Upper Tail Dependence,
            Log-Likelihood, AIC and the number of observations used, indexed by "Copula". When neither
            ticker is given, a square ticker-by-ticker grid of the winning copula family per pair (or,
            with full results, one row per pair instead, indexed by (Ticker A, Ticker B),
            with a "Best Copula" column and that family's fitted stats).

        Notes:
        - The method retrieves historical data based on the specified `period` and calibrates every
        supported copula family for the given pair, or every pair, of tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["V", "MA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        copulas = toolkit.risk.get_best_fitting_copula("V", "MA")

        copulas[["Lower Tail Dependence", "Upper Tail Dependence", "Log-Likelihood", "AIC"]]
        ```

        Which returns:

        | Copula    |   Lower Tail Dependence |   Upper Tail Dependence |   Log-Likelihood |      AIC |
        |:----------|------------------------:|------------------------:|-----------------:|---------:|
        | Student-T |                  0.5596 |                  0.5596 |          946.907 | -1889.81 |
        | Gaussian  |                  0      |                  0      |          912.208 | -1822.42 |
        | Gumbel    |                  0      |                  0.7388 |          892.134 | -1782.27 |
        | Frank     |                  0      |                  0      |          826.803 | -1651.61 |
        | Clayton   |                  0.7852 |                  0      |          757.472 | -1512.94 |
        """
        period = period if period else "daily"
        returns = self._get_price_column(period, column)

        fit_functions = {
            "Gaussian": copula_model.get_gaussian_copula_parameters,
            "Student-T": copula_model.get_student_t_copula_parameters,
            "Clayton": copula_model.get_clayton_copula_parameters,
            "Gumbel": copula_model.get_gumbel_copula_parameters,
            "Frank": copula_model.get_frank_copula_parameters,
        }

        ticker_pairs = self._resolve_ticker_pairs(ticker_a, ticker_b)

        if len(ticker_pairs) == 1:
            ticker_a, ticker_b = ticker_pairs[0]
            comparison_df = self._compare_copula_pair(
                ticker_a, ticker_b, returns, fit_functions
            )

            return apply_rounding(
                comparison_df, rounding if rounding is not None else self._rounding
            )

        # One row per pair: only the lowest-AIC family, which sorts first, is kept.
        comparison_df = pd.concat(
            {
                (pair_a, pair_b): self._compare_copula_pair(
                    pair_a, pair_b, returns, fit_functions
                ).iloc[[0]]
                for pair_a, pair_b in ticker_pairs
            }
        )
        comparison_df.index.names = ["Ticker A", "Ticker B", "Best Copula"]
        comparison_df = comparison_df.reset_index(level="Best Copula")

        if show_full_results:
            return apply_rounding(
                comparison_df, rounding if rounding is not None else self._rounding
            )

        tickers = sorted({ticker for pair in ticker_pairs for ticker in pair})
        grid = pd.DataFrame(index=tickers, columns=tickers, dtype=object)

        for (pair_a, pair_b), best_copula in comparison_df["Best Copula"].items():
            grid.loc[pair_a, pair_b] = best_copula
            grid.loc[pair_b, pair_a] = best_copula

        grid.index.name = "Ticker A"
        grid.columns.name = "Ticker B"

        return grid

    @staticmethod
    def _compare_copula_pair(
        pair_a: str, pair_b: str, returns: pd.DataFrame, fit_functions: dict
    ) -> pd.DataFrame:
        comparison = {
            copula_name: fit_function(returns[pair_a], returns[pair_b])
            for copula_name, fit_function in fit_functions.items()
        }

        pair_df = pd.DataFrame(comparison).T.sort_values("AIC")
        pair_df.index.name = "Copula"

        return pair_df

    @handle_errors
    def get_covar(
        self,
        ticker: str | None = None,
        conditioning_ticker: str | None = None,
        period: str | None = None,
        column: str = "Return",
        alpha: float = 0.05,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series:
        """
        Calculate the (Delta-)CoVaR of `ticker`: how much worse its Value at Risk gets when
        `conditioning_ticker` is in its own distress state, the spillover of tail risk.

        Without a pair, it measures how much each
        ticker's distress adds to the tail risk of the benchmark (the market as the
        "system"), as in Adrian & Brunnermeier (2016).

        Ordinary Value at Risk treats each asset in isolation, which misses systemic
        risk -- the fact that one asset's distress can spill over and worsen
        another's risk. CoVaR directly measures that spillover: it is the VaR of
        `ticker`, conditional on `conditioning_ticker` itself being at its own
        `alpha`-VaR, estimated via a linear Quantile Regression of `ticker`'s returns
        on `conditioning_ticker`'s returns at quantile `alpha`. The Delta-CoVaR
        isolates the marginal, distress-specific contribution by subtracting the same
        construction evaluated in the "normal" (median) state instead.

        CoVaR was introduced to measure systemic risk: in its original application `ticker`
        is a financial-system index (or a bank's peers) and `conditioning_ticker` a single
        institution, so the Delta-CoVaR reads as that institution's contribution to
        system-wide tail risk. Because the quantile regression is fitted at the `alpha`
        quantile, it needs several years of daily data, and the result is a statement about
        co-movement in the tails, not about causality.

        The formula is as follows (with the default alpha of 0.05):

        - Quantile_alpha(A | B) = Intercept + Slope * B
        - CoVaR = Intercept + Slope * VaR_B
        - Delta-CoVaR = Slope * (VaR_B - Median(B))

        Where A are the values of `ticker` and B those of `conditioning_ticker` on their common dates, the
        Intercept and Slope come from a linear quantile regression at quantile alpha (minimizing the check
        loss) and VaR_B is the empirical alpha-quantile of B.

        For more information about the method, see the following paper:

        - Adrian, T., & Brunnermeier, M.K. (2016). "CoVaR." American Economic Review,
        106(7), 1705-1741.

        Also known as: Conditional Value at Risk (systemic risk sense), Delta-CoVaR.

        Args:
            ticker (str | None, optional): The asset whose conditional VaR is being measured (the system),
            the benchmark by its own ticker (SPY) or as "Benchmark", or a ticker outside the Toolkit, whose
            returns are retrieved for the same dates (with column="Return"). Defaults to None, which uses the
            benchmark.
            conditioning_ticker (str | None, optional): The asset whose distress `ticker` is conditioned on.
            Defaults to None, which conditions `ticker` on every other ticker in the Toolkit instance in turn
            (the benchmark included when `ticker` is given), a row each.
            period (str, optional): The data frequency. Options are: "daily", "weekly", "monthly", "quarterly",
                "yearly". Defaults to "daily", since a dependence estimate needs far more observations than a
                lower frequency provides -- at "yearly" a decade of history is only ten
                observations.
            column (str, optional): The historical data column to use. Defaults to "Return".
            alpha (float, optional): The tail probability for both the tail quantile regression and
            the VaR of `conditioning_ticker` (e.g., 0.05 for the 95% confidence level). Defaults to 0.05.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Raises:
            ValueError: If one of the two tickers is missing while the Toolkit has no benchmark.

        Returns:
            pd.Series | pd.DataFrame: The CoVaR, the Delta-CoVaR, the tail (alpha-quantile) Quantile Regression
            slope and intercept, and the number of observations used -- a Series for a given pair, or a
            DataFrame with a row per conditioning ticker otherwise.

        Notes:
        - The method retrieves historical data based on the specified `period` for each pair of
        assets and estimates the CoVaR of `ticker` conditional on `conditioning_ticker`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["GS", "JPM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_covar()[["CoVaR", "Delta-CoVaR"]]
        ```

        Which returns:

        |     |   CoVaR |   Delta-CoVaR |
        |:----|--------:|--------------:|
        | GS  | -0.0232 |       -0.0104 |
        | JPM | -0.0247 |       -0.012  |
        """
        validate_alpha(alpha)
        period = period if period else "daily"
        returns = self._get_price_column(period, column)
        # The conditioning ticker varies and the system (`ticker`) defaults to the benchmark.
        pairs = self._benchmark_pairs(conditioning_ticker, ticker)
        if column == "Return":
            returns = self._add_other_returns(
                returns, [name for pair in pairs for name in pair], period
            )

        result = pd.DataFrame(
            {
                conditioning: covar_model.get_covar(
                    returns[system], returns[conditioning], alpha=alpha
                )
                for conditioning, system in pairs
            }
        ).T

        if ticker is not None and conditioning_ticker is not None:
            result = result.iloc[0].rename(None)

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    def _market_pairs(
        self, ticker: str | None, market_ticker: str | None
    ) -> tuple[list[str], str]:
        # The firms and the market of a systemic risk measure: every ticker against the benchmark by default. With another market (^GSPC) the benchmark is a market index too, not a firm.  # noqa: E501
        pairs = self._benchmark_pairs(ticker, market_ticker)
        firms = [firm for firm, _ in pairs if ticker is not None or firm != "Benchmark"]

        return firms, pairs[0][1]

    @handle_errors
    def get_marginal_expected_shortfall(
        self,
        ticker: str | None = None,
        market_ticker: str | None = None,
        period: str | None = None,
        alpha: float = 0.05,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Marginal Expected Shortfall (MES) of each ticker per period: its average
        return on the days the market (the benchmark) is in its worst `alpha` tail.

        Acharya, Pedersen, Philippon and Richardson (2017) show that a financial institution's
        contribution to the Expected Shortfall of the system is its MES, the return it is expected
        to make when the system as a whole is in its tail. Measured over a year of daily returns
        on the 5% worst days of the market, the MES of the financial firms before 2007 predicted
        which of them lost the most in the crisis. Where the CoVaR asks how much the system
        suffers when a firm is in distress, the MES asks how much a firm suffers when the system
        is; it is the market component of SRISK (see `get_srisk`).

        The formula is as follows (with the default alpha of 0.05):

        - VaR_alpha(Market) = Quantile(Market Returns within the period, alpha)
        - MES = MEAN(Returns_t | Market Return_t <= VaR_alpha(Market))

        Where the returns are the daily returns of the Adjusted Close within the period on the days the
        ticker and the market both have a return, and the quantile interpolates linearly between the two
        nearest returns. The MES is reported as a return, negative for a loss, like the Value at Risk and
        the Conditional Value at Risk (the paper reports the loss, -MES). With `rolling`, the days of the
        last `rolling` periods are used together.

        For more information about the method, see the following paper:

        - Acharya, V.V., Pedersen, L.H., Philippon, T., & Richardson, M. (2017). "Measuring Systemic
        Risk." The Review of Financial Studies, 30(1), 2-47.

        Also known as: MES, marginal expected shortfall, systemic expected shortfall, tail beta.

        Args:
            ticker (str | None, optional): The ticker whose MES to calculate. Defaults to None, which
                uses every ticker except the market and the benchmark.
            market_ticker (str | None, optional): The ticker that represents the market (the system): the
                benchmark by its own ticker (SPY) or as "Benchmark", one of the tickers, or a ticker outside
                the Toolkit (such as ^GSPC), whose returns are retrieved for the same dates. Defaults to None,
                which uses the benchmark.
            period (str, optional): The period of each estimate. Options are: "monthly", "quarterly",
                "yearly". Defaults to "yearly", as in the paper.
            alpha (float, optional): The tail probability of the market returns. Defaults to 0.05, the
                5% worst days of the market.
            rolling (int, optional): Use the daily returns of the last `rolling` periods instead of one
                period (e.g. period='monthly' and rolling=12 for a one-year window at every month end).
                At least 2. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the MES over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Raises:
            ValueError: If the period is not monthly, quarterly or yearly, or if the Toolkit has no
                benchmark and no market_ticker is given.

        Returns:
            pd.DataFrame: The MES as a return, with the periods as the index and the tickers as columns.

        Notes:
        - Each ticker is paired with the market on the days both have a return: a market return on a day the
        ticker did not trade is compounded into the next shared date, so the other tickers of the Toolkit (e.g.
        a cryptocurrency that trades every day) do not change a ticker's MES.
        - A period with fewer than 1 / alpha shared returns (20 at an alpha of 0.05) is NaN, as its tail would hold
        less than one day on average.
        - The market itself is left out: its MES is the average of its own worst days, its Conditional Value at
        Risk.
        - Closes on different exchanges are not synchronous (BNP.PA closes hours before SPY), so a market fall
        after the ticker's close shows up in its next day and the MES understates the tail dependence; use a
        local market index as market_ticker for such tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["JPM", "BAC", "C", "GS"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_marginal_expected_shortfall()
        ```

        Which returns:

        | Date   |     JPM |     BAC |       C |      GS |
        |:-------|--------:|--------:|--------:|--------:|
        | 2021   | -0.0126 | -0.0134 | -0.0152 | -0.0173 |
        | 2022   | -0.0282 | -0.0315 | -0.029  | -0.0284 |
        | 2023   | -0.0155 | -0.0205 | -0.0187 | -0.017  |
        | 2024   | -0.02   | -0.0196 | -0.0232 | -0.0233 |
        | 2025   | -0.029  | -0.0339 | -0.0393 | -0.0394 |
        """
        validate_alpha(alpha)
        period = period if period else "yearly"

        check_period(period, ["monthly", "quarterly", "yearly"])
        check_rolling(rolling, self._historical_data, period)

        tickers, market = self._market_pairs(ticker, market_ticker)
        returns = self._add_other_returns(
            self._historical_data["daily"]["Return"], [market], "daily"
        )

        marginal_expected_shortfall = covar_model.get_marginal_expected_shortfall(
            returns[tickers], returns[market], alpha, period, rolling
        )

        marginal_expected_shortfall = marginal_expected_shortfall.loc[
            self._start_date : self._end_date
        ]

        return finalize_dataset(
            dataset=marginal_expected_shortfall,
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

    @handle_errors
    def get_srisk(
        self,
        ticker: str | None = None,
        market_ticker: str | None = None,
        period: str | None = None,
        decline: float = 0.4,
        capital_ratio: float = 0.08,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate SRISK, the capital each firm is expected to need in a crisis, per period, with
        the Long-Run Marginal Expected Shortfall (LRMES) it is based on and each firm's share of
        the total.

        Brownlees and Engle (2017) define SRISK as the capital shortfall of a firm conditional
        on a severe market decline: in a crisis the market falls by 40% over six months, the
        firm's equity falls by its LRMES, and the firm has to hold a prudential share k of its
        assets (8% for U.S. firms) as equity. The firms with the largest SRISK contribute most to
        the undercapitalization of the financial system in a crisis, which NYU Stern's V-Lab
        publishes as its systemic risk ranking (SRISK%). It is meant for financial firms, whose
        liabilities are mostly debt; for other firms the liabilities include operating items.

        The formula is as follows (with the default decline of 0.4 and capital ratio of 0.08):

        - Beta = Covariance(Returns, Market Returns) / Variance(Market Returns)
        - LRMES = (1 - decline) ^ Beta - 1
        - SRISK = k * Total Liabilities - (1 - k) * (1 + LRMES) * Market Cap
        - SRISK Share = MAX(SRISK, 0) / SUM(MAX(SRISK, 0) of every ticker)

        Where Beta is the market beta from the daily returns of the Adjusted Close within the period on the
        days the ticker and the market (the benchmark) both have a return, k the capital ratio, Total
        Liabilities those of the balance sheet and Market Cap the Close on the ticker's last trading day of the
        period times the Weighted Average Shares Diluted. The LRMES is the approximation V-Lab
        publishes (1 - exp(log(1 - d) * Beta)), reported as a return, negative for a loss, like the Marginal
        Expected Shortfall; 1 + LRMES is the share of the equity left after the crisis. With `rolling`, the beta
        uses the days of the last `rolling` periods.

        For more information about the method, see the following papers:

        - Brownlees, C., & Engle, R.F. (2017). "SRISK: A Conditional Capital Shortfall Measure of Systemic
        Risk." The Review of Financial Studies, 30(1), 48-79.
        - Acharya, V.V., Engle, R.F., & Richardson, M. (2012). "Capital Shortfall: A New Approach to Ranking
        and Regulating Systemic Risks." American Economic Review, 102(3), 59-64.

        Also known as: systemic risk, capital shortfall, SRISK%, long-run marginal expected shortfall,
        LRMES.

        Shares similar logic:
        - empyrical, beta (Beta): https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L1314

        Args:
            ticker (str | None, optional): The ticker whose SRISK to calculate. Defaults to None, which
                uses every ticker except the market and the benchmark.
            market_ticker (str | None, optional): The ticker that represents the market: the benchmark by
                its own ticker (SPY) or as "Benchmark", one of the tickers, or a ticker outside the Toolkit
                (such as ^GSPC), whose returns are retrieved for the same dates. Defaults to None, which uses
                the benchmark.
            period (str, optional): The period of each estimate. Options are: "monthly", "quarterly",
                "yearly". Defaults to "yearly".
            decline (float, optional): The fall of the market over six months that defines the crisis, as a
                decimal. Defaults to 0.4, as in V-Lab.
            capital_ratio (float, optional): The prudential capital ratio k, the share of the assets a firm
                has to hold as equity. Defaults to 0.08, V-Lab's ratio for U.S. firms (5.5% for European
                firms, whose IFRS balance sheets report more liabilities).
            rolling (int, optional): Estimate the beta from the daily returns of the last `rolling` periods
                instead of one period (e.g. period='monthly' and rolling=12 for a one-year beta at every
                month end). At least 2. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the results over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Raises:
            ValueError: If the period is not monthly, quarterly or yearly, or if the Toolkit has no
                benchmark and no market_ticker is given.

        Returns:
            pd.DataFrame: The Beta, LRMES, Total Liabilities, Market Cap, SRISK and SRISK Share, with the
            periods as the index and (item, ticker) as columns. SRISK, the liabilities and the market cap are
            in currency units, the others decimals.

        Notes:
        - SRISK is positive for a capital shortfall and negative for a surplus. Each firm keeps its signed
        SRISK; the SRISK Share (V-Lab's SRISK%) only adds the shortfalls, as a surplus does not offset another
        firm's shortfall in a crisis, so a firm with a surplus has a share of 0.
        - The liabilities and the share count are those of the last fiscal period that ended on or before the
        last calendar day of the period (filed on or before it with point_in_time), so a fiscal year that ends
        on a weekend still belongs to its own year and a period never uses figures that were not yet reported.
        A quarterly Toolkit uses the latest quarter, as V-Lab does. The market cap is the traded
        (split-adjusted) Close times the diluted share count, which for periods that coincide with the fiscal
        periods equals `toolkit.ratios.get_market_cap`.
        - The Close is in the currency the ticker trades in, while the statements are in the currency they
        are reported in unless the Toolkit converts them to the currency of the prices (currency conversion);
        when the two differ without a conversion, a warning is logged.
        - V-Lab estimates the beta with a GJR-GARCH volatility and a Dynamic Conditional Correlation (DCC)
        model, a beta for every day, while this beta is the ordinary least squares beta over the period's daily
        returns, so the figures are of the same order but do not match V-Lab's.
        - The beta needs at least 15 shared returns in the window; the market and tickers without financial
        statements (such as an index or a cryptocurrency) are left out or NaN.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["JPM", "BAC", "C"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
            quarterly=True,
        )

        toolkit.risk.get_srisk().xs("C", level=1, axis=1)
        ```

        Which returns:

        | Date   |   Beta |   LRMES |   Total Liabilities |   Market Cap |       SRISK |   SRISK Share |
        |:-------|-------:|--------:|--------------------:|-------------:|------------:|--------------:|
        | 2021   | 0.9001 | -0.3686 |         2.08874e+12 |  1.20877e+11 | 9.68806e+10 |        0.832  |
        | 2022   | 0.9657 | -0.3894 |         2.21484e+12 |  8.84654e+10 | 1.27491e+11 |        0.544  |
        | 2023   | 1.1149 | -0.4342 |         2.20558e+12 |  9.8235e+10  | 1.25313e+11 |        0.5676 |
        | 2024   | 1.0891 | -0.4267 |         2.14358e+12 |  1.35923e+11 | 9.97953e+10 |        0.7761 |
        | 2025   | 1.2277 | -0.4659 |         2.44338e+12 |  2.17347e+11 | 8.86712e+10 |        0.8611 |
        """
        period = period if period else "yearly"

        check_period(period, ["monthly", "quarterly", "yearly"])
        check_rolling(rolling, self._historical_data, period)
        if self._capital_shortfall_function is None:
            raise ValueError(
                "SRISK needs the Total Liabilities and share count of the financial statements, "
                "which toolkit.risk of a Toolkit provides."
            )

        tickers, market = self._market_pairs(ticker, market_ticker)
        returns = self._add_other_returns(
            self._historical_data["daily"]["Return"], [market], "daily"
        )

        beta = covar_model.get_market_beta(
            returns[tickers], returns[market], period, rolling
        )
        if beta.dropna(how="all").empty:
            raise ValueError(
                f"The data holds fewer than the {rolling} {period} periods of daily returns the "
                "rolling market beta needs."
                if rolling
                else f"The data holds no {period} period with daily returns of both the "
                f"tickers and {market} between {self._start_date} and {self._end_date}."
            )
        long_run_marginal_expected_shortfall = (
            covar_model.get_long_run_marginal_expected_shortfall(beta, decline)
        )

        liabilities, liability_dates, shares, share_dates = (
            self._capital_shortfall_function()
        )
        close = self._get_prices("Close").reindex(columns=tickers)

        # The Close is NaN on the days a ticker did not trade, so this is the close of each ticker's last trading day of the period.  # noqa: E501
        period_close = (
            close.groupby(close.index.asfreq(PERIOD_TRANSLATION[period]))
            .last()
            .reindex(beta.index)
        )

        # The figures are those of the last fiscal period that ended (or, with point_in_time, was filed) on or before the last calendar day of the period, not its last trading day: a fiscal year ending on Saturday 31 December 2022 belongs to 2022 although the last trading day is the 30th. The current period stops at the end date of the Toolkit, so no figures of a later day are used.  # noqa: E501
        end_date = pd.Timestamp(self._end_date)
        period_ends = pd.DatetimeIndex(
            [
                min(end, end_date)
                for end in beta.index.to_timestamp(how="end").normalize()
            ]
        )
        period_liabilities, period_shares = (
            map_period_data_to_daily_data(
                period_data=figures,
                daily_dates=period_ends,
                quarterly=self._quarterly,
                period_end_dates=dates,
            )
            .set_axis(beta.index, axis=0)
            .reindex(columns=tickers)
            .where(period_close.notna())
            for figures, dates in (
                (liabilities, liability_dates),
                (shares, share_dates),
            )
        )

        period_market_cap = valuation_model.get_market_cap(period_close, period_shares)

        srisk = covar_model.get_srisk(
            period_liabilities,
            period_market_cap,
            long_run_marginal_expected_shortfall,
            capital_ratio,
        )

        result = pd.concat(
            {
                "Beta": beta,
                "LRMES": long_run_marginal_expected_shortfall,
                "Total Liabilities": period_liabilities,
                "Market Cap": period_market_cap,
                "SRISK": srisk,
                "SRISK Share": covar_model.get_srisk_share(srisk),
            },
            axis=1,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=result,
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
