"""Dependence Risk Module"""

__docformat__ = "google"

import itertools

import numpy as np
import pandas as pd

from financetoolkit.risk.base.base_controller import RiskBase
from financetoolkit.risk.dependence import copula_model, covar_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import (
    apply_rounding,
)

# ruff: noqa: E501

MINIMUM_TICKERS_FOR_ALL_PAIRS = 2


@filter_shown_columns
@track_sources
class DependenceRisk(RiskBase):
    """
    Dependence between assets: copula fits, simulations and the best fitting copula, the tail dependence coefficient and CoVaR.
    """

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
            raise ValueError(
                "Either provide both ticker_a and ticker_b, or neither (to use "
                "every pair of tickers in the Toolkit instance)."
            )

        return [(ticker_a, ticker_b)]

    def _benchmark_pairs(
        self, varying: str | None, counterpart: str | None
    ) -> list[tuple[str, str]]:
        # The (varying, counterpart) pairs of a pairwise measure: the counterpart defaults to the benchmark and the varying side to every other ticker, so a given ticker keeps the role of its own parameter.  # noqa: E501
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

    @handle_errors
    def get_tail_dependence_coefficient(
        self,
        ticker_a: str | None = None,
        ticker_b: str | None = None,
        period: str | None = None,
        column: str = "Return",
        q: float = 0.95,
        method: str = "empirical",
        dof: float = 4.0,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series:
        """
        Calculate the Upper and Lower Tail Dependence Coefficients between `ticker_a`
        and `ticker_b`. Without a pair, every ticker is paired with the benchmark, which
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

        The formula is as follows (with the default method="empirical" and q=0.95):

        - Upper Tail Dependence = #(A > Q_A(q) and B > Q_B(q)) / #(A > Q_A(q))
        - Lower Tail Dependence = #(A <= Q_A(1 - q) and B <= Q_B(1 - q)) / #(A <= Q_A(1 - q))

        Where A and B are the values of `ticker_a` and `ticker_b` on their common dates and Q_A and Q_B their
        empirical quantiles. With method="gaussian" both coefficients are 0 (1 only for a correlation of exactly
        1), and with method="student-t" both are 2 * t_(v+1)(-√((v + 1) * (1 - rho) / (1 + rho))), with v = `dof`,
        rho the Pearson correlation and t_(v+1) the Student-T distribution function.

        Also known as: tail dependence, extremal dependence coefficient.

        Args:
            ticker_a (str | None, optional): The first asset. Defaults to None, which pairs every other ticker
            in the Toolkit instance (the benchmark included when `ticker_b` is given) with `ticker_b`.
            ticker_b (str | None, optional): The second asset. Defaults to None, which uses the benchmark.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily", since a dependence estimate needs far more observations than a
                lower frequency provides -- at "yearly" a decade of history is only ten
                observations.
            column (str, optional): The historical data column to use. Defaults to "Return", since
            tail dependence between return series is the standard risk management application.
            q (float, optional): The threshold quantile used for the "empirical" method, in (0.5, 1).
            Defaults to 0.95.
            method (str, optional): The estimation method, one of "empirical", "gaussian" or
            "student-t". Defaults to "empirical".
            dof (float, optional): The degrees of freedom of the Student-T copula, only used when
            `method="student-t"`. Defaults to 4.0.
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
        Calibrate a bivariate copula between `ticker_a` and `ticker_b` on the
        pseudo-observations (ranks) of their returns: the Clayton, Gumbel and Frank
        parameters by maximum likelihood, the gaussian correlation from the normal
        scores, and the Student-T correlation from Kendall's tau with the degrees of
        freedom by maximum likelihood (Demarta & McNeil, 2005).

        When `ticker_a`/`ticker_b` are not given, every unique pair among the
        Toolkit's tickers is calibrated instead.

        A copula separates the dependence structure between two assets from
        their individual (marginal) return distributions (Sklar's theorem),
        letting the two be modeled independently. Five families are supported,
        each capturing a different shape of dependence:

        - "gaussian": zero tail dependence -- crashes and rallies are no more
        likely to happen together than the correlation alone implies. Included
        mainly as a baseline, since real asset returns typically show more
        joint tail risk than this.
        - "student-t": symmetric, nonzero tail dependence in both tails.
        - "clayton": nonzero *lower* tail dependence only -- assets crash
        together more than they rally together. The most common choice for
        equity return pairs.
        - "gumbel": nonzero *upper* tail dependence only -- assets rally
        together more than they crash together.
        - "frank": zero tail dependence in both tails, but (unlike gaussian) can
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

        Args:
            ticker_a (str, optional): The first asset. Defaults to None, meaning every unique pair of
            tickers in the Toolkit instance is calibrated (requires `ticker_b` to also be None).
            ticker_b (str, optional): The second asset. Defaults to None, see `ticker_a`.
            copula (str, optional): The copula family to fit, one of "gaussian",
            "student-t", "clayton", "gumbel" or "frank". Defaults to "gaussian".
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily", since a dependence estimate needs far more observations than a
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

        if copula not in fit_functions:
            raise ValueError(f"copula must be one of {', '.join(fit_functions)}.")

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
        n_simulations: int = 10_000,
        random_state: int = 42,
        empirical_margins: bool = True,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calibrate a bivariate copula (see `get_copula_parameters`) between
        `ticker_a` and `ticker_b`, and simulate joint scenarios from it.

        When `ticker_a`/`ticker_b` are not given, every unique pair among the
        Toolkit's tickers is calibrated and simulated instead, as columns
        grouped under a top-level (Ticker A, Ticker B) column per pair.

        This is what a copula-based portfolio Monte Carlo simulation needs: draws
        that preserve the calibrated dependence structure -- including tail
        dependence, for the "student-t", "clayton" and "gumbel" families -- rather
        than the (typically understated) joint crash risk a plain multivariate
        gaussian simulation would produce.

        The formula is as follows (with the default copula="gaussian" and empirical_margins=True):

        - (Z1, Z2) ~ N(0, [[1, Rho], [Rho, 1]])
        - U = Φ(Z1), V = Φ(Z2)
        - Simulated Return A = Quantile(A, U), Simulated Return B = Quantile(B, V)

        Where Rho is fitted as in `get_copula_parameters` and Quantile is the empirical quantile (linear
        interpolation) of each asset's own history. The Student-T copula scales (Z1, Z2) by √(v / W), with
        W ~ χ²(v), and maps them with the Student-T distribution function; the Clayton, Gumbel and Frank copulas
        draw U and T uniformly and solve C(V | U) = T for V.

        Also known as: copula Monte Carlo, dependence simulation, scenario
        generation.

        Args:
            ticker_a (str, optional): The first asset. Defaults to None, meaning every unique pair of
            tickers in the Toolkit instance is used (requires `ticker_b` to also be None).
            ticker_b (str, optional): The second asset. Defaults to None, see `ticker_a`.
            copula (str, optional): The copula family to fit and simulate from, one of "gaussian",
            "student-t", "clayton", "gumbel" or "frank". Defaults to "gaussian".
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily", since a dependence estimate needs far more observations than a
                lower frequency provides -- at "yearly" a decade of history is only ten
                observations.
            column (str, optional): The historical data column to use. Defaults to "Return".
            n_simulations (int, optional): The number of joint draws to simulate. Defaults to 10,000.
            random_state (int, optional): The seed for the random number generator. Defaults to 42.
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
            pd.DataFrame: `n_simulations` simulated joint draws, one column per asset (or "U"/"V" when
            `empirical_margins=False`) for a single given pair, or with columns grouped by
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

        toolkit.risk.get_copula_simulation("HD", "LOW", n_simulations=30).head(5)
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
                parameters["Rho"], n_simulations, random_state
            ),
            "student-t": lambda parameters: copula_model.get_student_t_copula_simulation(
                parameters["Rho"],
                parameters["Degrees of Freedom"],
                n_simulations,
                random_state,
            ),
            "clayton": lambda parameters: copula_model.get_clayton_copula_simulation(
                parameters["Theta"], n_simulations, random_state
            ),
            "gumbel": lambda parameters: copula_model.get_gumbel_copula_simulation(
                parameters["Theta"], n_simulations, random_state
            ),
            "frank": lambda parameters: copula_model.get_frank_copula_simulation(
                parameters["Theta"], n_simulations, random_state
            ),
        }

        if copula not in fit_functions:
            raise ValueError(f"copula must be one of {', '.join(fit_functions)}.")

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
        Calibrate all five supported copula families (see `get_copula_parameters`)
        between `ticker_a` and `ticker_b`, and compare them by AIC (Akaike
        Information Criterion) -- the lower the AIC, the better the fit relative
        to its number of parameters, so the top row is the best-fitting family.

        When `ticker_a`/`ticker_b` are not given, every unique pair among the
        Toolkit's tickers is compared instead.

        The formula is as follows:

        - AIC = 2 * k - 2 * Log-Likelihood

        Where the Log-Likelihood is that of each family fitted as in `get_copula_parameters` and k the number of
        fitted parameters (2 for the Student-T copula, 1 for the others).

        Also known as: copula selection, copula comparison.

        Args:
            ticker_a (str, optional): The first asset. Defaults to None, meaning every unique pair of
            tickers in the Toolkit instance is compared (requires `ticker_b` to also be None).
            ticker_b (str, optional): The second asset. Defaults to None, see `ticker_a`.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily", since a dependence estimate needs far more observations than a
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
            with `show_full_results=True`, one row per pair instead, indexed by (Ticker A, Ticker B),
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
        Calculate the (Delta-)CoVaR of `ticker` conditional on `conditioning_ticker`
        being in its own distress state. Without a pair, it measures how much each
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

        The formula is as follows (with the default alpha=0.05):

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
            ticker (str | None, optional): The asset whose conditional VaR is being measured (the system).
            Defaults to None, which uses the benchmark.
            conditioning_ticker (str | None, optional): The asset whose distress `ticker` is conditioned on.
            Defaults to None, which conditions `ticker` on every other ticker in the Toolkit instance in turn
            (the benchmark included when `ticker` is given), a row each.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily", since a dependence estimate needs far more observations than a
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
        period = period if period else "daily"
        returns = self._get_price_column(period, column)
        # The conditioning ticker varies and the system (`ticker`) defaults to the benchmark.
        pairs = self._benchmark_pairs(conditioning_ticker, ticker)

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
