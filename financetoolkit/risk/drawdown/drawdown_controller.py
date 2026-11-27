"""Drawdown Risk Module"""

__docformat__ = "google"

from typing import Literal

import pandas as pd

from financetoolkit.helpers import handle_portfolio
from financetoolkit.risk import risk_model
from financetoolkit.risk.drawdown import drawdown_model
from financetoolkit.risk.helpers import check_rolling
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import (
    InvalidArgumentError,
    check_period,
    handle_errors,
    validate_alpha,
)
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import finalize_dataset

# ruff: noqa: E501


@filter_shown_columns
@track_sources
class DrawdownRisk:
    """
    Drawdowns: the maximum drawdown, its duration and recovery time, the drawdown series, table and statistics, the average drawdown, the drawdown at risk and its conditional and entropic versions, the conditional expected drawdown, the ulcer index and the pain index.
    """

    @handle_portfolio
    @handle_errors
    def get_conditional_drawdown_at_risk(
        self,
        period: str | None = None,
        alpha: float = 0.05,
        within_period: bool = False,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        compound_returns: bool = True,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Conditional Drawdown at Risk (CDaR) of an investment portfolio or asset's returns: the
        average of the deepest drawdowns, those beyond the Drawdown at Risk.

        Conditional Drawdown at Risk (CDaR) extends the concept of Value at Risk and Conditional Value at
        Risk to the drawdown series instead of the return series. It is calculated as the average of the
        worst drawdowns that exceed the Drawdown at Risk (DaR), i.e. the alpha-quantile of the drawdown
        distribution, giving insight into the depth of the most severe drawdowns an investment portfolio or
        asset could experience.

        The formula is as follows (with the default alpha of 0.05):

        - Wealth(t) = Π (1 + Return(s)) for s <= t
        - Drawdown(t) = Wealth(t) / max(1, max(Wealth(s) for s <= t)) - 1
        - DaR = Quantile(Drawdown, alpha)
        - CDaR = Mean(Drawdown | Drawdown <= DaR)

        Every period counts towards the drawdown distribution, including those at a high (a drawdown of 0), the
        quantile interpolates linearly, and Wealth starts at 1, so a decline from the start of the window counts.
        Without compounding the drawdowns are measured on the summed returns instead, as an absolute decline
        from their peak (Drawdown(t) = Sum(t) - max(0, max(Sum(s) for s <= t)), with Sum(t) = Σ Return(s)), the
        uncompounded drawdown of Chekhlov, Uryasev and Zabarankin (2005).

        For more information, see the following sources:

        - Chekhlov, A., Uryasev, S. and Zabarankin, M. (2005). "Drawdown Measure in Portfolio Optimization."
        International Journal of Theoretical and Applied Finance, 8(1), 13-58.

        Also known as: CDaR, uncompounded CDaR, absolute CDaR.

        Shares similar logic:
        - Riskfolio-Lib, CDaR_Rel (compounded): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.CDaR_Rel
        - Riskfolio-Lib, CDaR_Abs (uncompounded): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.CDaR_Abs
        - skfolio, cdar: https://skfolio.org/generated/skfolio.measures.cdar.html
        - PerformanceAnalytics (R), CDD: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/maxDrawdown.R#L106

        Args:
            period (str, optional): The data frequency for returns. Options are: "daily", "weekly", "monthly",
                "quarterly", "yearly". Defaults to "daily".
            alpha (float, optional): The tail probability for the CDaR calculation (e.g., 0.05 for the 95%
            confidence level). Defaults to 0.05.
            within_period (bool, optional): Whether to calculate CDaR within the specified period or for the entire
            period. Thus whether to look at the CDaR within a specific year (if period = 'yearly') or look at the entirety
            of all years. Defaults to False. Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rolling (int, optional): The rolling window size to use for the calculation. If set, CDaR is
            calculated over a rolling window of this many periods across the full return history instead
            of per `period`. At least 2. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the CDaR values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            compound_returns (bool, optional): Whether to compound the returns into the wealth path the drawdowns are
            measured on, or to sum them (absolute drawdowns). Defaults to True.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: CDaR values with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates CDaR for each
        asset in the Toolkit instance.
        - With growth, the method calculates the growth of CDaR values using the specified `lag`.
        - Riskfolio-Lib's CDaR_Rel and CDaR_Abs and skfolio's cdar take the exact tail average of Rockafellar and
        Uryasev (2002), which counts the drawdown at the boundary with a fraction, and report the loss as a
        positive number; the Toolkit averages the drawdowns up to the interpolated quantile, as its historic
        CVaR does, which differs from them by up to 2e-4 on three years of daily returns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["^STOXX50E", "^N225", "^GSPC"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_conditional_drawdown_at_risk(period="yearly", within_period=True)
        ```

        Which returns:

        |      |   ^STOXX50E |   ^N225 |   ^GSPC |   Benchmark |
        |:-----|------------:|--------:|--------:|------------:|
        | 2021 |     -0.0629 | -0.1002 | -0.0421 |     -0.0414 |
        | 2022 |     -0.2395 | -0.131  | -0.2429 |     -0.2341 |
        | 2023 |     -0.091  | -0.0816 | -0.0833 |     -0.0805 |
        | 2024 |     -0.0826 | -0.1616 | -0.0585 |     -0.0576 |
        | 2025 |     -0.1185 | -0.1602 | -0.1445 |     -0.1432 |
        """
        validate_alpha(alpha)
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
            conditional_drawdown_at_risk = risk_model.get_conditional_drawdown_at_risk(
                returns, alpha, compound_returns=compound_returns, window=rolling
            )
        else:
            returns = self._get_returns(period, within_period)

            conditional_drawdown_at_risk = risk_model.get_conditional_drawdown_at_risk(
                returns, alpha, compound_returns=compound_returns
            )

        if rolling or within_period:
            conditional_drawdown_at_risk = conditional_drawdown_at_risk.loc[
                self._start_date : self._end_date
            ]

        return finalize_dataset(
            dataset=conditional_drawdown_at_risk,
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
    def get_conditional_expected_drawdown(
        self,
        period: str | None = None,
        window: int = 21,
        alpha: float = 0.05,
        within_period: bool = False,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        compound_returns: bool = True,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Conditional Expected Drawdown (CED) of an investment portfolio or asset's returns: the
        average of the worst Maximum Drawdowns over paths of a fixed length.

        The Maximum Drawdown of the whole history is a single number that depends on how long the history is.
        Goldberg and Mahmoud (2017) fix the length instead: the Maximum Drawdown of every path of `window`
        returns forms a distribution, and the CED is the tail average of it, as the CVaR is of the returns.
        With a window of 21 daily returns it reads as how deep the decline within a bad month gets, on average
        over the worst 5% of months. Where the Conditional Drawdown at Risk averages the drawdown of single
        days, the CED averages the worst decline of whole paths, and unlike the Maximum Drawdown it is a tail
        average, so a single crash moves it less and it does not grow with the length of the data.

        The formula is as follows (with the default window of 21 and alpha of 0.05):

        - Wealth(t) = Π (1 + Return(s)) for s <= t, starting at 1 before the first return of each path
        - Maximum Drawdown(s) = min over t in [s, s + window) of Wealth(t) / max(1, max(Wealth(u) for u <= t)) - 1
        - CED = (Σ_{i <= k} MDD(i) + (m · alpha − k) · MDD(k + 1)) / (m · alpha)

        Where MDD(1) <= MDD(2) <= ... are the m Maximum Drawdowns in ascending order and k = floor(m · alpha).
        The paths overlap, one starting at every return, so n returns give m = n - window + 1 paths. The tail
        average is the exact one of Rockafellar and Uryasev (2002), the average of the deepest m · alpha paths
        with the one at the boundary counted with its fraction. Overlapping paths often share their Maximum
        Drawdown, so a cut at an interpolated quantile, as the Toolkit's CDaR and historic CVaR make, would take
        or leave such ties on the last digit of the rounding. Without compounding the drawdowns are measured on the summed returns instead, as an absolute
        decline from their peak.

        For more information, see the following sources:

        - Goldberg, L.R. and Mahmoud, O. (2017). "Drawdown: From Practice to Theory and Back Again." Mathematics
        and Financial Economics, 11(3), 275-297.

        Also known as: CED, conditional expected drawdown, expected maximum drawdown in the tail.

        Shares similar logic:
        - RiskLabAI, conditional_expected_drawdown: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/RiskLabAI/backtest/robust_statistics.py#L88

        Args:
            period (str, optional): The data frequency for returns. Options are: "daily", "weekly", "monthly",
                "quarterly", "yearly". Defaults to "daily".
            window (int, optional): The length of each path in returns of `period`. Defaults to 21, a month of
                trading days.
            alpha (float, optional): The tail probability for the CED calculation (e.g., 0.05 for the worst 5% of
            paths). Defaults to 0.05.
            within_period (bool, optional): Whether to calculate the CED within the specified period or for the
            entire period. Thus whether to look at the CED within a specific year (if period = 'yearly') or look at
            the entirety of all years. Defaults to False. Note that this requires intraday data when period =
            'daily', since a single day only nests observations when intraday data was fetched.
            rolling (int, optional): The rolling window size to use for the calculation. If set, the CED is
            calculated over a rolling window of this many periods across the full return history instead
            of per `period`; it needs to be at least `window`. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the CED values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            compound_returns (bool, optional): Whether to compound the returns into the wealth path the drawdowns are
            measured on, or to sum them (absolute drawdowns). Defaults to True.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series | pd.DataFrame: CED values, negative (0 without a decline), per ticker, or per period (or
                date) and ticker.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates the CED
        for each asset in the Toolkit instance, each on its own returns. A period with fewer than `window`
        returns has no path and gives NaN.
        - RiskLabAI's conditional_expected_drawdown takes alpha as the confidence level (0.90 for the worst 10%)
        and reports the drawdown as a positive number; the Toolkit takes the tail probability, as for the
        CDaR, and keeps the sign of a return.
        - With fewer than 1 / alpha paths the tail holds less than one path and the CED is the deepest
        Maximum Drawdown of any path.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["^STOXX50E", "^N225", "^GSPC"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_conditional_expected_drawdown(period="yearly", within_period=True)
        ```

        Which returns:

        |      |   ^STOXX50E |   ^N225 |   ^GSPC |   Benchmark |
        |:-----|------------:|--------:|--------:|------------:|
        | 2021 |     -0.0769 | -0.1009 | -0.0464 |     -0.0454 |
        | 2022 |     -0.1575 | -0.1059 | -0.1269 |     -0.1263 |
        | 2023 |     -0.0646 | -0.0886 | -0.066  |     -0.0651 |
        | 2024 |     -0.0844 | -0.242  | -0.0799 |     -0.0792 |
        | 2025 |     -0.1587 | -0.1812 | -0.1374 |     -0.1372 |
        """
        validate_alpha(alpha)
        period = period if period else "daily"
        # A rolling window holds at least one path of `window` returns.
        returns = self._get_measure_returns(
            period, within_period, rolling, growth, max(window, 2)
        )

        conditional_expected_drawdown = (
            drawdown_model.get_conditional_expected_drawdown(
                returns,
                window=window,
                alpha=alpha,
                compound_returns=compound_returns,
                window_size=rolling,
            )
        )

        return self._finalize_measure(
            conditional_expected_drawdown,
            bool(rolling or within_period),
            rounding,
            growth,
            lag,
            standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_drawdown_at_risk(
        self,
        period: str | None = None,
        alpha: float = 0.05,
        compound_returns: bool = True,
        within_period: bool = False,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Drawdown at Risk (DaR) of an investment portfolio or asset's returns: how deep below its
        high the asset is in all but the worst alpha share of the periods.

        The Drawdown at Risk is the Value at Risk of the drawdown series: the depth below the running peak
        that the asset was at or beyond in only an alpha share of the periods (5% for the default alpha). Where
        the Maximum Drawdown rests on the single worst moment of the sample, the DaR describes how deep the
        drawdowns usually get in bad times, and the Conditional Drawdown at Risk (`get_conditional_drawdown_at_risk`)
        averages the drawdowns beyond it. Chekhlov, Uryasev and Zabarankin (2005) introduced it as a constraint
        for portfolio optimization: a fund manager whose clients leave after a deep drawdown cares about how
        often the portfolio is far below its high, not only about its volatility.

        The formula is as follows (with the default alpha of 0.05):

        - Wealth(t) = Π (1 + Return(s)) for s <= t
        - Drawdown(t) = Wealth(t) / max(1, max(Wealth(s) for s <= t)) - 1
        - DaR = Quantile(Drawdown, alpha)

        Every period counts towards the drawdown distribution, including those at a high (a drawdown of 0), the
        quantile interpolates linearly, and Wealth starts at 1, so a decline from the start of the window
        counts. Without compounding the drawdowns are measured on the summed returns instead, as an absolute
        decline from their peak (Drawdown(t) = Sum(t) - max(0, max(Sum(s) for s <= t)), with Sum(t) = Σ
        Return(s)). The result is a negative decimal (0 when the asset was at a high in more than 1 - alpha of
        the periods).

        For more information, see the following sources:

        - Chekhlov, A., Uryasev, S. and Zabarankin, M. (2005). "Drawdown Measure in Portfolio Optimization."
        International Journal of Theoretical and Applied Finance, 8(1), 13-58.

        Also known as: DaR, drawdown quantile, uncompounded DaR, absolute DaR.

        Shares similar logic:
        - Riskfolio-Lib, DaR_Rel (compounded): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.DaR_Rel
        - Riskfolio-Lib, DaR_Abs (uncompounded): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.DaR_Abs
        - skfolio, drawdown_at_risk: https://skfolio.org/generated/skfolio.measures.drawdown_at_risk.html

        Args:
            period (str, optional): The data frequency for returns. Options are: "daily", "weekly", "monthly",
                "quarterly", "yearly". Defaults to "daily".
            alpha (float, optional): The tail probability for the DaR calculation (e.g., 0.05 for the 95%
            confidence level). Defaults to 0.05.
            compound_returns (bool, optional): Whether to compound the returns into the wealth path the drawdowns are
            measured on, or to sum them (absolute drawdowns). Defaults to True.
            within_period (bool, optional): Whether to calculate the DaR within the specified period or for the
            entire period. Thus whether to look at the DaR within a specific year (if period = 'yearly') or look at
            the entirety of all years. Defaults to False. Note that this requires intraday data when period =
            'daily', since a single day only nests observations when intraday data was fetched.
            rolling (int, optional): The rolling window size to use for the calculation. If set, the DaR is
            calculated over a rolling window of this many periods across the full return history instead
            of per `period`, every window starting from a wealth of 1. At least 2. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the DaR values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series | pd.DataFrame: DaR values, per ticker, or per period (or date) and ticker.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates the DaR
        for each asset in the Toolkit instance, each on its own returns.
        - With growth, the method calculates the growth of DaR values using the specified `lag`.
        - Riskfolio-Lib's DaR_Rel and DaR_Abs and skfolio's drawdown_at_risk take the order statistic at
        ceil(alpha * n) of the sorted drawdowns instead of the interpolated quantile and report the drawdown
        as a positive number; on three years of daily returns they differ by up to 1e-3.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["^STOXX50E", "^N225", "^GSPC"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_drawdown_at_risk(period="yearly", within_period=True)
        ```

        Which returns:

        |      |   ^STOXX50E |   ^N225 |   ^GSPC |   Benchmark |
        |:-----|------------:|--------:|--------:|------------:|
        | 2021 |     -0.0527 | -0.0947 | -0.0385 |     -0.0375 |
        | 2022 |     -0.2303 | -0.1148 | -0.2335 |     -0.2244 |
        | 2023 |     -0.0821 | -0.0687 | -0.0715 |     -0.0693 |
        | 2024 |     -0.07   | -0.1379 | -0.0445 |     -0.0433 |
        | 2025 |     -0.077  | -0.1335 | -0.1164 |     -0.1122 |
        """
        validate_alpha(alpha)
        period = period if period else "daily"
        returns = self._get_measure_returns(period, within_period, rolling, growth)

        drawdown_at_risk = drawdown_model.get_drawdown_at_risk(
            returns, alpha=alpha, compound_returns=compound_returns, window_size=rolling
        )

        return self._finalize_measure(
            drawdown_at_risk,
            bool(rolling or within_period),
            rounding,
            growth,
            lag,
            standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_entropic_drawdown_at_risk(
        self,
        period: str | None = None,
        alpha: float = 0.05,
        compound_returns: bool = True,
        within_period: bool = False,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Entropic Drawdown at Risk (EDaR) of an investment portfolio or asset's returns: the
        tightest entropic bound on the deep drawdowns, between the CDaR and the Maximum Drawdown.

        The Entropic Drawdown at Risk is the historic Entropic Value at Risk (the historic method of
        `get_entropic_value_at_risk`) of the drawdown series: the tightest bound on the drawdowns that follows
        from the Chernoff inequality. It is at least as severe as the Drawdown at Risk and the Conditional
        Drawdown at Risk at the same alpha and at most as severe as the Maximum Drawdown, and as a coherent
        risk measure it is the drawdown counterpart of the EVaR in Cajas' (2021) entropic portfolio
        optimization. It weighs the deepest drawdowns more than the CDaR does, without resting on the single
        worst moment as the Maximum Drawdown does.

        The formula is as follows (with the default alpha of 0.05):

        - Wealth(t) = Π (1 + Return(s)) for s <= t
        - Drawdown(t) = Wealth(t) / max(1, max(Wealth(s) for s <= t)) - 1
        - EDaR = -inf over z > 0 of z * ln(mean(exp(-Drawdown(t) / z)) / alpha)

        Every period counts towards the drawdown distribution, including those at a high (a drawdown of 0), and
        Wealth starts at 1, so a decline from the start of the window counts. The bound is minimized over the
        risk aversion z numerically (on the logarithm of 1 / z, with scipy's brentq). Without compounding
        the drawdowns are measured on the summed returns instead, as an absolute decline from their peak
        (Drawdown(t) = Sum(t) - max(0, max(Sum(s) for s <= t)), with Sum(t) = Σ Return(s)). The result is a
        negative decimal.

        For more information, see the following sources:

        - Cajas, D. (2021). "Entropic Portfolio Optimization: a Disciplined Convex Programming Framework."
        SSRN Electronic Journal.
        - Ahmadi-Javid, A. (2012). "Entropic Value-at-Risk: A New Coherent Risk Measure." Journal of
        Optimization Theory and Applications, 155(3), 1105-1123.

        Also known as: EDaR, entropic drawdown, uncompounded EDaR, absolute EDaR.

        Referenced against:
        - skfolio, edar: https://skfolio.org/generated/skfolio.measures.edar.html
        - Riskfolio-Lib, EDaR_Rel (compounded): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.EDaR_Rel
        - Riskfolio-Lib, EDaR_Abs (uncompounded): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.EDaR_Abs

        Args:
            period (str, optional): The data frequency for returns. Options are: "daily", "weekly", "monthly",
                "quarterly", "yearly". Defaults to "daily".
            alpha (float, optional): The tail probability for the EDaR calculation (e.g., 0.05 for the 95%
            confidence level). Defaults to 0.05.
            compound_returns (bool, optional): Whether to compound the returns into the wealth path the drawdowns are
            measured on, or to sum them (absolute drawdowns). Defaults to True.
            within_period (bool, optional): Whether to calculate the EDaR within the specified period or for the
            entire period. Thus whether to look at the EDaR within a specific year (if period = 'yearly') or look
            at the entirety of all years. Defaults to False. Note that this requires intraday data when period =
            'daily', since a single day only nests observations when intraday data was fetched.
            rolling (int, optional): The rolling window size to use for the calculation. If set, the EDaR is
            calculated over a rolling window of this many periods across the full return history instead
            of per `period`, every window starting from a wealth of 1. At least 2. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the EDaR values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series | pd.DataFrame: EDaR values, per ticker, or per period (or date) and ticker.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates the EDaR
        for each asset in the Toolkit instance, each on its own returns.
        - With growth, the method calculates the growth of EDaR values using the specified `lag`.
        - Equals Riskfolio-Lib's EDaR_Rel and EDaR_Abs and skfolio's edar, which report the drawdown as a
        positive number.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["^STOXX50E", "^N225", "^GSPC"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_entropic_drawdown_at_risk(period="yearly", within_period=True)
        ```

        Which returns:

        |      |   ^STOXX50E |   ^N225 |   ^GSPC |   Benchmark |
        |:-----|------------:|--------:|--------:|------------:|
        | 2021 |     -0.0676 | -0.1046 | -0.0453 |     -0.0446 |
        | 2022 |     -0.2426 | -0.1427 | -0.2474 |     -0.2382 |
        | 2023 |     -0.0947 | -0.0864 | -0.0902 |     -0.0874 |
        | 2024 |     -0.0911 | -0.2002 | -0.0702 |     -0.0695 |
        | 2025 |     -0.1374 | -0.184  | -0.1606 |     -0.1592 |
        """
        validate_alpha(alpha)
        period = period if period else "daily"
        returns = self._get_measure_returns(period, within_period, rolling, growth)

        entropic_drawdown_at_risk = drawdown_model.get_entropic_drawdown_at_risk(
            returns, alpha=alpha, compound_returns=compound_returns, window_size=rolling
        )

        return self._finalize_measure(
            entropic_drawdown_at_risk,
            bool(rolling or within_period),
            rounding,
            growth,
            lag,
            standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_maximum_drawdown(
        self,
        period: str | None = None,
        within_period: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        compound_returns: bool = True,
        rolling: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Maximum Drawdown (MDD) of an investment portfolio or asset's returns: the largest decline
        from a high to a low, the worst loss an investor who bought at the top would have seen.

        Maximum Drawdown (MDD) is a risk management metric that quantifies the largest historical loss of
        an investment portfolio or asset experienced over a specified time horizon. It provides insights into
        the downside risk associated with an investment and helps investors make informed decisions about
        risk tolerance.

        The Maximum Drawdown is a single-path statistic: it depends entirely on the window chosen, so a
        figure over a window that includes a market crash and one over a window that starts after it are
        not comparable. It says how deep the worst fall was, not how long it lasted, which is why it is
        usually read together with the drawdown duration and recovery time (see
        `get_maximum_drawdown_duration` and `get_maximum_drawdown_recovery_time`), and it underlies
        return-to-risk ratios such as the Calmar ratio.

        The formula is as follows:

        - Wealth(t) = Π (1 + Return(s)) for s <= t
        - Drawdown(t) = Wealth(t) / max(1, max(Wealth(s) for s <= t)) - 1
        - Maximum Drawdown = min(Drawdown(t))

        Wealth starts at 1, so a decline from the start of the window counts as well, and the result is a
        negative decimal (0 when the value never fell below a previous high). Without compounding the
        drawdowns are measured on the summed returns, as an absolute decline from their peak
        (Drawdown(t) = Sum(t) - max(0, max(Sum(s) for s <= t)), with Sum(t) = Σ Return(s)), which is the
        Maximum Drawdown of Chekhlov, Uryasev and Zabarankin (2005) and Riskfolio-Lib's MDD_Abs. With
        `rolling`, every window starts from a wealth of 1 at its first return, so it is the deepest decline
        within the trailing window.

        For more information, see the following sources:

        - Magdon-Ismail, M. and Atiya, A.F. (2004). "Maximum Drawdown." Risk Magazine, 17(10), 99-102.
        - Chekhlov, A., Uryasev, S. and Zabarankin, M. (2005). "Drawdown Measure in Portfolio Optimization."
        International Journal of Theoretical and Applied Finance, 8(1), 13-58.

        Also known as: max drawdown, peak-to-trough decline, rolling maximum drawdown, uncompounded drawdown,
        absolute drawdown.

        Referenced against:
        - skfolio, max_drawdown: https://skfolio.org/generated/skfolio.measures.max_drawdown.html
        - Riskfolio-Lib, MDD_Rel (compounded): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.MDD_Rel
        - Riskfolio-Lib, MDD_Abs (uncompounded): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.MDD_Abs

        Shares similar logic:
        - empyrical, max_drawdown: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L352
        - empyrical, roll_max_drawdown (rolling): https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L405
        - PerformanceAnalytics (R), maxDrawdown: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/maxDrawdown.R#L44
        - quantstats, max_drawdown: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L2677

        Args:
            period (str, optional): The data frequency for returns. Options are: "daily", "weekly", "monthly",
                "quarterly", "yearly". Defaults to "daily".
            within_period (bool, optional): Whether to calculate the Maximum Drawdown within the specified period
            or for the entire period. Thus whether to look at the Maximum Drawdown within a specific year
            (if period = 'yearly') or look at the entirety of all years. Defaults to False.
            Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Maximum Drawdown values over time.
            Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            compound_returns (bool, optional): Whether to compound the returns into the wealth path the drawdowns are
            measured on, or to sum them (absolute drawdowns). Defaults to True.
            rolling (int, optional): The rolling window size to use for the calculation. If set, the Maximum
            Drawdown is calculated over a rolling window of this many periods across the full return history
            instead of per `period` (e.g. the deepest decline of the last 252 days). At least 2. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: Maximum Drawdown values with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates MMD for each
        asset in the Toolkit instance.
        - Without `within_period`, the drawdown is measured on the returns of the chosen period, so the
        yearly period only compares year-end values: it is 0 when every year in the range closed at a
        new high, even if prices fell within a year. Use the daily period for the deepest decline, or
        the within-period version for the deepest decline within each year.
        - With growth, the method calculates the growth of MMD values using the specified `lag`.
        - Equals empyrical's max_drawdown and roll_max_drawdown, Riskfolio-Lib's MDD_Rel and MDD_Abs and skfolio's
        max_drawdown, which report the decline as a positive number.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["META", "NFLX", "AMZN"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_maximum_drawdown(period="yearly", within_period=True)
        ```

        Which returns:

        |      |    META |    NFLX |    AMZN |   Benchmark |
        |:-----|--------:|--------:|--------:|------------:|
        | 2021 | -0.1972 | -0.173  | -0.1456 |     -0.051  |
        | 2022 | -0.7372 | -0.7237 | -0.5197 |     -0.245  |
        | 2023 | -0.1298 | -0.2752 | -0.1965 |     -0.0999 |
        | 2024 | -0.1842 | -0.1345 | -0.1949 |     -0.0841 |
        | 2025 | -0.3416 | -0.3073 | -0.3088 |     -0.1877 |
        """
        period = period if period else "daily"
        returns = self._get_measure_returns(period, within_period, rolling, growth)

        maximum_drawdown = drawdown_model.get_maximum_drawdown(
            returns, compound_returns=compound_returns, window_size=rolling
        )

        return self._finalize_measure(
            maximum_drawdown,
            bool(rolling or within_period),
            rounding,
            growth,
            lag,
            standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_maximum_drawdown_duration(
        self,
        period: str | None = None,
        within_period: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Maximum Drawdown Duration of an investment portfolio or asset's returns: how many periods
        the largest drawdown took from its peak to its lowest point.

        The Maximum Drawdown Duration is the number of periods between the peak and the lowest point
        of the largest drawdown, giving insight into how long the worst loss of value took to unfold.

        The formula is as follows:

        - Maximum Drawdown Duration = t(trough) - t(peak)

        Where t(trough) is the period of the lowest Drawdown (see `get_maximum_drawdown`) and t(peak) the last
        period at or before it in which Wealth stood at the high that drawdown is measured from. When that high is
        the starting value of 1, t(peak) is the period before the first return.

        Also known as: drawdown length.

        Shares similar logic:
        - PerformanceAnalytics (R), table.Drawdowns: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/table.Drawdowns.R#L54

        Args:
            period (str, optional): The data frequency for returns. Options are: "daily", "weekly", "monthly",
                "quarterly", "yearly". Defaults to "daily".
            within_period (bool, optional): Whether to calculate the duration within the specified period or
            for the entire period. Thus whether to look at the duration within a specific year (if period =
            'yearly') or look at the entirety of all years. Defaults to False.
            Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the duration values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: Maximum Drawdown Duration values, in number of periods, with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates the Maximum
        Drawdown Duration for each asset in the Toolkit instance.
        - With growth, the method calculates the growth of the duration values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["^GSPC", "^HSI", "^FTSE"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_maximum_drawdown_duration(period="yearly", within_period=True)
        ```

        Which returns:

        |      |   ^GSPC |   ^HSI |   ^FTSE |   Benchmark |
        |:-----|--------:|-------:|--------:|------------:|
        | 2021 |      22 |    217 |      15 |          22 |
        | 2022 |     201 |    200 |     173 |         201 |
        | 2023 |      64 |    225 |      98 |          64 |
        | 2024 |      14 |     35 |      58 |          14 |
        | 2025 |      34 |     13 |      27 |          34 |
        """
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
        if growth and not within_period:
            raise InvalidArgumentError(
                "Growth calculations need one value per period, while "
                "within_period=False produces a single value for the entire period. "
                "Set within_period=True to calculate growth."
            )

        returns = self._get_returns(period, within_period)

        maximum_drawdown_duration = risk_model.get_max_drawdown_duration(returns)

        if within_period:
            maximum_drawdown_duration = maximum_drawdown_duration.loc[
                self._start_date : self._end_date
            ]

        return finalize_dataset(
            dataset=maximum_drawdown_duration,
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
    def get_maximum_drawdown_recovery_time(
        self,
        period: str | None = None,
        within_period: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Maximum Drawdown Recovery Time of an investment portfolio or asset's returns: how many
        periods the asset took to climb back to its high after the bottom of its largest drawdown.

        The Maximum Drawdown Recovery Time is the number of periods it takes for the cumulative return
        to climb back to its previous peak after the lowest point of the largest drawdown. If the drawdown has not
        yet been recovered from within the selected period, this returns NaN.

        The formula is as follows:

        - Maximum Drawdown Recovery Time = t(recovery) - t(trough)

        Where t(trough) is the period of the lowest Drawdown (see `get_maximum_drawdown`) and t(recovery) the first
        period at or after it in which Wealth is back at or above the high that drawdown is measured from.

        Also known as: time to recovery, drawdown recovery.

        Shares similar logic:
        - PerformanceAnalytics (R), table.Drawdowns: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/table.Drawdowns.R#L54

        Args:
            period (str, optional): The data frequency for returns. Options are: "daily", "weekly", "monthly",
                "quarterly", "yearly". Defaults to "daily".
            within_period (bool, optional): Whether to calculate the recovery time within the specified period
            or for the entire period. Thus whether to look at the recovery time within a specific year (if
            period = 'yearly') or look at the entirety of all years. Defaults to False.
            Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the recovery time values over time.
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
            pd.Series: Maximum Drawdown Recovery Time values, in number of periods, with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates the Maximum
        Drawdown Recovery Time for each asset in the Toolkit instance.
        - With growth, the method calculates the growth of the recovery time values using the
        specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["TLT", "IEF", "HYG"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_maximum_drawdown_recovery_time(period="yearly", within_period=True)
        ```

        Which returns:

        |      |   TLT |   IEF |   HYG |   Benchmark |
        |:-----|------:|------:|------:|------------:|
        | 2021 |   nan |   nan |    19 |          12 |
        | 2022 |   nan |   nan |   nan |         nan |
        | 2023 |   nan |   nan |    98 |          23 |
        | 2024 |   nan |   nan |    14 |          32 |
        | 2025 |   100 |    49 |    23 |          54 |
        """
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
        if growth and not within_period:
            raise InvalidArgumentError(
                "Growth calculations need one value per period, while "
                "within_period=False produces a single value for the entire period. "
                "Set within_period=True to calculate growth."
            )

        returns = self._get_returns(period, within_period)

        maximum_drawdown_recovery_time = risk_model.get_max_drawdown_recovery_time(
            returns
        )

        if within_period:
            maximum_drawdown_recovery_time = maximum_drawdown_recovery_time.loc[
                self._start_date : self._end_date
            ]

        return finalize_dataset(
            dataset=maximum_drawdown_recovery_time,
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
    def get_drawdown_series(
        self,
        period: str | None = None,
        recovery: bool = False,
        deepest_since: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the drawdown in every period: how far an asset stands below its highest value so far,
        or the gain it needs to get back there.

        The drawdown series, also known as the underwater curve, shows the whole history of losses from a
        previous high instead of only the worst one. It is 0 at every new high and negative while the
        value is below one, so its lowest point is the Maximum Drawdown (see `get_maximum_drawdown`) and
        the stretches below 0 are the drawdowns that `get_drawdown_table` lists.

        Losses and gains are not symmetric: after a fall of 20% the asset needs a gain of 25% to be back
        at its high, after a fall of 50% a gain of 100%. The recovery version of the series shows that gain,
        the return needed from each period's value to get back to the running peak.

        With the Deepest Since dates, every period below the peak also gets the last earlier date with a
        drawdown at least as deep, the familiar headline ("the deepest drawdown since April"), as
        `get_move_significance` (Econometrics) gives the "largest since" date of a single move.

        The formula is as follows:

        - Wealth(t) = Π (1 + Return(s)) for s <= t
        - Peak(t) = max(1, max(Wealth(s) for s <= t))
        - Drawdown(t) = Wealth(t) / Peak(t) - 1
        - Recovery Needed(t) = Peak(t) / Wealth(t) - 1 = -Drawdown(t) / (1 + Drawdown(t))
        - Deepest Since(t) = the last date s < t with Drawdown(s) <= Drawdown(t), for Drawdown(t) < 0

        Returns are total returns (Adj Close). Wealth starts at 1 at the first close of the window, so a
        decline from the start of the window counts as well.

        Also known as: underwater curve, underwater plot, drawdown curve, distance from the high, gain needed
        to recover, breakeven return, recovery return.

        Shares similar logic:
        - PerformanceAnalytics (R), Drawdowns: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/Drawdowns.R#L3
        - quantstats, to_drawdown_series: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L2729
        - skfolio, get_drawdowns: https://skfolio.org/generated/skfolio.measures.get_drawdowns.html

        Args:
            period (str, optional): The data frequency for returns. Options are: "daily", "weekly", "monthly",
                "quarterly", "yearly". Defaults to "daily".
            recovery (bool, optional): Whether to return the gain needed to get back to the running peak
                (Recovery Needed) instead of the drawdown. Defaults to False.
            deepest_since (bool, optional): Whether to add the Deepest Since date of every period below the
                peak. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The drawdown per period as a negative decimal or, with recovery, the gain needed
                as a positive decimal (0 at a new high), with time as the index. With deepest since, the
                columns hold the measures (Drawdown or Recovery Needed, and Deepest Since, the date as text)
                and the tickers.

        Notes:
        - Each ticker keeps its own history: a ticker that listed during the window starts at its first close
        and is NaN before it.
        - A longer period only compares the closes of each period, so a monthly period shows shallower
        drawdowns than a daily one.
        - Deepest Since is NaN at a high and when no earlier period in the data was as deep, which makes
        the drawdown the deepest since the start_date. Choose an early start_date for a long history.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["META", "NFLX", "AMZN"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_drawdown_series(period="quarterly")
        ```

        Which returns:

        | Date   |    META |    NFLX |    AMZN |   Benchmark |
        |:-------|--------:|--------:|--------:|------------:|
        | 2021Q1 |  0      |  0      |  0      |      0      |
        | 2021Q2 |  0      |  0      |  0      |      0      |
        | 2021Q3 | -0.0239 |  0      | -0.0451 |      0      |
        | 2021Q4 | -0.0327 | -0.0129 | -0.0308 |      0      |
        | 2022Q1 | -0.3605 | -0.3862 | -0.0524 |     -0.0461 |
        | 2022Q2 | -0.5362 | -0.7134 | -0.3825 |     -0.1998 |
        | 2022Q3 | -0.6098 | -0.6143 | -0.3431 |     -0.2392 |
        | 2022Q4 | -0.6539 | -0.5168 | -0.5116 |     -0.1817 |
        | 2023Q1 | -0.3904 | -0.4339 | -0.3995 |     -0.1207 |
        | 2023Q2 | -0.1746 | -0.2782 | -0.2421 |     -0.0443 |
        | 2023Q3 | -0.1365 | -0.3813 | -0.261  |     -0.0751 |
        | 2023Q4 |  0      | -0.2021 | -0.1168 |      0      |
        | 2024Q1 |  0      | -0.0048 |  0      |      0      |
        | 2024Q2 |  0      |  0      |  0      |      0      |
        | 2024Q3 |  0      |  0      | -0.0358 |      0      |
        | 2024Q4 |  0      |  0      |  0      |      0      |
        | 2025Q1 | -0.0148 |  0      | -0.1328 |     -0.0427 |
        | 2025Q2 |  0      |  0      |  0      |      0      |
        | 2025Q3 | -0.0043 | -0.1047 |  0      |      0      |
        | 2025Q4 | -0.1043 | -0.2998 |  0      |      0      |
        """
        period = period if period else "daily"

        check_period(period, ["daily", "weekly", "monthly", "quarterly", "yearly"])

        # The first row holds the first close of the window (its return is NaN), the wealth of 1
        # the drawdowns are measured from; _get_returns would drop it.
        returns = self._historical_data[period]["Return"]

        drawdown_series = risk_model.get_drawdown_series(returns, recovery=recovery)

        if deepest_since:
            # Found on the drawdowns, which order the periods the same way as the gain needed.
            since = risk_model.get_deepest_since(
                drawdown_series
                if not recovery
                else risk_model.get_drawdown_series(returns)
            )
            drawdown_series = pd.concat(
                {
                    "Recovery Needed" if recovery else "Drawdown": drawdown_series,
                    "Deepest Since": since,
                },
                axis=1,
            )

        return finalize_dataset(
            dataset=drawdown_series,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rounding=rounding,
            axis="rows",
            apply_slice=False,
        )

    @handle_errors
    def get_drawdown_table(
        self,
        top: int = 5,
        period: str | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        List the largest drawdowns of each asset with their peak, trough and recovery dates, depth and length:
        whether an asset has one bad episode or recurring ones, and how long they took to recover.

        Where the Maximum Drawdown gives only the single worst decline, the drawdown table shows the
        largest declines side by side: when each started (the last high before it), when it bottomed
        out, when the asset was back at its high and how many periods each phase took. It shows whether
        an asset has one bad episode or recurring ones, and how long recoveries usually take, which a
        single figure hides. Recovery Needed is the gain needed from the trough to be back at the peak,
        which is larger than the drawdown itself: a 50% fall needs a 100% gain.

        The formula is as follows:

        - Drawdown(t) = Wealth(t) / max(1, max(Wealth(s) for s <= t)) - 1
        - To Trough = t(trough) - t(peak)
        - To Recovery = t(recovery) - t(trough)
        - Length = t(recovery) - t(peak)
        - Recovery Needed = -Drawdown / (1 + Drawdown)
        - Recovery Ratio = To Recovery / To Trough
        - Recovery Return = Wealth(recovery) / Wealth(trough) - 1
        - To Trough (Days) = date(trough) - date(peak), likewise To Recovery (Days) and Length (Days)
        - Length (Years) = Length (Days) / 365.25

        The Drawdown is the drawdown series of `get_drawdown_series`. A drawdown runs from the Peak (the last
        period at a high) to the Recovery (the first period back at that high), with its Trough at the lowest
        Drawdown in between. To Trough, To Recovery and Length count periods (trading days for daily data),
        the columns marked (Days) count calendar days between the same dates (the last day of each period for
        weekly or longer periods), and Length (Years) is Length (Days) in years of 365.25 days. Calendar time
        compares assets on exchanges with other holidays, and reads as the time an investor waited. The
        Recovery Ratio compares the way back with the way down: above 1 the recovery took longer than the
        decline. The Recovery Return is the gain from the trough to the close of the recovery, which can be
        more than the Recovery Needed when that close lies above the old peak.

        Each drawdown is dated by its peak, valley and recovery, so it starts on the peak rather than on
        the first period below it. The deepest
        drawdown matches `get_maximum_drawdown`, `get_maximum_drawdown_duration` (To Trough) and
        `get_maximum_drawdown_recovery_time` (To Recovery).

        Also known as: worst drawdowns, drawdown details, drawdown periods, gain needed to recover, recovery
        duration ratio, recovery return.

        Shares similar logic:
        - PerformanceAnalytics (R), table.Drawdowns: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/table.Drawdowns.R#L54
        - quantstats, drawdown_details: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L3301

        Args:
            top (int, optional): The number of drawdowns per asset, deepest first. Defaults to 5.
            period (str, optional): The data frequency for returns. Options are: "daily", "weekly", "monthly",
                "quarterly", "yearly". Defaults to "daily".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: One row per drawdown, indexed by ticker and rank, with the dates of the Peak, Trough
                and Recovery, the Drawdown as a negative decimal, the gain needed from the trough back to the peak
                (Recovery Needed) as a positive decimal, in periods the time from peak to trough
                (To Trough), from trough to recovery (To Recovery) and from peak to recovery (Length), the
                Recovery Ratio and the Recovery Return, the same in calendar days (To Trough (Days), To Recovery
                (Days), Length (Days)) and Length (Years).

        Notes:
        - A drawdown that has not recovered by the end of the window has no Recovery, To Recovery, Length,
        Recovery Ratio or Recovery Return, in periods or calendar time (NaN); `get_drawdown_statistics` gives
        the state of that open drawdown. Its Recovery Needed is the gain from its trough; the gain
        still needed from the latest value is the last Recovery Needed of `get_drawdown_series`.
        - Each ticker keeps its own history: a ticker that listed during the window starts at its first close.
        - The table depends on the window: a drawdown that started before the start date is measured from
        the first close of the window.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["META", "NFLX"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_drawdown_table(top=3)
        ```

        Which returns:

        |                  | Peak       | Trough     | Recovery   |   Drawdown |   Recovery Needed |   To Trough |   To Recovery |   Length |   Recovery Ratio |   Recovery Return |   To Trough (Days) |   To Recovery (Days) |   Length (Days) |   Length (Years) |
        |:-----------------|:-----------|:-----------|:-----------|-----------:|------------------:|------------:|--------------:|---------:|-----------------:|------------------:|-------------------:|---------------------:|----------------:|-----------------:|
        | ('META', 1)      | 2021-09-07 | 2022-11-03 | 2024-01-19 |    -0.7674 |            3.2985 |         293 |           302 |      595 |           1.0307 |            3.3128 |                422 |                  442 |             864 |           2.3655 |
        | ('META', 2)      | 2025-02-14 | 2025-04-21 | 2025-06-30 |    -0.3415 |            0.5186 |          44 |            48 |       92 |           1.0909 |            0.5241 |                 66 |                   70 |             136 |           0.3723 |
        | ('META', 3)      | 2025-08-12 | 2025-11-20 | NaT        |    -0.2537 |            0.34   |          71 |           nan |      nan |         nan      |          nan      |                100 |                  nan |             nan |         nan      |
        | ('NFLX', 1)      | 2021-11-17 | 2022-05-11 | 2024-08-20 |    -0.7594 |            3.1569 |         120 |           571 |      691 |           4.7583 |            3.1977 |                175 |                  832 |            1007 |           2.757  |
        | ('NFLX', 2)      | 2025-06-30 | 2025-12-10 | NaT        |    -0.3077 |            0.4444 |         114 |           nan |      nan |         nan      |          nan      |                163 |                  nan |             nan |         nan      |
        | ('NFLX', 3)      | 2025-02-14 | 2025-04-04 | 2025-04-24 |    -0.1915 |            0.2368 |          34 |            13 |       47 |           0.3824 |            0.2816 |                 49 |                   20 |              69 |           0.1889 |
        | ('Benchmark', 1) | 2022-01-03 | 2022-10-12 | 2023-12-13 |    -0.245  |            0.3244 |         195 |           294 |      489 |           1.5077 |            0.3403 |                282 |                  427 |             709 |           1.9411 |
        | ('Benchmark', 2) | 2025-02-19 | 2025-04-08 | 2025-06-26 |    -0.1876 |            0.2308 |          34 |            54 |       88 |           1.5882 |            0.2361 |                 48 |                   79 |             127 |           0.3477 |
        | ('Benchmark', 3) | 2024-07-16 | 2024-08-05 | 2024-09-19 |    -0.0841 |            0.0918 |          14 |            32 |       46 |           2.2857 |            0.1036 |                 20 |                   45 |              65 |           0.178  |
        """
        period = period if period else "daily"

        check_period(period, ["daily", "weekly", "monthly", "quarterly", "yearly"])
        if top < 1:
            raise ValueError("top must be at least 1.")

        # The first row holds the first close of the window, see get_drawdown_series.
        returns = self._historical_data[period]["Return"]

        drawdown_table = risk_model.get_drawdown_table(returns, top=top)
        decimals = rounding if rounding is not None else self._rounding

        if decimals is not None:
            rounded = [
                "Drawdown",
                "Recovery Needed",
                "Recovery Ratio",
                "Recovery Return",
                "Length (Years)",
            ]
            drawdown_table[rounded] = drawdown_table[rounded].round(decimals)

        return drawdown_table

    @handle_errors
    def get_drawdown_statistics(
        self,
        period: str | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Summarize the drawdowns of each asset: how many there were, how deep they were on average, how much
        their depth varied, how long they took to reach the bottom and to recover, how much of the time the
        asset spent below its high and where the drawdown that is still open stands.

        Where the drawdown table (`get_drawdown_table`) lists the largest drawdowns one by one, the statistics
        average over all of them, which is what Bacon (2008) uses to describe the drawdown risk of a fund: a
        manager with many shallow drawdowns that recover quickly and one with a few deep, long ones can have the
        same Maximum Drawdown. The Coverage is the share of the periods spent below the running high, and the
        Active measures describe the drawdown the asset is in at the end of the window: how deep it is, how long
        ago the high was and the bottom was, and how much of the decline has been made good.

        The formula is as follows:

        - Drawdown(t) = Wealth(t) / max(1, max(Wealth(s) for s <= t)) - 1
        - Average Drawdown = Σ Drawdown(j) / d
        - Drawdown Deviation = √(Σ Drawdown(j)^2 / n)
        - Average To Trough = Σ (t(trough, j) - t(peak, j)) / d
        - Average To Recovery = Σ (t(recovery, j) - t(trough, j)) / d_recovered
        - Average Length = Σ (t(recovery, j) - t(peak, j)) / d_recovered
        - Average Recovery Ratio = Σ (t(recovery, j) - t(trough, j)) / (t(trough, j) - t(peak, j)) / d_recovered
        - Average Recovery Return = Σ (Wealth(recovery, j) / Wealth(trough, j) - 1) / d_recovered
        - Coverage = the number of periods with Drawdown(t) < 0 / n
        - Active Drawdown = Drawdown(T)
        - Active Duration = T - t(peak), Active Recovery Duration = T - t(trough)
        - Active Recovery Return = (1 + Drawdown(T)) / (1 + Drawdown(trough)) - 1
        - Active Recovery = 1 - Drawdown(T) / Drawdown(trough)

        Where Drawdown(j) is the depth (the lowest Drawdown) of the j-th of the d drawdowns of
        `get_drawdown_table`, n the number of returns, d_recovered the number of drawdowns that recovered
        before the end of the window, T the last period and, for the Active measures, peak the last period at
        the high and trough the lowest Drawdown since. The lengths count periods (trading days for daily data)
        and Average Length (Days) the calendar days from peak to recovery. Average Drawdown is negative, Drawdown
        Deviation positive. The Active Recovery is the share of the decline from the peak to the trough that has
        been made good (0 at the trough, 1 back at the peak).

        For more information, see the following sources:

        - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution." John Wiley & Sons,
        2nd edition.

        Also known as: drawdown summary, average drawdown, drawdown deviation, average drawdown length, average
        recovery time, average drawdown days, time under water, current drawdown, open drawdown, recovery
        duration ratio.

        Shares similar logic:
        - PerformanceAnalytics (R), AverageDrawdown: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/maxDrawdown.R#L178
        - PerformanceAnalytics (R), DrawdownDeviation: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/maxDrawdown.R#L147
        - PerformanceAnalytics (R), AverageRecovery: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/maxDrawdown.R#L212
        - PerformanceAnalytics (R), AverageLength: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/maxDrawdown.R#L240
        - ffn, PerformanceStats.avg_drawdown: https://github.com/pmorissette/ffn/blob/v1.4.1/ffn/core.py#L377
        - vectorbt, Drawdowns.active_drawdown: https://github.com/polakowo/vectorbt/blob/v1.2.1/vectorbt/generic/drawdowns.py#L407
        - vectorbt, Drawdowns.recovery_duration_ratio: https://github.com/polakowo/vectorbt/blob/v1.2.1/vectorbt/generic/drawdowns.py#L392
        - vectorbt, Drawdowns.recovery_return: https://github.com/polakowo/vectorbt/blob/v1.2.1/vectorbt/generic/drawdowns.py#L328

        Args:
            period (str, optional): The data frequency for returns. Options are: "daily", "weekly", "monthly",
                "quarterly", "yearly". Defaults to "daily".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: One row per ticker with the number of Drawdowns, the Average Drawdown, the Drawdown
                Deviation, the Average To Trough, Average To Recovery and Average Length in periods, the
                Average Length (Days) in calendar days, the Average Recovery Ratio and Average Recovery Return,
                the Coverage and the Active Drawdown, Active Duration, Active Recovery Duration (in periods),
                Active Recovery Return and Active Recovery.

        Notes:
        - Each ticker keeps its own history: a ticker that listed during the window starts at its first close.
        - The Average Drawdown and Drawdown Deviation agree with PerformanceAnalytics' AverageDrawdown and
        DrawdownDeviation (which report the average as a positive number), and the Average Drawdown equals
        ffn's avg_drawdown. A drawdown that has not recovered by the end of the window has no recovery, so it
        is left out of the Average To Recovery, Average Length and the recovery averages; PerformanceAnalytics'
        AverageRecovery and AverageLength count it as recovered one period after the end. ffn's avg_drawdown_days
        counts calendar days from the first day below the peak, Average Length (Days) from the peak itself.
        - The Coverage, the Active measures and the recovery averages equal vectorbt's Drawdowns measures of the
        same names on the wealth path (the recovery averages over its recovered drawdowns).
        - At a high there is no open drawdown: the Active Drawdown and Active Duration are 0 and the other
        Active measures NaN.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["META", "NFLX"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_drawdown_statistics().T
        ```

        Which returns:

        |                          |    META |     NFLX |   Benchmark |
        |:-------------------------|--------:|---------:|------------:|
        | Drawdowns                | 39      |  30      |     87      |
        | Average Drawdown         | -0.072  |  -0.0735 |     -0.0174 |
        | Drawdown Deviation       |  0.0267 |   0.0249 |      0.0101 |
        | Average To Trough        | 15.5641 |  15      |      5.977  |
        | Average To Recovery      | 15.3684 |  25.931  |      7.3372 |
        | Average Length           | 29.4737 |  37.5172 |     13.3372 |
        | Average Length (Days)    | 42.7895 |  54.7241 |     19.3605 |
        | Average Recovery Ratio   |  1.4327 |   1.3625 |      1.1949 |
        | Average Recovery Return  |  0.1628 |   0.1717 |      0.0225 |
        | Coverage                 |  0.941  |   0.9466 |      0.8493 |
        | Active Drawdown          | -0.1632 |  -0.2998 |     -0.0123 |
        | Active Duration          | 98      | 128      |      4      |
        | Active Recovery Duration | 27      |  14      |      0      |
        | Active Recovery Return   |  0.1213 |   0.0113 |      0      |
        | Active Recovery          |  0.3568 |   0.0255 |      0      |
        """
        period = period if period else "daily"

        check_period(period, ["daily", "weekly", "monthly", "quarterly", "yearly"])

        # The first row holds the first close of the window, see get_drawdown_series.
        returns = self._historical_data[period]["Return"]

        drawdown_statistics = drawdown_model.get_drawdown_statistics(returns)
        decimals = rounding if rounding is not None else self._rounding

        return (
            drawdown_statistics.round(decimals)
            if decimals is not None
            else drawdown_statistics
        )

    @handle_portfolio
    @handle_errors
    def get_ulcer_index(
        self,
        period: str | None = None,
        window: int | None = 14,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        compound_returns: bool = True,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Ulcer Index of an investment portfolio or asset's returns: how deep and how long its
        drawdowns are, the root mean square of the declines from a recent high.

        The Ulcer Index is a financial metric used to assess the risk and volatility of an
        investment portfolio or asset. Developed by Peter Martin in the 1980s, the Ulcer Index
        is particularly useful for evaluating the downside risk and drawdowns associated with investments.

        The Ulcer Index differs from traditional volatility measures like standard deviation or variance
        because it focuses on the depth and duration of drawdowns rather than the dispersion of
        returns.

        The formula is as follows (with the default window of 14):

        - Ulcer Index = √(Σ ((Wealth(t) - Highest High(t)) / Highest High(t))^2 / n)

        Where Wealth(t) compounds the daily returns within each period from a starting value of 1, Highest
        High(t) is the highest Wealth over the trailing 14 observations (the starting value of 1 included
        while fewer are available) and n is the number of daily returns in the period.

        Martin measures every drawdown from the highest value so far (no window); the default
        trailing 14-period high is the variant StockCharts popularised. While fewer than `window`
        observations are available, the high is taken over the observations so far, so periods shorter
        than the window still have a value. The result is a decimal (0.05 is a root mean squared
        drawdown of 5%), where Martin quotes percentages.

        The Ulcer Index punishes both the depth and the length of drawdowns, so an asset that falls 10% and
        stays there for months scores worse than one that drops 10% and recovers within a week, which a
        volatility figure would not show. It is the risk measure behind the Martin ratio (excess return
        divided by the Ulcer Index) and is read best relative to peers over the same window, since the
        default 14-period lookback gives lower values than a since-inception high-water mark.

        Without compounding the drawdowns are the absolute declines of 1 plus the summed returns from their
        high, (Wealth(t) - Highest High(t)) with Wealth(t) = 1 + Σ Return(s), instead of the relative ones.

        For more information, see the following sources:

        - Martin, P.G. and McCann, B.B. (1989). "The Investor's Guide to Fidelity Funds." John Wiley & Sons.

        Also known as: UI, drawdown risk, uncompounded ulcer index.

        Referenced against:
        - Riskfolio-Lib, UCI_Rel (compounded, no window): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.UCI_Rel
        - Riskfolio-Lib, UCI_Abs (uncompounded, no window): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.UCI_Abs

        Shares similar logic:
        - PerformanceAnalytics (R), UlcerIndex: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/UlcerIndex.R#L31
        - quantstats, ulcer_index: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1829

        Args:
            period (str, optional): The data frequency for returns. Options are: "daily", "weekly", "monthly",
                "quarterly", "yearly". Defaults to "quarterly" for a quarterly Toolkit, otherwise "yearly".
            window (int | None, optional): The trailing lookback window used as the high-water
                mark reference for each day's drawdown. Pass None for an expanding
                (since-inception) high-water mark instead -- Martin's "Highest High"; a fixed
                int window is the common trailing-high variant (e.g. a 14-day trailing high).
                Defaults to 14.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the UI values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            compound_returns (bool, optional): Whether to compound the returns into the wealth path the drawdowns
                are measured on, or to sum them (absolute drawdowns). Defaults to True.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: UI values with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates UI for each
        asset in the Toolkit instance.
        - With growth, the method calculates the growth of UI values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["BA", "AIR.PA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_ulcer_index()
        ```

        Which returns:

        |      |     BA |   AIR.PA |   Benchmark |
        |:-----|-------:|---------:|------------:|
        | 2021 | 0.0564 |   0.041  |      0.0131 |
        | 2022 | 0.0917 |   0.0606 |      0.0481 |
        | 2023 | 0.0442 |   0.0289 |      0.0198 |
        | 2024 | 0.0696 |   0.0426 |      0.0164 |
        | 2025 | 0.0642 |   0.0503 |      0.027  |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        check_period(period, ["daily", "weekly", "monthly", "quarterly", "yearly"])
        if period == "daily" and self._historical_data["intraday"].empty:
            raise InvalidArgumentError(
                "Intraday data is required for daily calculations. Please initialise the "
                "Toolkit with an intraday_period or use a weekly or longer period."
            )

        returns = self._within_historical_data[period]["Return"]

        ulcer_index = risk_model.get_ui(
            returns, window, compound_returns=compound_returns
        )

        ulcer_index = ulcer_index.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=ulcer_index,
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
    def get_pain_index(
        self,
        period: str | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Pain Index of an investment portfolio or asset's returns: the average depth below its
        high over every day of a period, the area under water per day.

        The Pain Index, developed by Zephyr Associates and described by Bacon (2013), measures the
        downside risk of an investment portfolio or asset as the average depth of its drawdowns over
        every day of a period, the days at a new high counting as a drawdown of 0.

        Like the Ulcer Index it grows with both the depth and the duration of the drawdowns, so an asset
        that falls 10% and stays there for months scores worse than one that drops 10% and recovers
        within a week. Where the Ulcer Index takes the root mean square of the drawdowns, which makes the
        deepest ones dominate, the Pain Index takes their plain average: the area under water divided by
        the number of days. It is the risk measure behind the Pain Ratio (excess return divided by the Pain
        Index, see `toolkit.performance.get_pain_ratio`).

        The formula is as follows:

        - Wealth(t) = Π (1 + Return(s)) for s <= t
        - Drawdown(t) = Wealth(t) / max(1, max(Wealth(s) for s <= t)) - 1
        - Pain Index = Σ |Drawdown(t)| / n

        Where Wealth(t) compounds the daily returns within each period from a starting value of 1, so a
        loss on the first day of the period is a drawdown as well, the drawdowns are measured from the
        highest wealth since the start of the period and n is the number of daily returns in the period.
        The result is a positive decimal (0.05 is an average drawdown of 5%). Each ticker is measured over
        its own trading days, and the days before its first or after its last return (a later listing,
        a delisting) are no observations of it.

        For more information, see the following sources:

        - Bacon, C.R. (2013). "Practical Risk-Adjusted Performance Measurement." John Wiley & Sons,
        chapter 5 (the Pain Index of Zephyr Associates).

        Also known as: average drawdown, mean drawdown.

        Shares similar logic:
        - PerformanceAnalytics (R), PainIndex: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/PainIndex.R#L38

        Args:
            period (str, optional): The data frequency for returns. Options are: "daily", "weekly", "monthly",
                "quarterly", "yearly". Defaults to "quarterly" for a quarterly Toolkit, otherwise "yearly".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Pain Index values over time. Defaults
                to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Pain Index values with the periods as index and the tickers as columns.

        Notes:
        - The method retrieves the daily returns within each `period` and calculates the Pain Index for each
        asset in the Toolkit instance.
        - With growth, the method calculates the growth of the Pain Index values using the
        specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["BA", "AIR.PA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_pain_index()
        ```

        Which returns:

        |      |     BA |   AIR.PA |   Benchmark |
        |:-----|-------:|---------:|------------:|
        | 2021 | 0.1321 |   0.0404 |      0.009  |
        | 2022 | 0.2629 |   0.1135 |      0.139  |
        | 2023 | 0.0702 |   0.031  |      0.0223 |
        | 2024 | 0.3182 |   0.1124 |      0.0113 |
        | 2025 | 0.0715 |   0.0422 |      0.0278 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        check_period(period, ["daily", "weekly", "monthly", "quarterly", "yearly"])
        if period == "daily" and self._historical_data["intraday"].empty:
            raise InvalidArgumentError(
                "Intraday data is required for daily calculations. Please initialise the "
                "Toolkit with an intraday_period or use a weekly or longer period."
            )

        returns = self._within_historical_data[period]["Return"]

        pain_index = risk_model.get_pain_index(returns)

        pain_index = pain_index.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=pain_index,
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
    def get_average_drawdown(
        self,
        period: str | None = None,
        method: Literal["episode", "observation"] = "episode",
        compound_returns: bool = True,
        within_period: bool = False,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Average Drawdown of an investment portfolio or asset's returns: how deep a typical setback
        is, or how far below its high the asset typically was.

        The Average Drawdown has two definitions. Bacon (2008), PerformanceAnalytics and ffn average the depth
        of the drawdown episodes: every decline from a high until the asset is back at it counts once, with its
        deepest point. Chekhlov, Uryasev and Zabarankin (2005), Riskfolio-Lib and skfolio average the drawdown
        of every period instead, the periods at a high counting as 0, which weighs a drawdown by how long it
        lasted (the Pain Index, `get_pain_index`, is that average per year as a positive number). The episode
        average says how deep a typical setback is, the period average how far below its high the asset
        typically was. Both are less dominated by a single crash than the Maximum Drawdown.

        The formula for the episode average, the default, is as follows:

        - Wealth(t) = Π (1 + Return(s)) for s <= t
        - Drawdown(t) = Wealth(t) / max(1, max(Wealth(s) for s <= t)) - 1
        - Average Drawdown = Σ Trough(j) / d

        Where Trough(j) is the lowest Drawdown of the j-th of the d episodes below a high (one that has not
        recovered by the end of the window included). The formula for the observation average is as follows:

        - Average Drawdown = Σ Drawdown(t) / n

        Where n is the number of returns. Wealth starts at 1, so a decline from the start of the window counts.
        Without compounding the drawdowns are measured on the summed returns instead, as an absolute decline
        from their peak (Drawdown(t) = Sum(t) - max(0, max(Sum(s) for s <= t)), with Sum(t) = Σ Return(s)).
        The result is a negative decimal (0 when the value never fell below a previous high).

        For more information, see the following sources:

        - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution." John Wiley & Sons,
        2nd edition.
        - Chekhlov, A., Uryasev, S. and Zabarankin, M. (2005). "Drawdown Measure in Portfolio Optimization."
        International Journal of Theoretical and Applied Finance, 8(1), 13-58.

        Also known as: ADD, average drawdown depth, mean drawdown, uncompounded average drawdown.

        Referenced against:
        - skfolio, average_drawdown (observation): https://skfolio.org/generated/skfolio.measures.average_drawdown.html
        - Riskfolio-Lib, ADD_Rel (observation, compounded): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.ADD_Rel
        - Riskfolio-Lib, ADD_Abs (observation, uncompounded): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.ADD_Abs

        Shares similar logic:
        - PerformanceAnalytics (R), AverageDrawdown (episode): https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/maxDrawdown.R#L178
        - ffn, PerformanceStats.avg_drawdown (episode): https://github.com/pmorissette/ffn/blob/v1.4.1/ffn/core.py#L377

        Args:
            period (str, optional): The data frequency for returns. Options are: "daily", "weekly", "monthly",
                "quarterly", "yearly". Defaults to "daily".
            method (str, optional): The average. Options are: "episode" (the average depth of the drawdown
            episodes), "observation" (the average drawdown of every period). Defaults to "episode".
            compound_returns (bool, optional): Whether to compound the returns into the wealth path the drawdowns are
            measured on, or to sum them (absolute drawdowns). Defaults to True.
            within_period (bool, optional): Whether to calculate the Average Drawdown within the specified period
            or for the entire period. Thus whether to look at the Average Drawdown within a specific year (if
            period = 'yearly') or look at the entirety of all years. Defaults to False. Note that this requires
            intraday data when period = 'daily', since a single day only nests observations when intraday data
            was fetched.
            rolling (int, optional): The rolling window size to use for the calculation. If set, the Average
            Drawdown is calculated over a rolling window of this many periods across the full return history
            instead of per `period`, every window starting from a wealth of 1. At least 2. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Average Drawdown values over time.
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
            pd.Series | pd.DataFrame: Average Drawdown values, per ticker, or per period (or date) and ticker.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates the Average
        Drawdown for each asset in the Toolkit instance, each on its own returns.
        - With growth, the method calculates the growth of the Average Drawdown values using the
        specified `lag`.
        - The episode average agrees with PerformanceAnalytics' AverageDrawdown and ffn's avg_drawdown, the
        observation average with Riskfolio-Lib's ADD_Rel and ADD_Abs and skfolio's average_drawdown;
        PerformanceAnalytics and the Python libraries other than ffn report the drawdown as a positive number.
        PerformanceAnalytics' arithmetic version divides the decline of 1 + the summed returns by its peak, where the uncompounded drawdown here
        (and in Riskfolio-Lib and skfolio) takes the absolute decline.
        - The Sterling Ratio of `toolkit.performance` averages the depths of the 3 deepest episodes only (Bacon's
        Average Largest Drawdown); with at least as many as there are episodes it is the episode average.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["TSLA", "GM", "F"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_average_drawdown(period="yearly", within_period=True)
        ```

        Which returns:

        |      |    TSLA |      GM |       F |   Benchmark |
        |:-----|--------:|--------:|--------:|------------:|
        | 2021 | -0.1491 | -0.0916 | -0.0589 |     -0.0128 |
        | 2022 | -0.7272 | -0.5304 | -0.2025 |     -0.245  |
        | 2023 | -0.1077 | -0.0943 | -0.13   |     -0.0164 |
        | 2024 | -0.1443 | -0.0581 | -0.0914 |     -0.0129 |
        | 2025 | -0.1155 | -0.0296 | -0.0561 |     -0.0195 |
        """
        period = period if period else "daily"
        returns = self._get_measure_returns(period, within_period, rolling, growth)

        average_drawdown = drawdown_model.get_average_drawdown(
            returns,
            method=method,
            compound_returns=compound_returns,
            window_size=rolling,
        )

        return self._finalize_measure(
            average_drawdown,
            bool(rolling or within_period),
            rounding,
            growth,
            lag,
            standardize,
        )
