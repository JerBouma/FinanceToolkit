"""Liquidity Risk Module"""

__docformat__ = "google"

from collections.abc import Callable

import pandas as pd

from financetoolkit.helpers import handle_portfolio
from financetoolkit.performance import performance_model
from financetoolkit.ratios.helpers import map_period_data_to_daily_data
from financetoolkit.risk.liquidity import market_liquidity_model
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
class LiquidityRisk:
    """
    Market liquidity: the Amihud illiquidity measure, the Roll, Corwin-Schultz, Abdi-Ranaldo
    and EDGE spreads and the share turnover, and the idiosyncratic volatility.
    """

    # The Toolkit sets this to a function returning the Weighted Average Shares of the
    # statements (tickers as index, periods as columns) and the day each period's figures
    # apply from, which the Share Turnover divides the Volume by.
    _shares_outstanding_function: (
        Callable[[], tuple[pd.DataFrame, pd.DataFrame]] | None
    ) = None

    @handle_portfolio
    @handle_errors
    def get_amihud_illiquidity(
        self,
        period: str | None = None,
        within_period: bool = False,
        scale: float = 1_000_000,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Amihud (2002) Illiquidity ratio, per asset.

        The Amihud ratio measures the average price impact of trading -- how much the
        price moves per dollar of trading volume. A high value means that even a small
        amount of trading moves the price a lot (the asset is illiquid), while a low
        value means the asset can absorb a large amount of trading with little price
        impact (the asset is liquid).

        The Amihud ratio is mainly a cross-sectional measure: it ranks stocks by how much a
        given dollar volume moves the price and is used as a liquidity factor in asset
        pricing. Its level depends on the scale, the frequency and the price level of the
        market, so compare stocks within the same market and period, and expect the ratio
        of large caps to be several orders of magnitude below that of small caps.

        For more information about the method, see the following paper:

        - Amihud, Y. (2002). "Illiquidity and Stock Returns: Cross-Section and
        Time-Series Effects." Journal of Financial Markets, 5(1), 31-56.

        The formula is as follows (with the default scale of 1,000,000):

        - Amihud Illiquidity = Mean(|Return_t| / (Close_t * Volume_t)) * 1,000,000

        Where the mean runs over the periods of the chosen frequency (within each period with `within_period`),
        and periods without dollar volume are skipped.

        Also known as: Amihud illiquidity ratio, ILLIQ, price impact ratio.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            within_period (bool, optional): Whether to calculate the ratio within the specified period or
            for the entire period. Defaults to False. Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            scale (float, optional): A multiplier applied to the resulting ratio purely for readability.
            Defaults to 1,000,000.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Amihud Illiquidity values over
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
            pd.Series: Amihud Illiquidity values with time as the index.

        Notes:
        - The method retrieves historical return, Close price and Volume data based on the specified
        `period` and calculates the Amihud Illiquidity ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the Amihud Illiquidity values
        using the specified `lag`.

        A `scale` of one billion is used in the example below: liquid stocks trade so much dollar volume
        relative to their typical daily price move that the default `scale` of 1,000,000 (as in Amihud's
        original paper) can round their values to 0.0 at the default precision.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["SNAP", "ETSY", "LYFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_amihud_illiquidity(scale=1e9)
        ```

        Which returns:

        |           |   Value |
        |:----------|--------:|
        | SNAP      |  0.0634 |
        | ETSY      |  0.0768 |
        | LYFT      |  0.1277 |
        | Benchmark |  0.0002 |
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

        source_data = (
            self._within_historical_data[period]
            if within_period
            else self._historical_data[period]
        )

        # Missing returns are skipped per ticker, so a ticker with a shorter history does not cut the others.
        returns = source_data["Return"].dropna(how="all")
        dollar_volume = source_data["Close"] * source_data["Volume"]

        illiquidity = market_liquidity_model.get_amihud_illiquidity(
            returns, dollar_volume, scale
        )

        if within_period:
            illiquidity = illiquidity.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=illiquidity,
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
    def get_roll_spread(
        self,
        period: str | None = None,
        within_period: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Roll (1984) implied bid-ask spread, per asset.

        Roll's model shows that, under a stylized microstructure model in which the
        true (efficient) price follows a random walk and observed trade prices
        randomly bounce between the bid and the ask, the effective spread can be
        backed out purely from the serial covariance of consecutive price changes,
        without needing any actual quote data. If the estimated covariance is zero or
        positive, no valid estimate can be backed out and NaN is returned instead.

        The Roll measure is used when no quote data is available, typically for less liquid
        stocks or historical studies. On daily closes of liquid large caps it mostly picks
        up price noise rather than the bid-ask bounce, and in trending periods the
        autocovariance is positive and no estimate exists, so treat it as a rough, relative
        measure.

        For more information about the method, see the following paper:

        - Roll, R. (1984). "A Simple Implicit Measure of the Effective Bid-Ask Spread
        in an Efficient Market." The Journal of Finance, 39(4), 1127-1139.

        The formula is as follows:

        - Roll Spread = 2 * √(-Cov(ΔP_t, ΔP_(t-1)))
        - Relative Roll Spread = Roll Spread / Mean(P)

        Where P is the Close price, ΔP_t = P_t - P_(t-1) and Cov the sample autocovariance (ddof=1) of
        consecutive price changes; a covariance of zero or above gives NaN.

        Also known as: Roll's implied spread, Roll measure.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            within_period (bool, optional): Whether to calculate the spread within the specified period or
            for the entire period. Defaults to False. Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Roll Spread (in price units), the Relative Roll Spread (the Roll Spread divided
            by the mean price, as a decimal), the underlying lag-1 autocovariance, and whether that
            autocovariance was negative (i.e. whether a valid estimate could be backed out), per asset.

        Notes:
        - The method retrieves historical Close price data based on the specified `period` and calculates
        the Roll Spread for each asset in the Toolkit instance.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["KO", "PEP", "PG"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_roll_spread()
        ```

        Which returns:

        |                      |          KO |         PEP |         PG |   Benchmark |
        |:---------------------|------------:|------------:|-----------:|------------:|
        | Roll Spread          |  0.173128   |  0.72957    |  0.751753  |  1.97906    |
        | Relative Roll Spread |  0.0028038  |  0.00449337 |  0.0049553 |  0.00408282 |
        | Autocovariance       | -0.00749332 | -0.133068   | -0.141283  | -0.979167   |
        | Valid Estimate       |  1          |  1          |  1         |  1          |
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

        close_prices = self._get_prices("Close", period, within_period)

        result = market_liquidity_model.get_roll_spread(close_prices)

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_corwin_schultz_spread(
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
        Calculate the Corwin and Schultz (2012) high-low bid-ask spread, per asset and period.

        The daily High is almost always a trade at the ask and the daily Low a trade at the
        bid, so the range of a day holds both the volatility and the bid-ask spread. The
        volatility part of the range grows with the length of the window while the spread
        part does not, so comparing the ranges of two single days with the range over both
        days together backs out the spread from daily High and Low prices alone, without
        quote data. Each pair of consecutive trading days gives an estimate and the spread of
        a period is the average of the estimates within it.

        The estimator is mostly used where no quotes are available (historical studies,
        smaller markets) and ranks stocks well by their spread. For liquid large caps the
        spread is a few basis points and the estimate is dominated by the volatility within
        the day, so read it as a relative measure.

        For more information about the method, see the following paper:

        - Corwin, S. A., & Schultz, P. (2012). "A Simple Way to Estimate Bid-Ask Spreads from
        Daily High and Low Prices." The Journal of Finance, 67(2), 719-760.

        The formula is as follows:

        - β = [ln(H_t / L_t)]^2 + [ln(H_(t+1) / L_(t+1))]^2
        - γ = [ln(max(H_t, H_(t+1)) / min(L_t, L_(t+1)))]^2
        - α = (√(2β) - √β) / (3 - 2√2) - √(γ / (3 - 2√2))
        - Two-Day Spread = max(2 * (e^α - 1) / (1 + e^α), 0)
        - Corwin-Schultz Spread = Mean(Two-Day Spreads within the period)

        Where H and L are the daily High and Low of two consecutive trading days t and t+1 of
        the asset. Before β and γ are calculated, day t+1 is adjusted for the overnight return
        as in the paper: when its Low is above the Close of day t, its High and Low are
        lowered by the difference, and when its High is below that Close, they are raised by
        the difference. A negative two-day spread is set to 0, the paper's main convention,
        before the average is taken.

        Also known as: high-low spread estimator, HL spread, CS spread.

        Args:
            period (str, optional): The period to average the two-day spreads over (weekly, monthly, quarterly
                or yearly). Defaults to "monthly", as in the paper.
            rolling (int, optional): Average the two-day spreads of the last `rolling` periods instead of one
                period (e.g. period='monthly' and rolling=3 for a three-month spread at every month end).
                Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the spread over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The spread as a decimal of the price (0.005 is 0.5%), with the periods as the index
            and the assets as columns.

        Notes:
        - The method uses the daily High, Low and Close (split-adjusted) of each asset on its own trading
        days: a day on which only another ticker or the benchmark traded is not part of a pair, so the pair
        is always two consecutive trading days of the asset.
        - Both days of a pair fall within the period; a pair across two periods is left out.
        - A day with a price of 0 or below (a bad bar) is left out.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["SNAP", "ETSY", "LYFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-06-30",
        )

        toolkit.risk.get_corwin_schultz_spread()
        ```

        Which returns:

        | Date    |   SNAP |   ETSY |   LYFT |   Benchmark |
        |:--------|-------:|-------:|-------:|------------:|
        | 2025-01 | 0.0106 | 0.0086 | 0.0095 |      0.0017 |
        | 2025-02 | 0.011  | 0.0117 | 0.0115 |      0.002  |
        | 2025-03 | 0.0119 | 0.0096 | 0.0151 |      0.0046 |
        | 2025-04 | 0.0173 | 0.0174 | 0.0183 |      0.0099 |
        | 2025-05 | 0.0098 | 0.0086 | 0.0096 |      0.0031 |
        | 2025-06 | 0.0111 | 0.0067 | 0.0155 |      0.0024 |
        """
        period = period if period else "monthly"

        if period not in ["weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError("Period must be weekly, monthly, quarterly, or yearly.")

        spread = market_liquidity_model.get_corwin_schultz_spread(
            self._get_prices("High"),
            self._get_prices("Low"),
            self._get_prices("Close"),
            period,
            rolling,
        )

        spread = spread.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=spread,
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
    def get_abdi_ranaldo_spread(
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
        Calculate the Abdi and Ranaldo (2017) close-high-low bid-ask spread, per asset and period.

        The average of the daily High and Low (the mid-range) is a proxy for the efficient price
        without the spread, while the Close is a trade at either the bid or the ask. The distance of
        the Close to the mid-range of its own day and to that of the next day both hold half the
        spread, and their product removes the volatility of the efficient price, which backs out the
        spread from daily Close, High and Low prices alone, without quote data. Each pair of
        consecutive trading days gives an estimate and the spread of a period is the average of the
        estimates within it.

        Unlike the Corwin-Schultz spread, the estimator uses the Close as well and needs no adjustment
        for the overnight return (a weekend or a gap at the open). Like it, the estimate is mostly used
        where no quotes are available and ranks assets by their spread; for liquid large caps the spread
        is a few basis points and the estimate is dominated by the volatility, so read it as a relative
        measure.

        For more information about the method, see the following paper:

        - Abdi, F., & Ranaldo, A. (2017). "A Simple Estimation of Bid-Ask Spreads from Daily Close, High,
        and Low Prices." The Review of Financial Studies, 30(12), 4437-4480.

        The formula is as follows:

        - η_t = (ln H_t + ln L_t) / 2
        - Two-Day Spread = √(max(4 * (ln C_t - η_t) * (ln C_t - η_(t+1)), 0))
        - Abdi-Ranaldo Spread = Mean(Two-Day Spreads within the period)

        Where H, L and C are the daily High, Low and Close of two consecutive trading days t and t+1 of
        the asset. This is the "two-day corrected" version that the paper uses in its empirical analysis
        (equation 11): a negative two-day estimate of the squared spread is set to 0 before the square
        root and the average are taken.

        Also known as: close-high-low spread estimator, CHL spread, AR spread.

        Args:
            period (str, optional): The period to average the two-day spreads over (weekly, monthly, quarterly
                or yearly). Defaults to "monthly", as in the paper.
            rolling (int, optional): Average the two-day spreads of the last `rolling` periods instead of one
                period (e.g. period='monthly' and rolling=3 for a three-month spread at every month end).
                Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the spread over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The spread as a decimal of the price (0.005 is 0.5%), with the periods as the index
            and the assets as columns.

        Notes:
        - The method uses the daily High, Low and Close (split-adjusted) of each asset on its own trading
        days: a day on which only another ticker or the benchmark traded is not part of a pair, so the pair
        is always two consecutive trading days of the asset.
        - Both days of a pair fall within the period; a pair across two periods is left out.
        - A day with a price of 0 or below (a bad bar) is left out.
        - Averaging the square roots of the two-day estimates lowers the level when the daily volatility
        is large relative to the spread (in simulations with a 1% spread and a 1% daily volatility the
        estimate is about 0.8%). The paper accepts this for the closer correlation with the effective
        spread from trade and quote data; `get_edge_spread` has no such bias.
        - The paper discards months with fewer than 12 applicable days; no such minimum is applied here,
        as for `get_corwin_schultz_spread`, so a period with a single pair has an estimate.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["SNAP", "ETSY", "LYFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-06-30",
        )

        toolkit.risk.get_abdi_ranaldo_spread()
        ```

        Which returns:

        | Date    |   SNAP |   ETSY |   LYFT |   Benchmark |
        |:--------|-------:|-------:|-------:|------------:|
        | 2025-01 | 0.0115 | 0.0107 | 0.0084 |      0.0024 |
        | 2025-02 | 0.014  | 0.0087 | 0.0176 |      0.0035 |
        | 2025-03 | 0.0117 | 0.0065 | 0.0113 |      0.0053 |
        | 2025-04 | 0.0287 | 0.0116 | 0.0284 |      0.013  |
        | 2025-05 | 0.0168 | 0.0112 | 0.0137 |      0.0047 |
        | 2025-06 | 0.0109 | 0.007  | 0.0101 |      0.0029 |
        """
        period = period if period else "monthly"

        if period not in ["weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError("Period must be weekly, monthly, quarterly, or yearly.")

        spread = market_liquidity_model.get_abdi_ranaldo_spread(
            self._get_prices("High"),
            self._get_prices("Low"),
            self._get_prices("Close"),
            period,
            rolling,
        )

        spread = spread.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=spread,
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
    def get_edge_spread(
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
        Calculate the EDGE bid-ask spread of Ardia, Guidotti and Kroencke (2024), per asset and period.

        EDGE (the Efficient Discrete Generalized Estimator) estimates the spread from the daily Open,
        High, Low and Close. The mid-range (the average of the High and Low) is a proxy for the efficient
        price, while the Open and the Close are trades at the bid or the ask, so the returns between
        them around the open and around the close bounce with the spread. Four such estimators, two at
        the open and two at the close (one of them a generalization of the Abdi-Ranaldo spread), are
        combined with the weights of the Generalized Method of Moments that minimize the variance of the
        estimate. The probabilities in the formula correct for days without trades, which repeat the
        previous Close. The estimate is asymptotically unbiased and, in the paper's simulations and
        empirical tests, more precise than the Roll, Corwin-Schultz and Abdi-Ranaldo estimators.

        Unlike the Corwin-Schultz and Abdi-Ranaldo spreads, which average two-day estimates, EDGE is
        estimated from all pairs of consecutive trading days of the period at once. For liquid large caps
        the spread is a few basis points, close to the estimation error, so read it as a relative
        measure there.

        For more information about the method, see the following paper:

        - Ardia, D., Guidotti, E., & Kroencke, T. A. (2024). "Efficient Estimation of Bid-Ask Spreads from
        Open, High, Low, and Close Prices." Journal of Financial Economics, 161, 103916.

        The formula is as follows:

        - η_t = (h_t + l_t) / 2 and τ_t = 1 if h_t ≠ l_t or l_t ≠ c_(t-1), else 0
        - π_o = -8 / (P[o_t ≠ h_t, τ_t = 1] + P[o_t ≠ l_t, τ_t = 1])
        - π_c = -8 / (P[c_(t-1) ≠ h_(t-1), τ_t = 1] + P[c_(t-1) ≠ l_(t-1), τ_t = 1])
        - x_1 = π_o / 2 * (η_t - o_t) * (o_t - η_(t-1)) + π_c / 2 * (η_t - c_(t-1)) * (c_(t-1) - η_(t-1))
        - x_2 = π_o / 2 * (η_t - o_t) * (o_t - c_(t-1)) + π_c / 2 * (o_t - c_(t-1)) * (c_(t-1) - η_(t-1))
        - S^2 = (Var[x_2] * E[x_1] + Var[x_1] * E[x_2]) / (Var[x_1] + Var[x_2])
        - EDGE Spread = √(max(S^2, 0))

        Where o, h, l and c are the log Open, High, Low and Close of two consecutive trading days t-1 and
        t of the asset, and the probabilities, expectations and variances are taken over the pairs within
        the period. The returns η_t - o_t, η_t - c_(t-1) and o_t - c_(t-1) in the first factor of each
        product are de-meaned over the pairs with τ_t = 1, as in the authors' implementation. A negative
        squared spread of the period is set to 0, as in the paper (equation 26); setting the period's
        estimate rather than each two-day estimate to 0 keeps the estimate consistent.

        Also known as: Efficient Discrete Generalized Estimator, EDGE estimator, OHLC spread estimator.

        Args:
            period (str, optional): The period to estimate the spread over (weekly, monthly, quarterly or
                yearly). Defaults to "monthly", as in the paper.
            rolling (int, optional): Estimate the spread from the pairs of the last `rolling` periods together
                instead of one period (e.g. period='monthly' and rolling=3 for a three-month spread at every
                month end). Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the spread over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The spread as a decimal of the price (0.005 is 0.5%), with the periods as the index
            and the assets as columns.

        Notes:
        - The method uses the daily Open, High, Low and Close (split-adjusted) of each asset on its own
        trading days: a day on which only another ticker or the benchmark traded is not part of a pair, so
        the pair is always two consecutive trading days of the asset.
        - Both days of a pair fall within the period; a pair across two periods is left out, also in a
        rolling window.
        - A day with a price of 0 or below (a bad bar) is left out.
        - A period with fewer than two pairs with a price change, or in which the Open (or the previous
        Close) equals both the High and the Low on every pair with a price change (only single-price days),
        is NaN.
        - The estimates equal those of the authors' `bidask` package (`edge` on the days of the period,
        with negative squared spreads set to 0).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["SNAP", "ETSY", "LYFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-06-30",
        )

        toolkit.risk.get_edge_spread()
        ```

        Which returns:

        | Date    |   SNAP |   ETSY |   LYFT |   Benchmark |
        |:--------|-------:|-------:|-------:|------------:|
        | 2025-01 | 0      | 0.0032 | 0      |      0.0019 |
        | 2025-02 | 0.0121 | 0      | 0.0296 |      0.0045 |
        | 2025-03 | 0      | 0      | 0      |      0.003  |
        | 2025-04 | 0.0245 | 0.0176 | 0.0219 |      0.0135 |
        | 2025-05 | 0.0173 | 0.0134 | 0      |      0.007  |
        | 2025-06 | 0      | 0      | 0.0061 |      0.0017 |
        """
        period = period if period else "monthly"

        if period not in ["weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError("Period must be weekly, monthly, quarterly, or yearly.")

        spread = market_liquidity_model.get_edge_spread(
            self._get_prices("Open"),
            self._get_prices("High"),
            self._get_prices("Low"),
            self._get_prices("Close"),
            period,
            rolling,
        )

        spread = spread.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=spread,
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
    def get_share_turnover(
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
        Calculate the Share Turnover, the number of shares traded relative to the number of
        shares outstanding, per asset and period.

        Datar, Naik and Radcliffe (1998) use the turnover as a liquidity proxy: a share that
        changes hands often can be traded at a lower cost. They find that stocks with a lower
        turnover earn higher returns, also after controlling for size, the book-to-market
        ratio and the market beta. The turnover of a period is the share of the outstanding
        shares that was traded in it, the sum of the daily turnover within the period.

        The paper measures the turnover per month and averages it over the three months before
        the month of the returns, which `rolling=3` gives with the default monthly period.
        A turnover of 0.1 means that 10% of the shares outstanding changed hands; a value
        above 1 means more shares were traded than there are outstanding.

        For more information about the method, see the following paper:

        - Datar, V. T., Naik, N. Y., & Radcliffe, R. (1998). "Liquidity and Stock Returns: An
        Alternative Test." Journal of Financial Markets, 1(2), 203-219.

        The formula is as follows:

        - Daily Turnover = Volume_t / Weighted Average Shares_t
        - Share Turnover = SUM(Daily Turnover within the period)

        Where Volume is the number of shares traded on day t and Weighted Average Shares the
        number of shares of the last fiscal period that ended on or before day t. With
        `rolling`, the result is the mean of the Share Turnover of the last `rolling` periods.

        Also known as: turnover ratio, share turnover rate, trading turnover.

        Args:
            period (str, optional): The period to sum the daily turnover over (weekly, monthly, quarterly or
                yearly). Defaults to "monthly", as in the paper.
            rolling (int, optional): The mean of the turnover of the last `rolling` periods instead of the
                turnover per period (e.g. rolling=3 for the paper's three-month average). Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the turnover over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Share Turnover as a decimal, with the periods as the index and the assets as
            columns.

        Notes:
        - The number of shares is the Weighted Average Shares of the income statement, the share count the
        Toolkit's market capitalization uses as well. Each period's figure applies from the end of its fiscal
        period (from the filing day with point_in_time), so a day never uses a share count that was not yet
        reported. The days before the first fiscal period end in the statements have no share count, so a
        period with such a day is NaN rather than a partial sum, as is a period with a trading day without a
        positive share count.
        - The Volume is that of the asset's own trading days. The benchmark and other assets without
        financial statements are NaN.
        - The share count is not adjusted within a fiscal period, so a stock split during a period changes the
        turnover until the statements of the period after the split apply.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["SNAP", "ETSY", "LYFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-06-30",
        )

        toolkit.risk.get_share_turnover()
        ```

        Which returns:

        | Date    |   SNAP |   ETSY |   LYFT |   Benchmark |
        |:--------|-------:|-------:|-------:|------------:|
        | 2025-01 | 0.3433 | 0.4878 | 0.7121 |         nan |
        | 2025-02 | 0.3536 | 0.6891 | 1.2397 |         nan |
        | 2025-03 | 0.3493 | 0.7956 | 0.905  |         nan |
        | 2025-04 | 0.5942 | 0.9174 | 0.7491 |         nan |
        | 2025-05 | 0.4435 | 0.8986 | 1.1707 |         nan |
        | 2025-06 | 0.4057 | 1.2026 | 0.6827 |         nan |
        """
        period = period if period else "monthly"

        if period not in ["weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError("Period must be weekly, monthly, quarterly, or yearly.")
        if self._shares_outstanding_function is None:
            raise ValueError(
                "The Share Turnover needs the shares outstanding of the financial statements, "
                "which toolkit.risk of a Toolkit provides."
            )

        shares, period_end_dates = self._shares_outstanding_function()
        volume = self._get_prices("Volume")

        daily_shares = map_period_data_to_daily_data(
            period_data=shares,
            daily_dates=volume.index,
            quarterly=self._quarterly,
            period_end_dates=period_end_dates,
        )
        daily_shares = daily_shares.set_axis(volume.index, axis=0).reindex(
            columns=volume.columns
        )

        turnover = market_liquidity_model.get_share_turnover(
            volume, daily_shares, period, rolling
        )

        turnover = turnover.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=turnover,
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
    def get_idiosyncratic_volatility(
        self,
        period: str | None = None,
        model: str = "fama_french",
        rolling: int | None = None,
        annualize: bool = False,
        region: str = "United States",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Idiosyncratic Volatility, the volatility of the part of the returns the
        factors do not explain, per asset and period.

        Ang, Hodrick, Xing and Zhang (2006) regress the daily excess returns of every stock on
        the Fama and French three factors within each month and take the standard deviation
        of the residuals. Stocks with a high idiosyncratic volatility earn strikingly low
        returns in the following month, the idiosyncratic volatility puzzle: a diversified
        investor is not exposed to this risk, so it should not be priced at all, let alone
        negatively. With model="capm" the regression is the market model on the excess return
        of the benchmark instead.

        For more information about the method, see the following paper:

        - Ang, A., Hodrick, R. J., Xing, Y., & Zhang, X. (2006). "The Cross-Section of
        Volatility and Expected Returns." The Journal of Finance, 61(1), 259-299.

        The formula is as follows (with the default model="fama_french"):

        - Excess Return_t = α + β_1 * (Mkt-RF)_t + β_2 * SMB_t + β_3 * HML_t + ε_t
        - Idiosyncratic Volatility = Std(ε_t) * √(Trading Days per Period)

        And with model="capm":

        - Excess Return_t = α + β * Benchmark Excess Return_t + ε_t

        Where the regression runs on the daily returns within the period, Std is the sample
        standard deviation (ddof=1) of the residuals and the trading days per period are 252 / 12
        (monthly), 252 / 4 (quarterly) or 252 (yearly), the scaling of the Volatility of the Risk
        module; with annualize=True it is √252 for every period. A period with fewer than 15
        daily returns (the paper's minimum) is NaN.

        Also known as: IVOL, idiosyncratic risk, residual volatility, specific risk.

        Args:
            period (str, optional): The period each regression runs within (monthly, quarterly or yearly).
                Defaults to "monthly", as in the paper.
            model (str, optional): The factors to regress on: "fama_french" for the Fama and French three
                factors (Mkt-RF, SMB, HML) or "capm" for the excess return of the benchmark. Defaults to
                "fama_french".
            rolling (int, optional): Regress on the daily returns of the last `rolling` periods instead of one
                period (e.g. period='monthly' and rolling=12 for a one-year window at every month end).
                Defaults to None.
            annualize (bool, optional): Whether to scale the daily standard deviation of the residuals to a
                year (√252) instead of to the period. Defaults to False.
            region (str, optional): The region whose Fama and French factors to use with model="fama_french":
                "United States", "Developed", "Developed ex US", "Europe", "Japan", "Asia Pacific ex Japan"
                or "North America". Every region's factors are U.S. dollar returns. Defaults to
                "United States".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Idiosyncratic Volatility over time.
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
            pd.DataFrame: The Idiosyncratic Volatility, with the periods as the index and the assets as
            columns.

        Notes:
        - The returns are the daily returns of the Adjusted Close on the asset's own trading days.
        - With model="fama_french", the excess return is the return minus the risk-free rate (RF) of the
        Fama and French dataset, the one-month Treasury bill the factors are built with, as in the paper.
        The dataset is published with a delay of a month or more, so the most recent periods are NaN.
        - With model="capm", the excess returns are taken over the risk-free rate of the Toolkit, an annually
        compounded yield converted to a daily rate with (1 + rate)^(1 / 252) - 1, and the benchmark itself is
        left out, as its residuals are 0.
        - The regression runs on the days with a return for both the asset and every factor.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "TSLA", "KO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-06-30",
        )

        toolkit.risk.get_idiosyncratic_volatility()
        ```

        Which returns:

        | Date    |   AAPL |   TSLA |     KO |   Benchmark |
        |:--------|-------:|-------:|-------:|------------:|
        | 2025-01 | 0.077  | 0.1089 | 0.0393 |      0.0015 |
        | 2025-02 | 0.0598 | 0.1275 | 0.0602 |      0.0028 |
        | 2025-03 | 0.0576 | 0.168  | 0.0413 |      0.0037 |
        | 2025-04 | 0.0545 | 0.1187 | 0.053  |      0.011  |
        | 2025-05 | 0.0659 | 0.0962 | 0.0457 |      0.0035 |
        | 2025-06 | 0.043  | 0.188  | 0.0286 |      0.0023 |
        """
        period = period if period else "monthly"

        if period not in ["monthly", "quarterly", "yearly"]:
            raise ValueError("Period must be monthly, quarterly, or yearly.")
        if model not in ["fama_french", "capm"]:
            raise ValueError("Model must be fama_french or capm.")

        returns = self._historical_data["daily"]["Return"]

        if model == "fama_french":
            factors = performance_model.obtain_fama_and_french_three_factor_dataset(
                region=region
            )
            excess_returns = returns.sub(factors["RF"].reindex(returns.index), axis=0)
            factor_returns = factors[["Mkt-RF", "SMB", "HML"]].reindex(returns.index)
        else:
            risk_free_rate = self._risk_free_rate_data["daily"].reindex(returns.index)
            excess_returns = returns.sub(risk_free_rate, axis=0)
            factor_returns = excess_returns[["Benchmark"]]
            excess_returns = excess_returns.drop(columns="Benchmark")

        volatility = market_liquidity_model.get_idiosyncratic_volatility(
            excess_returns, factor_returns, period, rolling, annualize
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
