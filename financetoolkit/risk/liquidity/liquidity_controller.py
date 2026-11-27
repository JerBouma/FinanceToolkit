"""Liquidity Risk Module"""

__docformat__ = "google"

from collections.abc import Callable
from typing import Literal

import pandas as pd

from financetoolkit.helpers import handle_portfolio
from financetoolkit.performance import performance_model
from financetoolkit.ratios.helpers import map_period_data_to_daily_data
from financetoolkit.risk.helpers import check_rolling
from financetoolkit.risk.liquidity import market_liquidity_model
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


@filter_shown_columns
@track_sources
class LiquidityRisk:
    """
    Market liquidity: the Amihud illiquidity measure, the price impact (Kyle's, Amihud's and
    Hasbrouck's lambdas and the Roll impact), the Roll spread, the EDGE, Corwin-Schultz and
    Abdi-Ranaldo bid-ask spreads, the Bekker-Parkinson volatility, the share turnover, the
    idiosyncratic volatility and VPIN.
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
        Calculate the Amihud (2002) Illiquidity ratio per asset: how much the price moves per dollar traded,
        high for an illiquid asset.

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
            period (str, optional): The data frequency for returns. Options are: "daily", "weekly", "monthly",
                "quarterly", "yearly". Defaults to "daily".
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
        - With growth, the method calculates the growth of the Amihud Illiquidity values
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
    def get_price_impact(
        self,
        period: str | None = None,
        rolling: int | None = None,
        method: Literal["kyle", "amihud", "hasbrouck", "roll"] = "kyle",
        scale: float | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the price impact of trading per asset and period: how much the price moves per unit traded,
        with Kyle's, Amihud's or Hasbrouck's lambda as the regression slopes of López de Prado (2018, chapter
        19.4) or with the Roll Impact (section 19.3.2).

        Each lambda is the slope of a regression, without an intercept, of the daily price changes on a
        measure of the trading of the day, over the days within the period. A high value is an illiquid asset,
        in which trading moves the price a lot:

        - Kyle's lambda: Kyle (1985) models market makers who cannot tell informed from uninformed orders and
        move the price by λ per share of net order flow. The price change of a day is regressed on its Volume,
        signed with the tick rule (a rise is buying, a fall selling).
        - Amihud's lambda: the absolute log return of a day is regressed on its dollar volume. This is the
        regression form of the Amihud (2002) illiquidity ratio (`get_amihud_illiquidity`, the mean of the
        ratios), which weighs the days by their dollar volume, so a quiet day with a large move does not
        dominate it.
        - Hasbrouck's lambda: Hasbrouck (2009) uses the square root of the signed dollar volume, as the impact
        of an order grows less than proportionally with its size. The log return of a day is regressed on the
        square root of its dollar volume, signed with the tick rule.
        - Roll Impact: the Roll (1984) spread of the period (see `get_roll_spread`) divided by the mean
        dollar volume of its days, the spread paid per dollar traded.

        The measures are mostly used to compare assets and periods: a rising lambda is a market that has
        become thinner. Their level depends on the size of the asset (a large cap trades billions of dollars a
        day), so compare assets of a similar size or look at the changes over time.

        For more information about the methods, see the following sources:

        - Kyle, A. S. (1985). "Continuous Auctions and Insider Trading." Econometrica, 53(6), 1315-1335.
        - Amihud, Y. (2002). "Illiquidity and Stock Returns: Cross-Section and Time-Series Effects." Journal of
        Financial Markets, 5(1), 31-56.
        - Hasbrouck, J. (2009). "Trading Costs and Returns for U.S. Equities: Estimating Effective Costs from
        Daily Data." The Journal of Finance, 64(3), 1445-1477.
        - Roll, R. (1984). "A Simple Implicit Measure of the Effective Bid-Ask Spread in an Efficient Market."
        The Journal of Finance, 39(4), 1127-1139.
        - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, sections 19.3.2 and 19.4.

        The formula is as follows:

        For Kyle's lambda:

        - b_t = sign(ΔP_t), or b_(t-1) when ΔP_t = 0
        - ΔP_t = λ * (b_t * V_t) + ε_t
        - Kyle's Lambda = Σ(ΔP_t * b_t * V_t) / Σ((b_t * V_t)^2) * 1,000,000

        For Amihud's lambda:

        - |r_t| = λ * DV_t + ε_t
        - Amihud's Lambda = Σ(|r_t| * DV_t) / Σ(DV_t^2) * 1,000,000,000

        For Hasbrouck's lambda:

        - b_t = sign(r_t), or b_(t-1) when r_t = 0
        - r_t = λ * (b_t * √DV_t) + ε_t
        - Hasbrouck's Lambda = Σ(r_t * b_t * √DV_t) / Σ(DV_t) * 1,000,000

        For the Roll Impact:

        - Roll Spread = 2 * √(-Cov(ΔP_t, ΔP_(t-1)))
        - Roll Impact = Roll Spread / Mean(DV_t) * 1,000,000,000

        Where ΔP_t is the change of the Close, V_t the Volume, DV_t = Close_t * V_t the dollar volume and r_t
        the log return of the Adjusted Close (continuously compounded) on day t; the sums, the covariance
        (with n - 1 in the denominator) and the mean run over the days within the period. The default `scale` makes the measures of
        stocks readable: Kyle's lambda is the change of the price per million shares of net buying, Amihud's
        lambda the absolute log return per billion dollars traded (as `get_amihud_illiquidity` is shown),
        Hasbrouck's lambda the log return per square root of a dollar times 1,000,000 and the Roll Impact the
        spread in price units per billion dollars traded a day.

        Also known as: price impact coefficient, Kyle's lambda, market depth, Amihud's lambda, Hasbrouck's
        lambda, Roll impact.

        Args:
            period (str, optional): The period each regression runs within. Options are: "weekly", "monthly",
                "quarterly", "yearly". Defaults to "monthly".
            rolling (int, optional): Regress on the days of the last `rolling` periods together instead of one
                period (e.g. period='monthly' and rolling=3 for a three-month lambda at every month end).
                At least 2. Defaults to None.
            method (str, optional): The measure. Options are: "kyle" (Kyle's lambda), "amihud" (Amihud's lambda),
                "hasbrouck" (Hasbrouck's lambda), "roll" (Roll Impact). Defaults to "kyle".
            scale (float | None, optional): A multiplier applied to the measure for readability. Defaults to
                None, which uses 1,000,000 for method="kyle" and "hasbrouck" and 1,000,000,000 for "amihud" and
                "roll".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the measure over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The price impact with the periods as the index and the assets as columns.

        Notes:
        - The method uses the daily prices of each asset on its own trading days; the price change and log
        return of a day are those since the asset's previous trading day.
        - A day without a price change keeps the sign of the last change (the tick rule), as in the book.
        - A period with fewer than two days, or for the Roll Impact with fewer than three pairs of consecutive
        price changes within the period or an autocovariance of zero or above (no Roll spread), is NaN.
        - The lambdas are the regression slopes of the book, not the means of the ratios (ΔP_t / (b_t * V_t)
        and the like), which would weigh every day equally however little traded on it. For the Roll impact a
        positive autocovariance is NaN, as in `get_roll_spread`, and the spread is divided by the mean dollar
        volume of the period.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["SNAP", "ETSY", "LYFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-06-30",
        )

        toolkit.risk.get_price_impact()
        ```

        Which returns:

        | Date    |   SNAP |   ETSY |   LYFT |   Benchmark |
        |:--------|-------:|-------:|-------:|------------:|
        | 2025-01 | 0.0121 | 0.3753 | 0.018  |      0.091  |
        | 2025-02 | 0.008  | 0.2696 | 0.0139 |      0.089  |
        | 2025-03 | 0.008  | 0.2065 | 0.0151 |      0.0879 |
        | 2025-04 | 0.0079 | 0.2512 | 0.0307 |      0.1089 |
        | 2025-05 | 0.006  | 0.2008 | 0.0263 |      0.0672 |
        | 2025-06 | 0.0061 | 0.1986 | 0.0166 |      0.0462 |
        """
        check_option("method", method, ("kyle", "amihud", "hasbrouck", "roll"))

        period = period if period else "monthly"

        check_period(period, ["weekly", "monthly", "quarterly", "yearly"])
        check_rolling(rolling, self._historical_data, period)

        scale = (
            scale
            if scale is not None
            else {
                "kyle": 1_000_000,
                "amihud": 1_000_000_000,
                "hasbrouck": 1_000_000,
                "roll": 1_000_000_000,
            }[method]
        )
        close, volume = self._get_prices("Close"), self._get_prices("Volume")
        dollar_volume = close * volume

        if method == "kyle":
            impact = market_liquidity_model.get_kyle_lambda(
                close, volume, period, rolling, scale
            )
        elif method == "roll":
            impact = market_liquidity_model.get_roll_impact(
                close, dollar_volume, period, rolling, scale
            )
        else:
            estimator = (
                market_liquidity_model.get_amihud_lambda
                if method == "amihud"
                else market_liquidity_model.get_hasbrouck_lambda
            )
            impact = estimator(
                self._get_prices("Adj Close"), dollar_volume, period, rolling, scale
            )

        impact = impact.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=impact,
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
        Calculate the Roll (1984) implied bid-ask spread per asset: the cost of a round trip implied by the
        bounce of the price between the bid and the ask.

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

        Where P is the Close price, ΔP_t = P_t - P_(t-1) and Cov the sample autocovariance (with n - 1 in the
        denominator) of
        consecutive price changes; a covariance of zero or above gives NaN.

        Also known as: Roll's implied spread, Roll measure.

        Args:
            period (str, optional): The data frequency. Options are: "daily", "weekly", "monthly", "quarterly",
                "yearly". Defaults to "daily".
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

        close_prices = self._get_prices("Close", period, within_period)

        result = market_liquidity_model.get_roll_spread(close_prices)

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_bid_ask_spread(
        self,
        period: str | None = None,
        rolling: int | None = None,
        method: Literal["edge", "corwin_schultz", "abdi_ranaldo"] = "edge",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the bid-ask spread per asset and period: the cost of a round trip, estimated from daily
        prices alone, without quote data.

        It uses the EDGE estimator of Ardia, Guidotti and Kroencke (2024), the high-low estimator of Corwin and
        Schultz (2012) or the close-high-low estimator of Abdi and Ranaldo (2017).

        The daily High is almost always a trade at the ask and the daily Low a trade at the bid, while the
        Open and the Close are trades at either. The range of a day and the distance of the Open and Close
        to it therefore hold both the volatility and the bid-ask spread, and each estimator separates the
        two in its own way:

        - EDGE (the Efficient Discrete Generalized Estimator) takes the mid-range (the average of
        the High and Low) as a proxy for the efficient price; the returns between it and the Open and the
        Close bounce with the spread. Four such estimators, two at the open and two at the close (one of
        them a generalization of the Abdi-Ranaldo spread), are combined with the weights of the
        Generalized Method of Moments that minimize the variance of the estimate. The probabilities in the
        formula correct for days without trades, which repeat the previous Close. The estimate is
        asymptotically unbiased and, in the paper's simulations and empirical tests, more precise than the
        Roll, Corwin-Schultz and Abdi-Ranaldo estimators. It is estimated from all pairs of consecutive
        trading days of the period at once.
        - Corwin-Schultz: the volatility part of the range grows with the length of the window
        while the spread part does not, so comparing the ranges of two single days with the range over
        both days together backs out the spread from the daily High and Low. Each pair of consecutive
        trading days gives an estimate and the spread of a period is the average of the estimates within
        it.
        - Abdi-Ranaldo: the distance of the Close to the mid-range of its own day and to that of
        the next day both hold half the spread, and their product removes the volatility of the efficient
        price. Unlike the Corwin-Schultz spread, the estimator uses the Close as well and needs no
        adjustment for the overnight return (a weekend or a gap at the open). Each pair of consecutive
        trading days gives an estimate and the spread of a period is the average of the estimates within
        it.

        The estimators are mostly used where no quotes are available (historical studies, smaller markets)
        and rank assets well by their spread. For liquid large caps the spread is a few basis points, close
        to the estimation error and dominated by the volatility within the day, so read it as a relative
        measure there. The Roll spread, which uses the covariance of the close-to-close price changes, is
        `get_roll_spread`.

        For more information about the methods, see the following papers:

        - Ardia, D., Guidotti, E., & Kroencke, T. A. (2024). "Efficient Estimation of Bid-Ask Spreads from
        Open, High, Low, and Close Prices." Journal of Financial Economics, 161, 103916.
        - Corwin, S. A., & Schultz, P. (2012). "A Simple Way to Estimate Bid-Ask Spreads from
        Daily High and Low Prices." The Journal of Finance, 67(2), 719-760.
        - Abdi, F., & Ranaldo, A. (2017). "A Simple Estimation of Bid-Ask Spreads from Daily Close, High,
        and Low Prices." The Review of Financial Studies, 30(12), 4437-4480.

        The formula is as follows:

        For the EDGE spread:

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

        For the Corwin-Schultz spread:

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

        For the Abdi-Ranaldo spread:

        - η_t = (ln H_t + ln L_t) / 2
        - Two-Day Spread = √(max(4 * (ln C_t - η_t) * (ln C_t - η_(t+1)), 0))
        - Abdi-Ranaldo Spread = Mean(Two-Day Spreads within the period)

        Where H, L and C are the daily High, Low and Close of two consecutive trading days t and t+1 of
        the asset. This is the "two-day corrected" version that the paper uses in its empirical analysis
        (equation 11): a negative two-day estimate of the squared spread is set to 0 before the square
        root and the average are taken.

        Also known as: bid-ask spread estimator, effective spread, Efficient Discrete Generalized
        Estimator, EDGE spread, EDGE estimator, OHLC spread estimator, Corwin-Schultz spread, high-low
        spread estimator, HL spread, CS spread, Abdi-Ranaldo spread, close-high-low spread estimator, CHL
        spread, AR spread.

        Shares similar logic:
        - RiskLabAI, edge_estimator (EDGE): https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/features/microstructural_features/edge.py#L42
        - RiskLabAI, corwin_schultz_estimator (Corwin-Schultz): https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/features/microstructural_features/corwin_schultz.py#L106

        Args:
            period (str, optional): The period to estimate the spread over. Options are: "weekly", "monthly",
                "quarterly", "yearly". Defaults to "monthly", as in the papers.
            rolling (int, optional): Estimate the spread from the pairs of the last `rolling` periods together
                (method="edge"), or average the two-day spreads of the last `rolling` periods
                (method="corwin_schultz" and "abdi_ranaldo"), instead of one period (e.g. period='monthly' and
                rolling=3 for a three-month spread at every month end). At least 2. Defaults to None.
            method (str, optional): The estimator. Options are: "edge" (Ardia, Guidotti and Kroencke, from the
                Open, High, Low and Close), "corwin_schultz" (from the High and Low), "abdi_ranaldo" (from the
                Close, High and Low). Defaults to "edge".
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
        - The method uses the daily prices (split-adjusted) of each asset on its own trading days: a day on
        which only another ticker or the benchmark traded is not part of a pair, so the pair is always two
        consecutive trading days of the asset.
        - Both days of a pair fall within the period; a pair across two periods is left out, also in a
        rolling window.
        - A day with a price of 0 or below (a bad bar) is left out.
        - For the EDGE spread, a period with fewer than two pairs with a price change, or in which the Open
        (or the previous Close) equals both the High and the Low on every pair with a price change (only
        single-price days), is NaN. The estimate follows the equations of the authors' `bidask` package
        (`edge` on the days of the period, with negative squared spreads set to 0).
        - For the Abdi-Ranaldo spread, averaging the square roots of the two-day estimates lowers the level
        when the daily volatility is large relative to the spread (in simulations with a 1% spread and a 1%
        daily volatility the estimate is about 0.8%). The paper accepts this for the closer correlation
        with the effective spread from trade and quote data; the EDGE spread has no such bias.
        - Abdi and Ranaldo discard months with fewer than 12 applicable days; no such minimum is applied
        here, as for the Corwin-Schultz spread, so a period with a single pair has an estimate.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["SNAP", "ETSY", "LYFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-06-30",
        )

        toolkit.risk.get_bid_ask_spread()
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
        check_option("method", method, ("edge", "corwin_schultz", "abdi_ranaldo"))

        period = period if period else "monthly"

        check_period(period, ["weekly", "monthly", "quarterly", "yearly"])
        check_rolling(rolling, self._historical_data, period)

        high, low, close = (
            self._get_prices("High"),
            self._get_prices("Low"),
            self._get_prices("Close"),
        )
        if method == "edge":
            spread = market_liquidity_model.get_edge_spread(
                self._get_prices("Open"), high, low, close, period, rolling
            )
        elif method == "corwin_schultz":
            spread = market_liquidity_model.get_corwin_schultz_spread(
                high, low, close, period, rolling
            )
        else:
            spread = market_liquidity_model.get_abdi_ranaldo_spread(
                high, low, close, period, rolling
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
    def get_bekker_parkinson_volatility(
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
        Calculate the Bekker-Parkinson Volatility per asset and period: the volatility in the daily High/Low
        range once the bid-ask spread of Corwin and Schultz (2012) is taken out.

        The daily High is almost always a trade at the ask and the daily Low one at the bid, so the range of a
        day holds both the volatility and the spread, and a range-based volatility such as Parkinson's
        overstates the volatility of an asset with a wide spread. Corwin and Schultz write the expected squared
        range of one day and of two days in terms of the spread and the volatility; the spread part does not
        grow with the length of the window while the volatility part does. Solving the two equations for the
        spread gives the Corwin-Schultz spread (`get_bid_ask_spread`), and solving
        them for the volatility gives this estimate, as López de Prado (2018, section 19.3.4 and snippet 19.2)
        does; RiskLabAI names it the Bekker-Parkinson volatility.

        For more information about the method, see the following sources:

        - Corwin, S. A., & Schultz, P. (2012). "A Simple Way to Estimate Bid-Ask Spreads from Daily High and Low
        Prices." The Journal of Finance, 67(2), 719-760.
        - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, section 19.3.4.
        - Parkinson, M. (1980). "The Extreme Value Method for Estimating the Variance of the Rate of Return."
        Journal of Business, 53(1), 61-65.

        The formula is as follows:

        - β = [ln(H_t / L_t)]^2 + [ln(H_(t+1) / L_(t+1))]^2
        - γ = [ln(max(H_t, H_(t+1)) / min(L_t, L_(t+1)))]^2
        - k_2 = √(8 / π)
        - σ_t = max((√(β / 2) - √β) / (k_2 * (3 - 2√2)) + √(γ / (k_2^2 * (3 - 2√2))), 0)
        - Bekker-Parkinson Volatility = Mean(σ_t within the period) * √(Trading Days per Period)

        Where H and L are the daily High and Low of two consecutive trading days t and t+1 of the asset, with
        the overnight adjustment of day t+1 of the Corwin-Schultz spread. σ_t is the volatility of the log price
        (continuously compounded) over one day; the mean is scaled to the period as the Parkinson Volatility of
        `get_volatility` is, by the square root of the trading days per period (252 / 12 for monthly, or
        365 / 12 for a ticker that trades on weekends).

        Also known as: Corwin-Schultz volatility, spread-adjusted high-low volatility.

        Shares similar logic:
        - RiskLabAI, bekker_parkinson_volatility_estimates: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/features/microstructural_features/bekker_parkinson_volatility_estimator.py#L51

        Args:
            period (str, optional): The period to estimate the volatility over and scale to. Options are:
                "weekly", "monthly", "quarterly", "yearly". Defaults to "monthly", as for the Corwin-Schultz spread.
            rolling (int, optional): Average the two-day volatilities of the last `rolling` periods instead of
                one period (e.g. period='monthly' and rolling=3 for a three-month estimate at every month end).
                At least 2. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the volatility over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The volatility as a decimal, with the periods as the index and the assets as columns.

        Notes:
        - The pairs are those of the Corwin-Schultz spread: consecutive trading days of the asset of which both
        fall within the period, without days with a price of 0 or below.
        - A two-day volatility below zero (the range over both days is narrow relative to the single days) is
        set to 0 before the average is taken, as in López de Prado's snippet 19.2.
        - RiskLabAI's `bekker_parkinson_volatility_estimates` averages β over a rolling window of days but takes
        γ of the last pair only, and does not adjust for the overnight return. Here σ is taken per pair and
        averaged per period, as the Corwin-Schultz spread is; per pair the two use the same formula.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["SNAP", "ETSY", "LYFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-06-30",
        )

        toolkit.risk.get_bekker_parkinson_volatility()
        ```

        Which returns:

        | Date    |   SNAP |   ETSY |   LYFT |   Benchmark |
        |:--------|-------:|-------:|-------:|------------:|
        | 2025-01 | 0.1554 | 0.1103 | 0.1333 |      0.03   |
        | 2025-02 | 0.1062 | 0.1188 | 0.1899 |      0.0311 |
        | 2025-03 | 0.1375 | 0.1276 | 0.1383 |      0.0458 |
        | 2025-04 | 0.1692 | 0.1336 | 0.1422 |      0.0851 |
        | 2025-05 | 0.1086 | 0.1143 | 0.1353 |      0.0319 |
        | 2025-06 | 0.1184 | 0.1271 | 0.0643 |      0.0236 |
        """
        period = period if period else "monthly"

        check_period(period, ["weekly", "monthly", "quarterly", "yearly"])
        check_rolling(rolling, self._historical_data, period)

        volatility = market_liquidity_model.get_bekker_parkinson_volatility(
            self._get_prices("High"),
            self._get_prices("Low"),
            self._get_prices("Close"),
            period,
            rolling,
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
        Calculate the Share Turnover per asset and period: the number of shares traded relative to the
        number of shares outstanding, high for a liquid asset.

        Datar, Naik and Radcliffe (1998) use the turnover as a liquidity proxy: a share that
        changes hands often can be traded at a lower cost. They find that stocks with a lower
        turnover earn higher returns, also after controlling for size, the book-to-market
        ratio and the market beta. The turnover of a period is the share of the outstanding
        shares that was traded in it, the sum of the daily turnover within the period.

        The paper measures the turnover per month and averages it over the three months before
        the month of the returns, which a rolling window of 3 gives with the default monthly period.
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
            period (str, optional): The period to sum the daily turnover over. Options are: "weekly", "monthly",
                "quarterly", "yearly". Defaults to "monthly", as in the paper.
            rolling (int, optional): The mean of the turnover of the last `rolling` periods instead of the
                turnover per period (e.g. rolling=3 for the paper's three-month average). At least 2. Defaults to None.
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
        period (from the filing day with a point-in-time Toolkit), so a day never uses a share count that was not yet
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

        check_period(period, ["weekly", "monthly", "quarterly", "yearly"])
        check_rolling(rolling, self._historical_data, period)
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
        Calculate the Idiosyncratic Volatility per asset and period: the volatility of the part of the
        returns the factors do not explain.

        Ang, Hodrick, Xing and Zhang (2006) regress the daily excess returns of every stock on
        the Fama and French three factors within each month and take the standard deviation
        of the residuals. Stocks with a high idiosyncratic volatility earn strikingly low
        returns in the following month, the idiosyncratic volatility puzzle: a diversified
        investor is not exposed to this risk, so it should not be priced at all, let alone
        negatively. With the CAPM the regression is the market model on the excess return
        of the benchmark instead.

        For more information about the method, see the following paper:

        - Ang, A., Hodrick, R. J., Xing, Y., & Zhang, X. (2006). "The Cross-Section of
        Volatility and Expected Returns." The Journal of Finance, 61(1), 259-299.

        The formula is as follows (with the default Fama and French three factors):

        - Excess Return_t = α + β_1 * (Mkt-RF)_t + β_2 * SMB_t + β_3 * HML_t + ε_t
        - Idiosyncratic Volatility = Std(ε_t) * √(Trading Days per Period)

        And with the CAPM:

        - Excess Return_t = α + β * Benchmark Excess Return_t + ε_t

        Where the regression runs on the daily returns within the period, Std is the sample
        standard deviation (with n - 1 in the denominator) of the residuals and the trading days per period are 252 / 12
        (monthly), 252 / 4 (quarterly) or 252 (yearly), the scaling of the Volatility of the Risk
        module; annualized it is √252 for every period. A period with fewer than 15
        daily returns (the paper's minimum) is NaN.

        Also known as: IVOL, idiosyncratic risk, residual volatility, specific risk.

        Args:
            period (str, optional): The period each regression runs within. Options are: "monthly", "quarterly",
                "yearly". Defaults to "monthly", as in the paper.
            model (str, optional): The factors to regress on. Options are: "fama_french" (the Fama and French
                three factors Mkt-RF, SMB and HML), "capm" (the excess return of the benchmark). Defaults to
                "fama_french".
            rolling (int, optional): Regress on the daily returns of the last `rolling` periods instead of one
                period (e.g. period='monthly' and rolling=12 for a one-year window at every month end).
                At least 2. Defaults to None.
            annualize (bool, optional): Whether to scale the daily standard deviation of the residuals to a
                year (√252) instead of to the period. Defaults to False.
            region (str, optional): The region whose Fama and French factors to use with model="fama_french".
                Options are: "United States", "Developed", "Developed ex US", "Europe", "Japan", "Asia Pacific
                ex Japan", "North America". Every region's factors are U.S. dollar returns. Defaults to
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
        - With the Fama and French factors, the excess return is the return minus the risk-free rate (RF) of the
        Fama and French dataset, the one-month Treasury bill the factors are built with, as in the paper.
        The dataset is published with a delay of a month or more, so the most recent periods are NaN.
        - With the CAPM, the excess returns are taken over the risk-free rate of the Toolkit, an annually
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

        check_period(period, ["monthly", "quarterly", "yearly"])
        check_rolling(rolling, self._historical_data, period)
        check_option("model", model, ["fama_french", "capm"])

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

    @handle_portfolio
    @handle_errors
    def get_vpin(
        self,
        buckets: int = 50,
        window: int = 50,
        degrees_of_freedom: float | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Volume-Synchronized Probability of Informed Trading (VPIN) of Easley, López de Prado and
        O'Hara (2012) per asset and day, from daily bars: how one-sided, and thus toxic, the order flow is.

        VPIN measures the toxicity of the order flow, the share of the volume that is one-sided: informed traders
        buy (or sell) persistently, while uninformed trading roughly balances. The volume is cut into buckets of
        the same size (a volume clock rather than a time clock), the volume of each bucket is split into buys and
        sells with the Bulk Volume Classification (a large rise is mostly buying, a day without a change half
        buying and half selling) and VPIN is the mean order imbalance over the last 50 buckets. A high VPIN
        means a one-sided market in which market makers lose to informed traders and may withdraw; the paper
        finds that VPIN was at its highest in the hours before the flash crash of 6 May 2010.

        This is a coarse approximation of a measure designed for intraday data. The paper fills 50 buckets a day
        from one-minute bars, so its VPIN measures the toxicity within a day. On daily bars the finest
        classification is a whole day, which nets the buying and selling within it, and a bucket holds about a
        day of volume, so VPIN here is a slow-moving estimate of the one-sidedness of the volume over about 50
        trading days. Use it to compare assets and periods, not as the published intraday VPIN or as a warning
        of an intraday liquidity crash.

        For more information about the method, see the following sources:

        - Easley, D., López de Prado, M., & O'Hara, M. (2012). "Flow Toxicity and Liquidity in a High-Frequency
        World." The Review of Financial Studies, 25(5), 1457-1493.
        - Easley, D., López de Prado, M., & O'Hara, M. (2016). "Discerning Information from Trade Data." Journal
        of Financial Economics, 120(2), 269-285.
        - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, section 19.5.2.

        The formula is as follows:

        - ΔP_t = P_t - P_(t-1)
        - V^B_t = V_t * Z(ΔP_t / σ_ΔP)
        - V^B_τ = the buy volume of the volume in bucket τ, V^S_τ = V - V^B_τ
        - VPIN = Σ |V^S_τ - V^B_τ| / (n * V) over the last n buckets

        Where P is the Adjusted Close and V_t the Volume of day t, Z the cumulative distribution function of the
        standard normal distribution (or of the Student-t distribution with `degrees_of_freedom`), σ_ΔP the
        sample standard deviation (with n - 1 in the denominator) of the price changes of the last `window` days, V the volume of a
        bucket and n the number of `buckets`.

        Also known as: volume-synchronized probability of informed trading, flow toxicity, order flow toxicity,
        bulk volume classification, BVC.

        Shares similar logic:
        - RiskLabAI, volume_synchronized_pin: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/features/microstructural_features/order_flow.py#L157

        Args:
            buckets (int, optional): The number of buckets n VPIN is the mean over. Defaults to 50, as in the
                paper.
            window (int, optional): The number of days the standard deviation of the price changes of the
                classification is taken over. Defaults to 50.
            degrees_of_freedom (float | None, optional): The degrees of freedom of a Student-t distribution for
                the classification, as in Easley, López de Prado and O'Hara (2016), instead of the standard normal
                distribution. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: VPIN between 0 and 1, with the dates as the index and the assets as columns.

        Notes:
        - The volume of a bucket is the mean daily Volume of the asset over its first `window` days, the days
        before the first one classified, so a bucket holds about a day of trading at the start of the data and the
        volume clock is fixed in advance, as in the paper. Every VPIN thus uses the data up to its day only; when
        the volume grows over the years, the buckets fill faster and VPIN spans fewer days.
        - A day that fills more than the rest of a bucket carries its excess volume into the next bucket with its
        own buy share, as in the paper, so every bucket holds the same volume.
        - VPIN is updated on the day a bucket completes and kept on the days in between; it is NaN until the
        first `window` days are classified and `buckets` buckets are complete, so start the Toolkit about half
        a year before the first date of interest.
        - The days are those the asset traded on; the Adjusted Close keeps a dividend from counting as selling.
        - Every bucket holds the same volume V, so the mean imbalance of the buckets is divided by that volume.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["SNAP", "ETSY", "LYFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2024-07-01",
            end_date="2025-06-30",
        )

        toolkit.risk.get_vpin().tail()
        ```

        Which returns:

        | Date       |   SNAP |   ETSY |   LYFT |   Benchmark |
        |:-----------|-------:|-------:|-------:|------------:|
        | 2025-06-24 | 0.2988 | 0.5868 | 0.4158 |      0.2241 |
        | 2025-06-25 | 0.3125 | 0.5908 | 0.4132 |      0.2323 |
        | 2025-06-26 | 0.3138 | 0.6208 | 0.4149 |      0.2339 |
        | 2025-06-27 | 0.3097 | 0.6293 | 0.4178 |      0.2457 |
        | 2025-06-30 | 0.3153 | 0.6162 | 0.4109 |      0.2541 |
        """
        vpin = market_liquidity_model.get_vpin(
            self._get_prices("Adj Close"),
            self._get_prices("Volume"),
            buckets=buckets,
            window=window,
            degrees_of_freedom=degrees_of_freedom,
        )

        vpin = vpin.loc[self._start_date : self._end_date]

        return apply_rounding(
            vpin, rounding if rounding is not None else self._rounding
        )
