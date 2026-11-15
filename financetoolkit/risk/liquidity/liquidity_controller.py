"""Liquidity Risk Module"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.helpers import handle_portfolio
from financetoolkit.risk.base.base_controller import RiskBase
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
class LiquidityRisk(RiskBase):
    """
    Market liquidity: the Amihud illiquidity measure and the Roll spread.
    """

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
