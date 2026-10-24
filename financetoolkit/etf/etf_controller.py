"""ETF Module"""

__docformat__ = "google"

from collections.abc import Callable

import pandas as pd

from financetoolkit.etf import etf_model, sec_model
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.statistics_model import PERIODS_PER_YEAR, apply_rounding

# pylint: disable=too-many-instance-attributes,too-many-arguments
# The examples' output tables are wider than the line length.
# ruff: noqa: E501

logger = get_logger()

BENCHMARK_TICKER = "Benchmark"
# The trading days the liquidity is measured over: the last year.
TRADING_DAYS = 252
# The most days between the published NAV and a close to compare them.
NAV_PRICE_DAYS = 4
# Symbols are short; a longer identifier is an ISIN or a name.
MAXIMUM_SYMBOL_LENGTH = 10


class ETF:
    """
    The ETF module analyzes exchange traded funds (and other funds): what they cost, how they
    track their index, how concentrated and how much alike they are, what their holdings are
    valued at and how liquid they are.
    """

    def __init__(
        self,
        tickers: str | list[str],
        daily_historical: pd.DataFrame,
        information_function: Callable[[], pd.DataFrame],
        holdings_function: Callable[[], pd.DataFrame],
        ratios_function: Callable[[list[str]], pd.DataFrame],
        start_date: str | None = None,
        end_date: str | None = None,
        rounding: int | None = 4,
    ):
        """
        Initializes the ETF Controller Class, which the Toolkit creates as `toolkit.etf`.

        Args:
            tickers (str | list[str]): The tickers of the funds.
            daily_historical (pd.DataFrame): The daily historical data of the tickers and the
                Benchmark.
            information_function (Callable[[], pd.DataFrame]): Retrieves the profile of every
                fund (Toolkit.get_etf_information).
            holdings_function (Callable[[], pd.DataFrame]): Retrieves the holdings of every
                fund (Toolkit.get_etf_holdings).
            ratios_function (Callable[[list[str]], pd.DataFrame]): Retrieves the trailing twelve
                month ratios of a list of tickers, the holdings.
            start_date (str | None, optional): The start date. Defaults to None.
            end_date (str | None, optional): The end date. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["SPY", "VOO", "IVV"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.etf.get_profile()
        ```
        """
        self._tickers = [tickers] if isinstance(tickers, str) else list(tickers)
        self._daily_historical = daily_historical
        self._information_function = information_function
        self._holdings_function = holdings_function
        self._ratios_function = ratios_function
        self._start_date = start_date
        self._end_date = end_date
        self._rounding = rounding

    def _funds(self) -> list[str]:
        """The tickers other than the Benchmark and a Portfolio."""
        return [
            ticker
            for ticker in self._tickers
            if ticker not in (BENCHMARK_TICKER, "Portfolio")
        ]

    def _weights(self) -> dict[str, pd.Series]:
        """The weight of every holding per fund, from the holdings."""
        holdings = self._holdings_function()
        if holdings is None or holdings.empty:
            raise ValueError(
                "No holdings are available for these tickers; only ETFs and funds have them."
            )
        # The holdings of a single fund are not indexed by the fund.
        if holdings.index.nlevels == 1 and len(self._funds()) == 1:
            holdings = pd.concat({self._funds()[0]: holdings})
        weights = {}
        for fund in holdings.index.get_level_values(0).unique():
            if fund not in self._funds():
                continue
            fund_holdings = holdings.loc[fund]
            # A holding without a symbol (a bond, cash) is identified by its ISIN or name.
            keys = pd.Series(fund_holdings.index.astype(str), index=fund_holdings.index)
            for fallback in ("ISIN", "Name"):
                if fallback in fund_holdings:
                    missing = keys.str.strip().isin(["", "nan", "None"])
                    keys = keys.where(~missing, fund_holdings[fallback].astype(str))
            fund_weights = pd.to_numeric(fund_holdings["Weight"], errors="coerce")
            fund_weights.index = keys.to_numpy()
            fund_weights = fund_weights.groupby(level=0).sum().dropna()
            # Short positions and derivatives with a negative weight do not add to exposure.
            weights[fund] = fund_weights[fund_weights > 0]
        return weights

    def _round(self, data: pd.DataFrame, rounding: int | None) -> pd.DataFrame:
        return apply_rounding(
            data, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_profile(self, rounding: int | None = None) -> pd.DataFrame:
        """
        Get the profile of every fund: its name, issuer, asset class, domicile, inception date,
        expense ratio, assets under management, net asset value (NAV), latest price and the
        premium or discount of that price to the NAV, the number of holdings and the average
        daily volume and dollar volume. The expense ratio is what holding the fund costs every
        year; a premium means buying it costs more than the holdings are worth, which happens in
        stress for funds of less liquid assets such as bonds. The NAV is the latest Financial
        Modeling Prep publishes, so the premium is only given when the latest close is of about
        the same date (an end date in the past leaves it empty).

        Also known as: ETF profile, expense ratio, assets under management, premium discount.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the Toolkit.

        Returns:
            pd.DataFrame: The profile fields (rows) per fund (columns).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["VOO", "QQQ", "AGG"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-10-01",
        )

        toolkit.etf.get_profile()
        ```

        Which returns:

        |                         | VOO                  | QQQ                         | AGG                                  |
        |:------------------------|:---------------------|:----------------------------|:-------------------------------------|
        | Name                    | Vanguard S&P 500 ETF | Invesco QQQ Trust, Series 1 | iShares Core U.S. Aggregate Bond ETF |
        | ETF Company             | Vanguard             | Invesco                     | IShares                              |
        | Asset Class             | Large Cap Equity     | Equity                      | Fixed Income                         |
        | Domicile                | US                   | US                          | US                                   |
        | Inception Date          | 2010-09-07           | 1999-03-10                  | 2003-09-22                           |
        | Expense Ratio           | 0.0003               | 0.0018                      | 0.0003                               |
        | Assets Under Management | 1700000000000        | 475210352597                | 139388224538                         |
        | NAV                     | 698.45               | 704.38                      | 96.07                                |
        | Price                   | 698.58               | 704.4                       | 95.96                                |
        | Premium/Discount        | 0.0002               | 0.0                         | -0.0011                              |
        | Holdings Count          | 505                  | 102                         | 13445                                |
        | Average Volume          | 8676842.9444         | 49390737.9087               | 9471894.381                          |
        | Average Dollar Volume   | 5668940453.2057      | 32197925675.0503            | 933890010.5428                       |
        """
        information = self._information_function()
        if information is None or information.empty:
            raise ValueError("No fund information is available for these tickers.")

        funds = [fund for fund in self._funds() if fund in information.columns]
        close = self._daily_historical["Close"]
        volume = self._daily_historical["Volume"]
        last_year = close.index[-min(len(close), TRADING_DAYS) :]

        rows = {}
        for fund in funds:
            info = information[fund]
            price = close[fund].dropna()
            latest = float(price.iloc[-1]) if not price.empty else float("nan")
            nav = pd.to_numeric(info.get("NAV"), errors="coerce")
            # The NAV is the latest published, so it is only compared with a close of about
            # the same date.
            nav_date = pd.to_datetime(str(info.get("Updated At"))[:10], errors="coerce")
            price_date = price.index[-1].to_timestamp() if not price.empty else pd.NaT
            comparable = (
                pd.notna(nav_date)
                and pd.notna(price_date)
                and abs((nav_date - price_date).days) <= NAV_PRICE_DAYS
            )
            rows[fund] = {
                "Name": info.get("Name"),
                "ETF Company": info.get("ETF Company"),
                "Asset Class": info.get("Asset Class"),
                "Domicile": info.get("Domicile"),
                "Inception Date": info.get("Inception Date"),
                "Expense Ratio": pd.to_numeric(
                    info.get("Expense Ratio"), errors="coerce"
                ),
                "Assets Under Management": pd.to_numeric(
                    info.get("Assets Under Management"), errors="coerce"
                ),
                "NAV": nav,
                "Price": latest,
                "Premium/Discount": (
                    latest / nav - 1 if nav and comparable else float("nan")
                ),
                "Holdings Count": pd.to_numeric(
                    info.get("Holdings Count"), errors="coerce"
                ),
                "Average Volume": float(volume.loc[last_year, fund].mean()),
                "Average Dollar Volume": float(
                    (close.loc[last_year, fund] * volume.loc[last_year, fund]).mean()
                ),
            }

        rounding = rounding if rounding is not None else self._rounding
        profile = pd.DataFrame(rows)
        if rounding is not None:
            profile = profile.map(
                lambda value: (
                    round(value, rounding)
                    if isinstance(value, float) and not isinstance(value, bool)
                    else value
                )
            )
        return profile

    @handle_errors
    def get_fee_breakdown(
        self, sec_user_agent: str | None = None, rounding: int | None = None
    ) -> pd.DataFrame:
        """
        Get the fee table of the prospectus of every US fund from the SEC: the management fee,
        the distribution and service (12b-1) fee, the other expenses, the fees of the funds it
        invests in (acquired fund fees), the gross expense ratio, the fee waiver, the net expense
        ratio (what investors pay) and the portfolio turnover (the share of the portfolio traded
        in a year, which costs a fund trading costs on top of its expense ratio), as decimals.

        The data comes from the risk/return summaries the SEC publishes each quarter from the
        prospectuses (Form N-1A) of US funds, read back up to five quarters until every fund is
        found. A unit investment trust such as SPY files no such prospectus and is not covered.
        The first call downloads the quarterly files (about 80 MB each); the fees are cached
        for 90 days afterwards. The SEC asks every automated request to identify itself with a
        name and an email address, given as `sec_user_agent` or in the
        FINANCETOOLKIT_SEC_USER_AGENT environment variable.

        See definition: https://www.sec.gov/data-research/sec-markets-data/mutual-fund-prospectus-riskreturn-summary-data-sets

        Also known as: fee table, expense ratio breakdown, 12b-1 fee, management fee, portfolio
        turnover.

        Args:
            sec_user_agent (str | None, optional): Your name and email address, e.g.
                "Jane Doe jane.doe@example.com". Defaults to None, which reads the
                FINANCETOOLKIT_SEC_USER_AGENT environment variable.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the Toolkit.

        Returns:
            pd.DataFrame: The fees, turnover, prospectus date, series and class (rows) per fund
            (columns).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["VTI", "VOO", "AGG", "BND"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.etf.get_fee_breakdown(sec_user_agent="Jane Doe jane.doe@example.com")
        ```

        Which returns:

        |                                      | VTI        | VOO        | AGG        | BND        |
        |:-------------------------------------|:-----------|:-----------|:-----------|:-----------|
        | Management Fee                       | 0.0003     | 0.0002     | 0.0003     | 0.0002     |
        | Distribution and Service (12b-1) Fee | 0.0        | 0.0        | nan        | 0.0        |
        | Other Expenses                       | 0.0        | 0.0001     | 0.0        | 0.0001     |
        | Acquired Fund Fees                   | nan        | nan        | 0.0        | nan        |
        | Gross Expense Ratio                  | 0.0003     | 0.0003     | 0.0003     | 0.0003     |
        | Fee Waiver                           | nan        | nan        | 0.0        | nan        |
        | Net Expense Ratio                    | 0.0003     | 0.0003     | 0.0003     | 0.0003     |
        | Portfolio Turnover                   | 0.03       | 0.02       | 0.62       | 0.38       |
        | Prospectus Date                      | 2025-12-31 | 2025-12-31 | 2026-02-28 | 2025-12-31 |
        | Series                               | S000002848 | S000002839 | S000004362 | S000002564 |
        | Class                                | C000007808 | C000092055 | C000012092 | C000046844 |
        """
        user_agent = sec_model.resolve_user_agent(sec_user_agent)
        breakdown = sec_model.get_fee_breakdown(self._funds(), user_agent)
        if breakdown.empty:
            raise ValueError(
                "None of these tickers has a prospectus fee table at the SEC; it covers US "
                "funds registered under Form N-1A."
            )
        if missing := [fund for fund in self._funds() if fund not in breakdown.index]:
            logger.warning(
                "No SEC fee table was found for %s (a unit investment trust such as SPY, or a "
                "fund outside the US, files none).",
                ", ".join(missing),
            )
        numeric = breakdown.drop(columns=["Prospectus Date", "Series", "Class"]).apply(
            pd.to_numeric, errors="coerce"
        )
        numeric = self._round(numeric, rounding)
        return pd.concat(
            [numeric, breakdown[["Prospectus Date", "Series", "Class"]]], axis=1
        ).T

    @handle_errors
    def get_tracking(
        self,
        period: str = "daily",
        yearly: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Measure how closely every fund tracks the Benchmark of the Toolkit, which for an index
        fund should be the index it follows (set benchmark_ticker to a total return index or to
        another fund of the same index). Per fund:

        - Tracking Difference: the annualized average return difference with the Benchmark. An
          index fund trails its index by roughly its expense ratio; less points to securities
          lending income, more to a costly or loose replication.
        - Tracking Error: the annualized volatility of that difference, how far the fund strays
          from the index from day to day.
        - Correlation, Beta and R Squared of the fund's returns with the Benchmark's.

        With yearly=True the tracking difference is given per calendar year instead, the return
        of the fund minus that of the Benchmark. The daily closes of two funds can differ by the
        timing of their last trade, which adds noise to a daily tracking error; fund reports
        usually measure it from monthly returns.

        Also known as: tracking difference, tracking error, index tracking.

        Args:
            period (str, optional): The return frequency (daily, weekly, monthly, quarterly or
                yearly). Defaults to "daily".
            yearly (bool, optional): Whether to return the tracking difference per year.
                Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the Toolkit.

        Returns:
            pd.DataFrame: The tracking measures (rows) per fund (columns), or the tracking
            difference per year (rows).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["VOO", "IVV", "SPLG"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            benchmark_ticker="SPY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.etf.get_tracking(period="monthly")
        ```

        Which returns:

        |                     |    VOO |    IVV |    SPLG |
        |:--------------------|-------:|-------:|--------:|
        | Tracking Difference | 0.0007 | 0.0006 | -0      |
        | Tracking Error      | 0.0011 | 0.0018 |  0.0021 |
        | Correlation         | 1      | 0.9999 |  0.9999 |
        | Beta                | 0.9994 | 1.0014 |  0.9986 |
        | R Squared           | 0.9999 | 0.9999 |  0.9998 |
        """
        if period not in PERIODS_PER_YEAR:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly or yearly."
            )
        prices = self._daily_historical["Adj Close"]
        if BENCHMARK_TICKER not in prices:
            raise ValueError(
                "Tracking is measured against the Benchmark of the Toolkit."
            )
        prices = prices.loc[self._start_date : self._end_date]

        if yearly:
            yearly_prices = prices.groupby(prices.index.asfreq("Y")).last()
            first = prices.iloc[[0]].set_axis([yearly_prices.index[0] - 1])
            returns = (
                pd.concat([first, yearly_prices]).pct_change(fill_method=None).iloc[1:]
            )
            difference = returns[self._funds()].sub(returns[BENCHMARK_TICKER], axis=0)
            return self._round(difference, rounding)

        if period != "daily":
            frequency = {
                "weekly": "W",
                "monthly": "M",
                "quarterly": "Q",
                "yearly": "Y",
            }[period]
            prices = prices.groupby(prices.index.asfreq(frequency)).last()
        returns = prices.pct_change(fill_method=None)

        tracking = pd.DataFrame(
            {
                fund: etf_model.get_tracking(
                    returns[fund], returns[BENCHMARK_TICKER], PERIODS_PER_YEAR[period]
                )
                for fund in self._funds()
            }
        )
        return self._round(tracking, rounding)

    @handle_errors
    def get_concentration(self, rounding: int | None = None) -> pd.DataFrame:
        """
        Measure how concentrated every fund is from its holdings: the number of holdings, the
        largest holding and its weight, the weight of the ten largest, the
        Herfindahl-Hirschman index (the sum of the squared weights) and the effective number of
        holdings (one over that index). A fund of 500 holdings whose ten largest make up a third
        of it behaves like a fund of far fewer stocks; the effective number says how many
        equally weighted holdings would give the same concentration.

        Also known as: fund concentration, top 10 weight, Herfindahl index, effective number of
        holdings.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the Toolkit.

        Returns:
            pd.DataFrame: The concentration measures (rows) per fund (columns).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["VOO", "QQQ", "RSP"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.etf.get_concentration()
        ```

        Which returns:

                |                              | VOO     | QQQ     | RSP      |
        |:-----------------------------|:--------|:--------|:---------|
        | Number of Holdings           | 507.0   | 105.0   | 507.0    |
        | Largest Holding              | NVDA    | NVDA    | PTC      |
        | Largest Weight               | 0.084   | 0.0814  | 0.003    |
        | Top 10 Weight                | 0.3913  | 0.4665  | 0.0266   |
        | Herfindahl-Hirschman Index   | 0.023   | 0.0316  | 0.002    |
        | Effective Number of Holdings | 43.4661 | 31.6935 | 497.7398 |
        """
        concentration = pd.DataFrame(
            {
                fund: etf_model.get_concentration(weights)
                for fund, weights in self._weights().items()
            }
        )
        numeric = concentration.drop(index="Largest Holding").apply(
            pd.to_numeric, errors="coerce"
        )
        numeric = self._round(numeric.astype(float), rounding)
        return pd.concat([numeric, concentration.loc[["Largest Holding"]]]).reindex(
            concentration.index
        )

    @handle_errors
    def get_overlap(self, rounding: int | None = None) -> pd.DataFrame:
        """
        Measure how much the funds hold of the same: for every pair of funds the sum, over the
        holdings they share, of the smaller of the two weights. An overlap of 0.8 means 80% of
        the two portfolios is the same, so holding both adds little diversification.

        Also known as: ETF overlap, portfolio overlap, holdings overlap.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the Toolkit.

        Returns:
            pd.DataFrame: The overlap of every pair of funds, as decimals.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["VOO", "QQQ", "VTI", "SCHD"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.etf.get_overlap()
        ```

        Which returns:

                |      |    VOO |    QQQ |    VTI |   SCHD |
        |:-----|-------:|-------:|-------:|-------:|
        | VOO  | 1.0017 | 0.556  | 0.8911 | 0.0764 |
        | QQQ  | 0.556  | 1.0014 | 0.4994 | 0.0516 |
        | VTI  | 0.8911 | 0.4994 | 1.0005 | 0.071  |
        | SCHD | 0.0764 | 0.0516 | 0.071  | 1      |
        """
        return self._round(etf_model.get_overlap(self._weights()), rounding)

    @handle_errors
    def get_look_through_valuation(
        self, holdings: int = 25, rounding: int | None = None
    ) -> pd.DataFrame:
        """
        Value every fund through its holdings: the price-to-earnings, price-to-book,
        price-to-sales and price-to-free-cash-flow ratio, the dividend yield and the net profit
        margin of its largest holdings, weighted by their weight in the fund. The multiples are
        weighted harmonic means, the multiple of the fund as a whole, so one holding with a very
        high multiple does not dominate. The coverage is the share of the fund the valued
        holdings make up; more holdings raise it at the cost of more requests.

        The ratios are the trailing twelve month ratios Financial Modeling Prep calculates for
        every holding.

        Also known as: look-through valuation, fund P/E, weighted average P/E, portfolio
        valuation.

        Args:
            holdings (int, optional): The number of largest holdings to value per fund.
                Defaults to 25.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the Toolkit.

        Returns:
            pd.DataFrame: The valuation measures and the coverage (rows) per fund (columns).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["VOO", "QQQ", "SCHD"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.etf.get_look_through_valuation()
        ```

        Which returns:

                |                         |     VOO |     QQQ |    SCHD |
        |:------------------------|--------:|--------:|--------:|
        | Price-to-Earnings       | 26.616  | 31.6436 | 20.7484 |
        | Price-to-Book           |  8.3525 | 10.2411 |  4.7307 |
        | Price-to-Sales          |  7.5579 |  7.0579 |  2.2584 |
        | Price-to-Free-Cash-Flow | 59.957  | 53.445  | 17.0266 |
        | Dividend Yield          |  0.0047 |  0.0038 |  0.0308 |
        | Net Profit Margin       |  0.3673 |  0.3113 |  0.1583 |
        | Coverage                |  0.5324 |  0.7154 |  0.8077 |
        """
        weights = {
            fund: fund_weights.sort_values(ascending=False).iloc[:holdings]
            for fund, fund_weights in self._weights().items()
        }
        # Only holdings with a symbol (not bonds identified by their ISIN) have ratios.
        assets = sorted(
            {
                asset
                for fund_weights in weights.values()
                for asset in fund_weights.index
                if asset and len(asset) <= MAXIMUM_SYMBOL_LENGTH and " " not in asset
            }
        )
        if not assets:
            raise ValueError("The holdings of these funds have no symbols to value.")
        ratios = self._ratios_function(assets)
        if ratios is None or ratios.empty:
            raise ValueError("No ratios are available for the holdings of these funds.")
        ratios = ratios.T if set(assets) & set(ratios.columns) else ratios

        valuation = pd.DataFrame(
            {
                fund: etf_model.get_look_through_valuation(fund_weights, ratios)
                for fund, fund_weights in weights.items()
            }
        )
        return self._round(valuation, rounding)

    @handle_errors
    def get_liquidity(self, rounding: int | None = None) -> pd.DataFrame:
        """
        Measure how liquid every fund is to trade over the last year of the period: its average
        daily volume and dollar volume, that dollar volume relative to the fund's assets under
        management, and the price impact of trading 100 million dollars in a day (the Amihud,
        2002, illiquidity: the average absolute daily return per dollar traded, scaled to 100
        million). An ETF can also be traded through its holdings, so a thinly traded fund of
        liquid holdings is easier to trade than its own volume suggests; a fund whose holdings
        are themselves illiquid (small companies, high yield bonds) is not.

        For more information, see the following paper:

        - Amihud, Y. (2002). "Illiquidity and Stock Returns: Cross-Section and Time-Series
          Effects." Journal of Financial Markets, 5(1), 31-56.

        Also known as: ETF liquidity, dollar volume, Amihud illiquidity, price impact.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which keeps the full precision (the price impact of the largest
                funds is a fraction of a basis point).

        Returns:
            pd.DataFrame: The liquidity measures (rows) per fund (columns).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["VOO", "QQQ", "ARKK"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.etf.get_liquidity(rounding=6)
        ```

        Which returns:

        |                               |               VOO |                QQQ |             ARKK |
        |:------------------------------|------------------:|-------------------:|-----------------:|
        | Average Volume                |    7734399.056000 |    48666218.588000 |  10799597.760000 |
        | Average Dollar Volume         | 4400329488.944520 | 26740006060.354679 | 719035298.234000 |
        | Dollar Volume to Assets       |          0.002588 |           0.056270 |         0.090822 |
        | Price Impact per $100 Million |          0.000179 |           0.000036 |         0.002990 |
        """
        data = self._daily_historical.loc[self._start_date : self._end_date]
        data = data.iloc[-min(len(data), TRADING_DAYS) :]
        information = self._information_function()

        liquidity = {}
        for fund in self._funds():
            close, volume = data["Close"][fund], data["Volume"][fund]
            dollar_volume = close * volume
            assets = (
                pd.to_numeric(
                    information[fund].get("Assets Under Management"), errors="coerce"
                )
                if information is not None and fund in information
                else float("nan")
            )
            liquidity[fund] = {
                "Average Volume": float(volume.mean()),
                "Average Dollar Volume": float(dollar_volume.mean()),
                "Dollar Volume to Assets": (
                    float(dollar_volume.mean()) / assets if assets else float("nan")
                ),
                "Price Impact per $100 Million": etf_model.get_price_impact(
                    close.pct_change(fill_method=None), dollar_volume
                ),
            }
        # A price impact of a fraction of a basis point matters here, so the result is only
        # rounded when asked for.
        return apply_rounding(pd.DataFrame(liquidity), rounding)
