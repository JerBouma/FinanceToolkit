"""ETF Module"""

__docformat__ = "google"

from collections.abc import Callable

import pandas as pd

from financetoolkit.etf import etf_model
from financetoolkit.utilities.dataframe_model import filter_columns
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
# The most days the close compared with the NAV may lie before the NAV's date (a weekend).
NAV_PRICE_DAYS = 3
# Symbols are short; a longer identifier is an ISIN or a name.
MAXIMUM_SYMBOL_LENGTH = 10
# The NAV is struck at the 16:00 New York close, so an update at any time is of the last close
# 16 hours or more before it (an update before the open is of the previous day's close).
NAV_CLOSE_HOURS = 16
# A premium or discount beyond half the NAV is a data error (a NAV in another unit or
# currency, a stale price), not a market price.
MAXIMUM_PREMIUM = 0.5
# Some exchanges quote in a minor unit: London in pence (GBp, also written GBX), Johannesburg
# in cents (ZAc) and Tel Aviv in agorot (ILA). The NAV is in the major unit.
MINOR_CURRENCIES = {
    "GBp": ("GBP", 0.01),
    "GBX": ("GBP", 0.01),
    "ZAc": ("ZAR", 0.01),
    "ILA": ("ILS", 0.01),
}


def _major_currency(currency: object) -> tuple[str | None, float]:
    """The major currency of a currency code and the factor to it, e.g. GBp is GBP 0.01."""
    if not isinstance(currency, str) or not currency.strip():
        return None, 1.0
    currency = currency.strip()
    if currency in MINOR_CURRENCIES:
        return MINOR_CURRENCIES[currency]
    return currency.upper(), 1.0


class ETF:
    """
    The ETF module analyzes exchange traded funds (and other funds): what they hold, what they
    cost, how they track their index, how concentrated and how much alike they are, what their holdings are
    valued at and how liquid they are.
    """

    def __init__(
        self,
        tickers: str | list[str],
        historical_function: Callable[[], pd.DataFrame],
        fund_data_function: Callable[..., pd.DataFrame | None],
        ratios_function: Callable[[list[str]], pd.DataFrame],
        start_date: str | None = None,
        end_date: str | None = None,
        rounding: int | None = 4,
        currency_function: Callable[[], pd.Series] | None = None,
    ):
        """
        Initializes the ETF Controller Class, which the Toolkit creates as `toolkit.etf`.

        Args:
            tickers (str | list[str]): The tickers of the funds.
            historical_function (Callable[[], pd.DataFrame]): Retrieves the daily historical
                data of the tickers and the Benchmark, when an analysis first needs it.
            fund_data_function (Callable[..., pd.DataFrame | None]): Retrieves a fund dataset
                of the tickers (holdings, information, country_weightings or
                sector_weightings), with overwrite= to retrieve it again.
            ratios_function (Callable[[list[str]], pd.DataFrame]): Retrieves the trailing twelve
                month ratios of a list of tickers, the holdings.
            start_date (str | None, optional): The start date. Defaults to None.
            end_date (str | None, optional): The end date. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            currency_function (Callable[[], pd.Series] | None, optional): Retrieves the
                currency every ticker is listed (and its prices are quoted) in, to compare the
                close with a NAV of the same currency. Defaults to None, which leaves the
                listing currency unknown.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["SPY", "VOO", "IVV"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.etf.get_profile()
        ```
        """
        self._tickers = [tickers] if isinstance(tickers, str) else list(tickers)
        self._historical_function = historical_function
        self._daily_historical: pd.DataFrame | None = None
        self._fund_data_function = fund_data_function
        self._ratios_function = ratios_function
        self._start_date = start_date
        self._end_date = end_date
        self._rounding = rounding
        self._currency_function = currency_function

    def _listing_currencies(self) -> pd.Series:
        """The currency every ticker is listed in, empty when it cannot be retrieved."""
        if self._currency_function is None:
            return pd.Series(dtype=object)
        try:
            currencies = self._currency_function()
        # The listing currency only guards the premium, so failing to retrieve it must not
        # fail the profile.
        except Exception as error:  # noqa: BLE001
            logger.warning("The listing currencies could not be retrieved: %s", error)
            return pd.Series(dtype=object)
        return (
            currencies if isinstance(currencies, pd.Series) else pd.Series(dtype=object)
        )

    def _funds(self) -> list[str]:
        """The tickers other than the Benchmark and a Portfolio."""
        return [
            ticker
            for ticker in self._tickers
            if ticker not in (BENCHMARK_TICKER, "Portfolio")
        ]

    def _historical(self) -> pd.DataFrame:
        """The daily historical data of the tickers and the Benchmark."""
        if self._daily_historical is None:
            self._daily_historical = self._historical_function()
        return self._daily_historical

    def _weights(self) -> dict[str, pd.Series]:
        """The weight of every holding per fund, from the holdings."""
        holdings = self._fund_data_function("holdings")
        if holdings is None or holdings.empty:
            raise ValueError(
                "No holdings are available for these tickers; only ETFs and funds have them."
            )
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
        # A fund missing from the holdings has no weights, so its results are NaN.
        return {
            fund: weights.get(fund, pd.Series(dtype=float)) for fund in self._funds()
        }

    @staticmethod
    def _sorted_weightings(weightings: pd.DataFrame | None) -> pd.DataFrame | None:
        """The weightings with the largest weights of the first fund first."""
        if weightings is None or weightings.empty:
            return weightings
        # A fund lists only the sectors or countries it holds, so one another fund holds has
        # no weight in it rather than an unknown one.
        return weightings.fillna(0).sort_values(
            by=weightings.columns[0], ascending=False, kind="stable"
        )

    def _round(self, data: pd.DataFrame, rounding: int | None) -> pd.DataFrame:
        return apply_rounding(
            data, rounding if rounding is not None else self._rounding
        )

    def get_holdings(
        self,
        overwrite: bool = False,
        show_columns: list[str] | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame | None:
        """
        Get the holdings of every fund: each asset it holds with the number of shares, the
        market value and the weight in the fund (as a decimal). This shows what the fund is
        actually exposed to and how concentrated it is; get_concentration and get_overlap
        summarize the weights.

        Tickers that are not an ETF or fund have no holdings and return no data. They are not
        removed from the Toolkit instance.

        Also known as: fund holdings, ETF constituents, fund portfolio.

        Args:
            overwrite (bool, optional): Whether to retrieve the holdings again. Defaults to False.
            show_columns (list[str] | None, optional): A list of column names to keep in the
                result. Invalid names are reported and ignored. Defaults to None, which keeps
                every column.
            limit (int | None, optional): The number of largest holdings (by weight) to keep per
                fund, e.g. 10 for the top 10, or of the smallest for a negative number, e.g.
                -10. Defaults to None, which keeps every holding. The holdings carry no sector or
                country; get_sector_weightings and get_country_weightings break the fund down by
                those.

        Returns:
            pd.DataFrame | None: The holdings per fund, indexed by asset (only by asset for a
            single fund). None without an API key.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["QQQ", "VTI"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.etf.get_holdings().loc["QQQ"].head()
        ```

        Which returns:

        | Asset   | Name                       | ISIN         | CUSIP     |      Shares |    Weight |   Market Value | Updated At          |
        |:--------|:---------------------------|:-------------|:----------|------------:|----------:|---------------:|:--------------------|
        | NVDA    | NVIDIA Corp                | US67066G1040 | 67066G104 | 1.80964e+08 | 0.0850518 |    4.32937e+10 | 2026-10-07 14:14:58 |
        | AAPL    | Apple Inc                  | US0378331005 | 037833100 | 1.09586e+08 | 0.0718253 |    3.65611e+10 | 2026-10-07 14:14:58 |
        | MSFT    | Microsoft Corp             | US5949181045 | 594918104 | 5.57574e+07 | 0.057978  |    2.95124e+10 | 2026-10-07 14:14:58 |
        | MU      | Micron Technology Inc      | US5951121038 | 595112103 | 2.34119e+07 | 0.0480888 |    2.44785e+10 | 2026-10-07 14:14:58 |
        | AMD     | Advanced Micro Devices Inc | US0079031078 | 007903107 | 3.38406e+07 | 0.043174  |    2.19767e+10 | 2026-10-07 14:14:58 |
        """
        holdings = self._fund_data_function("holdings", overwrite=overwrite)
        if holdings is None:
            return None

        if limit is not None and not holdings.empty and "Weight" in holdings:
            # The largest holdings of each fund (or, for a negative limit, the smallest), as
            # a full fund can hold thousands.
            holdings = (
                holdings.assign(
                    _weight=pd.to_numeric(holdings["Weight"], errors="coerce")
                )
                .sort_values("_weight", ascending=limit < 0, kind="stable")
                .groupby(level=0, sort=False)
                .head(abs(limit))
                .drop(columns="_weight")
                .reindex(holdings.index.get_level_values(0).unique(), level=0)
            )

        if len(self._tickers) == 1 and not holdings.empty:
            return filter_columns(holdings.loc[self._tickers[0]], show_columns)

        return filter_columns(holdings, show_columns)

    def get_information(self, overwrite: bool = False) -> pd.DataFrame | None:
        """
        Get everything Financial Modeling Prep publishes about every fund: its name,
        identifiers, issuer, asset class, domicile, inception date, expense ratio (as a
        decimal), assets under management, net asset value, number of holdings, description
        and website. The expense ratio in particular determines how much of the return is lost
        to costs every year. get_profile adds the latest price, the premium or discount to the
        NAV and the dollar volume to the key figures.

        Tickers that are not an ETF or fund return no data. They are not removed from the
        Toolkit instance.

        Also known as: fund profile, ETF profile, fund information.

        Args:
            overwrite (bool, optional): Whether to retrieve the information again. Defaults to
                False.

        Returns:
            pd.DataFrame | None: The information (rows) per fund (columns). None without an API
            key.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["QQQ", "VTI"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.etf.get_information().drop(["Description", "Website"])
        ```

        Which returns:

        |                         | QQQ                         | VTI                                         |
        |:------------------------|:----------------------------|:--------------------------------------------|
        | Name                    | Invesco QQQ Trust, Series 1 | Vanguard Morningstar Total Stock Market ETF |
        | ISIN                    | US46090E1038                | US9229087690                                |
        | CUSIP                   | 46090E103                   | 922908769                                   |
        | Asset Class             | Equity                      | Large Cap Equity                            |
        | Domicile                | US                          | US                                          |
        | ETF Company             | Invesco                     | Vanguard                                    |
        | Inception Date          | 1999-03-10                  | 2001-05-24                                  |
        | Expense Ratio           | 0.0018                      | 0.0003                                      |
        | Assets Under Management | 508484484641                | 2300000000000                               |
        | Average Volume          | 39278732                    | 3259438                                     |
        | NAV                     | 759.5                       | 381.13                                      |
        | NAV Currency            | USD                         | USD                                         |
        | Holdings Count          | 102                         | 3598                                        |
        | Actively Trading        | True                        | True                                        |
        | Updated At              | 2026-10-08T01:48:10.006Z    | 2026-10-08T02:22:30.046Z                    |
        """
        return self._fund_data_function("information", overwrite=overwrite)

    def get_country_weightings(self, overwrite: bool = False) -> pd.DataFrame | None:
        """
        Get how every fund is allocated across countries, as decimals, the largest weights of
        the first fund first. Two funds tracking similar markets can differ considerably here,
        which matters for the currency and political risk the fund carries.

        Tickers that are not an ETF or fund return no data. They are not removed from the
        Toolkit instance.

        Also known as: country allocation, geographic exposure, country exposure.

        Args:
            overwrite (bool, optional): Whether to retrieve the weightings again. Defaults to
                False.

        Returns:
            pd.DataFrame | None: The weight per country (rows) and fund (columns). None without
            an API key.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["QQQ", "VTI"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.etf.get_country_weightings().head()
        ```

        Which returns:

        | Country        |    QQQ |    VTI |
        |:---------------|-------:|-------:|
        | United States  | 0.9445 | 0.9733 |
        | United Kingdom | 0.0164 | 0.0048 |
        | Netherlands    | 0.0135 | 0.0003 |
        | Canada         | 0.0096 | 0.0007 |
        | Singapore      | 0.0074 | 0.0025 |
        """
        return self._sorted_weightings(
            self._fund_data_function("country_weightings", overwrite=overwrite)
        )

    def get_sector_weightings(self, overwrite: bool = False) -> pd.DataFrame | None:
        """
        Get how every fund is allocated across sectors, as decimals, the largest weights of the
        first fund first. This shows whether a fund that looks broad is in fact concentrated
        in a few sectors, such as technology in many large-cap indices.

        Tickers that are not an ETF or fund return no data. They are not removed from the
        Toolkit instance.

        Also known as: sector allocation, sector exposure, sector breakdown.

        Args:
            overwrite (bool, optional): Whether to retrieve the weightings again. Defaults to
                False.

        Returns:
            pd.DataFrame | None: The weight per sector (rows) and fund (columns). None without an
            API key.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["QQQ", "VTI"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.etf.get_sector_weightings()
        ```

        Which returns:

        | Sector                 |      QQQ |      VTI |
        |:-----------------------|---------:|---------:|
        | Technology             | 0.603104 | 0.365041 |
        | Communication Services | 0.117791 | 0.086689 |
        | Consumer Cyclical      | 0.098732 | 0.089865 |
        | Consumer Defensive     | 0.057355 | 0.042514 |
        | Industrials            | 0.055378 | 0.088185 |
        | Healthcare             | 0.036579 | 0.099808 |
        | Utilities              | 0.010639 | 0.019731 |
        | Basic Materials        | 0.009198 | 0.022242 |
        | Cash & Others          | 0.004737 | 0.001929 |
        | Energy                 | 0.004538 | 0.037377 |
        | Financial Services     | 0.001949 | 0.123878 |
        | Real Estate            | 0        | 0.022741 |
        """
        return self._sorted_weightings(
            self._fund_data_function("sector_weightings", overwrite=overwrite)
        )

    @handle_errors
    def get_profile(
        self, stale_nav: bool = False, rounding: int | None = None
    ) -> pd.DataFrame:
        """
        Get the profile of every fund: its name, issuer, asset class, domicile, inception date,
        expense ratio, assets under management, net asset value (NAV), latest price and the
        premium or discount of that price to the NAV, the number of holdings and the average
        daily volume and dollar volume. The expense ratio is what holding the fund costs every
        year; a premium means buying it costs more than the holdings are worth, which happens in
        stress for funds of less liquid assets such as bonds. The NAV is the latest Financial
        Modeling Prep publishes and the NAV Date the New York close it was struck at (the last
        16:00 close before its update), so the premium compares the close on the NAV Date with
        the NAV (SEC Rule 6c-11), not the latest Price. It is empty without a close within three
        days before the NAV Date (an end date in the past), when the NAV is in another currency
        than the listing (a close in pence, GBp or GBX, is compared in pounds) and, with a logged
        warning, when it exceeds 50% either way, which is a data error rather than a price.
        With stale_nav=True the premium is given however old the NAV is: it compares the NAV
        with the last close on or before the NAV Date, or with the latest close when the prices
        end before the NAV Date or the NAV has no date. A NAV days older than the close then
        mostly shows how the market moved since, not what the fund trades at.

        Also known as: ETF profile, expense ratio, assets under management, premium discount.

        Args:
            stale_nav (bool, optional): Whether to give the premium for a NAV that is not
                current, comparing it with the nearest close available. Defaults to False,
                which leaves the premium empty without a close within three days before the
                NAV Date.
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
        | NAV Date                | 2026-10-23           | 2026-10-23                  | 2026-10-23                           |
        | Price                   | 698.58               | 704.4                       | 95.96                                |
        | Premium/Discount        | 0.0002               | 0.0                         | -0.0011                              |
        | Holdings Count          | 505                  | 102                         | 13445                                |
        | Average Volume          | 8676842.9444         | 49390737.9087               | 9471894.381                          |
        | Average Dollar Volume   | 5668940453.2057      | 32197925675.0503            | 933890010.5428                       |
        """
        information = self._fund_data_function("information")
        if information is None or information.empty:
            raise ValueError("No fund information is available for these tickers.")

        funds = [fund for fund in self._funds() if fund in information.columns]
        historical = self._historical()
        close, volume = historical["Close"], historical["Volume"]
        last_year = close.index[-min(len(close), TRADING_DAYS) :]

        currencies = self._listing_currencies()

        rows = {}
        for fund in funds:
            info = information[fund]
            price = close[fund].dropna()
            latest = float(price.iloc[-1]) if not price.empty else float("nan")
            nav = pd.to_numeric(info.get("NAV"), errors="coerce")
            # The NAV is struck at the 16:00 New York close and published in UTC, often after
            # midnight or the next morning, so its date is that of the last close 16 hours or
            # more before the update in New York; the premium compares it with the close of
            # that date (SEC Rule 6c-11), not with a later close.
            raw = str(info.get("Updated At") or "").strip()
            updated = pd.to_datetime(raw, utc=True, errors="coerce")
            if pd.isna(updated):
                nav_date = pd.NaT
            elif len(raw) <= 10:  # noqa: PLR2004
                # A date without a time is the NAV date itself.
                nav_date = updated.tz_localize(None).normalize()
            else:
                nav_date = (
                    updated.tz_convert("America/New_York").tz_localize(None)
                    - pd.Timedelta(hours=NAV_CLOSE_HOURS)
                ).normalize()
            dates = (
                price.index.to_timestamp() if not price.empty else pd.DatetimeIndex([])
            )
            before = price[dates <= nav_date] if pd.notna(nav_date) else price.iloc[:0]
            close_at_nav = (
                float(before.iloc[-1])
                if not before.empty
                and (nav_date - before.index[-1].to_timestamp()).days <= NAV_PRICE_DAYS
                else float("nan")
            )
            if pd.notna(close_at_nav) and before.index[-1].to_timestamp() != nav_date:
                if nav_date.dayofweek >= 5 or dates.max() > nav_date:  # noqa: PLR2004
                    # A weekend or holiday date from the 16 hours is reported as its close.
                    nav_date = before.index[-1].to_timestamp()
                else:
                    # The prices end before the NAV date, so its close is not known yet.
                    close_at_nav = float("nan")
            if stale_nav and pd.isna(close_at_nav):
                # Asked for regardless of age: the nearest close there is.
                close_at_nav = float(before.iloc[-1]) if not before.empty else latest
            premium = self._premium(
                fund, close_at_nav, nav, info.get("NAV Currency"), currencies.get(fund)
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
                "NAV Date": (
                    nav_date.strftime("%Y-%m-%d") if pd.notna(nav_date) else None
                ),
                "Price": latest,
                "Premium/Discount": premium,
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

    @staticmethod
    def _premium(
        fund: str,
        close: float,
        nav: float,
        nav_currency: object,
        listing_currency: object,
    ) -> float:
        """
        The premium (positive) or discount of a close to the NAV, NaN when they are in
        different currencies or the premium is implausible.
        """
        if not nav or pd.isna(nav) or pd.isna(close):
            return float("nan")
        nav_major, nav_factor = _major_currency(nav_currency)
        listing_major, listing_factor = _major_currency(listing_currency)
        if nav_major and listing_major and nav_major != listing_major:
            logger.info(
                "The NAV of %s is in %s and its close in %s, so no premium is given.",
                fund,
                nav_currency,
                listing_currency,
            )
            return float("nan")
        premium = (close * listing_factor) / (nav * nav_factor) - 1
        if abs(premium) > MAXIMUM_PREMIUM:
            logger.warning(
                "The premium of %s (%.2f) points to a NAV in another unit or currency than "
                "its close, so no premium is given.",
                fund,
                premium,
            )
            return float("nan")
        return premium

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

        - Tracking Difference: the annualized return of the fund minus that of the Benchmark
          over the period, as issuers report it. An index fund trails its index by roughly its
          expense ratio; less points to securities lending income, more to a costly or loose
          replication.
        - Tracking Error: the annualized volatility of the per-period return difference, how
          far the fund strays from the index from period to period.
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
        | Tracking Difference | 0.0008 | 0.0007 | -0      |
        | Tracking Error      | 0.0011 | 0.0018 |  0.0021 |
        | Correlation         | 1      | 0.9999 |  0.9999 |
        | Beta                | 0.9994 | 1.0014 |  0.9986 |
        | R Squared           | 0.9999 | 0.9999 |  0.9998 |
        """
        if period not in PERIODS_PER_YEAR:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly or yearly."
            )
        full = self._historical()["Adj Close"]
        if BENCHMARK_TICKER not in full:
            raise ValueError(
                "Tracking is measured against the Benchmark of the Toolkit."
            )
        prices = full.loc[self._start_date : self._end_date]

        # The first return starts from the last close before the range (the history holds
        # a year before the start date), so the first day, or period, is not lost.
        base = full[full.index < prices.index[0]].iloc[-1:]

        if yearly:
            yearly_prices = prices.groupby(prices.index.asfreq("Y")).last()
            first = (base if not base.empty else prices.iloc[[0]]).set_axis(
                [yearly_prices.index[0] - 1]
            )
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
            base = base.set_axis(base.index.asfreq(frequency))
        returns = (
            pd.concat([base, prices]).pct_change(fill_method=None).iloc[len(base) :]
        )

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
        largest holding and its weight, the weight of the ten largest (shares of the fund), the
        Herfindahl-Hirschman index (the sum of the squared weights, after scaling the weights of
        the holdings to sum to one, so cash and holdings without a weight do not dilute it) and
        the effective number of holdings (one over that index). A fund of 500 holdings whose ten largest make up a third
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
        numeric = self._round(numeric, rounding)
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
        high multiple does not dominate. The coverage is the share of the fund's weight held in
        holdings with a usable, non-zero price-to-earnings ratio; more holdings raise it at the
        cost of more requests.

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
        data = self._historical().loc[self._start_date : self._end_date]
        data = data.iloc[-min(len(data), TRADING_DAYS) :]
        information = self._fund_data_function("information")

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
                # Total returns, so a distribution on its ex-date is not counted as a
                # price move.
                "Price Impact per $100 Million": etf_model.get_price_impact(
                    data["Adj Close"][fund].pct_change(fill_method=None), dollar_volume
                ),
            }
        # A price impact of a fraction of a basis point matters here, so the result is only
        # rounded when asked for.
        return apply_rounding(pd.DataFrame(liquidity), rounding)
