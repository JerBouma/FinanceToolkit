"""Financial Constraints Models Module"""

__docformat__ = "google"

from datetime import datetime

import pandas as pd
import requests

from financetoolkit.economics.sources import bis_model
from financetoolkit.helpers import check_trailing, trailing_sum
from financetoolkit.models import helpers
from financetoolkit.models.constraints import constraints_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import (
    calculate_change_over_absolute_base,
    drop_non_finite,
    shift_periods,
)

# ruff: noqa: E501

logger = get_logger()

# The year Hadlock and Pierce (2010) express the book assets of the size-age index in.
SIZE_AGE_BASE_YEAR = "2004"

# The consumer price index of a fiscal period is that of the month the period ends in; a
# month the source has not published yet takes the latest published month, up to a year
# (and a bit) back.
PRICE_LEVEL_TOLERANCE_DAYS = 400


@check_trailing
@filter_shown_columns
@track_sources
class ConstraintsModels:
    """
    Financial constraint indices, which measure how hard it is for a firm to raise external finance: the Kaplan-Zingales, Whited-Wu and Hadlock-Pierce size-age indices.
    """

    def _income_before_extraordinary_items(self, trailing: int | None) -> pd.DataFrame:
        """
        The income before extraordinary items (Compustat item 18) the indices use: the net
        income from continuing operations, as Compustat's item also leaves out discontinued
        operations (and extraordinary items no longer exist under US GAAP and IFRS), and the
        net income where the statements do not report it. Summed over `trailing` periods.
        """
        income = self._rows(
            self._income_statement, "Net Income from Continuing Operations"
        ).combine_first(self._rows(self._income_statement, "Net Income"))

        return trailing_sum(income, trailing)

    def _depreciation_and_amortization(self, trailing: int | None) -> pd.DataFrame:
        """
        The depreciation and amortization (Compustat item 14) of the cash flow statement,
        which reports it more consistently than the income statement, summed over
        `trailing` periods.
        """
        return trailing_sum(
            self._rows(self._cash_flow_statement, "Depreciation and Amortization"),
            trailing,
        )

    def _profile_row(self, item: str) -> pd.Series:
        """An item of the company profile per ticker, NaN where it is not known."""
        profile = self._profile
        if profile is None or profile.empty or item not in profile.index:
            return pd.Series(float("nan"), index=self._tickers, dtype=object)

        values = profile.loc[item].reindex(self._tickers)

        # FinancialModelingPrep reports an unknown value as an empty string.
        return values.where(values.astype(str).str.strip() != "")

    def _price_level_since_base_year(self) -> pd.DataFrame:
        """
        The US consumer price level of every ticker's fiscal period end relative to the
        2004 average, per ticker (rows) and period (columns), which turns the book assets
        of the size-age index into 2004 dollars: the CPI-U (all items, not seasonally
        adjusted) of the month the period ends in, as the Bank for International
        Settlements publishes it, and for a month it has not published yet the latest
        published month. NaN, with a warning, when the index cannot be retrieved.
        """
        columns = self._balance_sheet_statement.columns
        unavailable = pd.DataFrame(float("nan"), index=self._tickers, columns=columns)
        market_dates = self._market_dates()

        # From a year before the earliest period end (or from the base year when that is
        # earlier still), so periods before 2004 are deflated as well, with the months the
        # tolerance looks back over.
        earliest = market_dates.min().min()
        start_year = int(SIZE_AGE_BASE_YEAR)
        if pd.notna(earliest):
            start_year = min(start_year, earliest.year - 1)

        try:
            prices = bis_model.get_consumer_prices(
                "consumer_price_index",
                f"{start_year}-01-01",
                datetime.today().strftime("%Y-%m-%d"),
            )
        except (requests.RequestException, ValueError) as error:
            logger.warning(
                "The US consumer price index could not be retrieved (%s), so the Size and "
                "the Size-Age Index are NaN.",
                error,
            )
            return unavailable

        if prices.empty or "United States" not in prices.columns:
            logger.warning(
                "The US consumer price index could not be retrieved, so the Size and the "
                "Size-Age Index are NaN."
            )
            return unavailable

        consumer_prices = prices["United States"].dropna()
        base = consumer_prices[
            consumer_prices.index.year == int(SIZE_AGE_BASE_YEAR)
        ].mean()
        price_level = (consumer_prices / base).to_timestamp()

        return helpers.values_on_or_before(
            pd.DataFrame({ticker: price_level for ticker in self._tickers}),
            market_dates,
            tolerance=PRICE_LEVEL_TOLERANCE_DAYS,
        )

    @handle_errors
    def get_kaplan_zingales_index(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculates the Kaplan-Zingales (KZ) index of financial constraints: how hard it is for a
        firm to fund its investments with external finance.

        It is the index as constructed by Lamont, Polk and Saa-Requejo (2001). Kaplan and
        Zingales (1997) read the annual reports of the 49 low-dividend manufacturing firms of
        Fazzari, Hubbard and Petersen (1988), classified every firm-year from not to undoubtedly
        financially constrained and related the classes to accounting ratios with an ordered
        logit. Lamont, Polk and Saa-Requejo use Steven Kaplan's re-estimation of that logit on
        the five ratios available in Compustat as an index.

        The formula is as follows:

            - Cash Flow to Capital = (Income Before Extraordinary Items + Depreciation and
              Amortization) / Lagged Net PP&E
            - Q = (Total Assets + Market Value of Equity - Common Equity - Deferred Taxes) /
              Total Assets
            - Debt to Total Capital = (Long-Term Debt + Short-Term Debt) /
              (Long-Term Debt + Short-Term Debt + Total Shareholder Equity)
            - Dividends to Capital = (Common + Preferred Dividends) / Lagged Net PP&E
            - Cash to Capital = Cash and Short-Term Investments / Lagged Net PP&E
            - KZ = -1.001909 * Cash Flow to Capital + 0.2826389 * Q +
              3.139193 * Debt to Total Capital - 39.3678 * Dividends to Capital -
              1.314759 * Cash to Capital

        The variables follow the appendix of Lamont, Polk and Saa-Requejo (Compustat items):

            - Income Before Extraordinary Items (item 18): the net income from continuing
              operations, or the net income where it is not reported.
            - Depreciation and Amortization (item 14): that of the cash flow statement.
            - Net PP&E (item 8): the net property, plant and equipment at the end of the
              previous period (on quarterly data, at the start of the trailing window).
            - Total Assets (item 6), Common Equity (item 60: the shareholders' equity less the
              preferred stock), Deferred Taxes (item 74: the deferred tax liabilities),
              Long-Term Debt (item 9), Short-Term Debt (item 34: the debt in current
              liabilities), Total Shareholder Equity (item 216: preferred stock included,
              minority interests excluded), Common and Preferred Dividends (items 21 and 19)
              and Cash and Short-Term Investments (item 1).
            - Market Value of Equity: the close at the end of the fiscal period times the
              period's weighted average shares (CRSP's December market equity in the paper).

        The Kaplan-Zingales index can be interpreted as follows:

            - A higher index means a more constrained firm. Lamont, Polk and Saa-Requejo call
              the top third of firms ranked on the index each year constrained and the bottom
              third unconstrained; the index has no absolute cut-off.
            - Whited and Wu (2006, Table 2) report quartile means of 0.25, 0.77, 1.15 and 1.81
              on quarterly Compustat data for 1975-2001, with the cash flow, the dividends, the
              cash and the long-term debt scaled by total assets instead of capital and total
              capital, so the levels of this index (ratios to net PP&E) are not comparable to
              theirs; compare firms by their rank.

        Also known as: KZ index, Kaplan-Zingales, financing constraints, external finance.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The number of periods the cash flow and the
                dividends are summed over, with the capital they are divided by taken that many
                periods earlier; the other balances are those at the end of the period. The
                window counts calendar periods, so a period without a report makes it NaN.
                Defaults to None, which is 4 on quarterly data (a trailing twelve months), as
                the index was estimated on yearly statements.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Kaplan-Zingales index and its five ratios per ticker and period.

        Notes:
            - The first period is NaN because the capital is that of the period before.
            - Kaplan and Zingales (1997) estimated the logit on low-dividend manufacturing firms
              with growing sales; Lamont, Polk and Saa-Requejo apply it to all manufacturing
              firms with growing real sales. For other industries (financial firms in particular,
              whose PP&E is small) the ratios to capital become large and the index is only
              indicative.
            - FinancialModelingPrep's net PP&E of some companies includes the right-of-use assets
              of operating leases, which Compustat's item 8 leaves out, and for some only in the
              fiscal year-end statements (Apple's, by about $10 billion, so its quarterly index
              jumps in every fiscal fourth quarter). The ratios to capital are lower where it does.
            - The index has been criticized as a measure of constraints: Whited and Wu (2006) and
              Hadlock and Pierce (2010) find that it classifies firms with characteristics of
              unconstrained firms as constrained, which get_whited_wu_index and
              get_size_age_index address.

        References:
        - Lamont, Owen, Christopher Polk and Jesus Saa-Requejo. "Financial Constraints and Stock
        Returns." The Review of Financial Studies, Vol. 14, No. 2, 2001, pp. 529-554.
        - Kaplan, Steven N. and Luigi Zingales. "Do Investment-Cash Flow Sensitivities Provide Useful
        Measures of Financing Constraints?" The Quarterly Journal of Economics, Vol. 112, No. 1,
        1997, pp. 169-215.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "F", "SMP"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2020-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_kaplan_zingales_index().xs("Kaplan-Zingales Index", level=1)
        ```

        Which returns:

        |      |     2020 |     2021 |     2022 |     2023 |    2024 |    2025 |
        |:-----|---------:|---------:|---------:|---------:|--------:|--------:|
        | AAPL | -15.9535 | -12.4285 | -10.9185 | -10.4628 | -9.5516 | -9.3348 |
        | F    |   1.4737 |   1.0274 |   0.4525 |  -1.6085 | -0.2481 |  0.3139 |
        | SMP  |  -4.1242 |  -7.6728 |  -6.1412 |  -6.2198 | -3.3562 | -2.8463 |
        """
        # The index was estimated on yearly statements, so quarterly data is summed into a
        # year (a trailing twelve months) unless a trailing period is given.
        if trailing is None and self._quarterly:
            trailing = 4
        balance = self._balance_sheet_statement

        # The capital at the start of the window the flows cover: a year back on yearly
        # data, `trailing` quarters back on quarterly data.
        lagged_capital = shift_periods(
            self._rows(balance, "Property, Plant and Equipment"), trailing or 1
        )
        total_assets = self._rows(balance, "Total Assets")
        stockholders_equity = self._rows(balance, "Total Shareholder Equity")
        common_equity = stockholders_equity - self._rows(
            balance, "Preferred Stock"
        ).fillna(0)
        market_value_of_equity = self._period_prices() * self._rows(
            self._income_statement, "Weighted Average Shares"
        )
        dividends = trailing_sum(
            self._rows(self._cash_flow_statement, "Dividends Paid"), trailing
        )

        kaplan_zingales = {
            "Cash Flow to Capital": constraints_model.get_cash_flow_to_capital(
                income_before_extraordinary_items=self._income_before_extraordinary_items(
                    trailing
                ),
                depreciation_and_amortization=self._depreciation_and_amortization(
                    trailing
                ),
                lagged_capital=lagged_capital,
            ),
            "Q": constraints_model.get_kaplan_zingales_q(
                total_assets=total_assets,
                market_value_of_equity=market_value_of_equity,
                common_equity=common_equity,
                deferred_taxes=self._rows(balance, "Deferred Tax Liabilities").fillna(
                    0
                ),
            ),
            "Debt to Total Capital": constraints_model.get_debt_to_total_capital(
                short_term_debt=self._rows(balance, "Short Term Debt"),
                long_term_debt=self._rows(balance, "Long Term Debt"),
                stockholders_equity=stockholders_equity,
            ),
            "Dividends to Capital": constraints_model.get_dividends_to_capital(
                dividends=dividends,
                lagged_capital=lagged_capital,
            ),
            "Cash to Capital": constraints_model.get_cash_to_capital(
                cash_and_short_term_investments=self._rows(
                    balance, "Cash and Short Term Investments"
                ),
                lagged_capital=lagged_capital,
            ),
        }
        kaplan_zingales["Kaplan-Zingales Index"] = (
            constraints_model.get_kaplan_zingales_index(
                cash_flow_to_capital=kaplan_zingales["Cash Flow to Capital"],
                q=kaplan_zingales["Q"],
                debt_to_total_capital=kaplan_zingales["Debt to Total Capital"],
                dividends_to_capital=kaplan_zingales["Dividends to Capital"],
                cash_to_capital=kaplan_zingales["Cash to Capital"],
            )
        )

        return self._finalize_components(
            kaplan_zingales, rounding, growth, lag, standardize
        )

    @handle_errors
    def get_whited_wu_index(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculates the Whited-Wu (WW) index of financial constraints: how hard it is for a firm
        to fund its investments with external finance.

        It is the shadow value of external funds that Whited and Wu (2006) estimate from the
        Euler equation of a firm's investment on quarterly Compustat data for 1975-2001. Where
        the Kaplan-Zingales index comes from a classification of a few firms, this index is the
        cost of external finance implied by how firms invest, and leaves out Tobin's Q, which
        measures investment opportunities with much error. Sales growth and the industry's sales
        growth take its place: a constrained firm is one with good investment opportunities (a
        growing industry) it cannot fund (it grows less).

        The formula is as follows:

            - CF = (Income Before Extraordinary Items + Depreciation and Amortization) /
              Total Assets
            - DIVPOS = 1 if the firm paid cash dividends (common or preferred) else 0
            - TLTD = Long-Term Debt / Total Assets
            - LNTA = ln(Total Assets in millions)
            - SG = (Revenue - Previous Revenue) / |Previous Revenue|
            - ISG = the average SG of the tickers in the firm's industry
            - WW = -0.091 * CF - 0.062 * DIVPOS + 0.021 * TLTD - 0.044 * LNTA +
              0.102 * ISG - 0.035 * SG

        The coefficients are those of equation 13 (column 4 of Table 1) of Whited and Wu (2006).
        The variables are those of Hennessy and Whited (2007) and Farre-Mensa and Ljungqvist (2016):

            - Income Before Extraordinary Items: the net income from continuing operations, or the
              net income where it is not reported.
            - Depreciation and Amortization: that of the cash flow statement.
            - Total Assets and Long-Term Debt: at the end of the period.
            - The sales growth compares with the period before on yearly data and with the same
              quarter a year earlier on quarterly data, which leaves out the seasons.

        The industry sales growth is the average sales growth (equally weighted, the firm itself
        included) of the tickers in the Toolkit that share the firm's industry in their company
        profile (FinancialModelingPrep's industry, close to the three-digit SIC industries of
        Whited and Wu). A firm without industry peers in the Toolkit, or without a profile, uses
        its own sales growth: add companies of its industry to the tickers for an industry figure.

        The Whited-Wu index can be interpreted as follows:

            - A higher index means a more constrained firm. As with the Kaplan-Zingales index, the
              literature calls the top third of firms constrained and the bottom third
              unconstrained, each year; the index has no absolute cut-off.
            - The size term dominates the level: -0.044 * ln(assets in millions) is about -0.20
              for a firm with $100 million of assets and -0.51 for one with $100 billion, so the
              index of most listed firms lies between about -0.6 and -0.1 on yearly data.

        Also known as: WW index, Whited-Wu, financing constraints, shadow cost of external funds.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The number of periods the cash flow, the dividends
                and the revenue are summed over; the balances are those at the end of the period.
                The window counts calendar periods, so a period without a report makes it NaN.
                Defaults to None, which is a single period: the quarter on quarterly data, as the
                index was estimated on quarterly statements.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Whited-Wu index and its six components per ticker and period.

        Notes:
            - The first period (the first four on quarterly data) is NaN because the sales growth
              needs a period to compare with.
            - On yearly statements the cash flow is a year's, four times a quarter's, so the CF
              term weighs more than in the quarterly estimation; this is how the index is
              commonly applied to yearly Compustat data (Hennessy and Whited 2007).
            - Whited and Wu's Tables 2 and 3 report quartile means of a "structural index" of
              0.55 to 0.80, a level equation 13 does not reproduce (it has no constant, and the
              Euler equation is estimated in first differences, which leaves the level
              unidentified). Compare firms by their rank on the index, as the literature does.
            - The total assets are taken in the currency of the statements, so for a company
              reporting in a currency other than the US dollar the LNTA term (and the index) shifts
              by -0.044 times the log of the exchange rate; compare such companies within their
              currency, or retrieve the statements converted to US dollars.

        References:
        - Whited, Toni M. and Guojun Wu. "Financial Constraints Risk." The Review of Financial Studies,
        Vol. 19, No. 2, 2006, pp. 531-559.
        - Hennessy, Christopher A. and Toni M. Whited. "How Costly Is External Financing? Evidence from
        a Structural Estimation." The Journal of Finance, Vol. 62, No. 4, 2007, pp. 1705-1745.
        - Farre-Mensa, Joan and Alexander Ljungqvist. "Do Measures of Financial Constraints Measure
        Financial Constraints?" The Review of Financial Studies, Vol. 29, No. 2, 2016, pp. 271-308.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["F", "GM", "LEA", "SMP"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2020-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_whited_wu_index().xs("Whited-Wu Index", level=1)
        ```

        Which returns:

        |     |    2020 |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:----|--------:|--------:|--------:|--------:|--------:|--------:|
        | F   | -0.6137 | -0.6083 | -0.5897 | -0.602  | -0.6054 | -0.6073 |
        | GM  | -0.6182 | -0.6054 | -0.6006 | -0.6063 | -0.609  | -0.6122 |
        | LEA | -0.4838 | -0.4732 | -0.4797 | -0.4865 | -0.4812 | -0.4755 |
        | SMP | -0.3814 | -0.3745 | -0.375  | -0.3751 | -0.3894 | -0.3925 |
        """
        balance = self._balance_sheet_statement
        total_assets = self._rows(balance, "Total Assets")
        dividends = trailing_sum(
            self._rows(self._cash_flow_statement, "Dividends Paid"), trailing
        )
        revenue = trailing_sum(self._rows(self._income_statement, "Revenue"), trailing)

        # Year over year: the quarter a year earlier on quarterly data, so the seasons of a
        # company's sales do not show up as growth.
        sales_growth = drop_non_finite(
            calculate_change_over_absolute_base(revenue, 4 if self._quarterly else 1)
        )

        # The basket's Total has no industry of its own and is no peer of its companies.
        industries = self._profile_row("Industry").reindex(self._peer_tickers)

        whited_wu = {
            "Cash Flow to Total Assets": constraints_model.get_cash_flow_to_total_assets(
                income_before_extraordinary_items=self._income_before_extraordinary_items(
                    trailing
                ),
                depreciation_and_amortization=self._depreciation_and_amortization(
                    trailing
                ),
                total_assets=total_assets,
            ),
            "Dividend Payer": constraints_model.get_dividend_payer_indicator(
                dividends=dividends
            ),
            "Long-Term Debt to Total Assets": constraints_model.get_long_term_debt_to_total_assets(
                long_term_debt=self._rows(balance, "Long Term Debt"),
                total_assets=total_assets,
            ),
            "Log of Total Assets": constraints_model.get_log_of_total_assets_in_millions(
                total_assets=total_assets
            ),
            "Industry Sales Growth": constraints_model.get_industry_sales_growth(
                sales_growth=sales_growth, industries=industries
            ),
            "Sales Growth": sales_growth,
        }
        whited_wu["Whited-Wu Index"] = constraints_model.get_whited_wu_index(
            cash_flow_to_total_assets=whited_wu["Cash Flow to Total Assets"],
            dividend_payer_indicator=whited_wu["Dividend Payer"],
            long_term_debt_to_total_assets=whited_wu["Long-Term Debt to Total Assets"],
            log_of_total_assets=whited_wu["Log of Total Assets"],
            industry_sales_growth=whited_wu["Industry Sales Growth"],
            sales_growth=whited_wu["Sales Growth"],
        )

        return self._finalize_components(whited_wu, rounding, growth, lag, standardize)

    @handle_errors
    def get_size_age_index(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculates the size-age (SA) index of financial constraints of Hadlock and Pierce
        (2010).

        They coded the financial constraints of a random sample of firms from the text of their
        10-K filings (1995-2004) and found that, of the usual measures, only size and age
        predict the constraints well: the Kaplan-Zingales ratios do not, once size and age are
        known. The index is their ordered logit on size and age alone, so it needs no market
        data and has few of the accounting choices that move the other indices.

        The formula is as follows:

            - Size = ln(min(Total Assets in millions of 2004 dollars, 4,500))
            - Age = min(Years Listed, 37)
            - SA = -0.737 * Size + 0.043 * Size^2 - 0.040 * Age

        The variables are as follows:

            - Total Assets in 2004 dollars: the total assets at the end of the fiscal period
              divided by the US consumer price index (CPI-U, all items, from the Bank for
              International Settlements) of the month the period ends in, relative to its 2004
              average; a month not published yet uses the latest published month.
            - Years Listed: the year of the end of the fiscal period minus the year of the
              company's IPO date in its profile (FinancialModelingPrep), the count of years a
              firm has a stock price in Compustat in Hadlock and Pierce. A period before the IPO
              is NaN.
            - The caps are those of Hadlock and Pierce: firms above $4.5 billion of assets or 37
              years listed take those values, beyond which size and age hardly add to the
              prediction.

        The size-age index can be interpreted as follows:

            - A higher (less negative) index means a more constrained firm: a small, young firm.
            - Its lowest value is -4.64, for a firm at both caps ($4.5 billion of 2004 assets and
              37 years listed). A new firm with $1 million of 2004 assets has 0, and a smaller one
              above 0 (its Size is negative). A firm listed 10 years with $100 million of 2004
              assets has -2.88.
            - As with the other indices, the literature calls the top third of firms constrained
              and the bottom third unconstrained, each year; the index has no absolute cut-off.

        Also known as: SA index, Hadlock-Pierce index, HP index, financing constraints.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The size-age index, its Size and Age per ticker and period.

        Notes:
            - The age needs the IPO date of the company profile, which requires an API key; without
              it, or for a company whose profile has no IPO date, the Age and the index are NaN (a
              warning names the tickers). The IPO date of a company that listed again (after a
              spin-off, merger or bankruptcy) is that of the latest listing, so its age counts
              from then.
            - The total assets are taken in the currency of the statements and deflated with the
              US price level, as in Hadlock and Pierce's US sample, so for a company reporting in
              another currency the Size is only indicative unless the statements are converted to
              US dollars.

        References:
        - Hadlock, Charles J. and Joshua R. Pierce. "New Evidence on Measuring Financial
        Constraints: Moving Beyond the KZ Index." The Review of Financial Studies, Vol. 23, No. 5,
        2010, pp. 1909-1940.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "F", "SMP", "ABNB"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2020-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_size_age_index().xs("Size-Age Index", level=1)
        ```

        Which returns:

        |      |    2020 |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:-----|--------:|--------:|--------:|--------:|--------:|--------:|
        | AAPL | -4.6369 | -4.6369 | -4.6369 | -4.6369 | -4.6369 | -4.6369 |
        | ABNB | -3.1569 | -3.1969 | -3.2369 | -3.2769 | -3.3169 | -3.3569 |
        | F    | -4.6369 | -4.6369 | -4.6369 | -4.6369 | -4.6369 | -4.6369 |
        | SMP  | -4.4611 | -4.4875 | -4.4849 | -4.4844 | -4.5306 | -4.5398 |
        """
        total_assets = self._rows(self._balance_sheet_statement, "Total Assets")
        market_dates = self._market_dates()

        listing_dates = pd.to_datetime(
            self._profile_row("IPO Date"), errors="coerce"
        ).reindex(self._tickers)
        # The basket's Total has no IPO date of its own.
        if without_date := (
            listing_dates[listing_dates.isna()]
            .index.intersection(self._peer_tickers)
            .tolist()
        ):
            logger.warning(
                "No IPO date is available for %s, so their Age and Size-Age Index are NaN.",
                ", ".join(without_date),
            )

        years_listed = market_dates.apply(lambda dates: dates.dt.year).sub(
            listing_dates.dt.year, axis=0
        )

        size_age = {
            "Size": constraints_model.get_size_age_size(
                total_assets=total_assets,
                price_level=self._price_level_since_base_year(),
            ),
            "Age": constraints_model.get_size_age_age(years_listed=years_listed),
        }
        size_age["Size-Age Index"] = constraints_model.get_size_age_index(
            size=size_age["Size"], age=size_age["Age"]
        )

        return self._finalize_components(size_age, rounding, growth, lag, standardize)
