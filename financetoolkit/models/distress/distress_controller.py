"""Distress Models Module"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.helpers import (
    add_score_zone,
    get_goodwill_and_intangible_assets,
)
from financetoolkit.models import helpers
from financetoolkit.models.base.base_controller import ModelsBase
from financetoolkit.models.distress import (
    altman_model,
    chs_model,
    credit_model,
    fulmer_model,
    grover_model,
    ohlson_model,
    springate_model,
    taffler_model,
    zmijewski_model,
)
from financetoolkit.models.valuation import damodaran_model
from financetoolkit.ratios import liquidity_model, valuation_model
from financetoolkit.utilities.dataframe_model import concat_frames, filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import (
    apply_rounding,
    finalize_dataset,
)

# ruff: noqa: E501

logger = get_logger()


@filter_shown_columns
@track_sources
class DistressModels(ModelsBase):
    """
    Bankruptcy and credit risk models: the Altman, Ohlson, Zmijewski, Springate, Grover, Fulmer, Taffler and Campbell-Hilscher-Szilagyi scores, Merton's distance to default and the synthetic credit rating.
    """

    @handle_errors
    def get_altman_z_score(
        self,
        diluted: bool = True,
        model: str = "original",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
        zone: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculates the Altman Z-Score, a financial metric used to predict the likelihood of a company going bankrupt.
        The Altman Z-Score is calculated using several financial ratios, including working capital to total assets,
        retained earnings to total assets, earnings before interest and taxes (EBIT) to total assets, equity to
        book value of total liabilities, and (for manufacturers) sales to total assets. Altman estimated four
        versions, chosen with `model`: the original Z-Score for listed manufacturers, the Z'-Score for private
        manufacturers, the Z''-Score for non-manufacturers and the Z''-EM score for emerging market companies.

        The formula is as follows:

            - Working Capital to Total Assets (X1) = Working Capital / Total Assets
            - Retained Earnings to Total Assets (X2) = Retained Earnings / Total Assets
            - EBIT to Total Assets (X3) = EBIT / Total Assets
            - Market Value to Total Liabilities (X4, "original") = Market Value of Equity / Total Liabilities
            - Book Value to Total Liabilities (X4, the other models) = Total Shareholder Equity / Total Liabilities
            - Sales to Total Assets (X5, "original" and "private") = Sales / Total Assets
            - Original (Altman 1968): Z = 1.2 X1 + 1.4 X2 + 3.3 X3 + 0.6 X4 + 1.0 X5
            - Private (Altman 1983): Z' = 0.717 X1 + 0.847 X2 + 3.107 X3 + 0.420 X4 + 0.998 X5
            - Non-manufacturing (Altman, Hartzell and Peck 1995): Z'' = 6.56 X1 + 3.26 X2 + 6.72 X3 + 1.05 X4
            - Emerging (Altman 2005): Z''-EM = 3.25 + 6.56 X1 + 3.26 X2 + 6.72 X3 + 1.05 X4

        The Altman Z-Score can be interpreted with the zones of each model (distress below the lower cut-off,
        grey zone in between, safe above the upper one):

            - Original: distress below 1.81, safe above 2.99.
            - Private: distress below 1.23, safe above 2.90.
            - Non-manufacturing: distress below 1.10, safe above 2.60.
            - Emerging: distress below 4.35, safe above 5.85 (the Z'' zones plus the constant of 3.25, which
            Altman set so that a score of zero or below matches a bond rated D).

        The original Z-Score was estimated on US listed manufacturers, and its thresholds are not meant for
        banks, insurers or most service companies; the Z''-Score drops the sales term, as asset turnover
        differs most between industries, so that it fits non-manufacturers, and none of the models is meant
        for banks or insurers. Share buybacks charged to retained earnings and accumulated losses both lower
        the retained earnings term, so a company that returns most of its earnings scores lower without being
        closer to distress, which is why the score is best read as a trend and next to the equity term.

        Also known as: Altman Z-score, Z'-Score, Z''-Score, Z''-EM, emerging market score, bankruptcy
        prediction, financial distress score.

        Args:
            diluted (bool, optional): Whether to use diluted shares outstanding in the market value of equity of
                the "original" model. Defaults to True.
            model (str, optional): The model to use: "original" (listed manufacturers, market value of
                equity), "private" (Z', book value of equity), "non-manufacturing" (Z'', book value of equity,
                no sales term) or "emerging" (Z''-EM, Z'' plus 3.25). Defaults to "original".
            rounding (int, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The trailing period to use for the calculation. Defaults to
                None, which is 4 on quarterly data (a trailing twelve months), as the score was estimated on
                yearly statements.
            zone (bool, optional): Whether to add a "Zone" row per ticker: "Distress" below the lower cut-off of
                the chosen model, "Grey" up to the upper one and "Safe" from it (see above). Not with growth or
                standardize. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: DataFrame containing the Altman Z-Score of the chosen model and its components.

        Notes:
            - The Altman Z-Score is a financial metric used to predict the likelihood of a company going bankrupt.
            - The original Z-Score is most effective when used to analyze listed manufacturing companies with
            assets of $1 million or more. The Z'-Score uses the book value of equity, so it needs no share
            price, and the Z''-Score and Z''-EM score leave out sales to total assets.
            - The book value of equity is the Total Shareholder Equity, which excludes minority interests.
            - EBIT is the reported Operating Income, as in the EBIT based ratios of the Ratios module.

        References:
            - Altman, E. I. (1968). Financial Ratios, Discriminant Analysis and the Prediction of Corporate
            Bankruptcy. The Journal of Finance, 23(4), 589-609.
            - Altman, E. I. (1983). Corporate Financial Distress: A Complete Guide to Predicting, Avoiding, and
            Dealing with Bankruptcy. John Wiley & Sons.
            - Altman, E. I. (2000). Predicting Financial Distress of Companies: Revisiting the Z-Score and ZETA
            Models. Working paper, New York University.
            - Altman, E. I., Hartzell, J. and Peck, M. (1995). Emerging Markets Corporate Bonds: A Scoring System.
            Salomon Brothers.
            - Altman, E. I. (2005). An Emerging Market Credit Scoring System for Corporate Bonds. Emerging
            Markets Review, 6(4), 311-323.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["MCD", "NFLX", "UBER"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_altman_z_score(model="non-manufacturing").xs("Altman Z-Score", level=1)
        ```

        Which returns:

        |      |    2021 |    2022 |    2023 |    2024 |   2025 |
        |:-----|--------:|--------:|--------:|--------:|-------:|
        | MCD  |  5.0735 |  5.1966 |  5.1302 |  5.3948 | 5.1966 |
        | NFLX |  2.379  |  2.8961 |  3.3806 |  4.3943 | 5.2949 |
        | UBER | -2.0375 | -3.3028 | -1.6181 | -0.0662 | 1.0641 |
        """
        # The score was estimated on yearly statements, so quarterly data is summed into a
        # year (a trailing twelve months) unless a trailing period is given.
        if trailing is None and self._quarterly:
            trailing = 4
        if model not in altman_model.ALTMAN_COEFFICIENTS:
            raise ValueError(
                f"Invalid model {model!r}. Please choose from: "
                f"{', '.join(altman_model.ALTMAN_COEFFICIENTS)}."
            )

        altman_z_score = {}

        def balance_item(item: str) -> pd.DataFrame:
            """A balance sheet item, averaged over the trailing periods when given."""
            values = self._balance_sheet_statement.loc[:, item, :]
            return values.T.rolling(trailing).mean().T if trailing else values

        total_assets = balance_item("Total Assets")
        total_liabilities = balance_item("Total Liabilities")

        working_capital = liquidity_model.get_working_capital(
            balance_item("Total Current Assets"),
            balance_item("Total Current Liabilities"),
        )

        altman_z_score["Working Capital to Total Assets"] = (
            altman_model.get_working_capital_to_total_assets_ratio(
                working_capital=working_capital,
                total_assets=total_assets,
            )
        )

        altman_z_score["Retained Earnings to Total Assets"] = (
            altman_model.get_retained_earnings_to_total_assets_ratio(
                retained_earnings=balance_item("Retained Earnings"),
                total_assets=total_assets,
            )
        )

        altman_z_score["EBIT to Total Assets"] = (
            altman_model.get_earnings_before_interest_and_taxes_to_total_assets_ratio(
                ebit=self._earnings_before_interest_and_taxes(trailing),
                total_assets=total_assets,
            )
        )

        if model == "original":
            years = self._balance_sheet_statement.columns
            begin, end = str(years[0]), str(years[-1])

            share_prices = (
                self._historical_data["quarterly" if self._quarterly else "yearly"]
                .loc[begin:end, "Close"][self._tickers]
                .T
            )

            average_shares = (
                self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
                if diluted
                else self._income_statement.loc[:, "Weighted Average Shares", :]
            )

            market_cap = valuation_model.get_market_cap(
                share_price=share_prices,
                total_shares_outstanding=(
                    average_shares.T.rolling(trailing).mean().T
                    if trailing
                    else average_shares
                ),
            )

            equity_ratio_name = "Market Value to Total Liabilities"
            altman_z_score[equity_ratio_name] = (
                altman_model.get_market_value_of_equity_to_book_value_of_total_liabilities_ratio(
                    market_value_of_equity=market_cap,
                    total_liabilities=total_liabilities,
                )
            )
        else:
            equity_ratio_name = "Book Value to Total Liabilities"
            altman_z_score[equity_ratio_name] = (
                altman_model.get_book_value_of_equity_to_book_value_of_total_liabilities_ratio(
                    book_value_of_equity=balance_item("Total Shareholder Equity"),
                    total_liabilities=total_liabilities,
                )
            )

        # The Z''-Score and Z''-EM score have no sales term.
        if model in ("original", "private"):
            revenue = self._income_statement.loc[:, "Revenue", :]
            altman_z_score["Sales to Total Assets"] = (
                altman_model.get_sales_to_total_assets_ratio(
                    sales=revenue.T.rolling(trailing).sum().T if trailing else revenue,
                    total_assets=total_assets,
                )
            )

        altman_z_score["Altman Z-Score"] = altman_model.get_altman_z_score(
            working_capital_to_total_assets_ratio=altman_z_score[
                "Working Capital to Total Assets"
            ],
            retained_earnings_to_total_assets_ratio=altman_z_score[
                "Retained Earnings to Total Assets"
            ],
            earnings_before_interest_and_taxes_to_total_assets_ratio=altman_z_score[
                "EBIT to Total Assets"
            ],
            equity_to_total_liabilities_ratio=altman_z_score[equity_ratio_name],
            sales_to_total_assets_ratio=altman_z_score.get("Sales to Total Assets"),
            model=model,
        )

        altman_results = (
            concat_frames(altman_z_score)
            .dropna(axis=1, how="all")
            .swaplevel(0, 1)
            .reindex(self._tickers, level=0)
        )

        # The models were estimated on non-financial companies. A company without any cost
        # of goods sold, such as a bank or an insurer, has no working capital or sales in
        # their sense, so its score (e.g. "distress") says nothing; the result says so.
        cost_of_goods_sold = self._income_statement.loc[:, "Cost of Goods Sold", :]

        if not_applicable := [
            ticker
            for ticker in cost_of_goods_sold.index
            # Most periods without, as a stray figure in one year does not make one.
            if (cost_of_goods_sold.loc[ticker].dropna() == 0).mean()
            >= 0.5  # noqa: PLR2004
        ]:
            logger.warning(
                "The Altman Z-Score does not apply to companies that report no cost of "
                "goods sold, as banks and insurers do: the models were estimated on "
                "non-financial companies, so it is no measure of financial distress for %s.",
                ", ".join(not_applicable),
            )

        altman_results = finalize_dataset(
            dataset=altman_results,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="columns",
        )

        if zone and not growth and not standardize:
            altman_results = add_score_zone(
                altman_results,
                "Altman Z-Score",
                list(altman_model.ALTMAN_CUT_OFFS[model]),
                ["Distress", "Grey", "Safe"],
            )

        return altman_results

    @handle_errors
    def get_ohlson_o_score(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculates the Ohlson O-Score, a financial metric used to predict the likelihood of a company
        going bankrupt. Unlike the Altman Z-Score, which is built with multiple discriminant analysis,
        the O-Score's coefficients come from a fitted logistic regression (logit) model. This is why
        the two models are usually reported side by side rather than one being treated as a replacement
        for the other: the Z-Score is only meaningful compared against Altman's empirically derived
        threshold bands, while the O-Score is directly interpretable as a probability of bankruptcy
        once passed through the logistic transform.

        The formula is as follows:

            - SIZE = ln(Total Assets)
            - TLTA = Total Liabilities / Total Assets
            - WCTA = Working Capital / Total Assets
            - CLCA = Current Liabilities / Current Assets
            - OENEG = 1 if Total Liabilities > Total Assets else 0
            - NITA = Net Income / Total Assets
            - FUTL = Operating Cash Flow / Total Liabilities
            - INTWO = 1 if Net Income was negative for the last two years else 0
            - CHIN = (Net Income (t) - Net Income (t-1)) / (|Net Income (t)| + |Net Income (t-1)|)
            - O-Score = -1.32 - 0.407 * SIZE + 6.03 * TLTA - 1.43 * WCTA + 0.0757 * CLCA
            - 1.72 * OENEG - 2.37 * NITA - 1.83 * FUTL + 0.285 * INTWO - 0.521 * CHIN
            - Bankruptcy Probability = 1 / (1 + e^(-O-Score))

        The Ohlson O-Score can be interpreted as follows:

            - Ohlson's (1980) original cutoff is a bankruptcy probability of 0.038 (3.8%), the
            threshold that minimized the sum of Type I and Type II misclassification errors on his
            sample. It is deliberately far below the naive 0.50 midpoint because bankruptcy is a
            rare event. Equivalently, in raw O-Score terms the cutoff sits at ln(0.038 / 0.962),
            i.e. approximately -3.23. Since this implementation omits Ohlson's price-level
            deflator, its scores sit about 2.6 below Ohlson's scale and this cutoff flags far too few
            companies (see the Notes).
            - A higher probability indicates a higher likelihood of bankruptcy.

        Also known as: Ohlson O-Score, bankruptcy prediction, financial distress score.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The trailing period to use for the calculation. Defaults to
                None, which is 4 on quarterly data (a trailing twelve months), as the score was estimated on
                yearly statements.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: DataFrame containing the Ohlson O-Score, its bankruptcy probability, and its
            components.

        Notes:
            - Ohlson's (1980) original SIZE term deflates Total Assets by the US GNP price-level index
            (rebased to 1968), a US-specific 1970s macro series that isn't available in this toolkit and
            isn't meaningful outside the original US sample. This implementation omits that deflator and
            uses ln(Total Assets) in nominal terms instead. This lowers every O-Score by roughly 0.407 *
            ln(price index); with the US price level of the 2020s at roughly six times its 1968 level (an
            index of about 600 to 650), that is about 2.6. The resulting probabilities are therefore much
            lower than in Ohlson's study and the 0.038 cutoff (-3.23) flags far too few companies; it
            corresponds to roughly -5.8 on this scale. The *ranking* of companies relative to one another
            within a given currency and period is unaffected.
            - Ohlson's original FUTL term uses "Funds from Operations", an accounting-flow concept that
            predates standardized cash flow statements. This implementation approximates Funds from
            Operations with Operating Cash Flow, the standard simplification used in modern
            reproductions of the O-Score.
            - The Beneish M-Score, Altman Z-Score, Piotroski F-Score and Ohlson O-Score are natural
            companions, using the same normalized financial statements as their input.
            - As with the other bankruptcy and distress models, this is a probabilistic, not a
            definitive, indicator and should be combined with further fundamental analysis.

        References:
        - Ohlson, James A. "Financial Ratios and the Probabilistic Prediction of Bankruptcy." Journal of
        Accounting Research, Vol. 18, No. 1, 1980, pp. 109-131.

        Note that the first year is NaN because the Change in Net Income and Negative Income
        Indicator components require a prior year to compare against. On quarterly data both
        compare with the same quarter a year earlier.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["CCL", "RCL", "NCLH"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_ohlson_o_score().xs("Ohlson O-Score", level=1)
        ```

        Which returns:

        |      |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:-----|--------:|--------:|--------:|--------:|--------:|
        | CCL  | -5.7604 | -5.4419 | -6.2136 | -6.8358 | -6.78   |
        | NCLH | -4.4192 | -4.1038 | -5.0639 | -5.269  | -4.9984 |
        | RCL  | -4.9997 | -5.0496 | -6.2194 | -6.3642 | -6.6839 |
        """
        # The score was estimated on yearly statements, so quarterly data is summed into a
        # year (a trailing twelve months) unless a trailing period is given.
        if trailing is None and self._quarterly:
            trailing = 4
        ohlson_o_score = {}

        current_assets = (
            self._balance_sheet_statement.loc[:, "Total Current Assets", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Current Assets", :]
        )
        current_liabilities = (
            self._balance_sheet_statement.loc[:, "Total Current Liabilities", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Current Liabilities", :]
        )
        total_assets = (
            self._balance_sheet_statement.loc[:, "Total Assets", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Assets", :]
        )
        total_liabilities = (
            self._balance_sheet_statement.loc[:, "Total Liabilities", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Liabilities", :]
        )
        net_income = (
            self._income_statement.loc[:, "Net Income", :].T.rolling(trailing).sum().T
            if trailing
            else self._income_statement.loc[:, "Net Income", :]
        )
        operating_cash_flow = (
            self._cash_flow_statement.loc[:, "Operating Cash Flow", :]
            .T.rolling(trailing)
            .sum()
            .T
            if trailing
            else self._cash_flow_statement.loc[:, "Operating Cash Flow", :]
        )

        working_capital = liquidity_model.get_working_capital(
            current_assets,
            current_liabilities,
        )

        ohlson_o_score["Log of Total Assets"] = ohlson_model.get_log_of_total_assets(
            total_assets=total_assets,
        )

        ohlson_o_score["Total Liabilities to Total Assets"] = (
            ohlson_model.get_total_liabilities_to_total_assets_ratio(
                total_liabilities=total_liabilities,
                total_assets=total_assets,
            )
        )

        ohlson_o_score["Working Capital to Total Assets"] = (
            ohlson_model.get_working_capital_to_total_assets_ratio(
                working_capital=working_capital,
                total_assets=total_assets,
            )
        )

        ohlson_o_score["Current Liabilities to Current Assets"] = (
            ohlson_model.get_current_liabilities_to_current_assets_ratio(
                current_liabilities=current_liabilities,
                current_assets=current_assets,
            )
        )

        ohlson_o_score["Negative Equity Indicator"] = (
            ohlson_model.get_negative_equity_indicator(
                total_liabilities=total_liabilities,
                total_assets=total_assets,
            )
        )

        ohlson_o_score["Net Income to Total Assets"] = (
            ohlson_model.get_net_income_to_total_assets_ratio(
                net_income=net_income,
                total_assets=total_assets,
            )
        )

        ohlson_o_score["Funds from Operations to Total Liabilities"] = (
            ohlson_model.get_funds_from_operations_to_total_liabilities_ratio(
                funds_from_operations=operating_cash_flow,
                total_liabilities=total_liabilities,
            )
        )

        # Ohlson defines both terms on the last two years, so quarterly data is compared with the
        # same quarter a year earlier, as Piotroski and Beneish do.
        comparison_lag = 4 if self._quarterly else 1

        ohlson_o_score["Negative Income Indicator"] = (
            ohlson_model.get_negative_income_indicator(
                net_income=net_income,
                lag=comparison_lag,
            )
        )

        ohlson_o_score["Change in Net Income"] = (
            ohlson_model.get_change_in_net_income_ratio(
                net_income=net_income,
                lag=comparison_lag,
            )
        )

        ohlson_o_score["Ohlson O-Score"] = ohlson_model.get_ohlson_o_score(
            log_of_total_assets=ohlson_o_score["Log of Total Assets"],
            total_liabilities_to_total_assets_ratio=ohlson_o_score[
                "Total Liabilities to Total Assets"
            ],
            working_capital_to_total_assets_ratio=ohlson_o_score[
                "Working Capital to Total Assets"
            ],
            current_liabilities_to_current_assets_ratio=ohlson_o_score[
                "Current Liabilities to Current Assets"
            ],
            negative_equity_indicator=ohlson_o_score["Negative Equity Indicator"],
            net_income_to_total_assets_ratio=ohlson_o_score[
                "Net Income to Total Assets"
            ],
            funds_from_operations_to_total_liabilities_ratio=ohlson_o_score[
                "Funds from Operations to Total Liabilities"
            ],
            negative_income_indicator=ohlson_o_score["Negative Income Indicator"],
            change_in_net_income_ratio=ohlson_o_score["Change in Net Income"],
        )

        ohlson_o_score["Ohlson Bankruptcy Probability"] = (
            ohlson_model.get_ohlson_bankruptcy_probability(
                ohlson_o_score=ohlson_o_score["Ohlson O-Score"],
            )
        )

        ohlson_results = (
            concat_frames(ohlson_o_score)
            .dropna(axis=1, how="all")
            .swaplevel(0, 1)
            .reindex(self._tickers, level=0)
        )

        ohlson_results = finalize_dataset(
            dataset=ohlson_results,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="columns",
        )

        return ohlson_results

    @handle_errors
    def get_zmijewski_score(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
        zone: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculates the Zmijewski Score, a financial metric used to predict the likelihood of a company
        going bankrupt. It is a simpler, three-variable model compared to both the Altman Z-Score and
        the Ohlson O-Score, and is commonly used as a robustness check when those two models disagree,
        since it is estimated on a different sample and with a different statistical technique (a
        probit model rather than discriminant analysis or logistic regression).

        The formula is as follows:

            - Net Income to Total Assets = Net Income / Total Assets
            - Total Liabilities to Total Assets = Total Liabilities / Total Assets
            - Current Assets to Current Liabilities = Current Assets / Current Liabilities
            - X = -4.336 - 4.513 * Net Income to Total Assets + 5.679 * Total Liabilities to Total Assets
            - 0.004 * Current Assets to Current Liabilities
            - Bankruptcy Probability = Phi(X), the standard normal cumulative distribution function

        The Zmijewski Score can be interpreted as follows:

            - A higher X (and therefore a higher bankruptcy probability) indicates a higher likelihood
            of financial distress.

        Also known as: Zmijewski Score, ZFC score, bankruptcy prediction, financial distress score.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The trailing period to use for the calculation. Defaults to
                None, which is 4 on quarterly data (a trailing twelve months), as the score was estimated on
                yearly statements.
            zone (bool, optional): Whether to add a "Zone" row per ticker: "Distress" from a score of 0, a
                bankruptcy probability of 0.5 and up (Zmijewski's classification), "Safe" below it. Not with
                growth or standardize. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: DataFrame containing the Zmijewski Score, its bankruptcy probability, and its
            components.

        Notes:
            - Unlike the Ohlson O-Score, which uses a logistic (sigmoid) link to convert its score into
            a probability, the Zmijewski Score uses a probit link (the standard normal CDF). Applying a
            logistic transform to the Zmijewski Score instead would produce an incorrect probability.
            - As with the Altman Z-Score, Piotroski F-Score, Beneish M-Score and Ohlson O-Score, this is
            a probabilistic, not a definitive, indicator and should be combined with further fundamental
            analysis.

        References:
        - Zmijewski, Mark E. "Methodological Issues Related to the Estimation of Financial Distress
        Prediction Models." Journal of Accounting Research, Vol. 22, 1984, pp. 59-82.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["DAL", "UAL", "AAL"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_zmijewski_score().xs("Zmijewski Score", level=1)
        ```

        Which returns:

        |     |   2021 |   2022 |   2023 |    2024 |    2025 |
        |:----|-------:|-------:|-------:|--------:|--------:|
        | AAL | 2.1018 | 1.8402 | 1.7502 |  1.6446 |  1.6755 |
        | DAL | 1.017  | 0.7425 | 0.2026 | -0.0177 | -0.3885 |
        | UAL | 1.0493 | 0.7082 | 0.4288 |  0.1763 |  0.0073 |
        """
        # The score was estimated on yearly statements, so quarterly data is summed into a
        # year (a trailing twelve months) unless a trailing period is given.
        if trailing is None and self._quarterly:
            trailing = 4
        zmijewski_score = {}

        current_assets = (
            self._balance_sheet_statement.loc[:, "Total Current Assets", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Current Assets", :]
        )
        current_liabilities = (
            self._balance_sheet_statement.loc[:, "Total Current Liabilities", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Current Liabilities", :]
        )
        total_assets = (
            self._balance_sheet_statement.loc[:, "Total Assets", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Assets", :]
        )
        total_liabilities = (
            self._balance_sheet_statement.loc[:, "Total Liabilities", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Liabilities", :]
        )
        net_income = (
            self._income_statement.loc[:, "Net Income", :].T.rolling(trailing).sum().T
            if trailing
            else self._income_statement.loc[:, "Net Income", :]
        )

        zmijewski_score["Net Income to Total Assets"] = (
            zmijewski_model.get_net_income_to_total_assets_ratio(
                net_income=net_income,
                total_assets=total_assets,
            )
        )

        zmijewski_score["Total Liabilities to Total Assets"] = (
            zmijewski_model.get_total_liabilities_to_total_assets_ratio(
                total_liabilities=total_liabilities,
                total_assets=total_assets,
            )
        )

        zmijewski_score["Current Assets to Current Liabilities"] = (
            zmijewski_model.get_current_assets_to_current_liabilities_ratio(
                current_assets=current_assets,
                current_liabilities=current_liabilities,
            )
        )

        zmijewski_score["Zmijewski Score"] = zmijewski_model.get_zmijewski_score(
            net_income_to_total_assets_ratio=zmijewski_score[
                "Net Income to Total Assets"
            ],
            total_liabilities_to_total_assets_ratio=zmijewski_score[
                "Total Liabilities to Total Assets"
            ],
            current_assets_to_current_liabilities_ratio=zmijewski_score[
                "Current Assets to Current Liabilities"
            ],
        )

        zmijewski_score["Zmijewski Bankruptcy Probability"] = (
            zmijewski_model.get_zmijewski_bankruptcy_probability(
                zmijewski_score=zmijewski_score["Zmijewski Score"],
            )
        )

        zmijewski_results = (
            concat_frames(zmijewski_score)
            .dropna(axis=1, how="all")
            .swaplevel(0, 1)
            .reindex(self._tickers, level=0)
        )

        zmijewski_results = finalize_dataset(
            dataset=zmijewski_results,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="columns",
        )

        if zone and not growth and not standardize:
            zmijewski_results = add_score_zone(
                zmijewski_results, "Zmijewski Score", [0.0], ["Safe", "Distress"]
            )

        return zmijewski_results

    @handle_errors
    def get_springate_score(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
        zone: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculates the Springate Score, a financial metric used to predict the likelihood of
        a company going bankrupt. It follows the same multiple discriminant analysis
        methodology as the Altman Z-Score, but was calibrated on a smaller, Canadian-firm
        sample using four financial ratios instead of five.

        The formula is as follows:

            - Working Capital to Total Assets = Working Capital / Total Assets
            - EBIT to Total Assets = EBIT / Total Assets
            - EBT to Current Liabilities = Earnings Before Taxes / Total Current Liabilities
            - Sales to Total Assets = Sales / Total Assets
            - Springate Score = 1.03 * Working Capital to Total Assets +
            3.07 * EBIT to Total Assets + 0.66 * EBT to Current Liabilities +
            0.4 * Sales to Total Assets

        The Springate Score can be interpreted as follows:

            - A Springate Score of less than 0.862 indicates a high likelihood of bankruptcy.
            - A Springate Score of greater than 0.862 indicates a low likelihood of bankruptcy.

        Also known as: Springate Score, S-Score, bankruptcy prediction, financial distress score.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The trailing period to use for the calculation. Defaults to
                None, which is 4 on quarterly data (a trailing twelve months), as the score was estimated on
                yearly statements.
            zone (bool, optional): Whether to add a "Zone" row per ticker: "Distress" below Springate's cut-off
                of 0.862, "Safe" from it. Not with growth or standardize. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: DataFrame containing the Springate Score and its components.

        Notes:
        - As with the Altman Z-Score, Piotroski F-Score, Beneish M-Score, Ohlson O-Score and
        Zmijewski Score, this is a probabilistic, not a definitive, indicator and should be
        combined with further fundamental analysis.
        - EBIT is the reported Operating Income, as in the Altman Z-Score calculation elsewhere
        in this module.

        References:
        - Springate, Gordon L.V. "Predicting the Possibility of Failure in a Canadian Firm."
        Unpublished M.B.A. Research Project, Simon Fraser University, 1978.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["M", "KSS", "GAP"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_springate_score().xs("Springate Score", level=1)
        ```

        Which returns:

        |     |   2021 |   2022 |   2023 |   2024 |
        |:----|-------:|-------:|-------:|-------:|
        | GAP | 0.8575 | 0.6249 | 0.9346 | 1.1942 |
        | KSS | 1.2226 | 0.5896 | 0.809  | 0.6373 |
        | M   | 1.2937 | 1.1721 | 0.7495 | 0.9653 |
        """
        # The score was estimated on yearly statements, so quarterly data is summed into a
        # year (a trailing twelve months) unless a trailing period is given.
        if trailing is None and self._quarterly:
            trailing = 4
        springate_score = {}

        current_assets = (
            self._balance_sheet_statement.loc[:, "Total Current Assets", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Current Assets", :]
        )
        current_liabilities = (
            self._balance_sheet_statement.loc[:, "Total Current Liabilities", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Current Liabilities", :]
        )
        total_assets = (
            self._balance_sheet_statement.loc[:, "Total Assets", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Assets", :]
        )
        income_before_tax = (
            self._income_statement.loc[:, "Income Before Tax", :]
            .T.rolling(trailing)
            .sum()
            .T
            if trailing
            else self._income_statement.loc[:, "Income Before Tax", :]
        )
        revenue = (
            self._income_statement.loc[:, "Revenue", :].T.rolling(trailing).sum().T
            if trailing
            else self._income_statement.loc[:, "Revenue", :]
        )

        working_capital = liquidity_model.get_working_capital(
            current_assets,
            current_liabilities,
        )

        springate_score["Working Capital to Total Assets"] = (
            springate_model.get_working_capital_to_total_assets_ratio(
                working_capital=working_capital,
                total_assets=total_assets,
            )
        )

        springate_score["EBIT to Total Assets"] = (
            springate_model.get_ebit_to_total_assets_ratio(
                ebit=self._earnings_before_interest_and_taxes(trailing),
                total_assets=total_assets,
            )
        )

        springate_score["EBT to Current Liabilities"] = (
            springate_model.get_ebt_to_current_liabilities_ratio(
                ebt=income_before_tax,
                current_liabilities=current_liabilities,
            )
        )

        springate_score["Sales to Total Assets"] = (
            springate_model.get_sales_to_total_assets_ratio(
                sales=revenue,
                total_assets=total_assets,
            )
        )

        springate_score["Springate Score"] = springate_model.get_springate_score(
            working_capital_to_total_assets_ratio=springate_score[
                "Working Capital to Total Assets"
            ],
            ebit_to_total_assets_ratio=springate_score["EBIT to Total Assets"],
            ebt_to_current_liabilities_ratio=springate_score[
                "EBT to Current Liabilities"
            ],
            sales_to_total_assets_ratio=springate_score["Sales to Total Assets"],
        )

        springate_results = (
            concat_frames(springate_score)
            .dropna(axis=1, how="all")
            .swaplevel(0, 1)
            .reindex(self._tickers, level=0)
        )

        springate_results = finalize_dataset(
            dataset=springate_results,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="columns",
        )

        if zone and not growth and not standardize:
            springate_results = add_score_zone(
                springate_results, "Springate Score", [0.862], ["Distress", "Safe"]
            )

        return springate_results

    @handle_errors
    def get_grover_score(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculates the Grover Score, a financial metric used to predict the likelihood of a
        company going bankrupt. It was developed by re-estimating the coefficients of a
        reduced-form Altman Z-Score and adding a Return on Assets term, using a sample that
        paired each of Altman's original bankrupt firms with a matched non-bankrupt firm from
        the same industry and year.

        The formula is as follows:

            - Working Capital to Total Assets = Working Capital / Total Assets
            - EBIT to Total Assets = EBIT / Total Assets
            - Return on Assets = Net Income / Total Assets
            - Grover Score = 1.65 * Working Capital to Total Assets +
            3.404 * EBIT to Total Assets - 0.016 * Return on Assets + 0.057

        The Grover Score can be interpreted as follows:

            - A Grover Score of -0.02 or lower indicates a high likelihood of bankruptcy.
            - A Grover Score of 0.01 or higher indicates a low likelihood of bankruptcy
            (per some secondary sources), leaving a gray area in between the two thresholds.

        Also known as: Grover Score, G-Score, bankruptcy prediction, financial distress score.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The trailing period to use for the calculation. Defaults to
                None, which is 4 on quarterly data (a trailing twelve months), as the score was estimated on
                yearly statements.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: DataFrame containing the Grover Score and its components.

        Notes:
        - As with the Altman Z-Score, Springate Score, Ohlson O-Score and Zmijewski Score,
        this is a probabilistic, not a definitive, indicator and should be combined with
        further fundamental analysis.
        - EBIT is the reported Operating Income, as in the Altman Z-Score calculation elsewhere
        in this module. Return on
        Assets uses the point-in-time Total Assets balance (not averaged), matching the
        original Grover (2001) specification.

        References:
        - Grover, Jeffrey S. "Validating the Grover Bankruptcy Model." Doctoral dissertation,
        University of North Texas, 2003.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["F", "GM", "STLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_grover_score().xs("Grover Score", level=1)
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |    2025 |
        |:-----|-------:|-------:|-------:|-------:|--------:|
        | F    | 0.2398 | 0.2688 | 0.2442 | 0.2194 |  0.1363 |
        | GM   | 0.2379 | 0.2474 | 0.2157 | 0.2846 |  0.1825 |
        | STLA | 0.477  | 0.5941 | 0.5796 | 0.2001 | -0.3018 |
        """
        # The score was estimated on yearly statements, so quarterly data is summed into a
        # year (a trailing twelve months) unless a trailing period is given.
        if trailing is None and self._quarterly:
            trailing = 4
        grover_score = {}

        current_assets = (
            self._balance_sheet_statement.loc[:, "Total Current Assets", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Current Assets", :]
        )
        current_liabilities = (
            self._balance_sheet_statement.loc[:, "Total Current Liabilities", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Current Liabilities", :]
        )
        total_assets = (
            self._balance_sheet_statement.loc[:, "Total Assets", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Assets", :]
        )
        net_income = (
            self._income_statement.loc[:, "Net Income", :].T.rolling(trailing).sum().T
            if trailing
            else self._income_statement.loc[:, "Net Income", :]
        )

        working_capital = liquidity_model.get_working_capital(
            current_assets,
            current_liabilities,
        )

        grover_score["Working Capital to Total Assets"] = (
            grover_model.get_working_capital_to_total_assets_ratio(
                working_capital=working_capital,
                total_assets=total_assets,
            )
        )

        grover_score["EBIT to Total Assets"] = (
            grover_model.get_ebit_to_total_assets_ratio(
                ebit=self._earnings_before_interest_and_taxes(trailing),
                total_assets=total_assets,
            )
        )

        grover_score["Return on Assets"] = grover_model.get_return_on_assets_ratio(
            net_income=net_income,
            total_assets=total_assets,
        )

        grover_score["Grover Score"] = grover_model.get_grover_score(
            working_capital_to_total_assets_ratio=grover_score[
                "Working Capital to Total Assets"
            ],
            ebit_to_total_assets_ratio=grover_score["EBIT to Total Assets"],
            return_on_assets_ratio=grover_score["Return on Assets"],
        )

        grover_results = (
            concat_frames(grover_score)
            .dropna(axis=1, how="all")
            .swaplevel(0, 1)
            .reindex(self._tickers, level=0)
        )

        grover_results = finalize_dataset(
            dataset=grover_results,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="columns",
        )

        return grover_results

    @handle_errors
    def get_fulmer_h_score(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculates the Fulmer H-Score, a financial metric used to predict the likelihood of a
        company going bankrupt. Unlike the Altman Z-Score, which was built on relatively
        large, listed manufacturing companies, the Fulmer H-Score was calibrated on a
        matched-pair sample of small, failed and non-failed U.S. companies, using nine
        financial ratios selected via step-wise multiple discriminant analysis.

        The formula is as follows:

        - V1 = Retained Earnings / Total Assets
        - V2 = Revenue / Total Assets
        - V3 = Income Before Tax (EBT) / Total Equity
        - V4 = Operating Cash Flow / Total Liabilities
        - V5 = Total Debt / Total Assets
        - V6 = Total Current Liabilities / Total Assets
        - V7 = ln(Total Assets - Goodwill and Intangible Assets)
        - V8 = (Total Current Assets - Total Current Liabilities) / Total Liabilities
        - V9 = ln(EBIT / Interest Expense), with EBIT the reported Operating Income
        - H-Score = 5.528 * V1 + 0.212 * V2 + 0.073 * V3 + 1.270 * V4 - 0.120 * V5 + 2.335 * V6 + 0.575 * V7 +
          1.083 * V8 + 0.894 * V9 - 6.075

        With `trailing`, the flows are summed and the balance sheet items averaged over the trailing periods.

        The Fulmer H-Score can be interpreted as follows:

            - An H-Score of less than 0 predicts failure.
            - An H-Score of greater than 0 predicts non-failure.

        Also known as: Fulmer H-Score, H factor, bankruptcy prediction, financial distress score.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The trailing period to use for the calculation. Defaults to
                None, which is 4 on quarterly data (a trailing twelve months), as the score was estimated on
                yearly statements.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: DataFrame containing the Fulmer H-Score and its components.

        Notes:
        - Because V7 and V9 involve a natural logarithm, periods where Tangible Total Assets
        is zero or negative, where EBIT and Interest Expense do not share the same sign, or
        where the company reports no Interest Expense at all, will produce NaN and therefore
        no H-Score for that period. This is a structural limitation of the log-linear Fulmer
        specification, not a bug — the same kind of limitation documented for the Graham
        Number elsewhere in this toolkit.
        - Because V7 uses an un-normalized dollar figure, the H-Score is sensitive to the
        absolute scale of a company's total assets; treat results for very large companies
        (far outside the small-firm sample the model was calibrated on) with additional
        caution.
        - EBIT is the reported Operating Income, as in the Altman Z-Score calculation elsewhere
        in this module.
        - As with the Altman Z-Score, Springate Score and Grover Score, this is a
        probabilistic, not a definitive, indicator and should be combined with further
        fundamental analysis.

        References:
        - Fulmer, John G., James E. Moon, Thomas A. Gavin, and Michael J. Erwin. "A
        Bankruptcy Classification Model for Small Firms." Journal of Commercial Bank
        Lending, Vol. 66, No. 11, 1984, pp. 25-37.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["DIS", "FOXA", "CMCSA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_fulmer_h_score().xs("Fulmer H-Score", level=1)
        ```

        Which returns:

        |       |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:------|--------:|--------:|--------:|--------:|--------:|
        | CMCSA | 11.6317 | 11.6248 | 11.7028 | 11.8104 | 11.9431 |
        | DIS   | 10.8945 | 11.5241 | 12.0105 | 12.4096 | 12.7101 |
        | FOXA  | 10.9776 | 11.0388 | 10.9996 | 11.0274 | 11.8441 |
        """
        # The score was estimated on yearly statements, so quarterly data is summed into a
        # year (a trailing twelve months) unless a trailing period is given.
        if trailing is None and self._quarterly:
            trailing = 4
        fulmer_h_score = {}

        current_assets = (
            self._balance_sheet_statement.loc[:, "Total Current Assets", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Current Assets", :]
        )
        current_liabilities = (
            self._balance_sheet_statement.loc[:, "Total Current Liabilities", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Current Liabilities", :]
        )
        total_assets = (
            self._balance_sheet_statement.loc[:, "Total Assets", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Assets", :]
        )
        total_liabilities = (
            self._balance_sheet_statement.loc[:, "Total Liabilities", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Liabilities", :]
        )
        total_debt = (
            self._balance_sheet_statement.loc[:, "Total Debt", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Debt", :]
        )
        total_equity = (
            self._balance_sheet_statement.loc[:, "Total Equity", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Equity", :]
        )
        retained_earnings = (
            self._balance_sheet_statement.loc[:, "Retained Earnings", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Retained Earnings", :]
        )
        goodwill_and_intangible_assets = (
            get_goodwill_and_intangible_assets(self._balance_sheet_statement)
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else get_goodwill_and_intangible_assets(self._balance_sheet_statement)
        )
        revenue = (
            self._income_statement.loc[:, "Revenue", :].T.rolling(trailing).sum().T
            if trailing
            else self._income_statement.loc[:, "Revenue", :]
        )
        income_before_tax = (
            self._income_statement.loc[:, "Income Before Tax", :]
            .T.rolling(trailing)
            .sum()
            .T
            if trailing
            else self._income_statement.loc[:, "Income Before Tax", :]
        )
        interest_expense = (
            self._income_statement.loc[:, "Interest Expense", :]
            .T.rolling(trailing)
            .sum()
            .T
            if trailing
            else self._income_statement.loc[:, "Interest Expense", :]
        )
        operating_cash_flow = (
            self._cash_flow_statement.loc[:, "Operating Cash Flow", :]
            .T.rolling(trailing)
            .sum()
            .T
            if trailing
            else self._cash_flow_statement.loc[:, "Operating Cash Flow", :]
        )

        working_capital = liquidity_model.get_working_capital(
            current_assets,
            current_liabilities,
        )
        ebit = self._earnings_before_interest_and_taxes(trailing)
        tangible_total_assets = fulmer_model.get_tangible_total_assets(
            total_assets=total_assets,
            goodwill_and_intangible_assets=goodwill_and_intangible_assets,
        )

        fulmer_h_score["Retained Earnings to Total Assets"] = (
            fulmer_model.get_retained_earnings_to_total_assets_ratio(
                retained_earnings=retained_earnings,
                total_assets=total_assets,
            )
        )

        fulmer_h_score["Sales to Total Assets"] = (
            fulmer_model.get_sales_to_total_assets_ratio(
                sales=revenue,
                total_assets=total_assets,
            )
        )

        fulmer_h_score["EBT to Total Equity"] = fulmer_model.get_ebt_to_equity_ratio(
            ebt=income_before_tax,
            total_equity=total_equity,
        )

        fulmer_h_score["Cash Flow to Total Liabilities"] = (
            fulmer_model.get_cash_flow_to_total_liabilities_ratio(
                cash_flow=operating_cash_flow,
                total_liabilities=total_liabilities,
            )
        )

        fulmer_h_score["Debt to Total Assets"] = (
            fulmer_model.get_debt_to_total_assets_ratio(
                total_debt=total_debt,
                total_assets=total_assets,
            )
        )

        fulmer_h_score["Current Liabilities to Total Assets"] = (
            fulmer_model.get_current_liabilities_to_total_assets_ratio(
                current_liabilities=current_liabilities,
                total_assets=total_assets,
            )
        )

        fulmer_h_score["Log of Tangible Total Assets"] = (
            fulmer_model.get_log_of_tangible_total_assets(
                tangible_total_assets=tangible_total_assets,
            )
        )

        fulmer_h_score["Working Capital to Total Liabilities"] = (
            fulmer_model.get_working_capital_to_total_liabilities_ratio(
                working_capital=working_capital,
                total_liabilities=total_liabilities,
            )
        )

        fulmer_h_score["Log of EBIT to Interest Expense"] = (
            fulmer_model.get_log_of_ebit_to_interest_expense_ratio(
                ebit=ebit,
                interest_expense=interest_expense,
            )
        )

        fulmer_h_score["Fulmer H-Score"] = fulmer_model.get_fulmer_h_score(
            retained_earnings_to_total_assets_ratio=fulmer_h_score[
                "Retained Earnings to Total Assets"
            ],
            sales_to_total_assets_ratio=fulmer_h_score["Sales to Total Assets"],
            ebt_to_equity_ratio=fulmer_h_score["EBT to Total Equity"],
            cash_flow_to_total_liabilities_ratio=fulmer_h_score[
                "Cash Flow to Total Liabilities"
            ],
            debt_to_total_assets_ratio=fulmer_h_score["Debt to Total Assets"],
            current_liabilities_to_total_assets_ratio=fulmer_h_score[
                "Current Liabilities to Total Assets"
            ],
            log_of_tangible_total_assets=fulmer_h_score["Log of Tangible Total Assets"],
            working_capital_to_total_liabilities_ratio=fulmer_h_score[
                "Working Capital to Total Liabilities"
            ],
            log_of_ebit_to_interest_expense_ratio=fulmer_h_score[
                "Log of EBIT to Interest Expense"
            ],
        )

        fulmer_results = (
            concat_frames(fulmer_h_score)
            .dropna(axis=1, how="all")
            .swaplevel(0, 1)
            .reindex(self._tickers, level=0)
        )

        fulmer_results = finalize_dataset(
            dataset=fulmer_results,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="columns",
        )

        return fulmer_results

    @handle_errors
    def get_taffler_z_score(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
        zone: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculates the Taffler z-score, a discriminant model of financial distress estimated on UK listed
        industrial companies (Taffler, 1983). It combines profitability, working capital, financial risk and
        liquidity, and Agarwal and Taffler (2007) showed that it kept its predictive ability over 25 years of
        UK data. A negative score places a company in the at-risk region.

        The formula is as follows:

            - Profit Before Tax to Current Liabilities (X1) = Income Before Tax / Total Current Liabilities
            - Current Assets to Total Liabilities (X2) = Total Current Assets / Total Liabilities
            - Current Liabilities to Total Assets (X3) = Total Current Liabilities / Total Assets
            - Daily Operating Expenses = (Revenue - Income Before Tax - Depreciation and Amortization) / Days
            - No-Credit Interval (X4) = (Total Current Assets - Inventory - Total Current Liabilities) /
            Daily Operating Expenses
            - Taffler z-score = 3.20 + 12.18 X1 + 2.50 X2 - 10.68 X3 + 0.029 X4

        The Taffler z-score can be interpreted as follows:

            - A score below zero indicates a financial profile like that of companies that failed (at risk).
            - A score above zero indicates a solvent profile; the higher, the safer.

        Also known as: Taffler Z-Score, UK z-score, bankruptcy prediction, financial distress score.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The trailing period to use for the calculation: flows are
                summed and balances averaged over it. Defaults to None, which is 4 on quarterly data (a
                trailing twelve months), as the score was estimated on yearly statements.
            zone (bool, optional): Whether to add a "Zone" row per ticker: "Distress" below zero (Taffler's at
                risk profile), "Safe" from it. Not with growth or standardize. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: DataFrame containing the Taffler z-score and its components.

        Notes:
        - The No-Credit Interval is in days. The days are those the flows cover: 365 a year, a quarter of
        that for a quarter, times the number of trailing periods.
        - Quick assets are the Total Current Assets less Inventory, and the Depreciation and Amortization
        stands in for the depreciation of the paper's daily operating expenses.
        - Agarwal and Taffler (2007, Table 1) attribute 53% of the discriminating power to X1, 13% to X2, 18%
        to X3 and 16% to X4. The coefficients were estimated on UK industrial companies, so the model is not
        meant for financial companies and is best read as a trend for companies elsewhere.
        - As with the Altman Z-Score and the other scores in this module, this is a probabilistic, not a
        definitive, indicator and should be combined with further fundamental analysis.

        References:
        - Taffler, R. J. (1983). The Assessment of Company Solvency and Performance Using a Statistical
        Model. Accounting and Business Research, 13(52), 295-308.
        - Agarwal, V. and Taffler, R. J. (2007). Twenty-Five Years of the Taffler z-Score Model: Does It
        Really Have Predictive Ability? Accounting and Business Research, 37(4), 285-300.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["M", "KSS", "GAP"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2020-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_taffler_z_score().xs("Taffler Z-Score", level=1)
        ```

        Which returns:

        |     |     2020 |   2021 |    2022 |   2023 |   2024 |
        |:----|---------:|-------:|--------:|-------:|-------:|
        | GAP |  -2.1624 | 0.7496 |  0.1718 | 3.1604 | 5.9418 |
        | KSS |   0.4845 | 5.7852 | -0.0302 | 2.4432 | 0.1935 |
        | M   | -11.3244 | 3.8969 |  3.4676 | 0.2883 | 2.4301 |
        """
        # The score was estimated on yearly statements, so quarterly data is summed into a
        # year (a trailing twelve months) unless a trailing period is given.
        if trailing is None and self._quarterly:
            trailing = 4
        taffler_score = {}

        def balance_item(item: str) -> pd.DataFrame:
            """A balance sheet item, averaged over the trailing periods when given."""
            values = self._balance_sheet_statement.loc[:, item, :]
            return values.T.rolling(trailing).mean().T if trailing else values

        def income_item(item: str) -> pd.DataFrame:
            """An income statement item, summed over the trailing periods when given."""
            values = self._income_statement.loc[:, item, :]
            return values.T.rolling(trailing).sum().T if trailing else values

        current_assets = balance_item("Total Current Assets")
        current_liabilities = balance_item("Total Current Liabilities")
        income_before_tax = income_item("Income Before Tax")

        taffler_score["Profit Before Tax to Current Liabilities"] = (
            taffler_model.get_profit_before_tax_to_current_liabilities_ratio(
                profit_before_tax=income_before_tax,
                current_liabilities=current_liabilities,
            )
        )

        taffler_score["Current Assets to Total Liabilities"] = (
            taffler_model.get_current_assets_to_total_liabilities_ratio(
                current_assets=current_assets,
                total_liabilities=balance_item("Total Liabilities"),
            )
        )

        taffler_score["Current Liabilities to Total Assets"] = (
            taffler_model.get_current_liabilities_to_total_assets_ratio(
                current_liabilities=current_liabilities,
                total_assets=balance_item("Total Assets"),
            )
        )

        # The paper spreads a year of operating expenses over 365 days, so the days follow the
        # length of the flows: a quarter of a year per quarter, times the trailing periods.
        taffler_score["No-Credit Interval"] = taffler_model.get_no_credit_interval(
            quick_assets=current_assets - balance_item("Inventory"),
            current_liabilities=current_liabilities,
            sales=income_item("Revenue"),
            profit_before_tax=income_before_tax,
            depreciation=income_item("Depreciation and Amortization"),
            days=(365 / 4 if self._quarterly else 365) * (trailing or 1),
        )

        taffler_score["Taffler Z-Score"] = taffler_model.get_taffler_z_score(
            profit_before_tax_to_current_liabilities_ratio=taffler_score[
                "Profit Before Tax to Current Liabilities"
            ],
            current_assets_to_total_liabilities_ratio=taffler_score[
                "Current Assets to Total Liabilities"
            ],
            current_liabilities_to_total_assets_ratio=taffler_score[
                "Current Liabilities to Total Assets"
            ],
            no_credit_interval=taffler_score["No-Credit Interval"],
        )

        taffler_results = (
            concat_frames(taffler_score)
            .dropna(axis=1, how="all")
            .swaplevel(0, 1)
            .reindex(self._tickers, level=0)
        )

        taffler_results = finalize_dataset(
            dataset=taffler_results,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="columns",
        )

        if zone and not growth and not standardize:
            taffler_results = add_score_zone(
                taffler_results, "Taffler Z-Score", [0.0], ["Distress", "Safe"]
            )

        return taffler_results

    def _period_risk_free_rate(self, columns: pd.PeriodIndex) -> pd.Series:
        """The risk free rate of every statement period."""
        risk_free = self._risk_free_rate_data[
            "quarterly" if self._quarterly else "yearly"
        ]["Adj Close"]
        risk_free.index = risk_free.index.asfreq(
            self._balance_sheet_statement.columns.freq
        )
        return risk_free.reindex(columns)

    @handle_errors
    def get_merton_distance_to_default(
        self,
        time_horizon: float = 1.0,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the distance to default and probability of default with the Merton
        (1974) structural credit model. Equity is seen as a call option on the company's
        assets with its debt as strike price: shareholders receive what is left after the
        debt is repaid. The market value and volatility of the assets, which cannot be
        observed, are solved from the market value and volatility of the equity:

            - E = V N(d1) - D e^(-rT) N(d2)
            - σ_E E = N(d1) σ_V V
            - Distance to Default = (ln(V/D) + (r - σ_V²/2) T) / (σ_V √T)
            - Probability of Default = N(-Distance to Default)

        where E is the market capitalization, σ_E the volatility of the share price over the
        year before, D the default point (short-term debt plus half of the long-term debt,
        the KMV convention), r the risk free rate and T the time horizon. The distance to
        default is the number of standard deviations the asset value is above the default
        point; it reacts to the share price immediately, long before a rating changes.

        The risk free rate is the Toolkit's (the 10-year Treasury yield unless set otherwise
        with `risk_free_rate`) at every period end, where Vassalou and Xing (2004) and
        Bharath and Shumway (2008) use the one-year rate for a one-year horizon; the
        difference moves the distance to default only slightly.

        For more information, see the following papers:

        - Merton, R.C. (1974). "On the Pricing of Corporate Debt: The Risk Structure of
          Interest Rates." The Journal of Finance, 29(2), 449-470.
        - Crosbie, P. and Bohn, J. (2003). "Modeling Default Risk." Moody's KMV.
        - Vassalou, M. and Xing, Y. (2004). "Default Risk in Equity Returns." The Journal of
          Finance, 59(2), 831-868.
        - Bharath, S.T. and Shumway, T. (2008). "Forecasting Default with the Merton Distance
          to Default Model." The Review of Financial Studies, 21(3), 1339-1369.

        Also known as: KMV model, structural credit model, expected default frequency.

        Args:
            time_horizon (float, optional): The horizon in years over which default is
                measured. Defaults to 1.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per ticker and period the Equity Value, Equity Volatility, Default
            Point, Asset Value, Asset Volatility, Distance to Default and Probability of
            Default (a decimal).

        Notes:
        - The probability is the model's risk-neutral probability; it ranks companies by
          credit risk well but tends to understate the actual default rate of safe ones.
        - Financial companies, whose debt is their raw material, are outside the model's
          intended use.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["F", "AAPL"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_merton_distance_to_default().loc["F"]
        ```

        Which returns:

        |                        |        2021 |        2022 |        2023 |        2024 |        2025 |
        |:-----------------------|------------:|------------:|------------:|------------:|------------:|
        | Equity Value           | 8.37862e+10 | 4.66828e+10 | 4.92598e+10 | 3.98079e+10 | 5.22045e+10 |
        | Equity Volatility      | 0.4163      | 0.491       | 0.3549      | 0.36        | 0.3319      |
        | Default Point          | 9.3892e+10  | 9.45665e+10 | 9.945e+10   | 1.07294e+11 | 1.1133e+11  |
        | Asset Value            | 1.76332e+11 | 1.38452e+11 | 1.44842e+11 | 1.42679e+11 | 1.58862e+11 |
        | Asset Volatility       | 0.1979      | 0.1665      | 0.1207      | 0.1005      | 0.1091      |
        | Distance to Default    | 3.1586      | 2.3846      | 3.3822      | 3.2046      | 3.598       |
        | Probability of Default | 0.0008      | 0.0085      | 0.0004      | 0.0007      | 0.0002      |
        """
        balance = self._balance_sheet_statement
        shares = self._rows(self._income_statement, "Weighted Average Shares Diluted")
        equity_value = (self._period_prices() * shares).reindex(
            index=self._tickers, columns=balance.columns
        )
        default_point = self._rows(balance, "Short Term Debt").fillna(
            0
        ) + 0.5 * self._rows(balance, "Long Term Debt").fillna(0)

        # The volatility of the daily returns over the 252 trading days (at least 60) up to
        # every period end, annualized, whether the statements are yearly or quarterly.
        daily_returns = self._historical_data["daily"]["Return"].reindex(
            columns=self._tickers
        )
        daily_returns.index = daily_returns.index.to_timestamp()
        volatility = daily_returns.rolling(252, min_periods=60).std() * (252**0.5)
        period_end = balance.columns.to_timestamp(how="end")
        equity_volatility = (
            volatility.reindex(volatility.index.union(period_end))
            .ffill()
            .reindex(period_end)
        )
        equity_volatility.index = balance.columns
        equity_volatility = equity_volatility.T

        risk_free = self._period_risk_free_rate(balance.columns)

        model_metrics = [
            "Default Point",
            "Asset Value",
            "Asset Volatility",
            "Distance to Default",
            "Probability of Default",
        ]
        results = {
            "Equity Value": equity_value,
            "Equity Volatility": equity_volatility,
            **{
                metric: pd.DataFrame(
                    index=self._tickers, columns=balance.columns, dtype=float
                )
                for metric in model_metrics
            },
        }

        for ticker in self._tickers:
            for period in balance.columns:
                model = credit_model.get_merton_model(
                    equity_value=float(equity_value.loc[ticker, period]),
                    equity_volatility=float(equity_volatility.loc[ticker, period]),
                    default_point=float(default_point.loc[ticker, period]),
                    risk_free_rate=float(risk_free.get(period, float("nan"))),
                    time_horizon=time_horizon,
                )
                for metric in model_metrics:
                    results[metric].loc[ticker, period] = model[metric]

        merton = self._combine(results).dropna(axis=1, how="all")
        merton = apply_rounding(
            merton, rounding if rounding is not None else self._rounding
        )

        return merton.loc[:, self._start_date : self._end_date]

    @handle_errors
    def get_campbell_hilscher_szilagyi_score(
        self,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the probability of failure of Campbell, Hilscher and Szilagyi (2008), a
        logit model that combines accounting figures with market data to predict failure: a
        bankruptcy filing, a delisting or a D (default) rating. It measures profitability
        and leverage against the market value of the assets rather than the book value, and
        adds the excess return of the shares, their volatility, the company's size, its
        cash, its market-to-book ratio and its share price. In the paper it predicts failure
        better than Shumway's (2001) model and than the Merton distance to default.

        The model is the 12-month-ahead logit of Table IV, column (3), of the paper:

            - LPFD = -9.164 - 20.264 NIMTAAVG + 1.416 TLMTA - 7.129 EXRETAVG + 1.411 SIGMA
              - 0.045 RSIZE - 2.132 CASHMTA + 0.075 MB - 0.058 PRICE
            - Failure Probability = 1 / (1 + e^(-LPFD))

        with:

            - MTA = Market Value of Equity + Total Liabilities (the market value of total
              assets)
            - NIMTA = Quarterly Net Income / MTA, and NIMTAAVG its average over four quarters
              with weights that halve every quarter: (1 - φ³) / (1 - φ¹²) × (NIMTA(t) +
              φ³ NIMTA(t-1) + φ⁶ NIMTA(t-2) + φ⁹ NIMTA(t-3)), φ = 2^(-1/3)
            - EXRET = ln(1 + Return) - ln(1 + S&P 500 Return) per month, and EXRETAVG its
              average over twelve months: (1 - φ) / (1 - φ¹²) × (EXRET(t) + φ EXRET(t-1) +
              ... + φ¹¹ EXRET(t-11))
            - TLMTA = Total Liabilities / MTA
            - CASHMTA = Cash and Short Term Investments / MTA
            - MB = Market Value of Equity / Adjusted Book Equity, where Adjusted Book Equity =
              Book Equity + 0.1 × (Market Value of Equity - Book Equity) (Cohen, Polk and
              Vuolteenaho, 2003), and a value that is still not positive becomes 1
            - SIGMA = sqrt(252 / (N - 1) × Σ r²), the annualized volatility around zero of
              the N daily returns of the three months before, NaN with fewer than five
              non-zero returns
            - RSIZE = ln(Market Value of Equity / Total Market Value of the S&P 500)
            - PRICE = ln(min(Share Price, 15))

        Book Equity is Total Shareholder Equity plus Deferred Tax Liabilities minus Preferred
        Stock (Davis, Fama and French, 2000), the latter two counted as zero when not
        reported. The market value of equity is the closing price times the diluted
        weighted average shares, the returns are total returns (Adj Close) against the S&P
        500 total return index (^SP500TR), and every market variable is measured on the day
        each period's figures apply from (the fiscal period end, or the filing day with
        point_in_time): no market data after that day is used.

        The probability is that of failing in the month 12 months ahead, given that the
        company survives until then. It is a monthly probability, not the probability of
        failing within a year. For comparison, between 0.4% and 3.7% of the companies in the
        paper's sample failed per year from 1983 to 2003 (Table I).

        Also known as: CHS model, failure probability, distress risk, default probability, LPFD.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per ticker and period the eight variables as they enter the model,
            the CHS Score (LPFD) and the Failure Probability (a decimal).

        Notes:
            - Winsorization: the paper winsorizes every variable at the 5th and 95th
              percentiles of the monthly cross-section of all US companies, which a
              calculation per company cannot repeat. Each variable (NIMTA per quarter, EXRET
              per month) is clipped instead to the range the winsorized variables span in
              the paper's full sample (Table II, minimum and maximum): NIMTA -0.069 to 0.028,
              TLMTA 0.036 to 0.923, EXRET -0.243 to 0.218, RSIZE -13.568 to -6.773, SIGMA
              0.153 to 1.353, CASHMTA 0.002 to 0.358, MB 0.358 to 6.471 and PRICE -0.065 to
              2.708. The variables are shown clipped, as they enter the score; a company
              larger than about 0.1% of the S&P 500 has the largest RSIZE the paper used.
            - RSIZE: the total market value of the S&P 500 is the sum of the latest market
              capitalizations of its current companies (FinancialModelingPrep, each company
              once), scaled back to each date with the S&P 500 price index (^GSPC). This
              holds the companies and share counts of the latest date: changes in the index
              membership and share counts that S&P offsets with its divisor are not
              followed, and the index weighs float-adjusted shares while the total counts all
              shares. As RSIZE enters with a coefficient of -0.045, an error of 5% in the
              total moves the score by about 0.002. Without an API key that has access to
              the constituents, RSIZE and the probability are NaN.
            - On yearly statements NIMTAAVG is a quarter of the yearly net income over the
              market value of total assets at the year end, equal weights for the four
              quarters instead of weights that halve every quarter; on quarterly statements
              it is calculated as in the paper. A missing quarter or month leaves the average
              NaN where the paper fills it with the cross-sectional mean.
            - PRICE uses the split-adjusted closing price, so for a company that split its
              shares (or reverse-split them) after the period, the price differs from the
              price traded then by the split ratio, which matters only below $15.
            - The paper lags the accounting figures by two months so they are known; set
              point_in_time on the Toolkit to measure on the filing day instead of the
              fiscal period end.
            - The coefficients were estimated on US companies from 1963 to 2003 with the S&P
              500 as the market. For a company listed in another currency, its market value
              is compared with the dollar total of the S&P 500, which moves RSIZE by the log
              of the exchange rate (about 0.004 on the score for the euro).

        References:
        - Campbell, J.Y., Hilscher, J. and Szilagyi, J. (2008). "In Search of Distress Risk."
          The Journal of Finance, 63(6), 2899-2939.
        - Cohen, R.B., Polk, C. and Vuolteenaho, T. (2003). "The Value Spread." The Journal
          of Finance, 58(2), 609-641.
        - Davis, J.L., Fama, E.F. and French, K.R. (2000). "Characteristics, Covariances, and
          Average Returns: 1929 to 1997." The Journal of Finance, 55(1), 389-406.
        - Shumway, T. (2001). "Forecasting Bankruptcy More Accurately: A Simple Hazard
          Model." The Journal of Business, 74(1), 101-124.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["F", "AAPL"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_campbell_hilscher_szilagyi_score().loc["F"]
        ```

        Which returns:

        |                                        |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:---------------------------------------|--------:|--------:|--------:|--------:|--------:|
        | Net Income to Market Assets (NIMTAAVG) |  0.0153 | -0.0019 |  0.0039 |  0.0052 | -0.0067 |
        | Liabilities to Market Assets (TLMTA)   |  0.7133 |  0.82   |  0.8239 |  0.8579 |  0.8291 |
        | Excess Return (EXRETAVG)               |  0.0573 | -0.0308 | -0.007  | -0.037  |  0.015  |
        | Volatility (SIGMA)                     |  0.4477 |  0.4187 |  0.4069 |  0.3133 |  0.3746 |
        | Relative Size (RSIZE)                  | -6.773  | -6.773  | -6.7827 | -7.2053 | -7.0859 |
        | Cash to Market Assets (CASHMTA)        |  0.1697 |  0.1699 |  0.1436 |  0.1369 |  0.126  |
        | Market to Book (MB)                    |  1.567  |  1.0379 |  1.1113 |  0.8788 |  1.3456 |
        | Log Price (PRICE)                      |  2.708  |  2.4536 |  2.5006 |  2.2925 |  2.5741 |
        | CHS Score                              | -8.3383 | -7.2758 | -7.5146 | -7.3842 | -7.4308 |
        | Failure Probability                    |  0.0002 |  0.0007 |  0.0005 |  0.0006 |  0.0006 |
        """
        balance = self._balance_sheet_statement
        periods = balance.columns
        tickers = self._tickers

        dates = self._period_end_dates
        if dates is None or dates.empty:
            dates = pd.DataFrame(
                {
                    period: period.to_timestamp(how="end").normalize()
                    for period in periods
                },
                index=tickers,
            )
        dates = dates.reindex(index=tickers, columns=periods).apply(pd.to_datetime)

        # The daily data from a year before the start date, so that the excess returns of
        # the first periods have their twelve months.
        daily = self._historical_data.get("lookback")
        if daily is None or daily.empty:
            daily = self._historical_data["daily"]
        close = daily["Close"].reindex(columns=tickers)
        close.index = close.index.to_timestamp()
        adjusted_close = daily["Adj Close"].reindex(columns=tickers)
        adjusted_close.index = adjusted_close.index.to_timestamp()
        daily_returns = daily["Return"].reindex(columns=tickers)
        daily_returns.index = daily_returns.index.to_timestamp()

        # The S&P 500 over the prices' dates and up to today, as its total market value is
        # known for the latest date only.
        index_data, latest_market_value = (
            self._sp500_function(
                close.index.min().strftime("%Y-%m-%d"),
                pd.Timestamp.today().strftime("%Y-%m-%d"),
            )
            if self._sp500_function is not None and not close.empty
            else (pd.DataFrame(), pd.Series(dtype=float))
        )
        price_index = index_data.get("Price Index", pd.Series(dtype=float))
        total_return_index = index_data.get(
            "Total Return Index", pd.Series(dtype=float)
        )
        index_market_value = chs_model.get_index_market_value(
            price_index, latest_market_value
        )
        if index_market_value.dropna().empty:
            logger.warning(
                "The total market value of the S&P 500 (its constituents and their market "
                "capitalizations from FinancialModelingPrep) is not available, so the "
                "relative size (RSIZE) and the failure probability are NaN."
            )

        price = helpers.values_on_or_before(close, dates)
        shares = self._rows(self._income_statement, "Weighted Average Shares Diluted")
        market_equity = price * shares.reindex(index=tickers, columns=periods)
        total_liabilities = self._rows(balance, "Total Liabilities")
        market_total_assets = chs_model.get_market_total_assets(
            market_equity, total_liabilities
        )
        book_equity = (
            self._rows(balance, "Total Shareholder Equity")
            + self._rows(balance, "Deferred Tax Liabilities").fillna(0)
            - self._rows(balance, "Preferred Stock").fillna(0)
        )

        # NIMTA is a quarterly figure: a yearly net income counts as four equal quarters.
        net_income = self._rows(self._income_statement, "Net Income")
        quarterly_net_income = net_income if self._quarterly else net_income / 4
        net_income_to_market_total_assets = chs_model.get_winsorized_variable(
            self._calendar(quarterly_net_income / market_total_assets), "NIMTA"
        )
        net_income_average = (
            chs_model.get_geometric_average(
                [net_income_to_market_total_assets.T.shift(lag).T for lag in range(4)],
                chs_model.MONTHLY_DECAY**3,
            ).reindex(columns=periods)
            if self._quarterly
            else net_income_to_market_total_assets.reindex(columns=periods)
        )

        monthly_excess_returns = chs_model.get_monthly_log_excess_returns(
            adjusted_close,
            total_return_index.dropna(),
            dates,
        )
        excess_return_average = chs_model.get_geometric_average(
            [
                chs_model.get_winsorized_variable(excess_return, "EXRET")
                for excess_return in monthly_excess_returns
            ],
            chs_model.MONTHLY_DECAY,
        )

        index_value = helpers.values_on_or_before(
            pd.DataFrame(
                {ticker: index_market_value for ticker in tickers},
                index=index_market_value.index,
            ),
            dates,
        )

        variables = {
            "NIMTAAVG": net_income_average,
            "TLMTA": chs_model.get_winsorized_variable(
                total_liabilities / market_total_assets, "TLMTA"
            ),
            "EXRETAVG": excess_return_average,
            "SIGMA": chs_model.get_winsorized_variable(
                chs_model.get_sigma(daily_returns, dates), "SIGMA"
            ),
            "RSIZE": chs_model.get_winsorized_variable(
                chs_model.get_relative_size(market_equity, index_value), "RSIZE"
            ),
            "CASHMTA": chs_model.get_winsorized_variable(
                self._rows(balance, "Cash and Short Term Investments")
                / market_total_assets,
                "CASHMTA",
            ),
            "MB": chs_model.get_winsorized_variable(
                market_equity
                / chs_model.get_adjusted_book_equity(book_equity, market_equity),
                "MB",
            ),
            "PRICE": chs_model.get_winsorized_variable(
                chs_model.get_log_price(price), "PRICE"
            ),
        }

        chs_score = chs_model.get_chs_score(
            net_income_to_market_total_assets_average=variables["NIMTAAVG"],
            total_liabilities_to_market_total_assets=variables["TLMTA"],
            excess_return_average=variables["EXRETAVG"],
            sigma=variables["SIGMA"],
            relative_size=variables["RSIZE"],
            cash_to_market_total_assets=variables["CASHMTA"],
            market_to_book=variables["MB"],
            log_price=variables["PRICE"],
        )

        results = {
            "Net Income to Market Assets (NIMTAAVG)": variables["NIMTAAVG"],
            "Liabilities to Market Assets (TLMTA)": variables["TLMTA"],
            "Excess Return (EXRETAVG)": variables["EXRETAVG"],
            "Volatility (SIGMA)": variables["SIGMA"],
            "Relative Size (RSIZE)": variables["RSIZE"],
            "Cash to Market Assets (CASHMTA)": variables["CASHMTA"],
            "Market to Book (MB)": variables["MB"],
            "Log Price (PRICE)": variables["PRICE"],
            "CHS Score": chs_score,
            "Failure Probability": chs_model.get_chs_failure_probability(chs_score),
        }

        chs = self._combine(
            {name: frame.astype(float) for name, frame in results.items()}
        ).dropna(axis=1, how="all")
        chs = apply_rounding(chs, rounding if rounding is not None else self._rounding)

        return chs.loc[:, self._start_date : self._end_date]

    @handle_errors
    def get_synthetic_credit_rating(
        self,
        firm_type: str = "auto",
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate a synthetic credit rating and the cost of debt that comes with it, from the
        interest coverage ratio (EBIT divided by interest expense), following Aswath
        Damodaran. Most companies have no traded bonds or rating of their own; the coverage
        ratio is the measure that explains ratings best, so it is mapped to the rating that
        companies with a similar ratio have, and to that rating's default spread over the
        risk free rate:

            - Interest Coverage Ratio = EBIT / Interest Expense
            - Rating and Default Spread from Damodaran's synthetic rating table
            - Pre-tax Cost of Debt = Risk Free Rate + Default Spread

        The table, with this year's default spreads, is read from Damodaran's website. Large
        and small companies use different coverage ranges, as smaller companies need a
        higher coverage for the same rating.

        For more information, see the following book and data:

        - Damodaran, A. (2012). "Investment Valuation." Wiley, 3rd edition, chapter 8.
        - Damodaran, A. "Ratings, Interest Coverage Ratios and Default Spread."
          https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ratings.html

        Also known as: implied credit rating, interest coverage rating, cost of debt.

        Args:
            firm_type (str, optional): The table to use: "large" (market capitalization above
                five billion), "small" (smaller and riskier firms), "financial" (banks and
                other financial firms) or "auto", which picks large or small from the market
                capitalization of every period. Defaults to "auto".
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per ticker and period the Interest Coverage Ratio, Rating, Default
            Spread, Risk Free Rate and Pre-tax Cost of Debt.

        Notes:
        - A company that reports an interest expense of zero gets the best rating; its
          coverage ratio, which is infinite, shows as NaN. A period without a reported
          interest expense gets no rating.
        - A coverage ratio below the lowest bound of the table gets the worst rating.
        - The market capitalization is compared with five billion in the reporting
          currency; for companies that do not report in US dollars, pass the firm type.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["F", "AAPL", "T"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_synthetic_credit_rating()
        ```

        Which returns:

        |                                     | 2021     | 2022    | 2023    | 2024    | 2025    |
        |:------------------------------------|:---------|:--------|:--------|:--------|:--------|
        | ('AAPL', 'Interest Coverage Ratio') | 41.1905  | 40.7496 | 29.062  | nan     | nan     |
        | ('AAPL', 'Rating')                  | Aaa/AAA  | Aaa/AAA | Aaa/AAA | Aaa/AAA | Aaa/AAA |
        | ('AAPL', 'Default Spread')          | 0.004    | 0.004   | 0.004   | 0.004   | 0.004   |
        | ('AAPL', 'Risk Free Rate')          | 0.0144   | 0.0296  | 0.0397  | 0.0421  | 0.0429  |
        | ('AAPL', 'Pre-tax Cost of Debt')    | 0.0184   | 0.0336  | 0.0437  | 0.0461  | 0.0469  |
        | ('F', 'Interest Coverage Ratio')    | 2.7915   | 5.0905  | 4.1306  | 4.5695  | 2.0191  |
        | ('F', 'Rating')                     | Baa2/BBB | A2/A    | A3/A-   | A2/A    | Ba2/BB  |
        | ('F', 'Default Spread')             | 0.0111   | 0.0078  | 0.0089  | 0.0078  | 0.0184  |
        | ('F', 'Risk Free Rate')             | 0.0144   | 0.0296  | 0.0397  | 0.0421  | 0.0429  |
        | ('F', 'Pre-tax Cost of Debt')       | 0.0255   | 0.0374  | 0.0486  | 0.0499  | 0.0613  |
        | ('T', 'Interest Coverage Ratio')    | 3.8675   | -0.7472 | 3.7011  | 3.5969  | 3.6743  |
        | ('T', 'Rating')                     | A3/A-    | D2/D    | A3/A-   | A3/A-   | A3/A-   |
        | ('T', 'Default Spread')             | 0.0089   | 0.19    | 0.0089  | 0.0089  | 0.0089  |
        | ('T', 'Risk Free Rate')             | 0.0144   | 0.0296  | 0.0397  | 0.0421  | 0.0429  |
        | ('T', 'Pre-tax Cost of Debt')       | 0.0233   | 0.2196  | 0.0486  | 0.051   | 0.0518  |
        """
        firm_type = firm_type.lower()
        if firm_type not in ("auto", "large", "small", "financial"):
            raise ValueError(
                "The firm type must be 'auto', 'large', 'small' or 'financial'."
            )

        table = damodaran_model.get_synthetic_rating_table()
        if table.empty:
            raise ValueError(
                "Damodaran's synthetic rating table could not be retrieved."
            )

        ebit = self._rows(self._income_statement, "Operating Income")
        interest = self._rows(self._income_statement, "Interest Expense").abs()
        # Only a reported interest expense of zero means there is nothing to cover (an
        # infinite ratio, the best rating); a missing one leaves the ratio unknown.
        coverage = (
            (ebit / interest.where(interest > 0))
            .mask(interest == 0, float("inf"))
            .where(ebit.notna())
        )

        market_cap = self._period_prices() * self._rows(
            self._income_statement, "Weighted Average Shares Diluted"
        )

        risk_free = self._period_risk_free_rate(coverage.columns)

        rows = {}
        for ticker in self._tickers:
            for period in coverage.columns:
                ratio = coverage.loc[ticker, period]
                if pd.isna(ratio):
                    continue
                kind = firm_type
                if kind == "auto":
                    cap = market_cap.loc[ticker, period]
                    kind = (
                        "large"
                        if pd.notna(cap) and cap > credit_model.LARGE_FIRM_MARKET_CAP
                        else "small"
                    )
                rating, spread = credit_model.get_synthetic_rating(
                    float(ratio), table[table["Firm Type"] == kind.capitalize()]
                )
                rate = float(risk_free.get(period, float("nan")))
                rows[(ticker, period)] = {
                    "Interest Coverage Ratio": (
                        float(ratio) if ratio != float("inf") else float("nan")
                    ),
                    "Rating": rating,
                    "Default Spread": spread,
                    "Risk Free Rate": rate,
                    "Pre-tax Cost of Debt": rate + spread,
                }

        if not rows:
            raise ValueError("No interest coverage ratio could be calculated.")

        long = pd.DataFrame(rows).T
        numeric = [
            "Interest Coverage Ratio",
            "Default Spread",
            "Risk Free Rate",
            "Pre-tax Cost of Debt",
        ]
        long[numeric] = apply_rounding(
            long[numeric].astype(float),
            rounding if rounding is not None else self._rounding,
        )
        result = long.stack().unstack(level=1)
        result.index.names = [None, None]
        order = {name: position for position, name in enumerate(long.columns)}
        result = result.sort_index(
            level=1, key=lambda names: names.map(order), sort_remaining=False
        ).reindex(self._tickers, level=0)

        return result.loc[:, self._start_date : self._end_date]
