"""Models Module"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.models.distress.distress_controller import DistressModels
from financetoolkit.models.profitability.profitability_controller import (
    ProfitabilityModels,
)
from financetoolkit.models.quality.quality_controller import QualityModels
from financetoolkit.models.valuation.valuation_controller import ValuationModels
from financetoolkit.utilities import factor_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import (
    apply_rounding,
)

# ruff: noqa: E501

# The models per collection that get_factor_analysis compares across tickers: those with one
# value per ticker and statement period. The valuation models (intrinsic value, dividend
# discount, residual income) give a currency amount whose level reflects the share price, and
# the analyses (DuPont, WACC, enterprise value breakdown) a set of components without one
# score, so they are not in the collections.
FACTOR_COLLECTIONS = {
    "scores": [
        "get_piotroski_score",
        "get_altman_z_score",
        "get_beneish_m_score",
        "get_ohlson_o_score",
        "get_zmijewski_score",
        "get_springate_score",
        "get_grover_score",
        "get_fulmer_h_score",
        "get_taffler_z_score",
        "get_campbell_hilscher_szilagyi_score",
        "get_dechow_f_score",
        "get_montier_c_score",
        "get_mohanram_g_score",
    ],
}

FACTOR_COLLECTIONS["all"] = FACTOR_COLLECTIONS["scores"] + [
    "get_accruals_ratio",
    "get_tobins_q_ratio",
    "get_sustainable_growth_rate",
    "get_internal_growth_rate",
    "get_magic_formula",
    "get_merton_distance_to_default",
]


@filter_shown_columns
@track_sources
class Models(ValuationModels, QualityModels, DistressModels, ProfitabilityModels):
    """
    The Models module is meant to execute well-known models such
    as DUPONT and the Discounted Cash Flow (DCF) model. These models
    are also directly related to the data retrieved from the Toolkit module.

    """

    @handle_errors
    def get_factor_analysis(
        self,
        metrics: str | list[str] = "scores",
        period: str = "monthly",
        horizons: list[int] | int | None = None,
        quantiles: int = 5,
        periods_apart: int = 1,
        detail: str | None = None,
        column: str | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Factor analysis (predictive power) of the models: whether the tickers that score high
        on a model, such as the Piotroski F-Score or the Altman Z-Score, go on to return more
        than those that score low. Per period, the information coefficient (IC) is the
        Spearman rank correlation between the model score of the tickers at the end of the
        period and their return over the following horizon: 1 means the score ranks the later
        returns perfectly, -1 in reverse and 0 that it says nothing about them. The IC measures
        skill in Grinold and Kahn's fundamental law of active management,
        IR ≈ IC × √breadth: a small but consistent IC across many tickers and periods gives a
        high information ratio.

        The summary has a row per model and horizon with the mean IC, its standard deviation,
        the IC information ratio (mean divided by standard deviation), the t-statistic of the
        mean with its p-value, the share of periods with a positive IC, the number of periods,
        the top minus bottom spread (the mean forward return of the highest quantile minus
        that of the lowest), the mean turnover of the top and bottom quantiles and the mean
        rank autocorrelation of the score from one period to the next. With detail, it returns
        one of those parts per period (or per quantile) for a single model.

        The models are calculated per statement period and each period's score applies from
        the end of the fiscal period, or with point_in_time from the day it was filed, never
        earlier, so no day uses figures that were not known yet. The returns use the dividend
        and split adjusted close and start at the close of the period the score belongs to.
        Each ticker keeps its own history: a ticker without a score or a return in a period is
        left out of that period only.

        Args:
            metrics (str | list[str], optional): A model method name with one value per
                ticker and period, e.g. "get_piotroski_score", a list of them, or a
                collection: "scores" or "all". Defaults to "scores", the Piotroski, Altman Z,
                Beneish M, Ohlson O, Zmijewski, Springate, Grover, Fulmer H, Taffler z,
                Dechow F, Montier C and Mohanram G scores. "all" adds the accruals ratio, Tobin's Q, the
                sustainable and internal growth rates, the magic formula and the Merton
                distance to default. The valuation models and the analyses with several
                components and no score are not in the collections.
            period (str, optional): The frequency at which the models are evaluated: daily,
                weekly, monthly, quarterly or yearly. Defaults to "monthly".
            horizons (list[int] | int | None, optional): The number of periods the forward
                returns cover. Defaults to None, which uses about a period, a quarter and a
                year ahead (1, 3 and 12 months for monthly).
            quantiles (int, optional): The number of quantiles the tickers are sorted into per
                period. Defaults to 5.
            periods_apart (int, optional): The number of periods between the dates the
                quantile turnover and the rank autocorrelation compare. Defaults to 1.
            detail (str | None, optional): Returns one part of the analysis of a single model
                instead of the summary: "information_coefficient" (per period and horizon),
                "quantile_returns" (the mean forward return per quantile and horizon),
                "quantile_turnover" (per period and quantile) or "rank_autocorrelation" (per
                period). Defaults to None.
            column (str | None, optional): The component to use of a model, e.g.
                "Working Capital to Total Assets" of the Altman Z-Score. Defaults to None,
                which uses the model's score (its last row).
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            **kwargs: Parameters passed to the models that take them, e.g. trailing=4 or
                diluted=False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The summary per model and horizon, or with detail that part of the
            analysis of the model.

        Notes:
            - The IC is reported as measured: for a distress score where a higher value means
              more risk (Ohlson O, Zmijewski, Beneish M), a negative IC means the riskier
              companies went on to return less. Read the sign as the direction of the
              relationship.
            - Scores that take few values (the Piotroski F-Score from 0 to 9) tie many
              tickers; ties are split over adjacent quantiles by the order of the tickers.
            - The cross-section is the tickers of the Toolkit, so use at least ten and
              preferably twenty or more for a meaningful rank correlation and quantiles.
            - A horizon longer than one period makes consecutive returns overlap, which
              inflates the t-statistic; judge those horizons by their mean IC and IC
              information ratio.
            - Based on alphalens (Quantopian) and on Grinold and Kahn (2000), Active Portfolio
              Management.

        Also known as: predictive power, information coefficient, factor testing, alpha signal

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "JNJ",
             "XOM", "PG", "KO", "WMT", "V", "UNH", "HD", "CVX"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2016-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_factor_analysis(
            ["get_piotroski_score", "get_altman_z_score"], horizons=12
        )
        ```

        Which returns:
        |                                      |   Mean IC |   IC Standard Deviation |   IC Information Ratio |   t-Statistic |   p-Value |   Positive Share |   Observations |   Top Minus Bottom |   Quantile Turnover |   Rank Autocorrelation |
        |:-------------------------------------|----------:|------------------------:|-----------------------:|--------------:|----------:|-----------------:|---------------:|-------------------:|--------------------:|-----------------------:|
        | ('get_piotroski_score', '12 Months') |   -0.1352 |                  0.3685 |                -0.3669 |       -3.6693 |    0.0004 |             0.31 |            100 |            -0.1482 |              0.0787 |                 0.93   |
        | ('get_altman_z_score', '12 Months')  |    0.1209 |                  0.3923 |                 0.3082 |        3.0824 |    0.0027 |             0.7  |            100 |             0.1264 |              0.0285 |                 0.9941 |
        """
        factors = factor_model.get_daily_factors(
            self,
            factor_model.select_metrics(metrics, FACTOR_COLLECTIONS),
            kwargs,
            {},
            self._tickers,
            column,
            self._period_end_dates,
        )
        result = factor_model.get_factor_analysis(
            factors,
            self._historical_data["daily"]["Adj Close"],
            period,
            horizons,
            quantiles,
            periods_apart,
            detail,
            self._start_date,
            self._end_date,
        )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )
