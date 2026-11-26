"""Statements Model: the statement figures SRISK measures the capital shortfall with."""

from collections.abc import Callable

import pandas as pd

from financetoolkit.utilities import logger_model

logger = logger_model.get_logger()


def get_capital_shortfall_data(
    balance_sheet: pd.DataFrame,
    income_statement: pd.DataFrame,
    start_dates: Callable[[pd.DataFrame], pd.DataFrame],
    statistics_statement: pd.DataFrame,
    listing_currencies: Callable[[], pd.Series],
    converted: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Select the liabilities and the diluted share count SRISK takes from the statements, each
    with the days its period's figures apply from, as get_market_cap does.

    Without a currency conversion the liabilities stay in the reported currency while the
    market cap is in the currency the ticker is listed in, so SRISK would mix the two; the
    tickers where they differ are reported with a warning.

    Args:
        balance_sheet (pd.DataFrame): The balance sheet statement.
        income_statement (pd.DataFrame): The income statement.
        start_dates (Callable[[pd.DataFrame], pd.DataFrame]): Returns the day each period of a
            statement applies from.
        statistics_statement (pd.DataFrame): The statistics statement, which holds the
            reported currency.
        listing_currencies (Callable[[], pd.Series]): Returns the currency each ticker is
            listed in, called only to compare it with the reported currency.
        converted (bool): Whether the statements are converted to the listing currency.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: The total liabilities,
            the days they apply from, the diluted share count and the days it applies from.
    """
    if not converted and not statistics_statement.empty:
        reported = (
            statistics_statement.xs("Reported Currency", axis=0, level=1)
            .T.ffill()
            .iloc[-1]
        )
        listed = listing_currencies().reindex(reported.index)
        mismatched = [
            f"{ticker} ({reported[ticker]} statements, {listed[ticker]} prices)"
            for ticker in reported.index
            if pd.notna(reported[ticker])
            and pd.notna(listed[ticker])
            and reported[ticker] != listed[ticker]
        ]
        if mismatched:
            logger.warning(
                "SRISK compares the liabilities with the market cap, but the statements "
                "are in another currency than the prices for %s; set convert_currency=True "
                "to convert the statements to the currency of the historical data.",
                ", ".join(mismatched),
            )

    return (
        balance_sheet.loc[:, "Total Liabilities", :],
        start_dates(balance_sheet),
        income_statement.loc[:, "Weighted Average Shares Diluted", :],
        start_dates(income_statement),
    )
