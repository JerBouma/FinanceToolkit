"""Peers Model: the valuation, profitability and growth figures of a company and its peers."""

from datetime import datetime

import pandas as pd
from dateutil.relativedelta import relativedelta

from financetoolkit.cache.cache_controller import Cache
from financetoolkit.fmp_model import get_stock_peers


def compare_with_peers(
    peers: dict[str, list[str]] | None,
    tickers: list[str],
    api_key: str,
    user_subscription: str,
    end_date: str,
    quarterly: bool,
    point_in_time: bool,
    convert_currency: bool,
    cache: Cache,
    enforce_source: str | None,
    sleep_timer: bool,
) -> pd.DataFrame:
    """
    Compare the latest valuation, profitability and growth figures of each ticker with those
    of its peers.

    The figures come from a Toolkit of the tickers and all their peers with the settings that
    change them (period, point in time, currency conversion, source), so the Company column
    equals what the Toolkit of the tickers shows for that period.

    Args:
        peers (dict[str, list[str]] | None): The peers per ticker; a ticker without them gets
            the peers Financial Modeling Prep lists.
        tickers (list[str]): The tickers to compare.
        api_key (str): The Financial Modeling Prep API key.
        user_subscription (str): The Financial Modeling Prep plan.
        end_date (str): The end date of the comparison (YYYY-MM-DD).
        quarterly (bool): Whether to compare quarterly figures.
        point_in_time (bool): Whether only the statements filed by each date are used.
        convert_currency (bool): Whether the statements are converted to the currency of the prices.
        cache (Cache): The cache the Toolkit of the peers reads and writes.
        enforce_source (str | None): The data source to use, None for the default order.
        sleep_timer (bool): Whether to wait when the API limit is reached.

    Returns:
        pd.DataFrame: The peer comparison of each ticker, keyed by ticker.
    """
    # Imported when first used: the Toolkit module imports this one, and only the peer
    # comparison needs the model.
    from financetoolkit.models.valuation import peers_model  # noqa: PLC0415
    from financetoolkit.toolkit_controller import Toolkit  # noqa: PLC0415

    # The tickers are upper case, also when the peers are given per lower case ticker.
    given_peers = {key.strip().upper(): value for key, value in (peers or {}).items()}
    peer_map = {
        ticker: list(given_peers.get(ticker) or [])
        or get_stock_peers(ticker, api_key, user_subscription)
        for ticker in tickers
    }
    universe = sorted(
        set(tickers) | {peer for group in peer_map.values() for peer in group}
    )
    # The latest period up to the end date is what is compared, so three years before it
    # suffice (the Toolkit adds the lookback that growth and trailing figures need). It is
    # unrounded, as the comparison is calculated (and rounded) by the caller.
    others = Toolkit(
        tickers=universe,
        api_key=api_key,
        start_date=(
            datetime.strptime(end_date, "%Y-%m-%d") - relativedelta(years=3)
        ).strftime("%Y-%m-%d"),
        end_date=end_date,
        quarterly=quarterly,
        point_in_time=point_in_time,
        convert_currency=convert_currency,
        use_cached_data=cache,
        enforce_source=enforce_source,
        sleep_timer=sleep_timer,
        benchmark_ticker=None,
        progress_bar=False,
        rounding=None,
    )
    ratios = others.ratios
    # Quarterly growth is over a year, as the growth of a quarter on the quarter before it
    # mixes in the seasons.
    income = others.get_income_statement(growth=True, lag=4 if quarterly else 1)
    metrics = pd.DataFrame(
        {
            "Price-to-Earnings": peers_model.get_latest_values(
                ratios.get_price_to_earnings_ratio()
            ),
            "EV-to-EBITDA": peers_model.get_latest_values(
                ratios.get_ev_to_ebitda_ratio()
            ),
            "Price-to-Book": peers_model.get_latest_values(
                ratios.get_price_to_book_ratio()
            ),
            "Price-to-Sales": peers_model.get_latest_values(
                ratios.get_price_to_sales_ratio()
            ),
            "Gross Margin": peers_model.get_latest_values(ratios.get_gross_margin()),
            "Operating Margin": peers_model.get_latest_values(
                ratios.get_operating_margin()
            ),
            "Net Profit Margin": peers_model.get_latest_values(
                ratios.get_net_profit_margin()
            ),
            "Return on Equity": peers_model.get_latest_values(
                ratios.get_return_on_equity()
            ),
            "Return on Invested Capital": peers_model.get_latest_values(
                ratios.get_return_on_invested_capital()
            ),
            "Revenue Growth": peers_model.get_latest_values(
                income.loc[:, "Revenue", :]
            ),
            "Debt-to-Equity": peers_model.get_latest_values(
                ratios.get_debt_to_equity_ratio()
            ),
            "Dividend Yield": peers_model.get_latest_values(
                ratios.get_dividend_yield()
            ),
        }
    )
    return pd.concat(
        {
            ticker: peers_model.get_peer_comparison(metrics, ticker, peer_map[ticker])
            for ticker in tickers
        }
    )
