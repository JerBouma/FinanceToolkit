"""Example: enrich FinanceToolkit historical data with Adanos sentiment."""

import os

from financetoolkit import Toolkit
from financetoolkit.utilities.external_dataset_model import (
    combine_external_dataset,
    get_adanos_sentiment,
)

ADANOS_API_KEY = os.getenv("ADANOS_API_KEY", "")


def main():
    """Fetch Adanos sentiment and merge it into a FinanceToolkit history frame."""
    toolkit = Toolkit(
        tickers=["AAPL", "MSFT", "NVDA"],
        benchmark_ticker="SPY",
        start_date="2024-01-01",
    )

    historical_data = toolkit.get_historical_data()

    reddit_sentiment = get_adanos_sentiment(
        ["AAPL", "MSFT", "NVDA"],
        api_key=ADANOS_API_KEY,
        source="reddit",
        days=30,
    )

    sentiment_enriched_history = combine_external_dataset(
        historical_data,
        reddit_sentiment,
        dataset_name="Adanos Reddit Sentiment",
    )

    print(sentiment_enriched_history.tail())


if __name__ == "__main__":
    main()
