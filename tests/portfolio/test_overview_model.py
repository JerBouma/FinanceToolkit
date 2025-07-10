"""Portfolio Overview Model Tests"""

# ruff: noqa: PLR2004


import pandas as pd
import pytest

from financetoolkit.portfolio.overview_model import (
    create_portfolio_overview,
    create_profit_and_loss_overview,
)


@pytest.fixture
def sample_portfolio_data():
    """Sample portfolio data for testing"""
    dates = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
    tickers = ["AAPL", "MSFT", "AMZN"]

    # Create MultiIndex for (Date, Ticker)
    index = pd.MultiIndex.from_product([dates, tickers], names=["Date", "Ticker"])

    return pd.DataFrame(
        {
            "Volume": [100, 50, 200, 0, 25, 100, 50, 0, 150],
            "Price": [150.0, 250.0, 100.0, 0.0, 260.0, 95.0, 155.0, 0.0, 98.0],
            "Costs": [1.0, 2.0, 3.0, 0.0, 1.5, 2.5, 1.0, 0.0, 2.0],
            "Name": ["Apple Inc"] * 3 + ["Microsoft Corp"] * 3 + ["Amazon.com Inc"] * 3,
        },
        index=index,
    )


@pytest.fixture
def sample_latest_returns():
    """Sample latest returns data"""
    return pd.Series([160.0, 255.0, 110.0], index=["AAPL", "MSFT", "AMZN"])


@pytest.fixture
def sample_benchmark_data():
    """Sample benchmark data"""
    return pd.Series([500.0, 510.0, 520.0], index=["AAPL", "MSFT", "AMZN"])


@pytest.fixture
def sample_volatilities():
    """Sample volatilities data"""
    return pd.Series(
        [0.25, 0.30, 0.35, 0.20], index=["AAPL", "MSFT", "AMZN", "Benchmark"]
    )


@pytest.fixture
def sample_betas():
    """Sample betas data"""
    return pd.Series([1.2, 0.8, 1.5], index=["AAPL", "MSFT", "AMZN"])


@pytest.mark.filterwarnings(
    "ignore:DataFrameGroupBy.apply operated on the grouping columns:DeprecationWarning"
)
def test_create_profit_and_loss_overview():
    """Test create_profit_and_loss_overview function"""
    transactions_overview = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            "Ticker": ["AAPL", "AAPL", "AAPL"],
            "Volume": [100, -50, 25],  # Buy, sell, buy
            "Price": [150.0, 160.0, 155.0],
            "Costs": [1.0, 2.0, 1.5],
        }
    )

    result = create_profit_and_loss_overview(
        transactions_overview=transactions_overview,
        ticker_column="Ticker",
        volume_column="Volume",
        price_column="Price",
        method="FIFO",
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert "PnL" in result.columns
    assert "Cumulative PnL" in result.columns


@pytest.mark.filterwarnings(
    "ignore:DataFrameGroupBy.apply operated on the grouping columns:DeprecationWarning"
)
def test_create_profit_and_loss_overview_lifo():
    """Test create_profit_and_loss_overview with LIFO method"""
    transactions_overview = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            "Ticker": ["AAPL", "AAPL", "AAPL"],
            "Volume": [100, -50, 25],
            "Price": [150.0, 160.0, 155.0],
            "Costs": [1.0, 2.0, 1.5],
        }
    )

    result = create_profit_and_loss_overview(
        transactions_overview=transactions_overview,
        ticker_column="Ticker",
        volume_column="Volume",
        price_column="Price",
        method="LIFO",
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert "PnL" in result.columns
    assert "Cumulative PnL" in result.columns


@pytest.mark.filterwarnings(
    "ignore:DataFrameGroupBy.apply operated on the grouping columns:DeprecationWarning"
)
def test_create_profit_and_loss_overview_average():
    """Test create_profit_and_loss_overview with AVERAGE method"""
    transactions_overview = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            "Ticker": ["AAPL", "AAPL", "AAPL"],
            "Volume": [100, -50, 25],
            "Price": [150.0, 160.0, 155.0],
            "Costs": [1.0, 2.0, 1.5],
        }
    )

    result = create_profit_and_loss_overview(
        transactions_overview=transactions_overview,
        ticker_column="Ticker",
        volume_column="Volume",
        price_column="Price",
        method="AVERAGE",
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert "PnL" in result.columns
    assert "Cumulative PnL" in result.columns


def test_create_portfolio_overview_with_missing_data():
    """Test create_portfolio_overview with missing data"""
    portfolio_name = pd.Series(["Apple Inc", "Microsoft Corp"], index=["AAPL", "MSFT"])
    portfolio_volume = pd.Series([100, 50], index=["AAPL", "MSFT"])
    portfolio_price = pd.Series([150.0, 250.0], index=["AAPL", "MSFT"])
    portfolio_costs = pd.Series([1.0, 2.0], index=["AAPL", "MSFT"])

    # Missing latest returns for MSFT
    latest_returns = pd.Series([160.0], index=["AAPL"])

    benchmark_prices = pd.Series([500.0, 510.0], index=["AAPL", "MSFT"])
    benchmark_latest_prices = pd.Series([520.0, 530.0], index=["AAPL", "MSFT"])
    volatilities = pd.Series([0.25, 0.30, 0.20], index=["AAPL", "MSFT", "Benchmark"])
    betas = pd.Series([1.2, 0.8], index=["AAPL", "MSFT"])

    with pytest.raises(KeyError):
        create_portfolio_overview(
            portfolio_name=portfolio_name,
            portfolio_volume=portfolio_volume,
            portfolio_price=portfolio_price,
            portfolio_costs=portfolio_costs,
            latest_returns=latest_returns,
            benchmark_prices=benchmark_prices,
            benchmark_latest_prices=benchmark_latest_prices,
            volatilities=volatilities,
            betas=betas,
            include_portfolio=True,
        )
