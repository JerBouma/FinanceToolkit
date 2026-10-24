"""Portfolio Overview Model Tests"""

# ruff: noqa: PLR2004


import numpy as np
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


def _dividend_paying_stock():
    """A stock that closes at 100 every day and pays two dividends of 5, with the back-adjusted close."""
    dates = pd.period_range("2022-01-03", periods=500, freq="D")
    close = pd.Series(100.0, index=dates)
    dividends = pd.Series(0.0, index=dates)
    dividends.iloc[[200, 400]] = 5.0
    factor = pd.Series(1.0, index=dates)
    for location in [200, 400]:
        factor.iloc[:location] *= 1 - 5.0 / 100.0
    historical = pd.DataFrame(
        {
            ("Close", "X"): close,
            ("Adj Close", "X"): close * factor,
            ("Dividends", "X"): dividends,
        }
    )
    historical.columns = pd.MultiIndex.from_tuples(historical.columns)
    return dates, historical


def test_portfolio_overview_adds_dividends_to_a_position_at_the_traded_close():
    """A flat stock bought at its close returns its dividends, as does the same stock as benchmark."""
    dates, _ = _dividend_paying_stock()
    index = pd.MultiIndex.from_tuples([(dates[0], "X")], names=["Date", "Identifier"])

    overview = create_portfolio_overview(
        portfolio_name=pd.Series(["X"], index=index),
        portfolio_volume=pd.Series([1.0], index=index),
        portfolio_price=pd.Series([100.0], index=index),
        portfolio_costs=pd.Series([0.0], index=index),
        latest_returns=pd.Series({"X": 100.0}),
        benchmark_prices=pd.Series([100.0], index=index),
        benchmark_latest_prices=pd.Series({"X": 100.0}),
        volatilities=pd.Series({"X": 0.0, "Benchmark": 0.0}),
        betas=pd.Series({"X": 1.0}),
        include_portfolio=True,
        dividends=pd.Series([10.0], index=index),
        benchmark_dividends=pd.Series([10.0], index=index),
    )

    assert overview.loc["X", "Latest Value"] == pytest.approx(100.0)
    assert overview.loc["X", "Dividends"] == pytest.approx(10.0)
    assert overview.loc["X", "Return"] == pytest.approx(0.10)
    assert overview.loc["X", "Benchmark Return"] == pytest.approx(0.10)
    assert overview.loc["X", "Alpha"] == pytest.approx(0.0)
    assert overview.loc["Portfolio", "Return"] == pytest.approx(0.10)


def test_positions_overview_is_a_total_return_from_the_first_day():
    """No loss on the day of purchase; each dividend is received by the units held the day before."""
    from financetoolkit.portfolio.overview_model import (  # noqa: PLC0415
        create_positions_overview,
    )

    dates, historical = _dividend_paying_stock()
    index = pd.MultiIndex.from_tuples(
        [(dates[0], "X"), (dates[400], "X")], names=["Date", "Identifier"]
    )
    transactions = pd.DataFrame(
        {"Volume": [1.0, 1.0], "Price": [100.0, 100.0], "Costs": [0.0, 0.0]},
        index=index,
    )

    positions = create_positions_overview(["X"], dates, transactions, historical)

    assert positions[("Cumulative Return", "X")].iloc[0] == pytest.approx(0.0)
    assert positions[("Current Value", "X")].iloc[0] == pytest.approx(100.0)
    # The unit bought on the second ex-date does not receive that dividend.
    assert positions[("Dividends", "X")].iloc[-1] == pytest.approx(10.0)
    assert positions[("Cumulative Return", "X")].iloc[-1] == pytest.approx(0.05)


def test_positions_overview_return_is_missing_before_the_first_purchase():
    """The return is undefined before any capital was deployed, the other columns are zero."""
    from financetoolkit.portfolio.overview_model import (  # noqa: PLC0415
        create_positions_overview,
    )

    dates, historical = _dividend_paying_stock()
    index = pd.MultiIndex.from_tuples([(dates[10], "X")], names=["Date", "Identifier"])
    transactions = pd.DataFrame(
        {"Volume": [1.0], "Price": [100.0], "Costs": [0.0]}, index=index
    )

    positions = create_positions_overview(["X"], dates, transactions, historical)

    assert positions[("Cumulative Return", "X")].iloc[:10].isna().all()
    assert (positions[("Invested Amount", "X")].iloc[:10] == 0).all()
    assert positions[("Cumulative Return", "X")].iloc[10] == pytest.approx(0.0)


def test_transactions_performance_follows_the_units_bought_in_the_period():
    """A sale of units bought in an earlier period does not enter the return of this period."""
    from financetoolkit.portfolio.overview_model import (  # noqa: PLC0415
        create_transactions_performance,
    )

    dates = pd.PeriodIndex(["2022-06-01", "2023-03-01", "2023-06-01"], freq="D")
    index = pd.MultiIndex.from_arrays(
        [dates, ["X", "X", "X"]], names=["Date", "Identifier"]
    )
    transactions = pd.DataFrame(
        {"Volume": [10.0, 1.0, -10.0], "Price": [50.0, 100.0, 100.0], "Costs": 0.0},
        index=index,
    )
    year_end = pd.DataFrame(
        {"X": [100.0, 110.0]}, index=pd.PeriodIndex(["2022", "2023"], freq="Y")
    )
    cumulative_dividends = pd.DataFrame(
        {"X": [0.0, 2.0]}, index=pd.PeriodIndex(["2022", "2023"], freq="Y")
    )

    performance = create_transactions_performance(
        transactions,
        "Identifier",
        "Date",
        "Volume",
        "Price",
        "Costs",
        period_prices=year_end,
        period_string="Y",
        benchmark_specific_prices=pd.Series([50.0, 100.0, 100.0], index=index),
        benchmark_period_prices=year_end["X"],
        period_dividends=cumulative_dividends,
        transaction_dividends=pd.Series([0.0, 0.0, 2.0], index=index),
        benchmark_period_dividends=cumulative_dividends["X"],
        benchmark_transaction_dividends=pd.Series([0.0, 0.0, 2.0], index=index),
    )

    row = performance.loc[(pd.Period("2023", freq="Y"), "X")]
    # One unit bought at 100, worth 110 at the end of 2023 plus a dividend of 2.
    assert row["Invested Amount"] == pytest.approx(100.0)
    assert row["Current Value"] == pytest.approx(110.0)
    assert row["Dividends"] == pytest.approx(2.0)
    assert row["Realized Proceeds"] == pytest.approx(1000.0)
    assert row["Return"] == pytest.approx(0.12)
    assert row["Alpha"] == pytest.approx(0.0)


def test_trailing_volatility_and_beta_use_the_last_252_trading_days():
    """A volatile first year does not count once 252 calmer days have passed."""
    from financetoolkit.portfolio.overview_model import (  # noqa: PLC0415
        calculate_trailing_beta,
        calculate_trailing_volatility,
    )

    dates = pd.period_range("2023-01-02", periods=600, freq="D")
    rng = np.random.default_rng(0)
    benchmark = pd.Series(rng.normal(0, 0.01, 600), index=dates)
    asset = pd.DataFrame({"X": benchmark * 3.0})
    asset.iloc[-252:, 0] = benchmark.iloc[-252:] * 0.5

    volatility = calculate_trailing_volatility(asset)
    expected = benchmark.iloc[-252:].mul(0.5).std() * np.sqrt(252)

    assert volatility["X"] == pytest.approx(expected)
    assert calculate_trailing_beta(asset, benchmark)["X"] == pytest.approx(0.5)
