"""Factor Analysis Tests: get_factor_analysis of the ratios, technicals, models and performance"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.models.models_controller import Models
from financetoolkit.performance.performance_controller import Performance
from financetoolkit.ratios import ratios_controller
from financetoolkit.ratios.ratios_controller import Ratios
from financetoolkit.technicals.technicals_controller import Technicals

TICKERS = [f"T{number}" for number in range(10)]


def daily_prices(seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", "2022-12-31")
    returns = rng.normal(0, 0.01, (len(dates), len(TICKERS) + 1))
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(returns, axis=0)),
        index=pd.PeriodIndex(dates, freq="D"),
        columns=[*TICKERS, "Benchmark"],
    )
    return pd.concat({"Adj Close": prices}, axis=1)


def future_monthly_return_factor():
    """A factor that knows each ticker's return over the next month: an IC of one."""
    prices = daily_prices()["Adj Close"][TICKERS]
    prices.index = prices.index.to_timestamp()
    month_end = prices.groupby(prices.index.to_period("M")).last()
    forward = month_end.shift(-1) / month_end - 1
    # Spread each month's value over its days, so the month-end sample picks it up.
    factor = forward.reindex(prices.index.to_period("M"))
    factor.index = prices.index
    return factor


class FakeModule:
    """The attributes get_factor_analysis reads; each test adds the metrics it needs."""

    _tickers = [*TICKERS, "Benchmark"]
    _historical_data = {"daily": daily_prices()}
    _daily_historical_data = daily_prices()
    _period_end_dates = None
    _rounding = None
    _start_date = "2020-01-01"
    _end_date = "2022-12-31"


class FakeRatios(FakeModule):
    get_factor_analysis = Ratios.get_factor_analysis

    def __init__(self):
        self.calls = []

    def get_price_to_earnings_ratio(
        self, show_daily=False, trailing=None, rounding=None
    ):
        self.calls.append(("get_price_to_earnings_ratio", show_daily, trailing))
        return future_monthly_return_factor()

    def get_price_to_book_ratio(self, show_daily=False, rounding=None):
        self.calls.append(("get_price_to_book_ratio", show_daily, None))
        return -future_monthly_return_factor()


class FakeTechnicals(FakeModule):
    get_factor_analysis = Technicals.get_factor_analysis

    def get_stochastic_oscillator(self, window=14, rounding=None):
        perfect = future_monthly_return_factor()
        noise = pd.DataFrame(
            np.random.default_rng(window).normal(size=perfect.shape),
            index=perfect.index,
            columns=perfect.columns,
        )
        values = pd.concat({"Stochastic %D": noise, "Stochastic %K": perfect}, axis=1)
        values.index = pd.PeriodIndex(values.index, freq="D")
        return values


def test_a_perfect_factor_of_your_own_has_an_ic_of_one_and_leaves_the_benchmark_out():
    performance = Performance(
        tickers=[*TICKERS, "Benchmark"],
        historical_data={"daily": daily_prices()},
        risk_free_rate_data={},
        rounding=None,
        start_date="2020-01-01",
        end_date="2022-12-31",
    )
    factor = future_monthly_return_factor()
    factor["Benchmark"] = 1.0

    summary = performance.get_factor_analysis(factor, horizons=1)
    ic = performance.get_factor_analysis(
        factor, horizons=1, detail="information_coefficient"
    )
    quantile_returns = performance.get_factor_analysis(
        factor, horizons=1, detail="quantile_returns"
    )

    assert list(summary.index) == [("Custom Factor", "1 Month")]
    assert summary.loc[("Custom Factor", "1 Month"), "Mean IC"] == 1
    assert np.allclose(ic["1 Month"], 1)
    assert quantile_returns["1 Month"].drop("Top Minus Bottom").is_monotonic_increasing


def test_the_ratios_default_to_the_valuation_collection_calculated_daily(monkeypatch):
    monkeypatch.setitem(
        ratios_controller.FACTOR_COLLECTIONS,
        "valuation",
        ["get_price_to_earnings_ratio", "get_price_to_book_ratio"],
    )
    ratios = FakeRatios()

    summary = ratios.get_factor_analysis(horizons=1, trailing=4)

    assert ratios.calls == [
        ("get_price_to_earnings_ratio", True, 4),
        ("get_price_to_book_ratio", True, None),
    ]
    # The IC is reported as measured: the reversed factor has an IC of minus one.
    assert summary["Mean IC"].tolist() == [1, -1]


def test_statement_figures_apply_from_the_period_end_only():
    # A yearly figure per ticker: the 2021 figure may only be used after its period end.
    periods = pd.PeriodIndex(["2020", "2021"], freq="Y")
    ratios = FakeRatios()
    ratios._period_end_dates = pd.DataFrame(
        [[pd.Timestamp("2020-12-31"), pd.Timestamp("2021-12-31")]] * 10,
        index=TICKERS,
        columns=periods,
    )
    ratios.get_return_on_equity = lambda rounding=None: pd.DataFrame(
        [[float(number), float(-number)] for number in range(10)],
        index=TICKERS,
        columns=periods,
    )

    autocorrelation = ratios.get_factor_analysis(
        "get_return_on_equity", detail="rank_autocorrelation"
    )

    # The ranking reverses once, in December 2021 when the 2021 figures become known, and
    # not at the start of 2021.
    assert autocorrelation.loc["2021-12", "Rank Autocorrelation"] == -1
    assert (autocorrelation.drop(pd.Period("2021-12", freq="M")) == 1).all().all()
    assert autocorrelation.index[0] == pd.Period("2021-01", freq="M")


def test_column_selects_the_output_of_an_indicator_with_several():
    technicals = FakeTechnicals()

    chosen = technicals.get_factor_analysis(
        "get_stochastic_oscillator", horizons=1, column="Stochastic %K"
    )
    first = technicals.get_factor_analysis("get_stochastic_oscillator", horizons=1)

    assert chosen["Mean IC"].iloc[0] == 1
    assert abs(first["Mean IC"].iloc[0]) < 0.2  # noqa: PLR2004


def test_a_model_gives_its_score_unless_column_names_a_component():
    periods = pd.PeriodIndex(["2020", "2021"], freq="Y")
    rows = pd.MultiIndex.from_product([TICKERS, ["Component", "Score"]])
    values = pd.DataFrame(
        [
            [float(number)] * 2 if label == "Score" else [float(-number), float(number)]
            for number, label in zip(np.repeat(range(10), 2), rows.get_level_values(1))
        ],
        index=rows,
        columns=periods,
    )
    models = FakeModule()
    models.get_piotroski_score = lambda rounding=None: values

    score = Models.get_factor_analysis(
        models, "get_piotroski_score", detail="rank_autocorrelation"
    )
    component = Models.get_factor_analysis(
        models, "get_piotroski_score", column="Component", detail="rank_autocorrelation"
    )

    # The score keeps its ranking from 2020 to 2021; the component reverses it.
    assert (score["Rank Autocorrelation"] == 1).all()
    assert component.loc["2021-12", "Rank Autocorrelation"] == -1


def test_detail_with_several_metrics_raises_a_clear_error(monkeypatch):
    monkeypatch.setenv("FINANCETOOLKIT_STRICT_ERRORS", "1")

    with pytest.raises(ValueError, match="one metric at a time"):
        FakeRatios().get_factor_analysis(
            ["get_price_to_earnings_ratio", "get_price_to_book_ratio"],
            detail="information_coefficient",
        )


def test_a_parameter_no_metric_takes_raises_before_calculating(monkeypatch):
    monkeypatch.setenv("FINANCETOOLKIT_STRICT_ERRORS", "1")
    ratios = FakeRatios()

    with pytest.raises(ValueError, match="window is not a parameter"):
        ratios.get_factor_analysis("get_price_to_book_ratio", window=14)
    assert ratios.calls == []


def test_fewer_than_three_tickers_or_an_unknown_metric_return_no_result():
    ratios = FakeRatios()
    ratios._tickers = TICKERS[:2]

    assert ratios.get_factor_analysis("get_price_to_earnings_ratio").empty
    assert FakeRatios().get_factor_analysis("get_unknown_ratio").empty
