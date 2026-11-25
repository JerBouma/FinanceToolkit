"""Factor Analysis Tests: get_factor_analysis of the ratios, technicals, models and performance"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.models.models_controller import Models
from financetoolkit.performance.performance_controller import Performance
from financetoolkit.ratios import ratios_controller
from financetoolkit.ratios.ratios_controller import Ratios
from financetoolkit.technicals.technicals_controller import Technicals
from financetoolkit.utilities import factor_model
from financetoolkit.utilities.error_model import InvalidArgumentError

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
    _profile_source = None
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


def test_fewer_than_three_tickers_return_no_result():
    ratios = FakeRatios()
    ratios._tickers = TICKERS[:2]

    assert ratios.get_factor_analysis("get_price_to_earnings_ratio").empty


def test_an_unknown_metric_is_a_mistake_in_the_call():
    # A misspelled metric raises with what to pass instead of reading as "no data".
    with pytest.raises(InvalidArgumentError, match="get_unknown_ratio is not a metric"):
        FakeRatios().get_factor_analysis("get_unknown_ratio")


def test_fewer_tickers_than_quantiles_say_so(monkeypatch):
    # Three tickers can't fill five quantiles: a clear error (logged, with an empty result)
    # instead of a missing column.
    performance = Performance(
        tickers=[*TICKERS[:3], "Benchmark"],
        historical_data={"daily": daily_prices()},
        risk_free_rate_data={},
        rounding=None,
        start_date="2020-01-01",
        end_date="2022-12-31",
    )
    factor = future_monthly_return_factor()[TICKERS[:3]]

    assert performance.get_factor_analysis(factor, horizons=1).empty
    monkeypatch.setenv("FINANCETOOLKIT_STRICT_ERRORS", "1")
    with pytest.raises(ValueError, match="5 quantiles need at least 5 tickers"):
        performance.get_factor_analysis(factor, horizons=1)
    assert not performance.get_factor_analysis(factor, horizons=1, quantiles=3).empty


def test_the_swiss_army_knife_filter_is_a_price_line_or_difference_by_its_filter(
    monkeypatch,
):
    # The low-pass filters return smoothed prices (a price line, close / line - 1), the
    # high-pass and band-pass filters oscillate around zero (a share of the close).
    close = daily_prices()["Adj Close"]
    technicals = FakeModule()
    technicals.get_swiss_army_knife_filter = (
        lambda window=20, filter="band_pass", rounding=None: 2 * close
    )
    captured = {}
    monkeypatch.setattr(
        factor_model,
        "get_factor_analysis",
        lambda factors, *args, **kwargs: captured.update(factors) or pd.DataFrame(),
    )

    for metrics, kwargs, expected in [
        ("get_swiss_army_knife_filter", {}, 2.0),
        ("get_swiss_army_knife_filter", {"filter": "butterworth"}, -0.5),
        ("get_swiss_army_knife_filter(filter='gaussian')", {}, -0.5),
        ("get_swiss_army_knife_filter(filter='high_pass')", {}, 2.0),
    ]:
        captured.clear()
        Technicals.get_factor_analysis(technicals, metrics, **kwargs)
        (factor,) = captured.values()
        assert np.allclose(factor.dropna(how="all").to_numpy(), expected)


def test_a_variant_label_need_not_be_in_a_collection(monkeypatch):
    monkeypatch.setenv("FINANCETOOLKIT_STRICT_ERRORS", "1")
    technicals = FakeModule()
    calls = []

    def get_moving_average(window=None, method="simple", rounding=None):
        calls.append((window, method))
        return daily_prices()["Adj Close"]

    technicals.get_moving_average = get_moving_average

    for label in [
        "get_moving_average(method='hull')",
        'get_moving_average(method="hull", window=10)',
    ]:
        Technicals.get_factor_analysis(technicals, label, horizons=1)
    assert calls == [(None, "hull"), (10, "hull")]

    # The parameters are checked against the method's signature.
    with pytest.raises(ValueError, match="size is not a parameter"):
        Technicals.get_factor_analysis(technicals, "get_moving_average(size=10)")


def performance_with_profile(profile=None):
    return Performance(
        tickers=[*TICKERS, "Benchmark"],
        historical_data={"daily": daily_prices()},
        risk_free_rate_data={},
        rounding=None,
        start_date="2020-01-01",
        end_date="2022-12-31",
        profile=profile,
    )


def test_group_sector_takes_the_sectors_from_the_company_profiles():
    sectors = ["Technology"] * 5 + ["Energy"] * 5
    profile = pd.DataFrame(
        [sectors, ["Software"] * 10], index=["Sector", "Industry"], columns=TICKERS
    )
    performance = performance_with_profile(lambda: profile)
    factor = future_monthly_return_factor()

    by_sector = performance.get_factor_analysis(
        factor, horizons=1, detail="group_information_coefficient", group="sector"
    )
    by_mapping = performance.get_factor_analysis(
        factor,
        horizons=1,
        detail="group_information_coefficient",
        group=dict(zip(TICKERS, sectors, strict=True)),
    )

    # A perfect factor ranks the later returns perfectly within every sector too.
    assert list(by_sector.index) == [("Energy", "1 Month"), ("Technology", "1 Month")]
    assert (by_sector["Mean IC"] == 1).all()
    pd.testing.assert_frame_equal(by_sector, by_mapping)


def test_group_without_company_profiles_names_the_alternative(monkeypatch):
    performance = performance_with_profile()
    factor = future_monthly_return_factor()

    assert performance.get_factor_analysis(factor, horizons=1, group="sector").empty
    monkeypatch.setenv("FINANCETOOLKIT_STRICT_ERRORS", "1")
    with pytest.raises(ValueError, match="Financial Modeling Prep API key.*group=\\{"):
        performance.get_factor_analysis(factor, horizons=1, group="sector")
    with pytest.raises(ValueError, match="need the groups"):
        performance.get_factor_analysis(factor, horizons=1, binning="group")


def test_every_controller_passes_the_group_options_on(monkeypatch):
    monkeypatch.setenv("FINANCETOOLKIT_STRICT_ERRORS", "1")
    groups = {
        ticker: "A" if number < 5 else "B" for number, ticker in enumerate(TICKERS)
    }

    ratios = FakeRatios()
    technicals = FakeTechnicals()
    summary = ratios.get_factor_analysis(
        "get_price_to_earnings_ratio", horizons=1, group=groups, demeaned=True
    )
    statistics = technicals.get_factor_analysis(
        "get_stochastic_oscillator",
        horizons=1,
        column="Stochastic %K",
        group=groups,
        binning="group",
        detail="quantile_statistics",
    )

    # The factor ranks the raw returns perfectly, but not those relative to each group.
    assert 0.5 < summary["Mean IC"].iloc[0] < 1  # noqa: PLR2004
    assert summary["Annualized Alpha"].iloc[0] > 0
    # Five tickers per group and five quantiles: one ticker of each group per quantile.
    assert statistics["Count Share"].tolist() == pytest.approx([0.2] * 5)
