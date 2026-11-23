"""Factor Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.utilities import factor_model

TICKERS = list("ABCDEFGHIJ")


def random_prices(periods=400, seed=0):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.01, (periods, len(TICKERS)))
    return pd.DataFrame(
        100 * np.exp(np.cumsum(returns, axis=0)),
        index=pd.bdate_range("2020-01-01", periods=periods),
        columns=TICKERS,
    )


def test_a_factor_equal_to_the_forward_return_has_an_ic_of_one():
    prices = factor_model.sample_by_period(random_prices(), "monthly")
    forward_returns = factor_model.get_forward_returns(prices, 1, "monthly")

    perfect = factor_model.get_information_coefficient(forward_returns, forward_returns)
    reversed_ = factor_model.get_information_coefficient(
        -forward_returns, forward_returns
    )

    assert np.allclose(perfect.dropna(), 1)
    assert np.allclose(reversed_.dropna(), -1)
    # The last month has no forward return, so no IC.
    assert np.isnan(perfect.iloc[-1])


def test_the_ic_is_the_spearman_correlation_of_the_tickers_with_both_values():
    factor = pd.DataFrame([[1.0, 2.0, 3.0, 4.0, np.nan]], columns=list("ABCDE"))
    forward_returns = pd.DataFrame([[0.1, 0.3, 0.2, 0.4, 0.5]], columns=list("ABCDE"))

    ic = factor_model.get_information_coefficient(factor, forward_returns)

    # Ranks 1-2-3-4 against 1-3-2-4: Spearman's rho = 1 - 6 * 2 / (4 * 15) = 0.8.
    assert ic.iloc[0] == pytest.approx(0.8)


def test_a_date_with_fewer_than_three_tickers_has_no_ic():
    factor = pd.DataFrame([[1.0, 2.0, np.nan]], columns=list("ABC"))
    forward_returns = pd.DataFrame([[0.1, 0.2, 0.3]], columns=list("ABC"))

    assert np.isnan(
        factor_model.get_information_coefficient(factor, forward_returns).iloc[0]
    )


def test_forward_returns_skip_the_days_a_ticker_does_not_trade():
    prices = pd.DataFrame(
        {"A": [100.0, 110.0, 121.0], "B": [100.0, np.nan, 120.0]},
        index=pd.period_range("2024-01-02", periods=3, freq="D"),
    )

    forward_returns = factor_model.get_forward_returns(prices, 1, "daily")

    assert forward_returns["A"].iloc[0] == pytest.approx(0.1)
    # B's next close after the first day is the third day's.
    assert forward_returns["B"].iloc[0] == pytest.approx(0.2)
    assert np.isnan(forward_returns["B"].iloc[1])


def test_sampling_takes_the_last_value_of_each_period_and_leaves_unlisted_tickers_empty():
    data = pd.DataFrame(
        {"A": [1.0, 2.0, 3.0, 4.0], "B": [np.nan, np.nan, 5.0, np.nan]},
        index=pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-28", "2024-02-29"]),
    )

    sampled = factor_model.sample_by_period(data, "monthly")

    assert sampled["A"].tolist() == [2.0, 4.0]
    assert np.isnan(sampled["B"].iloc[0])
    assert sampled["B"].iloc[1] == 5.0  # noqa: PLR2004


def test_quantiles_are_equally_sized_and_need_enough_tickers():
    factor = pd.DataFrame(
        [np.arange(10.0), [1.0, 2.0, 3.0] + [np.nan] * 7], columns=TICKERS
    )

    quantiles = factor_model.assign_quantiles(factor, 5)

    assert quantiles.iloc[0].tolist() == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert quantiles.iloc[1].isna().all()


def test_a_perfect_factor_orders_the_quantile_returns():
    prices = factor_model.sample_by_period(random_prices(), "monthly")
    forward_returns = factor_model.get_forward_returns(prices, 1, "monthly")
    quantiles = factor_model.assign_quantiles(forward_returns, 5)

    returns = factor_model.get_quantile_returns(quantiles, forward_returns)

    quantile_returns = returns.drop("Top Minus Bottom")
    assert quantile_returns.is_monotonic_increasing
    assert returns["Top Minus Bottom"] > 0
    # The spread is the average of the per-date differences, which with every quantile
    # present on every date equals the difference of the averages.
    assert returns["Top Minus Bottom"] == pytest.approx(
        returns["Quantile 5"] - returns["Quantile 1"]
    )


def test_an_unchanged_ranking_has_no_turnover_and_a_rank_autocorrelation_of_one():
    factor = pd.DataFrame([np.arange(10.0)] * 4, columns=TICKERS)
    quantiles = factor_model.assign_quantiles(factor, 5)

    turnover = factor_model.get_quantile_turnover(quantiles)
    autocorrelation = factor_model.get_factor_rank_autocorrelation(factor)

    assert turnover.iloc[0].isna().all()
    assert (turnover.iloc[1:] == 0).all().all()
    assert np.isnan(autocorrelation.iloc[0])
    assert np.allclose(autocorrelation.iloc[1:], 1)


def test_a_reversed_ranking_turns_every_outer_quantile_over():
    factor = pd.DataFrame([np.arange(10.0), -np.arange(10.0)], columns=TICKERS)
    quantiles = factor_model.assign_quantiles(factor, 5)

    turnover = factor_model.get_quantile_turnover(quantiles)

    assert turnover["Quantile 1"].iloc[1] == 1
    assert turnover["Quantile 5"].iloc[1] == 1
    assert factor_model.get_factor_rank_autocorrelation(factor).iloc[
        1
    ] == pytest.approx(-1)


def test_the_summary_follows_the_ic_per_date():
    ic = pd.DataFrame({"1 Month": [0.1, 0.3, -0.1, 0.2], "3 Months": [0.5] * 4})

    summary = factor_model.summarize_information_coefficient(ic)

    mean, deviation = ic["1 Month"].mean(), ic["1 Month"].std(ddof=1)
    assert summary.loc["Mean IC", "1 Month"] == pytest.approx(mean)
    assert summary.loc["IC Information Ratio", "1 Month"] == pytest.approx(
        mean / deviation
    )
    assert summary.loc["t-Statistic", "1 Month"] == pytest.approx(
        mean / (deviation / 2)
    )
    assert summary.loc["Positive Share", "1 Month"] == pytest.approx(0.75)
    assert summary.loc["Observations", "1 Month"] == 4  # noqa: PLR2004
    # A constant IC has no deviation and so no ratio.
    assert np.isnan(summary.loc["IC Information Ratio", "3 Months"])


def test_horizon_labels():
    assert factor_model.get_horizon_label(1, "monthly") == "1 Month"
    assert factor_model.get_horizon_label(21, "daily") == "21 Days"


def test_a_metric_too_few_tickers_have_is_left_out_of_a_collection(caplog):
    prices = random_prices()
    momentum = prices.pct_change(21)
    # Only two tickers have a value, as with preferred dividends most companies don't pay.
    thin = momentum.copy()
    thin.iloc[:, 2:] = np.nan

    result = factor_model.get_factor_analysis(
        {"get_momentum": momentum, "get_thin_ratio": thin}, prices
    )

    assert result.index.get_level_values("Metric").unique().tolist() == ["get_momentum"]
    assert "get_thin_ratio left out of the factor analysis" in caplog.text


def test_only_metrics_too_few_tickers_have_give_an_error_naming_them():
    prices = random_prices()
    thin = prices.pct_change(21)
    thin.iloc[:, 2:] = np.nan

    with pytest.raises(ValueError, match="get_thin_ratio has a value for fewer than 3"):
        factor_model.get_factor_analysis({"get_thin_ratio": thin}, prices)


def test_daily_forward_returns_skip_the_days_filled_for_another_ticker():
    from financetoolkit.historical_model import fill_non_trading_days

    dates = pd.period_range("2024-12-20", periods=8, freq="D")
    raw = pd.DataFrame(
        {
            "A": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
            "B": [100.0, 102.0, np.nan, np.nan, 108.0, 110.0, 111.0, 112.0],
        },
        index=dates,
    )
    returns = raw.apply(lambda column: column.dropna().pct_change()).reindex(dates)
    data = fill_non_trading_days(
        pd.concat({"Adj Close": raw, "Return": returns}, axis=1)
    )

    prices = factor_model.mask_non_trading_days(data["Adj Close"], data["Return"])
    forward_returns = factor_model.get_forward_returns(prices, 1, "daily")

    # B did not trade on the two filled days, so the return from its second day runs to
    # its next trading day (108 / 102 - 1) instead of a flat 0 over a filled day.
    assert forward_returns["B"].iloc[1] == pytest.approx(108 / 102 - 1)
    assert forward_returns["B"].iloc[2:4].isna().all()
    # The first price has no return yet and is kept.
    assert forward_returns["B"].iloc[0] == pytest.approx(0.02)
    assert forward_returns["A"].iloc[1] == pytest.approx(102 / 101 - 1)


def test_monthly_forward_returns_count_calendar_months():
    prices = pd.DataFrame(
        {"A": [100.0, 110.0, 121.0, 133.1], "B": [100.0, np.nan, 120.0, 132.0]},
        index=pd.period_range("2024-01", periods=4, freq="M"),
    )

    forward_returns = factor_model.get_forward_returns(prices, 1, "monthly")

    # B has no February price, so January has no one-month return rather than a
    # two-month one.
    assert np.isnan(forward_returns["B"].iloc[0])
    assert forward_returns["B"].iloc[2] == pytest.approx(0.1)
    assert forward_returns["A"].iloc[0] == pytest.approx(0.1)


def test_an_oscillator_defaults_to_its_main_line():
    dates = pd.date_range("2024-01-01", periods=3)
    columns = pd.MultiIndex.from_product(
        [["Histogram", "MACD Line", "Signal Line"], ["AAPL"]]
    )
    values = pd.DataFrame(
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
        index=dates,
        columns=columns,
    )

    factor = factor_model.to_daily_factor(values, ["AAPL"])

    assert (factor["AAPL"] == 2.0).all()
    assert (
        factor_model.to_daily_factor(values, ["AAPL"], "Histogram")["AAPL"] == 1.0
    ).all()


def test_a_metric_label_parses_into_the_method_and_its_parameters():
    assert factor_model.parse_metric_label("get_moving_average(method='hull')") == (
        "get_moving_average",
        {"method": "hull"},
    )
    assert factor_model.parse_metric_label(
        'get_moving_average(method="hull", window=10)'
    ) == ("get_moving_average", {"method": "hull", "window": 10})
    assert factor_model.parse_metric_label("get_moving_average") == (
        "get_moving_average"
    )
    # Not a call with keyword arguments: left as it is, to be reported as unknown.
    assert factor_model.parse_metric_label("get_moving_average('hull')") == (
        "get_moving_average('hull')"
    )

    selected = factor_model.select_metrics(
        "get_rate_of_change, get_moving_average(method='hull', window=10)",
        {"momentum": ["get_rate_of_change"]},
    )
    assert selected == [
        "get_rate_of_change",
        ("get_moving_average", {"method": "hull", "window": 10}),
    ]


def deterministic_factor_data():
    """Eight tickers in two groups with a smooth factor; the values of the alphalens tests."""
    tickers = [f"T{number}" for number in range(8)]
    dates = pd.bdate_range("2021-01-04", periods=60)
    time = np.arange(len(dates))[:, None]
    ticker = np.arange(8)[None, :]
    prices = pd.DataFrame(
        100
        * np.exp(np.cumsum(0.01 * np.sin(0.37 * time * (ticker + 1) + ticker), axis=0)),
        index=dates,
        columns=tickers,
    )
    factor = pd.DataFrame(
        np.cos(0.23 * time + 1.3 * ticker) + 0.5 * np.sin(0.11 * time * (ticker + 2)),
        index=dates,
        columns=tickers,
    )
    # alphalens drops the dates without every horizon's return; these have no factor.
    factor.iloc[-3:] = np.nan
    groups = pd.Series(
        {name: "A" if number < 4 else "B" for number, name in enumerate(tickers)}
    )

    return {"factor": factor}, prices, groups


def analyse_deterministic(**options):
    factors, prices, groups = deterministic_factor_data()
    return factor_model.get_factor_analysis(
        factors,
        prices,
        period="daily",
        horizons=[1, 3],
        quantiles=4,
        groups=options.pop("groups", groups),
        **options,
    )


# The values below are those of alphalens-reloaded 0.5.1 on the same prices and factor
# (get_clean_factor_and_forward_returns with quantiles=4 and periods=(1, 3)).


def test_the_summary_matches_alphalens():
    summary = analyse_deterministic(groups=None).loc["factor"]

    assert summary["Mean IC"].tolist() == pytest.approx(
        [0.026315789474, 0.062656641604]
    )
    assert summary["IC Skewness"].tolist() == pytest.approx(
        [-0.354942333396, 0.028818855841]
    )
    assert summary["IC Kurtosis"].tolist() == pytest.approx(
        [0.335427622028, -0.736409935571]
    )
    assert summary["Annualized Alpha"].tolist() == pytest.approx(
        [0.065993210888, 0.063218552399]
    )
    assert summary["Beta"].tolist() == pytest.approx([0.024868995927, 0.038568624188])


def test_the_group_neutral_analysis_matches_alphalens():
    summary = analyse_deterministic(demeaned=True).loc["factor"]
    by_group = analyse_deterministic(detail="group_information_coefficient")

    # The group-adjusted IC and the group-neutral portfolio's alpha and beta.
    assert summary["Mean IC"].tolist() == pytest.approx([0.02589807853, 0.065580618212])
    assert summary["Annualized Alpha"].tolist() == pytest.approx(
        [0.093077202018, 0.073641628512]
    )
    assert summary["Beta"].tolist() == pytest.approx([0.042115565901, 0.056862224084])
    assert by_group["Mean IC"].tolist() == pytest.approx(
        [0.087719298246, 0.122807017544, 0.021052631579, -0.024561403509]
    )


def test_the_quantile_return_statistics_match_alphalens():
    statistics = analyse_deterministic(
        detail="quantile_return_statistics", demeaned=True, groups=None
    ).loc["3 Days"]
    quantiles = statistics.drop("Top Minus Bottom")

    assert quantiles["Mean Return"].tolist() == pytest.approx(
        [-0.001140108598, 0.000669828758, -0.000510530358, 0.000980810198]
    )
    assert quantiles["Standard Error"].tolist() == pytest.approx(
        [0.000913913978, 0.000794695939, 0.000944373166, 0.000947614109]
    )
    assert quantiles["Mean Return Per Period"].tolist() == pytest.approx(
        [-0.000380180718, 0.000223226419, -0.000170205754, 0.000326829903]
    )
    assert quantiles["Standard Error Per Period"].tolist() == pytest.approx(
        [0.000527648481, 0.000458817915, 0.000545234102, 0.000547105261]
    )
    # The spread is the full difference (alphalens halves it) with √(SE² + SE²).
    spread = statistics.loc["Top Minus Bottom"]
    assert spread["Mean Return"] == pytest.approx(0.000980810198 + 0.001140108598)
    assert spread["Standard Error"] == pytest.approx(
        np.hypot(0.000947614109, 0.000913913978)
    )


def test_the_factor_returns_and_their_growth_match_alphalens():
    returns = analyse_deterministic(detail="factor_returns", groups=None)
    cumulative = analyse_deterministic(detail="cumulative_returns", groups=None)

    assert returns.mean().tolist() == pytest.approx([0.000256152725, 0.000740267509])
    # The growth starts at 1 and ends after the last holding period.
    assert cumulative[("1 Day", "Factor")].iloc[0] == 1
    assert cumulative[("1 Day", "Factor")].iloc[-1] == pytest.approx(1.014461813361)


def test_the_quantile_statistics_and_binnings_match_alphalens():
    factors, prices, _ = deterministic_factor_data()
    statistics = analyse_deterministic(detail="quantile_statistics", groups=None)
    factor = factors["factor"].iloc[:-3]

    assert statistics["Mean"].tolist() == pytest.approx(
        [-0.954787408311, -0.340603535638, 0.325418259198, 0.971555515138]
    )
    equal_width = factor_model.assign_bins(factor, 4, "equal_width")
    assert equal_width.stack().value_counts().sort_index().tolist() == [
        125,
        114,
        77,
        140,
    ]
    centred = factor.sub(factor.median(axis=1), axis=0)
    zero_aware = factor_model.assign_bins(centred, 4, "zero_aware")
    assert zero_aware.stack().value_counts().sort_index().tolist() == [114] * 4
    assert (zero_aware.where(centred < 0).stack().dropna() <= 2).all()  # noqa: PLR2004


def test_quantiles_within_groups_hold_every_group_equally():
    factors, _, groups = deterministic_factor_data()
    assignment = factor_model.assign_bins(factors["factor"], 4, "group", groups)

    # Four tickers per group and four quantiles: each group has one ticker per quantile.
    for name in ("A", "B"):
        members = assignment.loc[:, groups == name].dropna()
        assert (members.apply(sorted, axis=1).tolist()) == [[1, 2, 3, 4]] * len(members)


def test_the_event_returns_are_the_rebased_price_paths():
    factors, prices, _ = deterministic_factor_data()
    events = analyse_deterministic(detail="event_returns", groups=None)
    assignment = factor_model.assign_quantiles(factors["factor"].iloc[:-3], 4)

    # The path of quantile 1 three days after its dates, by hand.
    later = prices.shift(-3) / prices - 1
    by_hand = later.reindex(assignment.index).where(assignment == 1).mean(axis=1)
    assert events.loc[3, ("Quantile 1", "Mean")] == pytest.approx(by_hand.mean())
    assert events.loc[0, ("Quantile 1", "Mean")] == 0
    assert list(events.index) == [-3, -2, -1, 0, 1, 2, 3]


def test_the_mean_ic_per_calendar_period_needs_a_shorter_period():
    factors, prices, _ = deterministic_factor_data()
    monthly = analyse_deterministic(
        detail="information_coefficient_by_month", groups=None
    )
    ic = analyse_deterministic(detail="information_coefficient", groups=None)

    assert monthly.loc["2021-02", "1 Day"] == pytest.approx(
        ic.loc["2021-02", "1 Day"].mean()
    )
    with pytest.raises(ValueError, match="needs a period of a month or shorter"):
        factor_model.get_factor_analysis(
            factors,
            prices,
            period="quarterly",
            detail="information_coefficient_by_month",
        )
    with pytest.raises(ValueError, match="need the groups"):
        factor_model.get_factor_analysis(factors, prices, binning="group")
    with pytest.raises(ValueError, match="even number of quantiles"):
        factor_model.get_factor_analysis(factors, prices, binning="zero_aware")
