# Finance Toolkit v2.1.4

## Introduction

This is one of the larger feature releases to date. It roughly doubles the depth of three modules — Models, Performance, and Risk — and rounds out Technicals, Ratios, and Discovery with metrics that were previously missing.

Two capabilities now cut across nearly the whole toolkit:

- **`rolling` and `trailing` windows.** Many metrics used to return one value per reporting period. They can now be computed over a sliding window (`rolling=<n>`) or a trailing sum/average (`trailing=<n>`), turning a snapshot into a proper time series.
- **`standardize` (Z-Score).** Most `get_*` methods across Economics, Ratios, Technicals, Risk, Performance, Models, Options, and Fixed Income now accept `standardize=True`. It converts raw values into standard deviations from their own historical mean, so metrics on completely different scales become directly comparable — useful for ranking, scoring, or spotting an unusual reading at a glance.

On the fix side: swaption prices are discounted correctly now, a case-mismatch bug that was silently zeroing out diluted EPS is gone, and the new Forward P/E / Forward PEG ratios cache analyst estimates properly instead of double-fetching.

**`Volatility`, `Excess Return`, and `Excess Volatility` are gone from `toolkit.get_historical_data()`**. Use these instead:

- `toolkit.risk.get_volatility(period=...)`
- `toolkit.risk.get_excess_volatility(period=...)`
- `toolkit.risk.get_variance(period=...)`
- `toolkit.performance.get_excess_return(risk_free_rate=..., cumulative=...)`

Every method below hangs off the matching `Toolkit` sub-module (e.g. `toolkit.models.get_graham_number()`) or its standalone class (`Discovery(...)`, `Economics(...)`, `FixedIncome(...)`). Full runnable examples with real output live in the [documentation](https://www.jeroenbouma.com/projects/financetoolkit/docs) — each section below links to its module's page.

## New Features

### Standardization (Z-Score)

`standardize: bool = False` is now available on nearly every controller — Economics, Ratios, Technicals, Risk, Performance, Models, Options, and Fixed Income, roughly 270 methods in total.

```
Z-Score = (value - mean) / standard deviation
```

Mean and standard deviation come from each series' own history: per ticker for entity-indexed data like ratios, per column for date-indexed data like economic indicators. With `growth=True`, it standardizes the growth values instead of the raw ones. Use it to compare metrics that live on incompatible scales, build a quick cross-sectional ranking straight from toolkit output, or spot when a current reading is an unusual number of standard deviations from its own norm.

### Financial Models

Call these via `toolkit.models.<method>()`. Full examples: [docs/models](https://www.jeroenbouma.com/projects/financetoolkit/docs/models).

**`get_economic_value_added`** (EVA, economic profit) — whether a company earns more than its true cost of capital, something Net Income alone can't tell you.
- `NOPAT = EBIT × (1 − Effective Tax Rate)`
- `Invested Capital = Total Equity + Total Debt` (each smoothed as a 2-period average, or a `trailing`-period average, to soften a single noisy balance-sheet snapshot)
- `EVA = NOPAT − (WACC × Invested Capital)`
- Read it as: positive EVA creates value for capital providers; negative EVA destroys it, even in periods where net income is positive.

**`get_beneish_m_score`** (earnings manipulation score) — Messod Beneish's model for flagging likely earnings manipulation, a companion to the Altman Z-Score and Piotroski F-Score already in the toolkit.
- `M-Score = -4.84 + 0.92·DSRI + 0.528·GMI + 0.404·AQI + 0.892·SGI + 0.115·DEPI − 0.172·SGAI + 4.679·TATA − 0.327·LVGI`
- Each of the eight inputs is its own function in `beneish_model.py`: `get_days_sales_in_receivables_index` (DSRI — receivables outgrowing sales), `get_gross_margin_index` (GMI — margin deterioration), `get_asset_quality_index` (AQI — a rising share of non-core assets), `get_sales_growth_index` (SGI — fast growth raises manipulation pressure), `get_depreciation_index` (DEPI — a slowing depreciation rate flatters earnings), `get_selling_general_and_administrative_expenses_index` (SGAI), `get_leverage_index` (LVGI), and `get_total_accruals_to_total_assets` (TATA — the accrual share of earnings, a red flag across most accounting-fraud models).
- Read it as: above roughly **-1.78**, manipulation is more likely; below it, less likely. It's probabilistic, not a verdict — pair it with real fundamental analysis. The first period is always NaN since every input needs a prior period to compare against.
- Reference: Beneish, M. D. "The Detection of Earnings Manipulation." *Financial Analysts Journal*, Vol. 55, No. 5, 1999, pp. 24-36.

**`get_sustainable_growth_rate`** (SGR) — the fastest a company can grow revenue on internal funds alone, without new equity or added leverage.
- `SGR = Return on Equity × (1 − Dividend Payout Ratio)`
- Read it as: growing faster than SGR without outside financing means improving margins, cutting the payout, or taking on more debt — so actual growth versus SGR is a quick tell for how a company is funding its expansion.

**`get_internal_growth_rate`** (IGR) — a stricter version of SGR: growth funded from retained earnings only, with no debt added at all.
- `IGR = (ROA × Retention Ratio) / (1 − (ROA × Retention Ratio))`
- Read it as: IGR is always below SGR for a levered company. The gap between them is roughly how much of that company's "sustainable" growth actually depends on debt.

**`get_graham_number`** (Graham fair value) — Benjamin Graham's conservative ceiling on what a defensive investor should pay.
- `Graham Number = √(22.5 × EPS × Book Value per Share)`
- Read it as: most useful for stable, profitable companies with positive book value. Negative earnings or book value produce NaN (the square root of a negative number is undefined) — comparing the current price against this number is a quick margin-of-safety check.

### Ratios

Call these via `toolkit.ratios.<method>()`. Full examples: [docs/ratios](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios).

**Efficiency**
- **`get_research_and_development_ratio`**, **`get_selling_and_marketing_ratio`**, **`get_general_and_administrative_ratio`** — R&D, S&M, and G&A each as a share of Revenue. Most companies only report these lumped into SG&A; splitting them out lets you compare R&D intensity or customer-acquisition spend independent of admin overhead.
- **`get_stock_based_compensation_ratio`** — SBC ÷ Revenue. A high or rising number is a quality-of-earnings flag, since SBC gets added back in cash flow from operations (flattering FCF) while still diluting existing shareholders.
- **`get_deferred_revenue_ratio`** — Deferred Revenue ÷ Revenue. A leading indicator for subscription/SaaS businesses: a growing balance here signals revenue that hasn't hit the income statement yet.

**Profitability**
- **`get_cash_tax_rate`** — taxes actually *paid* in cash ÷ Income Before Tax, versus the accrual-based Effective Tax Rate.
- **`get_tax_rate_divergence`** — Cash Tax Rate minus Effective Tax Rate. Persistently positive means more cash tax paid than expensed (e.g. deferred liabilities unwinding); persistently negative can flag aggressive tax deferral.

**Solvency**
- **`get_debt_to_capital_ratio`** — `Total Debt / (Total Debt + Total Equity)`. Stays between 0 and 1, unlike Debt-to-Equity, which makes capital structure easier to compare across companies.
- **`get_preferred_dividend_coverage_ratio`** — Net Income ÷ |Preferred Dividends|: how many times earnings cover the obligation to preferred shareholders.
- **`get_interest_paid_to_expense_ratio`** — Interest *Paid* ÷ Interest *Expense*. Well below 1 suggests interest is being accrued rather than paid (e.g. payment-in-kind debt); well above 1 suggests catch-up payments or a cash/accrual timing gap.

**Valuation**
- **`get_ev_to_free_cash_flow_ratio`** — Enterprise Value ÷ Free Cash Flow, a multiple that (unlike EV/EBITDA) accounts for capex and working capital.
- **`get_buyback_yield`** — `-(Repurchases + Issuances) / Market Cap`. Positive means the company is a net repurchaser; negative means it's a net issuer, diluting shareholders.
- **`get_shareholder_yield`** — Dividend Yield + Buyback Yield: total cash returned to shareholders through both channels.
- **`get_sbc_adjusted_free_cash_flow`** — Free Cash Flow minus Stock-Based Compensation, treating SBC as the real cost it is instead of the non-cash add-back FCF normally lets it be.
- **`get_forward_price_earnings_ratio`** (Forward P/E) — `Stock Price / Estimated EPS Average`, using the analyst consensus estimate for a future period instead of trailing EPS. Needs a Premium FMP subscription; estimates are fetched once per `Toolkit` instance and cached automatically, so there's no need to call `get_analyst_estimates()` first.
- **`get_forward_price_earnings_growth_ratio`** (Forward PEG) — `Forward P/E / (Estimated EPS Growth Rate × 100)`, where growth bridges the analyst estimate to the company's most recent actual EPS (`get_estimated_eps_growth_rate`, its own reusable function) rather than comparing two historical periods the way `growth=True` does everywhere else. Read it as: near or below 1 is the classic (simplistic) heuristic for fairly priced growth; well above 1 means the market is paying a premium for it.

### Technical Indicators

Call these via `toolkit.technicals.<method>()`. Full examples: [docs/technicals](https://www.jeroenbouma.com/projects/financetoolkit/docs/technicals).

**Overlap**
- **`get_weighted_moving_average`** (WMA) — weights recent prices more, so it tracks price more closely than a Simple Moving Average while staying smoother than an Exponential one.
- **`get_hull_moving_average`** (HMA) — combines a half-window WMA, a full-window WMA, and a √window WMA specifically to cut moving-average lag without giving up smoothness.
- **`get_volume_weighted_average_price`** (VWAP) — typical price weighted by volume over a rolling window; the benchmark most execution algorithms are measured against.
- **`get_parabolic_sar`** — a trend-following stop-and-reverse level that flips sides when the trend reverses, with an acceleration factor (`af_start`/`af_increment`/`af_max`, defaults 0.02/0.02/0.2) that speeds convergence as a trend runs longer. Commonly used to trail stop-losses.
- **`get_pivot_points`** — Pivot Point plus Resistance/Support 1-3, derived from the *prior* period's high/low/close to flag likely next-period levels.

**Volatility**
- **`get_donchian_channels`** — highest high and lowest low over a rolling window, with the midpoint as the middle line; spots breakouts and shows how wide the recent range has been.
- **`get_volatility_cone`** — the historical percentile spread (min through max, plus current) of annualized realized volatility across several lookback windows (`[10, 20, 30, 60, 90, 120]` by default). Shows at a glance whether current volatility is cheap or expensive versus its own history — often used before comparing to options-implied volatility.

**Breadth** (cross-sectional across tickers at a point in time, not per-ticker over time)
- **`get_trin`** (Arms Index) — `(Advancing Issues / Declining Issues) / (Advancing Volume / Declining Volume)`. Below 1 suggests volume conviction is outpacing the number of advancing names (bullish); above 1 suggests the reverse.
- **`get_new_highs_new_lows`** — tickers at a window-period high minus tickers at a window-period low: a read on how broad a market move really is beneath the index-level headline.

### Performance

Call these via `toolkit.performance.<method>()`. Full examples: [docs/performance](https://www.jeroenbouma.com/projects/financetoolkit/docs/performance).

- **`get_calmar_ratio`** — `Return / |Maximum Drawdown|`. Penalizes purely by the single worst peak-to-through loss, not by volatility the way Sharpe/Sortino do.
- **`get_sterling_ratio`** — `Return / (|Average Drawdown| + adjustment)` (adjustment defaults to 10%), using the *average* of all drawdowns rather than just the worst one — softer and less driven by a single event than Calmar.
- **`get_burke_ratio`** — `Excess Return / Burke Drawdown Measure`, where the measure is `√(Σ drawdown²)` (new `get_burke_drawdown_measure`). Squaring before summing penalizes both frequency and depth of drawdowns, so two strategies sharing a max drawdown can still score very differently here.
- **`get_upside_capture_ratio`** / **`get_downside_capture_ratio`** — the asset's average return versus the benchmark's, split across periods where the benchmark was positive or negative. Upside capture above 1 and downside capture below 1 is the profile most investors want: gaining more in rallies than losing in selloffs.
- **`get_win_rate`** — the share of periods where the asset beat the benchmark. Simple, distribution-free complement to the capture ratios.
- **`get_kappa_ratio`** (default `order=3`) — Sortino generalized to a higher-order lower partial moment; Sortino itself is the `order=2` case. Higher orders punish larger losses more, making Kappa more tail-sensitive than Sortino.
- **`get_omega_ratio`** (default `minimum_acceptable_return=0.0`) — gains above a minimum acceptable return divided by losses below it, using the full return distribution rather than just mean and variance — so it captures skew and fat tails that Sharpe/Sortino miss.
- **`get_gain_to_pain_ratio`** — `Σ all returns / Σ |losses|`. A blunt, easy-to-explain measure of gain earned per unit of pain endured.
- **`get_returns` / `get_excess_return`** now take a `cumulative` parameter (pulled out of `get_historical_data`, see Breaking Changes) — `cumulative=True` compounds returns into a running index instead of period-over-period values.
- **Rolling variants** — `get_rolling_alpha`, `get_rolling_sortino_ratio`, `get_rolling_m2_ratio`, `get_rolling_tracking_error`, `get_rolling_information_ratio`, plus rolling support on Beta, CAPM, Jensen's Alpha, and Treynor Ratio. Pass `rolling=<window>` to see risk-adjusted performance evolve through time instead of getting one backward-looking number.

### Risk

Call these via `toolkit.risk.<method>()`. Full examples: [docs/risk](https://www.jeroenbouma.com/projects/financetoolkit/docs/risk).

- **`get_conditional_drawdown_at_risk`** (CDaR) — extends VaR/CVaR from returns to *drawdowns*: DaR is the alpha-quantile of the drawdown distribution, CDaR is the average of drawdowns at least that severe. Where VaR/CVaR ask "how bad could one period be," CDaR asks "how bad could the cumulative peak-to-through loss get."
- **`get_tail_ratio`** — `|percentile(1-alpha)| / |percentile(alpha)|` of returns. Above 1 means best-case gains outsize worst-case losses — a quick complement to skewness that doesn't assume a distribution shape.
- **`get_maximum_drawdown_duration`** — periods between the peak and through of the single largest drawdown: how long the worst decline took to play out.
- **`get_maximum_drawdown_recovery_time`** — periods from that through until cumulative return hits a new high (NaN if it hasn't yet). Paired with duration, this gives the full timeline of a strategy's worst stretch.
- **`get_downside_deviation`** (semi-deviation, default `minimum_acceptable_return=0.0`) — standard deviation of only the returns below a minimum acceptable return, so upside swings no longer count against the metric the way they do in plain standard deviation.
- **`get_mean_absolute_deviation`** (MAD) — mean absolute distance from the mean return. No squaring, so a handful of outliers move it less than they'd move variance.
- **`get_coefficient_of_variation`** (CV, relative standard deviation) — `standard deviation / mean` of returns, which lets you compare relative volatility across assets with very different average returns — something raw standard deviation can't do.
- **`get_ewma_volatility`** (RiskMetrics volatility, default `lambda_=0.94`) — `EWMA Variance(t) = λ·EWMA Variance(t-1) + (1-λ)·Return(t-1)²`. Weights recent observations more, so it reacts to a changing volatility regime faster than a fixed rolling window, while staying much simpler than a full GARCH fit.
- **`get_autocorrelation`** (default `lags=10`) — correlation between a series and its own lagged values; usually the first thing checked before deciding how many AR/MA terms a series needs.
- **`get_hurst_exponent`** (default `max_lag=20`) — estimated via rescaled range (R/S). `H < 0.5` mean-reverting, `H = 0.5` random walk, `H > 0.5` trending — a single number for whether a series tends to trend or revert.
- **Extreme Value Theory tail modelling** — `get_value_at_risk(distribution="evt")` now fits a Generalized Pareto Distribution to the worst losses via Peak-over-Threshold (new `get_var_evt`, plus a `threshold_percentile` parameter, default `0.95`, for how much of the tail to fit on). Models the tail's actual shape instead of assuming the whole distribution (Gaussian, Cornish-Fisher, Student-t) represents extreme outcomes well.
- **`get_volatility` / `get_variance` / `get_excess_volatility`** are now first-class Risk methods (pulled out of `get_historical_data`, see Breaking Changes).
- **Rolling variants** — `get_rolling_var_historic`, `get_rolling_cvar_historic`, `get_rolling_variance`, `get_rolling_volatility`, `get_rolling_excess_volatility`, `get_rolling_tail_ratio`, `get_rolling_conditional_drawdown_at_risk`, `get_rolling_skewness`, `get_rolling_kurtosis`, `get_rolling_downside_deviation`, `get_rolling_omega_ratio` — same sliding-window idea as Performance, across Risk.

### Economics

Call these via `Economics(...).get_*()`, or `toolkit.economics.<method>()` on a `Toolkit` instance. Full examples: [docs/economics](https://www.jeroenbouma.com/projects/financetoolkit/docs/economics).

- **`rolling` and `trailing` parameters** now cover essentially the whole module — GDP, CPI, inflation, trade, investment, consumption, confidence indices, government fiscal metrics, interest rates, unemployment, population, and more (45 methods). `rolling=<n>` smooths noisy monthly/quarterly prints with a simple moving average; `trailing=<n>` sums over a trailing window (e.g. a trailing 4-quarter sum to annualize a quarterly flow). Both apply before any `growth`/`standardize` step.

### Discovery & News

Call these via `Discovery(...).get_*()` / `search_*()`, or `toolkit.discovery.<method>()` for the ticker-scoped ones. Full examples: [docs/discovery](https://www.jeroenbouma.com/projects/financetoolkit/docs/discovery).

- **On the `Toolkit` itself:** `get_stock_news(pages, limit)` and `get_press_releases(pages, limit)` — ticker-scoped, filtered automatically to the Toolkit's `start_date`/`end_date`.
- **In `Discovery`, not ticker-scoped:** `get_stock_news`, `get_general_news` (broad market news), `get_press_releases`, `get_crypto_news`, `get_forex_news`, plus free-text search via `search_stock_news`, `search_press_releases`, `search_crypto_news`, `search_forex_news`.
- **Corporate events & calendars:** `get_ipo_calendar`, `get_ipo_disclosures` (SEC pre-IPO filings), `get_ipo_prospectuses` (S-1/424B4 pricing), `get_stock_splits_calendar`, `get_mergers_acquisitions_latest` (recent deals with acquirer/target and a link to the SEC filing).
- **Sector & industry data:** `get_sector_performance` / `get_industry_performance` and `get_sector_pe` / `get_industry_pe` — pass `date` for a snapshot across everything, or `sector`/`industry` for one name's history. Good for checking whether a move was sector-wide or idiosyncratic, or whether a sector looks rich or cheap against its own P/E history.

### Fixed Income

Call this via `FixedIncome().get_derivative_price(...)`. Full examples: [docs/fixedincome](https://www.jeroenbouma.com/projects/financetoolkit/docs/fixedincome).

- `tenor` and `payment_frequency` parameters on `get_derivative_price` — see Bug Fixes above; this is what makes swaption pricing account for the underlying swap's full payment schedule instead of treating it as a single-date payout.

## Bug Fixes

1. **Swaption pricing was missing the annuity leg.** `get_black_price` and `get_bachelier_price` used to discount the option payoff with a single factor to the expiration date, as if exercising a swaption paid out once. It doesn't — it grants the right to enter a swap that pays out at *every* payment date over the swap's tenor. Both functions now multiply by an annuity factor (the sum of discount factors across those payment dates) via a new `_get_annuity_factor()` helper. Two new parameters, `tenor` and `payment_frequency`, let you set the underlying swap's length and payment schedule separately from the option's own expiration. Previously tenor had zero effect on price; now a 1-year option into a 5-year swap correctly prices roughly 5x higher than one into a 1-year swap, since it covers five times as many payment dates.

2. **`trailing` rolled the wrong axis when combined with `show_daily`.** `get_enterprise_value`, `get_market_cap`, and related valuation ratios rolled the trailing window across tickers instead of across dates whenever both parameters were active together.

3. **`trailing` did nothing in `get_market_cap`.** Its if/else branches were identical, so the parameter was silently ignored.

4. **Cumulative return broke on gappy data.** `(1 + returns).cumprod()` became `(1 + returns.fillna(0)).cumprod()` in `historical_model.py` and `risk_model.py`, with `.replace([np.inf, -np.inf], np.nan)` added before the product to stop infinities from propagating through the series.

5. **Silenced a pandas `FutureWarning`.** Added `future_stack=True` to `stack()` calls in the performance controller ahead of pandas' default-behaviour change.

6. **Fixed a type annotation.** `pd.PeriodIndex` corrected to `pd.DatetimeIndex` in a `helpers.py` signature.

7. **Risk methods ignored the requested date range.** `get_value_at_risk`, `get_conditional_value_at_risk`, `get_entropic_value_at_risk`, `get_conditional_drawdown_at_risk`, `get_tail_ratio`, `get_maximum_drawdown`, `get_maximum_drawdown_duration`, `get_maximum_drawdown_recovery_time`, `get_ulcer_index`, `get_garch`, `get_skewness`, `get_kurtosis`, and `get_variance` could silently return data outside `start_date`/`end_date`, especially with `rolling` or `within_period` set. All now truncate correctly.

8. **A one-character typo zeroed out diluted EPS everywhere.** `financetoolkit/normalization/income.csv` mapped FMP's field as `epsdiluted` (lowercase), but the `stable` endpoint actually returns `epsDiluted` (capital D). Normalization fills any expected-but-missing column with `0` before renaming, so this mismatch meant `get_income_statement()` always reported "EPS Diluted" as zero — the real column was silently dropped instead of renamed. It stayed invisible until the new Forward PEG ratio started reading that field directly, at which point it returned exactly `0.0` for every ticker, every period. Fixed by correcting the key to `epsDiluted`.

9. **Forward-looking ratios fetched analyst estimates twice instead of caching once.** `get_forward_price_earnings_ratio()` and `get_forward_price_earnings_growth_ratio()` are meant to fetch Premium-only analyst estimates once and reuse them. The cache lived on the `Ratios` instance, but `toolkit.ratios` builds a brand-new `Ratios` instance on every access — so calling both methods back to back (exactly what the docs show) fetched estimates twice, doubling Premium API usage for nothing. Fixed by sharing a mutable cache dict from the `Toolkit` instance into every `Ratios` instance it builds, so the fetch-once behaviour survives across calls for the life of the `Toolkit` object.

10. **Log messages were mis-pluralized.** `"Obtaining %s data for %d tickers"` now reads `"...for %d ticker(s)"`, so single-ticker runs no longer say "for 1 tickers".

11. **Local dev setup was missing MCP dependencies.** The `dev` dependency group in `pyproject.toml` didn't pull in the project's own `mcp` extra, so a plain `uv sync` left `fastmcp`/`pydantic` and friends missing. Added `"financetoolkit[mcp]"` to `dev` plus a `[tool.uv.sources]` workspace entry.

## Test Coverage

New coverage for the Risk module's new metrics (`test_cvar_model.py`, expanded `test_risk_controller.py`, `test_risk_model.py`, `test_var_model.py`), the new technical indicators (Donchian Channels, Hull Moving Average, New Highs/New Lows, Parabolic SAR, Pivot Points, TRIN, VWAP, Weighted Moving Average, Volatility Cone), and the internal `helpers.py` split (`test_dataframe_model.py`, `test_requests_model.py`, `test_statistics_model.py`). Added missing snapshot tests for `collect_custom_ratios`, `get_cash_conversion_efficiency`, `get_effective_tax_rate`, `get_debt_service_coverage_ratio`, `get_weighted_dividend_yield`, `get_reinvestment_rate`, `get_ev_to_ebit`, `collect_all_metrics` (performance + risk), `get_compound_growth_rate`, and `get_estimated_eps_growth_rate`. Updated snapshots for valuation methods affected by the `trailing`/`show_daily` axis fix. Added `httpx2` as a dev dependency to resolve a `StarletteDeprecationWarning` in MCP OAuth tests.

## MCP Server

- Tool/category descriptions in `config.yaml` and parameter descriptions in `registry_controller.py` updated to cover all of the above: rolling/trailing on Economics, the new `standardize`/`cumulative`/`threshold_percentile`/`minimum_acceptable_return` params, the new Risk and Performance metrics, and the new Models functions.
- The `valuation` tool group now explicitly lists `get_forward_price_earnings_ratio` and `get_forward_price_earnings_growth_ratio` via a `method_override` entry. They're deliberately excluded from the bulk `collect_valuation_ratios` scan — including them there would trigger a Premium-only analyst-estimates fetch on every bulk request, even for free-tier users — so normal method discovery can't find them on its own.
- MCPB manifest and build script updated to match.
- Removed an unused `get_stock_quotes` function from `discovery_model.py` — dead code, since `fmp_model.get_quote` already covers per-ticker real-time quotes.

**Full comparison:** [v2.1.3...v2.1.4](https://github.com/JerBouma/FinanceToolkit/compare/v2.1.3...v2.1.4)
