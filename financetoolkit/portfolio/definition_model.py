"""Portfolio Definition Model"""

__docformat__ = "google"

from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from financetoolkit.portfolio import portfolio_model

# pylint: disable=too-many-instance-attributes

# How often a model portfolio is brought back to its target weights.
REBALANCING_FREQUENCIES = {
    "none": None,
    "monthly": "M",
    "quarterly": "Q",
    "yearly": "Y",
}

# Weights may sum to one within this margin before they are rejected.
WEIGHT_TOLERANCE = 1e-3

# Trades worth less than this (in the portfolio's currency) are not recorded.
MINIMUM_TRADE_VALUE = 1e-6


@dataclass
class Holding:
    """
    One holding of a portfolio definition.

    Args:
        ticker (str): The ticker, e.g. "VTI".
        weight (float): The target weight, as a decimal.
        name (str, optional): The name of the fund or company. Defaults to "".
        asset_class (str, optional): The asset class, e.g. "Equity", "Fixed Income",
            "Commodity" or "Cash". Defaults to "".
        currency (str, optional): The currency the holding trades in. Defaults to "".
        duration (float | None, optional): For a bond fund, its duration in years. Defaults
            to None.
    """

    ticker: str
    weight: float
    name: str = ""
    asset_class: str = ""
    currency: str = ""
    duration: float | None = None


@dataclass
class PortfolioDefinition:
    """
    A portfolio as weights per holding, with what is known about it: a name and
    description, the source of the allocation, how often it is rebalanced and, for a
    portfolio whose weights change over time (a target-date glide path), the weights per
    year from the start. It is plain data, so it can be stored as YAML or JSON and read by
    other tools.

    Args:
        name (str): A short name, e.g. "60/40 Balanced".
        holdings (list[Holding]): The holdings and their target weights.
        description (str, optional): What the portfolio is. Defaults to "".
        source (dict[str, str], optional): Where the allocation comes from, with a "text"
            (including the page or section) and a "link". Defaults to empty.
        rebalancing (str, optional): "none", "monthly", "quarterly" or "yearly". Defaults to
            "quarterly".
        schedule (list[dict], optional): Weights that change over time, as a list of
            {"year": years from the start, "weights": {ticker: weight}}; the weights are
            interpolated linearly between the years. Defaults to empty, fixed weights.
        aliases (list[str], optional): Other names the portfolio is known by, e.g. "60/40".
            Defaults to empty.
        identifier (str, optional): The file name of a preset, e.g. "balanced_60_40".
            Defaults to "".
    """

    name: str
    holdings: list[Holding]
    description: str = ""
    source: dict[str, str] = field(default_factory=dict)
    rebalancing: str = "quarterly"
    schedule: list[dict[str, Any]] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    identifier: str = ""

    def __post_init__(self):
        self.holdings = [
            holding if isinstance(holding, Holding) else Holding(**holding)
            for holding in self.holdings
        ]
        if not self.holdings:
            raise ValueError("A portfolio needs at least one holding.")
        total = sum(holding.weight for holding in self.holdings)
        if abs(total - 1) > WEIGHT_TOLERANCE:
            raise ValueError(f"The weights of {self.name!r} sum to {total:.4f}, not 1.")
        if self.rebalancing not in REBALANCING_FREQUENCIES:
            raise ValueError(
                "The rebalancing must be one of "
                f"{', '.join(REBALANCING_FREQUENCIES)}, not {self.rebalancing!r}."
            )
        tickers = set(self.tickers)
        for checkpoint in self.schedule:
            weights = checkpoint.get("weights", {})
            unknown = set(weights) - tickers
            if unknown:
                raise ValueError(
                    f"The schedule of {self.name!r} names tickers without a holding: "
                    f"{', '.join(sorted(unknown))}."
                )
            if abs(sum(weights.values()) - 1) > WEIGHT_TOLERANCE:
                raise ValueError(
                    f"The weights of year {checkpoint.get('year')} in the schedule of "
                    f"{self.name!r} do not sum to 1."
                )

    @property
    def tickers(self) -> list[str]:
        """The tickers of the holdings."""
        return [holding.ticker for holding in self.holdings]

    @property
    def weights(self) -> dict[str, float]:
        """The target weight per ticker."""
        return {holding.ticker: holding.weight for holding in self.holdings}

    def weights_at(self, years: float) -> dict[str, float]:
        """
        The target weights a number of years after the start: the fixed weights, or the
        schedule interpolated linearly between its years (flat before the first and after
        the last).

        Args:
            years (float): The years since the start.

        Returns:
            dict[str, float]: The weight per ticker.
        """
        if not self.schedule:
            return self.weights

        checkpoints = sorted(self.schedule, key=lambda checkpoint: checkpoint["year"])
        points = [float(checkpoint["year"]) for checkpoint in checkpoints]
        return {
            ticker: float(
                np.interp(
                    years,
                    points,
                    [
                        checkpoint["weights"].get(ticker, 0.0)
                        for checkpoint in checkpoints
                    ],
                )
            )
            for ticker in self.tickers
        }

    def to_dict(self) -> dict[str, Any]:
        """The definition as a plain dictionary."""
        data = asdict(self)
        if not data["identifier"]:
            data.pop("identifier")
        return data

    def to_yaml(self, path: str | Path | None = None) -> str:
        """
        The definition as YAML, written to a file when a path is given.

        Args:
            path (str | Path | None, optional): The file to write. Defaults to None.

        Returns:
            str: The YAML text.
        """
        text = yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PortfolioDefinition":
        """
        A definition from a dictionary such as `to_dict` returns.

        Args:
            data (dict[str, Any]): The definition.

        Returns:
            PortfolioDefinition: The definition.
        """
        return cls(**data)

    @classmethod
    def from_yaml(cls, source: str | Path) -> "PortfolioDefinition":
        """
        A definition from a YAML file or YAML text.

        Args:
            source (str | Path): The path to a YAML file, or the YAML text itself.

        Returns:
            PortfolioDefinition: The definition.
        """
        path = Path(source)
        text = (
            path.read_text(encoding="utf-8")
            if path.suffix in (".yaml", ".yml") and path.exists()
            else str(source)
        )
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError("The YAML does not describe a portfolio definition.")
        return cls.from_dict(data)

    @classmethod
    def from_weights(
        cls,
        weights: dict[str, float] | pd.Series | pd.DataFrame | str,
        name: str = "Custom Portfolio",
        description: str = "",
        rebalancing: str = "quarterly",
    ) -> "PortfolioDefinition":
        """
        A definition from weights: a dictionary, a Series, the table an optimizer of
        `toolkit.portfolio` returns (its Weight column), or text such as "VTI:0.6, BND:0.4".
        Holdings with a weight of zero are left out and the weights are scaled to sum to one.

        Args:
            weights (dict[str, float] | pd.Series | pd.DataFrame | str): The weights.
            name (str, optional): The name. Defaults to "Custom Portfolio".
            description (str, optional): The description. Defaults to "".
            rebalancing (str, optional): How often to rebalance. Defaults to "quarterly".

        Returns:
            PortfolioDefinition: The definition.
        """
        parsed = parse_weights(weights)
        return cls(
            name=name,
            description=description,
            rebalancing=rebalancing,
            holdings=[
                Holding(ticker=ticker, weight=weight)
                for ticker, weight in parsed.items()
            ],
        )


def parse_weights(
    weights: dict[str, float] | pd.Series | pd.DataFrame | str,
) -> dict[str, float]:
    """
    Weights from a dictionary, Series, optimizer table (Weight column, without the
    Portfolio row) or text such as "VTI:0.6, BND:0.4", without zero weights and scaled to
    sum to one.

    Args:
        weights (dict[str, float] | pd.Series | pd.DataFrame | str): The weights.

    Returns:
        dict[str, float]: The weight per ticker.
    """
    if isinstance(weights, pd.DataFrame):
        if "Weight" not in weights.columns:
            raise ValueError("A table of weights needs a Weight column.")
        weights = weights["Weight"].drop(["Portfolio", "Total"], errors="ignore")
    if isinstance(weights, str):
        pairs = [
            part.strip()
            for part in weights.replace(";", ",").split(",")
            if part.strip()
        ]
        parsed: dict[str, float] = {}
        for pair in pairs:
            if ":" not in pair:
                raise ValueError(f"Write the weights as 'TICKER:weight', not {pair!r}.")
            ticker, value = pair.split(":", 1)
            parsed[ticker.strip()] = float(value.strip().rstrip("%")) / (
                100 if value.strip().endswith("%") else 1
            )
        weights = parsed
    series = pd.Series(weights, dtype=float)
    series = series[series.abs() > 0]
    if series.empty or series.sum() <= 0:
        raise ValueError("The weights must contain at least one positive weight.")
    return (series / series.sum()).to_dict()


def presets_location() -> Path:
    """The folder with the classic portfolio definitions."""
    return Path(str(resources.files("financetoolkit.portfolio").joinpath("presets")))


def load_presets() -> dict[str, PortfolioDefinition]:
    """
    The classic portfolio definitions that ship with the Finance Toolkit, keyed by their
    identifier (the file name without .yaml).

    Returns:
        dict[str, PortfolioDefinition]: The definitions.
    """
    presets = {}
    for path in sorted(presets_location().glob("*.yaml")):
        definition = PortfolioDefinition.from_yaml(path)
        definition.identifier = path.stem
        presets[path.stem] = definition
    return presets


def _normalize(text: str) -> str:
    """A name compared without case, spaces, hyphens and underscores."""
    return "".join(
        character
        for character in text.lower()
        if character.isalnum() or character == "/"
    )


def find_preset(name: str) -> PortfolioDefinition:
    """
    A classic portfolio by its identifier, name or one of its aliases, e.g.
    "balanced_60_40", "60/40 Balanced" or "60/40".

    Args:
        name (str): The identifier, name or alias.

    Returns:
        PortfolioDefinition: The definition.

    Raises:
        ValueError: When no preset matches.
    """
    presets = load_presets()
    wanted = _normalize(name)
    for identifier, definition in presets.items():
        candidates = [identifier, definition.name, *definition.aliases]
        if wanted in {_normalize(candidate) for candidate in candidates}:
            return definition
    raise ValueError(
        f"No classic portfolio is called {name!r}. Choose from: "
        + ", ".join(
            f"{identifier} ({definition.name})"
            for identifier, definition in presets.items()
        )
    )


def _daily_periods(frame: pd.DataFrame) -> pd.DataFrame:
    """The frame on a daily PeriodIndex, the index the prices, dividends and rates share."""
    if isinstance(frame.index, pd.PeriodIndex):
        return frame
    frame = frame.copy()
    frame.index = pd.PeriodIndex(frame.index, freq="D")
    return frame


def build_transactions(
    definition: PortfolioDefinition,
    prices: pd.DataFrame,
    currencies: dict[str, str] | None = None,
    initial_value: float = 10000.0,
    years_offset: float = 0.0,
    currency: str | None = None,
    exchange_rates: pd.DataFrame | None = None,
    dividends: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    The transactions that invest a portfolio definition: the purchase of every holding on
    the first day all of them have a price, and the trades that bring the portfolio back
    to its target weights on every rebalancing date (the last trading day of each month,
    quarter or year). A schedule's weights are taken for the years since the first day plus
    the offset.

    Weights only add up when every price is in the same currency, so the prices are first
    converted to the portfolio's currency (pence to pounds included) with the exchange rate
    of each day, and every transaction is recorded in that currency.

    The dividends paid on the holdings are collected as cash and reinvested on the next
    rebalancing date: the portfolio is rebalanced to its target weights with the value of
    the holdings plus that cash, so the transactions form a total-return backtest. A
    dividend goes to the shares held at the close before its ex-dividend date. The cash
    collected after the last rebalancing date stays cash, as does all of it for a
    portfolio that is never rebalanced.

    Args:
        definition (PortfolioDefinition): The portfolio.
        prices (pd.DataFrame): The daily closing prices the transactions are made at, one
            column per ticker, each in the currency the ticker is quoted in.
        currencies (dict[str, str] | None, optional): The currency each ticker is quoted
            in, e.g. {"IGLT.L": "GBP"}. Defaults to None, the currency of each holding or
            otherwise the portfolio's currency.
        initial_value (float, optional): The amount invested at the start, in the
            portfolio's currency. Defaults to 10,000.
        years_offset (float, optional): The years to add to the time since the start when
            reading a schedule, e.g. to start a glide path part of the way. Defaults to 0.
        currency (str | None, optional): The portfolio's currency. Defaults to None, the
            currency with the largest share of the target weights (USD when no currency is
            known).
        exchange_rates (pd.DataFrame | None, optional): The daily exchange rates to convert
            the prices with, one column per rate such as USDGBP=X (the pounds per dollar).
            Defaults to None, for a portfolio whose holdings all trade in its currency.
        dividends (pd.DataFrame | None, optional): The cash dividend per share on every
            ex-dividend date, laid out as `prices` and in the same currencies. Defaults to
            None, for a portfolio without dividends to reinvest.

    Returns:
        pd.DataFrame: The transactions with Date, Identifier, Name, Price, Volume, Costs and
        Currency columns.

    Raises:
        ValueError: When a holding has no prices, when the holdings share no day with a
            price, or when a holding trades in another currency and its exchange rate is
            not available.
    """
    tickers = definition.tickers
    missing = [ticker for ticker in tickers if ticker not in prices.columns]
    if missing:
        raise ValueError(f"No prices are available for: {', '.join(missing)}.")

    known_currencies = {
        holding.ticker: (currencies or {}).get(holding.ticker) or holding.currency
        for holding in definition.holdings
    }
    known_currencies = {
        ticker: ticker_currency
        for ticker, ticker_currency in known_currencies.items()
        if ticker_currency
    }
    base_currency = portfolio_model.get_price_unit(
        currency
        or portfolio_model.get_base_currency(known_currencies, definition.weights)
    )[0]

    if exchange_rates is not None:
        exchange_rates = _daily_periods(exchange_rates)
    prices = portfolio_model.convert_prices(
        _daily_periods(prices[tickers]), known_currencies, base_currency, exchange_rates
    )

    complete = prices.dropna(how="any")
    if complete.empty:
        raise ValueError("The holdings have no day on which all of them have a price.")

    # The cumulative dividends per share in the portfolio's currency on every trading day, so the dividends between two rebalancing dates are a difference; a dividend on a day without a complete set of prices counts on the next complete day.  # noqa: E501
    if dividends is not None:
        dividends = portfolio_model.convert_prices(
            _daily_periods(
                dividends.reindex(columns=tickers).astype(float).fillna(0.0)
            ),
            known_currencies,
            base_currency,
            exchange_rates,
        ).fillna(0.0)
        cumulative_dividends = (
            dividends.cumsum()
            .reindex(dividends.index.union(complete.index))
            .ffill()
            .reindex(complete.index)
            .fillna(0.0)
        )
    else:
        cumulative_dividends = pd.DataFrame(0.0, index=complete.index, columns=tickers)

    start = complete.index[0]
    frequency = REBALANCING_FREQUENCIES[definition.rebalancing]
    if frequency:
        period_ends = complete.groupby(complete.index.asfreq(frequency)).tail(1).index
        rebalance_dates = [date for date in period_ends if date > start]
        # The current period has not ended yet, so it is not rebalanced.
        if rebalance_dates and rebalance_dates[-1] == complete.index[-1]:
            rebalance_dates = rebalance_dates[:-1]
    else:
        rebalance_dates = []

    names = {
        holding.ticker: holding.name or holding.ticker
        for holding in definition.holdings
    }

    rows = []
    shares = dict.fromkeys(tickers, 0.0)

    def trade(date: pd.Period, target: dict[str, float], value: float) -> None:
        for ticker in tickers:
            price = float(complete.loc[date, ticker])
            wanted = value * target.get(ticker, 0.0) / price
            volume = wanted - shares[ticker]
            if abs(volume) * price < MINIMUM_TRADE_VALUE:
                continue
            shares[ticker] = wanted
            rows.append(
                {
                    "Date": date.strftime("%Y-%m-%d"),
                    "Identifier": ticker,
                    "Name": names[ticker],
                    "Price": price,
                    "Volume": volume,
                    "Costs": 0.0,
                    "Currency": base_currency,
                }
            )

    trade(start, definition.weights_at(years_offset), initial_value)

    last_trade = start
    for date in rebalance_dates:
        # The dividends since the previous trade went to the shares held in between and are reinvested now.  # noqa: E501
        cash = sum(
            shares[ticker]
            * float(
                cumulative_dividends.loc[date, ticker]
                - cumulative_dividends.loc[last_trade, ticker]
            )
            for ticker in tickers
        )
        value = cash + sum(
            shares[ticker] * float(complete.loc[date, ticker]) for ticker in tickers
        )
        years = (date.to_timestamp() - start.to_timestamp()).days / 365.25
        trade(date, definition.weights_at(years + years_offset), value)
        last_trade = date

    return pd.DataFrame(rows)
