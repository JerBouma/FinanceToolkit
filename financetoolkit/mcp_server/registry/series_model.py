"""Indicator References for the Econometrics Tool"""

__docformat__ = "google"

import re
from typing import NamedTuple

import pandas as pd

from financetoolkit.econometrics.data_model import looks_like_dates
from financetoolkit.utilities.logger_model import get_logger

logger = get_logger()

# A reference has a tool, an indicator, its entities and optionally a column; without the
# tool it starts with the indicator.
PARTS = 3
PARTS_WITH_COLUMN = 4
PARTS_WITHOUT_TOOL = 2
# Each reference collects its own indicator, so a call names a handful, not hundreds.
MAX_REFERENCES = 20
# An argument of the indicator within a reference, e.g. "cumulative=true".
ARGUMENT = re.compile(r"(?P<name>[a-z_][a-z0-9_]*)\s*=(?P<value>.*)")


class SeriesReference(NamedTuple):
    """
    An indicator of another tool to analyze as a series, written as
    "tool:indicator:entities" with an optional ":column", e.g.
    "valuation:get_price_to_earnings_ratio:AAPL,MSFT" or
    "rates:get_government_bond_yield_curve:United States:10Y". The tool may be left out
    ("get_inflation_rate:Germany") when only one tool has the indicator. Arguments of the
    indicator follow as ":name=value" parts anywhere after the indicator, e.g.
    "rates:get_outstanding_mortgage_rates:United States:Below 4%:cumulative=true".
    """

    tool: str | None
    indicator: str
    entities: list[str]
    column: str | None
    text: str
    arguments: dict[str, str] = {}


def parse_series_references(text: str) -> list[SeriesReference]:
    """
    Reads the references of the `series` parameter, separated by semicolons or new lines.

    Args:
        text (str): The references, e.g. "valuation:get_price_to_earnings_ratio:AAPL;
            macroeconomics:get_inflation_rate:United States".

    Returns:
        list[SeriesReference]: The references in the order given.

    Raises:
        ValueError: When a reference does not name an indicator and its tickers or countries,
            or gives an argument more than once.
    """
    references = []
    for raw in str(text).replace("\n", ";").split(";"):
        reference = raw.strip()
        if not reference:
            continue
        parts = [part.strip() for part in reference.split(":")]
        # The arguments of the indicator (cumulative=true) are taken out first, so the
        # other parts keep their places; a part that is not a parameter name and a value
        # stays a ticker, country or column.
        matches = [match for part in parts[1:] if (match := ARGUMENT.fullmatch(part))]
        arguments = {match["name"]: match["value"].strip() for match in matches}
        # A name given twice (window=10:window=20) would otherwise keep the last value.
        if len(arguments) < len(matches):
            repeated = [
                name
                for name in arguments
                if sum(match["name"] == name for match in matches) > 1
            ]
            raise ValueError(
                f"The series '{reference}' gives {', '.join(repeated)} more than once; "
                "give each argument once."
            )
        parts = [parts[0]] + [
            part for part in parts[1:] if not ARGUMENT.fullmatch(part)
        ]
        # A reference without a tool starts with the indicator itself.
        if parts[0].startswith("get_") or len(parts) == PARTS_WITHOUT_TOOL:
            parts = ["", *parts]
        if len(parts) not in (PARTS, PARTS_WITH_COLUMN) or not parts[1]:
            raise ValueError(
                f"The series '{reference}' is not written as tool:indicator:entities (with an "
                "optional :column and :name=value arguments), e.g. "
                "'valuation:get_price_to_earnings_ratio:AAPL,MSFT' or "
                "'macroeconomics:get_inflation_rate:United States'."
            )
        indicator = parts[1] if parts[1].startswith("get_") else f"get_{parts[1]}"
        entities = [entity.strip() for entity in parts[2].split(",") if entity.strip()]
        # Without entities the indicator would be asked for every country or none.
        if not entities:
            raise ValueError(
                f"The series '{reference}' names no tickers or countries, e.g. "
                "'macroeconomics:get_inflation_rate:United States'; an indicator of one "
                "country only takes that country, e.g. "
                "'rates:get_outstanding_mortgage_rates:United States:Average Rate'."
            )
        references.append(
            SeriesReference(
                tool=parts[0] or None,
                indicator=indicator,
                entities=entities,
                column=(
                    parts[3] if len(parts) == PARTS_WITH_COLUMN and parts[3] else None
                ),
                text=reference,
                arguments=arguments,
            )
        )
    if not references:
        raise ValueError("The series parameter names no indicator.")
    if len(references) > MAX_REFERENCES:
        raise ValueError(
            f"The series parameter names {len(references)} indicators; at most "
            f"{MAX_REFERENCES} can be analyzed in one call."
        )
    return references


def _matches(level: object, wanted: str) -> bool:
    return str(level).strip().lower() == wanted.strip().lower()


def to_series_columns(
    result: object, reference: SeriesReference, benchmark: str | None = None
) -> pd.DataFrame:
    """
    Arranges the answer of an indicator as series by date: the dates become the rows,
    whether the indicator has them in its rows (economic indicators, risk metrics) or its
    columns (ratios, models), and only the requested tickers or countries are kept (so a
    benchmark the indicator adds is left out). The optional column picks a field, such as
    a maturity of a yield curve or a component of a model. A column of several levels is
    named by its levels ("AAPL Return on Equity"), and the answer of a single ticker or
    country, which has no level naming it, is named after it as well ("AAPL Revenue").

    Args:
        result (object): The indicator's answer.
        reference (SeriesReference): The reference it answers.
        benchmark (str | None, optional): The benchmark ticker the indicator was
            calculated with. The Toolkit reports a requested ticker that is also its
            benchmark under "Benchmark", which is then read as that ticker. Defaults to
            None, no requested ticker is the benchmark.

    Returns:
        pd.DataFrame: The series, a column each, by date.

    Raises:
        ValueError: When the answer has no dates or no matching column.
    """
    if isinstance(result, pd.Series):
        name = result.name
        if name is None:
            name = reference.entities[0] if len(reference.entities) == 1 else "Value"
        frame = result.to_frame(str(name))
    elif isinstance(result, pd.DataFrame):
        frame = result
    else:
        raise ValueError(
            f"{reference.indicator} does not return a series over time, so it cannot be analyzed."
        )

    if not looks_like_dates(frame.index) and looks_like_dates(frame.columns):
        frame = frame.T
    if frame.empty or not looks_like_dates(frame.index):
        raise ValueError(
            f"{reference.indicator} for {', '.join(reference.entities)} returned no values "
            "over time to analyze."
        )

    labels = [
        column if isinstance(column, tuple) else (column,) for column in frame.columns
    ]
    if benchmark and any(_matches(benchmark, entity) for entity in reference.entities):
        labels = [
            tuple(
                benchmark if _matches(level, "Benchmark") else level for level in label
            )
            for label in labels
        ]
    named = [
        any(_matches(level, entity) for level in label for entity in reference.entities)
        for label in labels
    ]
    keep = list(named)
    # An answer for one ticker or country often has no level naming it.
    if not any(keep):
        keep = [True] * len(labels)
    elif absent := [
        entity
        for entity in reference.entities
        if not any(_matches(level, entity) for label in labels for level in label)
    ]:
        # Reaches the answer's notes, so a series left out is never left out quietly.
        logger.warning(
            f"No data is available for {', '.join(absent)} in {reference.indicator}, so "
            f"{'it is' if len(absent) == 1 else 'they are'} left out of the series."
        )
    if reference.column:
        keep = [
            kept and any(_matches(level, reference.column) for level in label)
            for kept, label in zip(keep, labels, strict=True)
        ]
        if not any(keep):
            available = sorted(
                {
                    str(level)
                    for label in labels
                    for level in label
                    if not any(_matches(level, entity) for entity in reference.entities)
                }
            )
            raise ValueError(
                f"{reference.indicator} has no column '{reference.column}'. Choose from: "
                f"{', '.join(available[:40])}."
            )

    selected = frame.loc[:, keep]
    names = []
    for label, kept, has_entity in zip(labels, keep, named, strict=True):
        if kept:
            levels = [str(level) for level in label if str(level) != ""]
            # Named after its ticker or country, so the same field of two references
            # (AAPL's and MSFT's Revenue) stays two series.
            if not has_entity and len(reference.entities) == 1:
                levels = [reference.entities[0], *levels]
            names.append(" ".join(levels) if levels else reference.entities[0])
    selected.columns = names
    return selected


def combine_series(
    frames: list[tuple[SeriesReference, pd.DataFrame]],
) -> pd.DataFrame:
    """
    Combines the series of every reference. A name that two references share (AAPL's
    price-to-earnings and price-to-book ratio) gets the indicator added to it.

    Args:
        frames (list[tuple[SeriesReference, pd.DataFrame]]): The series per reference.

    Returns:
        pd.DataFrame: All series, a column each.
    """
    counts: dict[str, int] = {}
    for _, frame in frames:
        for name in frame.columns:
            counts[name] = counts.get(name, 0) + 1

    renamed = []
    for reference, frame in frames:
        indicator = reference.indicator.removeprefix("get_").replace("_", " ").title()
        renamed.append(
            frame.rename(
                columns={
                    name: f"{name} {indicator}"
                    for name in frame.columns
                    if counts[name] > 1
                }
            )
        )
    return pd.concat(renamed, axis=1)
