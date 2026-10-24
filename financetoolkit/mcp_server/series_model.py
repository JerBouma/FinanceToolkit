"""Indicator References for the Econometrics Tool"""

__docformat__ = "google"

from typing import NamedTuple

import pandas as pd

from financetoolkit.econometrics.data_model import looks_like_dates

# A reference has a tool, an indicator, its entities and optionally a column; without the
# tool it starts with the indicator.
PARTS = 3
PARTS_WITH_COLUMN = 4
PARTS_WITHOUT_TOOL = 2


class SeriesReference(NamedTuple):
    """
    An indicator of another tool to analyze as a series, written as
    "tool:indicator:entities" with an optional ":column", e.g.
    "valuation:get_price_to_earnings_ratio:AAPL,MSFT" or
    "rates:get_government_bond_yield_curve:United States:10Y". The tool may be left out
    ("get_inflation_rate:Germany") when only one tool has the indicator.
    """

    tool: str | None
    indicator: str
    entities: list[str]
    column: str | None
    text: str


def parse_series_references(text: str) -> list[SeriesReference]:
    """
    Reads the references of the `series` parameter, separated by semicolons or new lines.

    Args:
        text (str): The references, e.g. "valuation:get_price_to_earnings_ratio:AAPL;
            macroeconomics:get_inflation_rate:United States".

    Returns:
        list[SeriesReference]: The references in the order given.

    Raises:
        ValueError: When a reference does not name an indicator and its tickers or countries.
    """
    references = []
    for raw in str(text).replace("\n", ";").split(";"):
        reference = raw.strip()
        if not reference:
            continue
        parts = [part.strip() for part in reference.split(":")]
        # A reference without a tool starts with the indicator itself.
        if parts[0].startswith("get_") or len(parts) == PARTS_WITHOUT_TOOL:
            parts = ["", *parts]
        if len(parts) not in (PARTS, PARTS_WITH_COLUMN) or not parts[1] or not parts[2]:
            raise ValueError(
                f"The series '{reference}' is not written as tool:indicator:entities (with an "
                "optional :column), e.g. 'valuation:get_price_to_earnings_ratio:AAPL,MSFT' or "
                "'macroeconomics:get_inflation_rate:United States'."
            )
        indicator = parts[1] if parts[1].startswith("get_") else f"get_{parts[1]}"
        references.append(
            SeriesReference(
                tool=parts[0] or None,
                indicator=indicator,
                entities=[
                    entity.strip() for entity in parts[2].split(",") if entity.strip()
                ],
                column=(
                    parts[3] if len(parts) == PARTS_WITH_COLUMN and parts[3] else None
                ),
                text=reference,
            )
        )
    if not references:
        raise ValueError("The series parameter names no indicator.")
    return references


def _matches(level: object, wanted: str) -> bool:
    return str(level).strip().lower() == wanted.strip().lower()


def to_series_columns(result: object, reference: SeriesReference) -> pd.DataFrame:
    """
    Arranges the answer of an indicator as series by date: the dates become the rows,
    whether the indicator has them in its rows (economic indicators, risk metrics) or its
    columns (ratios, models), and only the requested tickers or countries are kept (so a
    benchmark the indicator adds is left out). The optional column picks a field, such as
    a maturity of a yield curve or a component of a model. A column of several levels is
    named by its levels ("AAPL Return on Equity").

    Args:
        result (object): The indicator's answer.
        reference (SeriesReference): The reference it answers.

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
    keep = [
        any(_matches(level, entity) for level in label for entity in reference.entities)
        for label in labels
    ]
    # An answer for one ticker or country often has no level naming it.
    if not any(keep):
        keep = [True] * len(labels)
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
    for label, kept in zip(labels, keep, strict=True):
        if kept:
            levels = [str(level) for level in label if str(level) != ""]
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
