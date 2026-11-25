"""Reads the answers of the MCP tools in the tests."""

import json


def records(table: dict) -> list[dict]:
    """A table of an answer ({"columns": [...], "rows": [[...], ...]}) as records."""
    return [dict(zip(table["columns"], row, strict=True)) for row in table["rows"]]


def read_answer(text: str):
    """
    An answer with each table as records: a list of records for a table alone, else the
    answer with the table's records under "data" next to its _notes and _sources, and a
    table per name for an answer of several tables (or of several indicators, each read the
    same way).
    """
    return _read(json.loads(text))


def _read(answer):
    """An answer, or the answer of one indicator in a call of several, with its tables as
    records."""
    if not isinstance(answer, dict):
        return answer
    if "columns" in answer and "rows" in answer:
        rest = {
            key: value
            for key, value in answer.items()
            if key not in ("columns", "rows")
        }
        return {"data": records(answer), **rest} if rest else records(answer)

    return {
        key: _read(value) if isinstance(value, dict) else value
        for key, value in answer.items()
    }
