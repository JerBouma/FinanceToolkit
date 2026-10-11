"""Tests for reading Excel workbooks with calamine and the default engine as fallback."""

import io

import pandas as pd
import pytest

from financetoolkit.utilities import excel_model


def _workbook() -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame({"Date": ["2026-10-06"], "Price": [84.31]}).to_excel(
            writer, sheet_name="Auctions", index=False
        )
        pd.DataFrame([[1, 2], [3, 4]]).to_excel(
            writer, sheet_name="Grid", header=False, index=False
        )
    return buffer.getvalue()


@pytest.mark.skipif(
    not excel_model.CALAMINE_AVAILABLE, reason="python-calamine is not installed"
)
def test_calamine_reads_the_same_values_as_the_default_engine(monkeypatch):
    workbook = _workbook()
    engines = []
    original = pd.read_excel

    def recording(source, **kwargs):
        engines.append(kwargs.get("engine"))
        return original(source, **kwargs)

    monkeypatch.setattr(excel_model.pd, "read_excel", recording)
    with_calamine = excel_model.read_excel(workbook, sheet_name="Auctions")

    monkeypatch.setattr(excel_model, "CALAMINE_AVAILABLE", False)
    without_calamine = excel_model.read_excel(workbook, sheet_name="Auctions")

    assert engines == ["calamine", None]
    pd.testing.assert_frame_equal(with_calamine, without_calamine)


def test_the_default_engine_reads_what_calamine_cannot(monkeypatch):
    calls = []
    original = pd.read_excel

    def failing_calamine(source, **kwargs):
        calls.append(kwargs.get("engine"))
        if kwargs.get("engine") == "calamine":
            raise ImportError("python-calamine is older than pandas requires")
        return original(source, **kwargs)

    monkeypatch.setattr(excel_model, "CALAMINE_AVAILABLE", True)
    monkeypatch.setattr(excel_model.pd, "read_excel", failing_calamine)

    grid = excel_model.read_excel(
        io.BytesIO(_workbook()), sheet_name="Grid", header=None
    )

    # The file-like object is rewound before the second attempt.
    assert calls == ["calamine", None]
    assert grid.to_numpy().tolist() == [[1, 2], [3, 4]]


def test_a_workbook_neither_engine_reads_raises(monkeypatch):
    monkeypatch.setattr(excel_model, "CALAMINE_AVAILABLE", True)

    with pytest.raises(ValueError):
        excel_model.read_excel(_workbook(), sheet_name="Missing")


def test_an_open_workbook_reads_several_worksheets(tmp_path):
    path = tmp_path / "workbook.xlsx"
    path.write_bytes(_workbook())

    workbook = excel_model.excel_file(str(path))

    assert workbook.sheet_names == ["Auctions", "Grid"]
    assert (
        excel_model.read_excel(workbook, sheet_name="Auctions")["Price"].iloc[0]
        == 84.31
    )
