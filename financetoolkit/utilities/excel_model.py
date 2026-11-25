"""Excel Module"""

__docformat__ = "google"

import importlib.util
import io
import warnings
from typing import Any

import pandas as pd

from financetoolkit.utilities.logger_model import get_logger

logger = get_logger()

# Calamine reads workbooks in Rust, several times faster than openpyxl for the large
# workbooks of the official sources (the Bank of England's yield curve archives, the EIOPA
# term structures, the ESRB scenarios). It is installed with the Finance Toolkit where a
# wheel exists; elsewhere, or where it cannot read a workbook, pandas' default engines
# (openpyxl for .xlsx, xlrd for .xls) read it instead.
CALAMINE_AVAILABLE = importlib.util.find_spec("python_calamine") is not None
CALAMINE = "calamine"

# The first bytes of a workbook: a zip archive (.xlsx) or an OLE2 compound file (.xls).
WORKBOOK_SIGNATURES = (b"PK\x03\x04", b"\xd0\xcf\x11\xe0")


def _check_workbook(source: Any, description: str) -> None:
    """
    Checks that downloaded bytes are a workbook, as a source that answers with an error
    page (HTML, plain text) would otherwise fail with a message about engines and formats.

    Args:
        source (Any): A path, bytes or a file-like object; only bytes are checked.
        description (str): What the workbook holds, used in the error message.

    Raises:
        ValueError: When the bytes are not an Excel workbook.
    """
    if isinstance(source, bytes | bytearray):
        start = bytes(source[:8])
    elif isinstance(source, io.BytesIO):
        start = source.getvalue()[:8]
    else:
        return

    if not start.startswith(WORKBOOK_SIGNATURES):
        raise ValueError(
            f"The {description} file is not an Excel workbook: the source answered with "
            f"something else (starting with {start!r}), such as an error page."
        )


def _rewind(source: Any) -> Any:
    """
    Returns a source that can be read again: a file-like object is rewound to its start.

    Args:
        source (Any): A path, bytes or a file-like object.

    Returns:
        Any: The source, ready to be read from the start.
    """
    if isinstance(source, bytes | bytearray):
        return io.BytesIO(source)
    if hasattr(source, "seek"):
        source.seek(0)

    return source


def _with_fallback(read, source: Any, **kwargs) -> Any:
    """
    Reads with calamine where it is available and with pandas' default engine otherwise,
    or when calamine cannot read the workbook or is older than the installed pandas
    requires.

    Args:
        read (Callable): pd.read_excel or pd.ExcelFile.
        source (Any): A path, bytes or a file-like object.
        **kwargs: The arguments for read.

    Returns:
        Any: What read returns.
    """
    if CALAMINE_AVAILABLE:
        try:
            return read(_rewind(source), engine=CALAMINE, **kwargs)
        # Any failure of the faster reader falls back to the default one, which raises the
        # error for a workbook neither can read.
        except Exception as error:  # noqa: BLE001
            logger.debug(
                "Calamine could not read the workbook (%s), so the default engine reads it.",
                error,
            )

    with warnings.catch_warnings():
        # openpyxl warns about workbook features it does not support, such as data
        # validation extensions or a missing default style, which do not affect the values.
        warnings.simplefilter("ignore", UserWarning)
        return read(_rewind(source), **kwargs)


def read_excel(
    source: Any, description: str = "downloaded", **kwargs
) -> pd.DataFrame | dict[Any, pd.DataFrame]:
    """
    Reads a worksheet (or several) of an Excel workbook (.xlsx or .xls), like pd.read_excel.

    Args:
        source (Any): A path, bytes or a file-like object, or a pd.ExcelFile of
            excel_file.
        description (str, optional): What the workbook holds, named when the bytes are not
            a workbook. Defaults to "downloaded".
        **kwargs: The arguments of pd.read_excel, such as sheet_name and header.

    Returns:
        pd.DataFrame | dict[Any, pd.DataFrame]: The worksheet, or a worksheet per name.

    Raises:
        ValueError: When the bytes are not an Excel workbook.
    """
    # A workbook that is already open is read with the engine it was opened with.
    if isinstance(source, pd.ExcelFile):
        return pd.read_excel(source, **kwargs)

    _check_workbook(source, description)

    return _with_fallback(pd.read_excel, source, **kwargs)


def excel_file(source: Any, description: str = "downloaded") -> pd.ExcelFile:
    """
    Opens an Excel workbook to read several of its worksheets, like pd.ExcelFile.

    Args:
        source (Any): A path, bytes or a file-like object.
        description (str, optional): What the workbook holds, named when the bytes are not
            a workbook. Defaults to "downloaded".

    Returns:
        pd.ExcelFile: The workbook.

    Raises:
        ValueError: When the bytes are not an Excel workbook.
    """
    _check_workbook(source, description)

    return _with_fallback(pd.ExcelFile, source)
