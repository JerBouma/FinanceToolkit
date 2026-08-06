"""Serialization Module"""

__docformat__ = "google"

import hashlib
import json
import pickle
import zlib
from typing import Any

import pandas as pd

# Pickle, not Parquet: PeriodIndex and MultiIndex columns must round-trip exactly.
COMPRESSION_LEVEL = 6


def encode_dataframe(data: pd.DataFrame | pd.Series) -> bytes:
    """
    Serialize a DataFrame or Series into compressed bytes for storage.

    Args:
        data (pd.DataFrame | pd.Series): The object to serialize.

    Returns:
        bytes: The zlib compressed pickle representation.

    Raises:
        TypeError: If the input is not a DataFrame or Series.
    """
    if not isinstance(data, pd.DataFrame | pd.Series):
        raise TypeError(
            f"Unsupported payload type ({type(data)}), expected a DataFrame or Series."
        )

    return zlib.compress(
        pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL), COMPRESSION_LEVEL
    )


def decode_dataframe(payload: bytes) -> pd.DataFrame | pd.Series:
    """
    Deserialize bytes previously produced by ``encode_dataframe``.

    Args:
        payload (bytes): The stored payload.

    Returns:
        pd.DataFrame | pd.Series: The restored object.
    """
    return pickle.loads(zlib.decompress(payload))  # noqa: S301


def encode_object(value: Any) -> bytes:
    """
    Serialize an arbitrary picklable object, such as a dictionary of metadata.

    Args:
        value (Any): The object to serialize.

    Returns:
        bytes: The zlib compressed pickle representation.
    """
    return zlib.compress(
        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL), COMPRESSION_LEVEL
    )


def decode_object(payload: bytes) -> Any:
    """
    Deserialize bytes previously produced by ``encode_object``.

    Args:
        payload (bytes): The stored payload.

    Returns:
        Any: The restored object.
    """
    return pickle.loads(zlib.decompress(payload))  # noqa: S301


def create_cache_key(source: str, dataset: str, parameters: dict[str, Any]) -> str:
    """
    Build the deterministic key that identifies a cached dataset.

    The key deliberately excludes the requested date range and the list of
    entities: those are tracked separately so that widening a date range or
    adding a ticker reuses everything already stored instead of invalidating it.
    Only parameters that genuinely change the shape or meaning of the returned
    data (interval, period, source, currency, and so on) belong here.

    Args:
        source (str): The external data source, e.g. "fmp" or "oecd".
        dataset (str): The dataset within that source, e.g. "historical".
        parameters (dict[str, Any]): Parameters that alter the returned data.

    Returns:
        str: A SHA256 hex digest uniquely identifying this dataset variant.
    """
    canonical = json.dumps(
        {"source": source, "dataset": dataset, "parameters": parameters},
        sort_keys=True,
        default=str,
    )

    return hashlib.sha256(canonical.encode()).hexdigest()
