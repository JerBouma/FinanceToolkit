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


def canonicalize(value: Any) -> Any:
    """
    Rewrite a parameter value into a form that hashes to exactly one key.

    A cache key is only trustworthy if it is a total function of the arguments:
    every value has to reach it, and two values that mean different things have
    to reach it differently. Plain JSON does neither. It cannot represent a set,
    a tuple or a date at all, so those would have to be dropped or stringified,
    and it renders a tuple and a list identically and a non-string dictionary key
    as its string form, so ``{1: "a"}`` and ``{"1": "a"}`` would collide.

    Containers are therefore tagged with their type, mappings are sorted by their
    canonical key so ordering cannot change the digest, sets are sorted so their
    iteration order cannot either, and anything else is tagged with its class name
    beside its representation so two unrelated objects that happen to print the
    same do not merge. Booleans are handled before integers because ``True`` and
    ``1`` are equal in Python and must not be for a key.

    Args:
        value (Any): The value to rewrite. Any object is accepted.

    Returns:
        Any: A JSON-serializable structure that is unique to the value.
    """
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", value]
    if isinstance(value, float):
        return ["float", repr(value)]
    if isinstance(value, dict):
        items = sorted(
            ([canonicalize(key), canonicalize(item)] for key, item in value.items()),
            key=lambda pair: json.dumps(pair[0]),
        )

        return ["dict", items]
    if isinstance(value, tuple):
        return ["tuple", [canonicalize(item) for item in value]]
    if isinstance(value, list):
        return ["list", [canonicalize(item) for item in value]]
    if isinstance(value, set | frozenset):
        return ["set", sorted((canonicalize(item) for item in value), key=json.dumps)]

    return [type(value).__name__, str(value)]


def create_cache_key(source: str, dataset: str, parameters: dict[str, Any]) -> str:
    """
    Build the deterministic key that identifies a cached dataset.

    The key deliberately excludes the requested date range and the list of
    entities: those are tracked separately so that widening a date range or
    adding a ticker reuses everything already stored instead of invalidating it.
    Only parameters that genuinely change the shape or meaning of the returned
    data (interval, period, source, currency, and so on) belong here.

    Every parameter passes through ``canonicalize`` first, so nested structures,
    lists and dictionaries all reach the digest and reach it in a fixed order.

    Args:
        source (str): The external data source, e.g. "fmp" or "oecd".
        dataset (str): The dataset within that source, e.g. "historical".
        parameters (dict[str, Any]): Parameters that alter the returned data.

    Returns:
        str: A SHA256 hex digest uniquely identifying this dataset variant.
    """
    canonical = json.dumps(
        {
            "source": source,
            "dataset": dataset,
            "parameters": canonicalize(parameters),
        },
        sort_keys=True,
    )

    return hashlib.sha256(canonical.encode()).hexdigest()
