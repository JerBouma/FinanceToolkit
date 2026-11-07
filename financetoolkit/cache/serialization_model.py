"""Serialization Module"""

__docformat__ = "google"

import base64
import datetime
import hashlib
import json
import zlib
from typing import Any

import numpy as np
import pandas as pd

COMPRESSION_LEVEL = 6

# Payloads are JSON, never pickle: reading a pickle runs whatever code it names, so a cache
# file someone else can write (a shared directory, a database that came with a downloaded
# project, FINANCE_TOOLKIT_CACHE_DB pointed elsewhere) would run that code on the first read.
# JSON only ever gives back data. Every payload starts with this marker, so a payload in any
# other format, such as a pickle, is refused rather than read and the data is fetched again.
PAYLOAD_MARKER = b"FTJSON1\n"


def _encode_bytes(array: np.ndarray) -> str:
    """
    Store a NumPy array of numbers or booleans as base64 text of its little-endian bytes.

    Args:
        array (np.ndarray): The array.

    Returns:
        str: The base64 text.
    """
    little_endian = array.astype(array.dtype.newbyteorder("<"), copy=False)

    return base64.b64encode(np.ascontiguousarray(little_endian).tobytes()).decode()


def _decode_bytes(text: str, dtype: np.dtype) -> np.ndarray:
    """
    Rebuild an array stored by ``_encode_bytes``.

    Args:
        text (str): The base64 text.
        dtype (np.dtype): The dtype of the array.

    Returns:
        np.ndarray: A writable array of the dtype.
    """
    return np.frombuffer(base64.b64decode(text), dtype=dtype.newbyteorder("<")).astype(
        dtype
    )


def _encode_array(array: Any) -> dict[str, Any]:
    """
    Turn the values of a Series, Index or column into JSON, keeping their dtype.

    NumPy numbers and booleans are stored as their raw little-endian bytes (base64), and
    dates, durations and periods as the bytes of their integers (in the dtype's unit, or the
    period ordinal): exact, compact and fast for the long price histories the cache holds.
    Every other dtype (object, str, nullable and categorical) has each value encoded on its
    own, so a Timestamp in an object column comes back as a Timestamp.

    Args:
        array (Any): The ``.array`` of a Series or Index.

    Returns:
        dict[str, Any]: The dtype and the values.
    """
    # Only the plain wrapper of a NumPy array: the str dtype's StringArray subclasses it.
    if type(array) is pd.arrays.NumpyExtensionArray:
        array = array.to_numpy()

    dtype = array.dtype

    if isinstance(dtype, pd.CategoricalDtype):
        return {
            "dtype": "category",
            "categories": _encode_index(dtype.categories),
            "ordered": bool(dtype.ordered),
            "codes": array.codes.tolist(),
        }
    if isinstance(dtype, pd.PeriodDtype) or dtype.kind in "mM":
        return {"dtype": str(dtype), "integers": _encode_bytes(array.asi8)}
    if isinstance(dtype, np.dtype) and dtype.kind in "biuf":
        return {"dtype": str(dtype), "numbers": _encode_bytes(array)}

    return {"dtype": str(dtype), "values": [_encode(value) for value in array]}


def _decode_array(encoded: dict[str, Any]) -> Any:
    """
    Rebuild the values ``_encode_array`` stored, with their original dtype.

    Args:
        encoded (dict[str, Any]): The output of ``_encode_array``.

    Returns:
        Any: An array to build a Series or Index from.
    """
    if encoded["dtype"] == "category":
        return pd.Categorical.from_codes(
            encoded["codes"],
            dtype=pd.CategoricalDtype(
                _decode_index(encoded["categories"]), encoded["ordered"]
            ),
        )

    dtype = pd.api.types.pandas_dtype(encoded["dtype"])

    if "numbers" in encoded:
        return _decode_bytes(encoded["numbers"], dtype)
    if "integers" in encoded:
        integers = _decode_bytes(encoded["integers"], np.dtype("int64"))

        if isinstance(dtype, pd.PeriodDtype):
            return pd.arrays.PeriodArray(integers, dtype=dtype)
        if isinstance(dtype, pd.DatetimeTZDtype):
            # The integers of a time zone aware date are UTC.
            utc = pd.array(integers.view(f"M8[{dtype.unit}]")).tz_localize("UTC")
            return utc.tz_convert(dtype.tz)

        return pd.array(integers.view(dtype))

    values = [_decode(value) for value in encoded["values"]]

    if dtype == np.dtype(object):
        # Filled one by one: NumPy would turn a list of lists into a two-dimensional array.
        array = np.empty(len(values), dtype=object)
        for position, value in enumerate(values):
            array[position] = value
        return array
    if isinstance(dtype, np.dtype):
        return np.asarray(values, dtype=dtype)

    return pd.array(values, dtype=dtype)


def _decode_series(
    encoded: dict[str, Any], index: pd.Index, name: Any = None
) -> pd.Series:
    """
    Rebuild a Series (or a column) from the values ``_encode_array`` stored.

    Args:
        encoded (dict[str, Any]): The output of ``_encode_array``.
        index (pd.Index): The index of the Series.
        name (Any): The name of the Series. Defaults to None.

    Returns:
        pd.Series: The Series, with the stored dtype.
    """
    array = _decode_array(encoded)

    # The dtype is passed on, or strings in an object column would become the str dtype.
    return pd.Series(array, index=index, dtype=array.dtype, name=name)


def _encode_index(index: pd.Index) -> dict[str, Any]:
    """
    Turn an index (rows or columns) into JSON, including a MultiIndex and its names.

    Args:
        index (pd.Index): The index.

    Returns:
        dict[str, Any]: The encoded index.
    """
    if isinstance(index, pd.MultiIndex):
        return {
            "kind": "multi",
            "levels": [_encode_index(level) for level in index.levels],
            "codes": [codes.tolist() for codes in index.codes],
            "names": [_encode(name) for name in index.names],
        }
    if isinstance(index, pd.RangeIndex):
        return {
            "kind": "range",
            "range": [index.start, index.stop, index.step],
            "name": _encode(index.name),
        }

    return {"kind": "index", "name": _encode(index.name), **_encode_array(index.array)}


def _decode_index(encoded: dict[str, Any]) -> pd.Index:
    """
    Rebuild an index stored by ``_encode_index``.

    Args:
        encoded (dict[str, Any]): The encoded index.

    Returns:
        pd.Index: The index.
    """
    if encoded["kind"] == "multi":
        return pd.MultiIndex(
            levels=[_decode_index(level) for level in encoded["levels"]],
            codes=encoded["codes"],
            names=[_decode(name) for name in encoded["names"]],
        )
    if encoded["kind"] == "range":
        return pd.RangeIndex(*encoded["range"], name=_decode(encoded["name"]))

    array = _decode_array(encoded)

    # The dtype is passed on, or strings in an object index would become the str dtype.
    return pd.Index(
        array, dtype=array.dtype, name=_decode(encoded["name"]), tupleize_cols=False
    )


def _encode(value: Any) -> Any:
    """
    Turn a value into JSON that ``_decode`` turns back into the same value and type.

    Strings, integers, floats (NaN and infinity included), booleans, None and lists are
    JSON values already. Everything else becomes an object whose "$" names its type, and
    every JSON object in a payload is such a tag: that keeps a tuple apart from a list and
    the non-string keys of a dictionary intact. A type without a tag raises a TypeError, so
    the value is not cached rather than cached wrongly.

    Args:
        value (Any): The value.

    Returns:
        Any: The JSON representation.

    Raises:
        TypeError: When the value has a type that cannot be stored.
    """
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, tuple):
        return {"$": "tuple", "items": [_encode(item) for item in value]}
    if isinstance(value, dict):
        return {
            "$": "dict",
            "items": [[_encode(key), _encode(item)] for key, item in value.items()],
        }
    if isinstance(value, pd.DataFrame):
        return {
            "$": "frame",
            "index": _encode_index(value.index),
            "columns": _encode_index(value.columns),
            # By position, so duplicate column names survive.
            "data": [
                _encode_array(value.iloc[:, position].array)
                for position in range(value.shape[1])
            ],
        }
    if isinstance(value, pd.Series):
        return {
            "$": "series",
            "index": _encode_index(value.index),
            "name": _encode(value.name),
            "data": _encode_array(value.array),
        }
    if isinstance(value, pd.Index):
        return {"$": "index", "index": _encode_index(value)}
    if value is pd.NaT:
        return {"$": "NaT"}
    if value is pd.NA:
        return {"$": "NA"}
    if isinstance(value, pd.Timestamp):
        return {
            "$": "timestamp",
            # Stored in UTC, so a wall time that occurs twice (a DST change) stays exact.
            "value": (
                value.tz_convert("UTC").tz_localize(None) if value.tz else value
            ).isoformat(),
            "tz": str(value.tz) if value.tz else None,
            "unit": value.unit,
        }
    if isinstance(value, pd.Period):
        return {"$": "period", "ordinal": value.ordinal, "freq": value.freqstr}
    if isinstance(value, pd.Timedelta):
        return {"$": "timedelta", "nanoseconds": value.value, "unit": value.unit}
    if isinstance(value, datetime.datetime):
        return {"$": "datetime", "value": value.isoformat()}
    if isinstance(value, datetime.date):
        return {"$": "date", "value": value.isoformat()}
    if isinstance(value, np.generic) and value.dtype.kind in "biuf":
        return {"$": "numpy", "dtype": str(value.dtype), "value": value.item()}
    if isinstance(value, np.ndarray) and value.ndim == 1:
        return {"$": "ndarray", "data": _encode_array(pd.Series(value).array)}

    raise TypeError(f"A {type(value).__name__} cannot be stored in the cache.")


def _decode(value: Any) -> Any:
    """
    Rebuild a value stored by ``_encode``.

    Args:
        value (Any): The JSON representation.

    Returns:
        Any: The value.

    Raises:
        ValueError: When a tag is not one ``_encode`` writes.
    """
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, dict):
        return value

    tag = value["$"]

    if tag == "tuple":
        return tuple(_decode(item) for item in value["items"])
    if tag == "dict":
        return {_decode(key): _decode(item) for key, item in value["items"]}
    if tag == "frame":
        index = _decode_index(value["index"])
        frame = pd.DataFrame(
            {
                position: _decode_series(data, index)
                for position, data in enumerate(value["data"])
            },
            index=index,
        )
        frame.columns = _decode_index(value["columns"])
        return frame
    if tag == "series":
        return _decode_series(
            value["data"], _decode_index(value["index"]), _decode(value["name"])
        )
    if tag == "index":
        return _decode_index(value["index"])
    if tag == "NaT":
        return pd.NaT
    if tag == "NA":
        return pd.NA
    if tag == "timestamp":
        timestamp = pd.Timestamp(value["value"]).as_unit(value["unit"])
        return (
            timestamp.tz_localize("UTC").tz_convert(value["tz"])
            if value["tz"]
            else timestamp
        )
    if tag == "period":
        return pd.Period(ordinal=value["ordinal"], freq=value["freq"])
    if tag == "timedelta":
        return pd.Timedelta(value["nanoseconds"], unit="ns").as_unit(value["unit"])
    if tag == "datetime":
        return datetime.datetime.fromisoformat(value["value"])
    if tag == "date":
        return datetime.date.fromisoformat(value["value"])
    if tag == "numpy":
        return np.dtype(value["dtype"]).type(value["value"])
    if tag == "ndarray":
        return np.asarray(_decode_array(value["data"]))

    raise ValueError(f"Unknown cache payload tag {tag!r}.")


def encode_dataframe(data: pd.DataFrame | pd.Series) -> bytes:
    """
    Serialize a DataFrame or Series into compressed bytes for storage.

    Args:
        data (pd.DataFrame | pd.Series): The object to serialize.

    Returns:
        bytes: The zlib compressed JSON representation.

    Raises:
        TypeError: If the input is not a DataFrame or Series.
    """
    if not isinstance(data, pd.DataFrame | pd.Series):
        raise TypeError(
            f"Unsupported payload type ({type(data)}), expected a DataFrame or Series."
        )

    return encode_object(data)


def decode_dataframe(payload: bytes) -> pd.DataFrame | pd.Series:
    """
    Deserialize bytes previously produced by ``encode_dataframe``.

    Args:
        payload (bytes): The stored payload.

    Returns:
        pd.DataFrame | pd.Series: The restored object.

    Raises:
        ValueError: When the payload is not in this format or holds something else.
    """
    data = decode_object(payload)

    if not isinstance(data, pd.DataFrame | pd.Series):
        raise ValueError("The cached payload is not a DataFrame or Series.")

    return data


def encode_object(value: Any) -> bytes:
    """
    Serialize a value for storage: a DataFrame, Series or Index, a scalar (dates, periods
    and NumPy numbers included), or a list, tuple or dictionary of these.

    Args:
        value (Any): The object to serialize.

    Returns:
        bytes: The zlib compressed JSON representation.

    Raises:
        TypeError: When the value holds a type that cannot be stored.
    """
    return zlib.compress(
        PAYLOAD_MARKER + json.dumps(_encode(value), separators=(",", ":")).encode(),
        COMPRESSION_LEVEL,
    )


def decode_object(payload: bytes) -> Any:
    """
    Deserialize bytes previously produced by ``encode_object``.

    Args:
        payload (bytes): The stored payload.

    Returns:
        Any: The restored object.

    Raises:
        ValueError: When the payload is not in this format, such as a pickle. It is never
            unpickled.
    """
    content = zlib.decompress(payload)

    if not content.startswith(PAYLOAD_MARKER):
        raise ValueError("The cached payload is not in the cache's JSON format.")

    return _decode(json.loads(content[len(PAYLOAD_MARKER) :]))


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
