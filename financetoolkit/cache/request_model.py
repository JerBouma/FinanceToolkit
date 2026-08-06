"""Request Module"""

__docformat__ = "google"

import re

# Credentials appear as query parameters on nearly every request the toolkit makes.
# They identify the caller, not the data, so they are stripped before a URL is used
# to identify a cache entry: two users asking the same question must share an entry,
# and a credential must never be written into the cache.
CREDENTIAL_PARAMETERS = ("apikey", "api_key", "token")

_CREDENTIAL_PATTERN = re.compile(
    r"([?&])(" + "|".join(CREDENTIAL_PARAMETERS) + r")=[^&]*", re.IGNORECASE
)


def redact_credentials(url: str) -> str:
    """
    Remove credential query parameters from a URL so it can identify a cache entry.

    Args:
        url (str): The request URL, possibly carrying an API key.

    Returns:
        str: The URL with every credential parameter removed, leaving the separator
            structure intact so that the remaining parameters still identify the
            request uniquely.
    """
    redacted = _CREDENTIAL_PATTERN.sub(r"\1", url)
    redacted = redacted.replace("?&", "?").replace("&&", "&")

    return redacted.rstrip("?&")
