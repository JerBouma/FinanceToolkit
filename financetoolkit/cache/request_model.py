"""Request Module"""

__docformat__ = "google"

import re

from financetoolkit.utilities.logger_model import CREDENTIAL_PATTERN, redact

# Credentials identify the caller, not the data, so they never reach a cache key.
CREDENTIAL_PARAMETERS = ("apikey", "api_key", "token")

# A query parameter with its value, the parameter's name in the second group.
_QUERY_PARAMETER = re.compile(r"([?&])([^?&=#\s]*)=[^&#\s]*")


def redact_credentials(url: str) -> str:
    """
    Remove credential query parameters from a URL so it can identify a cache entry or
    be shown in a message. A credential is what logger_model.CREDENTIAL_PATTERN redacts
    in the log: a parameter named key, apikey, api_key, token, secret or password, also
    after a prefix and an underscore (fred_api_key, access_token, ...), so a message never
    shows a key the log would hide.

    Args:
        url (str): The request URL, possibly carrying an API key.

    Returns:
        str: The URL with every credential parameter removed, leaving the separator
            structure intact so that the remaining parameters still identify the
            request uniquely. A credential outside a query string is replaced by "***".
    """
    redacted = _QUERY_PARAMETER.sub(
        # The pattern needs a value after the name, which "x" stands in for.
        lambda match: (
            match.group(1)
            if CREDENTIAL_PATTERN.search(f"{match.group(2)}=x")
            else match.group(0)
        ),
        url,
    )
    redacted = redacted.replace("?&", "?").replace("&&", "&").rstrip("?&")

    return redact(redacted)
