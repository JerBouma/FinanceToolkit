"""Requests Module"""

__docformat__ = "google"

import importlib.util
import os
import re
import ssl

import requests
from requests.adapters import HTTPAdapter

from financetoolkit.cache.request_model import CREDENTIAL_PARAMETERS, redact_credentials
from financetoolkit.utilities import logger_model

logger = logger_model.get_logger()

BROTLI_AVAILABLE = bool(
    importlib.util.find_spec("brotli") or importlib.util.find_spec("brotlicffi")
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/136.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    # Brotli is only offered when a decoder is installed: requests cannot decode it on
    # its own, and a server that prefers it would otherwise answer with unreadable bytes.
    "Accept-Encoding": "gzip, deflate, br" if BROTLI_AVAILABLE else "gzip, deflate",
    "Connection": "keep-alive",
}

# Some official sources (the Reserve Bank of Australia, De Nederlandsche Bank) turn away
# requests that present themselves as a browser without being one, so for those the
# Finance Toolkit names itself.
TOOLKIT_HEADERS = {
    "User-Agent": "financetoolkit (+https://github.com/JerBouma/FinanceToolkit)"
}

# Set to 1 to send requests that carry an API key without verifying the certificate when
# verification fails, which exposes the key to whoever intercepts the connection.
# Pointing REQUESTS_CA_BUNDLE to the certificate of a corporate proxy is the safe fix.
UNVERIFIED_SSL_ENVIRONMENT_VARIABLE = "FINANCETOOLKIT_ALLOW_UNVERIFIED_SSL"

CREDENTIAL_IN_URL = re.compile(
    r"[?&](" + "|".join(CREDENTIAL_PARAMETERS) + r")=", re.IGNORECASE
)

# Sized comfortably above the default number of worker threads (see helpers.DEFAULT_MAX_WORKERS) so that every concurrent API call can keep its connection alive; a larger worker count still works, urllib3 simply discards the surplus connections after use.  # noqa: E501
CONNECTION_POOL_SIZE = 32


def build_session() -> requests.Session:
    """
    Builds the shared Session that every request in the Finance Toolkit goes through.

    A single Session reuses TCP connections and TLS handshakes across calls to the same
    host, which matters a great deal here: collecting data for a large ticker universe
    means hundreds of requests to the same handful of API hosts, and without pooling
    every single one would pay for its own handshake. The Session is created once at
    import time and is safe to share across the worker threads used for concurrent
    API calls, since it is never mutated afterwards.

    Returns:
        requests.Session: The configured Session with enlarged connection pools.
    """
    session = requests.Session()

    adapter = HTTPAdapter(
        pool_connections=CONNECTION_POOL_SIZE,
        pool_maxsize=CONNECTION_POOL_SIZE,
    )

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


SESSION = build_session()


def get_request(
    url: str,
    timeout: int = 60,
    extra_headers: dict | None = None,
) -> requests.Response:
    """
    Make an HTTP GET request with automatic SSL fallback for corporate proxies
    and environments with self-signed certificates.

    Args:
        url (str): The URL to request.
        timeout (int): Request timeout in seconds.
        extra_headers (dict | None): Additional headers merged on top of the default HEADERS,
            e.g. {"Authorization": "Bearer <token>"}. Defaults to None.

    Returns:
        requests.Response: The HTTP response object.

    A request that carries an API key is not retried without verification unless
    FINANCETOOLKIT_ALLOW_UNVERIFIED_SSL is set to 1, since an intercepted connection would
    hand over the key; REQUESTS_CA_BUNDLE can point to a corporate proxy's certificate.

    Raises:
        requests.exceptions.SSLError: If the certificate cannot be verified for a request
            that carries an API key.
        requests.exceptions.RequestException: If the request fails even without SSL verification.
    """
    headers = {**HEADERS, **(extra_headers or {})}
    try:
        response = SESSION.get(url, headers=headers, timeout=timeout, verify=True)
        response.raise_for_status()
        return response
    except requests.exceptions.SSLError:
        # certifi has dropped some older roots that servers still chain to, such as
        # Comodo's "AAA Certificate Services" (STOXX), while the operating system still
        # trusts them, so the system's certificates are tried before not verifying.
        system_certificates = ssl.get_default_verify_paths().cafile
        if system_certificates and os.path.exists(system_certificates):
            try:
                response = SESSION.get(
                    url, headers=headers, timeout=timeout, verify=system_certificates
                )
                response.raise_for_status()
                return response
            except requests.exceptions.SSLError:
                pass

        carries_credentials = bool(CREDENTIAL_IN_URL.search(url)) or any(
            header.lower() == "authorization" for header in (extra_headers or {})
        )

        if (
            carries_credentials
            and os.environ.get(UNVERIFIED_SSL_ENVIRONMENT_VARIABLE) != "1"
        ):
            # An unverified connection could be intercepted, which would hand over the
            # API key, so a request carrying one is not retried without verification.
            logger.error(
                "SSL certificate verification failed for %s. The request carries an API "
                "key and is therefore not retried without verification. In a corporate "
                "network with its own certificates, set REQUESTS_CA_BUNDLE to the path of "
                "that certificate, or set %s=1 to retry without verification regardless.",
                redact_credentials(url),
                UNVERIFIED_SSL_ENVIRONMENT_VARIABLE,
            )
            raise

        logger.warning(
            "SSL certificate verification failed for %s. Retrying without verification. "
            "This is common in corporate networks with self-signed certificates.",
            redact_credentials(url),
        )
        response = SESSION.get(
            url, headers=headers, timeout=timeout, verify=False  # noqa
        )
        response.raise_for_status()
        return response


def convert_isin_to_ticker(isin_code: str) -> str:
    """
    Converts an ISIN code to a ticker symbol using Yahoo Finance search.

    Args:
        isin_code (str): The ISIN code to convert.

    Returns:
        str: The corresponding ticker symbol if found, otherwise the original ISIN code.
    """
    if bool(re.match("^([A-Z]{2})([A-Z0-9]{9})([0-9])$", isin_code)):
        try:
            response = get_request(
                f"https://query2.finance.yahoo.com/v1/finance/search?q={isin_code}",
                timeout=60,
            )

            data = response.json()

            if data.get("quotes"):
                symbol = data["quotes"][0]["symbol"]
                logger.info("Converted ISIN %s to ticker %s", isin_code, symbol)

                return symbol

            logger.warning(
                "Could not find a ticker for ISIN %s. Returning ISIN.", isin_code
            )
            return isin_code

        except requests.exceptions.RequestException as e:
            logger.warning(
                "Request failed for ISIN %s: %s. Returning ISIN.", isin_code, e
            )
            return isin_code
        except (KeyError, ValueError, IndexError):
            logger.warning(
                "Could not parse response for ISIN %s. Returning ISIN.", isin_code
            )
            return isin_code
    else:
        # If it's not a valid ISIN format, return the original input
        return isin_code
