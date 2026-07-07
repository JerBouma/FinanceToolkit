"""Requests Module"""

__docformat__ = "google"

import re

import requests

from financetoolkit.utilities import logger_model

logger = logger_model.get_logger()

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/136.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}


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

    Raises:
        requests.exceptions.RequestException: If the request fails even without SSL verification.
    """
    headers = {**HEADERS, **(extra_headers or {})}
    try:
        response = requests.get(url, headers=headers, timeout=timeout, verify=True)
        response.raise_for_status()
        return response
    except requests.exceptions.SSLError:
        logger.warning(
            "SSL certificate verification failed for %s. Retrying without verification. "
            "This is common in corporate networks with self-signed certificates.",
            url,
        )
        response = requests.get(
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
