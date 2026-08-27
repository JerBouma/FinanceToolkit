"""Requests Model Tests"""

# ruff: noqa: PLR2004, SIM117

from unittest.mock import MagicMock, patch

import requests

from financetoolkit.utilities import requests_model


def test_get_request_success():
    """Test get_request returns the response on a successful call."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    with patch.object(
        requests_model.SESSION, "get", return_value=mock_response
    ) as mock_get:
        result = requests_model.get_request("https://example.com")

        assert result is mock_response
        mock_get.assert_called_once()
        assert mock_get.call_args.kwargs["verify"] is True


def test_get_request_merges_extra_headers():
    """Test get_request merges extra_headers on top of the default HEADERS."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    with patch.object(
        requests_model.SESSION, "get", return_value=mock_response
    ) as mock_get:
        requests_model.get_request(
            "https://example.com", extra_headers={"Authorization": "Bearer token"}
        )

        headers = mock_get.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer token"
        assert headers["User-Agent"] == requests_model.HEADERS["User-Agent"]


def test_get_request_ssl_fallback():
    """Test get_request retries without SSL verification on SSLError."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    with patch.object(
        requests_model.SESSION,
        "get",
        side_effect=[requests.exceptions.SSLError("bad cert"), mock_response],
    ) as mock_get:
        result = requests_model.get_request("https://example.com")

        assert result is mock_response
        assert mock_get.call_count == 2
        assert mock_get.call_args.kwargs["verify"] is False


def test_get_request_raises_on_persistent_failure():
    """Test get_request propagates the exception when the retry also fails."""
    with patch.object(
        requests_model.SESSION,
        "get",
        side_effect=requests.exceptions.ConnectionError("network down"),
    ):
        try:
            requests_model.get_request("https://example.com")
            raise AssertionError("Expected ConnectionError to propagate")
        except requests.exceptions.ConnectionError:
            pass


def test_convert_isin_to_ticker_valid_isin():
    """Test converting valid ISIN to ticker."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"quotes": [{"symbol": "AAPL"}]}
    mock_response.raise_for_status = MagicMock()

    with patch.object(requests_model.SESSION, "get", return_value=mock_response):
        with patch("financetoolkit.utilities.requests_model.logger") as mock_logger:
            result = requests_model.convert_isin_to_ticker("US0378331005")

            assert result == "AAPL"
            mock_logger.info.assert_called_once()


def test_convert_isin_to_ticker_invalid_format():
    """Test converting invalid ISIN format."""
    result = requests_model.convert_isin_to_ticker("INVALID")

    assert result == "INVALID"


def test_convert_isin_to_ticker_no_quotes():
    """Test converting ISIN when no quotes found."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"quotes": []}
    mock_response.raise_for_status = MagicMock()

    with patch.object(requests_model.SESSION, "get", return_value=mock_response):
        with patch("financetoolkit.utilities.requests_model.logger") as mock_logger:
            result = requests_model.convert_isin_to_ticker("US0378331005")

            assert result == "US0378331005"
            mock_logger.warning.assert_called_once()


def test_convert_isin_to_ticker_request_exception():
    """Test converting ISIN when request fails."""
    with patch.object(
        requests_model.SESSION,
        "get",
        side_effect=requests.exceptions.RequestException("Network error"),
    ):
        with patch("financetoolkit.utilities.requests_model.logger") as mock_logger:
            result = requests_model.convert_isin_to_ticker("US0378331005")

            assert result == "US0378331005"
            mock_logger.warning.assert_called_once()


def test_convert_isin_to_ticker_json_parse_error():
    """Test converting ISIN when JSON parsing fails."""
    mock_response = MagicMock()
    mock_response.json.side_effect = ValueError("Invalid JSON")
    mock_response.raise_for_status = MagicMock()

    with patch.object(requests_model.SESSION, "get", return_value=mock_response):
        with patch("financetoolkit.utilities.requests_model.logger") as mock_logger:
            result = requests_model.convert_isin_to_ticker("US0378331005")

            assert result == "US0378331005"
            mock_logger.warning.assert_called_once()


def test_shared_session_connection_pooling():
    """Test that the shared Session mounts enlarged connection pools for both schemes."""
    for scheme in ["https://", "http://"]:
        adapter = requests_model.SESSION.get_adapter(f"{scheme}example.com")

        assert adapter._pool_maxsize == requests_model.CONNECTION_POOL_SIZE
        assert adapter._pool_connections == requests_model.CONNECTION_POOL_SIZE


def test_build_session_returns_fresh_session():
    """Test that build_session constructs a new Session rather than the shared one."""
    session = requests_model.build_session()

    assert isinstance(session, requests.Session)
    assert session is not requests_model.SESSION
