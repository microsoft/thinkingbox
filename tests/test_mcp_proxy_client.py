# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from thinkingbox.common.config_types import (
    ApiKeyCredentialConfig,
    SessionProxyConfig,
)
from thinkingbox.common.mcp_proxy_client import MCPProxyClient


def get_mock_async_client():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.json = MagicMock(return_value={"content": {"value": {"tools": {}}}})
    mock_response.raise_for_status = MagicMock()
    mock_response.status_code = 200
    mock_client.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
    return mock_client


def get_base_config(**kwargs) -> SessionProxyConfig:
    defaults = {"endpoint_url": "https://test.endpoint.com"}
    defaults.update(kwargs)
    return SessionProxyConfig(**defaults)


@pytest.mark.asyncio
@pytest.mark.parametrize("use_certificate", [True, False])
@patch("thinkingbox.common.http_client.Path.is_file", return_value=True)
@patch("thinkingbox.common.http_client.ssl.create_default_context")
@patch("thinkingbox.common.http_client.httpx.AsyncClient")
async def test_client_certificate(
    mock_async_client, mock_ssl_context, _, use_certificate
):
    mock_async_client.return_value = get_mock_async_client()

    mock_ssl_instance = MagicMock()
    mock_ssl_context.return_value = mock_ssl_instance

    client_certificate = "/path/to/client.pem" if use_certificate else None
    config = get_base_config(client_certificate=client_certificate)

    async with MCPProxyClient.session_context_from_config(
        config=config,
        server_config={},
        available_tools=[],
    ) as ctx:
        if use_certificate:
            assert ctx.client.ssl_context is mock_ssl_instance
        else:
            assert ctx.client.ssl_context is None

    if use_certificate:
        mock_ssl_context.assert_called()
    else:
        mock_ssl_context.assert_not_called()

    mock_async_client.assert_called()
    _, kwargs = mock_async_client.call_args
    if use_certificate:
        assert kwargs.get("verify") is mock_ssl_instance
    else:
        assert kwargs.get("verify", True) is True


@pytest.mark.asyncio
@pytest.mark.parametrize("use_trust_ca", [True, False])
@patch("thinkingbox.common.http_client.Path.is_file", return_value=True)
@patch("thinkingbox.common.http_client.Path.is_dir", return_value=False)
@patch("thinkingbox.common.http_client.ssl.create_default_context")
@patch("thinkingbox.common.http_client.httpx.AsyncClient")
async def test_trust_ca_path(mock_async_client, mock_ssl_context, _, __, use_trust_ca):
    mock_async_client.return_value = get_mock_async_client()

    mock_ssl_instance = MagicMock()
    mock_ssl_context.return_value = mock_ssl_instance

    trust_ca_path = "/path/to/ca.pem" if use_trust_ca else None
    config = get_base_config(trust_ca_path=trust_ca_path)

    async with MCPProxyClient.session_context_from_config(
        config=config,
        server_config={},
        available_tools=[],
    ) as ctx:
        if use_trust_ca:
            assert ctx.client.ssl_context is mock_ssl_instance
        else:
            assert ctx.client.ssl_context is None

    if use_trust_ca:
        mock_ssl_context.assert_called()
        mock_ssl_instance.load_verify_locations.assert_called()
    else:
        mock_ssl_context.assert_not_called()


def test_client_certificate_file_does_not_exist():
    config = get_base_config(client_certificate="/invalid/path/to/client.pem")
    with pytest.raises(ValueError, match="not found"):
        MCPProxyClient(
            endpoint=config.endpoint_url,
            client_certificate=config.client_certificate,
        )


def test_trust_ca_path_does_not_exist():
    config = get_base_config(trust_ca_path="/invalid/path/to/ca.pem")
    with pytest.raises(ValueError, match="not found"):
        MCPProxyClient(
            endpoint=config.endpoint_url,
            trust_ca_path=config.trust_ca_path,
        )


@pytest.mark.asyncio
@patch("thinkingbox.common.http_client.httpx.AsyncClient")
async def test_headers_passed_to_client(mock_async_client):
    mock_async_client.return_value = get_mock_async_client()

    headers = {"X-Custom-Header": "test-value", "Authorization": "Bearer token"}
    config = get_base_config(headers=headers)

    async with MCPProxyClient.session_context_from_config(
        config=config,
        server_config={},
        available_tools=[],
    ) as ctx:
        assert ctx.client.headers == headers

    mock_async_client.assert_called()
    _, kwargs = mock_async_client.call_args
    assert kwargs.get("headers") == headers


@pytest.mark.asyncio
@patch("thinkingbox.common.http_client.httpx.AsyncClient")
async def test_credential_from_config(mock_async_client):
    mock_async_client.return_value = get_mock_async_client()

    config = get_base_config(credential=ApiKeyCredentialConfig(api_key="test-api-key"))

    async with MCPProxyClient.session_context_from_config(
        config=config,
        server_config={},
        available_tools=[],
    ) as ctx:
        assert ctx.client.credential is not None


@pytest.mark.asyncio
@patch("thinkingbox.common.http_client.httpx.AsyncClient")
@patch("thinkingbox.common.mcp_proxy_client.BackoffAsyncClient")
async def test_retry_config_passed_to_backoff_client(
    mock_backoff_client, mock_async_client
):
    mock_async_client.return_value = get_mock_async_client()

    mock_backoff_instance = MagicMock()
    mock_backoff_instance.post = AsyncMock(
        return_value=MagicMock(
            json=MagicMock(return_value={"content": {"value": {"tools": {}}}}),
            status_code=200,
        )
    )
    mock_backoff_client.return_value = mock_backoff_instance

    max_retries = 10
    retryable_errors = (500, 502, 503)
    config = get_base_config(
        max_retries_server_error=max_retries,
        retryable_server_errors=retryable_errors,
    )

    async with MCPProxyClient.session_context_from_config(
        config=config,
        server_config={},
        available_tools=[],
    ) as ctx:
        assert ctx.client.max_retries_server_error == max_retries
        assert ctx.client.retryable_server_errors == retryable_errors

    mock_backoff_client.assert_called()
    _, kwargs = mock_backoff_client.call_args
    assert kwargs.get("max_retries_server_error") == max_retries
    assert kwargs.get("retryable_server_errors") == retryable_errors
    assert kwargs.get("max_retries_timeout") == 0


@pytest.mark.asyncio
@patch("thinkingbox.common.http_client.httpx.AsyncClient")
@patch("thinkingbox.common.mcp_proxy_client.BackoffAsyncClient")
async def test_credential_passed_to_backoff_client_on_request(
    mock_backoff_client, mock_async_client
):
    mock_async_client.return_value = get_mock_async_client()

    mock_backoff_instance = MagicMock()
    mock_backoff_instance.post = AsyncMock(
        return_value=MagicMock(
            json=MagicMock(return_value={"content": {"value": {"tools": {}}}}),
            status_code=200,
        )
    )
    mock_backoff_client.return_value = mock_backoff_instance

    config = get_base_config(credential=ApiKeyCredentialConfig(api_key="test-api-key"))

    async with MCPProxyClient.session_context_from_config(
        config=config,
        server_config={},
        available_tools=[],
    ):
        pass

    mock_backoff_client.assert_called()
    _, kwargs = mock_backoff_client.call_args
    assert kwargs.get("credential") is not None


@pytest.mark.asyncio
@patch("thinkingbox.common.http_client.httpx.AsyncClient")
async def test_timeout_config(mock_async_client):
    mock_async_client.return_value = get_mock_async_client()

    timeout = 300.0
    config = get_base_config(timeout=timeout)

    async with MCPProxyClient.session_context_from_config(
        config=config,
        server_config={},
        available_tools=[],
    ) as ctx:
        assert ctx.client.timeout_config.read == timeout
        assert ctx.client.timeout_config.connect == 30.0
        assert ctx.client.timeout_config.write == 30.0


@pytest.mark.asyncio
@patch("thinkingbox.common.http_client._g_dns_cache", new=MagicMock())
@patch("thinkingbox.common.http_client.DNSCacheTransport")
@patch("thinkingbox.common.http_client.httpx.AsyncClient")
async def test_use_dns_cache_config(mock_async_client, _):
    mock_async_client.return_value = get_mock_async_client()

    config = get_base_config(use_dns_cache=True)

    async with MCPProxyClient.session_context_from_config(
        config=config,
        server_config={},
        available_tools=[],
    ) as ctx:
        assert ctx.client.use_dns_cache is True
