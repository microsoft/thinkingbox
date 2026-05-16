# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import contextlib

import httpx
import pytest

from thinkingbox.common import http_client
from thinkingbox.common.http_client import (
    BackoffAsyncClient,
    DNSCacheTransport,
    get_httpx_client,
    initialize_dns_cache,
)
from thinkingbox.common.utils import CredentialBase

DUMMY_URL = "http://localhost/endpoint"


class DummyCredential(CredentialBase):
    def __init__(self, tokens):
        self._tokens = tokens
        self.get_token_calls = 0
        self.invalidate_calls = 0

    async def get_token(self) -> str:
        token = self._tokens[min(self.get_token_calls, len(self._tokens) - 1)]
        self.get_token_calls += 1
        return token

    async def invalidate(self) -> None:
        self.invalidate_calls += 1


class StubAsyncClient:
    """
    Simple programmable stub for httpx.AsyncClient.post
    Provide a sequence (list/iterable) of httpx.Response objects or Exceptions.
    """

    def __init__(self, sequence):
        self._iter = iter(sequence)
        self.calls = []  # list of (url, headers, kwargs)

    async def post(self, url, headers=None, **kwargs):
        self.calls.append((url, headers or {}, kwargs))
        nxt = next(self._iter)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class SleepRecorder:
    def __init__(self):
        self.calls = []

    async def __call__(self, delay: float):
        self.calls.append(delay)
        # simulate immediate passage of time
        return


def _make_response(
    status: int,
    url: str = DUMMY_URL,
    headers=None,
    response_headers=None,
) -> httpx.Response:
    req = httpx.Request("POST", url, headers=headers)
    return httpx.Response(status, request=req, headers=response_headers)


@pytest.mark.asyncio
async def test_post_success_no_retry(monkeypatch):
    stub = StubAsyncClient([_make_response(200)])
    client = BackoffAsyncClient(
        stub,
        credential=None,
        jitter=False,
        factor=0.2,
    )

    sleep_rec = SleepRecorder()
    monkeypatch.setattr("thinkingbox.common.http_client.asyncio.sleep", sleep_rec)

    resp = await client.post(DUMMY_URL, json={"k": "v"})
    resp.raise_for_status()
    assert resp.status_code == 200
    assert len(stub.calls) == 1
    assert sleep_rec.calls == []  # no retries


@pytest.mark.asyncio
async def test_retry_on_429_then_success(monkeypatch):
    # First attempt 429, second 200
    seq = [
        _make_response(429),
        _make_response(200),
    ]
    stub = StubAsyncClient(seq)
    client = BackoffAsyncClient(
        stub,
        credential=None,
        max_retries_too_many_requests=3,
        jitter=False,
        factor=0.5,
    )
    sleep_rec = SleepRecorder()
    monkeypatch.setattr("thinkingbox.common.http_client.asyncio.sleep", sleep_rec)

    resp = await client.post(DUMMY_URL)
    resp.raise_for_status()
    assert resp.status_code == 200
    assert len(stub.calls) == 2
    # One backoff sleep with factor (since retry_index=1)
    assert sleep_rec.calls == [0.5]


@pytest.mark.asyncio
async def test_429_exhausts_budget(monkeypatch):
    # Budget: max_retries_too_many_requests = 1 (only 1 retry allowed)
    # Sequence: 429, 429 -> second failure should raise
    seq = [
        _make_response(429),
        _make_response(429),
    ]
    stub = StubAsyncClient(seq)
    client = BackoffAsyncClient(
        stub,
        credential=None,
        max_retries_too_many_requests=1,
        jitter=False,
        factor=0.1,
    )
    sleep_rec = SleepRecorder()
    monkeypatch.setattr("thinkingbox.common.http_client.asyncio.sleep", sleep_rec)

    resp = await client.post(DUMMY_URL)
    with pytest.raises(httpx.HTTPStatusError) as exc:
        resp.raise_for_status()
    assert exc.value.response.status_code == 429
    assert len(stub.calls) == 2
    # Only first retry had a sleep
    assert sleep_rec.calls == [0.1]


@pytest.mark.asyncio
async def test_429_with_retry_after_header(monkeypatch):
    # 429 with Retry-After: 5 seconds, then success
    # Should use max(exponential_delay, retry_after) capped by max_delay
    seq = [
        _make_response(429, response_headers={"Retry-After": "5"}),
        _make_response(200),
    ]
    stub = StubAsyncClient(seq)
    client = BackoffAsyncClient(
        stub,
        credential=None,
        max_retries_too_many_requests=3,
        jitter=False,
        factor=1.0,  # Would give delay of 1.0 for first retry
        base=2.0,
        max_delay=10.0,
    )
    sleep_rec = SleepRecorder()
    monkeypatch.setattr("thinkingbox.common.http_client.asyncio.sleep", sleep_rec)

    resp = await client.post(DUMMY_URL)
    resp.raise_for_status()
    assert resp.status_code == 200
    assert len(stub.calls) == 2
    # Should respect Retry-After: 5 seconds (max of 1.0 exponential and 5.0 from header)
    assert sleep_rec.calls == [5.0]


@pytest.mark.asyncio
async def test_429_retry_after_exceeds_max_delay(monkeypatch):
    # 429 with Retry-After: 100 seconds, but max_delay is 10.0
    # Should cap at max_delay
    seq = [
        _make_response(429, response_headers={"Retry-After": "100"}),
        _make_response(200),
    ]
    stub = StubAsyncClient(seq)
    client = BackoffAsyncClient(
        stub,
        credential=None,
        max_retries_too_many_requests=3,
        jitter=False,
        factor=1.0,
        base=2.0,
        max_delay=10.0,
    )
    sleep_rec = SleepRecorder()
    monkeypatch.setattr("thinkingbox.common.http_client.asyncio.sleep", sleep_rec)

    resp = await client.post(DUMMY_URL)
    resp.raise_for_status()
    assert resp.status_code == 200
    assert len(stub.calls) == 2
    # Should cap at max_delay even though Retry-After says 100
    assert sleep_rec.calls == [10.0]


@pytest.mark.asyncio
async def test_429_retry_after_invalid_value(monkeypatch):
    # 429 with invalid Retry-After header, should fall back to exponential backoff
    seq = [
        _make_response(429, headers={"Retry-After": "invalid"}),
        _make_response(200),
    ]
    stub = StubAsyncClient(seq)
    client = BackoffAsyncClient(
        stub,
        credential=None,
        max_retries_too_many_requests=3,
        jitter=False,
        factor=0.5,
        base=2.0,
    )
    sleep_rec = SleepRecorder()
    monkeypatch.setattr("thinkingbox.common.http_client.asyncio.sleep", sleep_rec)

    resp = await client.post(DUMMY_URL)
    resp.raise_for_status()
    assert resp.status_code == 200
    assert len(stub.calls) == 2
    # Should use exponential backoff (0.5 * 2^0 = 0.5) since Retry-After is invalid
    assert sleep_rec.calls == [0.5]


@pytest.mark.asyncio
async def test_retry_on_timeout_then_success(monkeypatch):
    req = httpx.Request("POST", DUMMY_URL)
    timeout_exc = httpx.TimeoutException("Read timed out", request=req)
    seq = [
        timeout_exc,
        _make_response(200),
    ]
    stub = StubAsyncClient(seq)
    client = BackoffAsyncClient(
        stub,
        credential=None,
        max_retries_timeout=2,
        jitter=False,
        factor=0.25,
    )
    sleep_rec = SleepRecorder()
    monkeypatch.setattr("thinkingbox.common.http_client.asyncio.sleep", sleep_rec)

    resp = await client.post(DUMMY_URL)
    resp.raise_for_status()
    assert resp.status_code == 200
    assert len(stub.calls) == 2
    assert sleep_rec.calls == [0.25]


@pytest.mark.asyncio
async def test_retry_on_dns_connect_error(monkeypatch):
    req = httpx.Request("POST", DUMMY_URL)
    # Message must contain retryable DNS hints ("temporary failure")
    connect_exc = httpx.ConnectError(
        "Temporary failure in name resolution", request=req
    )
    seq = [
        connect_exc,
        _make_response(200),
    ]
    stub = StubAsyncClient(seq)
    client = BackoffAsyncClient(
        stub,
        credential=None,
        max_retries_dns=3,
        jitter=False,
        factor=0.3,
    )
    sleep_rec = SleepRecorder()
    monkeypatch.setattr("thinkingbox.common.http_client.asyncio.sleep", sleep_rec)

    resp = await client.post(DUMMY_URL)
    resp.raise_for_status()
    assert resp.status_code == 200
    assert len(stub.calls) == 2
    assert sleep_rec.calls == [0.3]


@pytest.mark.asyncio
async def test_retry_on_server_error_then_success(monkeypatch):
    seq = [
        _make_response(502),
        _make_response(200),
    ]
    stub = StubAsyncClient(seq)
    client = BackoffAsyncClient(
        stub,
        credential=None,
        max_retries_server_error=2,
        retryable_server_errors=(502, 503, 504),
        jitter=False,
        factor=0.4,
    )
    sleep_rec = SleepRecorder()
    monkeypatch.setattr("thinkingbox.common.http_client.asyncio.sleep", sleep_rec)

    resp = await client.post(DUMMY_URL)
    resp.raise_for_status()
    assert resp.status_code == 200
    assert len(stub.calls) == 2
    assert sleep_rec.calls == [0.4]


@pytest.mark.asyncio
async def test_token_refresh_on_401(monkeypatch):
    # 401 then 200
    seq = [
        _make_response(401),
        _make_response(200),
    ]
    stub = StubAsyncClient(seq)
    cred = DummyCredential(tokens=["tokenA", "tokenB"])
    client = BackoffAsyncClient(
        stub,
        credential=cred,
        max_retries_token_refresh=3,
        jitter=False,
        factor=0.9,  # should not matter for 401 path (delay=0)
    )
    sleep_rec = SleepRecorder()
    monkeypatch.setattr("thinkingbox.common.http_client.asyncio.sleep", sleep_rec)

    resp = await client.post(DUMMY_URL)
    resp.raise_for_status()
    assert resp.status_code == 200

    # Two POST attempts
    assert len(stub.calls) == 2

    # Headers used in each attempt
    first_headers = stub.calls[0][1]
    second_headers = stub.calls[1][1]

    assert first_headers.get("Authorization") == "Bearer tokenA"
    assert second_headers.get("Authorization") == "Bearer tokenB"

    # Credential interactions
    assert cred.get_token_calls == 2
    assert cred.invalidate_calls == 1

    # A zero-delay sleep still gets recorded once
    assert sleep_rec.calls == [0.0]


@pytest.mark.asyncio
async def test_no_retry_on_server_error_not_retriable(monkeypatch):
    seq = [
        _make_response(502),
        _make_response(200),
    ]
    stub = StubAsyncClient(seq)
    client = BackoffAsyncClient(
        stub,
        credential=None,
        max_retries_server_error=2,
        retryable_server_errors=(503, 504),
        jitter=False,
        factor=0.4,
    )
    sleep_rec = SleepRecorder()
    monkeypatch.setattr("thinkingbox.common.http_client.asyncio.sleep", sleep_rec)

    resp = await client.post(DUMMY_URL)
    with pytest.raises(httpx.HTTPStatusError) as exc:
        resp.raise_for_status()
    assert resp.status_code == 502
    assert len(stub.calls) == 1
    assert not sleep_rec.calls  # no retries


@contextlib.contextmanager
def _dns_cache_context():
    """Reverts global DNS cache changes on context exit"""
    old_dns_cache = http_client._g_dns_cache
    assert (
        old_dns_cache is None
    ), "DNS cache should not have been initialized in other tests"
    try:
        yield
    finally:
        http_client._g_dns_cache = None


@pytest.mark.asyncio
async def test_get_httpx_client_with_dns_cache_enabled():
    # DNSCacheTransport is used when the DNS cache is enabled
    with _dns_cache_context():
        initialize_dns_cache()
        async with get_httpx_client(use_dns_cache=True) as client:
            assert isinstance(client._transport, DNSCacheTransport)


@pytest.mark.asyncio
async def test_get_httpx_client_with_dns_cache_disabled():
    # by default DNSCacheTransport is not used
    async with get_httpx_client(use_dns_cache=False) as client:
        assert not isinstance(client._transport, DNSCacheTransport)
        assert isinstance(client._transport, httpx.AsyncHTTPTransport)


@pytest.mark.asyncio
async def test_dns_cache_multiple_requests_one_getaddrinfo(monkeypatch):
    """
    Test that when DNS caching is enabled, getaddrinfo is called only once
    for multiple requests to the same host.
    """

    # Patch getaddrinfo
    getaddrinfo_calls = []

    async def mock_getaddrinfo(host, port, family, type, *args, **kwargs):
        getaddrinfo_calls.append((host, port))
        host_map = {
            "example.com": "192.0.2.1",
            "test.com": "192.0.2.2",
        }
        addr = host_map.get(host, "192.0.2.99")
        # Return a fake address info structure
        # Format: [(family, type, proto, canonname, sockaddr)]
        return [(2, 1, 6, "", (addr, port))]

    original_loop = asyncio.get_event_loop()
    monkeypatch.setattr(original_loop, "getaddrinfo", mock_getaddrinfo)

    # Patch httpx transport request handler
    async def mock_handle_async_request(self, request):
        return httpx.Response(200, request=request)

    monkeypatch.setattr(
        httpx.AsyncHTTPTransport,
        "handle_async_request",
        mock_handle_async_request,
    )

    with _dns_cache_context():
        initialize_dns_cache()

        async with get_httpx_client(use_dns_cache=True) as client:
            # Make multiple requests to the same host
            test_requests = [
                "https://example.com/api/test",
                "https://example.com/main",
                "https://test.com",
                "https://test.com/v2",
            ]

            for url in test_requests:
                resp = await client.post(url)
                assert resp.status_code == 200

        # Verify getaddrinfo was called only once for the same host
        assert len(getaddrinfo_calls) == 2
        assert getaddrinfo_calls[0][0] == "example.com"
        assert getaddrinfo_calls[1][0] == "test.com"


@pytest.mark.asyncio
async def test_retry_on_read_error_then_success(monkeypatch):
    """Test that ReadError is retried when 'read-error' is in retryable_server_errors."""
    req = httpx.Request("POST", DUMMY_URL)
    read_error = httpx.ReadError("Server disconnected", request=req)
    seq = [
        read_error,
        _make_response(200),
    ]
    stub = StubAsyncClient(seq)
    client = BackoffAsyncClient(
        stub,
        credential=None,
        max_retries_server_error=3,
        retryable_server_errors=(502, 503, 504, "read-error"),
        jitter=False,
        factor=0.5,
    )
    sleep_rec = SleepRecorder()
    monkeypatch.setattr("thinkingbox.common.http_client.asyncio.sleep", sleep_rec)

    resp = await client.post(DUMMY_URL)
    resp.raise_for_status()
    assert resp.status_code == 200
    assert len(stub.calls) == 2
    assert sleep_rec.calls == [0.5]


@pytest.mark.asyncio
async def test_no_retry_on_read_error_when_not_configured(monkeypatch):
    """Test that ReadError is NOT retried when 'read-error' is not in retryable_server_errors."""
    req = httpx.Request("POST", DUMMY_URL)
    read_error = httpx.ReadError("Server disconnected", request=req)
    seq = [
        read_error,
        _make_response(200),
    ]
    stub = StubAsyncClient(seq)
    client = BackoffAsyncClient(
        stub,
        credential=None,
        max_retries_server_error=3,
        retryable_server_errors=(502, 503, 504),  # no "read-error"
        jitter=False,
        factor=0.5,
    )
    sleep_rec = SleepRecorder()
    monkeypatch.setattr("thinkingbox.common.http_client.asyncio.sleep", sleep_rec)

    with pytest.raises(httpx.ReadError):
        await client.post(DUMMY_URL)

    assert len(stub.calls) == 1
    assert sleep_rec.calls == []  # no retries


@pytest.mark.asyncio
async def test_read_error_exhausts_budget(monkeypatch):
    """Test that ReadError exhausts the server_error retry budget."""
    req = httpx.Request("POST", DUMMY_URL)
    read_error = httpx.ReadError("Server disconnected", request=req)
    seq = [
        read_error,
        read_error,
        read_error,
    ]
    stub = StubAsyncClient(seq)
    client = BackoffAsyncClient(
        stub,
        credential=None,
        max_retries_server_error=2,  # only 2 retries allowed
        retryable_server_errors=(502, "read-error"),
        jitter=False,
        factor=0.1,
    )
    sleep_rec = SleepRecorder()
    monkeypatch.setattr("thinkingbox.common.http_client.asyncio.sleep", sleep_rec)

    with pytest.raises(httpx.ReadError):
        await client.post(DUMMY_URL)

    # 3 attempts: initial + 2 retries
    assert len(stub.calls) == 3
    # Two backoff sleeps (0.1 * 2^0 = 0.1, 0.1 * 2^1 = 0.2)
    assert sleep_rec.calls == [0.1, 0.2]


@pytest.mark.asyncio
async def test_read_error_and_server_error_share_budget(monkeypatch):
    """Test that ReadError and HTTP server errors share the same retry budget."""
    req = httpx.Request("POST", DUMMY_URL)
    read_error = httpx.ReadError("Server disconnected", request=req)
    seq = [
        _make_response(502),  # uses 1 retry from server_error budget
        read_error,  # uses 1 more retry from same budget
        _make_response(200),
    ]
    stub = StubAsyncClient(seq)
    client = BackoffAsyncClient(
        stub,
        credential=None,
        max_retries_server_error=3,
        retryable_server_errors=(502, "read-error"),
        jitter=False,
        factor=0.1,
    )
    sleep_rec = SleepRecorder()
    monkeypatch.setattr("thinkingbox.common.http_client.asyncio.sleep", sleep_rec)

    resp = await client.post(DUMMY_URL)
    resp.raise_for_status()
    assert resp.status_code == 200
    assert len(stub.calls) == 3
    # Two backoff sleeps (both from server_error budget)
    assert sleep_rec.calls == [0.1, 0.2]
