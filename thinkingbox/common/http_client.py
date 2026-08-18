# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import random
import socket
import ssl
from pathlib import Path
from typing import Any, AsyncIterator, Mapping, NamedTuple

import httpx

from thinkingbox.common.utils import CredentialBase

logger = logging.getLogger(__name__)


_STRING_ERROR_CODES = {"read-error"}


@dataclasses.dataclass
class BackoffRetries:
    too_many_requests: int = 0
    dns: int = 0
    timeout: int = 0
    token_refresh: int = 0
    server_error: int = 0

    def all_lessequal(self, other: BackoffRetries):
        for f in dataclasses.fields(self):
            value = getattr(other, f.name)
            if (value >= 0) and getattr(self, f.name) > value:
                return False
        return True


class RetryAction(NamedTuple):
    retry: bool
    delay: float = 0.0
    invalidate_token: bool = False


class SSEResponseHeaders(NamedTuple):
    data: dict


class SSEResponse(NamedTuple):
    event_type: str | None = None
    data: str | None = None


def _is_retryable_dns_error(exc: httpx.ConnectError) -> bool:
    msg = str(exc).lower()
    return (
        "name or service not known" in msg
        or "temporary failure" in msg
        or "try again" in msg
    )


def _validate_retryable_server_error_code(code: int | str) -> int | str:
    if isinstance(code, str):
        if code in _STRING_ERROR_CODES:
            return code
        elif code.isdigit():
            return int(code)
    elif isinstance(code, int):
        return code
    raise ValueError(f"Invalid code in retryable_server_errors: {code!r}")


def _validate_retryable_server_errors(
    errors: tuple[int | str, ...],
) -> tuple[int | str, ...]:
    """
    Validate and normalize retryable_server_errors.
    """
    return tuple([_validate_retryable_server_error_code(code) for code in errors])


class BackoffAsyncClient:
    """
    Wrapper around an httpx.AsyncClient providing retry / backoff

    Features:
      * Exponential backoff (with optional full jitter)
      * Retry on DNS resolution transient errors in httpx.ConnectError
      * Token invalidation & refresh on HTTP 401 using a provided CredentialBase
      * Separation of retry budgets
      * Stops retrying once budgets are exhausted and re-raises the last error

    Order of handling in a failure case:
      1. If ConnectError and DNS-transient -> retry (dns budget)
      2. If HTTP 401 and credential present -> invalidate & retry (token_refresh budget)
      3. If HTTP 429 -> retry (too_many_requests budget)
      4. If server error in retryable_server_errors -> retry (server_error budget)
      5. If Timeout -> retry (timeout budget)
      6. Else -> raise

    Parameters
    ----------
    client : httpx.AsyncClient
        The underlying async client (not owned; we don't close it).
    credential : CredentialBase | None
        Credential used for bearer tokens. If provided and caller does not
        supply an Authorization header, one is injected.
    max_retries_too_many_requests : int
        Max retry attempts for 429 errors. Use -1 for unlimited.
    max_retries_dns : int
        Max retry attempts for retryable DNS-related ConnectError issues.
    max_retries_timeout : int
        Max retry attempts for timeout errors.
    max_retries_token_refresh : int
        Max number of HTTP 401 token refresh cycles.
    max_retries_server_error : int
        Max retry attempts for server errors matching retryable_server_errors.
    retryable_server_errors : tuple[int | str, ...]
        HTTP status codes to retry (e.g., 502, 503, 504). Also accepts
        "read-error" to retry on httpx.ReadError. Numeric strings are
        auto-converted to integers.
    base : float
        Exponentiation base for exponential delay (delay = factor * base ** n).
    factor : float
        Multiplicative factor for exponential delay (delay = factor * base ** n).
    max_delay : float
        Cap for the exponential delay.
    jitter : bool
        If True, apply "full jitter" (random.uniform(0, delay)).
    """

    logger = logging.getLogger("thinkingbox.backoff")

    def __init__(
        self,
        client: httpx.AsyncClient,
        credential: CredentialBase | None = None,
        max_retries_too_many_requests: int = -1,
        max_retries_dns: int = 10,
        max_retries_timeout: int = 10,
        max_retries_token_refresh: int = 3,
        max_retries_server_error: int = 5,
        retryable_server_errors: tuple[int | str, ...] = (502, 503, 504),
        base: float = 2.0,
        factor: float = 1.0,
        max_delay: float = 32.0,
        jitter: bool = True,
    ):
        self._client = client
        self._credential = credential
        self._max_retries = BackoffRetries(
            too_many_requests=max_retries_too_many_requests,
            dns=max_retries_dns,
            timeout=max_retries_timeout,
            token_refresh=max_retries_token_refresh,
            server_error=max_retries_server_error,
        )
        self._retryable_server_errors = _validate_retryable_server_errors(
            retryable_server_errors
        )
        self._base = base
        self._factor = factor
        self._max_delay = max_delay
        self._jitter = jitter

    async def post(self, url: str | httpx.URL, **kwargs) -> httpx.Response:
        """
        Perform a POST with retry/backoff + token refresh semantics.

        Any non-429 / non-timeout / non-retryable-ConnectError / non-401
        HTTP errors are propagated without retry.

        Returns
        -------
        httpx.Response
            A successful response if one was received before exhausting a retry budget,
            otherwise the last error response. The caller should inspect the response
            for errors or call response.raise_for_status().

        Raises
        ------
        httpx.TimeoutException
            If timeout and retry budget exhausted.
        httpx.ConnectError
            If DNS transient errors exceed retry budget.
        Exception
            Any other unexpected exception (not considered retryable).
        """
        # We'll preserve original headers for reconstruction each attempt
        base_headers = kwargs.pop("headers", None)

        retries = BackoffRetries()

        # We'll loop until we either return or raise
        while True:
            # Compose headers for this attempt
            headers = dict(base_headers) if base_headers else {}

            if (self._credential is not None) and not self._has_auth_header(headers):
                try:
                    token = await self._credential.get_token()
                except Exception:
                    # If token acquisition itself fails, we don't retry here automatically;
                    # raising surfaces underlying issue (could be extended if needed).
                    raise
                headers["Authorization"] = f"Bearer {token}"

            try:
                response = await self._client.post(url, headers=headers, **kwargs)
                # Convert non-2xx into HTTPStatusError so we can unify logic.
                response.raise_for_status()
                return response

            except (
                httpx.HTTPStatusError,
                httpx.TimeoutException,
                httpx.ConnectError,
                httpx.ReadError,
            ) as e:
                action = self._handle_exception(e, retries)
                if not action.retry:
                    if isinstance(e, httpx.HTTPStatusError):
                        self.logger.error("HTTP %d (%s)", e.response.status_code, e)
                        return e.response
                    self.logger.error("HTTP client timeout/connect/read error (%s)", e)
                    raise
            except Exception as e:
                # Any other error: do not treat as retryable
                self.logger.error("HTTP client error (%s)", e)
                raise

            # Invalidate token if needed
            if action.invalidate_token:
                try:
                    await self._credential.invalidate()
                except Exception as e:
                    self._log_retry("Error invalidating token", e)

            # wait
            await asyncio.sleep(action.delay)

    async def post_iter_sse_chunks(
        self, url: str | httpx.URL, **kwargs
    ) -> AsyncIterator[SSEResponse | SSEResponseHeaders]:
        """Perform a POST with retry/backoff and stream SSE chunks.

        Yields:
        - SSEResponseHeaders(data) where data contains headers, then
        - SSEResponse(event_type, data) where data is the chunk content as string

        The stream terminates on `data: [DONE]`.

        Unlike the `post` method, non-retryable HTTP errors are always raised
        (there is no response object to return).

        Raises
        ------
        httpx.HTTPStatusError
            If the server returns a non-retryable error status.
        httpx.TimeoutException
            If timeout and retry budget exhausted.
        httpx.ConnectError
            If DNS transient errors exceed retry budget.
        """
        base_headers = kwargs.pop("headers", None)

        retries = BackoffRetries()

        while True:
            headers = dict(base_headers) if base_headers else {}

            if (self._credential is not None) and not self._has_auth_header(headers):
                token = await self._credential.get_token()
                headers["Authorization"] = f"Bearer {token}"

            try:
                async with self._client.stream(
                    "POST", url, headers=headers, **kwargs
                ) as response:
                    response.raise_for_status()

                    yield SSEResponseHeaders(data=dict(response.headers.items()))

                    event_type: str | None = None
                    event_data: list[str] = []
                    async for line in response.aiter_lines():
                        if line.startswith("event: "):
                            event_type = line[7:]
                        elif line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                return
                            event_data.append(data)
                        elif line == "":
                            if event_data:
                                full_data = "\n".join(event_data)
                                yield SSEResponse(
                                    event_type=event_type,
                                    data=full_data,
                                )
                                event_type = None
                                event_data = []
                    return

            except (
                httpx.HTTPStatusError,
                httpx.TimeoutException,
                httpx.ConnectError,
                httpx.ReadError,
            ) as e:
                action = self._handle_exception(e, retries)
                if not action.retry:
                    if isinstance(e, httpx.HTTPStatusError):
                        self.logger.error("HTTP %d (%s)", e.response.status_code, e)
                    else:
                        self.logger.error(
                            "HTTP client timeout/connect/read error (%s)", e
                        )
                    raise
            except Exception as e:
                self.logger.error("HTTP client error (%s)", e)
                raise

            if action.invalidate_token:
                try:
                    await self._credential.invalidate()
                except Exception as e:
                    self._log_retry("Error invalidating token", e)

            await asyncio.sleep(action.delay)

    @property
    def client(self) -> httpx.AsyncClient:
        return self._client

    def _handle_exception(self, e: Exception, retries: BackoffRetries):
        # DNS / transient ConnectError
        if isinstance(e, httpx.ConnectError) and _is_retryable_dns_error(e):
            retries.dns += 1
            if not retries.all_lessequal(self._max_retries):
                return RetryAction(retry=False)
            delay = self._compute_delay(retries.dns)
            self._log_retry(
                f"Retryable DNS/connect error (attempt {retries.dns}/{self._max_retries.dns})",
                e,
                delay,
            )
            return RetryAction(retry=True, delay=delay)

        # HTTP status errors (including 401 / 429 handling)
        if isinstance(e, httpx.HTTPStatusError):
            status = e.response.status_code

            # Token expiration (401)
            if status == 401 and self._credential is not None:
                retries.token_refresh += 1
                if not retries.all_lessequal(self._max_retries):
                    return RetryAction(retry=False)
                self._log_retry(
                    f"HTTP 401: invalidating token (attempt {retries.token_refresh}/{self._max_retries.token_refresh})",
                    e,
                    delay=0.0,
                )
                return RetryAction(retry=True, delay=0.0, invalidate_token=True)

            # 429 Too Many Requests
            if status == 429:
                retries.too_many_requests += 1
                if not retries.all_lessequal(self._max_retries):
                    return RetryAction(retry=False)
                delay = self._compute_delay_with_retry_after(
                    retries.too_many_requests,
                    e.response.headers,
                )
                self._log_retry(
                    f"HTTP 429 (attempt {retries.too_many_requests}/{self._max_retries.too_many_requests})",
                    e,
                    delay,
                )
                return RetryAction(retry=True, delay=delay)

            # 5xx
            if status in self._retryable_server_errors:
                retries.server_error += 1
                if not retries.all_lessequal(self._max_retries):
                    return RetryAction(retry=False)
                delay = self._compute_delay(retries.server_error)
                self._log_retry(
                    f"HTTP {status} (attempt {retries.server_error}/{self._max_retries.server_error})",
                    e,
                    delay,
                )
                return RetryAction(retry=True, delay=delay)

            return RetryAction(retry=False)

        # Timeout
        if isinstance(e, httpx.TimeoutException):
            retries.timeout += 1
            if not retries.all_lessequal(self._max_retries):
                return RetryAction(retry=False)
            delay = self._compute_delay(retries.timeout)
            self._log_retry(
                f"Timeout (attempt {retries.timeout}/{self._max_retries.timeout})",
                e,
                delay,
            )
            return RetryAction(retry=True, delay=delay)

        # ReadError
        if (
            isinstance(e, httpx.ReadError)
            and "read-error" in self._retryable_server_errors
        ):
            retries.server_error += 1
            if not retries.all_lessequal(self._max_retries):
                return RetryAction(retry=False)
            delay = self._compute_delay(retries.server_error)
            self._log_retry(
                f"ReadError (attempt {retries.server_error}/{self._max_retries.server_error})",
                e,
                delay,
            )
            return RetryAction(retry=True, delay=delay)

        # Anything else
        return RetryAction(retry=False)

    def _compute_delay(self, n: int) -> float:
        """
        n: 1-based index of the retry (first retry => 1).
        """
        # factor * base ** n
        raw = self._factor * (self._base ** (n - 1))
        capped = min(raw, self._max_delay)
        if self._jitter:
            return random.uniform(0, capped)
        return capped

    def _compute_delay_with_retry_after(
        self, n: int, headers: Mapping[str, str]
    ) -> float:
        delay = self._compute_delay(n)
        retry_after = headers.get("Retry-After")
        if retry_after is None:
            return delay
        try:
            retry_after_seconds = int(retry_after.strip())
        except ValueError:
            return delay
        delay = max(
            min(float(retry_after_seconds), self._max_delay),
            delay,
        )
        return delay

    @staticmethod
    def _has_auth_header(headers: dict[str, str]) -> bool:
        return any(k.lower() == "authorization" for k in headers)

    def _log_retry(self, context: str, exc: BaseException, delay: float | None = None):
        if delay is not None:
            self.logger.warning("%s; retrying in %.2fs (%s)", context, delay, exc)
        else:
            self.logger.warning("%s (%s)", context, exc)


class DNSCacheContext:
    def __init__(self, cache: dict | None = None):
        self.lock = asyncio.Lock()
        self.cache: dict[str, str] = cache.copy() if cache else {}

    async def copy(self) -> dict:
        async with self.lock:
            value = self.cache.copy()
        return value

    async def get(self, key: str) -> str | None:
        async with self.lock:
            value = self.cache.get(key)
        return value

    async def set(self, key: str, value: str):
        async with self.lock:
            if key not in self.cache:
                self.cache[key] = value

    async def remove(self, key: str):
        async with self.lock:
            self.cache.pop(key, None)


class DNSCacheTransport(httpx.AsyncHTTPTransport):
    """
    Custom transport that caches successful DNS resolutions.

    The cache dictionary is shared across all DNSCache instances, so DNS
    resolutions are cached globally. Each client gets its own transport instance.

    Once a hostname is successfully resolved, the resolved address is cached
    and reused for subsequent requests to that hostname. Errors are not cached,
    allowing retry on the next request.
    """

    def __init__(self, cache_context: DNSCacheContext, /, **kwargs):
        super().__init__(**kwargs)
        self.cache_context = cache_context

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """
        Handle request with DNS caching.

        If the hostname has been successfully resolved before, use the cached
        address. Otherwise, attempt resolution and cache on success.
        """
        host = request.url.host

        # Check if we have a cached resolution
        cached_ip: str | None = None
        if host:
            cached_ip = await self.cache_context.get(host)

        if cached_ip is None:
            loop = asyncio.get_event_loop()
            try:
                addr_info = await loop.getaddrinfo(
                    host,
                    request.url.port or (443 if request.url.scheme == "https" else 80),
                    family=socket.AF_UNSPEC,
                    type=socket.SOCK_STREAM,
                )
            except Exception as e:
                raise httpx.ConnectError(
                    str(e),
                    request=request,
                ) from e
            if not addr_info:
                raise httpx.ConnectError(
                    "Name or service not known",
                    request=request,
                )
            cached_ip = addr_info[0][4][0]
            await self.cache_context.set(host, cached_ip)

        # Create a new request with the cached IP address
        # Modify the URL to use the cached IP but preserve the original host header
        original_url = request.url
        modified_url = original_url.copy_with(host=cached_ip)

        # Preserve the original hostname for SNI (SSL certificate verification)
        extensions = dict(request.extensions)
        if "sni_hostname" not in extensions:
            extensions["sni_hostname"] = host.encode("ascii")

        request = httpx.Request(
            method=request.method,
            url=modified_url,
            headers=request.headers,
            content=request.content,
            extensions=extensions,
        )
        # Ensure Host header preserves original hostname
        request.headers["Host"] = host

        try:
            # Attempt the request with the parent transport
            return await super().handle_async_request(request)
        except Exception as e:
            # On error, do not cache - let it retry on next request
            # If this was a DNS error and we used cache, remove from cache
            if (
                host
                and isinstance(e, httpx.ConnectError)
                and _is_retryable_dns_error(e)
            ):
                await self.cache_context.remove(host)
            raise


_g_dns_cache: DNSCacheContext | None = None


def initialize_dns_cache(cache: dict | None = None):
    """
    Initialize the global DNS cache.

    This should be called once at application startup to enable DNS caching
    for all HTTP clients created with get_httpx_client(). Each client will get
    its own transport instance, but they will all share the same DNS cache.
    """
    global _g_dns_cache
    if _g_dns_cache is None:
        _g_dns_cache = DNSCacheContext(cache)
        logger.debug("Initialized DNS Cache")


def get_dns_cache() -> DNSCacheContext:
    return _g_dns_cache


_CLIENT_TRANSPORT_KWARGS = {"verify", "cert", "trust_env", "http1", "http2", "limits"}


def create_custom_ssl_context(
    client_certificate: str | Path | None = None,
    trust_ca_path: str | Path | None = None,
) -> ssl.SSLContext:
    """
    Create an SSL context that optionally:
    - trusts a custom CA in addition to system CAs
    - uses a client certificate for authentication
    """
    context = ssl.create_default_context()

    if trust_ca_path is not None:
        trust_ca_path = Path(trust_ca_path).expanduser()
        if trust_ca_path.is_dir():
            context.load_verify_locations(capath=trust_ca_path)
        elif trust_ca_path.is_file():
            context.load_verify_locations(cafile=trust_ca_path)
        else:
            raise ValueError(f"trust_ca_path not found at {trust_ca_path!s}")

    if client_certificate is not None:
        client_certificate = Path(client_certificate).expanduser()
        if not client_certificate.is_file():
            raise ValueError(f"client_certificate not found at {client_certificate!s}")
        context.load_cert_chain(certfile=client_certificate)

    return context


@contextlib.asynccontextmanager
async def get_httpx_client(
    use_dns_cache: bool = False,
    **kwargs,
):
    """
    Create an httpx.AsyncClient with optional DNS caching.

    If initialize_dns_cache() has been called, the returned client will use
    a new DNSCache transport instance that shares the global cache. Otherwise,
    a standard client is returned.

    Parameters
    ----------
    **kwargs
        Keyword arguments to pass to httpx.AsyncClient.

    Yields
    ------
    httpx.AsyncClient
        An async HTTP client with DNS caching if enabled.
    """

    if use_dns_cache:
        if _g_dns_cache is None:
            raise RuntimeError("Initialize the DNS cache with initialize_dns_cache()")
        if "transport" in kwargs:
            raise ValueError(
                "use_dns_cache=True cannot be used if transport is already set"
            )
        # Create a new transport instance that uses the shared cache
        transport_kwargs: dict[str, Any] = {}
        for key in _CLIENT_TRANSPORT_KWARGS & set(kwargs):
            transport_kwargs[key] = kwargs.pop(key)

        kwargs["transport"] = DNSCacheTransport(_g_dns_cache, **transport_kwargs)

    async with httpx.AsyncClient(**kwargs) as client:
        yield client
