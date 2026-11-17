"""
Moon AI — Sovereign Safe HTTP Client
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Active Sovereign Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY
Training Gate: AUTHORIZED_BY_PREM = False (LOCKED)

Restricted HTTP Client for Moon Web Intelligence:
1. Strict Anti-SSRF and DNS-rebinding immunity via IP-pinned connection pools.
2. GET requests only; zero arbitrary JavaScript execution (no headless browsers).
3. Hard streaming response cap (2 MB) with early abort.
4. Strict Content-Type whitelisting for supported document formats.
5. SSRF-validated redirect following (maximum 3 redirects).
6. Complete isolation from ambient environment proxies (HTTP_PROXY / HTTPS_PROXY).
7. Structured, honest status reporting (FetchResult).
"""

import socket
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import urllib3
from urllib3.exceptions import (
    ConnectTimeoutError,
    HTTPError,
    MaxRetryError,
    ReadTimeoutError,
    SSLError as Urllib3SSLError,
)

from .ssrf_guard import (
    MaxRedirectsExceededError,
    SSRFSecurityError,
    ValidatedURL,
    is_ip_allowed,
    validate_redirect,
    validate_url,
)


class FetchStatus(str, Enum):
    """Honest status classifications for web fetch operations."""
    SUCCESS = "SUCCESS"
    BLOCKED_SSRF = "BLOCKED_SSRF"
    REDIRECT_LOOP = "REDIRECT_LOOP"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
    UNSUPPORTED_CONTENT_TYPE = "UNSUPPORTED_CONTENT_TYPE"
    LOGIN_REQUIRED = "LOGIN_REQUIRED"        # HTTP 401
    ACCESS_DENIED = "ACCESS_DENIED"          # HTTP 403
    NOT_FOUND = "NOT_FOUND"                  # HTTP 404, 410
    RATE_LIMITED = "RATE_LIMITED"            # HTTP 429
    SERVER_ERROR = "SERVER_ERROR"            # HTTP 5xx
    TIMEOUT = "TIMEOUT"                      # Connect or read timeout
    SSL_ERROR = "SSL_ERROR"                  # TLS verification failure
    NETWORK_ERROR = "NETWORK_ERROR"          # Connection dropped or reset


@dataclass
class FetchResult:
    """Immutable record of an HTTP fetch result."""
    status: FetchStatus
    status_code: Optional[int]
    url: str
    final_url: str
    content_text: str = ""
    raw_bytes: bytes = b""
    content_type: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    ip_address: Optional[str] = None
    redirect_history: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    duration_ms: float = 0.0

    @property
    def is_success(self) -> bool:
        """Returns True if the fetch completed successfully with readable content."""
        return self.status == FetchStatus.SUCCESS and self.status_code == 200

    def to_dict(self) -> Dict[str, Any]:
        """Serializes result to a clean dictionary for Moon Chat consumption."""
        return {
            "status": self.status.value,
            "status_code": self.status_code,
            "url": self.url,
            "final_url": self.final_url,
            "content_type": self.content_type,
            "content_length_bytes": len(self.raw_bytes),
            "content_text_chars": len(self.content_text),
            "ip_address": self.ip_address,
            "redirect_history": self.redirect_history,
            "error_message": self.error_message,
            "duration_ms": round(self.duration_ms, 2),
        }


# Default Whitelisted Content-Types (Document, Text, and API Data Formats)
DEFAULT_ALLOWED_CONTENT_TYPES: Set[str] = {
    "text/html",
    "text/plain",
    "application/xhtml+xml",
    "application/json",
    "text/xml",
    "application/xml",
    "text/markdown",
    "text/csv",
    "application/rss+xml",
    "application/atom+xml",
}


class SafeHTTPClient:
    """
    Hardened HTTP Client engineered for sovereign web intelligence retrieval.
    Guarantees strict network sandboxing, DNS-rebinding immunity, stream size enforcement,
    and fail-closed error handling.
    """

    def __init__(
        self,
        connect_timeout: float = 5.0,
        read_timeout: float = 10.0,
        max_response_size: int = 2 * 1024 * 1024,  # 2 MB hard limit
        max_redirects: int = 3,
        allowed_content_types: Optional[Set[str]] = None,
        user_agent: Optional[str] = None,
        socket_resolver: Optional[Callable] = None,
        pool_factory: Optional[Callable] = None,
    ):
        self.connect_timeout = float(connect_timeout)
        self.read_timeout = float(read_timeout)
        self.max_response_size = int(max_response_size)
        self.max_redirects = int(max_redirects)
        self.allowed_content_types = (
            {ct.lower() for ct in allowed_content_types}
            if allowed_content_types is not None
            else DEFAULT_ALLOWED_CONTENT_TYPES
        )
        self.user_agent = (
            user_agent
            or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 MoonAI/1.0"
        )
        self.socket_resolver = socket_resolver
        self.pool_factory = pool_factory

    def _create_pinned_pool(
        self,
        scheme: str,
        pinned_ip: str,
        port: int,
        server_hostname: str,
    ) -> Union[urllib3.HTTPConnectionPool, urllib3.HTTPSConnectionPool]:
        """
        Creates an IP-pinned connection pool to eliminate DNS rebinding TOCTOU windows.
        
        The TCP connection connects directly to the pre-validated `pinned_ip`.
        In HTTPS mode, TLS SNI and certificate hostname verification are bound to `server_hostname`.
        System proxies are completely disabled.
        """
        if self.pool_factory is not None:
            return self.pool_factory(scheme, pinned_ip, port, server_hostname)

        timeout = urllib3.Timeout(connect=self.connect_timeout, read=self.read_timeout)

        if scheme == "https":
            return urllib3.HTTPSConnectionPool(
                host=pinned_ip,
                port=port,
                server_hostname=server_hostname,
                assert_hostname=server_hostname,
                timeout=timeout,
                retries=False,
            )
        else:
            return urllib3.HTTPConnectionPool(
                host=pinned_ip,
                port=port,
                timeout=timeout,
                retries=False,
            )

    def _parse_content_type(self, raw_header: Optional[str]) -> Tuple[str, str]:
        """
        Extracts lowercase MIME type and charset from a Content-Type header.
        Defaults to 'text/html' and 'utf-8' if unspecified.
        """
        if not raw_header:
            return "text/html", "utf-8"

        parts = [p.strip() for p in raw_header.split(";")]
        mime_type = parts[0].lower() if parts else "text/html"

        charset = "utf-8"
        for part in parts[1:]:
            if part.lower().startswith("charset="):
                eq_idx = part.find("=")
                if eq_idx != -1:
                    raw_cs = part[eq_idx + 1:].strip("\"' ")
                    if raw_cs:
                        charset = raw_cs.lower()
                break

        return mime_type, charset

    def get(
        self,
        url: str,
        custom_headers: Optional[Dict[str, str]] = None,
    ) -> FetchResult:
        """
        Executes a secure, streaming HTTP GET request under strict SSRF and size bounds.

        Args:
            url: The destination URL to fetch.
            custom_headers: Optional HTTP headers to include.

        Returns:
            FetchResult: Structured response record.
        """
        start_time = time.time()
        current_url = url
        redirect_history: List[str] = []
        redirect_count = 0

        while True:
            # 1. SSRF URL and DNS Validation
            try:
                validated: ValidatedURL = validate_url(
                    current_url,
                    socket_resolver=self.socket_resolver,
                )
            except SSRFSecurityError as e:
                duration_ms = (time.time() - start_time) * 1000
                return FetchResult(
                    status=FetchStatus.BLOCKED_SSRF,
                    status_code=None,
                    url=url,
                    final_url=current_url,
                    redirect_history=redirect_history,
                    error_message=str(e),
                    duration_ms=duration_ms,
                )
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                return FetchResult(
                    status=FetchStatus.NETWORK_ERROR,
                    status_code=None,
                    url=url,
                    final_url=current_url,
                    redirect_history=redirect_history,
                    error_message=f"URL validation error: {e}",
                    duration_ms=duration_ms,
                )

            pinned_ip = validated.primary_ip

            # 2. Build Request Headers (No proxy headers accepted, validate CRLF)
            if custom_headers:
                for k, v in custom_headers.items():
                    k_str = str(k)
                    v_str = str(v)
                    if any(c in k_str or c in v_str for c in ("\r", "\n", "\0")):
                        duration_ms = (time.time() - start_time) * 1000
                        return FetchResult(
                            status=FetchStatus.NETWORK_ERROR,
                            status_code=None,
                            url=url,
                            final_url=current_url,
                            redirect_history=redirect_history,
                            error_message=f"Custom header contains illegal control characters (CR/LF/NULL): '{k_str}'",
                            duration_ms=duration_ms,
                        )

            host_val = f"[{validated.hostname}]" if (":" in validated.hostname and not validated.hostname.startswith("[")) else validated.hostname
            request_headers: Dict[str, str] = {
                "Host": host_val if (validated.port in (80, 443)) else f"{host_val}:{validated.port}",
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/json,text/plain;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Connection": "close",
            }
            if custom_headers:
                for k, v in custom_headers.items():
                    if str(k).lower() not in ("host", "proxy-authorization"):
                        request_headers[str(k)] = str(v)

            # 3. Create IP-Pinned Connection Pool & Execute Request
            pool = None
            response = None
            try:
                try:
                    pool = self._create_pinned_pool(
                        scheme=validated.scheme,
                        pinned_ip=pinned_ip,
                        port=validated.port,
                        server_hostname=validated.hostname,
                    )
                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    return FetchResult(
                        status=FetchStatus.NETWORK_ERROR,
                        status_code=None,
                        url=url,
                        final_url=current_url,
                        ip_address=pinned_ip,
                        redirect_history=redirect_history,
                        error_message=f"Failed to create connection pool: {e}",
                        duration_ms=duration_ms,
                    )

                # 4. Execute Streaming GET Request
                request_path = validated.path
                if validated.query:
                    request_path = f"{request_path}?{validated.query}"

                try:
                    response = pool.request(
                        "GET",
                        request_path,
                        headers=request_headers,
                        preload_content=False,
                    )
                except (ConnectTimeoutError, ReadTimeoutError) as e:
                    duration_ms = (time.time() - start_time) * 1000
                    return FetchResult(
                        status=FetchStatus.TIMEOUT,
                        status_code=None,
                        url=url,
                        final_url=current_url,
                        ip_address=pinned_ip,
                        redirect_history=redirect_history,
                        error_message=f"Request timed out: {e}",
                        duration_ms=duration_ms,
                    )
                except Urllib3SSLError as e:
                    duration_ms = (time.time() - start_time) * 1000
                    return FetchResult(
                        status=FetchStatus.SSL_ERROR,
                        status_code=None,
                        url=url,
                        final_url=current_url,
                        ip_address=pinned_ip,
                        redirect_history=redirect_history,
                        error_message=f"TLS/SSL verification failed: {e}",
                        duration_ms=duration_ms,
                    )
                except (HTTPError, OSError) as e:
                    duration_ms = (time.time() - start_time) * 1000
                    return FetchResult(
                        status=FetchStatus.NETWORK_ERROR,
                        status_code=None,
                        url=url,
                        final_url=current_url,
                        ip_address=pinned_ip,
                        redirect_history=redirect_history,
                        error_message=f"Network error: {e}",
                        duration_ms=duration_ms,
                    )

                status_code = response.status
                raw_headers = {str(k): str(v) for k, v in response.headers.items()}

                # 5. Handle HTTP Redirects (301, 302, 303, 307, 308)
                if status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("Location") or response.headers.get("location")
                    if not location:
                        duration_ms = (time.time() - start_time) * 1000
                        return FetchResult(
                            status=FetchStatus.NETWORK_ERROR,
                            status_code=status_code,
                            url=url,
                            final_url=current_url,
                            ip_address=pinned_ip,
                            redirect_history=redirect_history,
                            error_message="Redirect response missing Location header.",
                            duration_ms=duration_ms,
                        )

                    redirect_history.append(current_url)
                    redirect_count += 1

                    try:
                        next_validated = validate_redirect(
                            current_url=current_url,
                            redirect_location=location,
                            redirect_count=redirect_count,
                            max_redirects=self.max_redirects,
                            socket_resolver=self.socket_resolver,
                        )
                    except MaxRedirectsExceededError as e:
                        duration_ms = (time.time() - start_time) * 1000
                        return FetchResult(
                            status=FetchStatus.REDIRECT_LOOP,
                            status_code=status_code,
                            url=url,
                            final_url=current_url,
                            ip_address=pinned_ip,
                            redirect_history=redirect_history,
                            error_message=str(e),
                            duration_ms=duration_ms,
                        )
                    except SSRFSecurityError as e:
                        duration_ms = (time.time() - start_time) * 1000
                        return FetchResult(
                            status=FetchStatus.BLOCKED_SSRF,
                            status_code=status_code,
                            url=url,
                            final_url=location,
                            ip_address=pinned_ip,
                            redirect_history=redirect_history,
                            error_message=f"Redirect target blocked by SSRF security: {e}",
                            duration_ms=duration_ms,
                        )

                    current_url = next_validated.normalized_url
                    continue  # Loop to follow the redirect safely; finally cleans up pool and response!

                # 6. Check Specific HTTP Status Codes
                if status_code == 401:
                    duration_ms = (time.time() - start_time) * 1000
                    return FetchResult(
                        status=FetchStatus.LOGIN_REQUIRED,
                        status_code=status_code,
                        url=url,
                        final_url=current_url,
                        headers=raw_headers,
                        ip_address=pinned_ip,
                        redirect_history=redirect_history,
                        error_message="HTTP 401: Authentication required by target server.",
                        duration_ms=duration_ms,
                    )
                if status_code == 403:
                    duration_ms = (time.time() - start_time) * 1000
                    return FetchResult(
                        status=FetchStatus.ACCESS_DENIED,
                        status_code=status_code,
                        url=url,
                        final_url=current_url,
                        headers=raw_headers,
                        ip_address=pinned_ip,
                        redirect_history=redirect_history,
                        error_message="HTTP 403: Access forbidden or blocked by target server.",
                        duration_ms=duration_ms,
                    )
                if status_code in (404, 410):
                    duration_ms = (time.time() - start_time) * 1000
                    return FetchResult(
                        status=FetchStatus.NOT_FOUND,
                        status_code=status_code,
                        url=url,
                        final_url=current_url,
                        headers=raw_headers,
                        ip_address=pinned_ip,
                        redirect_history=redirect_history,
                        error_message=f"HTTP {status_code}: Resource not found or deleted.",
                        duration_ms=duration_ms,
                    )
                if status_code == 429:
                    duration_ms = (time.time() - start_time) * 1000
                    return FetchResult(
                        status=FetchStatus.RATE_LIMITED,
                        status_code=status_code,
                        url=url,
                        final_url=current_url,
                        headers=raw_headers,
                        ip_address=pinned_ip,
                        redirect_history=redirect_history,
                        error_message="HTTP 429: Target server rate limit exceeded.",
                        duration_ms=duration_ms,
                    )
                if status_code >= 500:
                    duration_ms = (time.time() - start_time) * 1000
                    return FetchResult(
                        status=FetchStatus.SERVER_ERROR,
                        status_code=status_code,
                        url=url,
                        final_url=current_url,
                        headers=raw_headers,
                        ip_address=pinned_ip,
                        redirect_history=redirect_history,
                        error_message=f"HTTP {status_code}: Upstream server encountered an error.",
                        duration_ms=duration_ms,
                    )

                # 7. Validate Content-Type
                raw_content_type = response.headers.get("Content-Type") or response.headers.get("content-type")
                mime_type, charset = self._parse_content_type(raw_content_type)

                if mime_type not in self.allowed_content_types:
                    duration_ms = (time.time() - start_time) * 1000
                    return FetchResult(
                        status=FetchStatus.UNSUPPORTED_CONTENT_TYPE,
                        status_code=status_code,
                        url=url,
                        final_url=current_url,
                        content_type=mime_type,
                        headers=raw_headers,
                        ip_address=pinned_ip,
                        redirect_history=redirect_history,
                        error_message=(
                            f"Unsupported Content-Type '{mime_type}'. "
                            f"Only document/text formats ({sorted(self.allowed_content_types)}) are permitted."
                        ),
                        duration_ms=duration_ms,
                    )

                # 8. Check Content-Length Header if Present
                content_length_header = response.headers.get("Content-Length") or response.headers.get("content-length")
                if content_length_header:
                    try:
                        declared_len = int(content_length_header)
                        if declared_len > self.max_response_size:
                            duration_ms = (time.time() - start_time) * 1000
                            return FetchResult(
                                status=FetchStatus.PAYLOAD_TOO_LARGE,
                                status_code=status_code,
                                url=url,
                                final_url=current_url,
                                content_type=mime_type,
                                headers=raw_headers,
                                ip_address=pinned_ip,
                                redirect_history=redirect_history,
                                error_message=(
                                    f"Declared Content-Length ({declared_len} bytes) exceeds "
                                    f"maximum permitted limit of {self.max_response_size} bytes (2 MB)."
                                ),
                                duration_ms=duration_ms,
                            )
                    except ValueError:
                        pass  # Non-integer Content-Length, proceed to stream limit check

                # 9. Stream Response Body with Hard Cap
                chunks: List[bytes] = []
                total_bytes = 0
                chunk_size = 16384  # 16 KB chunks

                try:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        total_bytes += len(chunk)
                        if total_bytes > self.max_response_size:
                            duration_ms = (time.time() - start_time) * 1000
                            return FetchResult(
                                status=FetchStatus.PAYLOAD_TOO_LARGE,
                                status_code=status_code,
                                url=url,
                                final_url=current_url,
                                content_type=mime_type,
                                headers=raw_headers,
                                ip_address=pinned_ip,
                                redirect_history=redirect_history,
                                error_message=(
                                    f"Streamed response exceeded maximum permitted limit of "
                                    f"{self.max_response_size} bytes (2 MB). Stream terminated."
                                ),
                                duration_ms=duration_ms,
                            )
                        chunks.append(chunk)
                except (ReadTimeoutError, socket.timeout) as e:
                    duration_ms = (time.time() - start_time) * 1000
                    return FetchResult(
                        status=FetchStatus.TIMEOUT,
                        status_code=status_code,
                        url=url,
                        final_url=current_url,
                        content_type=mime_type,
                        headers=raw_headers,
                        ip_address=pinned_ip,
                        redirect_history=redirect_history,
                        error_message=f"Read timeout while streaming body: {e}",
                        duration_ms=duration_ms,
                    )
                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    return FetchResult(
                        status=FetchStatus.NETWORK_ERROR,
                        status_code=status_code,
                        url=url,
                        final_url=current_url,
                        content_type=mime_type,
                        headers=raw_headers,
                        ip_address=pinned_ip,
                        redirect_history=redirect_history,
                        error_message=f"Error reading response stream: {e}",
                        duration_ms=duration_ms,
                    )

                raw_body = b"".join(chunks)

                # 10. Decode Text Content Safely
                try:
                    decoded_text = raw_body.decode(charset, errors="replace")
                except (LookupError, UnicodeDecodeError):
                    decoded_text = raw_body.decode("utf-8", errors="replace")

                duration_ms = (time.time() - start_time) * 1000
                return FetchResult(
                    status=FetchStatus.SUCCESS,
                    status_code=status_code,
                    url=url,
                    final_url=current_url,
                    content_text=decoded_text,
                    raw_bytes=raw_body,
                    content_type=mime_type,
                    headers=raw_headers,
                    ip_address=pinned_ip,
                    redirect_history=redirect_history,
                    duration_ms=duration_ms,
                )
            finally:
                if response is not None:
                    try:
                        response.close()
                    except Exception:
                        pass
                    try:
                        response.release_conn()
                    except Exception:
                        pass
                if pool is not None:
                    try:
                        pool.close()
                    except Exception:
                        pass
