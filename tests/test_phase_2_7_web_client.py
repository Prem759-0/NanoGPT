"""
Moon AI — Phase 2.7 Milestone 1: Safe HTTP Client Test Suite
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Active Sovereign Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY
Training Gate: AUTHORIZED_BY_PREM = False (LOCKED)

Deterministic Unit Tests for SafeHTTPClient:
- IP Pinning & DNS Rebinding Immunity
- 2 MB Stream Cap & Early Content-Length Abort
- Content-Type Whitelisting
- Safe SSRF-Guarded Redirects
- Honest Status Reporting (401, 403, 404, 429, 500, Timeout, SSL)
- Zero Ambient Proxy Inheritance
- Zero JavaScript Execution
"""

import io
import socket
import unittest
from unittest.mock import MagicMock, patch
import urllib3
from urllib3.exceptions import ConnectTimeoutError, ReadTimeoutError, SSLError as Urllib3SSLError

from core.tools.web.client import FetchResult, FetchStatus, SafeHTTPClient


class MockStreamResponse:
    """Mock HTTP response simulating streaming behavior."""

    def __init__(
        self,
        status: int = 200,
        headers: dict = None,
        body: bytes = b"",
        chunk_size: int = 16384,
    ):
        self.status = status
        self.headers = headers or {"Content-Type": "text/html; charset=utf-8"}
        self._body_stream = io.BytesIO(body)
        self._chunk_size = chunk_size
        self.released = False
        self.closed = False

    def read(self, amt: int = None) -> bytes:
        amt = amt or self._chunk_size
        return self._body_stream.read(amt)

    def close(self):
        self.closed = True

    def release_conn(self):
        self.released = True


class TestSafeHTTPClient(unittest.TestCase):
    """Deterministic, offline test suite for SafeHTTPClient."""

    def setUp(self):
        # Default mock resolver returning public IP 93.184.216.34
        self.public_ip = "93.184.216.34"

        def mock_resolver(host, port, **kwargs):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (self.public_ip, port))]

        self.mock_resolver = mock_resolver

    def test_01_successful_html_fetch(self):
        """Standard 200 OK HTML response must return FetchStatus.SUCCESS with decoded text."""
        html_bytes = b"<!DOCTYPE html><html><head><title>Test</title></head><body>Hello Moon AI</body></html>"
        mock_resp = MockStreamResponse(status=200, headers={"Content-Type": "text/html; charset=utf-8"}, body=html_bytes)

        mock_pool = MagicMock()
        mock_pool.request.return_value = mock_resp

        client = SafeHTTPClient(
            socket_resolver=self.mock_resolver,
            pool_factory=lambda *args: mock_pool,
        )

        res = client.get("https://example.com/test")
        self.assertTrue(res.is_success)
        self.assertEqual(res.status, FetchStatus.SUCCESS)
        self.assertEqual(res.status_code, 200)
        self.assertIn("Hello Moon AI", res.content_text)
        self.assertEqual(res.content_type, "text/html")
        self.assertEqual(res.ip_address, self.public_ip)
        self.assertTrue(mock_resp.released)

    def test_02_ssrf_destination_blocked_early(self):
        """Private IP and localhost requests must be blocked before socket creation."""
        pool_created = False

        def spy_pool_factory(*args):
            nonlocal pool_created
            pool_created = True
            return MagicMock()

        client = SafeHTTPClient(
            socket_resolver=self.mock_resolver,
            pool_factory=spy_pool_factory,
        )

        # Localhost URL
        res = client.get("http://127.0.0.1:80/admin")
        self.assertEqual(res.status, FetchStatus.BLOCKED_SSRF)
        self.assertIsNone(res.status_code)
        self.assertFalse(pool_created)
        self.assertIn("prohibited", res.error_message.lower())

        # Cloud Metadata URL
        res_meta = client.get("http://169.254.169.254/latest/meta-data")
        self.assertEqual(res_meta.status, FetchStatus.BLOCKED_SSRF)
        self.assertFalse(pool_created)

    def test_03_ip_pinning_anti_dns_rebinding(self):
        """Client must pass pre-validated IP to connection pool while binding hostname to SNI."""
        captured_args = {}

        def capturing_pool_factory(scheme, pinned_ip, port, server_hostname):
            captured_args["scheme"] = scheme
            captured_args["pinned_ip"] = pinned_ip
            captured_args["port"] = port
            captured_args["server_hostname"] = server_hostname
            mock_pool = MagicMock()
            mock_pool.request.return_value = MockStreamResponse(status=200, body=b"OK")
            return mock_pool

        client = SafeHTTPClient(
            socket_resolver=self.mock_resolver,
            pool_factory=capturing_pool_factory,
        )

        res = client.get("https://example.com/page")
        self.assertEqual(res.status, FetchStatus.SUCCESS)
        self.assertEqual(captured_args["pinned_ip"], self.public_ip)
        self.assertEqual(captured_args["server_hostname"], "example.com")
        self.assertEqual(captured_args["port"], 443)
        self.assertEqual(captured_args["scheme"], "https")

    def test_04_streaming_payload_too_large_via_content_length(self):
        """Responses declaring Content-Length > 2 MB must abort immediately without reading body."""
        read_called = False

        class MonitoredStream(MockStreamResponse):
            def read(self, amt=None):
                nonlocal read_called
                read_called = True
                return b"data"

        mock_resp = MonitoredStream(
            status=200,
            headers={"Content-Type": "text/html", "Content-Length": "3145728"},  # 3 MB
            body=b"large",
        )
        mock_pool = MagicMock()
        mock_pool.request.return_value = mock_resp

        client = SafeHTTPClient(
            socket_resolver=self.mock_resolver,
            pool_factory=lambda *args: mock_pool,
            max_response_size=2 * 1024 * 1024,
        )

        res = client.get("https://example.com/huge")
        self.assertEqual(res.status, FetchStatus.PAYLOAD_TOO_LARGE)
        self.assertFalse(read_called, "Body stream should not have been read!")
        self.assertTrue(mock_resp.released)

    def test_05_streaming_payload_too_large_via_chunk_overflow(self):
        """Responses exceeding 2 MB during streaming must terminate stream and release socket."""
        # 2.5 MB body
        oversized_body = b"X" * (int(2.5 * 1024 * 1024))
        mock_resp = MockStreamResponse(
            status=200,
            headers={"Content-Type": "text/html"},
            body=oversized_body,
            chunk_size=65536,
        )
        mock_pool = MagicMock()
        mock_pool.request.return_value = mock_resp

        client = SafeHTTPClient(
            socket_resolver=self.mock_resolver,
            pool_factory=lambda *args: mock_pool,
            max_response_size=2 * 1024 * 1024,
        )

        res = client.get("https://example.com/stream-overflow")
        self.assertEqual(res.status, FetchStatus.PAYLOAD_TOO_LARGE)
        self.assertIn("exceeded maximum permitted limit", res.error_message.lower())
        self.assertTrue(mock_resp.released)

    def test_06_unsupported_content_type_rejected(self):
        """Binary/executable content types must be rejected before reading stream."""
        read_called = False

        class MonitoredStream(MockStreamResponse):
            def read(self, amt=None):
                nonlocal read_called
                read_called = True
                return b"data"

        bad_types = ["application/octet-stream", "video/mp4", "application/x-executable", "image/png"]
        for ct in bad_types:
            with self.subTest(content_type=ct):
                read_called = False
                mock_resp = MonitoredStream(status=200, headers={"Content-Type": ct})
                mock_pool = MagicMock()
                mock_pool.request.return_value = mock_resp

                client = SafeHTTPClient(
                    socket_resolver=self.mock_resolver,
                    pool_factory=lambda *args: mock_pool,
                )

                res = client.get("https://example.com/binary")
                self.assertEqual(res.status, FetchStatus.UNSUPPORTED_CONTENT_TYPE)
                self.assertFalse(read_called, f"Stream should not be read for {ct}")
                self.assertTrue(mock_resp.released)

    def test_07_safe_redirect_followed(self):
        """Safe 302 redirect to valid relative URL must be followed up to destination."""
        first_resp = MockStreamResponse(status=302, headers={"Location": "/target-article"})
        second_resp = MockStreamResponse(status=200, body=b"<h1>Final Destination</h1>")

        responses = [first_resp, second_resp]

        def sequenced_pool(*args):
            mock_pool = MagicMock()
            mock_pool.request.side_effect = lambda *a, **kw: responses.pop(0)
            return mock_pool

        client = SafeHTTPClient(
            socket_resolver=self.mock_resolver,
            pool_factory=sequenced_pool,
        )

        res = client.get("https://example.com/redirect-me")
        self.assertEqual(res.status, FetchStatus.SUCCESS)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.final_url, "https://example.com/target-article")
        self.assertEqual(res.redirect_history, ["https://example.com/redirect-me"])
        self.assertIn("Final Destination", res.content_text)

    def test_08_redirect_to_private_ip_blocked(self):
        """Redirect attempting to steer client to private IP or localhost must be blocked."""
        evil_redirect_resp = MockStreamResponse(
            status=302,
            headers={"Location": "http://192.168.1.1/admin"},
        )
        mock_pool = MagicMock()
        mock_pool.request.return_value = evil_redirect_resp

        client = SafeHTTPClient(
            socket_resolver=self.mock_resolver,
            pool_factory=lambda *args: mock_pool,
        )

        res = client.get("https://example.com/open-redirect")
        self.assertEqual(res.status, FetchStatus.BLOCKED_SSRF)
        self.assertIn("redirect target blocked", res.error_message.lower())

    def test_09_redirect_loop_and_limit_enforced(self):
        """Redirect chains exceeding max_redirects (3) must return REDIRECT_LOOP status."""
        loop_resp = MockStreamResponse(status=302, headers={"Location": "/loop-step"})
        mock_pool = MagicMock()
        mock_pool.request.return_value = loop_resp

        client = SafeHTTPClient(
            socket_resolver=self.mock_resolver,
            pool_factory=lambda *args: mock_pool,
            max_redirects=3,
        )

        res = client.get("https://example.com/loop-step")
        self.assertEqual(res.status, FetchStatus.REDIRECT_LOOP)
        self.assertIn("maximum redirect limit of 3 exceeded", res.error_message.lower())

    def test_10_http_status_codes_reported_honestly(self):
        """HTTP error status codes must map to explicit, honest FetchStatus enums."""
        status_map = {
            401: FetchStatus.LOGIN_REQUIRED,
            403: FetchStatus.ACCESS_DENIED,
            404: FetchStatus.NOT_FOUND,
            410: FetchStatus.NOT_FOUND,
            429: FetchStatus.RATE_LIMITED,
            500: FetchStatus.SERVER_ERROR,
            503: FetchStatus.SERVER_ERROR,
        }

        for code, expected_status in status_map.items():
            with self.subTest(status_code=code):
                mock_resp = MockStreamResponse(status=code)
                mock_pool = MagicMock()
                mock_pool.request.return_value = mock_resp

                client = SafeHTTPClient(
                    socket_resolver=self.mock_resolver,
                    pool_factory=lambda *args: mock_pool,
                )

                res = client.get(f"https://example.com/error-{code}")
                self.assertEqual(res.status, expected_status)
                self.assertEqual(res.status_code, code)
                self.assertTrue(mock_resp.released)

    def test_11_timeout_handling(self):
        """ConnectTimeoutError and ReadTimeoutError must map to FetchStatus.TIMEOUT."""
        # Connect timeout
        connect_pool = MagicMock()
        connect_pool.request.side_effect = ConnectTimeoutError("Connection timed out after 5.0s")

        client = SafeHTTPClient(
            socket_resolver=self.mock_resolver,
            pool_factory=lambda *args: connect_pool,
        )
        res_connect = client.get("https://example.com/slow-connect")
        self.assertEqual(res_connect.status, FetchStatus.TIMEOUT)
        self.assertIn("timed out", res_connect.error_message.lower())

        # Read timeout
        read_pool = MagicMock()
        read_pool.request.side_effect = ReadTimeoutError(read_pool, "https://example.com", "Read timed out")
        client_read = SafeHTTPClient(
            socket_resolver=self.mock_resolver,
            pool_factory=lambda *args: read_pool,
        )
        res_read = client_read.get("https://example.com/slow-read")
        self.assertEqual(res_read.status, FetchStatus.TIMEOUT)

    def test_12_ssl_error_handling(self):
        """TLS verification failures must return FetchStatus.SSL_ERROR."""
        ssl_pool = MagicMock()
        ssl_pool.request.side_effect = Urllib3SSLError("Certificate verify failed: self-signed certificate")

        client = SafeHTTPClient(
            socket_resolver=self.mock_resolver,
            pool_factory=lambda *args: ssl_pool,
        )
        res = client.get("https://untrusted-cert.example.com")
        self.assertEqual(res.status, FetchStatus.SSL_ERROR)
        self.assertIn("tls/ssl", res.error_message.lower())

    def test_13_no_ambient_proxy_inheritance(self):
        """Real connection pools must be constructed with retries=False and direct host IP without proxy."""
        client = SafeHTTPClient()
        pool = client._create_pinned_pool(
            scheme="https",
            pinned_ip="93.184.216.34",
            port=443,
            server_hostname="example.com",
        )
        self.assertIsInstance(pool, urllib3.HTTPSConnectionPool)
        self.assertEqual(pool.host, "93.184.216.34")
        self.assertEqual(pool.port, 443)
        self.assertEqual(pool.conn_kw.get("server_hostname"), "example.com")
        self.assertEqual(getattr(pool, "assert_hostname", None), "example.com")

    def test_14_no_javascript_execution_verification(self):
        """Client must return static raw bytes and text without invoking any browser or JS engine."""
        payload = b"<html><script>window.location='https://evil.com';</script><h1>Plain Content</h1></html>"
        mock_resp = MockStreamResponse(status=200, body=payload)
        mock_pool = MagicMock()
        mock_pool.request.return_value = mock_resp

        client = SafeHTTPClient(
            socket_resolver=self.mock_resolver,
            pool_factory=lambda *args: mock_pool,
        )

        res = client.get("https://example.com/static-page")
        self.assertEqual(res.status, FetchStatus.SUCCESS)
        self.assertIn("<script>", res.content_text)
        self.assertIn("<h1>Plain Content</h1>", res.content_text)
        # Verify result dictionary format
        d = res.to_dict()
        self.assertEqual(d["status"], "SUCCESS")
        self.assertEqual(d["status_code"], 200)
        self.assertEqual(d["content_type"], "text/html")

    def test_15_connection_pool_and_response_closed_on_success(self):
        """Pool and response must be deterministically closed on successful fetch."""
        mock_resp = MockStreamResponse(status=200, body=b"Hello Moon AI")
        mock_pool = MagicMock()
        mock_pool.request.return_value = mock_resp

        client = SafeHTTPClient(
            socket_resolver=self.mock_resolver,
            pool_factory=lambda *args: mock_pool,
        )
        res = client.get("https://example.com/page")
        self.assertEqual(res.status, FetchStatus.SUCCESS)
        self.assertTrue(mock_resp.closed, "Response should be closed")
        self.assertTrue(mock_resp.released, "Connection should be released")
        mock_pool.close.assert_called_once()

    def test_16_connection_pool_closed_on_failures_and_timeouts(self):
        """Pool must be deterministically closed on connect timeouts, read timeouts, SSL errors, and network errors."""
        # 1. Connect timeout (no response exists)
        pool1 = MagicMock()
        pool1.request.side_effect = ConnectTimeoutError()
        c1 = SafeHTTPClient(socket_resolver=self.mock_resolver, pool_factory=lambda *args: pool1)
        res1 = c1.get("https://example.com/timeout")
        self.assertEqual(res1.status, FetchStatus.TIMEOUT)
        pool1.close.assert_called_once()

        # 2. SSL error
        pool2 = MagicMock()
        pool2.request.side_effect = Urllib3SSLError("SSL cert mismatch")
        c2 = SafeHTTPClient(socket_resolver=self.mock_resolver, pool_factory=lambda *args: pool2)
        res2 = c2.get("https://example.com/ssl-err")
        self.assertEqual(res2.status, FetchStatus.SSL_ERROR)
        pool2.close.assert_called_once()

        # 3. Read timeout during stream (response exists, interrupted halfway)
        resp3 = MockStreamResponse(status=200, body=b"partial data")
        resp3.read = MagicMock(side_effect=ReadTimeoutError(pool2, "https://example.com", "Read timed out"))
        pool3 = MagicMock()
        pool3.request.return_value = resp3
        c3 = SafeHTTPClient(socket_resolver=self.mock_resolver, pool_factory=lambda *args: pool3)
        res3 = c3.get("https://example.com/stream-timeout")
        self.assertEqual(res3.status, FetchStatus.TIMEOUT)
        self.assertTrue(resp3.closed)
        pool3.close.assert_called_once()

    def test_17_connection_pool_closed_on_streaming_limit_exceeded(self):
        """When streaming payload exceeds 2 MB, response and pool must both be closed immediately."""
        big_chunk = b"X" * (500 * 1024)  # 500 KB per chunk
        resp = MockStreamResponse(status=200, body=big_chunk * 5, chunk_size=500 * 1024)  # 2.5 MB total
        mock_pool = MagicMock()
        mock_pool.request.return_value = resp

        client = SafeHTTPClient(
            max_response_size=2 * 1024 * 1024,
            socket_resolver=self.mock_resolver,
            pool_factory=lambda *args: mock_pool,
        )
        res = client.get("https://example.com/oversized")
        self.assertEqual(res.status, FetchStatus.PAYLOAD_TOO_LARGE)
        self.assertTrue(resp.closed)
        mock_pool.close.assert_called_once()

    def test_18_connection_pools_closed_across_redirect_chain(self):
        """In a redirect chain, the initial pool must be closed before following the next hop."""
        pool1 = MagicMock()
        resp1 = MockStreamResponse(status=302, headers={"Location": "https://example.com/target"})
        pool1.request.return_value = resp1

        pool2 = MagicMock()
        resp2 = MockStreamResponse(status=200, body=b"Final destination")
        pool2.request.return_value = resp2

        pools = [pool1, pool2]
        client = SafeHTTPClient(
            socket_resolver=self.mock_resolver,
            pool_factory=lambda *args: pools.pop(0),
        )
        res = client.get("https://example.com/source")
        self.assertEqual(res.status, FetchStatus.SUCCESS)
        self.assertEqual(res.final_url, "https://example.com/target")
        self.assertTrue(resp1.closed)
        self.assertTrue(resp2.closed)
        pool1.close.assert_called_once()
        pool2.close.assert_called_once()

    def test_19_custom_headers_crlf_injection_rejection(self):
        """Custom headers containing CR/LF control characters must be rejected fail-closed."""
        client = SafeHTTPClient(socket_resolver=self.mock_resolver)
        bad_headers = {
            "X-Injected\r\nHeader": "value",
            "X-Normal": "normal_val",
        }
        res = client.get("https://example.com/page", custom_headers=bad_headers)
        self.assertEqual(res.status, FetchStatus.NETWORK_ERROR)
        self.assertIn("illegal control characters", res.error_message)

        bad_val_headers = {
            "X-Header": "value\r\nInjected-Key: bad",
        }
        res2 = client.get("https://example.com/page", custom_headers=bad_val_headers)
        self.assertEqual(res2.status, FetchStatus.NETWORK_ERROR)
        self.assertIn("illegal control characters", res2.error_message)

    def test_20_ipv6_host_header_bracket_formatting(self):
        """When target is an IPv6 literal, Host header must be enclosed in RFC 3986 brackets."""
        mock_resp = MockStreamResponse(status=200, body=b"IPv6 response")
        mock_pool = MagicMock()
        mock_pool.request.return_value = mock_resp

        client = SafeHTTPClient(pool_factory=lambda *args: mock_pool)
        res = client.get("https://[2607:f8b0:4005:805::200e]:8443/test")
        self.assertEqual(res.status, FetchStatus.SUCCESS)

        # Inspect headers passed to pool.request
        call_kwargs = mock_pool.request.call_args[1]
        req_headers = call_kwargs["headers"]
        self.assertEqual(req_headers["Host"], "[2607:f8b0:4005:805::200e]:8443")


if __name__ == "__main__":
    unittest.main()
