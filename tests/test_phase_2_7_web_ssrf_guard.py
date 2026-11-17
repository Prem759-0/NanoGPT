"""
Moon AI — Phase 2.7 Milestone 1: SSRF Guard Test Suite
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Active Sovereign Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY
Training Gate: AUTHORIZED_BY_PREM = False (LOCKED)

Deterministic Unit Tests for Anti-SSRF, DNS Rebinding Defense, and IP Validation.
Zero live internet requests. Zero external network dependencies.
"""

import socket
import unittest
from typing import List, Tuple

from core.tools.web.ssrf_guard import (
    ALLOWED_WEB_PORTS,
    PROHIBITED_PORTS,
    MaxRedirectsExceededError,
    SSRFSecurityError,
    ValidatedURL,
    is_hostname_allowed,
    is_ip_allowed,
    is_port_allowed,
    resolve_and_validate_host,
    validate_redirect,
    validate_url,
)


class TestSSRFGuard(unittest.TestCase):
    """Rigorous unit tests for Anti-SSRF boundaries and IP whitelisting."""

    def test_01_ipv4_private_and_loopback_rejection(self):
        """IPv4 loopback and RFC 1918 private addresses must be strictly rejected."""
        private_ips = [
            "127.0.0.1",
            "127.0.0.53",
            "127.255.255.254",
            "10.0.0.1",
            "10.254.1.99",
            "172.16.0.1",
            "172.24.10.5",
            "172.31.255.254",
            "192.168.0.1",
            "192.168.1.254",
        ]
        for ip in private_ips:
            with self.subTest(ip=ip):
                allowed, reason = is_ip_allowed(ip)
                self.assertFalse(allowed, f"Expected {ip} to be rejected, but was allowed.")
                self.assertTrue(len(reason) > 0)

    def test_02_ipv4_link_local_and_cgnat_rejection(self):
        """Link-local addresses and Carrier-Grade NAT (RFC 6598) must be strictly rejected."""
        prohibited = [
            "169.254.0.1",
            "169.254.169.254",  # AWS/GCP/Azure IMDS
            "169.254.255.254",
            "100.64.0.1",       # CGNAT
            "100.100.50.25",    # CGNAT
            "100.127.255.254",  # CGNAT
        ]
        for ip in prohibited:
            with self.subTest(ip=ip):
                allowed, reason = is_ip_allowed(ip)
                self.assertFalse(allowed, f"Expected {ip} to be rejected, but was allowed.")
                self.assertTrue(len(reason) > 0)

    def test_03_ipv4_multicast_reserved_and_broadcast_rejection(self):
        """Multicast, 0.0.0.0, documentation, and class E reserved IPs must be rejected."""
        prohibited = [
            "0.0.0.0",
            "224.0.0.1",        # Multicast
            "239.255.255.250",  # Multicast SSDP
            "240.0.0.1",        # Reserved
            "255.255.255.255",  # Broadcast
            "192.0.2.1",        # TEST-NET-1
            "198.51.100.1",     # TEST-NET-2
            "203.0.113.1",      # TEST-NET-3
            "198.18.0.1",       # Benchmarking
        ]
        for ip in prohibited:
            with self.subTest(ip=ip):
                allowed, reason = is_ip_allowed(ip)
                self.assertFalse(allowed, f"Expected {ip} to be rejected, but was allowed.")

    def test_04_cloud_metadata_explicit_endpoints(self):
        """Explicit cloud provider metadata and wire endpoints must be rejected."""
        metadata_targets = [
            "169.254.169.254",  # Universal cloud metadata
            "168.63.129.16",    # Azure wire server
            "100.100.100.200",  # Alibaba internal metadata
            "fd00:ec2::254",    # AWS IMDSv6
        ]
        for target in metadata_targets:
            with self.subTest(target=target):
                allowed, reason = is_ip_allowed(target)
                self.assertFalse(allowed, f"Expected cloud metadata {target} to be rejected.")
                self.assertIn("cloud metadata", reason.lower())

    def test_05_ipv6_loopback_unspecified_and_private_rejection(self):
        """IPv6 loopback, unspecified, ULA, link-local, and multicast must be rejected."""
        prohibited = [
            "::1",                      # Loopback
            "::",                       # Unspecified
            "fc00::1",                  # ULA
            "fd12:3456:789a:1::1",      # ULA
            "fe80::1",                  # Link-local
            "fe80::200:5aee:feaa:20a2", # Link-local
            "ff02::1",                  # Multicast
            "2001:db8::1",              # Documentation
        ]
        for ip in prohibited:
            with self.subTest(ip=ip):
                allowed, reason = is_ip_allowed(ip)
                self.assertFalse(allowed, f"Expected IPv6 {ip} to be rejected.")

    def test_06_ipv4_mapped_ipv6_rejection(self):
        """IPv4-mapped IPv6 representations of prohibited IPv4 addresses must be rejected."""
        prohibited_mapped = [
            "::ffff:127.0.0.1",
            "::ffff:10.0.0.1",
            "::ffff:192.168.1.1",
            "::ffff:169.254.169.254",
            "::ffff:172.16.5.5",
        ]
        for ip in prohibited_mapped:
            with self.subTest(ip=ip):
                allowed, reason = is_ip_allowed(ip)
                self.assertFalse(allowed, f"Expected IPv4-mapped {ip} to be rejected.")
                self.assertIn("mapped to prohibited ipv4", reason.lower())

    def test_07_public_ip_acceptance(self):
        """Standard public routable IPv4 and IPv6 addresses must be accepted."""
        public_ips = [
            "8.8.8.8",                  # Google DNS
            "1.1.1.1",                  # Cloudflare DNS
            "93.184.216.34",            # example.com
            "151.101.1.140",            # Reddit / Fastly
            "2606:4700:4700::1111",      # Cloudflare DNS IPv6
            "2001:4860:4860::8888",      # Google DNS IPv6
        ]
        for ip in public_ips:
            with self.subTest(ip=ip):
                allowed, reason = is_ip_allowed(ip)
                self.assertTrue(allowed, f"Expected public IP {ip} to be allowed, got error: {reason}")
                self.assertEqual(reason, "")

    def test_08_dns_resolution_with_mixed_addresses_rejects_host(self):
        """If DNS returns ANY private IP alongside public IPs, the host must be rejected."""
        def mock_mixed_resolver(host, port, **kwargs):
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port)),
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port)),
            ]

        with self.assertRaises(SSRFSecurityError) as ctx:
            resolve_and_validate_host("evil-split-horizon.com", 443, socket_resolver=mock_mixed_resolver)
        self.assertIn("prohibited IP '127.0.0.1'", str(ctx.exception))

    def test_09_dns_rebinding_simulation(self):
        """Simulated DNS rebinding where resolver flips to private IP must be caught."""
        def mock_rebind_resolver(host, port, **kwargs):
            # Simulates attacker nameserver returning metadata IP
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", port)),
            ]

        with self.assertRaises(SSRFSecurityError) as ctx:
            resolve_and_validate_host("rebind.attacker.com", 80, socket_resolver=mock_rebind_resolver)
        self.assertIn("cloud metadata", str(ctx.exception).lower())

    def test_10_localhost_and_internal_hostnames_rejected(self):
        """Local hostnames and internal suffixes must be rejected before DNS resolution."""
        bad_hosts = [
            "localhost",
            "localhost.localdomain",
            "app.localhost",
            "database.local",
            "cluster.internal",
            "router.lan",
            "server.home.arpa",
            "intranet.corp",
            "metadata.google.internal",
            "instance-data",
        ]
        for host in bad_hosts:
            with self.subTest(host=host):
                allowed, reason = is_hostname_allowed(host)
                self.assertFalse(allowed, f"Expected host {host} to be rejected.")
                self.assertTrue(len(reason) > 0)

    def test_11_prohibited_schemes_and_credentials_rejected(self):
        """Schemes other than http/https and URLs with credentials must raise SSRFSecurityError."""
        invalid_urls = [
            "ftp://example.com/file.txt",
            "file:///etc/passwd",
            "gopher://example.com:70/",
            "data:text/html,<h1>Hello</h1>",
            "javascript:alert(1)",
            "http://user:password@example.com/page",
            "https://admin@example.com/dashboard",
        ]
        for bad_url in invalid_urls:
            with self.subTest(url=bad_url):
                with self.assertRaises(SSRFSecurityError):
                    validate_url(bad_url)

    def test_12_port_whitelisting(self):
        """Only standard web ports must be permitted; internal daemon ports rejected."""
        self.assertTrue(is_port_allowed(80)[0])
        self.assertTrue(is_port_allowed(443)[0])
        self.assertTrue(is_port_allowed(8080)[0])
        self.assertTrue(is_port_allowed(8443)[0])

        for bad_port in [22, 25, 53, 3306, 5432, 6379, 8000, 9200, 27017]:
            with self.subTest(port=bad_port):
                allowed, reason = is_port_allowed(bad_port)
                self.assertFalse(allowed, f"Port {bad_port} should be prohibited.")

    def test_13_redirect_validation_and_limits(self):
        """Redirects must be validated; redirects to private IPs and limit exceedances must fail."""
        def mock_public_resolver(host, port, **kwargs):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]

        # Relative redirect to valid public path -> OK
        res = validate_redirect(
            current_url="https://example.com/old-path",
            redirect_location="/new-path",
            redirect_count=1,
            max_redirects=3,
            socket_resolver=mock_public_resolver,
        )
        self.assertEqual(res.normalized_url, "https://example.com/new-path")

        # Redirect to localhost -> blocked
        with self.assertRaises(SSRFSecurityError):
            validate_redirect(
                current_url="https://example.com/page",
                redirect_location="http://127.0.0.1:80/admin",
                redirect_count=1,
                max_redirects=3,
                socket_resolver=mock_public_resolver,
            )

        # Redirect limit exceeded -> MaxRedirectsExceededError
        with self.assertRaises(MaxRedirectsExceededError):
            validate_redirect(
                current_url="https://example.com/step3",
                redirect_location="/step4",
                redirect_count=3,
                max_redirects=3,
                socket_resolver=mock_public_resolver,
            )

    def test_14_validated_url_immutability(self):
        """ValidatedURL must be an immutable frozen record providing primary_ip."""
        v = ValidatedURL(
            original_url="https://example.com/",
            normalized_url="https://example.com/",
            scheme="https",
            hostname="example.com",
            port=443,
            path="/",
            query="",
            validated_ips=("93.184.216.34", "93.184.216.35"),
        )
        self.assertEqual(v.primary_ip, "93.184.216.34")
        with self.assertRaises(Exception):
            v.hostname = "tampered.com"  # type: ignore

    def test_15_bracketed_ipv6_with_custom_port(self):
        """Bracketed IPv6 destination with non-standard port must normalize brackets and port correctly."""
        v = validate_url("https://[2607:f8b0:4005:805::200e]:8443/api/v1/query?q=moon")
        self.assertEqual(v.scheme, "https")
        self.assertEqual(v.port, 8443)
        self.assertEqual(v.hostname, "2607:f8b0:4005:805::200e")
        self.assertEqual(v.primary_ip, "2607:f8b0:4005:805::200e")
        self.assertEqual(v.normalized_url, "https://[2607:f8b0:4005:805::200e]:8443/api/v1/query?q=moon")

    def test_16_redirect_port_restriction(self):
        """Redirect to disallowed port (SSH:22, Redis:6379, etc.) must be strictly rejected."""
        def mock_public_resolver(host, port, **kwargs):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]

        # Redirect to port 22
        with self.assertRaises(SSRFSecurityError):
            validate_redirect(
                current_url="https://example.com/page",
                redirect_location="https://example.com:22/ssh",
                redirect_count=1,
                max_redirects=3,
                socket_resolver=mock_public_resolver,
            )

        # Redirect to port 6379
        with self.assertRaises(SSRFSecurityError):
            validate_redirect(
                current_url="https://example.com/page",
                redirect_location="http://example.com:6379/info",
                redirect_count=1,
                max_redirects=3,
                socket_resolver=mock_public_resolver,
            )

    def test_17_mixed_dns_resolution_rejection(self):
        """If DNS returns ANY private IP among public IPs, host must be rejected fail-closed."""
        def mock_mixed_resolver(host, port, **kwargs):
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port)),
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", port)),
            ]

        with self.assertRaises(SSRFSecurityError) as ctx:
            resolve_and_validate_host("dualhomed.example.com", 443, socket_resolver=mock_mixed_resolver)
        self.assertIn("prohibited IP", str(ctx.exception))

    def test_18_redirect_to_cloud_metadata_blocked(self):
        """Redirect targeting AWS/GCP cloud metadata IP must be blocked immediately."""
        with self.assertRaises(SSRFSecurityError):
            validate_redirect(
                current_url="https://example.com/index.html",
                redirect_location="http://169.254.169.254/latest/meta-data/",
                redirect_count=1,
                max_redirects=3,
            )


if __name__ == "__main__":
    unittest.main()
