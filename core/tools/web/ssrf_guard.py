"""
Moon AI — Sovereign Anti-SSRF & Network Boundary Guard
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Active Sovereign Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY
Training Gate: AUTHORIZED_BY_PREM = False (LOCKED)

Defends against Server-Side Request Forgery (SSRF), DNS rebinding, internal network
probing, and cloud metadata theft across all external web intelligence operations.
"""

import ipaddress
import re
import socket
import urllib.parse
from dataclasses import dataclass
from typing import Callable, List, Optional, Set, Tuple, Union


class SSRFSecurityError(Exception):
    """Raised when a URL, IP, or network destination violates SSRF security boundaries."""
    pass


class MaxRedirectsExceededError(SSRFSecurityError):
    """Raised when HTTP redirect depth exceeds maximum allowed limit."""
    pass


@dataclass(frozen=True)
class ValidatedURL:
    """Immutable record of an SSRF-validated, DNS-verified network destination."""
    original_url: str
    normalized_url: str
    scheme: str
    hostname: str
    port: int
    path: str
    query: str
    validated_ips: Tuple[str, ...]

    @property
    def primary_ip(self) -> str:
        """Returns the primary validated IP address for pinned socket connection."""
        if not self.validated_ips:
            raise SSRFSecurityError("No validated IP addresses available.")
        return self.validated_ips[0]


# -----------------------------------------------------------------------------
# CIDR Blacklist & Prohibited Networks
# -----------------------------------------------------------------------------

# Prohibited IPv4 Networks (Private, Loopback, Link-Local, CGNAT, Multicast, Reserved)
PROHIBITED_IPV4_NETWORKS: List[ipaddress.IPv4Network] = [
    ipaddress.ip_network("0.0.0.0/8"),          # Current network (RFC 1122)
    ipaddress.ip_network("10.0.0.0/8"),         # Private-Use (RFC 1918)
    ipaddress.ip_network("100.64.0.0/10"),      # Shared Address Space / Carrier-Grade NAT (RFC 6598)
    ipaddress.ip_network("127.0.0.0/8"),        # Loopback (RFC 1122)
    ipaddress.ip_network("169.254.0.0/16"),     # Link-Local / Cloud Metadata (RFC 3927)
    ipaddress.ip_network("172.16.0.0/12"),      # Private-Use (RFC 1918)
    ipaddress.ip_network("192.0.0.0/24"),       # IETF Protocol Assignments (RFC 6890)
    ipaddress.ip_network("192.0.2.0/24"),       # Documentation / TEST-NET-1 (RFC 5737)
    ipaddress.ip_network("192.168.0.0/16"),     # Private-Use (RFC 1918)
    ipaddress.ip_network("198.18.0.0/15"),      # Benchmarking (RFC 2544)
    ipaddress.ip_network("198.51.100.0/24"),    # Documentation / TEST-NET-2 (RFC 5737)
    ipaddress.ip_network("203.0.113.0/24"),     # Documentation / TEST-NET-3 (RFC 5737)
    ipaddress.ip_network("224.0.0.0/4"),        # Multicast (RFC 5771)
    ipaddress.ip_network("240.0.0.0/4"),        # Reserved for Future Use / Broadcast (RFC 1112)
]

# Prohibited IPv6 Networks (Loopback, Unspecified, Unique Local, Link-Local, Multicast, Doc)
PROHIBITED_IPV6_NETWORKS: List[ipaddress.IPv6Network] = [
    ipaddress.ip_network("::/128"),             # Unspecified (RFC 4291)
    ipaddress.ip_network("::1/128"),           # Loopback (RFC 4291)
    ipaddress.ip_network("fc00::/7"),           # Unique Local Address (ULA) (RFC 4193)
    ipaddress.ip_network("fe80::/10"),          # Link-Local Unicast (RFC 4291)
    ipaddress.ip_network("ff00::/8"),           # Multicast (RFC 4291)
    ipaddress.ip_network("2001:db8::/32"),      # Documentation (RFC 3849)
]

# Specific Cloud Metadata & Infrastructure IP Targets (Explicit Defense-in-Depth)
EXPLICIT_PROHIBITED_IPS: Set[str] = {
    "169.254.169.254",   # AWS, GCP, Azure, Oracle, OpenStack, DigitalOcean Metadata
    "168.63.129.16",     # Azure Virtual Network Wire Server
    "100.100.100.200",   # Alibaba Cloud Internal Metadata & DNS
    "fd00:ec2::254",     # AWS IMDSv6 IPv6 Metadata Endpoint
}

# Prohibited Local & Internal Hostnames / Suffixes
PROHIBITED_HOSTNAMES: Set[str] = {
    "localhost",
    "localhost.localdomain",
    "ip6-localhost",
    "ip6-loopback",
    "instance-data",
    "metadata.google.internal",
    "metadata.internal",
}

PROHIBITED_HOSTNAME_SUFFIXES: Tuple[str, ...] = (
    ".localhost",
    ".local",
    ".internal",
    ".lan",
    ".home.arpa",
    ".corp",
    ".test",
    ".localdomain",
    ".invalid",
)

# Standard Allowed Web Ports
ALLOWED_WEB_PORTS: Set[int] = {80, 443, 8080, 8443}

# Explicit Dangerous Ports (Internal Services & Daemons)
PROHIBITED_PORTS: Set[int] = {
    21, 22, 23, 25, 53, 69, 110, 111, 135, 137, 138, 139, 143, 389,
    445, 465, 587, 636, 1433, 1521, 2049, 2375, 2376, 2379, 2380,
    3306, 3389, 5000, 5432, 5672, 5900, 6379, 6443, 8000, 8008, 8500,
    8888, 9000, 9200, 9300, 10250, 10255, 11211, 27017
}


# -----------------------------------------------------------------------------
# Core IP Validation
# -----------------------------------------------------------------------------

def is_ip_allowed(ip_input: Union[str, ipaddress.IPv4Address, ipaddress.IPv6Address]) -> Tuple[bool, str]:
    """
    Validates whether an IP address is a safe, routable public address.
    Strictly rejects private, loopback, link-local, carrier-grade NAT, multicast,
    reserved, unspecified, documentation, cloud metadata, and IPv4-mapped IPv6 ranges.

    Returns:
        Tuple[bool, str]: (is_allowed, rejection_reason_if_disallowed)
    """
    try:
        if isinstance(ip_input, (ipaddress.IPv4Address, ipaddress.IPv6Address)):
            ip = ip_input
        else:
            # Strip brackets if IPv6 string formatted like '[::1]'
            cleaned = str(ip_input).strip("[]")
            ip = ipaddress.ip_address(cleaned)
    except ValueError as e:
        return False, f"Invalid IP address format: {e}"

    ip_str = str(ip)

    # Check explicit prohibited list
    if ip_str in EXPLICIT_PROHIBITED_IPS:
        return False, f"IP address {ip_str} is a known cloud metadata or internal platform endpoint."

    # Handle IPv6 and IPv4-mapped IPv6
    if isinstance(ip, ipaddress.IPv6Address):
        # Handle IPv4-mapped IPv6 (::ffff:x.x.x.x) first
        if ip.ipv4_mapped:
            mapped_ipv4 = ip.ipv4_mapped
            mapped_allowed, mapped_reason = is_ip_allowed(mapped_ipv4)
            if not mapped_allowed:
                return False, f"IPv4-mapped IPv6 address {ip_str} mapped to prohibited IPv4 {mapped_ipv4}: {mapped_reason}"
            return True, ""

        # Check standard IPv6 properties
        if ip.is_loopback:
            return False, f"IPv6 address {ip_str} is a loopback address (::1)."
        if ip.is_unspecified:
            return False, f"IPv6 address {ip_str} is an unspecified address (::)."
        if ip.is_link_local:
            return False, f"IPv6 address {ip_str} is a link-local address (fe80::/10)."
        if ip.is_multicast:
            return False, f"IPv6 address {ip_str} is a multicast address (ff00::/8)."
        if ip.is_private:
            return False, f"IPv6 address {ip_str} is a unique-local private address (fc00::/7)."
        if ip.is_reserved:
            return False, f"IPv6 address {ip_str} is in an IETF reserved range."

        # Check against prohibited IPv6 CIDR blocks
        for net in PROHIBITED_IPV6_NETWORKS:
            if ip in net:
                return False, f"IPv6 address {ip_str} belongs to prohibited network {net}."

        # Check 6to4 addresses (2002::/16) - bits 16..47 contain an IPv4 address
        if ip in ipaddress.ip_network("2002::/16"):
            v4_int = (int(ip) >> 80) & 0xFFFFFFFF
            embedded_v4 = ipaddress.IPv4Address(v4_int)
            emb_allowed, emb_reason = is_ip_allowed(embedded_v4)
            if not emb_allowed:
                return False, f"6to4 IPv6 address {ip_str} embeds prohibited IPv4 {embedded_v4}: {emb_reason}"

        # Check NAT64 prefix (64:ff9b::/96) - last 32 bits contain an IPv4 address
        if ip in ipaddress.ip_network("64:ff9b::/96"):
            v4_int = int(ip) & 0xFFFFFFFF
            embedded_v4 = ipaddress.IPv4Address(v4_int)
            emb_allowed, emb_reason = is_ip_allowed(embedded_v4)
            if not emb_allowed:
                return False, f"NAT64 IPv6 address {ip_str} embeds prohibited IPv4 {embedded_v4}: {emb_reason}"

        return True, ""

    # Handle IPv4
    if isinstance(ip, ipaddress.IPv4Address):
        if ip.is_loopback:
            return False, f"IPv4 address {ip_str} is a loopback address (127.0.0.0/8)."
        if ip.is_private:
            return False, f"IPv4 address {ip_str} is a private RFC 1918 address."
        if ip.is_link_local:
            return False, f"IPv4 address {ip_str} is a link-local address (169.254.0.0/16)."
        if ip.is_multicast:
            return False, f"IPv4 address {ip_str} is a multicast address (224.0.0.0/4)."
        if ip.is_reserved:
            return False, f"IPv4 address {ip_str} is in an IETF reserved range."
        if ip.is_unspecified:
            return False, f"IPv4 address {ip_str} is an unspecified address (0.0.0.0)."

        # Check against prohibited IPv4 CIDR blocks
        for net in PROHIBITED_IPV4_NETWORKS:
            if ip in net:
                return False, f"IPv4 address {ip_str} belongs to prohibited network {net}."

        return True, ""

    return False, "Unknown IP address type."


# -----------------------------------------------------------------------------
# Hostname & Port Security Checks
# -----------------------------------------------------------------------------

def is_hostname_allowed(hostname: str) -> Tuple[bool, str]:
    """
    Validates that a hostname is syntactically safe and does not resolve
    to localhost, internal TLDs, or platform metadata aliases.
    """
    if not hostname:
        return False, "Hostname cannot be empty."

    cleaned_host = hostname.strip().lower()

    # Reject null bytes, control characters, or embedded whitespace
    if any(ord(c) < 32 or ord(c) == 127 for c in cleaned_host):
        return False, "Hostname contains unprintable or control characters."
    if " " in cleaned_host:
        return False, "Hostname cannot contain spaces."

    # Reject prohibited literal hostnames
    if cleaned_host in PROHIBITED_HOSTNAMES:
        return False, f"Hostname '{cleaned_host}' is a prohibited local or metadata hostname."

    # Reject prohibited internal TLDs and domain suffixes
    for suffix in PROHIBITED_HOSTNAME_SUFFIXES:
        if cleaned_host.endswith(suffix):
            return False, f"Hostname '{cleaned_host}' uses prohibited internal suffix '{suffix}'."

    # Safe IDNA normalization
    try:
        ascii_host = cleaned_host.encode("idna").decode("ascii")
    except Exception as e:
        return False, f"Hostname cannot be converted to valid IDNA ASCII: {e}"

    # Basic RFC 1123 label format checks
    labels = ascii_host.split(".")
    for label in labels:
        if not label:
            return False, "Hostname contains empty label (e.g. consecutive dots)."
        if len(label) > 63:
            return False, f"Hostname label '{label}' exceeds maximum length of 63 characters."
        if not re.match(r"^[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?$", label):
            return False, f"Hostname label '{label}' contains invalid characters."

    return True, ""


def is_port_allowed(port: int) -> Tuple[bool, str]:
    """
    Validates port against allowed standard web ports and blocks internal daemon ports.
    """
    if not (1 <= port <= 65535):
        return False, f"Port {port} is out of valid range (1-65535)."

    if port in PROHIBITED_PORTS:
        return False, f"Port {port} is a prohibited internal service/daemon port."

    if port not in ALLOWED_WEB_PORTS:
        return False, f"Port {port} is not in the allowed standard web port whitelist ({sorted(ALLOWED_WEB_PORTS)})."

    return True, ""


# -----------------------------------------------------------------------------
# DNS Resolution & Verification
# -----------------------------------------------------------------------------

def resolve_and_validate_host(
    hostname: str,
    port: int,
    socket_resolver: Optional[Callable] = None
) -> List[str]:
    """
    Resolves hostname to IP addresses and rigorously verifies that EVERY returned
    address is a safe public routable IP.

    If ANY resolved address belongs to a private, loopback, or reserved network,
    the host is immediately rejected to prevent DNS rebinding or dual-homed attacks.

    Args:
        hostname: The hostname or IP string to resolve.
        port: The destination port.
        socket_resolver: Optional callable (default: socket.getaddrinfo) for mocking.

    Returns:
        List[str]: A list of verified public IP address strings.

    Raises:
        SSRFSecurityError: If DNS fails, returns no addresses, or returns any prohibited IP.
    """
    resolver = socket_resolver or socket.getaddrinfo

    # If hostname is already an IP literal, validate directly
    is_literal = False
    cleaned_ip = hostname.strip("[]")
    try:
        ip_obj = ipaddress.ip_address(cleaned_ip)
        is_literal = True
    except ValueError:
        pass  # Hostname is a domain name, proceed to DNS resolution

    if is_literal:
        allowed, reason = is_ip_allowed(ip_obj)
        if not allowed:
            raise SSRFSecurityError(f"Direct IP destination {hostname} is prohibited: {reason}")
        return [str(ip_obj)]

    # Validate hostname syntax and local blacklist first
    host_ok, host_reason = is_hostname_allowed(hostname)
    if not host_ok:
        raise SSRFSecurityError(f"Hostname validation failed: {host_reason}")

    # Perform DNS resolution
    try:
        addr_info = resolver(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise SSRFSecurityError(f"DNS resolution failed for host '{hostname}': {e}")
    except Exception as e:
        raise SSRFSecurityError(f"Unexpected error resolving host '{hostname}': {e}")

    if not addr_info:
        raise SSRFSecurityError(f"DNS resolution returned zero records for host '{hostname}'.")

    # Extract unique resolved IP strings
    resolved_ips: List[str] = []
    seen: Set[str] = set()

    for item in addr_info:
        sockaddr = item[4]
        ip_str = sockaddr[0]
        if ip_str not in seen:
            seen.add(ip_str)
            resolved_ips.append(ip_str)

    if not resolved_ips:
        raise SSRFSecurityError(f"No IP addresses could be parsed from DNS response for '{hostname}'.")

    # Audit EVERY resolved address
    # If ANY resolved address is private/reserved, reject the host completely!
    for ip_str in resolved_ips:
        allowed, reason = is_ip_allowed(ip_str)
        if not allowed:
            raise SSRFSecurityError(
                f"Security interception: DNS resolution for host '{hostname}' returned prohibited IP '{ip_str}'. "
                f"Reason: {reason}. Host rejected."
            )

    return resolved_ips


# -----------------------------------------------------------------------------
# Comprehensive URL Validation
# -----------------------------------------------------------------------------

def validate_url(
    url: str,
    socket_resolver: Optional[Callable] = None
) -> ValidatedURL:
    """
    Rigorously validates a target URL against all SSRF boundaries:
    1. Scheme must be strictly 'http' or 'https'.
    2. Embedded credentials (user:pass@) are strictly rejected.
    3. Hostname cannot be empty, localhost, or internal TLD.
    4. Port must be in allowed web ports.
    5. Hostname DNS is resolved and ALL returned IPs are verified public.

    Returns:
        ValidatedURL: Fully validated URL record with pinned public IPs.

    Raises:
        SSRFSecurityError: If any security boundary is violated.
    """
    if not url or not isinstance(url, str):
        raise SSRFSecurityError("URL must be a non-empty string.")

    cleaned_url = url.strip()

    # Reject null bytes or unprintable characters
    if any(ord(c) < 32 or ord(c) == 127 for c in cleaned_url):
        raise SSRFSecurityError("URL contains illegal control or null characters.")

    try:
        parsed = urllib.parse.urlsplit(cleaned_url)
    except Exception as e:
        raise SSRFSecurityError(f"Failed to parse URL: {e}")

    # 1. Scheme Check
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        raise SSRFSecurityError(
            f"Prohibited URL scheme '{scheme}'. Only 'http' and 'https' protocols are permitted."
        )

    # 2. Reject Embedded Credentials (RFC 3986 userinfo)
    if parsed.username or parsed.password or ("@" in (parsed.netloc.split(":")[0] if parsed.netloc else "")):
        raise SSRFSecurityError("Prohibited URL format: URLs with embedded credentials are not permitted.")

    # 3. Hostname Check
    hostname = parsed.hostname
    if not hostname:
        raise SSRFSecurityError("URL must contain a valid, non-empty hostname.")

    # 4. Port Check
    port = parsed.port
    if port is None:
        port = 443 if scheme == "https" else 80

    port_ok, port_reason = is_port_allowed(port)
    if not port_ok:
        raise SSRFSecurityError(f"URL port validation failed: {port_reason}")

    # 5. DNS Resolution and IP Validation
    validated_ips = resolve_and_validate_host(hostname, port, socket_resolver=socket_resolver)

    # Build normalized URL
    path = parsed.path or "/"
    query = parsed.query or ""
    
    # Standardize netloc for normalized URL
    default_port = 443 if scheme == "https" else 80
    host_for_netloc = f"[{hostname}]" if (":" in hostname and not hostname.startswith("[")) else hostname
    if port == default_port:
        netloc = host_for_netloc
    else:
        netloc = f"{host_for_netloc}:{port}"

    normalized_url = urllib.parse.urlunsplit((scheme, netloc, path, query, ""))

    return ValidatedURL(
        original_url=cleaned_url,
        normalized_url=normalized_url,
        scheme=scheme,
        hostname=hostname,
        port=port,
        path=path,
        query=query,
        validated_ips=tuple(validated_ips),
    )


# -----------------------------------------------------------------------------
# Redirect Validation
# -----------------------------------------------------------------------------

def validate_redirect(
    current_url: str,
    redirect_location: str,
    redirect_count: int,
    max_redirects: int = 3,
    socket_resolver: Optional[Callable] = None
) -> ValidatedURL:
    """
    Validates an HTTP redirect target:
    1. Enforces maximum redirect count (default 3).
    2. Resolves relative redirects safely against current URL.
    3. Runs complete SSRF validation on the target destination.

    Returns:
        ValidatedURL: Validated destination for the next request.

    Raises:
        MaxRedirectsExceededError: If redirect count exceeds max_redirects.
        SSRFSecurityError: If redirect destination violates SSRF boundaries.
    """
    if redirect_count >= max_redirects:
        raise MaxRedirectsExceededError(
            f"Maximum redirect limit of {max_redirects} exceeded (current count: {redirect_count})."
        )

    if not redirect_location or not isinstance(redirect_location, str):
        raise SSRFSecurityError("Redirect location header is empty or invalid.")

    # Safely join relative redirect location with current URL
    try:
        resolved_target = urllib.parse.urljoin(current_url, redirect_location.strip())
    except Exception as e:
        raise SSRFSecurityError(f"Failed to resolve redirect target '{redirect_location}': {e}")

    # Full SSRF validation on target
    return validate_url(resolved_target, socket_resolver=socket_resolver)
