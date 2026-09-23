"""Bounded HTTPS fetching with allowlists, address pinning and redirect revalidation."""
import asyncio
import base64
import hashlib
import http.client
import ipaddress
import re
import socket
import ssl
import zlib
from urllib.parse import urljoin, urlsplit

from .models import IngestRequest, Source, stable_id

MIMES = {"text/plain", "text/markdown", "text/html", "application/json", "message/rfc822",
         "application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}


class PinnedHTTPS(http.client.HTTPSConnection):
    def __init__(self, hostname, address):
        super().__init__(hostname, port=443, timeout=15, context=ssl.create_default_context())
        self.address = address

    def connect(self):
        sock = socket.create_connection((self.address, 443), self.timeout)
        try:
            self.sock = self._context.wrap_socket(sock, server_hostname=self.host)
        except BaseException:
            sock.close()
            raise


class SafeFetcher:
    def __init__(self, allowed_hosts, max_bytes, public_web=False, mimes=MIMES):
        self.allowed_hosts = {h.strip().lower() for h in allowed_hosts if h.strip()}
        self.max_bytes = max_bytes
        self.public_web = public_web
        self.mimes = mimes

    def validate(self, url):
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower()
        if (parsed.scheme != "https" or not host or (not self.public_web and host not in self.allowed_hosts) or parsed.username or parsed.password
                or parsed.port not in (None, 443) or "\\" in url or any(ord(c) < 32 for c in url)):
            raise ValueError("Only HTTPS URLs on explicitly allowed source hosts are permitted")
        addresses = {row[4][0] for row in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)}
        if not addresses or any(not ipaddress.ip_address(ip).is_global for ip in addresses):
            raise ValueError("Non-public destination denied")
        # Prefer IPv4: containers commonly lack IPv6 routes ("Network is unreachable").
        return parsed, min(addresses, key=lambda ip: (":" in ip, ip))

    def fetch_sync(self, url):
        for _ in range(4):
            parsed, address = self.validate(url)
            connection = PinnedHTTPS(parsed.hostname, address)
            try:
                path = parsed.path or "/"
                if parsed.query:
                    path += "?" + parsed.query
                # Some search endpoints answer "Accept-Encoding: identity" with a bot challenge (HTTP 202).
                connection.request("GET", path, headers={"Accept-Encoding": "gzip", "User-Agent": "GraphMemory/0.1"})
                response = connection.getresponse()
                if response.status in (301, 302, 303, 307, 308):
                    location = response.getheader("Location")
                    if not location:
                        raise ValueError("Redirect has no destination")
                    url = urljoin(url, location)
                    continue
                if response.status != 200:
                    raise ValueError(f"Source returned HTTP {response.status}")
                mime = response.getheader("Content-Type", "").split(";", 1)[0].lower()
                if mime not in self.mimes:
                    raise ValueError("Unsupported source content type")
                length = response.getheader("Content-Length")
                if length and int(length) > self.max_bytes:
                    raise ValueError("Source exceeds size limit")
                data = response.read(self.max_bytes + 1)
                encoding = (response.getheader("Content-Encoding") or "identity").lower()
                if encoding == "gzip" and len(data) <= self.max_bytes:
                    # Bounded inflation: a compression bomb stops at the same size limit.
                    data = zlib.decompressobj(wbits=31).decompress(data, self.max_bytes + 1)
                elif encoding != "identity":
                    raise ValueError("Unsupported content encoding")
                if len(data) > self.max_bytes:
                    raise ValueError("Source exceeds size limit")
                return data, mime, url
            finally:
                connection.close()
        raise ValueError("Too many redirects")

    async def fetch(self, url):
        return await asyncio.to_thread(self.fetch_sync, url)


class SourceService:
    def __init__(self, settings):
        self.settings = settings
        self.fetcher = SafeFetcher(settings.allowed_source_hosts.split(","), settings.max_source_bytes, settings.public_web_enabled)

    async def identify(self, request: IngestRequest):
        uri = None
        mime = request.mime_type.split(";", 1)[0].lower()
        if request.url:
            data, mime, uri = await self.fetcher.fetch(request.url)
        elif request.content_base64 is not None:
            if len(request.content_base64) > (self.settings.max_source_bytes * 4 // 3 + 8):
                raise ValueError("Source exceeds size limit")
            data = base64.b64decode(request.content_base64, validate=True)
        else:
            data = request.text.encode("utf-8")
            # Retriever adapters may supply already-fetched text and its original URI.
            original = request.metadata.get("original_uri")
            if isinstance(original, str) and urlsplit(original).scheme in ("https", "http"):
                uri = original
        if not data or len(data) > self.settings.max_source_bytes:
            raise ValueError("Empty source or source exceeds size limit")
        if mime not in MIMES:
            raise ValueError("Unsupported source content type")
        filename = re.sub(r"[^\w. -]", "_", re.split(r"[/\\]", request.filename or "")[-1])[:200] or None
        digest = hashlib.sha256(data).hexdigest()
        identity = f"{uri or filename or request.title}:{digest}"
        source = Source(id=stable_id("source", identity), label=request.title,
                        source_type="url" if uri else request.source_type, uri=uri, filename=filename,
                        mime_type=mime, author=request.author, content_hash=digest,
                        published_at=request.published_at.isoformat() if request.published_at else None,
                        metadata=request.metadata)
        return source, data
