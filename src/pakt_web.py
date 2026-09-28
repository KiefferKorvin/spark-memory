"""Small, stdlib-only Obscura adapter shared by PAKT and standalone Memory Atlas."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
from html.parser import HTMLParser
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit

_SLOTS = threading.BoundedSemaphore(3)


def command(configured=None):
    if os.environ.get('PAKT_WEB_BROWSER', 'obscura') == 'direct':
        return None
    configured = configured or os.environ.get('PAKT_OBSCURA_COMMAND')
    if configured:
        return configured
    local = Path(__file__).resolve().parents[2] / '.tools/obscura/obscura.exe'
    return shutil.which('obscura') or (str(local) if local.is_file() else None)


def fetch(url, max_bytes, validate, *, executable=None, obey_robots=True, timeout=25):
    """Render one public page. Caller validation runs before and after navigation.

    Obscura also checks resolved destinations for redirects and JS subrequests.
    Processes have isolated cookies, bounded output and a hard lifetime.
    """
    validate(url)
    executable = executable or command()
    if not executable:
        raise ValueError('Obscura is not installed')
    env = dict(os.environ, OBSCURA_ALLOW_PRIVATE_NETWORK='0', OBSCURA_NAV_CHAIN_LIMIT='5',
               OBSCURA_NAV_TIMEOUT_MS=str(timeout * 1000), OBSCURA_SCRIPT_DEADLINE_MS=str(timeout * 1000))
    env.pop('OBSCURA_PROXY', None)
    args = [executable, '--stealth', '--v8-flags', '--max-old-space-size=256']
    if obey_robots:
        args.append('--obey-robots')
    args += ['fetch', url, '--quiet', '--timeout', str(timeout), '--eval',
             'JSON.stringify({url:location.href,html:document.documentElement.outerHTML,contentType:document.contentType})']
    with _SLOTS, tempfile.TemporaryFile() as output:
        try:
            with subprocess.Popen(args, stdout=output, stderr=subprocess.DEVNULL, env=env,
                                  creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)) as process:
                deadline = time.monotonic() + timeout + 2
                try:
                    while process.poll() is None:
                        if time.monotonic() > deadline or os.fstat(output.fileno()).st_size > max_bytes * 6 + 4096:
                            raise ValueError('Obscura exceeded its time or output limit')
                        time.sleep(.05)
                    if process.returncode:
                        raise ValueError('Obscura could not read this public page')
                finally:
                    if process.poll() is None:
                        process.kill()
                    process.wait()
        except OSError as exc:
            raise ValueError('Obscura could not be started') from exc
        output.seek(0)
        try:
            page = json.loads(output.read(max_bytes * 6 + 4097))
            final, data = page['url'], page['html'].encode('utf-8')
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ValueError('Obscura returned an invalid page') from exc
    if page.get('contentType') not in (None, '', 'text/html', 'application/xhtml+xml', 'text/plain'):
        raise ValueError('Non-HTML source requires the direct reader')
    validate(final)
    if not data or len(data) > max_bytes:
        raise ValueError('Obscura page exceeds size limit or is empty')
    return data, 'text/html', final


class SearchResults(HTMLParser):
    """DuckDuckGo HTML results, including both direct and redirect links."""
    def __init__(self):
        super().__init__()
        self.results = {}
        self.url = None
        self.title = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'a' and 'result__a' in attrs.get('class', '').split():
            href = urljoin('https://html.duckduckgo.com', attrs.get('href', ''))
            try:
                parsed = urlsplit(href)
                if parsed.hostname in ('duckduckgo.com', 'html.duckduckgo.com'):
                    href = parse_qs(parsed.query).get('uddg', [''])[0]
                parsed = urlsplit(href)
                if parsed.scheme in ('https', 'http') and parsed.hostname and not parsed.username and not parsed.password:
                    self.url, self.title = href, []
            except ValueError:
                pass

    def handle_data(self, text):
        if self.url:
            self.title.append(text)

    def handle_endtag(self, tag):
        if tag == 'a' and self.url:
            title = ''.join(self.title).strip()
            if title:
                self.results.setdefault(self.url, dict(url=self.url, title=title, description=''))
            self.url = None


def search(query, max_bytes, validate, executable=None, limit=12):
    data, _, _ = fetch('https://html.duckduckgo.com/html/?' + urlencode({'q': query}),
                       max_bytes, validate, executable=executable, obey_robots=False)
    parser = SearchResults()
    parser.feed(data.decode('utf-8', 'replace'))
    return list(parser.results.values())[:limit]
