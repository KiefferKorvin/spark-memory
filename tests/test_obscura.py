import json
import subprocess
import sys

import pytest

import pakt_web
from graph_memory.external import WebRetriever


def test_process_output_limits_validation_and_cleanup(monkeypatch):
    real_popen = subprocess.Popen
    processes = []
    script = ["print(" + repr(json.dumps({'url': 'https://example.org/final', 'html': '<h1>Café</h1>'})) + ")"]
    def popen(args, **kwargs):
        assert '--stealth' in args and '--obey-robots' in args
        assert kwargs['env']['OBSCURA_ALLOW_PRIVATE_NETWORK'] == '0'
        assert 'OBSCURA_PROXY' not in kwargs['env']
        process = real_popen([sys.executable, '-c', script[0]], **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(pakt_web.subprocess, 'Popen', popen)
    monkeypatch.setenv('OBSCURA_ALLOW_PRIVATE_NETWORK', '1')
    monkeypatch.setenv('OBSCURA_PROXY', 'http://localhost:8888')
    checked = []
    data, mime, final = pakt_web.fetch('https://example.org', 1000, checked.append, executable='fake')
    assert data.decode() == '<h1>Café</h1>' and mime == 'text/html'
    assert checked == ['https://example.org', final]
    script[0] = 'import time; time.sleep(30)'
    with pytest.raises(ValueError, match='time or output'):
        pakt_web.fetch('https://example.org', 1000, checked.append, executable='fake', timeout=0)
    assert processes[-1].poll() is not None
    for body in ('not json', json.dumps({'url': 'https://example.org', 'html': 'x' * 1001})):
        script[0] = 'print(' + repr(body) + ')'
        with pytest.raises(ValueError):
            pakt_web.fetch('https://example.org', 1000, checked.append, executable='fake')
    def deny(url):
        raise ValueError('Non-public destination denied')
    count = len(processes)
    with pytest.raises(ValueError, match='Non-public'):
        pakt_web.fetch('https://localhost', 1000, deny, executable='fake')
    assert len(processes) == count
    script[0] = 'print(' + repr(json.dumps({'url': 'https://localhost', 'html': '<p>private</p>'})) + ')'
    def public_only(url):
        if 'localhost' in url:
            raise ValueError('Non-public destination denied')
    with pytest.raises(ValueError, match='Non-public'):
        pakt_web.fetch('https://example.org', 1000, public_only, executable='fake')


def test_search_parses_links_without_attribute_order_assumptions():
    parser = pakt_web.SearchResults()
    parser.feed('''<a href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fa&amp;x=1" class="result__a">A <b>&amp; B</b></a>
        <a href="https://example.org/a" class="result__a">duplicate</a>
        <a class="result__a" href="javascript:alert(1)">bad</a>
        <a class="result__a" href="https://user:password@example.org">credentials</a>
        <a class="result__a" href="https://example.org/b">Direct</a>''')
    assert list(parser.results.values()) == [
        {'url': 'https://example.org/a', 'title': 'A & B', 'description': ''},
        {'url': 'https://example.org/b', 'title': 'Direct', 'description': ''}]


async def test_atlas_browser_provenance_and_failure_fallback(monkeypatch):
    monkeypatch.setattr(pakt_web, 'command', lambda *a: 'obscura')
    retriever = WebRetriever(10000)
    monkeypatch.setattr(pakt_web, 'search', lambda *a: [dict(url='https://example.org/start', title='Result')])
    monkeypatch.setattr(pakt_web, 'fetch', lambda *a, **k: (b'<h1>Rendered</h1><p>Evidence from JavaScript.</p>', 'text/html', 'https://example.org/final'))
    results = await retriever.search('research', 2)
    assert results[0].url == 'https://example.org/final' and 'JavaScript' in results[0].text
    def failure(*a, **k):
        raise ValueError('Browser unavailable')
    monkeypatch.setattr(pakt_web, 'fetch', failure)
    async def direct(url):
        return b'Fallback evidence', 'text/plain', url
    monkeypatch.setattr(retriever.fetcher, 'fetch', direct)
    assert (await retriever.page('https://example.org', 'Fallback')).text == 'Fallback evidence'
