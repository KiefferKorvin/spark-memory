import asyncio
import html
import itertools
import json
import logging
import re
from typing import Protocol
from urllib.parse import unquote, urlencode, urlsplit
from xml.etree import ElementTree

from pydantic import Field
import pakt_web

from .models import IngestRequest, Strict
from .parsing import extract
from .sources import MIMES, SafeFetcher

logger = logging.getLogger(__name__)
# Retrieved text is ingested and summarized; cap it so one long page cannot dominate a query's budget.
MAX_SOURCE_CHARS = 12000


class RetrievedSource(Strict):
    title: str
    url: str
    text: str
    mime_type: str = "text/plain"
    metadata: dict = Field(default_factory=dict)


class ExternalRetriever(Protocol):
    async def search(self, query: str, limit: int) -> list[RetrievedSource]: ...


class WikipediaRetriever:
    """Keyless Wikipedia search returning each hit's lead section."""
    def __init__(self, max_bytes, host="en.wikipedia.org"):
        self.host = host
        self.fetcher = SafeFetcher([host], max_bytes)

    async def search(self, query, limit):
        # One request per search: Wikimedia rate-limits bursts (HTTP 429), and TextExtracts only returns
        # several extracts per request for lead sections (exintro), never for full pages.
        data, _, _ = await self.fetcher.fetch(f"https://{self.host}/w/api.php?" + urlencode({
            "action": "query", "format": "json", "formatversion": 2, "generator": "search", "gsrsearch": query,
            "gsrlimit": limit, "prop": "extracts|info", "exintro": 1, "explaintext": 1, "exlimit": "max", "inprop": "url"}))
        pages = sorted(json.loads(data).get("query", {}).get("pages", []), key=lambda p: p.get("index", 0))
        return [RetrievedSource(title=p["title"], url=p["fullurl"], text=p["extract"][:MAX_SOURCE_CHARS],
                                metadata={"retriever": "wikipedia", "page_id": p["pageid"]})
                for p in pages if p.get("extract")]


class PubMedRetriever:
    """Keyless NCBI E-utilities search returning titles and abstracts."""
    BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    def __init__(self, max_bytes):
        self.fetcher = SafeFetcher(["eutils.ncbi.nlm.nih.gov"], max_bytes, mimes=MIMES | {"text/xml", "application/xml"})

    async def search(self, query, limit):
        data, _, _ = await self.fetcher.fetch(self.BASE + "esearch.fcgi?" + urlencode(
            {"db": "pubmed", "term": query, "retmax": limit, "retmode": "json", "sort": "relevance"}))
        ids = json.loads(data)["esearchresult"]["idlist"]
        if not ids:
            return []
        data, _, _ = await self.fetcher.fetch(self.BASE + "efetch.fcgi?" + urlencode({"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}))
        results = []
        for article in ElementTree.fromstring(data).iter("PubmedArticle"):
            pmid = article.findtext(".//PMID")
            node = article.find(".//ArticleTitle")
            title = "".join(node.itertext()).strip() if node is not None else f"PubMed {pmid}"
            abstract = "\n\n".join((part.get("Label") + ": " if part.get("Label") else "") + "".join(part.itertext()).strip()
                                   for part in article.iter("AbstractText"))
            if not abstract:
                continue
            authors = ", ".join(f"{a.findtext('LastName')} {a.findtext('Initials') or ''}".strip()
                                for a in article.iter("Author") if a.findtext("LastName"))
            journal, year = article.findtext(".//Journal/Title") or "", article.findtext(".//PubDate/Year") or ""
            results.append(RetrievedSource(title=title, url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                text=f"{title}\n\nAuthors: {authors}\nJournal: {journal} {year}\nPMID: {pmid}\n\n{abstract}",
                metadata={"retriever": "pubmed", "pmid": pmid}))
        return results


class WebRetriever:
    """Obscura web research with keyless DuckDuckGo search and pinned HTTPS fallback."""
    RESULT = re.compile(r'class="result__a" href="[^"]*?uddg=([^&"]+)[^"]*"[^>]*>(.*?)</a>', re.S)

    def __init__(self, max_bytes, web_browser='obscura', obscura_command=''):
        self.max_bytes = max_bytes
        self.obscura = pakt_web.command(obscura_command) if web_browser == 'obscura' else None
        self.search_fetcher = SafeFetcher(["html.duckduckgo.com"], max_bytes)
        self.fetcher = SafeFetcher([], max_bytes, public_web=True)

    async def search(self, query, limit):
        if self.obscura:
            try:
                hits = await asyncio.to_thread(pakt_web.search, query, self.max_bytes,
                                               self.search_fetcher.validate, self.obscura, limit)
                if hits:
                    return [s for s in await asyncio.gather(*(self.page(h['url'], h['title']) for h in hits)) if s]
            except ValueError:
                logger.info('Obscura search unavailable; trying direct web search')
        data, _, _ = await self.search_fetcher.fetch("https://html.duckduckgo.com/html/?" + urlencode({"q": query}))
        results = {}
        for url, title in self.RESULT.findall(data.decode("utf-8", "replace")):
            url, host = unquote(url), urlsplit(unquote(url)).hostname or ""
            # Wikipedia has its own retriever; search-engine hosts are ads or redirects.
            if url.startswith("https://") and not host.endswith(("duckduckgo.com", "wikipedia.org", "bing.com")):
                results.setdefault(url, html.unescape(re.sub(r"<[^>]+>", "", title)).strip())
        return [s for s in await asyncio.gather(*(self.page(u, t) for u, t in list(results.items())[:limit])) if s]

    async def page(self, url, title):
        try:
            data = None
            if self.obscura and not urlsplit(url).path.lower().endswith('.pdf'):
                try:
                    data, mime, final = await asyncio.to_thread(pakt_web.fetch, url, self.max_bytes,
                                                               self.fetcher.validate, executable=self.obscura)
                except ValueError:
                    logger.info('Obscura page unavailable; trying direct web reader')
            if data is None:
                data, mime, final = await self.fetcher.fetch(url)
            text = await asyncio.to_thread(extract, data, mime, self.max_bytes)
        except (ValueError, OSError) as exc:
            logger.info("Web result %s skipped: %s", url, exc)
            return None
        return RetrievedSource(title=title or final, url=final, text=text[:MAX_SOURCE_CHARS],
                               mime_type="text/plain" if mime == "text/plain" else "text/markdown",
                               metadata={"retriever": "web"})


class CompositeRetriever:
    """Queries every backend concurrently and interleaves results so each contributes before any repeats."""
    def __init__(self, retrievers):
        self.retrievers = retrievers

    async def search(self, query, limit):
        outcomes = await asyncio.gather(*(r.search(query, limit) for r in self.retrievers), return_exceptions=True)
        failures = [o for o in outcomes if isinstance(o, BaseException)]
        for retriever, outcome in zip(self.retrievers, outcomes):
            if isinstance(outcome, BaseException):
                logger.warning("%s failed: %s", type(retriever).__name__, outcome)
        if failures and len(failures) == len(outcomes):
            raise failures[0]
        lists = [o for o in outcomes if not isinstance(o, BaseException)]
        return [s for group in itertools.zip_longest(*lists) for s in group if s][:limit]


RETRIEVERS = {"wikipedia": WikipediaRetriever, "pubmed": PubMedRetriever, "web": WebRetriever}


def build_retriever(names, max_bytes, web_browser='obscura', obscura_command=''):
    names = list(dict.fromkeys(n.strip().lower() for n in names.split(",") if n.strip()))
    if unknown := set(names) - RETRIEVERS.keys():
        raise ValueError(f"Unknown EXTERNAL_RETRIEVERS: {', '.join(sorted(unknown))}")
    return CompositeRetriever([WebRetriever(max_bytes, web_browser, obscura_command) if n == 'web'
                               else RETRIEVERS[n](max_bytes) for n in names]) if names else None


def source_request(source):
    # SDK retrievers already downloaded the source. Keep its URI explicitly as provenance.
    return IngestRequest(title=source.title[:300], text=source.text, mime_type=source.mime_type,
                         source_type="api", metadata={**source.metadata, "original_uri": source.url})
