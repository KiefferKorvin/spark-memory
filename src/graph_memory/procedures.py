"""Procedural memory, starting with retrieval know-how: which web hosts and retrievers give evidence that is
accepted and used, which were forgotten or blocked. Online search uses it to skip what never helps."""
from urllib.parse import urlsplit

from .models import now


def host(url):
    return (urlsplit(url or "").hostname or "").removeprefix("www.")


def score(stats):
    """Share of found sources that proved useful (accepted or used), with a neutral prior for unknown hosts."""
    return (stats.get("accepted", 0) + stats.get("used", 0) + 1) / (stats.get("found", 0) + 2) / (1 + stats.get("forgotten", 0))


class RetrievalKnowHow:
    # ponytail: read-modify-write counters, so concurrent queries can lose an increment; fine for a trust signal.
    def __init__(self, repository, min_trials=5):
        self.repository, self.min_trials = repository, min_trials

    async def stats(self, kind, name):
        return await self.repository.read_record("procedure", f"{kind}:{name}") or {"kind": kind, "name": name}

    async def count(self, kind, name, **increments):
        if not name:
            return
        stats = await self.stats(kind, name)
        for field, value in increments.items():
            stats[field] = stats.get(field, 0) + value
        await self.repository.record("procedure", f"{kind}:{name}", {**stats, "updated_at": now()})

    async def skip_reason(self, url):
        """Why a found source should not even be relevance-checked, or None."""
        stats = await self.stats("host", host(url))
        if stats.get("blocked"):
            return "host blocked: " + stats["blocked"]["reason"]
        if self.min_trials and stats.get("found", 0) >= self.min_trials and not stats.get("accepted") and not stats.get("used"):
            return f"host never useful (0 of {stats['found']} found sources accepted)"
        return None

    async def block(self, name, reason=None):
        stats = await self.stats("host", name)
        await self.repository.record("procedure", f"host:{name}", {**stats, "updated_at": now(),
                                                                  "blocked": {"reason": reason, "at": now()} if reason else None})

    async def listing(self):
        rows = await self.repository.records("procedure")
        return sorted(({**r, "score": round(score(r), 3)} for r in rows), key=lambda r: (r["kind"], -r["score"], r["name"]))
