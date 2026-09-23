"""Ontology adapters: SKOS, normalized JSON, and streaming SNOMED CT RF2 Snapshot."""
import argparse
import asyncio
import csv
import hashlib
import io
import json
import sqlite3
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

from .models import Concept, Edge, Relation, node_from, stable_id
from .unesco import Taxonomy, parse_skos

IS_A = "116680003"
PREFERRED = "900000000000548007"
FSN = "900000000000003001"


def validate_authority(authority):
    if not authority.strip() or authority == "LOCAL" or len(authority) > 80:
        raise ValueError("Choose a nonempty authority other than LOCAL (max 80 characters)")


def normalized_json(path, authority):
    """Custom adapters can emit this stable node/edge/manifest interchange format."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes = [node_from(n) for n in data["nodes"]]
    edges = [Edge.model_validate(e) for e in data["edges"]]
    validate_snapshot(nodes, edges, authority)
    from .graph import validate_edges
    validate_edges({n.id: n for n in nodes}, edges)
    digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    return Taxonomy(nodes, edges, {**data.get("manifest", {}), "authority": authority,
        "fingerprint": digest, "concepts": sum(n.kind == "Concept" for n in nodes), "groups": sum(n.kind == "TaxonomyGroup" for n in nodes)})


def validate_snapshot(nodes, edges, authority):
    validate_authority(authority)
    if any(n.kind not in ("Concept", "TaxonomyGroup") or n.origin != authority or not n.uri for n in nodes):
        raise ValueError("Imported nodes must be concepts/groups with the selected authority and stable URI")
    if len({n.id for n in nodes}) != len(nodes) or len({n.uri for n in nodes}) != len(nodes):
        raise ValueError("Duplicate ontology IDs or URIs")
    if any(e.relation not in (Relation.BROADER_THAN, Relation.HAS_MEMBER, Relation.RELATED_TO, Relation.SEMANTIC_RELATION) for e in edges):
        raise ValueError("Unsupported ontology relationship")


class RF2Snapshot:
    """Stage RF2 rows in temporary SQLite, never load the full RDF/OWL graph in RAM."""
    def __init__(self, path, authority="SNOMED-CT", language="fr"):
        validate_authority(authority)
        self.authority, self.language = authority, language
        self.temp = tempfile.TemporaryDirectory(prefix="graph-memory-rf2-")
        self.db = sqlite3.connect(str(Path(self.temp.name) / "snapshot.sqlite"))
        try:
            self._load(Path(path))
        except BaseException:
            self.close()
            raise

    def close(self):
        self.db.close()
        self.temp.cleanup()

    def _load(self, path):
        if path.is_dir():
            archives = list(path.rglob("*.zip"))
            if len(archives) != 1:
                raise ValueError("Select one RF2 Snapshot ZIP explicitly")
            path = archives[0]
        self.db.executescript("""
          PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
          CREATE TABLE concept(id TEXT PRIMARY KEY, metadata TEXT);
          CREATE TABLE description(id TEXT PRIMARY KEY, concept TEXT, lang TEXT, type TEXT, term TEXT);
          CREATE TABLE preferred(id TEXT PRIMARY KEY);
          CREATE TABLE relationship(id TEXT PRIMARY KEY, source TEXT, target TEXT, type TEXT, metadata TEXT);
        """)
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if "/Snapshot/" in n and n.endswith(".txt")]
            concept_files = [n for n in names if "/sct2_Concept_" in n]
            relationship_files = [n for n in names if "/sct2_Relationship_" in n]
            if not concept_files or not relationship_files:
                raise ValueError("RF2 archive must include Concept and inferred Relationship Snapshot files")
            for name in names:
                if "/sct2_Concept_" in name:
                    sql = "INSERT INTO concept VALUES (?,?)"
                    convert = lambda r: (r['id'], json.dumps(r))
                elif "/sct2_Description_" in name:
                    sql = "INSERT INTO description VALUES (?,?,?,?,?)"
                    convert = lambda r: (r['id'], r['conceptId'], r['languageCode'], r['typeId'], r['term'])
                elif "/der2_cRefset_Language" in name:
                    sql = "INSERT OR IGNORE INTO preferred VALUES (?)"
                    convert = lambda r: (r['referencedComponentId'],) if r['acceptabilityId'] == PREFERRED else None
                elif "/sct2_Relationship_" in name:
                    sql = "INSERT INTO relationship VALUES (?,?,?,?,?)"
                    convert = lambda r: (r['id'], r['sourceId'], r['destinationId'], r['typeId'], json.dumps(r))
                else:
                    continue
                with archive.open(name) as stream:
                    rows = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig"), delimiter="\t", quoting=csv.QUOTE_NONE)
                    def active():
                        for row in rows:
                            if row['active'] == '1':
                                value = convert(row)
                                if value is not None:
                                    yield value
                    self.db.executemany(sql, active())
            self.db.commit()
        self.db.executescript("CREATE INDEX descriptions_concept ON description(concept); CREATE INDEX relations_source ON relationship(source); CREATE INDEX relations_target ON relationship(target);")
        dangling = self.db.execute("SELECT r.id FROM relationship r LEFT JOIN concept a ON a.id=r.source LEFT JOIN concept b ON b.id=r.target WHERE a.id IS NULL OR b.id IS NULL LIMIT 1").fetchone()
        missing = self.db.execute("SELECT c.id FROM concept c WHERE NOT EXISTS (SELECT 1 FROM description d WHERE d.concept=c.id) LIMIT 1").fetchone()
        if dangling or missing:
            raise ValueError("Incomplete RF2 release: missing dependency concepts or active descriptions")
        # Disk-backed Kahn traversal. Never expand all hierarchy paths in Python or Neo4j.
        self.db.executescript("CREATE TABLE degrees AS SELECT c.id,count(r.id) AS degree FROM concept c LEFT JOIN relationship r ON r.source=c.id AND r.type='116680003' GROUP BY c.id; CREATE UNIQUE INDEX degree_id ON degrees(id);")
        while True:
            self.db.execute("CREATE TEMP TABLE ready AS SELECT id FROM degrees WHERE degree=0")
            self.db.execute("CREATE UNIQUE INDEX ready_id ON ready(id)")
            count = self.db.execute("SELECT count(*) FROM ready").fetchone()[0]
            if not count:
                self.db.execute("DROP TABLE ready")
                break
            self.db.execute("UPDATE degrees SET degree=degree-(SELECT count(*) FROM relationship r JOIN ready p ON r.target=p.id WHERE r.source=degrees.id AND r.type=?) WHERE id IN (SELECT source FROM relationship r JOIN ready p ON r.target=p.id WHERE r.type=?)", (IS_A, IS_A))
            self.db.execute("DELETE FROM degrees WHERE id IN (SELECT id FROM ready)")
            self.db.execute("DROP TABLE ready")
        if self.db.execute("SELECT count(*) FROM degrees").fetchone()[0]:
            raise ValueError("RF2 is-a hierarchy contains a cycle")
        with path.open('rb') as f:
            fingerprint = hashlib.file_digest(f, 'sha256').hexdigest()
        self.manifest = {"authority": self.authority, "format": "snomed-rf2-snapshot", "language": self.language,
            "fingerprint": hashlib.sha256(f"rf2-v1:{self.authority}:{self.language}:{fingerprint}".encode()).hexdigest(),
            "source_file": str(path), "file_sha256": fingerprint,
            "concepts": self.db.execute("SELECT count(*) FROM concept").fetchone()[0], "groups": 0,
            "relations": self.db.execute("SELECT count(*) FROM relationship").fetchone()[0],
            "semantics": "Active inferred is-a hierarchy and typed RF2 attributes; no OWL reasoning or inactive/historical concepts"}

    def identity(self, code):
        return stable_id("ontology:" + self.authority, "http://snomed.info/id/" + code)

    def nodes(self):
        for code, metadata in self.db.execute("SELECT id,metadata FROM concept ORDER BY id"):
            pref, aliases, fsns = defaultdict(list), defaultdict(list), defaultdict(list)
            rows = self.db.execute("SELECT d.lang,d.term,d.type,p.id IS NOT NULL FROM description d LEFT JOIN preferred p ON p.id=d.id WHERE d.concept=? ORDER BY d.lang,d.term", (code,))
            for lang, term, kind, preferred in rows:
                aliases[lang].append(term)
                if kind == FSN:
                    fsns[lang].append(term)
                elif preferred:
                    pref[lang].append(term)
            for lang in aliases:
                if not pref[lang]:
                    pref[lang] = fsns[lang] or aliases[lang][:1]
            label = (pref.get(self.language) or pref.get("en") or next(iter(pref.values())))[0]
            yield Concept(id=self.identity(code), origin=self.authority, uri="http://snomed.info/id/" + code,
                external_id=code, label=label, preferred_label=label, ontology_status="external",
                pref_labels=dict(pref), alt_labels=dict(aliases), descriptions=dict(fsns),
                aliases=sorted({t for terms in aliases.values() for t in terms} - {label}),
                description="; ".join(fsns.get(self.language) or fsns.get('en') or []),
                routing_summary=f"{self.authority}: {label}", metadata={"rf2": json.loads(metadata)})

    def edges(self):
        for code, source, target, kind, metadata in self.db.execute("SELECT id,source,target,type,metadata FROM relationship ORDER BY id"):
            broader = kind == IS_A
            yield Edge(source=self.identity(target if broader else source), target=self.identity(source if broader else target),
                relation=Relation.BROADER_THAN if broader else Relation.SEMANTIC_RELATION,
                metadata={"authority": self.authority, "predicate": "http://snomed.info/id/" + kind, "rf2": json.loads(metadata)})


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path')
    parser.add_argument('--authority', required=True)
    parser.add_argument('--format', choices=['skos', 'snomed-rf2', 'json'], default='skos')
    parser.add_argument('--language', default='fr')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--report')
    args = parser.parse_args()
    validate_authority(args.authority)
    snapshot = RF2Snapshot(args.path, args.authority, args.language) if args.format == 'snomed-rf2' else (
        normalized_json(args.path, args.authority) if args.format == 'json' else parse_skos(args.path, args.authority, args.language))
    try:
        if not args.dry_run:
            from .config import Settings
            from .graph import Neo4jGraph
            repo = Neo4jGraph(Settings())
            try:
                await repo.initialize()
                await repo.import_taxonomy(snapshot.nodes, snapshot.edges, snapshot.manifest)
            finally:
                await repo.close()
        report = {**snapshot.manifest, 'status': 'validated' if args.dry_run else 'imported'}
        if args.report:
            Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps(report, indent=2))
    finally:
        if isinstance(snapshot, RF2Snapshot):
            snapshot.close()


if __name__ == '__main__':
    asyncio.run(main())
