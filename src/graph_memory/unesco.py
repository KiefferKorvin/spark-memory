"""Lossless SKOS fields, canonical graph relationships, repeatable local import."""
import argparse
import asyncio
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from rdflib import Graph, Literal, Namespace, RDF, SKOS, URIRef

from .models import Concept, Edge, Relation, TaxonomyGroup, stable_id
from .parsing import truncate

BASE = "http://vocabularies.unesco.org/thesaurus/"
SCHEME = BASE.rstrip("/")
ISO = Namespace("http://purl.org/iso25964/skos-thes#")
UNESCO = Namespace("http://vocabularies.unesco.org/ontology#")
VERSION = "unesco-skos-1"


@dataclass
class Taxonomy:
    nodes: list
    edges: list[Edge]
    manifest: dict


def literals(graph, subject, predicate):
    """SKOS documentation properties may point to a resource carrying rdf:value."""
    result = defaultdict(set)
    for value in graph.objects(subject, predicate):
        values = [value] if isinstance(value, Literal) else graph.objects(value, RDF.value)
        for literal in values:
            if isinstance(literal, Literal):
                result[literal.language or "und"].add(str(literal))
    return {lang: sorted(values) for lang, values in sorted(result.items())}


def preferred(values, language):
    for lang in [language, "en", "fr", *sorted(values)]:
        if values.get(lang):
            return values[lang][0]
    return ""


def parse_thesaurus(path, language="fr"):
    return parse_skos(path, "UNESCO", language, SCHEME, BASE)


def parse_skos(path, authority, language="fr", scheme_uri=None, base_uri=None):
    if not authority.strip() or authority == "LOCAL":
        raise ValueError("An imported ontology needs a named authority other than LOCAL")
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*.ttl")) or sorted(path.glob("*.rdf"))
        if len(files) != 1:
            raise ValueError("Choose one .ttl or .rdf file explicitly")
        path = files[0]
    formats = {".ttl": "turtle", ".rdf": "xml"}
    if path.suffix.lower() not in formats or path.stat().st_size > 50_000_000:
        raise ValueError("Expected a local RDF/XML or Turtle file smaller than 50 MB")
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    if "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
        raise ValueError("XML DTDs and entities are not permitted")
    # Explicit formats and data= never dereference locations, owl:imports or remote contexts.
    graph = Graph().parse(data=text, format=formats[path.suffix.lower()], publicID=scheme_uri or path.resolve().as_uri())
    scheme_uri = scheme_uri or next((str(s) for s in graph.subjects(RDF.type, SKOS.ConceptScheme)), "")
    concepts = set(graph.subjects(RDF.type, SKOS.Concept))
    groups = set(graph.subjects(RDF.type, SKOS.Collection)) - concepts
    subjects = concepts | groups
    if not concepts or any(not isinstance(s, URIRef) or (base_uri and not str(s).startswith(base_uri)) for s in subjects):
        raise ValueError("Expected SKOS concepts with stable URIs in the configured namespace")
    nodes, edges = [], {}
    documentation = (SKOS.definition, SKOS.scopeNote, SKOS.note, SKOS.historyNote,
                     SKOS.editorialNote, SKOS.changeNote)
    for subject in sorted(subjects):
        pref = literals(graph, subject, SKOS.prefLabel)
        alt = literals(graph, subject, SKOS.altLabel)
        hidden = literals(graph, subject, SKOS.hiddenLabel)
        notes = {str(p): literals(graph, subject, p) for p in documentation if list(graph.objects(subject, p))}
        descriptions = defaultdict(set)
        for values in notes.values():
            for lang, entries in values.items():
                descriptions[lang].update(entries)
        descriptions = {lang: sorted(values) for lang, values in sorted(descriptions.items())}
        label = preferred(pref, language)
        if not label:
            raise ValueError(f"Missing preferred label: {subject}")
        aliases = sorted({v for fields in (pref, alt, hidden) for values in fields.values() for v in values} - {label})
        # Keep original predicates/URIs even when inverse and symmetric links normalize to one edge.
        properties = {}
        for predicate in sorted(set(graph.predicates(subject))):
            properties[str(predicate)] = sorted(
                [{"value": str(o), "language": o.language, "datatype": str(o.datatype) if o.datatype else None}
                 if isinstance(o, Literal) else {"uri": str(o)} for o in graph.objects(subject, predicate)],
                key=lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False))
        fields = dict(id=stable_id("unesco" if authority == "UNESCO" else "ontology:" + authority, str(subject)), label=label, preferred_label=label,
            origin=authority, ontology_status="external", uri=str(subject), external_id=str(subject)[len(base_uri):] if base_uri else str(subject).rsplit("/", 1)[-1],
            aliases=aliases, pref_labels=pref, alt_labels=alt, hidden_labels=hidden, descriptions=descriptions,
            description=preferred(descriptions, language), summary=preferred(descriptions, language),
            routing_summary=f"{authority}: {label}. " + preferred(descriptions, language),
            metadata={"scheme_uri": scheme_uri, "skos_properties": properties, "documentation": notes,
                      "is_top_concept": (subject, SKOS.topConceptOf, URIRef(scheme_uri)) in graph})
        if subject in groups:
            role = "domain" if (subject, RDF.type, UNESCO.Domain) in graph else (
                "microthesaurus" if (subject, RDF.type, UNESCO.MicroThesaurus) in graph else "collection")
            node = TaxonomyGroup(**fields, group_type=role)
        else:
            node = Concept(**fields)
        nodes.append(node)
    mappings = {SKOS.broader: Relation.BROADER_THAN, SKOS.narrower: Relation.BROADER_THAN,
                SKOS.related: Relation.RELATED_TO, SKOS.member: Relation.HAS_MEMBER,
                ISO.subGroup: Relation.HAS_MEMBER, ISO.superGroup: Relation.HAS_MEMBER}
    self_related = []
    for subject, predicate, target in graph:
        if subject not in subjects or predicate not in mappings:
            continue
        if target not in subjects:
            raise ValueError(f"Dangling taxonomy relationship: {subject} {predicate} {target}")
        start, end = (target, subject) if predicate in (SKOS.broader, ISO.superGroup) else (subject, target)
        if predicate == SKOS.related:
            start, end = sorted((start, end))
            if start == end:
                self_related.append(str(start))
        edge = Edge(source=stable_id("unesco" if authority == "UNESCO" else "ontology:" + authority, str(start)), target=stable_id("unesco" if authority == "UNESCO" else "ontology:" + authority, str(end)),
                    relation=mappings[predicate], metadata={"authority": authority, "skos_triples": []})
        edges.setdefault(edge.id, edge).metadata["skos_triples"].append([str(subject), str(predicate), str(target)])
    links = sorted(edges.values(), key=lambda e: e.id)
    for edge in links:
        edge.metadata["skos_triples"].sort()
    from .graph import validate_edges
    validate_edges({n.id: n for n in nodes}, links)
    # The fingerprint identifies the thesaurus content, not memory's storage fields (scope is always shared here):
    # hashing those made every stored snapshot "differ" when Node gained a field, and refused startup.
    canonical = json.dumps({"nodes": [n.model_dump(exclude={"scope"}) for n in nodes], "edges": [e.model_dump() for e in links]},
                           sort_keys=True, ensure_ascii=False)
    scheme_properties = {str(p): sorted(str(o) for o in graph.objects(URIRef(scheme_uri), p))
                         for p in set(graph.predicates(URIRef(scheme_uri)))}
    manifest = {"authority": authority, "edition": "2026" if authority == "UNESCO" else "local", "scheme_uri": scheme_uri,
        "parser_version": VERSION, "language": language, "fingerprint": hashlib.sha256(canonical.encode()).hexdigest(),
        "source_file": str(path.resolve()), "file_sha256": hashlib.sha256(raw).hexdigest(),
        "triples": len(graph), "concepts": len(concepts), "groups": len(groups),
        "relations": dict(Counter(e.relation.value for e in links)), "scheme_properties": scheme_properties,
        "warnings": [f"Preserved self-related concept: {uri}" for uri in sorted(set(self_related))]}
    return Taxonomy(nodes, links, manifest)


async def import_thesaurus(repository, path, language="fr"):
    taxonomy = await asyncio.to_thread(parse_thesaurus, path, language)
    await repository.import_taxonomy(taxonomy.nodes, taxonomy.edges, taxonomy.manifest)
    return taxonomy.manifest


def concept_text(node, language):
    """What an imported concept means, for its ontology embedding: labels in the display language and English
    (the thesaurus is multilingual, extracted concepts mostly English), then its scope note, bounded."""
    labels = [node.label, *(v for lang in dict.fromkeys([language, "en"]) for v in node.pref_labels.get(lang, []) + node.alt_labels.get(lang, []))]
    note = preferred(node.descriptions, language)
    return truncate((" ; ".join(dict.fromkeys(labels)) + (". " + note if note else ""))[:1600], 200)


async def embed_taxonomy(repository, models, settings, authority=None, batch=32):
    """Embeds imported concepts for semantic classification candidates. The import itself stays model-free.

    Resumable and idempotent: each concept stores a key of its text and the embedding model, so a rerun embeds
    only concepts that are new, changed or embedded with another model."""
    authority = authority or settings.primary_ontology
    stamp = f"{settings.embedding_model}:{settings.embedding_dimensions}"
    concepts, todo = await repository.taxonomy_concepts(authority), []
    for node, key in concepts:
        text = concept_text(node, settings.unesco_label_language)
        wanted = hashlib.sha256(f"{stamp}:{text}".encode()).hexdigest()[:24]
        if key != wanted:
            todo.append((node.id, wanted, text))
    # Providers take 1.5-16 s per request, so batches go 8 at a time (4,500 concepts: minutes, not half an hour).
    gate = asyncio.Semaphore(8)

    async def one(part):
        async with gate:
            vectors = await models.embed_batch([text for _, _, text in part])
            await repository.set_ontology_embeddings([(i, key, v) for (i, key, _), v in zip(part, vectors)])
    await asyncio.gather(*(one(todo[start:start + batch]) for start in range(0, len(todo), batch)))
    await repository.record("taxonomy_embedding", authority, {"model": settings.embedding_model,
                                                               "dimensions": settings.embedding_dimensions, "concepts": len(concepts)})
    return {"authority": authority, "concepts": len(concepts), "embedded": len(todo)}


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="Local UNESCO .ttl/.rdf file or directory to import")
    parser.add_argument("--embed", action="store_true",
                        help="Embed imported concepts for classification (paid embeddings; resumable, skips unchanged)")
    parser.add_argument("--clear-classification-cache", action="store_true",
                        help="Forget cached skip verdicts so every concept is classified again")
    parser.add_argument("--language", default="fr")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-formats", action="store_true", help="Compare .ttl and .rdf semantic content")
    parser.add_argument("--report", help="Write import manifest as JSON")
    args = parser.parse_args()
    if not (args.path or args.embed or args.clear_classification_cache):
        parser.error("give a thesaurus path to import, --embed and/or --clear-classification-cache")
    if args.path:
        await import_command(args)
    if args.embed or args.clear_classification_cache:
        from .config import Settings
        from .graph import Neo4jGraph
        from .llm import OpenRouter
        settings = Settings()
        repository = Neo4jGraph(settings)
        models = OpenRouter(settings, repository)
        try:
            await repository.initialize()
            if args.clear_classification_cache:
                await repository.delete_records("classification_skip", settings.primary_ontology + ":")
                print("classification skip cache cleared")
            if args.embed:
                print(json.dumps(await embed_taxonomy(repository, models, settings), indent=2))
        finally:
            await models.close()
            await repository.close()


async def import_command(args):
    taxonomy = parse_thesaurus(args.path, args.language)
    if args.verify_formats:
        root = Path(args.path) if Path(args.path).is_dir() else Path(args.path).parent
        for path in [*root.glob("*.rdf"), *root.glob("*.ttl")]:
            other = parse_thesaurus(path, args.language)
            if other.manifest["fingerprint"] != taxonomy.manifest["fingerprint"]:
                raise ValueError("RDF and Turtle semantic contents differ")
        taxonomy.manifest["formats_verified"] = True
    if not args.dry_run:
        from .config import Settings
        from .graph import Neo4jGraph
        settings = Settings()
        repository = Neo4jGraph(settings)  # No model credentials or paid embeddings needed for import.
        try:
            await repository.initialize()
            await repository.import_taxonomy(taxonomy.nodes, taxonomy.edges, taxonomy.manifest)
        finally:
            await repository.close()
    report = {**taxonomy.manifest, "status": "validated" if args.dry_run else "imported"}
    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
