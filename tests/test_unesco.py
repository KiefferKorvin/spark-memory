from pathlib import Path

import pytest
from rdflib import Graph

from graph_memory.graph import InMemoryGraph
from graph_memory.llm import ProviderError
from graph_memory.models import Concept, ConceptSpec, Edge, Relation, TaxonomyResolution
from graph_memory.ontology import OntologyService
from graph_memory.unesco import parse_thesaurus


TTL = '''@prefix s: <http://www.w3.org/2004/02/skos/core#> .
@prefix u: <http://vocabularies.unesco.org/thesaurus/> .
@prefix r: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix o: <http://vocabularies.unesco.org/ontology#> .
u:domain3 a s:Collection, o:Domain; s:prefLabel "Culture"@en; s:member u:music .
u:art a s:Concept; s:prefLabel "Art"@en; s:narrower u:music .
u:sound a s:Concept; s:prefLabel "Sound"@en; s:narrower u:music .
u:music a s:Concept; s:prefLabel "Music"@en, "Musique"@fr;
 s:altLabel "Instrumental music"@en; s:hiddenLabel "musics"@en;
 s:broader u:art, u:sound; s:related u:music;
 s:definition u:definition .
u:definition r:value "Organized sound"@en .
'''


@pytest.fixture
def taxonomy(tmp_path):
    path = tmp_path / 'unesco.ttl'
    path.write_text(TTL, encoding='utf-8')
    return parse_thesaurus(path)


def test_rdf_ttl_equivalence_multilingual_polyhierarchy(taxonomy, tmp_path):
    path = tmp_path / 'unesco.rdf'
    Graph().parse(data=TTL, format='turtle').serialize(path, format='xml')
    assert parse_thesaurus(path).manifest['fingerprint'] == taxonomy.manifest['fingerprint']
    music = next(n for n in taxonomy.nodes if n.external_id == 'music')
    assert music.pref_labels == {'en': ['Music'], 'fr': ['Musique']}
    assert music.descriptions['en'] == ['Organized sound']
    assert 'musics' in music.aliases
    assert taxonomy.manifest['relations'] == {'BROADER_THAN': 2, 'HAS_MEMBER': 1, 'RELATED_TO': 1}
    assert len(taxonomy.manifest['warnings']) == 1


async def loaded(taxonomy):
    repo = InMemoryGraph()
    await repo.import_taxonomy(taxonomy.nodes, taxonomy.edges, taxonomy.manifest)
    return repo


async def test_idempotent_immutable_official_taxonomy(taxonomy):
    repo = await loaded(taxonomy)
    await repo.import_taxonomy(taxonomy.nodes, taxonomy.edges, taxonomy.manifest)
    assert len(repo.nodes) == 4 and len(repo.edges) == 4
    music = await repo.exact_concept('Music')
    assert (await repo.exact_concept('Musique')).id == music.id
    music.aliases.append('invented alias')
    await repo.put([music], [])
    assert 'invented alias' not in (await repo.get(music.id)).aliases
    art = await repo.exact_concept('Art')
    with pytest.raises(ValueError, match='Official'):
        await repo.put([], [Edge(source=music.id, target=art.id, relation=Relation.BROADER_THAN)])
    with pytest.raises(ValueError, match='snapshot'):
        await repo.import_taxonomy(taxonomy.nodes, taxonomy.edges, {**taxonomy.manifest, 'fingerprint': 'changed'})


class Classifier:
    async def structured(self, operation, payload, schema, query_id=None):
        assert operation == 'taxonomy_resolution'
        label = payload['candidate']['label']
        parent_label = 'Piano' if label == 'Rootless voicings' else 'Musique'
        parent = next(n for n in payload['existing'] if n['label'] == parent_label)
        return TaxonomyResolution(reuse_id=None, parent_ids=[parent['id']], confidence=.95)


async def test_local_hierarchy_and_about_targets(taxonomy):
    repo = await loaded(taxonomy)
    ontology = OntologyService(repo, Classifier())
    pending = {}
    linked, edges, skipped = await ontology.link([
        ConceptSpec(label='Rootless voicings', broader=['Piano']),
        ConceptSpec(label='Piano', broader=['Music']),
        ConceptSpec(label='Music')], pending)
    assert not skipped
    assert {'Musique', 'Piano', 'Rootless voicings'} <= {n.label for nodes in linked.values() for n in nodes}
    await repo.put(list(pending.values()), edges)
    piano = await repo.exact_concept('Piano')
    rootless = await repo.exact_concept('Rootless voicings')
    assert piano.origin == rootless.origin == 'LOCAL'
    assert any(e.source == piano.id and e.target == rootless.id for e in repo.edges.values())
    assert {'Art', 'Sound', 'Musique'} == {n.label for n in await repo.unesco_ancestors(rootless.id)}
    before = len(repo.nodes)
    pending = {}
    _, edges, _ = await ontology.link([ConceptSpec(label='Piano')], pending)
    await repo.put(list(pending.values()), edges)
    assert len(repo.nodes) == before


async def test_uncertain_classification_is_skipped_without_orphan(taxonomy):
    class Uncertain:
        async def structured(self, *args):
            return TaxonomyResolution(reuse_id=None, parent_ids=[], confidence=.2)
    repo = await loaded(taxonomy)
    pending = {}
    # One unclassifiable concept must not reject the whole document; it is reported and left unlinked.
    linked, edges, skipped = await OntologyService(repo, Uncertain()).link([ConceptSpec(label='unknown'), ConceptSpec(label='Music')], pending)
    assert [(k['label'], k['reason']) for k in skipped] == [('unknown', 'low_confidence')] and [n.label for n in linked['music']] == ['Musique']
    assert not edges and all(n.origin == 'UNESCO' for n in pending.values())
    assert len(repo.nodes) == 4


async def test_document_ingestion_creates_about_and_deduplicates(taxonomy):
    from graph_memory.config import Settings
    from graph_memory.demo import DemoModels
    from graph_memory.models import IngestRequest
    from graph_memory.service import Memory
    repo = await loaded(taxonomy)
    memory = Memory(Settings(memory_mode='demo', _env_file=None), repo, DemoModels())
    await memory.initialize()
    request = IngestRequest(title='The Jazz Piano Book', mime_type='text/plain', text='Music: Piano players use rootless voicings to leave room for the bassist.')
    # Thesaurus entries without attached content are never retrieval candidates.
    assert not await repo.candidates('music', [], 10)
    result = await memory.ingest(request)
    assert 'Musique' in {n.label for n, _ in await repo.candidates('music', [], 10)}
    about = [e.target for e in repo.edges.values() if e.source == result['document_id'] and e.relation == Relation.ABOUT]
    assert {'Musique', 'Piano', 'Rootless voicings'} <= {repo.nodes[i].label for i in about}
    before = len(repo.nodes), len(repo.edges)
    assert (await memory.ingest(request))['duplicate']
    assert (len(repo.nodes), len(repo.edges)) == before
    await memory.close()


def decide(reuse=None, parents=(), confidence=.9):
    return lambda ids: TaxonomyResolution(reuse_id=ids.get(reuse, reuse), parent_ids=[ids.get(p, p) for p in parents], confidence=confidence)


@pytest.mark.parametrize('reason,label,decision', [
    ('low_confidence', 'unknown', decide(parents=['Musique'], confidence=.2)),
    ('unknown_reuse_id', 'unknown', decide(reuse='invented')),
    ('unanchored_reuse', 'Orphan', decide(reuse='Orphan')),
    ('no_parent', 'unknown', decide()),
    ('unknown_parent_id', 'unknown', decide(parents=['invented'])),
    ('no_anchor', 'Orphan', decide(parents=['Orphan'])),
    ('provider_error', 'unknown', None),
])
async def test_skip_records_name_their_reason(taxonomy, reason, label, decision):
    repo = await loaded(taxonomy)
    await repo.put([Concept(id='orphan', label='Orphan', preferred_label='Orphan')], [])  # LOCAL, no UNESCO anchor
    class Model:
        async def structured(self, operation, payload, schema, query_id=None):
            if decision is None:
                raise ProviderError('unavailable')
            return decision({e['label']: e['id'] for e in payload['existing']})
    _, edges, skipped = await OntologyService(repo, Model()).link([ConceptSpec(label=label, broader=['Music'])], {})
    assert [(k['label'], k['reason']) for k in skipped] == [(label, reason)] and not edges
    if reason == 'low_confidence':
        assert skipped[0]['confidence'] == .2 and [p['label'] for p in skipped[0]['parents']] == ['Musique']
        assert skipped[0]['candidates'] >= 1 and skipped[0]['reuse'] is None


async def test_ingestion_reports_and_persists_skips(taxonomy):
    from graph_memory.config import Settings
    from graph_memory.demo import DemoModels
    from graph_memory.models import IngestRequest
    from graph_memory.service import Memory
    memory = Memory(Settings(memory_mode='demo', _env_file=None), await loaded(taxonomy), DemoModels())
    # Demo concepts: Music is an exact UNESCO label; Acoustics (broader Science) has no parent in this thesaurus.
    result = await memory.ingest(IngestRequest(title='Note', text='Acoustics and music.'))
    assert result['concepts_extracted'] == 2 and result['classification_skips'] == {'low_confidence': 1}
    assert result['unclassified_concepts'] == ['Acoustics']
    review = await memory.repository.read_record('classification_review', result['document_id'])
    assert [k['label'] for k in review['skipped']] == ['Acoustics'] and review['skipped'][0]['confidence'] == 0
    await memory.close()
