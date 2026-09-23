import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { reduceEvents, type Trace } from './state.js';

describe('trace replay', () => {
  it('reconstructs needs, candidates, routes, evidence, pruning and external sources deterministically', () => {
    const events = [
      { event_type:'QUERY_DECOMPOSED', metadata:{information_needs:[{id:'N1',description:'Need one'}]} },
      { event_type:'CANDIDATES_GENERATED', information_need_id:'N1', metadata:{nodes:[{node_id:'a',label:'A',kind:'Concept'},{node_id:'x',label:'X',kind:'Chunk'}]} },
      { event_type:'NODE_PRUNED', node_id:'x', information_need_id:'N1', metadata:{} },
      { event_type:'BRANCH_SPAWNED', node_id:'a', information_need_id:'N1', branch_id:'b1', metadata:{path:['a']} },
      { event_type:'NODE_EXPANDED', node_id:'a', information_need_id:'N1', metadata:{node:{node_id:'a',label:'A',kind:'Concept'}} },
      { event_type:'CANDIDATES_GENERATED', node_id:'a', information_need_id:'N1', metadata:{nodes:[{node_id:'b',label:'B',kind:'Chunk'}]} },
      { event_type:'EDGE_TRAVERSED', node_id:'b', metadata:{source:'a',target:'b'},branch_id:'b2' },
      { event_type:'BRANCH_SPAWNED', node_id:'b', information_need_id:'N1', branch_id:'b2', metadata:{path:['a','b']} },
      { event_type:'EVIDENCE_FOUND', node_id:'b', metadata:{evidence:{id:'e',source_node_id:'b',text:'Original'}} },
      { event_type:'NODE_PRUNED', node_id:'b', metadata:{} },
      { event_type:'CANDIDATES_GENERATED', information_need_id:'N1', metadata:{nodes:[{node_id:'a',label:'A',kind:'Concept'}]} },
      { event_type:'EXTERNAL_SOURCE_FOUND', metadata:{url:'https://x.org',title:'X',need_id:'N1',retriever:'web'} },
      { event_type:'SOURCE_INGESTED', metadata:{url:'https://x.org',title:'X',document_id:'d'} },
    ].map((event,i) => ({ ...event,event_id:String(i),query_id:'q',sequence:i+1,timestamp:'' })) as Trace[];
    const early = reduceEvents(events.slice(0, 2));
    assert.equal(early.nodes['need:N1'].kind, 'Need');
    assert.equal(early.nodes.a.state, 'CANDIDATE');
    assert.equal(early.edges['need:N1:a'].caption, 'candidate');
    const final = reduceEvents(events);
    assert.equal(final.nodes.x.state, 'PRUNED');
    assert.equal(final.edges['need:N1:a'].caption, 'route');
    assert.equal(final.edges['a:b'].caption, 'route');
    assert.equal(final.nodes.b.state, 'EVIDENCE');
    assert.equal(final.nodes.a.state, 'EXPANDED');  // reappearing as a candidate does not reset it
    assert.equal(final.nodes.e.kind, 'Evidence');
    assert.equal(final.edges['https://x.org:d'].caption, 'ingested');
    assert.deepEqual(reduceEvents(events), final);
  });
});
