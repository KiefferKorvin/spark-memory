import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { InteractiveNvlWrapper } from '@neo4j-nvl/react';
import type { Node as NvlNode, Relationship } from '@neo4j-nvl/base';
import { api, stream } from './api';
import { branchColor, reduceEvents, type GraphNode, type Trace } from './state';
import './style.css';
import { Answer, FormattedText } from './Answer';

const kinds: Record<string, string> = { Need: '#f5f0e1', Concept: '#aabaff', Document: '#80c7bf', Section: '#c4acd5', Chunk: '#e2be87', Assertion: '#ed9f94', Evidence: '#71dec0', Source: '#9aadc8', ExternalConcept: '#9daac9' };
const states: Record<string, string> = { PRUNED: '#3b4254', DEAD_END: '#654e5b', EVIDENCE: '#79e7c4', ACTIVE: '#ffffff', SELECTED: '#f3cf83' };

function App() {
  const [token, setToken] = useState('');
  const [query, setQuery] = useState('Why are rootless voicings useful and how should I practice them?');
  const [queryId, setQueryId] = useState('');
  const [events, setEvents] = useState<Trace[]>([]);
  const [cursor, setCursor] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [result, setResult] = useState<any>(null);
  const [selected, setSelected] = useState<any>(null);
  const [manual, setManual] = useState<{ nodes: Record<string, GraphNode>; edges: Record<string, any> }>({ nodes: {}, edges: {} });
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [mode, setMode] = useState('');
  const [note, setNote] = useState('');
  const [sourceUrl, setSourceUrl] = useState('');
  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const [inputMode, setInputMode] = useState('text');
  const [title, setTitle] = useState('');
  const [savedQueryId, setSavedQueryId] = useState('');
  const [pulse, setPulse] = useState(false);
  const [showPruned, setShowPruned] = useState(false);
  const abort = useRef<AbortController | null>(null);
  const latestSequence = useRef(0);
  const graph = useMemo(() => reduceEvents(events.slice(0, cursor ?? events.length)), [events, cursor]);
  const visibleNodes = { ...graph.nodes, ...manual.nodes };
  const visibleEdges = { ...graph.edges, ...manual.edges };
  // Pruned candidates are hidden by default so the explored route stays readable; the timeline keeps them.
  const pruned = Object.values(visibleNodes).filter(n => n.state === 'PRUNED').length;
  const nodes: NvlNode[] = Object.values(visibleNodes).filter(n => !hidden.has(n.id) && (showPruned || n.state !== 'PRUNED')).map(n => ({ id: n.id, caption: n.origin ? `${n.origin}: ${n.label}` : n.label, captionSize: 12,
    color: states[n.state] || kinds[n.kind] || '#aabaff', size: n.kind === 'Need' ? 34 : n.kind === 'Concept' ? 29 : n.kind === 'Evidence' ? 15 : 21,
    selected: selected?.id === n.id, pinned: false }));
  const ids = new Set(nodes.map(n => n.id));
  const rels: Relationship[] = Object.values(visibleEdges).filter(e => ids.has(e.from) && ids.has(e.to)).map(e => ({
    id: e.id, from: e.from, to: e.to, caption: e.caption === 'candidate' ? '' : e.caption || e.relation,
    color: e.caption === 'candidate' ? '#3d4660' : branchColor(e.branch), width: e.active ? (pulse ? 5 : 2) : e.caption === 'route' ? 2 : 1,
  }));

  useEffect(() => { api('/health', token).then(r => setMode(r.mode)).catch(e => setError(e.message)); }, [token]);
  useEffect(() => () => abort.current?.abort(), []);
  useEffect(() => {
    if ((!busy && !playing) || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const timer = setInterval(() => setPulse(current => !current), 400);
    return () => clearInterval(timer);
  }, [busy, playing]);
  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(() => setCursor(current => {
      const next = (current ?? 0) + 1;
      if (next >= events.length) { setPlaying(false); return events.length; }
      return next;
    }), 220);
    return () => clearInterval(timer);
  }, [playing, events.length]);

  async function inspect(id: string) {
    const local = visibleNodes[id];
    const evidence = result?.evidence?.find((e: any) => e.id === id);
    if (evidence) { setSelected({ id, label: 'Evidence', ...evidence }); return; }
    setSelected(local);
    if (local?.kind === 'Need' || (local?.kind === 'Source' && id.startsWith('https:'))) return;
    try { setSelected({ ...local, ...await api(`/nodes/${id}`, token) }); }
    catch (e) { if (local?.kind !== 'Evidence') setError((e as Error).message); }
  }

  async function expand() {
    try {
      const data = await api(`/nodes/${selected.id}/neighbors`, token);
      setManual(current => ({ nodes: { ...current.nodes, ...Object.fromEntries(data.nodes.map((n: any) => [n.id, { ...n, state: 'UNVISITED' }])) },
        edges: { ...current.edges, ...Object.fromEntries(data.edges.map((e: any) => [e.id, { ...e, from: e.source, to: e.target }])) } }));
      setHidden(current => { const next = new Set(current); data.nodes.forEach((n: any) => next.delete(n.id)); return next; });
    } catch (e) { setError((e as Error).message); }
  }

  async function browseTaxonomy() {
    try {
      const data = await api('/taxonomy', token);
      if (!data.manifest) { setNotice('The selected ontology has not been imported into this database.'); return; }
      setManual(current => ({ ...current, nodes: { ...current.nodes, ...Object.fromEntries(data.roots.map((n: any) => [n.id, { ...n, state: 'UNVISITED' }])) } }));
      setHidden(new Set());
      setNotice(`${data.authority} ${data.manifest.edition}: ${data.manifest.concepts} concepts. Select a domain and expand its neighbors.`);
    } catch (e) { setError((e as Error).message); }
  }

  function collapse() {
    const remove = new Set<string>(), pending = [selected.id];
    while (pending.length) {
      const parent = pending.pop();
      for (const edge of Object.values(visibleEdges)) if (edge.from === parent && edge.to !== selected.id && !remove.has(edge.to)) {
        remove.add(edge.to); pending.push(edge.to);
      }
    }
    setHidden(current => new Set([...current, ...remove]));
  }

  async function follow(id: string) {
    abort.current?.abort(); abort.current = new AbortController();
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        await stream(id, token, latestSequence.current, abort.current.signal, event => {
          // The stream is ordered and resumes after the last sequence, so a cursor check deduplicates.
          if (event.sequence <= latestSequence.current) return;
          latestSequence.current = event.sequence;
          setEvents(current => [...current, event]);
        });
        const completed = await api(`/query/${id}`, token);
        if (completed.status !== 'running') { setResult(completed); if (completed.error) setError(completed.error); return; }
      } catch (e) { if (abort.current.signal.aborted || attempt === 2) throw e; }
      await new Promise(resolve => setTimeout(resolve, 500 * (attempt + 1)));
    }
    throw new Error('Stream interrupted. Open this query ID to resume.');
  }

  async function run() {
    setBusy(true); setError(''); setEvents([]); setCursor(null); setPlaying(false); setResult(null); setSelected(null);
    setManual({ nodes: {}, edges: {} }); setHidden(new Set()); latestSequence.current = 0;
    try { const data = await api('/query', token, { method: 'POST', body: JSON.stringify({ query, allow_external: true }) }); setQueryId(data.query_id); await follow(data.query_id); }
    catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  }

  async function load() {
    setBusy(true); setError('');
    try {
      const id = savedQueryId.trim(), loaded = await api(`/query/${id}/events`, token);
      setQueryId(id); setEvents(loaded); setCursor(null); setPlaying(false); setManual({ nodes: {}, edges: {} }); setHidden(new Set());
      latestSequence.current = loaded.at(-1)?.sequence || 0;
      setResult(await api(`/query/${id}`, token)); await follow(id);
    } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  }

  async function ingest() {
    setError(''); setBusy(true);
    try {
      let body: any = { title, text: note, mime_type: 'text/markdown' };
      if (inputMode === 'url') body = { title, url: sourceUrl };
      if (inputMode === 'file') {
        if (!sourceFile) throw new Error('Choose a file first');
        if (sourceFile.size > 5_000_000) throw new Error('File exceeds the 5 MB upload limit');
        const encoded = await new Promise<string>((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(String(reader.result).split(',')[1]); reader.onerror = () => reject(new Error('Could not read file')); reader.readAsDataURL(sourceFile); });
        const extension = sourceFile.name.split('.').pop()?.toLowerCase();
        const mime: Record<string, string> = { pdf: 'application/pdf', docx: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', md: 'text/markdown', txt: 'text/plain' };
        if (!extension || !mime[extension]) throw new Error('Choose PDF, DOCX, Markdown or plain text');
        body = { title, filename: sourceFile.name, mime_type: mime[extension], content_base64: encoded, source_type: 'file' };
      }
      const response = await api('/ingest', token, { method: 'POST', body: JSON.stringify(body) }); setNote(''); setNotice(response.duplicate ? 'This source is already in memory.' : 'Source added to memory.'); }
    catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  }

  const shown = events.slice(0, cursor ?? events.length);
  return <div className="app">
    <header><a className="brand" href="#">◈ <span>Memory Atlas</span></a><span className="subtitle">Progressive graph memory</span><span className="mode">{mode === 'demo' ? 'OFFLINE DEMO' : mode === 'online_demo' ? 'ONLINE DEMO · DeepSeek' : 'LIVE MEMORY'}</span></header>
    <main>
      <section className="workspace">
        <div className="intro"><div className="eyebrow">FOLLOW THE EVIDENCE</div><h1>Watch a question become knowledge.</h1><p>Explore connected sources, inspect decisions, and see where memory still has gaps.</p></div>
        <form className="query" onSubmit={e => { e.preventDefault(); void run(); }}><label className="sr-only" htmlFor="question">Your question</label><textarea id="question" value={query} onChange={e => setQuery(e.target.value)} rows={2}/><button disabled={busy || !query.trim()}>{busy ? 'Exploring…' : 'Explore graph ↗'}</button></form>
        {error && <p role="alert" className="error">{error}</p>}
        {notice && <p role="status" className="muted">{notice}</p>}
        <div className="graph-shell">
          <div className="graph-toolbar"><span><i className={busy ? 'pulse' : ''}/> {busy ? `Exploration live${shown.length ? ' · ' + shown[shown.length - 1].event_type.replaceAll('_', ' ').toLowerCase() : ''}` : events.length ? 'Exploration recorded' : 'Ready to explore'}</span><span>{nodes.length} nodes · {rels.length} connections</span><button onClick={() => setShowPruned(v => !v)} disabled={!pruned}>{showPruned ? 'Hide' : 'Show'} pruned ({pruned})</button><button onClick={() => void browseTaxonomy()}>Browse ontology</button><button onClick={() => setHidden(new Set())}>Show all</button></div>
          <div className="graph" aria-label="Interactive knowledge graph">
            {nodes.length ? <InteractiveNvlWrapper nodes={nodes} rels={rels} nvlOptions={{ layout: 'd3Force', renderer: 'canvas', initialZoom: 0.75, disableTelemetry: true }} mouseEventCallbacks={{ onNodeClick: node => { void inspect(node.id); }, onPan: true, onZoom: true, onDrag: true }} /> : <div className="empty"><div className="orbit">◈</div><h2>Your memory, connected.</h2><p>Ingest a source or load the demo, then ask a question.<br/>The graph will grow as exploration unfolds.</p>{['demo', 'online_demo'].includes(mode) && <button disabled={busy} onClick={async () => { setBusy(true); try { await api('/demo/seed', token, { method: 'POST' }); setNotice('Three synthetic sources loaded. Ready to explore.'); } catch(e) { setError((e as Error).message); } finally {setBusy(false);} }}>Load synthetic sources</button>}</div>}
          </div>
          <div className="legend">{Object.entries(kinds).filter(([k]) => k !== 'ExternalConcept').map(([kind, color]) => <span key={kind}><b style={{background: color}}/>{kind}</span>)}<span><b style={{background:'#3b4254'}}/>Pruned</span></div>
        </div>
        <div className="replay"><button disabled={!events.length || busy} onClick={() => { setCursor(0); setPlaying(true); }}>▶ Replay</button><button disabled={!playing} onClick={() => setPlaying(false)}>Pause</button><input aria-label="Exploration timeline position" type="range" min="0" max={events.length} value={cursor ?? events.length} onChange={e => {setPlaying(false);setCursor(Number(e.target.value));}}/><span>{cursor ?? events.length} / {events.length}</span><button onClick={() => {setCursor(null);setPlaying(false);}}>Latest</button></div>
        <div className="lower"><section className="panel"><h2>Evidence & coverage</h2>{result?.coverage?.coverage.map((c: any) => <div className="coverage" key={c.information_need_id}><span className={c.status.toLowerCase()}>{c.status}</span><div><strong>{result.information_needs.find((n: any) => n.id === c.information_need_id)?.description}</strong><p>{c.missing || `${c.evidence_ids.length} supporting passages`}</p></div></div>)}{result?.answer ? <Answer text={result.answer} evidence={result.evidence || []} inspect={id => void inspect(id)}/> : <p className="muted">Grounded answers and missing information appear here.</p>}</section><section className="panel timeline"><h2>Execution timeline <small>{shown.length} events</small></h2><ol>{shown.map(e => <li key={e.event_id}><time>{new Date(e.timestamp).toLocaleTimeString()}</time><button onClick={() => e.node_id && void inspect(e.node_id)} disabled={!e.node_id}>{e.event_type.replaceAll('_', ' ').toLowerCase()}</button><span style={{color:branchColor(e.branch_id)}}>{e.information_need_id}</span></li>)}</ol></section></div>
      </section>
      <aside><section className="panel inspector"><div className="eyebrow">NODE INSPECTOR</div>{selected ? <><h2>{selected.label}</h2><span className="tag">{selected.kind || selected.source_type} · {selected.state || ''}</span>{selected.metadata?.content_format === "text" ? <pre className="source-plain">{selected.metadata.formatted_content}</pre> : <FormattedText text={selected.metadata?.formatted_content || selected.text || selected.summary || ""}/>}<h3>Routing summary</h3><p>{selected.routing_summary || 'Evidence leaf'}</p>{selected.need && <p>Information need: {selected.need}</p>}<div className="actions"><button onClick={() => void expand()}>Expand neighbors</button><button onClick={collapse}>Collapse branch</button></div><h3>Provenance</h3><pre>{JSON.stringify(selected.provenance || {}, null, 2)}</pre></> : <><h2>Every route has a reason.</h2><p className="muted">Select a node to inspect its routing summary, source and supporting evidence.</p></>}</section>
        <section className="panel"><h2>Add to memory</h2><label>Title (optional — generated automatically)<input value={title} onChange={e => setTitle(e.target.value)}/></label><label>Source type<select value={inputMode} onChange={e => setInputMode(e.target.value)}><option value="text">Text / Markdown</option><option value="file">PDF / DOCX / text file</option><option value="url">Public website</option></select></label>{inputMode === "url" ? <label>Public HTTPS URL<input type="url" value={sourceUrl} onChange={e => setSourceUrl(e.target.value)}/></label> : inputMode === "file" ? <label>Document<input type="file" accept=".pdf,.docx,.txt,.md" onChange={e => setSourceFile(e.target.files?.[0] || null)}/></label> : <label>Text or Markdown<textarea rows={5} value={note} onChange={e => setNote(e.target.value)} placeholder="Paste a note, article, or structured document…"/></label>}<button disabled={busy || (inputMode === "text" ? !note.trim() : inputMode === "url" ? !sourceUrl.trim() : !sourceFile)} onClick={() => void ingest()}>Ingest source</button></section>
        <details className="panel"><summary>Connection & saved explorations</summary><label>API bearer token<input type="password" value={token} onChange={e => setToken(e.target.value)} autoComplete="off"/></label><label>Saved query ID<input value={savedQueryId} onChange={e => setSavedQueryId(e.target.value)}/></label><button disabled={busy || !savedQueryId.trim()} onClick={() => void load()}>Open exploration</button>{queryId && <p className="query-id">Current: {queryId}</p>}</details>
      </aside>
    </main>
    <footer>Structure gives context. Sources give evidence. <span>Graph Memory / 0.1</span></footer>
  </div>;
}

createRoot(document.getElementById('root')!).render(<React.StrictMode><App/></React.StrictMode>);
