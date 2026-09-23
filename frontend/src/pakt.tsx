import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { InteractiveNvlWrapper } from '@neo4j-nvl/react';
import { FormattedText } from './Answer';
import './pakt.css';

type GraphNode = { id: string; label: string; kind: string; text?: string; summary?: string; uri?: string; metadata?: any; provenance?: any };
type Edge = { id: string; source: string; target: string; relation: string };
const palette: Record<string, [string, string]> = {
  TaxonomyGroup: ['#b48648', 'Domaine'], Concept: ['#738bc2', 'Concept'], Document: ['#389487', 'Document'],
  Section: ['#a086bc', 'Section'], Chunk: ['#bc9653', 'Passage'], Assertion: ['#bd7169', 'Assertion'],
  Source: ['#788995', 'Source'], ExternalConcept: ['#93a0bc', 'Concept externe'],
};
const host = document.getElementById('atlas-explorer')!;
const seeds: string[] = JSON.parse(host.dataset.seeds || '[]');
const MAX_NODES = 240;

async function get(path: string) {
  const response = await fetch('/api/atlas/' + path, { headers: { Accept: 'application/json' } });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || 'Exploration indisponible. Recharge la page puis réessaie.');
  return body;
}
function safeUrl(value?: string) {
  try { const url = new URL(value || ''); return ['http:', 'https:'].includes(url.protocol) && !url.username && !url.password ? url.href : null; }
  catch { return null; }
}

function Explorer() {
  const [graph, setGraph] = useState<{ nodes: Record<string, GraphNode>; edges: Record<string, Edge> }>({ nodes: {}, edges: {} });
  const [selected, setSelected] = useState<GraphNode | null>(null);
  const [busy, setBusy] = useState(false), [error, setError] = useState(''), [notice, setNotice] = useState('');
  const [filter, setFilter] = useState(''), [view, setView] = useState(0);
  const requestId = useRef(0);
  const nodes = Object.values(graph.nodes), edges = Object.values(graph.edges);
  const matches = nodes.filter(n => n.label.toLocaleLowerCase().includes(filter.toLocaleLowerCase()));

  function merge(incoming: GraphNode[], links: Edge[]) {
    setGraph(current => {
      const next = { ...current.nodes };
      for (const node of incoming) if (next[node.id] || Object.keys(next).length < MAX_NODES) next[node.id] = node;
      const rels = { ...current.edges };
      for (const edge of links) if (next[edge.source] && next[edge.target]) rels[edge.id] = edge;
      return { nodes: next, edges: rels };
    });
  }

  async function load() {
    const ticket = ++requestId.current;
    setBusy(true); setError(''); setSelected(null); setFilter(''); setGraph({ nodes: {}, edges: {} });
    try {
      const data = await get('taxonomy');
      if (ticket !== requestId.current) return;
      merge(data.roots || [], []);
      setNotice(data.roots?.length ? 'Choisis un domaine, puis explore ses connexions.' : 'Aucun domaine disponible. Pose une question ci-dessous pour explorer ses sources.');
      if (seeds.length) {
        const evidence = await Promise.all([...new Set(seeds)].slice(0, 12).map(id => get('nodes/' + encodeURIComponent(id))));
        if (ticket !== requestId.current) return;
        merge(evidence, []);
        const links = await Promise.all(evidence.slice(0, 3).map(node => get('nodes/' + node.id + '/neighbors')));
        if (ticket !== requestId.current) return;
        links.forEach(data => merge(data.nodes, data.edges));
        setNotice('Les sources de ta réponse sont ajoutées au graphe. Sélectionne-les pour suivre leurs liens.');
      }
      setView(v => v + 1);
    } catch (e) { if (ticket === requestId.current) setError((e as Error).message); }
    finally { if (ticket === requestId.current) setBusy(false); }
  }

  useEffect(() => { void load(); return () => { requestId.current++; }; }, []);

  async function inspect(id: string) {
    if (busy) return;
    const ticket = ++requestId.current;
    setSelected(graph.nodes[id]); setBusy(true); setError('');
    try {
      const detail = await get('nodes/' + encodeURIComponent(id));
      if (ticket === requestId.current) setSelected(detail);
    } catch (e) { if (ticket === requestId.current) setError((e as Error).message); }
    finally { if (ticket === requestId.current) setBusy(false); }
  }

  async function expand() {
    if (!selected || busy) return;
    setBusy(true); setError('');
    try {
      const data = await get('nodes/' + encodeURIComponent(selected.id) + '/neighbors');
      merge(data.nodes, data.edges);
      setNotice(data.edges.length ? `${data.edges.length} connexions chargées. Les voisinages sont limités à 60 liens par ouverture.` : 'Ce nœud n’a pas encore de connexion.');
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  }

  function focus() {
    if (!selected) return;
    setGraph(current => {
      const links = Object.values(current.edges).filter(e => e.source === selected.id || e.target === selected.id);
      const ids = new Set([selected.id, ...links.flatMap(e => [e.source, e.target])]);
      return { nodes: Object.fromEntries(Object.entries(current.nodes).filter(([id]) => ids.has(id))), edges: Object.fromEntries(links.map(e => [e.id, e])) };
    });
    setFilter(''); setView(v => v + 1);
  }

  const sources = selected?.provenance?.sources || [];
  return <div className="pakt-graph">
    <div className="ag-heading"><div><span className="ag-eyebrow">LA MÉMOIRE, EN CONNEXIONS</span><h2>Explorer le graphe</h2></div><span className="ag-count">{nodes.length} nœuds · {edges.length} liens</span></div>
    <p className="ag-intro">Des domaines aux concepts, des documents à leurs sources. Déplace les nœuds, zoome et suis les connexions.</p>
    <div className="ag-toolbar"><button type="button" onClick={() => void load()} disabled={busy}>Revenir aux domaines</button><button type="button" onClick={() => setView(v => v + 1)} disabled={!nodes.length}>Recentrer la vue</button><span role="status">{busy ? 'Chargement…' : notice}</span></div>
    {error && <p className="ag-error" role="alert">{error} <button type="button" disabled={busy} onClick={() => void load()}>Réessayer</button></p>}
    {nodes.length >= MAX_NODES && <p role="status">Vue limitée à {MAX_NODES} nœuds. Isole un voisinage pour poursuivre l’exploration.</p>}
    <div className="ag-layout">
      <div className="ag-canvas" aria-label="Graphe interactif de la mémoire">
        {nodes.length ? <InteractiveNvlWrapper key={view}
          nodes={nodes.map(n => ({ id: n.id, caption: n.label, captionAlign: 'bottom' as const, captionSize: 2, color: palette[n.kind]?.[0] || '#788995', size: n.kind === 'TaxonomyGroup' ? 30 : n.kind === 'Concept' ? 24 : 18, selected: n.id === selected?.id }))}
          rels={edges.map(e => ({ id: e.id, from: e.source, to: e.target, caption: e.relation, color: '#728c91' }))}
          nvlOptions={{ layout: 'd3Force', renderer: 'canvas', initialZoom: 1, disableTelemetry: true }}
          mouseEventCallbacks={{ onNodeClick: n => { void inspect(n.id); }, onPan: true, onZoom: true, onDrag: true }} />
          : <div className="ag-empty"><span aria-hidden="true">◈</span><h3>{busy ? 'Ouverture de la mémoire…' : 'Le graphe commence par une source.'}</h3><p>Les domaines et les sources des réponses apparaîtront ici.</p></div>}
      </div>
      <aside className="ag-inspector" aria-label="Détails du nœud">
        <label htmlFor="ag-filter">Trouver un nœud affiché<input id="ag-filter" type="search" value={filter} onChange={e => setFilter(e.target.value)} placeholder="Concept, document, source…"/></label>
        <label htmlFor="ag-node">Sélectionner un nœud<select id="ag-node" value={matches.some(n => n.id === selected?.id) ? selected!.id : ''} disabled={busy} onChange={e => e.target.value && void inspect(e.target.value)}><option value="">{matches.length} nœuds disponibles</option>{matches.map(n => <option value={n.id} key={n.id}>{n.label} · {palette[n.kind]?.[1] || n.kind}</option>)}</select></label>
        {selected ? <><span className="ag-kind">{palette[selected.kind]?.[1] || selected.kind}</span><h3>{selected.label}</h3>
          <div className="ag-actions"><button type="button" disabled={busy} onClick={() => void expand()}>Explorer les connexions</button><button type="button" disabled={busy} onClick={focus}>Isoler ce voisinage</button></div>
          <div className="ag-content"><FormattedText text={selected.metadata?.formatted_content || selected.text || selected.summary || 'Sélectionne les connexions pour découvrir les connaissances liées.'}/></div>
          {safeUrl(selected.uri) && <a href={safeUrl(selected.uri)!} target="_blank" rel="noopener noreferrer">Ouvrir la source ↗</a>}
          {!!sources.length && <><h4>Provenance</h4>{sources.map((s: any) => <p key={s.id}>{safeUrl(s.uri) ? <a href={safeUrl(s.uri)!} target="_blank" rel="noopener noreferrer">{s.label || s.uri} ↗</a> : s.label}<small>{s.author || ''}{s.source_type === 'generated' ? ' · Contenu généré' : ''}</small></p>)}</>}
          <details><summary>Identifiant du nœud</summary><code>{selected.id}</code></details>
        </> : <div className="ag-hint"><h3>Chaque lien ouvre une piste.</h3><p>Clique sur un nœud ou sélectionne-le dans la liste pour consulter son contenu et sa provenance.</p></div>}
      </aside>
    </div>
    <div className="ag-legend" aria-label="Légende">{Object.entries(palette).filter(([kind]) => nodes.some(n => n.kind === kind)).map(([kind, [color, label]]) => <span key={kind}><i style={{ background: color }}/>{label}</span>)}</div>
    <p className="ag-footnote">Vue progressive de l’Atlas partagé. Explorer le graphe ne lance aucune recherche web ni génération IA.</p>
  </div>;
}

createRoot(host).render(<Explorer/>);
