export type NodeState = 'UNVISITED' | 'CANDIDATE' | 'ACTIVE' | 'EXPANDED' | 'SELECTED' | 'EVIDENCE' | 'PRUNED' | 'DEAD_END';
export type GraphNode = { id: string; label: string; kind: string; origin?: string; uri?: string; state: NodeState; routing_summary?: string; summary?: string; branch?: string; need?: string };
export type GraphEdge = { id: string; from: string; to: string; active?: boolean; branch?: string; caption?: string };
export type Trace = { event_id: string; query_id: string; sequence: number; timestamp: string; event_type: string; node_id?: string; branch_id?: string; information_need_id?: string; metadata: Record<string, any> };
export type GraphState = { nodes: Record<string, GraphNode>; edges: Record<string, GraphEdge> };

const explored = (state?: NodeState) => state === 'EXPANDED' || state === 'SELECTED' || state === 'EVIDENCE';
export const needId = (id?: string) => 'need:' + id;

export function reduceEvents(events: Trace[]): GraphState {
  const graph: GraphState = { nodes: {}, edges: {} };
  const upsert = (raw: any, state: NodeState) => {
    const id = raw.node_id || raw.id;
    if (!id) return;
    const current = graph.nodes[id]?.state;
    // A node explored for one need stays explored when it reappears as another need's candidate.
    graph.nodes[id] = { ...graph.nodes[id], ...raw, id, state: current === 'EVIDENCE' || (explored(current) && !explored(state)) ? current! : state };
  };
  const link = (from: string | undefined, to: string | undefined, extra: Partial<GraphEdge>) => {
    if (!from || !to || from === to) return;
    const id = `${from}:${to}`;
    graph.edges[id] = { ...graph.edges[id], id, from, to, ...extra };
  };
  for (const event of events) {
    const meta = event.metadata;
    const origin = event.node_id || needId(event.information_need_id);
    if (event.event_type === 'QUERY_DECOMPOSED') for (const need of meta.information_needs || []) upsert({ id: needId(need.id), label: need.description, kind: 'Need' }, 'ACTIVE');
    if (event.event_type === 'CANDIDATES_GENERATED') for (const node of meta.nodes || []) {
      upsert(node, 'CANDIDATE');
      if (!graph.edges[`${origin}:${node.node_id}`]) link(origin, node.node_id, { caption: 'candidate' });
    }
    if (meta.node) upsert(meta.node, event.event_type === 'NODE_SELECTED' ? 'SELECTED' : 'EXPANDED');
    if (event.event_type === 'BRANCH_SPAWNED' && event.node_id) {
      // The route edge follows the branch path; first hops start at their information need.
      const path: string[] = meta.path || [];
      link(path.length > 1 ? path[path.length - 2] : needId(event.information_need_id), event.node_id, { branch: event.branch_id, caption: 'route' });
    }
    if (event.node_id && graph.nodes[event.node_id]) {
      const node = graph.nodes[event.node_id];
      const state: Record<string, NodeState> = { BRANCH_SPAWNED: 'ACTIVE', NODE_PRUNED: 'PRUNED', EVIDENCE_FOUND: 'EVIDENCE', BACKTRACK: 'DEAD_END' };
      if (state[event.event_type] && node.state !== 'EVIDENCE' && !(event.event_type === 'NODE_PRUNED' && explored(node.state))) node.state = state[event.event_type];
      if (['BRANCH_SPAWNED', 'NODE_EXPANDED', 'NODE_SELECTED', 'EVIDENCE_FOUND'].includes(event.event_type)) {
        node.branch = event.branch_id; node.need = event.information_need_id;
      }
    }
    if (event.event_type === 'EDGE_TRAVERSED') {
      for (const edge of Object.values(graph.edges)) edge.active = false;
      link(meta.source, meta.target, { active: true, branch: event.branch_id, caption: 'route' });
    }
    if (event.event_type === 'EVIDENCE_FOUND') {
      const e = meta.evidence;
      upsert({ id: e.id, label: 'Evidence', kind: 'Evidence', summary: e.text, need: event.information_need_id }, 'EVIDENCE');
      link(e.source_node_id, e.id, { caption: 'supports' });
    }
    if (event.event_type === 'EXTERNAL_SOURCE_FOUND') {
      upsert({ id: meta.url, label: meta.title, kind: 'Source', summary: meta.url, origin: meta.retriever }, 'ACTIVE');
      link(needId(meta.need_id), meta.url, { caption: 'searched' });
    }
    if (event.event_type === 'EXTERNAL_SOURCE_REJECTED' && graph.nodes[meta.url]) graph.nodes[meta.url].state = 'PRUNED';
    if (event.event_type === 'SOURCE_INGESTED') {
      upsert({ id: meta.document_id, label: meta.title || 'Ingested source', kind: 'Document' }, 'ACTIVE');
      link(meta.url, meta.document_id, { caption: 'ingested' });
    }
  }
  return graph;
}

export function branchColor(branch?: string): string {
  const colors = ['#a9b7ff', '#ffba85', '#6ed9c5', '#e7a3dc', '#e6d779', '#9bbcf1'];
  return colors[Array.from(branch || '').reduce((sum, c) => sum + c.charCodeAt(0), 0) % colors.length];
}
