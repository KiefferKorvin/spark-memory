import React, { useEffect, useId, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

// Adapted from stoic_med/frontend/SYNAPSE.html: citLink, citCardShow and refsBlock.
// React escapes source metadata; a fixed portal keeps the hover card outside scroll clips.
function Citation({ evidence, number, inspect }: { evidence: any; number: number; inspect: (id: string) => void }) {
  const [open, setOpen] = useState(false);
  const trigger = useRef<HTMLButtonElement>(null), card = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState({ left: 10, top: 10 });
  const source = evidence.provenance?.sources?.[0] || {};
  const bibliography = source.metadata?.bibliography || {};
  const title = source.label || bibliography.title || 'Source passage';
  useLayoutEffect(() => {
    if (!open || !trigger.current || !card.current) return;
    const r = trigger.current.getBoundingClientRect(), c = card.current.getBoundingClientRect();
    setPosition({ left: Math.max(10, Math.min(r.left + r.width / 2 - c.width / 2, window.innerWidth - c.width - 10)),
      top: Math.max(8, Math.min(r.top - c.height - 8 < 8 ? r.bottom + 8 : r.top - c.height - 8, window.innerHeight - c.height - 8)) });
  }, [open]);
  useEffect(() => {
    const close = () => setOpen(false);
    window.addEventListener('scroll', close, true); window.addEventListener('resize', close);
    return () => { window.removeEventListener('scroll', close, true); window.removeEventListener('resize', close); };
  }, []);
  const tipId = useId();
  return <><button ref={trigger} className="citation-dot" aria-label={`Reference ${number}: ${title}`} aria-describedby={open ? tipId : undefined}
    onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)} onFocus={() => setOpen(true)} onBlur={() => setOpen(false)}
    onKeyDown={e => { if (e.key === 'Escape') setOpen(false); }} onClick={() => inspect(evidence.id)}>●</button>
    {open && createPortal(<div ref={card} id={tipId} role="tooltip" className="citation-card" style={position}>
      <strong>{title}</strong><div>{[source.author || bibliography.author, bibliography.publisher, source.published_at || bibliography.published_at].filter(Boolean).join(' · ')}</div>
      {evidence.provenance?.section_path?.length > 0 && <small>{evidence.provenance.section_path.join(' › ')}</small>}
      <blockquote>{evidence.text.slice(0, 650)}{evidence.text.length > 650 ? '…' : ''}</blockquote>
      <small>Reference {number} · Click the dot to inspect the evidence</small>
    </div>, document.body)}</>;
}

export function FormattedText({ text }: { text: string }) {
  return <div className="formatted"><Markdown remarkPlugins={[remarkGfm]}>{text}</Markdown></div>;
}

export function Answer({ text, evidence, inspect }: { text: string; evidence: any[]; inspect: (id: string) => void }) {
  const byId = new Map(evidence.map(e => [e.id, e]));
  const cited: string[] = [];
  for (const match of text.matchAll(/\[([^\[\]\n]+)\]/g)) if (byId.has(match[1]) && !cited.includes(match[1])) cited.push(match[1]);
  function citations() {
    return (tree: any) => {
      const visit = (node: any) => {
        if (!node.children || ['link', 'code', 'inlineCode'].includes(node.type)) return;
        node.children = node.children.flatMap((child: any) => {
          if (child.type !== 'text') { visit(child); return [child]; }
          const parts: any[] = []; let start = 0;
          for (const match of child.value.matchAll(/\[([^\[\]\n]+)\]/g)) {
            if (!byId.has(match[1])) continue;
            parts.push({ type: 'text', value: child.value.slice(start, match.index) });
            parts.push({ type: 'link', url: '#evidence-' + encodeURIComponent(match[1]), children: [{ type: 'text', value: match[1] }] });
            start = match.index + match[0].length;
          }
          return [...parts, { type: 'text', value: child.value.slice(start) }];
        });
      };
      visit(tree);
    };
  }
  return <div className="formatted answer"><Markdown remarkPlugins={[remarkGfm, citations]} components={{ a: ({ href, children }) => {
    const id = href?.startsWith('#evidence-') ? decodeURIComponent(href.slice(10)) : '';
    return byId.has(id) ? <Citation evidence={byId.get(id)} number={cited.indexOf(id) + 1} inspect={inspect}/> : <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>;
  } }}>{text}</Markdown>
    {cited.length > 0 && <details className="references"><summary>References ({cited.length})</summary><ol>{cited.map(id => {
      const item = byId.get(id), source = item.provenance?.sources?.[0];
      return <li key={id}><button onClick={() => inspect(id)}>{source?.label || 'Source passage'}</button>{source?.uri && /^https?:\/\//.test(source.uri) && <a href={source.uri} target="_blank" rel="noopener noreferrer"> Open source ↗</a>}</li>;
    })}</ol></details>}
  </div>;
}
