import type { Trace } from './state';

export async function api(path: string, token: string, init: RequestInit = {}) {
  const response = await fetch('/memory' + path, { ...init, headers: { 'Content-Type': 'application/json', ...(token ? { Authorization: `Bearer ${token}` } : {}), ...init.headers } });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === 'string' ? body.detail : `Request failed (${response.status})`);
  }
  return response.json();
}

export async function stream(queryId: string, token: string, after: number, signal: AbortSignal, onEvent: (event: Trace) => void) {
  const response = await fetch(`/memory/query/${queryId}/stream?after=${after}`, { signal, headers: token ? { Authorization: `Bearer ${token}` } : {} });
  if (!response.ok || !response.body) throw new Error('Could not open exploration stream');
  const reader = response.body.getReader(), decoder = new TextDecoder();
  let buffer = '';
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let boundary;
      while ((boundary = buffer.indexOf('\n\n')) !== -1) {
        const block = buffer.slice(0, boundary); buffer = buffer.slice(boundary + 2);
        if (block.startsWith('event: done')) return;
        const data = block.split('\n').filter(line => line.startsWith('data:')).map(line => line.slice(5).trim()).join('\n');
        if (data) onEvent(JSON.parse(data));
      }
    }
  } finally { reader.releaseLock(); }
}

