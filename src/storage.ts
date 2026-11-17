import type { ChatMessage, Env, SourceRef } from './types';

const MAX_HISTORY_MESSAGES = 12;
const HISTORY_TTL_SECONDS = 60 * 60 * 24 * 7; // 7 days

// In-memory fallback for local dev / unit tests when D1/KV are not bound
const memoryStore = new Map<string, ChatMessage[]>();

function historyKey(sessionId: string): string {
  return `session:${sessionId}`;
}

export async function getHistory(
  env: Pick<Env, 'DB' | 'CHAT_HISTORY'>,
  sessionId: string,
): Promise<ChatMessage[]> {
  if (!sessionId) return [];

  if (env.DB) {
    try {
      const result = await env.DB.prepare(
        'SELECT role, content, created_at as timestamp FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?',
      )
        .bind(sessionId, MAX_HISTORY_MESSAGES)
        .all<{ role: 'user' | 'assistant'; content: string; timestamp: string }>();
      return (result.results || []).reverse();
    } catch {
      // Fall through if table not migrated yet
    }
  }

  if (env.CHAT_HISTORY) {
    const raw = await env.CHAT_HISTORY.get(historyKey(sessionId));
    if (!raw) return [];
    try {
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }

  return memoryStore.get(sessionId) ?? [];
}

export async function appendHistory(
  env: Pick<Env, 'DB' | 'CHAT_HISTORY'>,
  sessionId: string,
  userContent: string,
  assistantContent: string,
  sources: SourceRef[] = [],
): Promise<void> {
  if (!sessionId) return;

  if (env.DB) {
    try {
      const sourcesJson = JSON.stringify(sources);
      await env.DB.batch([
        env.DB.prepare(
          'INSERT INTO messages (session_id, role, content, sources_json) VALUES (?, ?, ?, ?)',
        ).bind(sessionId, 'user', userContent, null),
        env.DB.prepare(
          'INSERT INTO messages (session_id, role, content, sources_json) VALUES (?, ?, ?, ?)',
        ).bind(sessionId, 'assistant', assistantContent, sourcesJson),
      ]);
      return;
    } catch {
      // Fall through to KV / memory
    }
  }

  const existing = await getHistory(env, sessionId);
  const now = new Date().toISOString();
  const updated = [
    ...existing,
    { role: 'user' as const, content: userContent, timestamp: now },
    { role: 'assistant' as const, content: assistantContent, timestamp: now },
  ].slice(-MAX_HISTORY_MESSAGES);

  if (env.CHAT_HISTORY) {
    await env.CHAT_HISTORY.put(historyKey(sessionId), JSON.stringify(updated), {
      expirationTtl: HISTORY_TTL_SECONDS,
    });
  } else {
    memoryStore.set(sessionId, updated);
  }
}

export async function clearHistory(
  env: Pick<Env, 'DB' | 'CHAT_HISTORY'>,
  sessionId: string,
): Promise<void> {
  if (!sessionId) return;

  if (env.DB) {
    try {
      await env.DB.prepare('DELETE FROM messages WHERE session_id = ?').bind(sessionId).run();
    } catch {
      // ignore
    }
  }

  if (env.CHAT_HISTORY) {
    await env.CHAT_HISTORY.delete(historyKey(sessionId));
  }
  memoryStore.delete(sessionId);
}

export async function saveFeedback(
  env: Pick<Env, 'DB'>,
  sessionId: string,
  question: string,
  answer: string,
  rating: number,
): Promise<boolean> {
  if (!env.DB) return false;
  try {
    await env.DB.prepare(
      'INSERT INTO feedback (session_id, question, answer, rating) VALUES (?, ?, ?, ?)',
    )
      .bind(sessionId || 'anonymous', question, answer.slice(0, 2000), rating)
      .run();
    return true;
  } catch {
    return false;
  }
}
