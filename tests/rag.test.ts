import { describe, expect, it, vi, afterEach } from 'vitest';
import worker from '../src/index';
import {
  buildSystemPrompt,
  chunkMarkdown,
  cosineSimilarity,
  dedupeSources,
  keywordSearch,
  parseFrontmatter,
  reciprocalRankFusion,
} from '../src/rag';
import type { Env, KnowledgeChunk } from '../src/types';

function createTestEnv(overrides: Partial<Env> = {}): Env {
  return {
    ASSETS: {
      fetch: async () => new Response('Not found', { status: 404 }),
    } as unknown as Fetcher,
    NVIDIA_BASE_URL: 'https://integrate.api.nvidia.com/v1',
    NVIDIA_CHAT_MODEL: 'meta/llama-3.3-70b-instruct',
    NVIDIA_EMBED_MODEL: 'nvidia/nv-embedqa-e5-v5',
    ...overrides,
  };
}

describe('RAG utilities', () => {
  it('parses frontmatter metadata and body', () => {
    const raw = `---\ntitle: Test Doc\nurl: https://example.com\n---\n\n## Section\nHello world.`;
    const { meta, body } = parseFrontmatter(raw);
    expect(meta.title).toBe('Test Doc');
    expect(meta.url).toBe('https://example.com');
    expect(body).toBe('## Section\nHello world.');
  });

  it('chunks markdown by section and respects maxChars', () => {
    const md = `## Part One\nShort text.\n\n## Part Two\n${'word '.repeat(200)}`;
    const chunks = chunkMarkdown(md, 'doc1', 'Doc 1', 'https://example.com', 300, 50);
    expect(chunks.length).toBeGreaterThan(2);
    expect(chunks[0].id).toBe('doc1#0');
    expect(chunks[0].title).toBe('Doc 1');
  });

  it('computes cosine similarity accurately', () => {
    expect(cosineSimilarity([1, 0], [1, 0])).toBeCloseTo(1);
    expect(cosineSimilarity([1, 0], [0, 1])).toBeCloseTo(0);
    expect(cosineSimilarity([1, 1], [1, 1])).toBeCloseTo(1);
  });

  it('retrieves relevant chunks via keywordSearch fallback', () => {
    const chunks: KnowledgeChunk[] = [
      { id: '1', title: 'Admissions', url: 'https://ejust.edu.eg/admissions', text: 'Admission requires Thanaweya Amma or STEM.' },
      { id: '2', title: 'Campus', url: 'https://ejust.edu.eg/campus', text: 'Located in New Borg El Arab City.' },
    ];
    const results = keywordSearch('Where is E-JUST located?', chunks, 3);
    expect(results[0].id).toBe('2');
  });

  it('dedupes sources by URL keeping highest score', () => {
    const sources = dedupeSources([
      { id: 'a', title: 'A', url: 'https://ejust.edu.eg/a', text: 't1', score: 0.4 },
      { id: 'b', title: 'A', url: 'https://ejust.edu.eg/a', text: 't2', score: 0.8 },
    ]);
    expect(sources).toHaveLength(1);
    expect(sources[0].score).toBe(0.8);
  });

  it('merges ranked lists using Reciprocal Rank Fusion', () => {
    const dense = [
      { id: 'a', title: 'A', url: 'u1', text: 't1', score: 0.9 },
      { id: 'b', title: 'B', url: 'u2', text: 't2', score: 0.7 },
    ];
    const lexical = [
      { id: 'b', title: 'B', url: 'u2', text: 't2', score: 5.0 },
      { id: 'c', title: 'C', url: 'u3', text: 't3', score: 3.0 },
    ];
    const fused = reciprocalRankFusion([dense, lexical]);
    expect(fused[0].id).toBe('b');
    expect(fused.map((x) => x.id)).toEqual(['b', 'a', 'c']);
  });

  it('builds bilingual system prompt with context and suggestion format', () => {
    const promptAr = buildSystemPrompt([], 'ar');
    expect(promptAr).toContain('ممكن كمان تسأل عن');
    const promptEn = buildSystemPrompt(
      [{ id: '1', title: 'About', url: 'https://ejust.edu.eg', text: 'E-JUST is in Alexandria.', score: 0.9 }],
      'en',
    );
    expect(promptEn).toContain('E-JUST is in Alexandria.');
    expect(promptEn).toContain('You might also want to ask:');
  });
});

describe('Worker HTTP routes', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('returns health status', async () => {
    const env = createTestEnv();
    const res = await worker.fetch(new Request('http://localhost/health'), env);
    expect(res.status).toBe(200);
    const data = (await res.json()) as Record<string, unknown>;
    expect(data.status).toBe('ok');
    expect(data.provider).toBe('nvidia-nim');
    expect(data.api_key_configured).toBe(false);
  });

  it('validates empty question on POST /ask', async () => {
    const env = createTestEnv({ NVIDIA_API_KEY: 'nvapi-test-key' });
    const res = await worker.fetch(
      new Request('http://localhost/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: '   ' }),
      }),
      env,
    );
    expect(res.status).toBe(400);
  });

  it('answers question using retrieved context and stores history', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({
          choices: [
            {
              message: {
                content:
                  'E-JUST is located in New Borg El Arab City, Alexandria.\n\n---\n**You might also want to ask:**\n1. What programs does E-JUST offer?',
              },
            },
          ],
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      ),
    );

    const env = createTestEnv({ NVIDIA_API_KEY: 'nvapi-test-key' });
    const sessionId = 'test-session-1';

    const askRes = await worker.fetch(
      new Request('http://localhost/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: 'Where is E-JUST located?', session_id: sessionId }),
      }),
      env,
    );

    expect(askRes.status).toBe(200);
    const askData = (await askRes.json()) as {
      answer: string;
      sources: Array<{ title: string }>;
      retrieval_mode: string;
    };
    expect(askData.answer).toContain('New Borg El Arab');
    expect(askData.sources.length).toBeGreaterThan(0);
    expect(askData.retrieval_mode).toBe('keyword-fallback');
    expect(fetchMock).toHaveBeenCalledTimes(1);

    const histRes = await worker.fetch(
      new Request(`http://localhost/history?session_id=${sessionId}`),
      env,
    );
    const histData = (await histRes.json()) as { history: Array<{ role: string }> };
    expect(histData.history).toHaveLength(2);

    const clearRes = await worker.fetch(
      new Request('http://localhost/clear', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId }),
      }),
      env,
    );
    expect(clearRes.status).toBe(200);

    const histAfter = await worker.fetch(
      new Request(`http://localhost/history?session_id=${sessionId}`),
      env,
    );
    const histAfterData = (await histAfter.json()) as { history: unknown[] };
    expect(histAfterData.history).toHaveLength(0);
  });

  it('streams SSE tokens on POST /ask with stream=true and handles /feedback', async () => {
    const ssePayload = [
      'data: {"choices":[{"delta":{"content":"Hello "}}]}\n\n',
      'data: {"choices":[{"delta":{"content":"E-JUST!"}}]}\n\n',
      'data: [DONE]\n\n',
    ].join('');

    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(ssePayload, {
        status: 200,
        headers: { 'Content-Type': 'text/event-stream' },
      }),
    );

    const env = createTestEnv({ NVIDIA_API_KEY: 'nvapi-test-key' });
    const res = await worker.fetch(
      new Request('http://localhost/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: 'Tell me about E-JUST', session_id: 'sse-1', stream: true }),
      }),
      env,
    );

    expect(res.status).toBe(200);
    expect(res.headers.get('Content-Type')).toContain('text/event-stream');
    const bodyText = await res.text();
    expect(bodyText).toContain('"token":"Hello "');
    expect(bodyText).toContain('"token":"E-JUST!"');
    expect(bodyText).toContain('"done":true');

    const fbRes = await worker.fetch(
      new Request('http://localhost/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: 'sse-1', question: 'Hi', answer: 'Hello', rating: 1 }),
      }),
      env,
    );
    expect(fbRes.status).toBe(200);
  });
});
