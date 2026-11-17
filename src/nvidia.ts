import type { ChatMessage, Env, RetrievedChunk } from './types';

const DEFAULT_BASE_URL = 'https://integrate.api.nvidia.com/v1';
const DEFAULT_CHAT_MODEL = 'nvidia/nemotron-3.5-bolt-e4b';
const CHAT_FALLBACK_MODELS = [
  'nvidia/nemotron-3.5-lightning-30b-a3b',
  'z-ai/glm-5.3-flash',
  'minimaxai/minimax-m3.1-flash',
  'google/gemma-3-12b-it',
];
const DEFAULT_EMBED_MODEL = 'nvidia/embed-qa-4';
const EMBED_FALLBACK_MODELS = [
  'nvidia/nemotron-3.5-embed-1b',
  'nvidia/nemotron-3-embed-1b',
  'nvidia/llama-3.2-nv-embedqa-1b-v1',
];

export class NvidiaApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message);
    this.name = 'NvidiaApiError';
  }
}

function getBaseUrl(env: Pick<Env, 'NVIDIA_BASE_URL'>): string {
  return (env.NVIDIA_BASE_URL || DEFAULT_BASE_URL).replace(/\/+$/, '');
}

function requireApiKey(env: Pick<Env, 'NVIDIA_API_KEY'>): string {
  const key = env.NVIDIA_API_KEY?.trim();
  if (!key || key === 'nvapi-your-free-nvidia-key-here') {
    throw new NvidiaApiError(
      'NVIDIA_API_KEY is not configured. Add your free key from build.nvidia.com.',
      503,
    );
  }
  return key;
}

function usesAsymmetricInputType(model: string): boolean {
  const lower = model.toLowerCase();
  return lower.includes('nv-embed') || lower.includes('e5') || lower.includes('qa') || lower.includes('nemotron');
}

export async function embedTexts(
  texts: string[],
  inputType: 'query' | 'passage',
  env: Pick<Env, 'NVIDIA_API_KEY' | 'NVIDIA_BASE_URL' | 'NVIDIA_EMBED_MODEL'>,
): Promise<number[][]> {
  if (texts.length === 0) return [];
  const apiKey = requireApiKey(env);
  const primary = env.NVIDIA_EMBED_MODEL?.trim() || DEFAULT_EMBED_MODEL;
  const candidates = Array.from(new Set([primary, ...EMBED_FALLBACK_MODELS]));
  const baseUrl = getBaseUrl(env);
  let lastErr: NvidiaApiError | null = null;

  for (const model of candidates) {
    const payload: Record<string, unknown> = {
      model,
      input: texts,
      encoding_format: 'float',
      truncate: 'END',
    };

    if (usesAsymmetricInputType(model)) {
      payload.input_type = inputType;
    }

    const res = await fetch(`${baseUrl}/embeddings`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(15_000),
    });

    if (res.ok) {
      const data = (await res.json()) as {
        data?: Array<{ embedding: number[]; index: number }>;
      };
      if (!data.data || !Array.isArray(data.data)) {
        lastErr = new NvidiaApiError(`Invalid response from NVIDIA Embeddings API [${model}]`, 502);
        continue;
      }
      return data.data
        .sort((a, b) => a.index - b.index)
        .map((item) => {
          // Ensure 1024-dim output for Vectorize (Matryoshka slice + L2 norm if > 1024)
          if (item.embedding.length <= 1024) return item.embedding;
          const sliced = item.embedding.slice(0, 1024);
          const norm = Math.sqrt(sliced.reduce((acc, v) => acc + v * v, 0)) || 1;
          return sliced.map((v) => v / norm);
        });
    }

    const errBody = await res.text().catch(() => '');
    lastErr = new NvidiaApiError(
      `NVIDIA Embeddings API error (${res.status}) [${model}]: ${errBody.slice(0, 200)}`,
      res.status === 401 ? 502 : res.status,
    );
    if (res.status === 401) throw lastErr;
    if (![400, 404, 410, 429, 500, 502, 503, 504].includes(res.status)) {
      throw lastErr;
    }
  }

  throw lastErr ?? new NvidiaApiError('All NVIDIA embedding models failed', 502);
}

export async function rerankChunks(
  query: string,
  chunks: RetrievedChunk[],
  topK: number,
  env: Pick<Env, 'NVIDIA_API_KEY' | 'NVIDIA_RERANK_MODEL'>,
): Promise<{ chunks: RetrievedChunk[]; reranked: boolean }> {
  const model = env.NVIDIA_RERANK_MODEL?.trim();
  if (chunks.length <= 1 || !env.NVIDIA_API_KEY || !model) {
    return { chunks: chunks.slice(0, topK), reranked: false };
  }

  const endpoint = `https://ai.api.nvidia.com/v1/retrieval/${model}/reranking`;

  try {
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${env.NVIDIA_API_KEY.trim()}`,
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify({
        model,
        query: { text: query },
        passages: chunks.map((c) => ({ text: `${c.title}\n${c.text}`.slice(0, 1500) })),
        truncate: 'END',
      }),
      signal: AbortSignal.timeout(3_500),
    });

    if (!res.ok) {
      return { chunks: chunks.slice(0, topK), reranked: false };
    }

    const data = (await res.json()) as {
      rankings?: Array<{ index: number; logit: number }>;
    };

    if (!data.rankings || !Array.isArray(data.rankings)) {
      return { chunks: chunks.slice(0, topK), reranked: false };
    }

    const ranked = data.rankings
      .filter((r) => r.index >= 0 && r.index < chunks.length)
      .sort((a, b) => b.logit - a.logit)
      .slice(0, topK)
      .map((r) => ({
        ...chunks[r.index],
        score: Number((1 / (1 + Math.exp(-r.logit))).toFixed(4)),
      }));

    return ranked.length > 0
      ? { chunks: ranked, reranked: true }
      : { chunks: chunks.slice(0, topK), reranked: false };
  } catch {
    return { chunks: chunks.slice(0, topK), reranked: false };
  }
}

export async function rewriteQuery(
  question: string,
  history: ChatMessage[],
  env: Pick<Env, 'NVIDIA_API_KEY' | 'NVIDIA_BASE_URL' | 'NVIDIA_CHAT_MODEL'>,
): Promise<string> {
  const hasArabic = /[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]/.test(question);
  if ((history.length === 0 && !hasArabic) || !env.NVIDIA_API_KEY) return question;

  const recent = history.length
    ? `Conversation:\n${history
        .slice(-4)
        .map((m) => `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.content.slice(0, 250)}`)
        .join('\n')}\n\n`
    : '';

  try {
    const res = await fetch(`${getBaseUrl(env)}/chat/completions`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${env.NVIDIA_API_KEY.trim()}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: env.NVIDIA_CHAT_MODEL || DEFAULT_CHAT_MODEL,
        messages: [
          {
            role: 'system',
            content:
              'Translate and rewrite the user question into a single concise, self-contained English search query for E-JUST (Egypt-Japan University of Science and Technology) knowledge retrieval. Return ONLY the English search query, nothing else.',
          },
          {
            role: 'user',
            content: `${recent}Question: ${question}`,
          },
        ],
        temperature: 0,
        max_tokens: 100,
        stream: false,
      }),
      signal: AbortSignal.timeout(6_000),
    });

    if (!res.ok) return question;
    const data = (await res.json()) as {
      choices?: Array<{ message?: { content?: string } }>;
    };
    const rewritten = data.choices?.[0]?.message?.content?.trim();
    return rewritten && rewritten.length > 3 ? rewritten : question;
  } catch {
    return question;
  }
}

function getCandidateChatModels(env: Pick<Env, 'NVIDIA_CHAT_MODEL'>): string[] {
  const primary = env.NVIDIA_CHAT_MODEL?.trim() || DEFAULT_CHAT_MODEL;
  return Array.from(new Set([primary, ...CHAT_FALLBACK_MODELS]));
}

export async function generateChatCompletion(
  messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>,
  env: Pick<Env, 'NVIDIA_API_KEY' | 'NVIDIA_BASE_URL' | 'NVIDIA_CHAT_MODEL'>,
): Promise<{ content: string; model: string }> {
  const apiKey = requireApiKey(env);
  const baseUrl = getBaseUrl(env);
  const candidates = getCandidateChatModels(env);
  let lastErr: NvidiaApiError | null = null;

  for (const model of candidates) {
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(new Error(`Timeout after 12s [${model}]`)), 12_000);
    let res: Response;
    try {
      res = await fetch(`${baseUrl}/chat/completions`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model,
          messages,
          temperature: 0.25,
          top_p: 0.9,
          max_tokens: 2048,
          stream: false,
        }),
        signal: ac.signal,
      });
    } catch (err) {
      clearTimeout(timer);
      const msg = err instanceof Error ? err.message : 'Request timed out';
      lastErr = new NvidiaApiError(`NVIDIA Chat API error [${model}]: ${msg}`, 504);
      continue;
    }
    clearTimeout(timer);

    if (res.ok) {
      const data = (await res.json()) as {
        choices?: Array<{ message?: { content?: string } }>;
      };
      const content = data.choices?.[0]?.message?.content?.trim();
      if (content) return { content, model };
      lastErr = new NvidiaApiError(`Empty completion from NVIDIA Chat API [${model}]`, 502);
      continue;
    }

    const errBody = await res.text().catch(() => '');
    lastErr = new NvidiaApiError(
      `NVIDIA Chat API error (${res.status}) [${model}]: ${errBody.slice(0, 240)}`,
      res.status === 401 ? 502 : res.status,
    );

    if (res.status === 401) throw lastErr;
    // Retry next fallback model on 404/410 (retired), 429 (rate limit), or 5xx
    if (![404, 410, 429, 500, 502, 503, 504].includes(res.status)) {
      throw lastErr;
    }
  }

  throw lastErr ?? new NvidiaApiError('All NVIDIA chat models failed', 502);
}

export async function streamChatCompletion(
  messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>,
  env: Pick<Env, 'NVIDIA_API_KEY' | 'NVIDIA_BASE_URL' | 'NVIDIA_CHAT_MODEL'>,
): Promise<{ stream: ReadableStream<string>; model: string }> {
  const apiKey = requireApiKey(env);
  const baseUrl = getBaseUrl(env);
  const candidates = getCandidateChatModels(env);
  let res: Response | null = null;
  let chosenModel = candidates[0];
  let lastErr: NvidiaApiError | null = null;

  for (const model of candidates) {
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(new Error(`Timeout after 12s [${model}]`)), 12_000);
    let attempt: Response;
    try {
      attempt = await fetch(`${baseUrl}/chat/completions`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model,
          messages,
          temperature: 0.25,
          top_p: 0.9,
          max_tokens: 2048,
          stream: true,
        }),
        signal: ac.signal,
      });
    } catch (err) {
      clearTimeout(timer);
      const msg = err instanceof Error ? err.message : 'Request timed out';
      lastErr = new NvidiaApiError(`NVIDIA Chat Stream error [${model}]: ${msg}`, 504);
      continue;
    }
    clearTimeout(timer);

    if (attempt.ok && attempt.body) {
      res = attempt;
      chosenModel = model;
      break;
    }

    const errBody = await attempt.text().catch(() => '');
    lastErr = new NvidiaApiError(
      `NVIDIA Chat Stream error (${attempt.status}) [${model}]: ${errBody.slice(0, 240)}`,
      attempt.status === 401 ? 502 : attempt.status,
    );
    if (attempt.status === 401) throw lastErr;
    if (![404, 410, 429, 500, 502, 503, 504].includes(attempt.status)) {
      throw lastErr;
    }
  }

  if (!res || !res.body) {
    throw lastErr ?? new NvidiaApiError('All NVIDIA chat stream models failed', 502);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();

  const tokenStream = new ReadableStream<string>({
    async start(controller) {
      let buffer = '';
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() ?? '';

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed.startsWith('data:')) continue;
            const payload = trimmed.slice(5).trim();
            if (payload === '[DONE]') continue;
            try {
              const parsed = JSON.parse(payload) as {
                choices?: Array<{ delta?: { content?: string } }>;
              };
              const token = parsed.choices?.[0]?.delta?.content;
              if (token) controller.enqueue(token);
            } catch {
              // Ignore malformed partial SSE frames
            }
          }
        }
      } catch (err) {
        controller.error(err);
        return;
      }
      controller.close();
    },
    cancel() {
      reader.cancel().catch(() => {});
    },
  });

  return { stream: tokenStream, model: chosenModel };
}

export function formatHistoryForModel(
  history: ChatMessage[],
  maxTurns = 4,
): Array<{ role: 'user' | 'assistant'; content: string }> {
  return history.slice(-maxTurns * 2).map((m) => ({
    role: m.role,
    content: m.content,
  }));
}

export async function generateChatTitle(
  messages: Array<{ role: 'user' | 'assistant'; content: string }>,
  env: Pick<Env, 'NVIDIA_API_KEY' | 'NVIDIA_BASE_URL' | 'NVIDIA_CHAT_MODEL'>,
): Promise<string> {
  const snippet = messages
    .slice(-10)
    .map((m) => `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.content.slice(0, 220)}`)
    .join('\n');

  const fallback =
    messages.find((m) => m.role === 'user')?.content.trim().slice(0, 48) || 'New Chat';

  if (!env.NVIDIA_API_KEY) return fallback;

  try {
    const { content } = await generateChatCompletion(
      [
        {
          role: 'system',
          content:
            'Summarize the conversation into a concise, descriptive 3 to 6 word chat title. Return ONLY the title text — no quotes, no markdown, no trailing punctuation.',
        },
        {
          role: 'user',
          content: snippet,
        },
      ],
      env,
    );
    const cleaned = content.replace(/^["'`*#\s]+|["'`*#.\s]+$/g, '').slice(0, 64);
    return cleaned || fallback;
  } catch {
    return fallback;
  }
}
