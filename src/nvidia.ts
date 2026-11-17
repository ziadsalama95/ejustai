import type { ChatMessage, Env, RetrievedChunk } from './types';

const DEFAULT_BASE_URL = 'https://integrate.api.nvidia.com/v1';
const DEFAULT_CHAT_MODEL = 'meta/llama-3.3-70b-instruct';
const DEFAULT_EMBED_MODEL = 'nvidia/nv-embedqa-e5-v5';
const DEFAULT_RERANK_MODEL = 'nvidia/nv-rerankqa-mistral-4b-v3';

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
  return lower.includes('nv-embed') || lower.includes('e5') || lower.includes('qa');
}

export async function embedTexts(
  texts: string[],
  inputType: 'query' | 'passage',
  env: Pick<Env, 'NVIDIA_API_KEY' | 'NVIDIA_BASE_URL' | 'NVIDIA_EMBED_MODEL'>,
): Promise<number[][]> {
  if (texts.length === 0) return [];
  const apiKey = requireApiKey(env);
  const model = env.NVIDIA_EMBED_MODEL || DEFAULT_EMBED_MODEL;
  const baseUrl = getBaseUrl(env);

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
    signal: AbortSignal.timeout(20_000),
  });

  if (!res.ok) {
    const errBody = await res.text().catch(() => '');
    throw new NvidiaApiError(
      `NVIDIA Embeddings API error (${res.status}): ${errBody.slice(0, 200)}`,
      res.status === 401 ? 502 : res.status,
    );
  }

  const data = (await res.json()) as {
    data?: Array<{ embedding: number[]; index: number }>;
  };

  if (!data.data || !Array.isArray(data.data)) {
    throw new NvidiaApiError('Invalid response from NVIDIA Embeddings API', 502);
  }

  return data.data
    .sort((a, b) => a.index - b.index)
    .map((item) => item.embedding);
}

export async function rerankChunks(
  query: string,
  chunks: RetrievedChunk[],
  topK: number,
  env: Pick<Env, 'NVIDIA_API_KEY' | 'NVIDIA_RERANK_MODEL'>,
): Promise<{ chunks: RetrievedChunk[]; reranked: boolean }> {
  if (chunks.length <= 1 || !env.NVIDIA_API_KEY) {
    return { chunks: chunks.slice(0, topK), reranked: false };
  }

  const model = env.NVIDIA_RERANK_MODEL || DEFAULT_RERANK_MODEL;
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
      signal: AbortSignal.timeout(10_000),
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
  if (history.length === 0 || !env.NVIDIA_API_KEY) return question;

  const recent = history
    .slice(-4)
    .map((m) => `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.content.slice(0, 250)}`)
    .join('\n');

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
              'Rewrite the user follow-up question into a single self-contained search query for E-JUST (Egypt-Japan University of Science and Technology). Preserve the original language (English or Arabic). Return ONLY the rewritten query, nothing else.',
          },
          {
            role: 'user',
            content: `Conversation:\n${recent}\n\nFollow-up question: ${question}`,
          },
        ],
        temperature: 0,
        max_tokens: 90,
        stream: false,
      }),
      signal: AbortSignal.timeout(8_000),
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

export async function generateChatCompletion(
  messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>,
  env: Pick<Env, 'NVIDIA_API_KEY' | 'NVIDIA_BASE_URL' | 'NVIDIA_CHAT_MODEL'>,
): Promise<{ content: string; model: string }> {
  const apiKey = requireApiKey(env);
  const model = env.NVIDIA_CHAT_MODEL || DEFAULT_CHAT_MODEL;
  const baseUrl = getBaseUrl(env);

  const res = await fetch(`${baseUrl}/chat/completions`, {
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
      max_tokens: 1024,
      stream: false,
    }),
    signal: AbortSignal.timeout(45_000),
  });

  if (!res.ok) {
    const errBody = await res.text().catch(() => '');
    throw new NvidiaApiError(
      `NVIDIA Chat API error (${res.status}): ${errBody.slice(0, 240)}`,
      res.status === 401 ? 502 : res.status,
    );
  }

  const data = (await res.json()) as {
    choices?: Array<{ message?: { content?: string } }>;
  };

  const content = data.choices?.[0]?.message?.content?.trim();
  if (!content) {
    throw new NvidiaApiError('Empty completion from NVIDIA Chat API', 502);
  }

  return { content, model };
}

export async function streamChatCompletion(
  messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>,
  env: Pick<Env, 'NVIDIA_API_KEY' | 'NVIDIA_BASE_URL' | 'NVIDIA_CHAT_MODEL'>,
): Promise<{ stream: ReadableStream<string>; model: string }> {
  const apiKey = requireApiKey(env);
  const model = env.NVIDIA_CHAT_MODEL || DEFAULT_CHAT_MODEL;
  const baseUrl = getBaseUrl(env);

  const res = await fetch(`${baseUrl}/chat/completions`, {
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
      max_tokens: 1024,
      stream: true,
    }),
    signal: AbortSignal.timeout(50_000),
  });

  if (!res.ok || !res.body) {
    const errBody = await res.text().catch(() => '');
    throw new NvidiaApiError(
      `NVIDIA Chat Stream error (${res.status}): ${errBody.slice(0, 240)}`,
      res.status === 401 ? 502 : res.status,
    );
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

  return { stream: tokenStream, model };
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
