import {
  formatHistoryForModel,
  generateChatCompletion,
  NvidiaApiError,
  rewriteQuery,
  streamChatCompletion,
} from './nvidia';
import { buildSystemPrompt, dedupeSources, retrieveContext } from './rag';
import { appendHistory, clearHistory, getHistory, saveFeedback } from './storage';
import type { AskRequest, AskResponse, Env, FeedbackRequest } from './types';

const CORS_HEADERS: Record<string, string> = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
};

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      'Content-Type': 'application/json; charset=utf-8',
      ...CORS_HEADERS,
    },
  });
}

function detectLanguage(text: string, hint?: string): 'en' | 'ar' {
  if (hint === 'ar' || hint === 'en') return hint;
  const arabicChars = (text.match(/[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]/g) || []).length;
  const cleanLen = text.replace(/\s/g, '').length || 1;
  return arabicChars / cleanLen > 0.25 ? 'ar' : 'en';
}

function normalizePath(pathname: string): string {
  return pathname.startsWith('/api/') ? pathname.slice(4) : pathname;
}

export default {
  async fetch(request: Request, env: Env, ctx?: ExecutionContext): Promise<Response> {
    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: CORS_HEADERS });
    }

    const url = new URL(request.url);
    const route = normalizePath(url.pathname);

    try {
      if (route === '/health' && request.method === 'GET') {
        return jsonResponse({
          status: 'ok',
          service: 'ejustai',
          provider: 'nvidia-nim',
          chat_model: env.NVIDIA_CHAT_MODEL || 'meta/llama-3.3-70b-instruct',
          embed_model: env.NVIDIA_EMBED_MODEL || 'nvidia/nv-embedqa-e5-v5',
          rerank_model: env.NVIDIA_RERANK_MODEL || 'nvidia/nv-rerankqa-mistral-4b-v3',
          vectorize_bound: Boolean(env.VECTORIZE),
          d1_bound: Boolean(env.DB),
          kv_bound: Boolean(env.CHAT_HISTORY),
          api_key_configured: Boolean(
            env.NVIDIA_API_KEY && env.NVIDIA_API_KEY !== 'nvapi-your-free-nvidia-key-here',
          ),
        });
      }

      if (route === '/history' && request.method === 'GET') {
        const sessionId = url.searchParams.get('session_id')?.trim() || '';
        const history = await getHistory(env, sessionId);
        return jsonResponse({ history });
      }

      if (route === '/clear' && request.method === 'POST') {
        const body = (await request.json().catch(() => ({}))) as { session_id?: string };
        const sessionId = body.session_id?.trim() || '';
        await clearHistory(env, sessionId);
        return jsonResponse({ cleared: true });
      }

      if (route === '/feedback' && request.method === 'POST') {
        const body = (await request.json().catch(() => ({}))) as FeedbackRequest;
        const saved = await saveFeedback(
          env,
          body.session_id?.trim() || 'anonymous',
          body.question?.trim() || '',
          body.answer?.trim() || '',
          typeof body.rating === 'number' ? body.rating : 1,
        );
        return jsonResponse({ saved });
      }

      if (route === '/ask' && request.method === 'POST') {
        const body = (await request.json().catch(() => ({}))) as AskRequest;
        const question = body.question?.trim() || '';
        const sessionId = body.session_id?.trim() || 'default';
        const faculty = body.faculty?.trim() || undefined;
        const wantsStream =
          body.stream === true ||
          (request.headers.get('Accept') || '').includes('text/event-stream');

        if (!question) {
          return jsonResponse({ error: 'Question is required.' }, 400);
        }
        if (question.length > 2000) {
          return jsonResponse({ error: 'Question exceeds 2000 character limit.' }, 400);
        }

        const lang = detectLanguage(question, body.lang);
        const history = await getHistory(env, sessionId);
        const searchQuery = await rewriteQuery(question, history, env);
        const { chunks, mode } = await retrieveContext(searchQuery, env, faculty);
        const sources = dedupeSources(chunks);

        const systemPrompt = buildSystemPrompt(chunks, lang);
        const messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }> = [
          { role: 'system', content: systemPrompt },
          ...formatHistoryForModel(history),
          { role: 'user', content: question },
        ];

        if (wantsStream) {
          const { stream, model } = await streamChatCompletion(messages, env);
          const encoder = new TextEncoder();
          const reader = stream.getReader();
          let fullAnswer = '';

          const sseStream = new ReadableStream<Uint8Array>({
            async start(controller) {
              const sendEvent = (data: unknown) => {
                controller.enqueue(encoder.encode(`data: ${JSON.stringify(data)}\n\n`));
              };

              try {
                sendEvent({
                  meta: true,
                  lang,
                  model,
                  retrieval_mode: mode,
                  sources,
                });

                while (true) {
                  const { done, value } = await reader.read();
                  if (done) break;
                  fullAnswer += value;
                  sendEvent({ token: value });
                }

                sendEvent({ done: true, answer: fullAnswer, sources, lang, model, retrieval_mode: mode });
                const savePromise = appendHistory(env, sessionId, question, fullAnswer, sources);
                if (ctx) ctx.waitUntil(savePromise);
                else await savePromise;
              } catch (err) {
                const msg = err instanceof Error ? err.message : 'Stream error';
                sendEvent({ error: msg });
              } finally {
                controller.close();
              }
            },
            cancel() {
              reader.cancel().catch(() => {});
            },
          });

          return new Response(sseStream, {
            status: 200,
            headers: {
              'Content-Type': 'text/event-stream; charset=utf-8',
              'Cache-Control': 'no-cache',
              ...CORS_HEADERS,
            },
          });
        }

        const { content, model } = await generateChatCompletion(messages, env);
        await appendHistory(env, sessionId, question, content, sources);

        const response: AskResponse = {
          answer: content,
          sources,
          lang,
          model,
          retrieval_mode: mode,
          ...(searchQuery !== question ? { rewritten_query: searchQuery } : {}),
        };

        return jsonResponse(response);
      }

      if (env.ASSETS) {
        return env.ASSETS.fetch(request);
      }

      return jsonResponse({ error: 'Not found' }, 404);
    } catch (err) {
      if (err instanceof NvidiaApiError) {
        return jsonResponse({ error: err.message }, err.status);
      }
      const message = err instanceof Error ? err.message : 'Internal server error';
      return jsonResponse({ error: message }, 500);
    }
  },
};
