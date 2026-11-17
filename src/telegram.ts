import type { Env, RetrievalMode, SourceRef } from './types';

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function truncate(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen - 1) + '…';
}

function sanitizeTag(sessionId: string): string {
  return sessionId.replace(/[^a-zA-Z0-9_]/g, '_');
}

async function sendTelegramMessage(
  env: Pick<Env, 'TELEGRAM_BOT_TOKEN' | 'TELEGRAM_CHAT_ID'>,
  htmlText: string,
): Promise<void> {
  const token = env.TELEGRAM_BOT_TOKEN?.trim();
  const chatId = env.TELEGRAM_CHAT_ID?.trim();
  if (!token || !chatId) return;

  try {
    await fetch(`https://api.telegram.org/bot${token}/sendMessage`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        chat_id: chatId,
        text: htmlText.slice(0, 4090),
        parse_mode: 'HTML',
        disable_web_page_preview: true,
      }),
      signal: AbortSignal.timeout(8_000),
    });
  } catch {
    // Never let logging failures break user requests
  }
}

export async function logChatToTelegram(
  env: Pick<Env, 'TELEGRAM_BOT_TOKEN' | 'TELEGRAM_CHAT_ID'>,
  params: {
    sessionId: string;
    userName?: string;
    question: string;
    rewrittenQuery?: string;
    answer: string;
    model: string;
    lang: string;
    faculty?: string;
    retrievalMode: RetrievalMode;
    sources: SourceRef[];
    latencyMs: number;
  },
): Promise<void> {
  const tag = `#s_${sanitizeTag(params.sessionId)}`;
  const userLabel = params.userName ? escapeHtml(params.userName) : 'Anonymous';
  const rewrittenLine =
    params.rewrittenQuery && params.rewrittenQuery !== params.question
      ? `\n🔎 <b>Rewritten Query:</b> <i>${escapeHtml(truncate(params.rewrittenQuery, 240))}</i>`
      : '';

  const sourcesList = params.sources.length
    ? params.sources.map((s, i) => `${i + 1}. ${escapeHtml(truncate(s.title, 60))}`).join('\n')
    : 'None';

  const msg = [
    `💬 <b>E-JUST AI — Chat Interaction</b>`,
    `👤 <b>User:</b> <b>${userLabel}</b> | 🆔 <b>Session:</b> <code>${escapeHtml(params.sessionId)}</code> (${tag})`,
    `⚙️ <b>Model:</b> <code>${escapeHtml(params.model)}</code> | <b>Mode:</b> <code>${escapeHtml(params.retrievalMode)}</code>`,
    `🌐 <b>Lang:</b> ${escapeHtml(params.lang.toUpperCase())} | <b>Faculty:</b> ${escapeHtml(params.faculty || 'All')} | <b>Latency:</b> ${params.latencyMs}ms`,
    ``,
    `👤 <b>Prompt:</b>`,
    `<pre>${escapeHtml(truncate(params.question, 900))}</pre>${rewrittenLine}`,
    ``,
    `🤖 <b>Response:</b>`,
    `<pre>${escapeHtml(truncate(params.answer, 1800))}</pre>`,
    ``,
    `📚 <b>Sources (${params.sources.length}):</b>`,
    sourcesList,
  ].join('\n');

  await sendTelegramMessage(env, msg);
}

export async function logErrorToTelegram(
  env: Pick<Env, 'TELEGRAM_BOT_TOKEN' | 'TELEGRAM_CHAT_ID'>,
  params: {
    sessionId?: string;
    userName?: string;
    route: string;
    method: string;
    status: number;
    error: string;
    question?: string;
  },
): Promise<void> {
  const sid = params.sessionId || 'unknown';
  const tag = `#s_${sanitizeTag(sid)}`;
  const userLabel = params.userName ? escapeHtml(params.userName) : 'Anonymous';
  const promptBlock = params.question
    ? `\n\n👤 <b>Prompt:</b>\n<pre>${escapeHtml(truncate(params.question, 800))}</pre>`
    : '';

  const msg = [
    `🚨 <b>E-JUST AI — Error Alert</b>`,
    `👤 <b>User:</b> <b>${userLabel}</b> | 🆔 <b>Session:</b> <code>${escapeHtml(sid)}</code> (${tag})`,
    `📍 <b>Endpoint:</b> <code>${escapeHtml(params.method)} ${escapeHtml(params.route)}</code> | <b>Status:</b> <code>${params.status}</code>`,
    ``,
    `❌ <b>Error:</b>`,
    `<pre>${escapeHtml(truncate(params.error, 1800))}</pre>${promptBlock}`,
  ].join('\n');

  await sendTelegramMessage(env, msg);
}

export async function logFeedbackToTelegram(
  env: Pick<Env, 'TELEGRAM_BOT_TOKEN' | 'TELEGRAM_CHAT_ID'>,
  params: {
    sessionId: string;
    userName?: string;
    question: string;
    answer: string;
    rating: number;
  },
): Promise<void> {
  const icon = params.rating > 0 ? '👍' : '👎';
  const tag = `#s_${sanitizeTag(params.sessionId)}`;
  const userLabel = params.userName ? escapeHtml(params.userName) : 'Anonymous';

  const msg = [
    `${icon} <b>E-JUST AI — User Feedback</b>`,
    `👤 <b>User:</b> <b>${userLabel}</b> | 🆔 <b>Session:</b> <code>${escapeHtml(params.sessionId)}</code> (${tag})`,
    `⭐ <b>Rating:</b> ${params.rating > 0 ? 'Helpful (+1)' : 'Removed (0)'}`,
    ``,
    `👤 <b>Prompt:</b>`,
    `<pre>${escapeHtml(truncate(params.question || 'N/A', 700))}</pre>`,
    ``,
    `🤖 <b>Response:</b>`,
    `<pre>${escapeHtml(truncate(params.answer || 'N/A', 1200))}</pre>`,
  ].join('\n');

  await sendTelegramMessage(env, msg);
}
