import type { Env, SourceRef } from './types';

type TelegramEnv = Pick<Env, 'TELEGRAM_BOT_TOKEN' | 'TELEGRAM_CHAT_ID'>;

const ANSWER_LIMIT = 2400;
const PROMPT_LIMIT = 800;

function esc(text: string): string {
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function cut(text: string, max: number): string {
  const clean = text.trim();
  return clean.length <= max ? clean : clean.slice(0, max - 1).trimEnd() + '…';
}

function safeUrl(url: string): string | null {
  if (!/^https?:\/\//i.test(url)) return null;
  try {
    return esc(encodeURI(decodeURI(url))).replace(/"/g, '%22');
  } catch {
    return null;
  }
}

/** Converts the model's Markdown answer into Telegram-safe HTML with clickable [n] citations. */
function formatAnswer(answer: string, sources: SourceRef[]): string {
  const body = answer.replace(/\n?---\s*\n\*\*[^*\n]+\*\*[\s\S]*$/, '');
  let html = esc(cut(body, ANSWER_LIMIT));

  html = html
    .replace(/^#{1,3}\s+(.+)$/gm, '<b>$1</b>')
    .replace(/\*\*([^*\n]+?)\*\*/g, '<b>$1</b>')
    .replace(/^\s*[*\-+]\s+/gm, '• ')
    .replace(/,\s*(?=\[\d+\])/g, ' ');

  html = html.replace(/\[(\d+)\]/g, (match, n: string) => {
    const src = sources[Number(n) - 1];
    const href = src ? safeUrl(src.url) : null;
    return href ? `<a href="${href}">[${n}]</a>` : match;
  });

  return html;
}

function userLabel(name?: string): string {
  return name?.trim() ? esc(name.trim()) : 'Unknown';
}

async function send(env: TelegramEnv, html: string): Promise<void> {
  const token = env.TELEGRAM_BOT_TOKEN?.trim();
  const chatId = env.TELEGRAM_CHAT_ID?.trim();
  if (!token || !chatId) return;

  try {
    await fetch(`https://api.telegram.org/bot${token}/sendMessage`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        chat_id: chatId,
        text: html,
        parse_mode: 'HTML',
        disable_web_page_preview: true,
      }),
      signal: AbortSignal.timeout(8_000),
    });
  } catch {
    // Logging must never affect the user experience
  }
}

export async function logChatToTelegram(
  env: TelegramEnv,
  p: {
    sessionId: string;
    userName?: string;
    question: string;
    answer: string;
    sources: SourceRef[];
  },
): Promise<void> {
  await send(
    env,
    [
      `💬 <b>New Conversation Turn</b>`,
      `━━━━━━━━━━━━━━━━━━`,
      `👤 <b>${userLabel(p.userName)}</b>`,
      `🆔 <code>${esc(p.sessionId)}</code>`,
      ``,
      `🗨 <b>Prompt</b>`,
      `<blockquote>${esc(cut(p.question, PROMPT_LIMIT))}</blockquote>`,
      `🤖 <b>Response</b>`,
      `<blockquote>${formatAnswer(p.answer, p.sources)}</blockquote>`,
    ].join('\n'),
  );
}

export async function logErrorToTelegram(
  env: TelegramEnv,
  p: {
    sessionId?: string;
    userName?: string;
    status: number;
    error: string;
    question?: string;
  },
): Promise<void> {
  const lines = [
    `🚨 <b>Something Went Wrong</b>`,
    `━━━━━━━━━━━━━━━━━━`,
    `👤 <b>${userLabel(p.userName)}</b>`,
    `🆔 <code>${esc(p.sessionId || 'unknown')}</code>`,
    ``,
  ];
  if (p.question) {
    lines.push(`🗨 <b>Prompt</b>`, `<blockquote>${esc(cut(p.question, PROMPT_LIMIT))}</blockquote>`);
  }
  lines.push(
    `❌ <b>Error (${p.status})</b>`,
    `<blockquote>${esc(cut(p.error, 900))}</blockquote>`,
  );
  await send(env, lines.join('\n'));
}

export async function logFeedbackToTelegram(
  env: TelegramEnv,
  p: {
    sessionId: string;
    userName?: string;
    question: string;
    answer: string;
    rating: number;
  },
): Promise<void> {
  const positive = p.rating > 0;
  await send(
    env,
    [
      positive ? `👍 <b>Marked as Helpful</b>` : `↩️ <b>Helpful Mark Removed</b>`,
      `━━━━━━━━━━━━━━━━━━`,
      `👤 <b>${userLabel(p.userName)}</b>`,
      `🆔 <code>${esc(p.sessionId)}</code>`,
      ``,
      `🗨 <b>Prompt</b>`,
      `<blockquote>${esc(cut(p.question || 'N/A', 500))}</blockquote>`,
      `🤖 <b>Response</b>`,
      `<blockquote>${esc(cut(p.answer || 'N/A', 900))}</blockquote>`,
    ].join('\n'),
  );
}

export async function logProfileToTelegram(
  env: TelegramEnv,
  p: { sessionId: string; newName: string; oldName?: string },
): Promise<void> {
  const changed = Boolean(p.oldName?.trim()) && p.oldName!.trim() !== p.newName.trim();
  if (p.oldName?.trim() && !changed) return;

  const lines = changed
    ? [
        `✏️ <b>Name Updated</b>`,
        `━━━━━━━━━━━━━━━━━━`,
        `Previous: <b>${esc(p.oldName!.trim())}</b>`,
        `Now: <b>${esc(p.newName.trim())}</b>`,
        `🆔 <code>${esc(p.sessionId)}</code>`,
      ]
    : [
        `🎉 <b>New User Joined</b>`,
        `━━━━━━━━━━━━━━━━━━`,
        `👤 <b>${esc(p.newName.trim())}</b>`,
        `🆔 <code>${esc(p.sessionId)}</code>`,
      ];

  await send(env, lines.join('\n'));
}
