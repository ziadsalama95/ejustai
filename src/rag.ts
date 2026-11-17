import kbChunksData from './data/kb-chunks.json';
import { embedTexts, rerankChunks } from './nvidia';
import type { Env, KnowledgeChunk, RetrievalMode, RetrievedChunk, SourceRef } from './types';

const DEFAULT_TOP_K = 5;
const DEFAULT_MIN_SCORE = 0.22;

const STOP_WORDS = new Set([
  'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
  'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'about',
  'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why', 'how',
  'do', 'does', 'did', 'can', 'could', 'will', 'would', 'should', 'may',
  'and', 'or', 'but', 'if', 'then', 'than', 'that', 'this', 'these', 'those',
  'me', 'my', 'i', 'you', 'your', 'it', 'its', 'they', 'their', 'we', 'our',
]);

export interface DocMeta {
  title?: string;
  url?: string;
  faculty?: string;
  label?: string;
  lang?: string;
  source_type?: string;
}

export function parseFrontmatter(raw: string): {
  meta: DocMeta;
  body: string;
} {
  const match = raw.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n([\s\S]*)$/);
  if (!match) return { meta: {}, body: raw.trim() };

  const meta: DocMeta = {};
  for (const line of match[1].split(/\r?\n/)) {
    const idx = line.indexOf(':');
    if (idx === -1) continue;
    const key = line.slice(0, idx).trim().toLowerCase();
    const val = line.slice(idx + 1).trim();
    if (key === 'title') meta.title = val;
    if (key === 'url') meta.url = val;
    if (key === 'faculty') meta.faculty = val;
    if (key === 'label') meta.label = val;
    if (key === 'lang') meta.lang = val;
    if (key === 'source_type') meta.source_type = val;
  }

  return { meta, body: match[2].trim() };
}

export function chunkMarkdown(
  markdown: string,
  docId: string,
  title: string,
  url: string,
  maxChars = 750,
  overlapChars = 100,
  extraMeta: Partial<Pick<KnowledgeChunk, 'faculty' | 'label' | 'lang' | 'source_type'>> = {},
): KnowledgeChunk[] {
  const { body } = parseFrontmatter(markdown);
  const sections = body.split(/\n(?=##\s)/g).map((s) => s.trim()).filter(Boolean);
  const chunks: KnowledgeChunk[] = [];
  let idx = 0;

  for (const section of sections) {
    if (section.length <= maxChars) {
      chunks.push({
        id: `${docId}#${idx++}`,
        title,
        url,
        ...extraMeta,
        text: section,
      });
      continue;
    }

    // Split oversized sections by paragraph, and split oversized paragraphs by length
    const rawParagraphs = section.split(/\n{2,}/);
    const paragraphs: string[] = [];
    for (const p of rawParagraphs) {
      if (p.length <= maxChars) {
        paragraphs.push(p);
      } else {
        for (let start = 0; start < p.length; start += maxChars - overlapChars) {
          paragraphs.push(p.slice(start, start + maxChars).trim());
        }
      }
    }

    let buffer = '';
    for (const para of paragraphs) {
      if (buffer && buffer.length + para.length + 2 > maxChars) {
        chunks.push({
          id: `${docId}#${idx++}`,
          title,
          url,
          ...extraMeta,
          text: buffer.trim(),
        });
        const tail = buffer.slice(-overlapChars).trim();
        buffer = tail ? `${tail}\n\n${para}` : para;
      } else {
        buffer = buffer ? `${buffer}\n\n${para}` : para;
      }
    }

    if (buffer.trim()) {
      chunks.push({
        id: `${docId}#${idx++}`,
        title,
        url,
        ...extraMeta,
        text: buffer.trim(),
      });
    }
  }

  return chunks;
}

export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s-]/gu, ' ')
    .split(/\s+/)
    .filter((t) => t.length > 1 && !STOP_WORDS.has(t));
}

export function keywordSearch(
  query: string,
  chunks: KnowledgeChunk[],
  topK = DEFAULT_TOP_K,
  faculty?: string,
): RetrievedChunk[] {
  const queryTokens = tokenize(query);
  if (queryTokens.length === 0) return [];

  const querySet = new Set(queryTokens);
  const scored: RetrievedChunk[] = [];

  for (const chunk of chunks) {
    if (faculty && faculty !== 'All' && chunk.faculty && chunk.faculty !== faculty) {
      continue;
    }
    const hay = `${chunk.title} ${chunk.text}`.toLowerCase();
    let hits = 0;
    for (const tok of querySet) {
      if (hay.includes(tok)) hits += 1;
    }
    if (hits === 0) continue;

    const coverage = hits / querySet.size;
    const titleBoost = queryTokens.some((t) => chunk.title.toLowerCase().includes(t)) ? 0.15 : 0;
    const score = Math.min(1, coverage * 0.85 + titleBoost);
    scored.push({ ...chunk, score });
  }

  return scored.sort((a, b) => b.score - a.score).slice(0, topK);
}

export function reciprocalRankFusion(
  lists: RetrievedChunk[][],
  k = 60,
): RetrievedChunk[] {
  const scores = new Map<string, number>();
  const items = new Map<string, RetrievedChunk>();

  for (const list of lists) {
    list.forEach((item, rank) => {
      const rrf = 1 / (k + rank + 1);
      scores.set(item.id, (scores.get(item.id) ?? 0) + rrf);
      if (!items.has(item.id)) {
        items.set(item.id, item);
      }
    });
  }

  return Array.from(items.values())
    .map((item) => ({
      ...item,
      score: Number((scores.get(item.id) ?? 0).toFixed(5)),
    }))
    .sort((a, b) => b.score - a.score);
}

function buildFtsQuery(raw: string): string {
  const tokens = tokenize(raw).slice(0, 8);
  if (tokens.length === 0) return '';
  return tokens.map((t) => `"${t.replace(/"/g, '')}"`).join(' OR ');
}

async function fts5Search(
  query: string,
  env: Pick<Env, 'DB'>,
  limit = 15,
  faculty?: string,
): Promise<RetrievedChunk[]> {
  if (!env.DB) return [];
  const ftsQuery = buildFtsQuery(query);
  if (!ftsQuery) return [];

  try {
    const hasFacultyFilter = faculty && faculty !== 'All';
    const sql = hasFacultyFilter
      ? `SELECT c.id, c.title, c.url, c.faculty, c.label, c.lang, c.source_type, c.text, -bm25(chunks_fts) AS rank_score
         FROM chunks_fts f
         JOIN chunks c ON c.rowid = f.rowid
         WHERE chunks_fts MATCH ? AND c.faculty = ?
         ORDER BY bm25(chunks_fts)
         LIMIT ?`
      : `SELECT c.id, c.title, c.url, c.faculty, c.label, c.lang, c.source_type, c.text, -bm25(chunks_fts) AS rank_score
         FROM chunks_fts f
         JOIN chunks c ON c.rowid = f.rowid
         WHERE chunks_fts MATCH ?
         ORDER BY bm25(chunks_fts)
         LIMIT ?`;

    const stmt = hasFacultyFilter
      ? env.DB.prepare(sql).bind(ftsQuery, faculty, limit)
      : env.DB.prepare(sql).bind(ftsQuery, limit);

    const res = await stmt.all<KnowledgeChunk & { rank_score: number }>();
    return (res.results || []).map((row) => ({
      id: row.id,
      title: row.title,
      url: row.url,
      faculty: row.faculty,
      label: row.label,
      lang: row.lang,
      source_type: row.source_type,
      text: row.text,
      score: Math.max(0.01, row.rank_score ?? 0.5),
    }));
  } catch {
    return [];
  }
}

async function vectorizeSearch(
  query: string,
  env: Env,
  limit = 15,
  minScore = DEFAULT_MIN_SCORE,
  faculty?: string,
): Promise<RetrievedChunk[]> {
  if (!env.VECTORIZE || !env.NVIDIA_API_KEY) return [];
  try {
    const [queryVector] = await embedTexts([query], 'query', env);
    if (!queryVector?.length) return [];

    const filter =
      faculty && faculty !== 'All' ? { faculty: { $eq: faculty } } : undefined;

    const matches = await env.VECTORIZE.query(queryVector, {
      topK: limit,
      returnMetadata: 'all',
      filter,
    });

    return (matches.matches || [])
      .filter((m) => (m.score ?? 0) >= minScore)
      .map((m) => {
        const meta = (m.metadata || {}) as Record<string, string>;
        return {
          id: m.id,
          title: meta.title || 'E-JUST Knowledge Base',
          url: meta.url || 'https://www.ejust.edu.eg',
          faculty: meta.faculty || 'EJUST General',
          label: meta.label || 'general',
          lang: meta.lang || 'en',
          text: meta.text || '',
          score: m.score ?? 0,
        };
      })
      .filter((c) => c.text.length > 0);
  } catch {
    return [];
  }
}

export async function retrieveContext(
  query: string,
  env: Env,
  faculty?: string,
): Promise<{ chunks: RetrievedChunk[]; mode: RetrievalMode }> {
  const topK = Number(env.RAG_TOP_K) || DEFAULT_TOP_K;
  const minScore = Number(env.RAG_MIN_SCORE) || DEFAULT_MIN_SCORE;
  const localChunks = kbChunksData as KnowledgeChunk[];

  const [vecResults, ftsResults] = await Promise.all([
    vectorizeSearch(query, env, 15, minScore, faculty),
    fts5Search(query, env, 15, faculty),
  ]);

  if (vecResults.length > 0 && ftsResults.length > 0) {
    const fused = reciprocalRankFusion([vecResults, ftsResults]).slice(0, 15);
    const { chunks, reranked } = await rerankChunks(query, fused, topK, env);
    return {
      chunks,
      mode: reranked ? 'hybrid-reranked' : 'hybrid-rrf',
    };
  }

  if (vecResults.length > 0) {
    const { chunks, reranked } = await rerankChunks(query, vecResults.slice(0, 15), topK, env);
    return {
      chunks,
      mode: reranked ? 'hybrid-reranked' : 'vectorize',
    };
  }

  if (ftsResults.length > 0) {
    const { chunks, reranked } = await rerankChunks(query, ftsResults.slice(0, 15), topK, env);
    return {
      chunks,
      mode: reranked ? 'hybrid-reranked' : 'fts5',
    };
  }

  return {
    chunks: keywordSearch(query, localChunks, topK, faculty),
    mode: 'keyword-fallback',
  };
}

export function dedupeSources(chunks: RetrievedChunk[]): SourceRef[] {
  const seen = new Map<string, SourceRef>();
  for (const c of chunks) {
    const key = c.url || c.title;
    const existing = seen.get(key);
    if (!existing || c.score > (existing.score ?? 0)) {
      seen.set(key, {
        title: c.title,
        url: c.url,
        faculty: c.faculty,
        score: Number(c.score.toFixed(3)),
      });
    }
  }
  return Array.from(seen.values());
}

export function buildSystemPrompt(chunks: RetrievedChunk[], lang: 'en' | 'ar'): string {
  const contextBlock = chunks.length
    ? chunks
        .map(
          (c, i) =>
            `[Source ${i + 1}: ${c.title}${c.faculty ? ` | ${c.faculty}` : ''} (${c.url})]\n${c.text}`,
        )
        .join('\n\n---\n\n')
    : 'NO RELEVANT KNOWLEDGE BASE DOCUMENTS WERE RETRIEVED.';

  const langInstruction =
    lang === 'ar'
      ? 'Respond in clear, natural Arabic (Egyptian or Modern Standard Arabic matching the user tone). Keep official English program names or course codes accurate.'
      : 'Respond in clear, concise English using well-structured Markdown (bullet points, tables when comparing items, bold key terms).';

  const suggestionHeader =
    lang === 'ar' ? '**ممكن كمان تسأل عن:**' : '**You might also want to ask:**';

  return `You are the E-JUST AI Assistant — the authoritative, grounded RAG assistant for Egypt-Japan University of Science and Technology (E-JUST) serving students, prospective applicants, faculty, and staff.

RULES:
1. Ground every factual claim in the RETRIEVED CONTEXT below. Cite supporting sources inline using [1], [2], etc. matching the source numbers below.
2. Never invent tuition numbers, deadlines, GPAs, or policies that are not explicitly present in the retrieved context.
3. If the retrieved context does not contain enough information to answer the question, explicitly say:
   - In English: "I don't have that information in my current knowledge base. Please verify on the official E-JUST website (https://www.ejust.edu.eg)."
   - In Arabic: "معنديش المعلومة دي في قاعدة البيانات الحالية — يفضل تتأكد من الموقع الرسمي للجامعة (https://www.ejust.edu.eg)."
4. ${langInstruction}
5. At the very end of your response, always include 2-3 short follow-up questions in the exact format below:

---
${suggestionHeader}
1. <short follow-up question>
2. <short follow-up question>

RETRIEVED CONTEXT:
${contextBlock}`;
}
