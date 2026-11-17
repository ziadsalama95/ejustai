import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { basename, extname, join, relative, resolve } from 'node:path';
import { execSync } from 'node:child_process';
import { embedTexts } from '../src/nvidia';
import { chunkMarkdown, parseFrontmatter } from '../src/rag';
import type { KnowledgeChunk } from '../src/types';

const ROOT = resolve(process.cwd());
const KNOWLEDGE_DIR = join(ROOT, 'knowledge');
const CHUNKS_OUT = join(ROOT, 'src', 'data', 'kb-chunks.json');
const NDJSON_OUT = join(ROOT, 'vectors.ndjson');
const SQL_OUT = join(ROOT, 'chunks-seed.sql');
const INDEX_NAME = process.env.VECTORIZE_INDEX || 'ejustai-index';
const D1_DB_NAME = process.env.D1_DATABASE || 'ejustai-db';

function loadDevVars(): void {
  const devVarsPath = join(ROOT, '.dev.vars');
  if (!existsSync(devVarsPath)) return;
  const lines = readFileSync(devVarsPath, 'utf-8').split(/\r?\n/);
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    const eq = trimmed.indexOf('=');
    if (eq === -1) continue;
    const key = trimmed.slice(0, eq).trim();
    const val = trimmed.slice(eq + 1).trim();
    if (!process.env[key]) {
      process.env[key] = val;
    }
  }
}

function collectMarkdownFiles(dir: string): string[] {
  const results: string[] = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      results.push(...collectMarkdownFiles(full));
    } else if (['.md', '.txt'].includes(extname(entry.name).toLowerCase())) {
      results.push(full);
    }
  }
  return results.sort();
}

function sqlEscape(val: string): string {
  return `'${val.replace(/'/g, "''")}'`;
}

function buildD1SeedSql(chunks: KnowledgeChunk[]): string {
  const lines = ['DELETE FROM chunks;'];
  for (const c of chunks) {
    lines.push(
      `INSERT OR REPLACE INTO chunks (id, title, url, faculty, label, lang, source_type, text) VALUES (${[
        sqlEscape(c.id),
        sqlEscape(c.title),
        sqlEscape(c.url),
        sqlEscape(c.faculty || 'EJUST General'),
        sqlEscape(c.label || 'general'),
        sqlEscape(c.lang || 'en'),
        sqlEscape(c.source_type || 'webpage'),
        sqlEscape(c.text),
      ].join(', ')});`,
    );
  }
  return lines.join('\n') + '\n';
}

async function main() {
  loadDevVars();
  const args = new Set(process.argv.slice(2));
  const shouldUpload = args.has('--upload') || args.has('--upload-only');
  const uploadOnly = args.has('--upload-only');
  const chunksOnly = args.has('--chunks-only');

  if (!existsSync(KNOWLEDGE_DIR)) {
    throw new Error(`Knowledge directory not found: ${KNOWLEDGE_DIR}`);
  }

  const files = collectMarkdownFiles(KNOWLEDGE_DIR);
  if (files.length === 0) {
    throw new Error(`No .md or .txt files found in ${KNOWLEDGE_DIR}`);
  }

  const allChunks: KnowledgeChunk[] = [];
  for (const fullPath of files) {
    const raw = readFileSync(fullPath, 'utf-8');
    const { meta } = parseFrontmatter(raw);
    const rel = relative(KNOWLEDGE_DIR, fullPath).replace(/\\/g, '/');
    const docId = rel.replace(/\.[^.]+$/, '').replace(/[^a-zA-Z0-9_-]/g, '_');
    const title = meta.title || basename(fullPath, extname(fullPath)).replace(/^\d+-/, '').replace(/-/g, ' ');
    const url = meta.url || 'https://www.ejust.edu.eg';

    const chunks = chunkMarkdown(raw, docId, title, url, 750, 100, {
      faculty: meta.faculty || 'EJUST General',
      label: meta.label || 'general',
      lang: meta.lang || 'en',
      source_type: meta.source_type || 'webpage',
    });
    allChunks.push(...chunks);
  }

  mkdirSync(join(ROOT, 'src', 'data'), { recursive: true });
  // Keep up to 300 core chunks in local JSON fallback so Worker bundle stays small
  writeFileSync(CHUNKS_OUT, JSON.stringify(allChunks.slice(0, 300), null, 2), 'utf-8');
  writeFileSync(SQL_OUT, buildD1SeedSql(allChunks), 'utf-8');
  console.log(`Processed ${allChunks.length} chunks from ${files.length} files.`);
  console.log(`Wrote fallback chunks -> ${CHUNKS_OUT}`);
  console.log(`Wrote D1 FTS5 seed SQL -> ${SQL_OUT}`);

  const apiKey = process.env.NVIDIA_API_KEY?.trim();
  if (chunksOnly || !apiKey || apiKey === 'nvapi-your-free-nvidia-key-here') {
    console.log('Skipping NVIDIA embeddings (no NVIDIA_API_KEY set or --chunks-only passed).');
    return;
  }

  let ndjsonLines: string[] = [];

  if (uploadOnly && existsSync(NDJSON_OUT)) {
    ndjsonLines = readFileSync(NDJSON_OUT, 'utf-8').split('\n').filter(Boolean);
    console.log(`Reusing ${ndjsonLines.length} existing vectors from ${NDJSON_OUT}`);
  } else {
    console.log('Generating contextual embeddings via NVIDIA NIM...');
    const batchSize = 16;

    for (let i = 0; i < allChunks.length; i += batchSize) {
      const batch = allChunks.slice(i, i + batchSize);
      const contextualInputs = batch.map(
        (c) => `[${c.title} | ${c.faculty || 'EJUST General'} | ${c.label || 'general'}]\n${c.text}`,
      );

      const vectors = await embedTexts(contextualInputs, 'passage', {
        NVIDIA_API_KEY: apiKey,
        NVIDIA_BASE_URL: process.env.NVIDIA_BASE_URL,
        NVIDIA_EMBED_MODEL: process.env.NVIDIA_EMBED_MODEL,
      });

      for (let j = 0; j < batch.length; j++) {
        const chunk = batch[j];
        ndjsonLines.push(
          JSON.stringify({
            id: chunk.id,
            values: vectors[j],
            metadata: {
              title: chunk.title,
              url: chunk.url,
              faculty: chunk.faculty || 'EJUST General',
              label: chunk.label || 'general',
              lang: chunk.lang || 'en',
              text: chunk.text,
            },
          }),
        );
      }

      if ((i + batchSize) % 160 === 0 || i + batchSize >= allChunks.length) {
        console.log(`Embedded ${Math.min(i + batchSize, allChunks.length)}/${allChunks.length} chunks`);
      }
    }

    writeFileSync(NDJSON_OUT, ndjsonLines.join('\n') + '\n', 'utf-8');
    console.log(`Wrote Vectorize NDJSON -> ${NDJSON_OUT}`);
  }

  if (shouldUpload) {
    console.log(`Uploading chunks to Cloudflare D1 (${D1_DB_NAME})...`);
    execSync(`npx wrangler d1 execute ${D1_DB_NAME} --remote --yes --file="${SQL_OUT}"`, {
      stdio: 'inherit',
    });

    console.log(`Upserting ${ndjsonLines.length} vectors to Cloudflare Vectorize (${INDEX_NAME})...`);
    const upsertBatchSize = 500;
    for (let i = 0; i < ndjsonLines.length; i += upsertBatchSize) {
      const partPath = join(ROOT, `vectors.part.${Math.floor(i / upsertBatchSize)}.ndjson`);
      writeFileSync(partPath, ndjsonLines.slice(i, i + upsertBatchSize).join('\n') + '\n', 'utf-8');
      execSync(`npx wrangler vectorize upsert ${INDEX_NAME} --file="${partPath}"`, {
        stdio: 'inherit',
      });
    }
  } else {
    console.log('To upload to Cloudflare D1 & Vectorize, run: npm run ingest:remote');
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
