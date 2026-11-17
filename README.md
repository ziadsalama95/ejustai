# E-JUST AI Assistant (`ejustai`)

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot built for **Cloudflare Workers** (with Static Assets, Vectorize, and KV) and powered by **free NVIDIA NIM models**.

---

## What Was Missing in the Original Repository?

The original repository only contained a single static `index.html` file designed to connect to an external `ngrok-free.app` tunnel. To make this a complete, production-ready RAG application on Cloudflare, the following pieces were missing and have now been built:

1. **Backend API Server** — Replaced the missing Colab/ngrok server with a TypeScript **Cloudflare Worker** (`src/index.ts`) serving `/health`, `/ask`, `/history`, `/clear`, and static frontend assets from the same origin.
2. **Free NVIDIA NIM Integration** (`src/nvidia.ts`) — Connects to NVIDIA's OpenAI-compatible API (`https://integrate.api.nvidia.com/v1`) using free models (`meta/llama-3.3-70b-instruct` for chat and `nvidia/nv-embedqa-e5-v5` for embeddings).
3. **Knowledge Base & Ingestion Pipeline** (`knowledge/` + `scripts/ingest.ts`) — Added structured Markdown documents, a chunking pipeline, and an embedding/upload script for **Cloudflare Vectorize**.
4. **Vector Retrieval + Graceful Fallback** (`src/rag.ts`) — Queries Cloudflare Vectorize in production and automatically falls back to local keyword scoring over `src/data/kb-chunks.json` during local development or when Vectorize is not yet populated.
5. **Persistent Session History** (`src/storage.ts`) — Stores conversation history in **Cloudflare KV** (with an in-memory fallback for local dev).
6. **Frontend Auto-Connection & Bug Fixes** (`public/index.html`) — Auto-connects to the built-in Cloudflare Worker API on load (no manual ngrok URL required) and fixes the missing `#footerHint` element bug.

---

## Project Structure

```text
ejustai/
├── knowledge/          # Markdown knowledge base files (add your own .md/.txt here)
├── public/index.html   # Bilingual (EN/AR) E-JUST AI Assistant frontend
├── scripts/ingest.ts   # Chunks knowledge/ and embeds via NVIDIA NIM for Vectorize
├── src/
│   ├── index.ts        # Cloudflare Worker router + static asset serving
│   ├── nvidia.ts       # NVIDIA NIM chat & embeddings client
│   ├── rag.ts          # Chunking, Vectorize retrieval, keyword fallback & prompts
│   ├── storage.ts      # Cloudflare KV / in-memory chat history
│   ├── types.ts        # TypeScript interfaces
│   └── data/kb-chunks.json
├── tests/rag.test.ts   # Vitest unit & Worker route tests
└── wrangler.jsonc      # Cloudflare Workers configuration
```

---

## Quick Start (Local Development)

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Get a free NVIDIA API key:**
   - Sign in at [build.nvidia.com](https://build.nvidia.com) and generate an API key (`nvapi-...`).
   - Copy `.dev.vars.example` to `.dev.vars` and paste your key:
     ```bash
     cp .dev.vars.example .dev.vars
     ```

3. **Run tests & typecheck:**
   ```bash
   npm run typecheck
   npm test
   ```

4. **Start local dev server:**
   ```bash
   npm run dev
   ```

---

## Deploy to Cloudflare

1. **Authenticate Wrangler:**
   ```bash
   npx wrangler login
   ```

2. **Create Vectorize index (1024 dims for `nvidia/nv-embedqa-e5-v5`):**
   ```bash
   npx wrangler vectorize create ejustai-index --dimensions=1024 --metric=cosine
   ```

3. **Create KV namespace for chat history:**
   ```bash
   npx wrangler kv namespace create CHAT_HISTORY
   ```
   Uncomment the `vectorize` and `kv_namespaces` blocks in `wrangler.jsonc` and paste your KV namespace `id`.

4. **Set your NVIDIA API key as a Cloudflare secret:**
   ```bash
   npx wrangler secret put NVIDIA_API_KEY
   ```

5. **Embed & upload your knowledge base:**
   ```bash
   npm run ingest:remote
   ```

6. **Deploy:**
   ```bash
   npm run deploy
   ```
