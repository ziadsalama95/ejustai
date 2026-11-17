# E-JUST AI Assistant (`ejustai`)

A production **Hybrid Retrieval-Augmented Generation (RAG)** assistant for **Egypt-Japan University of Science and Technology (E-JUST)**, deployed 100% on the **Cloudflare Free Tier** (Workers, Static Assets, Vectorize, D1 FTS5) and powered by **free NVIDIA NIM models**.

---

## Architecture

```text
User (Web / Mobile)
   │
   ▼  SSE Streaming / REST
Cloudflare Worker (src/index.ts) + Static Assets (public/index.html)
   │
   ├──► Query Rewriter (NVIDIA NIM) ──► Rewrites follow-up questions using chat history
   │
   ├──► Hybrid Retrieval (src/rag.ts)
   │      ├── Cloudflare Vectorize (1024-dim semantic search via nvidia/embed-qa-4)
   │      └── Cloudflare D1 FTS5  (BM25 lexical search for exact codes, names & acronyms)
   │      └── Reciprocal Rank Fusion (RRF) merges both ranked lists
   │
   ├──► Grounded Generation (src/nvidia.ts)
   │      └── Primary: nvidia/nemotron-3.5-bolt-e4b (with automatic multi-model fallback)
   │
   └──► Async Observability (ctx.waitUntil)
          ├── Cloudflare D1 (chat history & user feedback)
          └── Telegram Bot (session-grouped chat logs, errors & feedback)
```

---

## Key Features

- **Hybrid RAG (Dense + Lexical):** Combines **Cloudflare Vectorize** semantic search with **Cloudflare D1 FTS5** full-text search via **Reciprocal Rank Fusion (RRF)**.
- **Real E-JUST Knowledge Base:** Includes **170+ scraped E-JUST web pages** and **25+ parsed PDFs** (handbooks, admission policies, academic programs) in [`knowledge/`](knowledge/).
- **Resilient Free NVIDIA NIM Models:** Uses `nvidia/nemotron-3.5-bolt-e4b` for fast streaming chat and `nvidia/embed-qa-4` for 1024-dim embeddings, with automatic failover across active NVIDIA models.
- **Clean Light-Theme UI:** Mobile-first, responsive interface ([`public/index.html`](public/index.html)) with SSE token streaming, `[1]` source citations, Markdown tables/code blocks, follow-up suggestion pills, and bilingual English/Arabic support.
- **Local Chat Sessions & Auto-Titling:** Saves conversations in `localStorage` with a collapsible **Recent Chats** sidebar and AI-generated titles (`POST /title`) after the 1st turn and every 5th turn.
- **Telegram Observability:** Optional real-time logging of prompts, responses, errors, and feedback to a Telegram group tagged by `session_id`.

---

## Project Structure

```text
ejustai/
├── knowledge/                  # Curated + scraped E-JUST Markdown & PDF text
├── migrations/0001_init.sql    # Single D1 schema (chunks, FTS5 triggers, messages, feedback)
├── public/index.html           # Responsive Light-Theme chat UI
├── scripts/
│   ├── scrape_ejust.py         # E-JUST crawler + PyMuPDF PDF-to-Markdown extractor
│   └── ingest.ts               # Contextual chunker + D1 & Vectorize uploader
├── src/
│   ├── index.ts                # Worker router, SSE streaming & error handling
│   ├── nvidia.ts               # NVIDIA NIM chat, embeddings, rewrite & title client
│   ├── rag.ts                  # Vectorize + D1 FTS5 hybrid search, RRF & prompts
│   ├── storage.ts              # D1 / KV / in-memory session & feedback storage
│   ├── telegram.ts             # Async Telegram group logger
│   └── types.ts                # TypeScript interfaces
├── tests/rag.test.ts           # Vitest unit & Worker integration tests
├── EJUST_Cloud_Ingest.ipynb    # Zero-local Google Colab ingestion notebook
└── wrangler.jsonc              # Cloudflare Workers configuration
```

---

## Quick Start (Deploy Your Own Instance)

### 1. Prerequisites
- [Node.js 20+](https://nodejs.org/)
- A free [Cloudflare](https://dash.cloudflare.com/) account
- A free [NVIDIA NIM API key](https://build.nvidia.com/) (`nvapi-...`)

### 2. Install & Verify Locally
```bash
git clone https://github.com/ziadsalama95/ejustai.git
cd ejustai
npm install
cp .dev.vars.example .dev.vars   # Add your NVIDIA_API_KEY
npm run typecheck
npm test
npm run dev                      # Starts local server
```

### 3. Create Cloudflare Resources
```bash
npx wrangler login
npx wrangler vectorize create ejustai-index --dimensions=1024 --metric=cosine
npx wrangler d1 create ejustai-db
```
Update `database_id` in [`wrangler.jsonc`](wrangler.jsonc) with the ID printed by `wrangler d1 create`, then apply the single migration:
```bash
npx wrangler d1 migrations apply ejustai-db --remote
```

### 4. Configure Secrets & Deploy
```bash
npx wrangler secret put NVIDIA_API_KEY

# Optional: enable Telegram group logging
npx wrangler secret put TELEGRAM_BOT_TOKEN
npx wrangler secret put TELEGRAM_CHAT_ID

npm run deploy
```

### 5. Populate D1 & Vectorize
Upload the included `knowledge/` dataset (or re-scrape fresh data):
```bash
# Option A: Upload existing knowledge/ files from your machine
npm run ingest:remote

# Option B: Re-scrape live E-JUST pages & PDFs in Google Colab (zero local compute)
# Open EJUST_Cloud_Ingest.ipynb in Google Colab and run Cells 1 → 4.
```

---

## Environment Variables & Secrets

| Name | Type | Description |
| :--- | :--- | :--- |
| `NVIDIA_API_KEY` | Secret | Free API key from `build.nvidia.com` |
| `NVIDIA_CHAT_MODEL` | Var | Primary chat model (default: `nvidia/nemotron-3.5-bolt-e4b`) |
| `NVIDIA_EMBED_MODEL` | Var | 1024-dim embedding model (default: `nvidia/embed-qa-4`) |
| `RAG_TOP_K` | Var | Number of retrieved chunks passed to the LLM (default: `5`) |
| `RAG_MIN_SCORE` | Var | Minimum cosine similarity threshold (default: `0.22`) |
| `TELEGRAM_BOT_TOKEN` | Secret *(optional)* | Telegram Bot token for chat/error/feedback logs |
| `TELEGRAM_CHAT_ID` | Secret *(optional)* | Target Telegram group/channel ID |
