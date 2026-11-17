# E-JUST AI Assistant

**Live Demo:** [ejustai.ziad-dev9.workers.dev](https://ejustai.ziad-dev9.workers.dev)

E-JUST AI is a production **Hybrid Retrieval-Augmented Generation (RAG)** assistant built for **Egypt-Japan University of Science and Technology (E-JUST)** students, applicants, and staff. It runs entirely on the **Cloudflare Edge** (Workers, Static Assets, Vectorize, and D1) and is powered by **NVIDIA NIM** inference and embedding models.

---

## Why I Built It This Way

Most student-facing university chatbots fail on two things: they either hallucinate policies when retrieval misses exact terms (like faculty codes or PDF bylaws), or they break completely when a student asks a question in Arabic against an English-heavy knowledge base.

I designed this system to solve both problems in production without paying for dedicated GPU servers:

- **Hybrid Retrieval (Semantic + Lexical):** Every query runs **Cloudflare Vectorize** (1024-dim cosine similarity via `nvidia/embed-qa-4`) in parallel with **Cloudflare D1 FTS5** (BM25 full-text search). Both ranked lists are merged using **Reciprocal Rank Fusion (RRF)** so exact acronyms (`CSIT`, `FIBH`, `TICAD`) and natural-language questions both surface the right chunks.
- **Cross-Lingual Arabic Support:** Most official E-JUST handbooks and admission PDFs are in English. When a user asks a question in Arabic, a fast query-rewriting step translates the search intent into English for retrieval while instructing the generator to respond in natural Arabic with proper RTL formatting.
- **Real Data Pipeline:** A custom Python crawler and `PyMuPDF` extractor ([`scripts/scrape_ejust.py`](scripts/scrape_ejust.py)) parses **170+ E-JUST web pages** and **25+ official PDFs** (preserving Markdown headings and tuition/curriculum tables) into **4,577 contextualized chunks** stored across D1 and Vectorize.
- **Resilient Inference & Streaming:** Built around `nvidia/nemotron-3.5-bolt-e4b` with automatic multi-model failover, per-model connection timeouts, and live **Server-Sent Events (SSE)** streaming.
- **Designed Like a Real Product:** A clean Light-Theme interface ([`public/index.html`](public/index.html)) with collapsible numbered references that reveal linked `[1]` inline citation badges on expand, `localStorage` chat persistence with AI-generated titles (`POST /title`), and automatic per-message RTL handling.
- **Production Observability:** Asynchronous **Telegram** logging (`ctx.waitUntil`) streams session-grouped prompts, responses, latency, retrieval modes, user feedback, and any server/client errors to a monitoring group without adding latency to the user.

---

## Architecture

```text
Browser (SSE Stream + Local Session History)
   │
   ▼
Cloudflare Worker (TypeScript)
   ├── Query Rewriter (Follow-up resolution + Arabic → English retrieval translation)
   ├── Hybrid Search
   │      ├── Cloudflare Vectorize (1024-dim semantic search)
   │      └── Cloudflare D1 FTS5   (BM25 lexical search)
   │      └── Reciprocal Rank Fusion (RRF)
   ├── NVIDIA NIM Chat Completion (SSE streaming + automatic model fallback)
   └── Async Background Tasks (ctx.waitUntil)
          ├── Cloudflare D1 (conversation history & feedback)
          └── Telegram Bot  (session-tagged logs & error alerts)
```

---

## Tech Stack

- **Edge & Storage:** Cloudflare Workers, Workers Static Assets, Cloudflare Vectorize, Cloudflare D1 (SQLite + FTS5)
- **AI Models (NVIDIA NIM):** `nvidia/nemotron-3.5-bolt-e4b` (chat & query rewriting), `nvidia/embed-qa-4` (1024-dim embeddings)
- **Data Ingestion:** Python (`BeautifulSoup`, `PyMuPDF`), TypeScript (`tsx`, `wrangler`)
- **Testing:** Vitest (`npm test`) & TypeScript strict checking (`npm run typecheck`)
