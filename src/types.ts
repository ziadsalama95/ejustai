export interface Env {
  ASSETS: Fetcher;
  VECTORIZE?: VectorizeIndex;
  DB?: D1Database;
  CHAT_HISTORY?: KVNamespace;

  // Secrets & configuration
  NVIDIA_API_KEY?: string;
  NVIDIA_BASE_URL?: string;
  NVIDIA_CHAT_MODEL?: string;
  NVIDIA_EMBED_MODEL?: string;
  NVIDIA_RERANK_MODEL?: string;
  RAG_TOP_K?: string;
  RAG_MIN_SCORE?: string;
}

export interface KnowledgeChunk {
  id: string;
  title: string;
  url: string;
  faculty?: string;
  label?: string;
  lang?: string;
  source_type?: string;
  text: string;
}

export interface RetrievedChunk extends KnowledgeChunk {
  score: number;
}

export interface SourceRef {
  title: string;
  url: string;
  faculty?: string;
  score?: number;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp?: string;
}

export interface AskRequest {
  question?: string;
  session_id?: string;
  lang?: 'en' | 'ar';
  faculty?: string;
  stream?: boolean;
}

export type RetrievalMode = 'hybrid-reranked' | 'hybrid-rrf' | 'vectorize' | 'fts5' | 'keyword-fallback';

export interface AskResponse {
  answer: string;
  sources: SourceRef[];
  lang: 'en' | 'ar';
  model: string;
  retrieval_mode: RetrievalMode;
  rewritten_query?: string;
}

export interface FeedbackRequest {
  session_id?: string;
  question?: string;
  answer?: string;
  rating?: number;
}
