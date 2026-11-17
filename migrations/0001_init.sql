-- E-JUST AI Assistant D1 Schema (Chunks + FTS5 Hybrid Search + Chat History + Feedback)

CREATE TABLE IF NOT EXISTS chunks (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  url TEXT NOT NULL,
  faculty TEXT NOT NULL DEFAULT 'EJUST General',
  label TEXT NOT NULL DEFAULT 'general',
  lang TEXT NOT NULL DEFAULT 'en',
  source_type TEXT NOT NULL DEFAULT 'webpage',
  text TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  id UNINDEXED,
  title,
  faculty,
  label,
  text,
  content='chunks',
  content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, id, title, faculty, label, text)
  VALUES (new.rowid, new.id, new.title, new.faculty, new.label, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, id, title, faculty, label, text)
  VALUES ('delete', old.rowid, old.id, old.title, old.faculty, old.label, old.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, id, title, faculty, label, text)
  VALUES ('delete', old.rowid, old.id, old.title, old.faculty, old.label, old.text);
  INSERT INTO chunks_fts(rowid, id, title, faculty, label, text)
  VALUES (new.rowid, new.id, new.title, new.faculty, new.label, new.text);
END;

CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  sources_json TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);

CREATE TABLE IF NOT EXISTS feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  question TEXT NOT NULL,
  answer TEXT NOT NULL,
  rating INTEGER NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
