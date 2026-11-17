-- Fix FTS5 delete/update triggers to include old.title (7 values for 7 columns)

DROP TRIGGER IF EXISTS chunks_ad;
DROP TRIGGER IF EXISTS chunks_au;

CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, id, title, faculty, label, text)
  VALUES ('delete', old.rowid, old.id, old.title, old.faculty, old.label, old.text);
END;

CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, id, title, faculty, label, text)
  VALUES ('delete', old.rowid, old.id, old.title, old.faculty, old.label, old.text);
  INSERT INTO chunks_fts(rowid, id, title, faculty, label, text)
  VALUES (new.rowid, new.id, new.title, new.faculty, new.label, new.text);
END;
