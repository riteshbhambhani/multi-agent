PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS users (
  user_id TEXT PRIMARY KEY,
  name TEXT,
  email TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  title TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS messages (
  message_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  role TEXT CHECK(role IN ('user','assistant','system')),
  content TEXT NOT NULL,
  agent TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS checkpoints (
  checkpoint_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  session_id TEXT NOT NULL,
  pending_agent TEXT NOT NULL,
  pending_question TEXT NOT NULL,
  context_snapshot TEXT NOT NULL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS benefits (
  benefit_id TEXT PRIMARY KEY,
  member_id TEXT,
  coverage TEXT,
  deductible REAL,
  copay REAL,
  coinsurance REAL,
  out_of_pocket REAL,
  source_file TEXT NOT NULL,
  content TEXT NOT NULL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS claims (
  claim_id TEXT PRIMARY KEY,
  member_id TEXT,
  service_date TEXT,
  provider TEXT,
  status TEXT,
  billed_amount REAL,
  allowed_amount REAL,
  paid_amount REAL,
  source_file TEXT NOT NULL,
  content TEXT NOT NULL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS vector_index (
  idx_name TEXT PRIMARY KEY,
  location TEXT NOT NULL,
  dim INTEGER NOT NULL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS provenance (
  prov_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  agent TEXT NOT NULL,
  model_name TEXT NOT NULL,
  quantization TEXT,
  sources TEXT NOT NULL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ragas_runs (
  run_id TEXT PRIMARY KEY,
  dataset_name TEXT,
  metrics_json TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
