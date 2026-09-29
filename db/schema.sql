-- Схема идемпотентна: API применяет её при старте, поэтому старые тома БД обновляются сами.
CREATE TABLE IF NOT EXISTS searches (
  id SERIAL PRIMARY KEY,
  query TEXT NOT NULL,
  query_en TEXT,
  translation_method TEXT,
  n_sources INT NOT NULL DEFAULT 0,
  n_candidates INT NOT NULL DEFAULT 0,
  n_signals INT NOT NULL DEFAULT 0,
  n_confident INT NOT NULL DEFAULT 0,
  n_rejected INT NOT NULL DEFAULT 0,
  llm_model TEXT,
  elapsed_sec DOUBLE PRECISION,
  created_at TIMESTAMP DEFAULT NOW()
);
ALTER TABLE searches ADD COLUMN IF NOT EXISTS meta JSONB;  -- перевод, статистика, статус LLM и модели

CREATE TABLE IF NOT EXISTS documents (
  id SERIAL PRIMARY KEY,
  query TEXT NOT NULL,
  title TEXT NOT NULL,
  url TEXT NOT NULL UNIQUE,
  published_at DATE,
  source_name TEXT NOT NULL,
  source_type TEXT NOT NULL, -- gov, edu, patent, paper, conf, industry_media, analytics, blog, social, press
  lang TEXT NOT NULL,        -- ru / en / other
  trust_level TEXT NOT NULL, -- high / medium / low
  trust_score DOUBLE PRECISION NOT NULL,
  snippet TEXT,
  created_at TIMESTAMP DEFAULT NOW()
);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS search_id INT REFERENCES searches(id) ON DELETE SET NULL;

CREATE TABLE IF NOT EXISTS signals (
  id SERIAL PRIMARY KEY,
  query TEXT NOT NULL,
  tech_name TEXT NOT NULL,
  description_ru TEXT NOT NULL,
  advantage_ru TEXT NOT NULL,
  case_example TEXT NOT NULL,
  weak_score DOUBLE PRECISION NOT NULL,   -- 0..1 уверенность модели+правил
  ml_proba DOUBLE PRECISION,
  rule_score DOUBLE PRECISION,
  predictors JSONB NOT NULL,              -- ключевые предикторы с объяснением
  maturity_verdict TEXT NOT NULL,         -- weak / mature_rejected / noise_rejected / low_trust
  reject_reason TEXT,
  llm_model TEXT NOT NULL,                -- какая модель сгенерировала текст (требование ТЗ)
  llm_prompt_hash TEXT,
  source_ids INT[] DEFAULT '{}',
  created_at TIMESTAMP DEFAULT NOW()
);
ALTER TABLE signals ADD COLUMN IF NOT EXISTS search_id INT REFERENCES searches(id) ON DELETE CASCADE;
ALTER TABLE signals ADD COLUMN IF NOT EXISTS rank INT;
ALTER TABLE signals ADD COLUMN IF NOT EXISTS card JSONB;   -- полная карточка для страницы инсайта

CREATE TABLE IF NOT EXISTS llm_log (
  id SERIAL PRIMARY KEY,
  model TEXT NOT NULL,
  purpose TEXT NOT NULL, -- query_expand / summarize / hypothesis
  prompt_hash TEXT NOT NULL,
  sources_used INT[] DEFAULT '{}',
  created_at TIMESTAMP DEFAULT NOW()
);
ALTER TABLE llm_log ADD COLUMN IF NOT EXISTS search_id INT REFERENCES searches(id) ON DELETE CASCADE;
ALTER TABLE llm_log ADD COLUMN IF NOT EXISTS ok BOOLEAN;

CREATE INDEX IF NOT EXISTS idx_documents_query ON documents(query);
CREATE INDEX IF NOT EXISTS idx_signals_query ON signals(query);
CREATE INDEX IF NOT EXISTS idx_signals_search ON signals(search_id);
CREATE INDEX IF NOT EXISTS idx_searches_created ON searches(created_at DESC);
