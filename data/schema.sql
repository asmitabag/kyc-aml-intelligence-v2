-- Canonical database schema, KYC-AML Intelligence API contract v1.0
-- Owner: Saloni. Column names, types and enum values follow docs/API_CONTRACTS_V1.md exactly.
-- Timestamps are ISO 8601 UTC strings ("2026-10-09T12:00:00Z"). SQLite has no native datetime type.
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS customers (
    customer_id    TEXT PRIMARY KEY,
    full_name      TEXT NOT NULL,
    date_of_birth  TEXT,
    created_at     TEXT NOT NULL,
    customer_type  TEXT NOT NULL CHECK (customer_type IN ('INDIVIDUAL', 'BUSINESS')),
    data_origin    TEXT NOT NULL CHECK (data_origin IN ('SYNTHETIC', 'CONSENTED_DEMO', 'IMPORTED'))
);

CREATE TABLE IF NOT EXISTS accounts (
    account_id    TEXT PRIMARY KEY,
    customer_id   TEXT NOT NULL REFERENCES customers (customer_id),
    account_type  TEXT NOT NULL,
    currency      TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kyc_results (
    kyc_id                       TEXT PRIMARY KEY,
    customer_id                  TEXT NOT NULL REFERENCES customers (customer_id),
    document_authenticity_score  REAL CHECK (document_authenticity_score IS NULL OR document_authenticity_score BETWEEN 0 AND 1),
    face_match_score             REAL CHECK (face_match_score IS NULL OR face_match_score BETWEEN 0 AND 1),
    liveness_score               REAL CHECK (liveness_score IS NULL OR liveness_score BETWEEN 0 AND 1),
    kyc_risk_score               REAL CHECK (kyc_risk_score IS NULL OR kyc_risk_score BETWEEN 0 AND 1),
    document_status              TEXT NOT NULL CHECK (document_status   IN ('PASS', 'REVIEW', 'FAIL', 'NOT_RUN')),
    face_match_status            TEXT NOT NULL CHECK (face_match_status IN ('PASS', 'REVIEW', 'FAIL', 'NOT_RUN')),
    liveness_status              TEXT NOT NULL CHECK (liveness_status   IN ('PASS', 'REVIEW', 'FAIL', 'NOT_RUN')),
    overall_status               TEXT NOT NULL CHECK (overall_status IN ('VERIFIED', 'REVIEW', 'REJECTED', 'PENDING')),
    model_version                TEXT NOT NULL,
    created_at                   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    document_id   TEXT PRIMARY KEY,
    customer_id   TEXT NOT NULL REFERENCES customers (customer_id),
    document_type TEXT NOT NULL,
    storage_uri   TEXT NOT NULL,
    data_origin   TEXT NOT NULL CHECK (data_origin IN ('SYNTHETIC', 'CONSENTED_DEMO', 'IMPORTED')),
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id       TEXT PRIMARY KEY,
    from_account_id      TEXT NOT NULL REFERENCES accounts (account_id),
    to_account_id        TEXT NOT NULL REFERENCES accounts (account_id),
    amount               REAL NOT NULL CHECK (amount > 0),
    currency             TEXT NOT NULL,
    timestamp            TEXT NOT NULL,
    transaction_type     TEXT NOT NULL,
    data_origin          TEXT NOT NULL CHECK (data_origin IN ('SYNTHETIC', 'IMPORTED')),
    -- Training/evaluation only. NEVER an inference feature.
    is_suspicious_label  INTEGER CHECK (is_suspicious_label IS NULL OR is_suspicious_label IN (0, 1))
);

CREATE TABLE IF NOT EXISTS risk_assessments (
    assessment_id       TEXT PRIMARY KEY,
    customer_id         TEXT NOT NULL REFERENCES customers (customer_id),
    account_id          TEXT REFERENCES accounts (account_id),
    kyc_risk_score      REAL CHECK (kyc_risk_score IS NULL OR kyc_risk_score BETWEEN 0 AND 1),
    gnn_risk_score      REAL CHECK (gnn_risk_score IS NULL OR gnn_risk_score BETWEEN 0 AND 1),
    xgboost_risk_score  REAL CHECK (xgboost_risk_score IS NULL OR xgboost_risk_score BETWEEN 0 AND 1),
    final_risk_score    REAL CHECK (final_risk_score IS NULL OR final_risk_score BETWEEN 0 AND 1),
    risk_level          TEXT NOT NULL CHECK (risk_level IN ('LOW', 'MEDIUM', 'HIGH', 'REVIEW')),
    top_risk_factors    TEXT NOT NULL CHECK (json_valid(top_risk_factors)),
    model_versions      TEXT NOT NULL CHECK (json_valid(model_versions)),
    created_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id       TEXT PRIMARY KEY,
    customer_id    TEXT NOT NULL REFERENCES customers (customer_id),
    assessment_id  TEXT NOT NULL REFERENCES risk_assessments (assessment_id),
    severity       TEXT NOT NULL CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH')),
    reason_codes   TEXT NOT NULL CHECK (json_valid(reason_codes)),
    status         TEXT NOT NULL CHECK (status IN ('OPEN', 'UNDER_REVIEW', 'CLOSED')),
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sar_drafts (
    sar_id         TEXT PRIMARY KEY,
    alert_id       TEXT NOT NULL REFERENCES alerts (alert_id),
    customer_id    TEXT NOT NULL REFERENCES customers (customer_id),
    draft_text     TEXT NOT NULL,
    evidence_refs  TEXT NOT NULL CHECK (json_valid(evidence_refs)),
    status         TEXT NOT NULL CHECK (status IN ('DRAFT', 'IN_REVIEW', 'APPROVED', 'REJECTED')),
    created_at     TEXT NOT NULL
);

-- Provenance: links every canonical ID back to its source record, and records how
-- each customer<->account link was established (SOURCE_IDENTITY vs SYNTHETIC_ASSOCIATION).
CREATE TABLE IF NOT EXISTS id_mappings (
    source_name           TEXT NOT NULL,
    source_entity_type    TEXT NOT NULL,
    source_id             TEXT NOT NULL,
    canonical_entity_type TEXT NOT NULL,
    canonical_id          TEXT NOT NULL,
    link_type             TEXT NOT NULL CHECK (link_type IN ('SOURCE_IDENTITY', 'SYNTHETIC_ASSOCIATION')),
    notes                 TEXT,
    PRIMARY KEY (source_name, source_entity_type, source_id, canonical_entity_type)
);

CREATE INDEX IF NOT EXISTS idx_accounts_customer   ON accounts (customer_id);
CREATE INDEX IF NOT EXISTS idx_tx_from             ON transactions (from_account_id);
CREATE INDEX IF NOT EXISTS idx_tx_to               ON transactions (to_account_id);
CREATE INDEX IF NOT EXISTS idx_tx_time             ON transactions (timestamp);
CREATE INDEX IF NOT EXISTS idx_kyc_customer        ON kyc_results (customer_id, created_at);
