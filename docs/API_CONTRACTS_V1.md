# KYC–AML Intelligence: Frozen Schema & API Contracts v1.0

This is the shared contract that Asmita, Saloni, and Sounak must agree on before coding independently. Once finalized, nobody should rename fields, change data types, or alter API responses without agreement from all three.

The key design decision is to separate three things:

- Unified dataset: Saloni owns its creation, versioning, customer/account mappings, and data splits.
- Model interfaces: Asmita owns KYC inference; Saloni owns GNN inference; Sounak owns XGBoost, explanations, SAR generation, and the application.
- Live application database: The application stores newly registered customers, KYC results, account links, transactions, alerts, and SAR drafts using the same IDs.

One important rule: training datasets can originate from different sources. The application connects them through a common schema; it must not pretend unrelated people in source datasets are the same real person.

## 1. Frozen architecture

Frontend — Sounak

Streamlit dashboard and live customer onboarding

FastAPI — Sounak

Authentication, orchestration, validation, shared API contracts

KYC

Asmita

GNN

Saloni

XAI + SAR

Sounak

Shared database and versioned dataset

Customers • Accounts • Documents • KYC results • Transactions • Alerts • SAR drafts

Frozen technical choices for v1: Python, FastAPI, Streamlit, Pydantic schemas, a relational database, and JSON responses. Use SQLite for local development and a shared configured database for the integrated demo. Store large images and videos as files or object storage references, not binary blobs in the customer table.

## 2. Canonical database schema

Saloni creates and maintains these tables and publishes the dataset-generation scripts. All members use the same column names and relationships.

### Table A — `customers`

One row per enrolled customer.

| Field           | Type        | Rule                                      |
| --------------- | ----------- | ----------------------------------------- |
| `customer_id`   | string      | Primary key, e.g. `C000001`               |
| `full_name`     | string      | Required for enrollment                   |
| `date_of_birth` | date / null | Optional                                  |
| `created_at`    | datetime    | UTC, ISO 8601                             |
| `customer_type` | enum        | `INDIVIDUAL`, `BUSINESS`                  |
| `data_origin`   | enum        | `SYNTHETIC`, `CONSENTED_DEMO`, `IMPORTED` |

### Table B — `accounts`

A customer may have multiple accounts.

| Field          | Type     | Rule                          |
| -------------- | -------- | ----------------------------- |
| `account_id`   | string   | Primary key, e.g. `A000001`   |
| `customer_id`  | string   | Foreign key → `customers`     |
| `account_type` | string   | E.g. `SAVINGS`, `CURRENT`     |
| `currency`     | string   | ISO currency code, e.g. `INR` |
| `created_at`   | datetime | UTC                           |

### Table C — `kyc_results`

One row per KYC verification attempt.

| Field                         | Type         | Rule                                        |
| ----------------------------- | ------------ | ------------------------------------------- |
| `kyc_id`                      | string       | Primary key                                 |
| `customer_id`                 | string       | Foreign key                                 |
| `document_authenticity_score` | float / null | Range 0–1                                   |
| `face_match_score`            | float / null | Range 0–1                                   |
| `liveness_score`              | float / null | Range 0–1                                   |
| `kyc_risk_score`              | float / null | Range 0–1                                   |
| `document_status`             | enum         | `PASS`, `REVIEW`, `FAIL`, `NOT_RUN`         |
| `face_match_status`           | enum         | `PASS`, `REVIEW`, `FAIL`, `NOT_RUN`         |
| `liveness_status`             | enum         | `PASS`, `REVIEW`, `FAIL`, `NOT_RUN`         |
| `overall_status`              | enum         | `VERIFIED`, `REVIEW`, `REJECTED`, `PENDING` |
| `model_version`               | string       | Version of deployed model                   |
| `created_at`                  | datetime     | UTC                                         |

Score convention: All component scores are normalized to 0–1. For authenticity and matching, higher means stronger evidence of authenticity/match. For `kyc_risk_score`, higher means higher risk. The KYC model owner must document the precise calculation and calibrated thresholds.

A score of `null` means not measured or unavailable; it does not mean zero risk.

### Table D — `documents`

Metadata only; store files separately.

| Field           | Type     | Rule                                      |
| --------------- | -------- | ----------------------------------------- |
| `document_id`   | string   | Primary key                               |
| `customer_id`   | string   | Foreign key                               |
| `document_type` | string   | E.g. `PASSPORT`, `ID_CARD`                |
| `storage_uri`   | string   | Private file/object reference             |
| `data_origin`   | enum     | `SYNTHETIC`, `CONSENTED_DEMO`, `IMPORTED` |
| `created_at`    | datetime | UTC                                       |

### Table E — `transactions`

| Field                 | Type           | Rule                                                 |
| --------------------- | -------------- | ---------------------------------------------------- |
| `transaction_id`      | string         | Primary key                                          |
| `from_account_id`     | string         | Foreign key → `accounts`                             |
| `to_account_id`       | string         | Foreign key → `accounts`                             |
| `amount`              | decimal        | Greater than zero                                    |
| `currency`            | string         | ISO currency code                                    |
| `timestamp`           | datetime       | UTC                                                  |
| `transaction_type`    | string         | Transfer, payment, etc.                              |
| `data_origin`         | enum           | `SYNTHETIC`, `IMPORTED`                              |
| `is_suspicious_label` | integer / null | Training/evaluation only; never an inference feature |

The transaction graph is built from account IDs and transaction edges. The associated customer is resolved through the account foreign key; do not duplicate a potentially inconsistent customer ID on every transaction.

### Table F — `risk_assessments`

One row per AML assessment.

| Field                | Type          | Rule                              |
| -------------------- | ------------- | --------------------------------- |
| `assessment_id`      | string        | Primary key                       |
| `customer_id`        | string        | Foreign key                       |
| `account_id`         | string / null | Assessed account, if applicable   |
| `kyc_risk_score`     | float / null  | Snapshot of available KYC risk    |
| `gnn_risk_score`     | float / null  | Range 0–1                         |
| `xgboost_risk_score` | float / null  | Range 0–1                         |
| `final_risk_score`   | float / null  | Range 0–1                         |
| `risk_level`         | enum          | `LOW`, `MEDIUM`, `HIGH`, `REVIEW` |
| `top_risk_factors`   | JSON array    | Human-readable reasons            |
| `model_versions`     | JSON object   | Model/version provenance          |
| `created_at`         | datetime      | UTC                               |

### Table G — `alerts`

| Field           | Type       | Rule                             |
| --------------- | ---------- | -------------------------------- |
| `alert_id`      | string     | Primary key                      |
| `customer_id`   | string     | Foreign key                      |
| `assessment_id` | string     | Foreign key                      |
| `severity`      | enum       | `LOW`, `MEDIUM`, `HIGH`          |
| `reason_codes`  | JSON array | Evidence-linked reasons          |
| `status`        | enum       | `OPEN`, `UNDER_REVIEW`, `CLOSED` |
| `created_at`    | datetime   | UTC                              |

### Table H — `sar_drafts`

| Field           | Type       | Rule                                         |
| --------------- | ---------- | -------------------------------------------- |
| `sar_id`        | string     | Primary key                                  |
| `alert_id`      | string     | Foreign key                                  |
| `customer_id`   | string     | Foreign key                                  |
| `draft_text`    | text       | Generated draft                              |
| `evidence_refs` | JSON array | IDs of supporting transactions/findings      |
| `status`        | enum       | `DRAFT`, `IN_REVIEW`, `APPROVED`, `REJECTED` |
| `created_at`    | datetime   | UTC                                          |

SARs are drafts for authorized human review, not automatically filed reports.

## 3. Shared field definitions that must never change casually

These identifiers are the backbone of integration:

| Field            | Meaning                         | Created/managed by                             |
| ---------------- | ------------------------------- | ---------------------------------------------- |
| `customer_id`    | One customer in the application | Shared registration service                    |
| `account_id`     | One banking account             | Dataset/import pipeline or banking integration |
| `transaction_id` | One financial transaction       | Dataset/import pipeline                        |
| `kyc_id`         | One KYC verification attempt    | Asmita's KYC service                           |
| `assessment_id`  | One risk assessment             | Sounak's orchestration/risk service            |
| `alert_id`       | One suspicious-activity alert   | AML workflow                                   |
| `sar_id`         | One SAR draft                   | SAR service                                    |

IDs are opaque strings. Do not assume a customer's ID equals their account ID, or that `C000001` must correspond to `A000001`. The explicit foreign-key mapping is authoritative.

For live registration, the backend creates the customer ID. For imported datasets, Saloni preserves source-to-canonical mapping in a separate mapping table. If the source datasets contain unrelated identities, no identity link is inferred merely because records have similar names or matching row numbers.

## 4. Frozen API contracts

All routes use the prefix `/api/v1`. JSON requests and responses use `snake_case`. Dates/times are ISO 8601 UTC. Errors use a consistent JSON structure.

### API 1 — Create a customer

Sounak + Saloni

`POST /api/v1/customers`

Request:

```
{
  "full_name": "Demo Customer",
  "date_of_birth": null,
  "customer_type": "INDIVIDUAL",
  "data_origin": "CONSENTED_DEMO"
}
```

Response — `201 Created`:

```
{
  "customer_id": "C000001",
  "full_name": "Demo Customer",
  "customer_type": "INDIVIDUAL",
  "data_origin": "CONSENTED_DEMO",
  "created_at": "2026-10-09T12:00:00Z"
}
```

The IDs and timestamps above are illustrative. The server generates the real ID and timestamp.

### API 2 — Run KYC

Asmita

`POST /api/v1/customers/{customer_id}/kyc`

Use `multipart/form-data` for the document image and recorded/live verification video. The backend validates the upload and forwards it to the KYC service. Live webcam capture happens in the browser/dashboard; a short captured session is then uploaded for assessment.

Inputs:

- `document_file`: image file.
- `video_file`: short video file or captured session.
- `document_type`: supported document category.

Response — `200 OK`:

```
{
  "kyc_id": "KYC000001",
  "customer_id": "C000001",
  "document_authenticity_score": 0.93,
  "face_match_score": 0.96,
  "liveness_score": 0.94,
  "kyc_risk_score": 0.08,
  "document_status": "PASS",
  "face_match_status": "PASS",
  "liveness_status": "PASS",
  "overall_status": "VERIFIED",
  "model_version": "kyc-v1"
}
```

The response illustrates the format, not a guaranteed result. Actual decisions depend on model outputs and validation-tested thresholds.

### API 3 — Register or import an account

Saloni

`POST /api/v1/accounts`

```
{
  "customer_id": "C000001",
  "account_type": "SAVINGS",
  "currency": "INR"
}
```

Response — `201 Created`:

```
{
  "account_id": "A000001",
  "customer_id": "C000001",
  "account_type": "SAVINGS",
  "currency": "INR"
}
```

For a real bank integration, account creation and ownership come from the bank's authoritative system. The demo can create simulated accounts.

### API 4 — Import or retrieve transactions

Saloni

`POST /api/v1/transactions/import`

```
{
  "transactions": [
    {
      "transaction_id": "T000001",
      "from_account_id": "A000001",
      "to_account_id": "A000002",
      "amount": 45000.0,
      "currency": "INR",
      "timestamp": "2026-10-09T10:00:00Z",
      "transaction_type": "TRANSFER",
      "data_origin": "SYNTHETIC"
    }
  ]
}
```

Response:

```
{
  "imported_count": 1,
  "rejected_count": 0,
  "errors": []
}
```

The service validates account references, transaction IDs, timestamps, currency and amounts. The import endpoint must not accept `is_suspicious_label` as a model input feature.

To retrieve a customer's transactions:

`GET /api/v1/customers/{customer_id}/transactions`

The response contains the transactions for all accounts owned by that customer.

### API 5 — Run AML analysis

Saloni + Sounak

`POST /api/v1/customers/{customer_id}/aml-analysis`

Request:

```
{
  "account_id": "A000001",
  "include_kyc_features": true
}
```

The orchestration layer loads the linked transactions, graph neighborhood, and latest available KYC result. Saloni's GNN produces the graph risk score; Sounak's XGBoost service produces its risk score and explanations.

Response:

```
{
  "assessment_id": "ASM000001",
  "customer_id": "C000001",
  "account_id": "A000001",
  "kyc_risk_score": 0.08,
  "gnn_risk_score": 0.82,
  "xgboost_risk_score": 0.79,
  "final_risk_score": 0.81,
  "risk_level": "HIGH",
  "top_risk_factors": [
    {
      "code": "RAPID_FUND_MOVEMENT",
      "description": "Funds moved rapidly through connected accounts",
      "source": "GNN"
    }
  ],
  "model_versions": {
    "gnn": "gnn-v1",
    "xgboost": "xgb-v1"
  }
}
```

These are example values. The `final_risk_score` must come from an explicitly defined, validated triage rule/model—not arbitrary averaging. If KYC is unavailable, its score is `null` and the response indicates that the feature was unavailable; it must not silently substitute zero.

### API 6 — Retrieve an alert

Sounak

`GET /api/v1/alerts/{alert_id}`

Response:

```
{
  "alert_id": "ALT000001",
  "customer_id": "C000001",
  "assessment_id": "ASM000001",
  "severity": "HIGH",
  "reason_codes": [
    "RAPID_FUND_MOVEMENT"
  ],
  "status": "OPEN"
}
```

An alert should be created according to a documented threshold or investigation rule. A high model score is a signal for review, not proof of criminal activity.

### API 7 — Generate a SAR draft

Sounak

`POST /api/v1/alerts/{alert_id}/sar-draft`

Request:

```
{
  "include_transaction_evidence": true
}
```

Response:

```
{
  "sar_id": "SAR000001",
  "alert_id": "ALT000001",
  "customer_id": "C000001",
  "status": "DRAFT",
  "draft_text": "Draft narrative based on the validated evidence...",
  "evidence_refs": [
    "T000001"
  ]
}
```

The generated narrative must be grounded in stored evidence. Unsupported claims must not be invented, and a human reviewer must approve the draft before any formal submission.

### API 8 — Health checks

Sounak

`GET /api/v1/health`

```
{
  "status": "ok",
  "api_version": "v1",
  "models": {
    "kyc": "loaded",
    "gnn": "loaded",
    "xgboost": "loaded"
  }
}
```

Use `loaded`, `unavailable`, or `error` for model states. The health endpoint should reflect the actual application state.

## 5. Standard error response

Every endpoint should return the same error format:

```
{
  "error": {
    "code": "CUSTOMER_NOT_FOUND",
    "message": "The requested customer does not exist.",
    "request_id": "req-example-001"
  }
}
```

Use appropriate HTTP statuses:

| Status | Meaning                              |
| ------ | ------------------------------------ |
| `200`  | Successful read or analysis          |
| `201`  | Resource created                     |
| `400`  | Invalid request or file              |
| `404`  | Customer, account or alert not found |
| `409`  | Duplicate/conflicting resource       |
| `422`  | Schema validation failed             |
| `500`  | Unexpected server error              |
| `503`  | Required model/service unavailable   |

## 6. Dataset versioning and model evaluation contract

Saloni must publish a manifest for every dataset release, for example `dataset_v1`.

It must specify:

- Source dataset names and versions, licensing and provenance.
- Number of customers, accounts, transactions, documents and videos.
- Canonical ID mappings and how links were established.
- Which records are synthetic, imported or consented demo data.
- Feature definitions, missing-value conventions and label meanings.
- Train/validation/test splits and leakage-prevention rules.

Evaluation rule: Ground-truth AML labels must never be included as inference features. Use held-out evaluation and report precision, recall, F1, PR-AUC, ROC-AUC and confusion matrices where applicable. For KYC, report document, face-match and genuine-versus-spoof liveness metrics separately.

The models' training records need not all refer to the same real people. The application registry and test/evaluation datasets must have documented, valid relationships. The team must not manufacture matching identities between unrelated source records.

## 7. What all three should do now

Step 1 — Freeze these contracts

Save this specification as `docs/API_CONTRACTS_V1.md`. Agree on any changes before implementation begins.

Step 2 — Saloni creates the database and fixture records

Implement the schema, foreign keys, migrations, and small versioned fixture dataset. Publish the dataset-generation plan and source manifest.

Step 3 — Asmita implements the KYC endpoint

Return the exact KYC response schema even if some optional scores are unavailable during early development.

Step 4 — Sounak implements the orchestration and application

Create the API skeleton, request/response models, dashboard and fixture-based integration tests.

Step 5 — Test the contracts before the models finish

Every service must pass its contract tests with fixtures before being integrated with actual model inference.

Final rule: These field names, types, enum values, endpoint paths, and response structures are the team's proposed v1 contract. Treat them as frozen after all three members approve and commit them. Model internals can evolve independently as long as they continue to satisfy these interfaces.