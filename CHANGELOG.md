# Changelog

All notable changes to this project are documented here.

## [0.1.2] - 2026-04-16

### Added
- Background ingestion jobs API:
  - `POST /ingest-jobs` to queue ingestion
  - `GET /ingest-jobs/{job_id}` to inspect state
  - `GET /ingest-jobs` to list recent jobs
- In-memory job manager with lifecycle states (`queued`, `running`, `completed`, `failed`)
- Config knobs for async ingestion workers, retention, and TTL

### Improved
- Validation for ingestion paths requests now enforces non-empty `paths`
- API tests now cover async ingestion job lifecycle and disabled-mode behavior

## [0.1.1] - 2026-04-07

### Added
- Multimodal query support via CLI image input: `mmrag ask "..." --image <path>`
- New `POST /query-multimodal` API endpoint for question + optional image multipart queries

### Improved
- Citation-rich query responses are preserved across both text-only and multimodal query paths
- README usage examples updated for multimodal querying
