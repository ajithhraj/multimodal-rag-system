# Changelog

All notable changes to this project are documented here.

## [0.1.1] - 2026-04-07

### Added
- Multimodal query support via CLI image input: `mmrag ask "..." --image <path>`
- New `POST /query-multimodal` API endpoint for question + optional image multipart queries

### Improved
- Citation-rich query responses are preserved across both text-only and multimodal query paths
- README usage examples updated for multimodal querying
