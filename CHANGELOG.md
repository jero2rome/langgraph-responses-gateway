# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-08-17

### Added
- Full OpenAI Responses API spec compliance with correct `input` parameter
- Conversation chaining support via `previous_response_id`
- Response storage with `store` parameter for conversation continuity
- Proper token usage extraction from LangGraph `response_metadata`
- Multi-part input support for array of input parts
- Error event handling in SSE streams
- Comprehensive test examples for all features

### Changed
- **BREAKING**: `model` parameter is now required per OpenAI spec
- Primary input parameter changed from `messages` to `input` (maintains backward compatibility)
- SSE event names corrected to match OpenAI spec exactly:
  - `response.text.delta` → `response.output_text.delta`
  - Added proper `response.output_item.added` and `response.output_item.done` events
- Non-streaming response structure now matches OpenAI spec with `object: "response"`
- Token usage now extracted from LangGraph AIMessage metadata when available

### Fixed
- Correct SSE event ordering and structure
- Proper usage tracking with fallback to estimation
- Response format compliance with OpenAI Responses API

## [0.1.0] - 2025-08-17

### Added
- Initial release
- Basic gateway implementation
- FastAPI-based server
- SSE streaming support
- Non-streaming response support
- Health check endpoint
- Models listing endpoint
- Basic examples with LangGraph ReAct agent

[0.2.0]: https://github.com/jero2rome/langgraph-responses-gateway/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/jero2rome/langgraph-responses-gateway/releases/tag/v0.1.0