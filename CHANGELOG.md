# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-08-17

### Added
- Full OpenAI Responses API spec compliance based on openai-agents-python SDK
- Dual conversation management mechanisms:
  - `previous_response_id`: OpenAI's stateless conversation continuation
  - `thread_id` + `user`: LangGraph's stateful checkpointer-based management
- Response storage with `store` parameter for conversation continuity
- System instructions support via `instructions` parameter
- Proper token usage extraction from LangGraph `response_metadata`
- Multi-part input support for array of input parts
- Error event handling in SSE streams
- Generation parameters: `temperature`, `top_p`, `max_output_tokens`
- Additional SDK parameters: `truncation`, `service_tier`, `user`, `include`
- Comprehensive test examples for all features

### Changed
- **BREAKING**: `model` parameter is now required per OpenAI SDK
- **BREAKING**: Removed backward compatibility with `messages` - gateway now only accepts `input`
- Gateway now purely translates from OpenAI Responses API to LangGraph format
- SSE event names confirmed to match SDK exactly:
  - Uses `response.output_text.delta` (correct per SDK)
  - Proper event sequence: `response.created` → `response.output_item.added` → `response.output_text.delta` → `response.output_item.done` → `response.completed`
- Non-streaming response structure matches OpenAI SDK with `object: "response"`
- Token usage extracted from LangGraph AIMessage metadata when available
- Proper LangGraph config with composite thread ID (`{user}:{thread_id}`)

### Fixed
- Correct SSE event ordering and structure per SDK
- Proper usage tracking with fallback to estimation
- Full compliance with openai-agents-python SDK types
- Correct mapping between OpenAI concepts and LangGraph's thread management

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