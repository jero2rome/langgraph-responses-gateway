# LangGraph Responses Gateway

[![GitHub](https://img.shields.io/github/license/jero2rome/langgraph-responses-gateway)](https://github.com/jero2rome/langgraph-responses-gateway/blob/main/LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/jero2rome/langgraph-responses-gateway)](https://github.com/jero2rome/langgraph-responses-gateway)
[![GitHub Issues](https://img.shields.io/github/issues/jero2rome/langgraph-responses-gateway)](https://github.com/jero2rome/langgraph-responses-gateway/issues)
[![GitHub Actions](https://github.com/jero2rome/langgraph-responses-gateway/actions/workflows/test.yml/badge.svg)](https://github.com/jero2rome/langgraph-responses-gateway/actions)

Bridge any LangGraph agent to OpenAI's Responses API format with zero configuration.

## Overview

`langgraph-responses-gateway` is a lightweight Python package that exposes any LangGraph agent or multi-agent system as an OpenAI Responses API endpoint. This enables seamless integration with modern AI clients like Vercel AI SDK, OpenAI's client libraries, and any tool that supports the Responses API format.

## Why This Package?

The AI ecosystem has a gap: while LangGraph provides powerful agent orchestration capabilities, and OpenAI's Responses API has become the de facto standard for AI interactions, there's no simple way to bridge the two. This package fills that gap.

### Key Benefits

- **Zero Configuration**: Works with any LangGraph CompiledGraph out of the box
- **Full Streaming Support**: Native SSE streaming with correct OpenAI event names
- **Conversation Chaining**: Support for `previous_response_id` to maintain context
- **OpenAI Spec Compliant**: Uses `input` parameter and exact event structure
- **Vercel AI SDK Compatible**: Seamless integration with modern web frameworks
- **Backward Compatible**: Still accepts `messages` for easy migration
- **Production Ready**: Built on FastAPI with robust error handling

## Installation

```bash
pip install langgraph-responses-gateway
```

Or with uv:

```bash
uv add langgraph-responses-gateway
```

## Quick Start

Transform your LangGraph agent into an OpenAI-compatible API in just 3 lines:

```python
from langgraph_responses_gateway import ResponsesGateway
from your_agent import create_agent_graph

# 1. Create your LangGraph
graph = create_agent_graph()

# 2. Wrap it as Responses API
gateway = ResponsesGateway(graph, name="My Agent")

# 3. Run the server
import uvicorn
uvicorn.run(gateway.app, host="0.0.0.0", port=8000)
```

Your agent is now accessible at `http://localhost:8000/v1/responses`!

## Usage Examples

### Basic Chat Request

```python
import httpx

# Using 'input' (OpenAI Responses API spec)
response = httpx.post(
    "http://localhost:8000/v1/responses",
    json={
        "model": "langgraph-agent",  # Required
        "input": "Hello!",
        "stream": False
    }
)
print(response.json()["output"][0]["content"][0]["text"])
```

### Streaming with SSE

```python
import httpx
import json

with httpx.stream("POST", "http://localhost:8000/v1/responses",
                  json={
                      "model": "langgraph-agent",
                      "input": "Tell me a story",
                      "stream": True
                  }) as r:
    for line in r.iter_lines():
        if line.startswith("data: "):
            event = json.loads(line[6:])
            if event["type"] == "response.output_text.delta":
                print(event["delta"], end="", flush=True)
```

### Conversation Chaining

```python
# First message
response1 = httpx.post(
    "http://localhost:8000/v1/responses",
    json={
        "model": "langgraph-agent",
        "input": "My name is Alice",
        "store": True,  # Store for chaining
        "stream": False
    }
)
response_id = response1.json()["id"]

# Chained follow-up
response2 = httpx.post(
    "http://localhost:8000/v1/responses",
    json={
        "model": "langgraph-agent",
        "input": "What's my name?",
        "previous_response_id": response_id,
        "stream": False
    }
)
# Will remember context from previous message
```

### With Vercel AI SDK

```typescript
import { openai } from '@ai-sdk/openai'
import { streamText } from 'ai'

const result = await streamText({
  model: openai('langgraph-agent'),
  baseURL: 'http://localhost:8000/v1',
  prompt: 'Hello!',  // Vercel AI SDK uses prompt
})

for await (const chunk of result.textStream) {
  console.log(chunk)
}
```

### Advanced Configuration

```python
from langgraph_responses_gateway import ResponsesGateway

gateway = ResponsesGateway(
    graph=your_graph,
    name="Advanced Agent",
    version="2.0.0",
    base_path="/api/v2",  # Custom base path
    model_name="my-custom-model"  # Model name reported to clients
)

# Access additional request data
class CustomGateway(ResponsesGateway):
    def _prepare_graph_input(self, user_input: str, req, previous_context=None):
        messages = []
        if previous_context:  # Continue conversation
            messages.extend(previous_context["messages"])
        messages.append({"role": "user", "content": user_input})
        
        return {
            "messages": messages,
            "thread_id": req.thread_id,  # Thread management
            "user_id": req.user_id,      # User isolation
            "metadata": req.metadata,     # Custom metadata
        }
```

## API Endpoints

The gateway exposes the following endpoints:

### POST `/v1/responses`
Create a response from the agent. Supports both streaming and non-streaming modes.

**Request Body:**
```json
{
  "model": "langgraph-agent",              // Required
  "input": "Your message",                // Or array of input parts
  "stream": true,                         // Enable SSE streaming
  "previous_response_id": "resp_xxx",     // Chain conversations
  "store": true,                          // Store for chaining
  "thread_id": "optional-thread-id",      // Thread management
  "user_id": "optional-user-id",          // User isolation
  "metadata": {}                          // Custom data
}
```

**Backward Compatibility:**
The gateway still accepts `messages` array for easy migration:
```json
{
  "model": "langgraph-agent",
  "messages": [{"role": "user", "content": "Your message"}],
  "stream": false
}
```

### GET `/v1/models`
List available models (returns your configured model name).

### GET `/health`
Health check endpoint for monitoring.

## Customization

### Custom Content Extraction

Override the content extraction methods to handle your specific LangGraph output format:

```python
class MyGateway(ResponsesGateway):
    def _extract_content_from_step(self, step):
        # Custom logic for your graph's streaming format
        if "my_custom_field" in step:
            return step["my_custom_field"]
        return super()._extract_content_from_step(step)
```

### Custom Input Preparation

Customize how user input is prepared for your graph:

```python
class MyGateway(ResponsesGateway):
    def _prepare_graph_input(self, user_input: str, req, previous_context=None):
        return {
            "query": user_input,
            "history": previous_context.get("messages", []) if previous_context else [],
            "context": req.metadata.get("context", {}),
            "config": {"temperature": req.temperature or 0.7}
        }
```

## Architecture

The gateway works by:

1. **Input Translation**: Converting OpenAI Responses API requests to LangGraph input format
2. **Graph Execution**: Running your LangGraph with the prepared input
3. **Output Streaming**: Converting LangGraph's streaming output to SSE events
4. **Format Compliance**: Ensuring all responses match OpenAI's Responses API specification

## Requirements

- Python 3.9+
- LangGraph 0.5.0+
- FastAPI 0.100.0+
- Pydantic 2.0.0+

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/jero2rome/langgraph-responses-gateway
cd langgraph-responses-gateway

# Install with development dependencies
uv add --dev pytest pytest-asyncio pytest-cov black ruff mypy

# Run tests
uv run pytest

# Format code
uv run black src/
uv run ruff check --fix src/
```

### Running Tests

```bash
# Run all tests
uv run pytest

# With coverage
uv run pytest --cov=langgraph_responses_gateway

# Specific test file
uv run pytest tests/test_gateway.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Created by [Jerome Mohanan](https://github.com/jero2rome)

## Acknowledgments

- Built on top of [LangGraph](https://github.com/langchain-ai/langgraph) by LangChain
- Implements [OpenAI's Responses API](https://platform.openai.com/docs/api-reference/responses) specification
- Inspired by the need for better LangGraph ↔ OpenAI compatibility

## Support

- **Issues**: [GitHub Issues](https://github.com/jero2rome/langgraph-responses-gateway/issues)
- **Discussions**: [GitHub Discussions](https://github.com/jero2rome/langgraph-responses-gateway/discussions)
- **Documentation**: [Read the Docs](https://langgraph-responses-gateway.readthedocs.io) (coming soon)

## Roadmap

- [ ] Tool calling support (convert LangGraph tools to Responses API format)
- [ ] Reasoning output for complex agent traces  
- [ ] Automatic token counting with tiktoken
- [ ] Built-in authentication and rate limiting
- [ ] Support for other agent frameworks (CrewAI, AutoGen)
- [ ] WebSocket support for bidirectional streaming
- [ ] OpenTelemetry instrumentation

## Related Projects

- [LangGraph](https://github.com/langchain-ai/langgraph) - Build robust agents with LLMs
- [Vercel AI SDK](https://sdk.vercel.ai) - Build AI-powered web applications
- [FastAPI](https://fastapi.tiangolo.com) - Modern web framework for building APIs