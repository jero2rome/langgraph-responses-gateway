"""Comprehensive tests for LangGraph Responses Gateway."""

import json
import time
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from langgraph_responses_gateway import ResponsesGateway


@pytest.fixture
def mock_graph():
    """Create a mock LangGraph for testing."""
    graph = MagicMock()

    # Mock AIMessage with token usage
    mock_message = MagicMock()
    mock_message.content = "Test response from agent"
    mock_message.response_metadata = {
        "token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    }

    # Set up async methods
    graph.ainvoke = AsyncMock(return_value={"messages": [mock_message]})

    # Mock streaming
    async def mock_stream(*args, **kwargs):
        # Yield updates
        yield {"agent": {"messages": [mock_message]}}

    graph.astream = mock_stream

    return graph


@pytest.fixture
def gateway(mock_graph):
    """Create a gateway instance with mock graph."""
    return ResponsesGateway(mock_graph, name="Test Gateway", model_name="test-model")


@pytest.fixture
def client(gateway):
    """Create a test client."""
    return TestClient(gateway.app)


class TestBasicRequests:
    """Test basic request handling."""

    def test_health_check(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["agent"] == "Test Gateway"

    def test_list_models(self, client):
        """Test models listing."""
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "test-model"

    def test_missing_model_parameter(self, client):
        """Test that model parameter is required."""
        response = client.post("/v1/responses", json={"input": "Hello"})
        assert response.status_code == 400
        assert "model is required" in response.json()["detail"]

    def test_missing_input_parameter(self, client):
        """Test that input is required."""
        response = client.post("/v1/responses", json={"model": "test-model"})
        assert response.status_code == 400
        assert "No input found" in response.json()["detail"]


class TestInputFormats:
    """Test different input formats."""

    def test_string_input(self, client):
        """Test simple string input."""
        response = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "Hello, how are you?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "response"
        assert data["output"][0]["content"][0]["text"] == "Test response from agent"

    def test_array_input(self, client):
        """Test array of input parts."""
        response = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": [
                    {"type": "input_text", "text": "Part 1"},
                    {"type": "input_text", "text": "Part 2"},
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "response"

    def test_system_instructions(self, client):
        """Test system instructions parameter."""
        response = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "Hello",
                "instructions": "You are a helpful assistant",
            },
        )
        assert response.status_code == 200


class TestConversationManagement:
    """Test conversation continuation features."""

    def test_store_and_retrieve_response(self, client, gateway):
        """Test storing response for continuation."""
        # First request with store=true
        response1 = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "First message", "store": True},
        )
        assert response1.status_code == 200
        resp_id = response1.json()["id"]

        # Check response was stored
        assert resp_id in gateway._response_store
        stored = gateway._response_store[resp_id]
        assert len(stored["messages"]) > 0
        assert stored["messages"][-1]["content"] == "Test response from agent"

    def test_previous_response_id(self, client, gateway):
        """Test conversation continuation with previous_response_id."""
        # Store a response first
        resp_id = f"resp_{uuid.uuid4().hex}"
        gateway._response_store[resp_id] = {
            "messages": [
                {"role": "user", "content": "First message"},
                {"role": "assistant", "content": "First response"},
            ],
            "timestamp": time.time(),
        }

        # Continue conversation
        response = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "Second message",
                "previous_response_id": resp_id,
            },
        )
        assert response.status_code == 200

    def test_thread_management(self, client, mock_graph):
        """Test thread_id and user parameters for LangGraph."""
        response = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "Hello",
                "thread_id": "test-thread",
                "user": "test-user",
            },
        )
        assert response.status_code == 200

        # Verify the graph was called with proper config
        mock_graph.ainvoke.assert_called_once()
        call_args = mock_graph.ainvoke.call_args
        config = call_args[1]["config"]
        assert "configurable" in config
        assert config["configurable"]["thread_id"] == "test-user:test-thread"


class TestTokenUsage:
    """Test token usage tracking."""

    def test_extract_token_usage_from_metadata(self, client):
        """Test extracting token usage from response_metadata."""
        response = client.post(
            "/v1/responses", json={"model": "test-model", "input": "Hello"}
        )
        assert response.status_code == 200
        data = response.json()

        # Check usage is included
        assert "usage" in data
        usage = data["usage"]
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 5
        assert usage["total_tokens"] == 15

    def test_fallback_token_estimation(self, client, mock_graph):
        """Test fallback to estimation when no metadata."""
        # Remove token usage from mock
        mock_message = MagicMock()
        mock_message.content = "Response"
        mock_message.response_metadata = {}  # No token usage

        mock_graph.ainvoke = AsyncMock(return_value={"messages": [mock_message]})

        response = client.post(
            "/v1/responses", json={"model": "test-model", "input": "Hello"}
        )
        assert response.status_code == 200
        data = response.json()

        # Should have estimated tokens
        assert "usage" in data
        assert data["usage"]["total_tokens"] > 0


class TestStreamingResponse:
    """Test SSE streaming responses."""

    @pytest.mark.asyncio
    async def test_streaming_events(self, gateway):
        """Test that streaming produces correct event sequence."""
        from langgraph_responses_gateway.gateway import ResponsesRequest

        req = ResponsesRequest(model="test-model", input="Hello", stream=True)

        events = []
        async for event_data in gateway._stream_response(
            {"messages": [{"role": "user", "content": "Hello"}]}, req
        ):
            # Parse SSE format
            if event_data.startswith("data: "):
                event = json.loads(event_data[6:])
                events.append(event["type"])

        # Check event sequence matches SDK
        assert "response.created" in events
        assert "response.output_item.added" in events
        assert "response.output_text.delta" in events
        assert "response.output_item.done" in events
        assert "response.completed" in events

    def test_stream_endpoint(self, client):
        """Test streaming via HTTP endpoint."""
        with client.stream(
            "POST",
            "/v1/responses",
            json={"model": "test-model", "input": "Hello", "stream": True},
        ) as response:
            assert response.status_code == 200
            assert (
                response.headers["content-type"] == "text/event-stream; charset=utf-8"
            )

            # Read some events
            events = []
            for line in response.iter_lines():
                if line.startswith("data: "):
                    event = json.loads(line[6:])
                    events.append(event["type"])
                    if event["type"] == "response.completed":
                        break

            assert len(events) > 0
            assert "response.output_text.delta" in events


class TestSDKParameters:
    """Test additional SDK parameters."""

    def test_generation_parameters(self, client):
        """Test temperature, top_p, max_output_tokens."""
        response = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "Hello",
                "temperature": 0.7,
                "top_p": 0.9,
                "max_output_tokens": 1000,
            },
        )
        assert response.status_code == 200

    def test_additional_sdk_parameters(self, client):
        """Test truncation, service_tier, user, include."""
        response = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "Hello",
                "truncation": "auto",
                "service_tier": "default",
                "user": "user-123",
                "include": ["message.output_text.logprobs"],
            },
        )
        assert response.status_code == 200

    def test_metadata_parameter(self, client):
        """Test custom metadata."""
        response = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "Hello",
                "metadata": {"custom": "value"},
            },
        )
        assert response.status_code == 200


class TestErrorHandling:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_graph_error_handling(self, gateway, mock_graph):
        """Test handling of graph execution errors."""
        from langgraph_responses_gateway.gateway import ResponsesRequest

        # Make graph raise an error
        mock_graph.ainvoke = AsyncMock(side_effect=Exception("Graph error"))

        req = ResponsesRequest(model="test-model", input="Hello")

        response = await gateway._create_response(
            {"messages": [{"role": "user", "content": "Hello"}]}, req
        )

        assert response.status_code == 500
        body = json.loads(response.body)
        assert "error" in body
        assert "Graph error" in body["error"]["message"]

    @pytest.mark.asyncio
    async def test_streaming_error_handling(self, gateway, mock_graph):
        """Test error handling in streaming mode."""
        from langgraph_responses_gateway.gateway import ResponsesRequest

        # Make stream raise an error
        async def error_stream(*args, **kwargs):
            yield {"agent": {"messages": []}}
            raise Exception("Stream error")

        mock_graph.astream = error_stream

        req = ResponsesRequest(model="test-model", input="Hello", stream=True)

        events = []
        async for event_data in gateway._stream_response(
            {"messages": [{"role": "user", "content": "Hello"}]}, req
        ):
            if event_data.startswith("data: "):
                event = json.loads(event_data[6:])
                events.append(event)

        # Should have error event
        error_events = [e for e in events if e["type"] == "error"]
        assert len(error_events) > 0
        assert "Stream error" in error_events[0]["error"]["message"]


class TestResponseStorage:
    """Test response storage and cleanup."""

    def test_response_cleanup(self, gateway):
        """Test old responses are cleaned up."""
        # Add old response
        old_id = "resp_old"
        gateway._response_store[old_id] = {
            "messages": [],
            "timestamp": time.time() - 7200,  # 2 hours ago
        }

        # Add recent response
        new_id = "resp_new"
        gateway._response_store[new_id] = {"messages": [], "timestamp": time.time()}

        # Trigger cleanup by storing another response
        gateway._store_response("resp_trigger", {"messages": []}, "output")

        # Old should be removed, new should remain
        assert old_id not in gateway._response_store
        assert new_id in gateway._response_store

    def test_store_with_thread_context(self, gateway):
        """Test storing thread context with response."""
        resp_id = "resp_test"
        gateway._store_response(
            resp_id,
            {
                "messages": [{"role": "user", "content": "Hi"}],
                "thread_id": "thread-123",
                "user_id": "user-456",
            },
            "Hello",
        )

        stored = gateway._response_store[resp_id]
        assert stored["thread_id"] == "thread-123"
        assert stored["user_id"] == "user-456"
        assert len(stored["messages"]) == 2
        assert stored["messages"][-1]["content"] == "Hello"
