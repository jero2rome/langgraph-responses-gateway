"""Test the service implementation."""

import pytest

from src.langgraph_responses_gateway import (
    ResponsesGatewayService,
    ResponsesRequest,
)


class MockGraph:
    """Mock LangGraph for testing."""

    async def ainvoke(self, input_data, config=None):
        """Mock ainvoke method."""
        return {
            "messages": [
                {"role": "assistant", "content": "Test response from mock graph"}
            ]
        }

    async def astream(self, input_data, config=None, stream_mode="updates"):
        """Mock astream method."""
        yield {
            "agent": {
                "messages": [
                    {"role": "assistant", "content": "Streaming test response"}
                ]
            }
        }


@pytest.mark.asyncio
async def test_service_initialization():
    """Test that the service can be initialized without web dependencies."""
    graph = MockGraph()
    service = ResponsesGatewayService(
        graph=graph, model_name="test-model", name="Test Agent", version="1.0.0"
    )

    assert service.graph == graph
    assert service.model_name == "test-model"
    assert service.name == "Test Agent"
    assert service.version == "1.0.0"


@pytest.mark.asyncio
async def test_process_request():
    """Test processing a non-streaming request."""
    graph = MockGraph()
    service = ResponsesGatewayService(graph=graph)

    request = ResponsesRequest(model="test-model", input="Test input", stream=False)

    response = await service.process_request(request)

    assert response["object"] == "response"
    assert response["status"] == "completed"
    assert response["model"] == "test-model"
    assert len(response["output"]) == 1
    assert response["output"][0]["type"] == "message"
    assert response["output"][0]["role"] == "assistant"
    assert "Test response from mock graph" in str(response["output"][0]["content"])
    assert "usage" in response
    assert "prompt_tokens" in response["usage"]
    assert "completion_tokens" in response["usage"]
    assert "total_tokens" in response["usage"]


@pytest.mark.asyncio
async def test_stream_response():
    """Test streaming a response."""
    graph = MockGraph()
    service = ResponsesGatewayService(graph=graph)

    request = ResponsesRequest(
        model="test-model", input="Test streaming input", stream=True
    )

    events = []
    async for event in service.stream_response(request):
        events.append(event)

    # Should have multiple SSE events
    assert len(events) > 0

    # Check for required event types
    event_types = []
    for event in events:
        if event.startswith("data: "):
            import json

            data = json.loads(event[6:].strip())
            if "type" in data:
                event_types.append(data["type"])

    assert "response.created" in event_types
    assert "response.output_item.added" in event_types
    assert "response.completed" in event_types


@pytest.mark.asyncio
async def test_request_validation():
    """Test request validation."""
    graph = MockGraph()
    service = ResponsesGatewayService(graph=graph)

    # Test missing model
    request = ResponsesRequest(input="Test input")
    with pytest.raises(ValueError, match="model is required"):
        await service.process_request(request)

    # Test missing input
    request = ResponsesRequest(model="test-model")
    with pytest.raises(ValueError, match="No input found"):
        await service.process_request(request)


@pytest.mark.asyncio
async def test_conversation_chaining():
    """Test conversation chaining with previous_response_id."""
    graph = MockGraph()
    service = ResponsesGatewayService(graph=graph)

    # First request
    request1 = ResponsesRequest(model="test-model", input="First message", store=True)

    response1 = await service.process_request(request1)
    response_id = response1["id"]

    # Second request with chaining
    request2 = ResponsesRequest(
        model="test-model", input="Second message", previous_response_id=response_id
    )

    response2 = await service.process_request(request2)
    assert response2["status"] == "completed"
