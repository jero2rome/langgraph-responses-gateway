"""Integration tests for LangGraph Responses Gateway with real agents."""

import json
import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from langgraph_responses_gateway import ResponsesGateway
from langgraph_responses_gateway.gateway import ResponsesRequest


class AgentState(TypedDict):
    """State for test agent."""

    messages: list = add_messages


@pytest.fixture
def simple_graph():
    """Create a simple graph for testing."""

    def agent_node(state):
        """Simple agent that echoes messages."""
        messages = state.get("messages", [])
        if messages:
            last_msg = messages[-1]
            response = f"Echo: {last_msg.content if hasattr(last_msg, 'content') else str(last_msg)}"
            return {"messages": [AIMessage(content=response)]}
        return {"messages": []}

    # Build graph
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", END)

    return workflow.compile()


class TestIntegrationWithRealAgent:
    """Integration tests with actual LangGraph agents."""

    @pytest.mark.asyncio
    async def test_simple_graph_integration(self, simple_graph):
        """Test with a simple echo graph."""
        gateway = ResponsesGateway(simple_graph, name="Echo Agent")

        req = ResponsesRequest(model="echo-model", input="Hello, world!", stream=False)

        response = await gateway._create_response(
            {"messages": [HumanMessage(content="Hello, world!")]}, req
        )

        assert response.status_code == 200
        data = json.loads(response.body)
        assert "output" in data
        assert "Echo: Hello, world!" in data["output"][0]["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_streaming_integration(self, simple_graph):
        """Test streaming with a real graph."""
        gateway = ResponsesGateway(simple_graph, name="Stream Agent")

        req = ResponsesRequest(
            model="stream-model", input="Stream this message", stream=True
        )

        events = []
        async for event_data in gateway._stream_response(
            {"messages": [HumanMessage(content="Stream this message")]}, req
        ):
            if event_data.startswith("data: "):
                event = json.loads(event_data[6:])
                events.append(event["type"])

        # Verify event sequence
        assert "response.created" in events
        assert "response.output_item.added" in events
        assert "response.completed" in events

    @pytest.mark.asyncio
    async def test_conversation_storage(self, simple_graph):
        """Test conversation response storage."""
        gateway = ResponsesGateway(simple_graph, name="Chain Agent")

        # First message
        req1 = ResponsesRequest(
            model="chain-model", input="First message", store=True, stream=False
        )

        response1 = await gateway._create_response(
            {"messages": [HumanMessage(content="First message")]}, req1
        )

        assert response1.status_code == 200
        data1 = json.loads(response1.body)
        resp_id = data1["id"]

        # Verify stored
        assert resp_id in gateway._response_store
        stored = gateway._response_store[resp_id]
        assert len(stored["messages"]) >= 2  # At least user + assistant
        assert stored["messages"][-1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_thread_management(self, simple_graph):
        """Test thread ID management for LangGraph checkpointer."""
        gateway = ResponsesGateway(simple_graph, name="Thread Agent")

        req = ResponsesRequest(
            model="thread-model",
            input="Test message",
            thread_id="test-thread-123",
            user="test-user",
            stream=False,
        )

        # Prepare config
        config = gateway._prepare_langgraph_config({"messages": []}, req)

        # Verify config structure
        assert "configurable" in config
        assert config["configurable"]["thread_id"] == "test-user:test-thread-123"
        assert config["recursion_limit"] == 150

    @pytest.mark.asyncio
    async def test_sdk_parameters_handling(self, simple_graph):
        """Test handling of all SDK parameters."""
        gateway = ResponsesGateway(simple_graph, name="SDK Agent")

        req = ResponsesRequest(
            model="sdk-model",
            input="Test input",
            instructions="Be helpful",
            temperature=0.7,
            top_p=0.9,
            max_output_tokens=100,
            truncation="auto",
            service_tier="default",
            user="user-123",
            include=["message.output_text.logprobs"],
            metadata={"session": "test"},
            stream=False,
        )

        # Prepare graph input
        graph_input = gateway._prepare_graph_input("Test input", req)

        # Verify instructions are added as system message
        assert len(graph_input["messages"]) == 2
        assert graph_input["messages"][0]["role"] == "system"
        assert graph_input["messages"][0]["content"] == "Be helpful"

        # Verify config
        config = gateway._prepare_langgraph_config(graph_input, req)
        # Check that thread_id contains the user prefix
        assert config["configurable"]["thread_id"].startswith("user-123:chat-")
        assert config["metadata"] == {"session": "test"}

    @pytest.mark.asyncio
    async def test_error_recovery(self, simple_graph):
        """Test error handling and recovery."""
        gateway = ResponsesGateway(simple_graph, name="Error Agent")

        # Make graph raise an error
        async def error_invoke(*args, **kwargs):
            raise Exception("Test error")

        simple_graph.ainvoke = error_invoke

        req = ResponsesRequest(model="error-model", input="Cause error", stream=False)

        response = await gateway._create_response(
            {"messages": [HumanMessage(content="Cause error")]}, req
        )

        assert response.status_code == 500
        data = json.loads(response.body)
        assert "error" in data
        assert "Test error" in data["error"]["message"]


class TestResponseStorage:
    """Test response storage and cleanup."""

    def test_storage_cleanup(self):
        """Test that old responses are cleaned up."""
        from unittest.mock import MagicMock

        graph = MagicMock()
        gateway = ResponsesGateway(graph)

        # Add old response (2 hours ago)
        old_id = "resp_old"
        gateway._response_store[old_id] = {
            "messages": [],
            "timestamp": time.time() - 7200,
        }

        # Add recent response
        new_id = "resp_new"
        gateway._response_store[new_id] = {"messages": [], "timestamp": time.time()}

        # Trigger cleanup
        gateway._store_response("resp_trigger", {"messages": []}, "output")

        # Verify cleanup
        assert old_id not in gateway._response_store
        assert new_id in gateway._response_store

    def test_conversation_context_storage(self):
        """Test that conversation context is properly stored."""
        from unittest.mock import MagicMock

        graph = MagicMock()
        gateway = ResponsesGateway(graph)

        # Store response with context
        resp_id = "resp_test"
        gateway._store_response(
            resp_id,
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                ],
                "thread_id": "thread-123",
                "user_id": "user-456",
            },
            "New response",
        )

        # Verify stored correctly
        stored = gateway._response_store[resp_id]
        # The messages list should have the new response appended
        assert len(stored["messages"]) == 3
        assert stored["messages"][-1]["content"] == "New response"
        assert stored["messages"][-1]["role"] == "assistant"
        assert stored["thread_id"] == "thread-123"
        assert stored["user_id"] == "user-456"
