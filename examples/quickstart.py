#!/usr/bin/env python
"""
Quick Start Example for LangGraph Responses Gateway

This example shows how to expose any LangGraph agent as an OpenAI Responses API endpoint.
"""

import asyncio
import json

import httpx
import uvicorn
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from langgraph_responses_gateway import ResponsesGateway

# ============================================
# Step 1: Create Your LangGraph Agent
# ============================================


# Define some tools for the agent
@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Search results for: {query}"


@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"The result is {result}"
    except:
        return "Invalid expression"


# Create a ReAct agent with tools
def create_agent():
    """Create a ReAct agent with tools."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [search, calculate]
    return create_react_agent(llm, tools)


# ============================================
# Step 2: Wrap Your Agent as Responses API
# ============================================


def run_server():
    """Run the gateway server."""
    # Create your agent
    agent = create_agent()

    # Wrap it as an OpenAI Responses API
    gateway = ResponsesGateway(agent, name="My LangGraph Agent", model_name="my-agent")

    # Run the server
    print(" Starting server on http://localhost:8000")
    print(" API docs available at http://localhost:8000/docs")
    uvicorn.run(gateway.app, host="0.0.0.0", port=8000)


# ============================================
# Step 3: Test Your API
# ============================================


async def test_api():
    """Test the API with various examples."""
    BASE_URL = "http://localhost:8000"

    print("\n" + "=" * 50)
    print("Testing LangGraph Responses Gateway")
    print("=" * 50)

    # Test 1: Basic Request
    print("\n1. Basic Request:")
    response = httpx.post(
        f"{BASE_URL}/v1/responses",
        json={"model": "my-agent", "input": "What is 15 + 27?"},
    )
    result = response.json()
    print(f"   Response: {result['output'][0]['content'][0]['text']}")

    # Test 2: Streaming Request
    print("\n2. Streaming Request:")
    print("   ", end="")
    with httpx.stream(
        "POST",
        f"{BASE_URL}/v1/responses",
        json={
            "model": "my-agent",
            "input": "Search for information about Python",
            "stream": True,
        },
    ) as response:
        for line in response.iter_lines():
            if line.startswith("data: "):
                event = json.loads(line[6:])
                if event["type"] == "response.output_text.delta":
                    print(event["delta"], end="", flush=True)
    print()

    # Test 3: Conversation Chaining
    print("\n3. Conversation Chaining (OpenAI style):")

    # First message
    response1 = httpx.post(
        f"{BASE_URL}/v1/responses",
        json={
            "model": "my-agent",
            "input": "Remember that my favorite number is 42",
            "store": True,
        },
    )
    resp_id = response1.json()["id"]
    print(f"   First response ID: {resp_id}")

    # Continue conversation
    response2 = httpx.post(
        f"{BASE_URL}/v1/responses",
        json={
            "model": "my-agent",
            "input": "What's my favorite number multiplied by 2?",
            "previous_response_id": resp_id,
        },
    )
    print(f"   Continued: {response2.json()['output'][0]['content'][0]['text']}")

    # Test 4: Thread Management (LangGraph style)
    print("\n4. Thread Management (LangGraph style):")

    # Note: This requires a checkpointer configured in your graph
    response3 = httpx.post(
        f"{BASE_URL}/v1/responses",
        json={
            "model": "my-agent",
            "input": "Hello, I'm Alice",
            "thread_id": "conversation-123",
            "user": "alice",
        },
    )
    print(
        f"   Thread response: {response3.json()['output'][0]['content'][0]['text'][:50]}..."
    )

    print("\n" + "=" * 50)
    print(" All tests completed!")
    print("=" * 50)


# ============================================
# Main Entry Point
# ============================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run tests
        asyncio.run(test_api())
    else:
        # Run server
        print("\n Welcome to LangGraph Responses Gateway!")
        print(
            "\nThis example shows how to expose your LangGraph agent as an OpenAI API."
        )
        print("\nUsage:")
        print("  python quickstart.py        # Run the server")
        print("  python quickstart.py test   # Test the API (in another terminal)")
        print("\n" + "-" * 50)

        run_server()

