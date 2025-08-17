"""Basic example of using langgraph-responses-gateway with a ReAct agent.

This example demonstrates how to:
1. Create a LangGraph ReAct agent using the prebuilt helper
2. Wrap it with ResponsesGateway
3. Run it as an OpenAI-compatible API server

Author: Jerome Mohanan
"""

# Import the gateway
import os
import sys
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Load .env file from parent directory
load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, "../src")
from langgraph_responses_gateway import ResponsesGateway


# Define some example tools for the agent
@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    # This is a mock implementation
    return f"The weather in {location} is sunny and 72°F."


@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        # Safe evaluation of simple math expressions
        result = eval(expression, {"__builtins__": {}}, {})
        return f"The result of {expression} is {result}"
    except Exception:
        return "Invalid mathematical expression"


@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    # This is a mock implementation
    return f"Here are some search results for '{query}': This is a demo result showing that the agent can use tools."


def create_react_agent_graph():
    """Create a ReAct agent using LangGraph's prebuilt helper."""
    # Define the tools
    tools = [get_weather, calculate, search_web]

    # Create a mock LLM (in production, use a real model)
    # For demo purposes, we'll use a simple mock that doesn't require API keys
    try:
        # Try to use OpenAI if API key is available
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY", "sk-mock-key-for-demo"),
        )
    except Exception:
        # Fallback to a mock implementation
        from langchain_core.language_models import FakeListChatModel

        llm = FakeListChatModel(
            responses=[
                "I'll help you with that request using the available tools.",
                "Let me search for that information.",
                "Based on my calculations, here's the result.",
            ]
        )

    # Create the ReAct agent using the prebuilt helper
    agent_graph = create_react_agent(llm, tools=tools)

    return agent_graph


def main():
    """Run the example server."""
    # Create the LangGraph ReAct agent
    graph = create_react_agent_graph()

    # Wrap it with ResponsesGateway
    gateway = ResponsesGateway(
        graph=graph, name="ReAct Agent", version="1.0.0", model_name="react-agent"
    )

    print("Starting ReAct Agent server...")
    print("API available at: http://localhost:8003/v1/responses")
    print("\nExample requests:")
    print("\n1. Ask about weather:")
    print("curl -X POST http://localhost:8003/v1/responses \\")
    print('  -H "Content-Type: application/json" \\')
    print(
        '  -d \'{"messages": [{"role": "user", "content": "What is the weather in San Francisco?"}], "stream": false}\''
    )
    print("\n2. Perform calculations:")
    print("curl -X POST http://localhost:8003/v1/responses \\")
    print('  -H "Content-Type: application/json" \\')
    print(
        '  -d \'{"messages": [{"role": "user", "content": "Calculate 125 * 8 + 200"}], "stream": false}\''
    )
    print("\n3. Search the web (with streaming):")
    print("curl -X POST http://localhost:8003/v1/responses \\")
    print('  -H "Content-Type: application/json" \\')
    print(
        '  -d \'{"messages": [{"role": "user", "content": "Search for information about LangGraph"}], "stream": true}\''
    )
    print("\nPress Ctrl+C to stop the server\n")

    # Run the server
    uvicorn.run(gateway.app, host="0.0.0.0", port=8003)


if __name__ == "__main__":
    main()
