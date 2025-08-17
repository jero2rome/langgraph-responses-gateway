"""Test client for interacting with the ResponsesGateway server.

This example shows different ways to interact with the gateway:
1. Non-streaming requests
2. Streaming with SSE
3. Using the OpenAI Python client

Author: Jerome Mohanan
"""

import asyncio
import json

import httpx


async def test_non_streaming():
    """Test non-streaming request."""
    print("=" * 50)
    print("Testing Non-Streaming Request")
    print("=" * 50)

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8003/v1/responses",
            json={
                "messages": [
                    {"role": "user", "content": "What is the capital of France?"}
                ],
                "stream": False,
            },
        )

        if response.status_code == 200:
            data = response.json()
            print(f"Response ID: {data['id']}")
            print(f"Status: {data['status']}")

            # Extract the text content
            if data.get("output"):
                content = data["output"][0]["content"][0]["text"]
                print(f"Response: {content}")
        else:
            print(f"Error: {response.status_code} - {response.text}")

    print()


async def test_streaming():
    """Test streaming request with SSE."""
    print("=" * 50)
    print("Testing Streaming Request")
    print("=" * 50)

    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "http://localhost:8003/v1/responses",
            json={
                "messages": [
                    {"role": "user", "content": "Tell me a short story about a robot"}
                ],
                "stream": True,
            },
        ) as response:
            print("Streaming response:")
            accumulated_text = ""

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        event = json.loads(line[6:])

                        if event["type"] == "response.created":
                            print(f"Response ID: {event['response']['id']}")
                            print("Content: ", end="", flush=True)

                        elif event["type"] == "response.text.delta":
                            delta = event.get("delta", "")
                            print(delta, end="", flush=True)
                            accumulated_text += delta

                        elif event["type"] == "response.completed":
                            print("\n")
                            print(f"Status: {event['response']['status']}")
                            usage = event["response"].get("usage", {})
                            print(
                                f"Tokens - Input: {usage.get('input_tokens')}, Output: {usage.get('output_tokens')}"
                            )

                    except json.JSONDecodeError:
                        pass

    print()


async def test_with_thread_id():
    """Test conversation threading."""
    print("=" * 50)
    print("Testing Conversation Threading")
    print("=" * 50)

    thread_id = "test-thread-123"

    async with httpx.AsyncClient() as client:
        # First message
        print("Message 1:")
        response1 = await client.post(
            "http://localhost:8003/v1/responses",
            json={
                "messages": [{"role": "user", "content": "My name is Alice"}],
                "thread_id": thread_id,
                "stream": False,
            },
        )
        if response1.status_code == 200:
            content = response1.json()["output"][0]["content"][0]["text"]
            print(f"Response: {content}")

        # Second message in same thread
        print("\nMessage 2 (same thread):")
        response2 = await client.post(
            "http://localhost:8003/v1/responses",
            json={
                "messages": [{"role": "user", "content": "What's my name?"}],
                "thread_id": thread_id,
                "stream": False,
            },
        )
        if response2.status_code == 200:
            content = response2.json()["output"][0]["content"][0]["text"]
            print(f"Response: {content}")

    print()


async def test_with_openai_client():
    """Test using the OpenAI Python client (if available)."""
    print("=" * 50)
    print("Testing with OpenAI Client")
    print("=" * 50)

    try:
        from openai import OpenAI

        # Configure client to use our gateway
        client = OpenAI(
            base_url="http://localhost:8003/v1",
            api_key="not-needed",  # Gateway doesn't require auth
        )

        # Note: OpenAI client expects /chat/completions endpoint
        # This would need the gateway to also expose that endpoint
        print("Note: OpenAI client requires /chat/completions endpoint")
        print("The gateway currently only exposes /responses endpoint")
        print("Use httpx for direct Responses API access")

    except ImportError:
        print("OpenAI client not installed. Install with: pip install openai")

    print()


async def test_health_check():
    """Test the health check endpoint."""
    print("=" * 50)
    print("Testing Health Check")
    print("=" * 50)

    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8003/health")

        if response.status_code == 200:
            data = response.json()
            print(f"Status: {data['status']}")
            print(f"Agent: {data['agent']}")
            print(f"Version: {data['version']}")
        else:
            print(f"Error: {response.status_code}")

    print()


async def test_list_models():
    """Test listing available models."""
    print("=" * 50)
    print("Testing List Models")
    print("=" * 50)

    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8003/v1/models")

        if response.status_code == 200:
            data = response.json()
            print("Available models:")
            for model in data.get("data", []):
                print(f"  - {model['id']} (owned by: {model['owned_by']})")
        else:
            print(f"Error: {response.status_code}")

    print()


async def main():
    """Run all tests."""
    print("\n" + "=" * 50)
    print("LangGraph Responses Gateway Test Client")
    print("=" * 50)
    print("\nMake sure the server is running on http://localhost:8003")
    print("Run the basic_agent.py example first!\n")

    try:
        # Run tests sequentially
        await test_health_check()
        await test_list_models()
        await test_non_streaming()
        await test_streaming()
        await test_with_thread_id()
        await test_with_openai_client()

        print("=" * 50)
        print("All tests completed!")
        print("=" * 50)

    except httpx.ConnectError:
        print("ERROR: Could not connect to server at http://localhost:8003")
        print("Please make sure the server is running (run basic_agent.py first)")


if __name__ == "__main__":
    asyncio.run(main())
