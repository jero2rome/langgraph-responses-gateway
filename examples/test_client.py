"""Test client for interacting with the ResponsesGateway server.

This example shows different ways to interact with the gateway:
1. Non-streaming requests using 'input' parameter
2. Streaming with SSE
3. Conversation chaining with previous_response_id
4. Backward compatibility with 'messages' parameter

Author: Jerome Mohanan
"""

import asyncio
import json

import httpx


async def test_non_streaming():
    """Test non-streaming request with 'input' parameter (OpenAI spec)."""
    print("=" * 50)
    print("Testing Non-Streaming Request with 'input'")
    print("=" * 50)

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8003/v1/responses",
            json={
                "model": "langgraph-agent",  # Required per spec
                "input": "What is the capital of France?",
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
                
            # Show usage
            usage = data.get("usage", {})
            print(f"Tokens - Prompt: {usage.get('prompt_tokens')}, Completion: {usage.get('completion_tokens')}")
        else:
            print(f"Error: {response.status_code} - {response.text}")

    print()


async def test_streaming():
    """Test streaming request with SSE using 'input' parameter."""
    print("=" * 50)
    print("Testing Streaming Request with 'input'")
    print("=" * 50)

    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "http://localhost:8003/v1/responses",
            json={
                "model": "langgraph-agent",
                "input": "Tell me a short story about a robot",
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

                        elif event["type"] == "response.output_text.delta":
                            delta = event.get("delta", "")
                            print(delta, end="", flush=True)
                            accumulated_text += delta

                        elif event["type"] == "response.completed":
                            print("\n")
                            print(f"Status: {event['response']['status']}")
                            usage = event["response"].get("usage", {})
                            print(
                                f"Tokens - Prompt: {usage.get('prompt_tokens')}, "
                                f"Completion: {usage.get('completion_tokens')}, "
                                f"Total: {usage.get('total_tokens')}"
                            )

                    except json.JSONDecodeError:
                        pass

    print()


async def test_conversation_chaining():
    """Test conversation chaining with previous_response_id."""
    print("=" * 50)
    print("Testing Conversation Chaining")
    print("=" * 50)

    async with httpx.AsyncClient() as client:
        # First message
        print("Message 1:")
        response1 = await client.post(
            "http://localhost:8003/v1/responses",
            json={
                "model": "langgraph-agent",
                "input": "My name is Alice and I love robotics",
                "stream": False,
                "store": True,  # Store for chaining
            },
        )
        
        if response1.status_code == 200:
            data1 = response1.json()
            response_id1 = data1["id"]
            content = data1["output"][0]["content"][0]["text"]
            print(f"Response: {content}")
            print(f"Response ID: {response_id1}")

            # Second message chained to first
            print("\nMessage 2 (chained):")
            response2 = await client.post(
                "http://localhost:8003/v1/responses",
                json={
                    "model": "langgraph-agent",
                    "input": "What's my name and what do I love?",
                    "previous_response_id": response_id1,  # Chain to previous
                    "stream": False,
                },
            )
            
            if response2.status_code == 200:
                data2 = response2.json()
                content = data2["output"][0]["content"][0]["text"]
                print(f"Response: {content}")
            else:
                print(f"Error: {response2.status_code} - {response2.text}")
        else:
            print(f"Error: {response1.status_code} - {response1.text}")

    print()


async def test_multipart_input():
    """Test with array of input parts."""
    print("=" * 50)
    print("Testing Multi-part Input")
    print("=" * 50)

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8003/v1/responses",
            json={
                "model": "langgraph-agent",
                "input": [
                    {"type": "input_text", "text": "Calculate the sum of "},
                    {"type": "input_text", "text": "15 and 27"},
                ],
                "stream": False,
            },
        )

        if response.status_code == 200:
            data = response.json()
            content = data["output"][0]["content"][0]["text"]
            print(f"Response: {content}")
        else:
            print(f"Error: {response.status_code} - {response.text}")

    print()


async def test_backward_compatibility():
    """Test backward compatibility with 'messages' parameter."""
    print("=" * 50)
    print("Testing Backward Compatibility with 'messages'")
    print("=" * 50)

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8003/v1/responses",
            json={
                "model": "langgraph-agent",
                "messages": [
                    {"role": "user", "content": "What's 2+2?"}
                ],
                "stream": False,
            },
        )

        if response.status_code == 200:
            data = response.json()
            content = data["output"][0]["content"][0]["text"]
            print(f"Response: {content}")
            print(" Backward compatibility working")
        else:
            print(f"Error: {response.status_code} - {response.text}")

    print()


async def test_error_handling():
    """Test error handling with missing required fields."""
    print("=" * 50)
    print("Testing Error Handling")
    print("=" * 50)

    async with httpx.AsyncClient() as client:
        # Test missing model
        print("1. Missing 'model' parameter:")
        response = await client.post(
            "http://localhost:8003/v1/responses",
            json={
                "input": "Hello",
                "stream": False,
            },
        )
        
        if response.status_code == 400:
            print(f" Correctly rejected: {response.json()}")
        else:
            print(f" Should have failed: {response.status_code}")

        # Test missing input
        print("\n2. Missing 'input' and 'messages':")
        response = await client.post(
            "http://localhost:8003/v1/responses",
            json={
                "model": "langgraph-agent",
                "stream": False,
            },
        )
        
        if response.status_code == 400:
            print(f" Correctly rejected: {response.json()}")
        else:
            print(f" Should have failed: {response.status_code}")

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
        await test_multipart_input()
        await test_conversation_chaining()
        await test_backward_compatibility()
        await test_error_handling()

        print("=" * 50)
        print("All tests completed!")
        print("=" * 50)

    except httpx.ConnectError:
        print("ERROR: Could not connect to server at http://localhost:8003")
        print("Please make sure the server is running (run basic_agent.py first)")


if __name__ == "__main__":
    asyncio.run(main()