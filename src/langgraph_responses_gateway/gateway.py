"""Core gateway implementation for exposing LangGraph agents as OpenAI Responses API.

This module provides the main ResponsesGateway class that wraps any LangGraph
CompiledGraph and exposes it as an OpenAI Responses API endpoint.

Author: Jerome Mohanan
"""

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel


class ResponsesRequest(BaseModel):
    """Request model for OpenAI Responses API."""

    model: Optional[str] = None
    messages: list[dict] = []
    stream: bool = False
    temperature: Optional[float] = None
    tools: Optional[list] = None
    thread_id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Optional[dict] = None


class ResponsesGateway:
    """Gateway to expose LangGraph agents as OpenAI Responses API.

    This class wraps any LangGraph CompiledGraph and exposes it through
    the OpenAI Responses API format, supporting both streaming and non-streaming
    responses.

    Args:
        graph: The compiled LangGraph to expose
        name: Name of the agent/platform
        version: Version string
        base_path: Base path for API endpoints (default: /v1)
        model_name: Model name to report in responses

    Example:
        ```python
        from langgraph_responses_gateway import ResponsesGateway
        from your_agent import create_agent_graph

        graph = create_agent_graph()
        gateway = ResponsesGateway(graph, name="My Agent")

        # Run with uvicorn
        import uvicorn
        uvicorn.run(gateway.app, host="0.0.0.0", port=8000)
        ```
    """

    def __init__(
        self,
        graph: Any,  # Accept any compiled graph (CompiledStateGraph, etc.)
        *,
        name: str = "LangGraph Agent",
        version: str = "1.0.0",
        base_path: str = "/v1",
        model_name: str = "langgraph-agent",
    ):
        """Initialize the gateway with a LangGraph."""
        self.graph = graph
        self.name = name
        self.version = version
        self.base_path = base_path
        self.model_name = model_name
        self.app = self._create_app()

    def _create_app(self) -> FastAPI:
        """Create FastAPI application with Responses API endpoints."""
        app = FastAPI(
            title=self.name,
            version=self.version,
            description=f"{self.name} exposed as OpenAI Responses API",
        )

        @app.post(f"{self.base_path}/responses")
        async def create_response(request: Request):
            """Handle OpenAI Responses API requests."""
            body = await request.json()
            req = ResponsesRequest(**body)

            # Extract user message
            user_message = self._extract_user_message(req.messages)
            if not user_message:
                raise HTTPException(400, "No user message found")

            # Prepare input for LangGraph
            graph_input = self._prepare_graph_input(user_message, req)

            if req.stream:
                return StreamingResponse(
                    self._stream_response(graph_input, req),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                    },
                )
            else:
                return await self._create_response(graph_input, req)

        @app.get(f"{self.base_path}/models")
        async def list_models():
            """List available models."""
            return {
                "object": "list",
                "data": [
                    {
                        "id": self.model_name,
                        "object": "model",
                        "owned_by": "langgraph",
                    }
                ],
            }

        @app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "agent": self.name,
                "version": self.version,
            }

        return app

    def _extract_user_message(self, messages: list[dict]) -> str:
        """Extract the user message from the messages list."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    # Handle multimodal content
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                    return " ".join(text_parts)
        return ""

    def _prepare_graph_input(self, user_message: str, req: ResponsesRequest) -> dict:
        """Prepare input for the LangGraph."""
        # Basic input structure - can be customized based on your graph
        return {
            "messages": [{"role": "user", "content": user_message}],
            "thread_id": req.thread_id,
            "user_id": req.user_id,
            "metadata": req.metadata or {},
        }

    async def _stream_response(
        self, graph_input: dict, req: ResponsesRequest
    ) -> AsyncIterator[str]:
        """Stream response in OpenAI Responses API SSE format."""
        response_id = f"resp_{uuid.uuid4().hex}"
        item_id = f"item_{uuid.uuid4().hex}"

        # Send response.created event
        yield self._format_sse(
            {
                "type": "response.created",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created_at": int(time.time()),
                    "model": self.model_name,
                    "status": "in_progress",
                },
            }
        )

        # Send output item added event
        yield self._format_sse(
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": item_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": ""}],
                },
            }
        )

        try:
            accumulated_content = ""

            # Stream from LangGraph
            async for step in self.graph.astream(graph_input):
                # Extract content from LangGraph step
                content = self._extract_content_from_step(step)

                if content and len(content) > len(accumulated_content):
                    # Send only the new delta
                    delta = content[len(accumulated_content) :]
                    accumulated_content = content

                    # Send text delta event
                    yield self._format_sse(
                        {
                            "type": "response.text.delta",
                            "response_id": response_id,
                            "item_id": item_id,
                            "output_index": 0,
                            "content_index": 0,
                            "delta": delta,
                        }
                    )

            # Send output item done event
            yield self._format_sse(
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": {
                        "id": item_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": accumulated_content,
                                "annotations": [],
                            }
                        ],
                    },
                }
            )

            # Send response completed event
            yield self._format_sse(
                {
                    "type": "response.completed",
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "created_at": int(time.time()),
                        "model": self.model_name,
                        "status": "completed",
                        "usage": self._estimate_usage(accumulated_content),
                        "incomplete_details": None,
                    },
                }
            )

        except Exception as e:
            # Send error event
            yield self._format_sse(
                {
                    "type": "error",
                    "error": {
                        "type": "server_error",
                        "message": str(e),
                    },
                }
            )

    async def _create_response(
        self, graph_input: dict, req: ResponsesRequest
    ) -> JSONResponse:
        """Create non-streaming response."""
        try:
            # Run the graph
            result = await self.graph.ainvoke(graph_input)

            # Extract content from result
            content = self._extract_content_from_result(result)

            response_id = f"resp_{uuid.uuid4().hex}"
            item_id = f"item_{uuid.uuid4().hex}"

            response_data = {
                "id": response_id,
                "object": "response",
                "created_at": int(time.time()),
                "model": self.model_name,
                "status": "completed",
                "output": [
                    {
                        "id": item_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": content,
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "usage": self._estimate_usage(content),
                "incomplete_details": None,
            }

            return JSONResponse(response_data)

        except Exception as e:
            raise HTTPException(500, f"Error processing response: {str(e)}")

    def _extract_content_from_step(self, step: Any) -> str:
        """Extract content from a LangGraph streaming step.

        This method should be customized based on your graph's output format.
        """
        # Common patterns for LangGraph output
        if isinstance(step, dict):
            # Check for messages in the step
            for key, value in step.items():
                if isinstance(value, dict):
                    if "messages" in value and value["messages"]:
                        msg = value["messages"][-1]
                        if hasattr(msg, "content"):
                            return msg.content
                        elif isinstance(msg, dict) and "content" in msg:
                            return msg["content"]

            # Check for direct content
            if "content" in step:
                return step["content"]
            if "output" in step:
                return step["output"]

        return ""

    def _extract_content_from_result(self, result: Any) -> str:
        """Extract content from a LangGraph result.

        This method should be customized based on your graph's output format.
        """
        if isinstance(result, dict):
            # Check for messages
            if "messages" in result and result["messages"]:
                msg = result["messages"][-1]
                if hasattr(msg, "content"):
                    return msg.content
                elif isinstance(msg, dict) and "content" in msg:
                    return msg["content"]

            # Check for direct output
            if "output" in result:
                return str(result["output"])
            if "content" in result:
                return str(result["content"])

        return str(result)

    def _format_sse(self, event: dict) -> str:
        """Format an event as Server-Sent Event."""
        return f"data: {json.dumps(event)}\n\n"

    def _estimate_usage(self, content: str) -> dict:
        """Estimate token usage (simplified)."""
        # Simple estimation - 1 token per 4 characters
        output_tokens = len(content) // 4
        input_tokens = 50  # Rough estimate

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
