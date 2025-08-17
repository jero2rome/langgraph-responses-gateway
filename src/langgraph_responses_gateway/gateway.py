"""Core gateway implementation for exposing LangGraph agents as OpenAI Responses API.

This module provides the main ResponsesGateway class that wraps any LangGraph
CompiledGraph and exposes it as an OpenAI Responses API endpoint.

Author: Jerome Mohanan
"""

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Optional, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel


class ResponsesRequest(BaseModel):
    """Request model for OpenAI Responses API."""

    model: str  # Required per OpenAI spec
    # Accept both 'input' (correct) and 'messages' (backward compat)
    input: Optional[Union[str, list[dict]]] = None
    messages: Optional[list[dict]] = None
    stream: bool = False
    temperature: Optional[float] = None
    tools: Optional[list] = None
    # For conversation chaining
    previous_response_id: Optional[str] = None
    # Custom fields for thread/user management
    thread_id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Optional[dict] = None
    store: Optional[bool] = False  # Whether to store the response


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
        # Store for conversation chaining (simple in-memory for MVP)
        self._response_store = {}

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

            # Validate model is provided
            if not req.model:
                raise HTTPException(400, "model is required")

            # Extract input (prefer 'input' over 'messages' for compatibility)
            user_input = self._extract_user_input(req)
            if not user_input:
                raise HTTPException(400, "No input or user message found")

            # Get previous context if chaining
            previous_context = None
            if req.previous_response_id:
                previous_context = self._response_store.get(req.previous_response_id)

            # Prepare input for LangGraph
            graph_input = self._prepare_graph_input(user_input, req, previous_context)

            if req.stream:
                return StreamingResponse(
                    self._stream_response(graph_input, req),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                        "Connection": "keep-alive",
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

    def _extract_user_input(self, req: ResponsesRequest) -> str:
        """Extract the user input from the request.

        Supports both 'input' (OpenAI Responses API spec) and 'messages' (backward compat).
        """
        # Prefer 'input' field (correct per spec)
        if req.input is not None:
            if isinstance(req.input, str):
                return req.input
            elif isinstance(req.input, list):
                # Handle array of input parts
                text_parts = []
                for part in req.input:
                    if isinstance(part, dict):
                        if part.get("type") == "input_text":
                            text_parts.append(part.get("text", ""))
                        elif part.get("type") == "text":  # Fallback
                            text_parts.append(part.get("text", ""))
                return " ".join(text_parts)

        # Fallback to 'messages' for backward compatibility
        if req.messages:
            for msg in reversed(req.messages):
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

    def _prepare_graph_input(
        self,
        user_input: str,
        req: ResponsesRequest,
        previous_context: Optional[dict] = None,
    ) -> dict:
        """Prepare input for the LangGraph."""
        # Build messages list with context if available
        messages = []

        # Add previous context if chaining
        if previous_context and "messages" in previous_context:
            messages.extend(previous_context["messages"])

        # Add current user input
        messages.append({"role": "user", "content": user_input})

        return {
            "messages": messages,
            "thread_id": req.thread_id,
            "user_id": req.user_id,
            "metadata": req.metadata or {},
            "temperature": req.temperature,
        }

    async def _stream_response(
        self, graph_input: dict, req: ResponsesRequest
    ) -> AsyncIterator[str]:
        """Stream response in OpenAI Responses API SSE format."""
        response_id = f"resp_{uuid.uuid4().hex}"
        item_id = f"item_{uuid.uuid4().hex}"
        created_at = int(time.time())

        # Send response.created event
        yield self._format_sse(
            {
                "type": "response.created",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created_at": created_at,
                    "model": req.model or self.model_name,
                    "status": "in_progress",
                },
            }
        )

        # Send response.output_item.added event
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
            # Track token usage from LangGraph messages (following OpenFlowLab pattern)
            total_tokens = {"prompt": 0, "completion": 0, "total": 0}

            # Stream from LangGraph
            async for step in self.graph.astream(graph_input, stream_mode="updates"):
                # Extract token usage from messages (OpenFlowLab pattern)
                for key, value in step.items():
                    if isinstance(value, dict) and "messages" in value:
                        for msg in value["messages"]:
                            # Check for AIMessage with response_metadata
                            if (
                                hasattr(msg, "response_metadata")
                                and msg.response_metadata
                            ):
                                usage = msg.response_metadata.get("token_usage", {})
                                if usage:
                                    total_tokens["prompt"] += usage.get(
                                        "prompt_tokens", 0
                                    )
                                    total_tokens["completion"] += usage.get(
                                        "completion_tokens", 0
                                    )
                                    total_tokens["total"] += usage.get(
                                        "total_tokens", 0
                                    )

                # Extract content from LangGraph step
                content = self._extract_content_from_step(step)

                if content and len(content) > len(accumulated_content):
                    # Send only the new delta
                    delta = content[len(accumulated_content) :]
                    accumulated_content = content

                    # Send response.output_text.delta event (correct name per spec)
                    yield self._format_sse(
                        {
                            "type": "response.output_text.delta",
                            "response_id": response_id,
                            "item_id": item_id,
                            "output_index": 0,
                            "content_index": 0,
                            "delta": delta,
                        }
                    )

            # Use real token counts if available, otherwise estimate
            if total_tokens["total"] == 0:
                # Fallback to estimation if no token usage from LangGraph
                total_tokens["prompt"] = self._estimate_tokens(str(graph_input))
                total_tokens["completion"] = self._estimate_tokens(accumulated_content)
                total_tokens["total"] = (
                    total_tokens["prompt"] + total_tokens["completion"]
                )

            # Send response.output_item.done event
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
                            }
                        ],
                    },
                }
            )

            # Send response.completed event
            yield self._format_sse(
                {
                    "type": "response.completed",
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "created_at": created_at,
                        "model": req.model or self.model_name,
                        "status": "completed",
                        "usage": {
                            "prompt_tokens": total_tokens["prompt"],
                            "completion_tokens": total_tokens["completion"],
                            "total_tokens": total_tokens["total"],
                        },
                    },
                }
            )

            # Store response if requested
            if req.store:
                self._store_response(response_id, graph_input, accumulated_content)

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
        """Create non-streaming response matching OpenAI structure exactly."""
        try:
            # Run the graph
            result = await self.graph.ainvoke(graph_input)

            # Extract content from result
            content = self._extract_content_from_result(result)

            response_id = f"resp_{uuid.uuid4().hex}"
            created_at = int(time.time())

            # Extract token usage from LangGraph messages (OpenFlowLab pattern)
            total_tokens = {"prompt": 0, "completion": 0, "total": 0}
            if isinstance(result, dict) and "messages" in result:
                for msg in result["messages"]:
                    if hasattr(msg, "response_metadata") and msg.response_metadata:
                        usage = msg.response_metadata.get("token_usage", {})
                        if usage:
                            total_tokens["prompt"] += usage.get("prompt_tokens", 0)
                            total_tokens["completion"] += usage.get(
                                "completion_tokens", 0
                            )
                            total_tokens["total"] += usage.get("total_tokens", 0)

            # Use real token counts if available, otherwise estimate
            if total_tokens["total"] == 0:
                # Fallback to estimation if no token usage from LangGraph
                total_tokens["prompt"] = self._estimate_tokens(str(graph_input))
                total_tokens["completion"] = self._estimate_tokens(content)
                total_tokens["total"] = (
                    total_tokens["prompt"] + total_tokens["completion"]
                )

            # Build response matching OpenAI spec exactly
            response_data = {
                "object": "response",
                "id": response_id,
                "created_at": created_at,
                "model": req.model or self.model_name,
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": content,
                            }
                        ],
                    }
                ],
                "usage": {
                    "prompt_tokens": total_tokens["prompt"],
                    "completion_tokens": total_tokens["completion"],
                    "total_tokens": total_tokens["total"],
                },
            }

            # Store response if requested
            if req.store:
                self._store_response(response_id, graph_input, content)

            return JSONResponse(response_data)

        except Exception as e:
            # Return error in proper format
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "type": "server_error",
                        "message": f"Error processing response: {str(e)}",
                    }
                },
            )

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

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (simplified).

        For production, consider using tiktoken for accurate counts.
        """
        # Simple estimation - 1 token per 4 characters
        return max(1, len(text) // 4)

    def _store_response(self, response_id: str, input_data: dict, output: str):
        """Store response for conversation chaining.

        This is a simple in-memory store. For production, use a proper database.
        """
        # Store the conversation context
        messages = input_data.get("messages", [])
        messages.append({"role": "assistant", "content": output})

        self._response_store[response_id] = {
            "messages": messages,
            "timestamp": time.time(),
        }

        # Simple cleanup - remove old responses (older than 1 hour)
        current_time = time.time()
        to_remove = [
            rid
            for rid, data in self._response_store.items()
            if current_time - data["timestamp"] > 3600
        ]
        for rid in to_remove:
            del self._response_store[rid]
