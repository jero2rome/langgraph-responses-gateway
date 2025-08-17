"""Core gateway implementation for exposing LangGraph agents as OpenAI Responses API.

This module provides the main ResponsesGateway class that wraps any LangGraph
CompiledGraph and exposes it as an OpenAI Responses API endpoint.

Follows the openai-agents-python SDK type definitions for full compatibility.

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
    """Request model for OpenAI Responses API.

    Based on openai-agents-python SDK types.
    """

    # Required parameters
    model: Optional[str] = None  # We'll validate this manually
    input: Optional[Union[str, list[dict]]] = None  # OpenAI Responses API format

    # Optional parameters from SDK
    stream: bool = False
    instructions: Optional[str] = None  # System instructions
    previous_response_id: Optional[str] = None  # For conversation chaining
    store: Optional[bool] = False  # Whether to store the response

    # Generation parameters
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_output_tokens: Optional[int] = None

    # Additional SDK parameters
    truncation: Optional[str] = None  # "auto" or "disabled"
    service_tier: Optional[str] = None  # "auto", "default", "flex", "scale", "priority"
    user: Optional[str] = None  # End-user identifier for caching/abuse detection
    include: Optional[list[str]] = None  # Additional output data to include

    # Tools (not fully implemented yet)
    tools: Optional[list] = None
    tool_choice: Optional[Union[str, dict]] = None
    parallel_tool_calls: Optional[bool] = None

    # Additional fields for LangGraph context
    thread_id: Optional[str] = None  # For LangGraph thread management
    user_id: Optional[str] = None  # User isolation (maps to SDK's 'user')
    metadata: Optional[dict] = None  # Custom metadata


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

            # Extract input from OpenAI Responses API format
            user_input = self._extract_user_input(req)
            if not user_input:
                raise HTTPException(400, "No input found")

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
        """Extract the user input from the OpenAI Responses API request.

        Handles both string and array of input parts format.
        """
        if req.input is not None:
            if isinstance(req.input, str):
                return req.input
            elif isinstance(req.input, list):
                # Handle array of input parts (for multimodal)
                text_parts = []
                for part in req.input:
                    if isinstance(part, dict):
                        if part.get("type") == "input_text":
                            text_parts.append(part.get("text", ""))
                        # Note: input_image, input_audio etc. not supported yet
                return " ".join(text_parts)

        return ""

    def _prepare_graph_input(
        self,
        user_input: str,
        req: ResponsesRequest,
        previous_context: Optional[dict] = None,
    ) -> dict:
        """Prepare input for LangGraph by converting OpenAI format to LangGraph's messages format.

        Translates from OpenAI Responses API 'input' to LangGraph's 'messages' structure.
        Handles both previous_response_id (for conversation continuation) and thread management.
        """
        # Build messages list for LangGraph (it expects messages format)
        messages = []

        # Add system instructions if provided (OpenAI SDK parameter)
        if req.instructions:
            messages.append({"role": "system", "content": req.instructions})

        # Add previous context if chaining via previous_response_id
        if previous_context and "messages" in previous_context:
            # Skip any system messages from previous context to avoid duplication
            for msg in previous_context["messages"]:
                if msg.get("role") != "system":
                    messages.append(msg)

        # Convert OpenAI input to LangGraph message format
        messages.append({"role": "user", "content": user_input})

        # Prepare config for LangGraph
        # Note: LangGraph's thread management is separate from OpenAI's previous_response_id
        # - previous_response_id: Links to a specific response for continuation
        # - thread_id + user_id: LangGraph's conversation state management
        config = {
            "messages": messages,  # LangGraph expects messages format
        }

        # Add LangGraph-specific thread management if provided
        # These are separate from OpenAI's previous_response_id concept
        if req.thread_id or req.user_id or req.user:
            # Use 'user' from SDK or fallback to 'user_id'
            effective_user = req.user or req.user_id
            config["thread_id"] = req.thread_id
            config["user_id"] = effective_user

        # Add metadata
        if req.metadata:
            config["metadata"] = req.metadata

        # Add generation parameters if provided
        if req.temperature is not None:
            config["temperature"] = req.temperature
        if req.top_p is not None:
            config["top_p"] = req.top_p
        if req.max_output_tokens is not None:
            config["max_output_tokens"] = req.max_output_tokens

        return config

    async def _stream_response(
        self, graph_input: dict, req: ResponsesRequest
    ) -> AsyncIterator[str]:
        """Stream response in OpenAI Responses API SSE format."""
        response_id = f"resp_{uuid.uuid4().hex}"
        item_id = f"item_{uuid.uuid4().hex}"
        created_at = int(time.time())

        # Prepare LangGraph config with proper thread management
        config = self._prepare_langgraph_config(graph_input, req)

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
            # Track token usage from LangGraph messages
            total_tokens = {"prompt": 0, "completion": 0, "total": 0}

            # Stream from LangGraph with proper config
            # Extract messages from graph_input for the actual invocation
            messages_input = {"messages": graph_input.get("messages", [])}
            async for step in self.graph.astream(
                messages_input, config=config, stream_mode="updates"
            ):
                # Extract token usage from messages
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
            # Prepare LangGraph config with proper thread management
            config = self._prepare_langgraph_config(graph_input, req)

            # Run the graph with proper config
            # Extract messages from graph_input for the actual invocation
            messages_input = {"messages": graph_input.get("messages", [])}
            result = await self.graph.ainvoke(messages_input, config=config)

            # Extract content from result
            content = self._extract_content_from_result(result)

            response_id = f"resp_{uuid.uuid4().hex}"
            created_at = int(time.time())

            # Extract token usage from LangGraph messages
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
        """Store response for conversation chaining via previous_response_id.

        This stores the full conversation context that can be retrieved when
        previous_response_id is provided in a subsequent request.

        Note: This is separate from LangGraph's thread management which uses
        thread_id + user_id for persistent conversation state.

        This is a simple in-memory store. For production, use a proper database.
        """
        # Store the conversation context (make a copy to avoid mutation)
        messages = list(input_data.get("messages", []))
        # Always append the assistant response
        messages.append({"role": "assistant", "content": output})

        self._response_store[response_id] = {
            "messages": messages,
            "timestamp": time.time(),
            # Store thread context if available for proper continuation
            "thread_id": input_data.get("thread_id"),
            "user_id": input_data.get("user_id"),
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

    def _prepare_langgraph_config(
        self, graph_input: dict, req: ResponsesRequest
    ) -> dict:
        """Prepare LangGraph configuration with proper thread management.

        Maps OpenAI's concepts to LangGraph's configuration:
        - thread_id + user_id -> composite thread_id for checkpointer
        - metadata -> passed through
        - generation params -> included if LangGraph supports them
        """
        config = {}

        # Handle LangGraph thread management
        if req.thread_id or req.user_id or req.user:
            # Use 'user' from SDK or fallback to 'user_id'
            effective_user = req.user or req.user_id or "guest"
            thread_id = req.thread_id or f"chat-{uuid.uuid4().hex[:8]}"

            # Create composite thread ID for checkpointer
            composite_thread_id = f"{effective_user}:{thread_id}"

            config["configurable"] = {"thread_id": composite_thread_id}

        # Add metadata if provided
        if req.metadata or graph_input.get("metadata"):
            config["metadata"] = req.metadata or graph_input.get("metadata", {})

        # Add recursion limit (allow enough steps for complex agents)
        config["recursion_limit"] = 150

        return config
