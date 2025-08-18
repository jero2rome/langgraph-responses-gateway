"""Pure service layer for OpenAI Responses API.

This module provides a framework-agnostic service for exposing LangGraph agents
through the OpenAI Responses API format, without any web framework dependencies.
"""

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Optional, Protocol, Union

from pydantic import BaseModel, Field


class LangGraphProtocol(Protocol):
    """Protocol for LangGraph compatibility."""

    async def ainvoke(self, input_data: dict, config: Optional[dict] = None) -> dict:
        """Invoke the graph asynchronously."""
        ...

    async def astream(
        self,
        input_data: dict,
        config: Optional[dict] = None,
        stream_mode: str = "updates",
    ) -> AsyncIterator[dict]:
        """Stream the graph execution asynchronously."""
        ...


class ResponsesRequest(BaseModel):
    """Request model for OpenAI Responses API."""

    # Required parameters
    model: Optional[str] = None
    input: Optional[Union[str, list[dict]]] = None

    # Optional parameters
    stream: bool = False
    instructions: Optional[str] = None
    previous_response_id: Optional[str] = None
    store: Optional[bool] = False

    # Generation parameters
    temperature: Optional[float] = Field(None, ge=0, le=2)
    top_p: Optional[float] = Field(None, ge=0, le=1)
    max_output_tokens: Optional[int] = Field(None, gt=0)

    # Additional parameters
    truncation: Optional[str] = None
    service_tier: Optional[str] = None
    user: Optional[str] = None
    include: Optional[list[str]] = None

    # Tools (not fully implemented yet)
    tools: Optional[list] = None
    tool_choice: Optional[Union[str, dict]] = None
    parallel_tool_calls: Optional[bool] = None

    # LangGraph context
    thread_id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Optional[dict] = None


class ResponsesGatewayService:
    """
    Pure service layer for OpenAI Responses API.

    This service handles the business logic of processing requests
    through a LangGraph, without any HTTP/web framework concerns.
    Token usage tracking is preserved from the original implementation.
    """

    def __init__(
        self,
        graph: LangGraphProtocol,
        model_name: str = "gpt-5-nano-2025-08-07",
        name: str = "LangGraph Agent",
        version: str = "1.0.0",
    ):
        """
        Initialize the service with a LangGraph.

        Args:
            graph: Any LangGraph-compatible graph
            model_name: Model identifier for responses
            name: Name of the agent/platform
            version: Version string
        """
        self.graph = graph
        self.model_name = model_name
        self.name = name
        self.version = version
        # Store for conversation chaining (simple in-memory for MVP)
        self._response_store = {}

    async def process_request(self, request: ResponsesRequest) -> dict:
        """
        Process a request and return the response.

        Args:
            request: The OpenAI Responses API request

        Returns:
            Response dictionary following OpenAI Responses format
        """
        # Validate model is provided
        if not request.model:
            raise ValueError("model is required")

        # Extract input
        user_input = self._extract_user_input(request)
        if not user_input:
            raise ValueError("No input found")

        # Get previous context if chaining
        previous_context = None
        if request.previous_response_id:
            previous_context = self._response_store.get(request.previous_response_id)

        # Prepare input for LangGraph
        graph_input = self._prepare_graph_input(user_input, request, previous_context)

        # Prepare LangGraph config
        config = self._prepare_langgraph_config(graph_input, request)

        # Run the graph
        messages_input = {"messages": graph_input.get("messages", [])}
        result = await self.graph.ainvoke(messages_input, config=config)

        # Extract content from result
        content = self._extract_content_from_result(result)

        response_id = f"resp_{uuid.uuid4().hex[:16]}"
        created_at = int(time.time())

        # Extract token usage from LangGraph messages
        total_tokens = self._extract_token_usage(result)

        # Use real token counts if available, otherwise estimate
        if total_tokens["total"] == 0:
            total_tokens = self._estimate_token_usage(graph_input, content)

        # Build response
        response_data = {
            "object": "response",
            "id": response_id,
            "created_at": created_at,
            "model": request.model or self.model_name,
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
        if request.store:
            self._store_response(response_id, graph_input, content)

        return response_data

    async def stream_response(self, request: ResponsesRequest) -> AsyncIterator[str]:
        """
        Stream a response as Server-Sent Events.

        Args:
            request: The OpenAI Responses API request

        Yields:
            SSE-formatted strings ready to send to client
        """
        # Validate model is provided
        if not request.model:
            raise ValueError("model is required")

        # Extract input
        user_input = self._extract_user_input(request)
        if not user_input:
            raise ValueError("No input found")

        # Get previous context if chaining
        previous_context = None
        if request.previous_response_id:
            previous_context = self._response_store.get(request.previous_response_id)

        # Prepare input for LangGraph
        graph_input = self._prepare_graph_input(user_input, request, previous_context)

        response_id = f"resp_{uuid.uuid4().hex}"
        item_id = f"item_{uuid.uuid4().hex}"
        created_at = int(time.time())

        # Prepare LangGraph config
        config = self._prepare_langgraph_config(graph_input, request)

        # Send response.created event
        yield self._format_sse(
            {
                "type": "response.created",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created_at": created_at,
                    "model": request.model or self.model_name,
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
            total_tokens = {"prompt": 0, "completion": 0, "total": 0}

            # Stream from LangGraph
            messages_input = {"messages": graph_input.get("messages", [])}
            async for step in self.graph.astream(
                messages_input, config=config, stream_mode="updates"
            ):
                # Extract token usage from messages
                for key, value in step.items():
                    if isinstance(value, dict) and "messages" in value:
                        for msg in value["messages"]:
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

                # Extract content from step
                content = self._extract_content_from_step(step)

                if content and len(content) > len(accumulated_content):
                    # Send only the new delta
                    delta = content[len(accumulated_content) :]
                    accumulated_content = content

                    # Send response.output_text.delta event
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
                total_tokens = self._estimate_token_usage(
                    graph_input, accumulated_content
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
                        "model": request.model or self.model_name,
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
            if request.store:
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

    def _extract_user_input(self, req: ResponsesRequest) -> str:
        """Extract the user input from the request."""
        if req.input is not None:
            if isinstance(req.input, str):
                return req.input
            elif isinstance(req.input, list):
                text_parts = []
                for part in req.input:
                    if isinstance(part, dict):
                        if part.get("type") == "input_text":
                            text_parts.append(part.get("text", ""))
                return " ".join(text_parts)
        return ""

    def _prepare_graph_input(
        self,
        user_input: str,
        req: ResponsesRequest,
        previous_context: Optional[dict] = None,
    ) -> dict:
        """Prepare input for LangGraph."""
        messages = []

        # Add system instructions if provided
        if req.instructions:
            messages.append({"role": "system", "content": req.instructions})

        # Add previous context if chaining
        if previous_context and "messages" in previous_context:
            for msg in previous_context["messages"]:
                if msg.get("role") != "system":
                    messages.append(msg)

        # Add current user input
        messages.append({"role": "user", "content": user_input})

        # Prepare config
        config = {"messages": messages}

        # Add thread management
        if req.thread_id or req.user_id or req.user:
            effective_user = req.user or req.user_id
            config["thread_id"] = req.thread_id
            config["user_id"] = effective_user

        # Add metadata
        if req.metadata:
            config["metadata"] = req.metadata

        # Add generation parameters
        if req.temperature is not None:
            config["temperature"] = req.temperature
        if req.top_p is not None:
            config["top_p"] = req.top_p
        if req.max_output_tokens is not None:
            config["max_output_tokens"] = req.max_output_tokens

        return config

    def _prepare_langgraph_config(
        self, graph_input: dict, req: ResponsesRequest
    ) -> dict:
        """Prepare LangGraph configuration."""
        config = {}

        # Handle thread management
        if req.thread_id or req.user_id or req.user:
            effective_user = req.user or req.user_id or "guest"
            thread_id = req.thread_id or f"chat-{uuid.uuid4().hex[:8]}"
            composite_thread_id = f"{effective_user}:{thread_id}"
            config["configurable"] = {"thread_id": composite_thread_id}

        # Add metadata
        if req.metadata or graph_input.get("metadata"):
            config["metadata"] = req.metadata or graph_input.get("metadata", {})

        # Add recursion limit
        config["recursion_limit"] = 150

        return config

    def _extract_content_from_step(self, step: Any) -> str:
        """Extract content from a LangGraph streaming step."""
        if isinstance(step, dict):
            for key, value in step.items():
                if isinstance(value, dict):
                    if "messages" in value and value["messages"]:
                        msg = value["messages"][-1]
                        if hasattr(msg, "content"):
                            return msg.content
                        elif isinstance(msg, dict) and "content" in msg:
                            return msg["content"]

            if "content" in step:
                return step["content"]
            if "output" in step:
                return step["output"]

        return ""

    def _extract_content_from_result(self, result: Any) -> str:
        """Extract content from a LangGraph result."""
        if isinstance(result, dict):
            if "messages" in result and result["messages"]:
                msg = result["messages"][-1]
                if hasattr(msg, "content"):
                    return msg.content
                elif isinstance(msg, dict) and "content" in msg:
                    return msg["content"]

            if "output" in result:
                return str(result["output"])
            if "content" in result:
                return str(result["content"])

        return str(result)

    def _extract_token_usage(self, result: Any) -> dict:
        """Extract token usage from LangGraph result."""
        total_tokens = {"prompt": 0, "completion": 0, "total": 0}

        if isinstance(result, dict) and "messages" in result:
            for msg in result["messages"]:
                if hasattr(msg, "response_metadata") and msg.response_metadata:
                    usage = msg.response_metadata.get("token_usage", {})
                    if usage:
                        total_tokens["prompt"] += usage.get("prompt_tokens", 0)
                        total_tokens["completion"] += usage.get("completion_tokens", 0)
                        total_tokens["total"] += usage.get("total_tokens", 0)

        return total_tokens

    def _estimate_token_usage(self, graph_input: dict, content: str) -> dict:
        """Estimate token usage if not provided by LangGraph."""
        # Simple estimation - 1 token per 4 characters
        prompt_tokens = max(1, len(str(graph_input)) // 4)
        completion_tokens = max(1, len(content) // 4)

        return {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": prompt_tokens + completion_tokens,
        }

    def _format_sse(self, event: dict) -> str:
        """Format an event as Server-Sent Event."""
        return f"data: {json.dumps(event)}\n\n"

    def _store_response(self, response_id: str, input_data: dict, output: str):
        """Store response for conversation chaining."""
        messages = list(input_data.get("messages", []))
        messages.append({"role": "assistant", "content": output})

        self._response_store[response_id] = {
            "messages": messages,
            "timestamp": time.time(),
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
