"""FastAPI integration for langgraph-responses-gateway.

This module provides routers and utilities for integrating with FastAPI,
following modern best practices where the user maintains control of their app.
"""

import json
import uuid
from typing import Any, Callable, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..service import ResponsesGatewayService, ResponsesRequest


def create_responses_router(
    graph: Any,
    *,
    service_name: str = "assistant",
    public_model_name: Optional[str] = None,
    internal_model_name: str = "gpt-5-nano-2025-08-07",
    auth_handler: Optional[Callable[[Request], bool]] = None,
    session_handler: Optional[Callable[[Request], str]] = None,
) -> APIRouter:
    """
    Create a FastAPI router for OpenAI Responses API endpoints.

    This follows modern best practices where the library provides a router
    that users mount in their own FastAPI app, maintaining full control.

    Args:
        graph: LangGraph instance to expose
        service_name: Name for your service (used as default public model name)
        public_model_name: Public-facing model name (defaults to service_name)
        internal_model_name: Internal model to use (hidden from clients)
        auth_handler: Optional auth validation function
        session_handler: Optional function to extract session/thread ID from request

    Returns:
        APIRouter ready to be mounted in your FastAPI app

    Example:
        ```python
        from fastapi import FastAPI
        from langgraph_responses_gateway.integrations.fastapi import create_responses_router

        app = FastAPI(title="My Lab API")

        # Create and mount the router
        graph = await create_my_graph()
        router = create_responses_router(
            graph,
            service_name="lab-assistant",
            internal_model_name="gpt-5-nano-2025-08-07"
        )
        app.include_router(router, prefix="/v1")

        # Add your own routes, middleware, etc.
        @app.get("/custom")
        async def my_custom_endpoint():
            return {"custom": "endpoint"}
        ```
    """

    # Use service_name as public model name if not specified
    if public_model_name is None:
        public_model_name = service_name

    # Create the service
    service = ResponsesGatewayService(
        graph=graph,
        model_name=public_model_name,
        name=service_name,
    )

    # Create router
    router = APIRouter()

    @router.post("/responses")
    async def responses_endpoint(request: Request):
        """OpenAI Responses API endpoint."""

        # Optional auth check
        if auth_handler and not auth_handler(request):
            raise HTTPException(401, "Unauthorized")

        # Parse request body
        try:
            body = await request.json()
        except Exception as e:
            raise HTTPException(400, f"Invalid JSON: {str(e)}")

        # Auto-inject model if not provided
        if "model" not in body:
            body["model"] = internal_model_name

        # Handle session/thread ID
        if "thread_id" not in body:
            if session_handler:
                # Use custom session handler
                body["thread_id"] = session_handler(request)
            else:
                # Generate a thread ID
                body["thread_id"] = f"session-{uuid.uuid4().hex[:8]}"

        # Validate request
        try:
            req = ResponsesRequest(**body)
        except Exception as e:
            raise HTTPException(400, f"Invalid request: {str(e)}")

        # Process request
        if req.stream:
            # Streaming response
            async def stream_with_model_masking():
                async for event in service.stream_response(req):
                    # Mask internal model name in SSE events
                    yield _mask_model_in_sse(event, public_model_name)

            return StreamingResponse(
                stream_with_model_masking(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive",
                },
            )
        else:
            # Non-streaming response
            try:
                result = await service.process_request(req)
                # Mask internal model name
                result["model"] = public_model_name
                return JSONResponse(result)
            except ValueError as e:
                raise HTTPException(400, str(e))
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": {
                            "type": "server_error",
                            "message": str(e),
                        }
                    },
                )

    @router.get("/models")
    async def list_models():
        """List available models."""
        return {
            "object": "list",
            "data": [
                {
                    "id": public_model_name,
                    "object": "model",
                    "owned_by": "system",
                }
            ],
        }

    return router


def _mask_model_in_sse(event: str, public_model_name: str) -> str:
    """Replace internal model references with public name in SSE events."""
    if not event.startswith("data: "):
        return event

    try:
        # Parse the JSON data
        data = json.loads(event[6:])

        # Replace model references
        if "response" in data and "model" in data["response"]:
            data["response"]["model"] = public_model_name
        if "model" in data:
            data["model"] = public_model_name

        # Re-serialize
        return f"data: {json.dumps(data)}\n\n"
    except:
        # If parsing fails, return as-is
        return event


class ResponsesAPIConfig:
    """
    Configuration helper for common setups.

    This provides convenience methods for common authentication
    and session management patterns.
    """

    @staticmethod
    def bearer_auth_handler(expected_token: str) -> Callable[[Request], bool]:
        """Create a simple Bearer token auth handler."""

        def handler(request: Request) -> bool:
            auth = request.headers.get("Authorization", "")
            return auth == f"Bearer {expected_token}"

        return handler

    @staticmethod
    def header_session_handler(
        header_name: str = "X-Session-Id",
    ) -> Callable[[Request], str]:
        """Create a session handler that reads from a header."""

        def handler(request: Request) -> str:
            session_id = request.headers.get(header_name)
            if session_id:
                return f"session-{session_id}"
            return f"session-{uuid.uuid4().hex[:8]}"

        return handler

    @staticmethod
    def cookie_session_handler(
        cookie_name: str = "session_id",
    ) -> Callable[[Request], str]:
        """Create a session handler that reads from a cookie."""

        def handler(request: Request) -> str:
            session_id = request.cookies.get(cookie_name)
            if session_id:
                return f"session-{session_id}"
            return f"session-{uuid.uuid4().hex[:8]}"

        return handler
