"""Framework integrations for langgraph-responses-gateway."""

from typing import TYPE_CHECKING, Any

# Lazy imports to avoid requiring FastAPI when not needed
if TYPE_CHECKING:
    from .fastapi import ResponsesAPIConfig, create_responses_router

__all__ = ["create_responses_router", "ResponsesAPIConfig"]


def __getattr__(name: str) -> Any:
    """Lazy import for optional dependencies."""
    if name == "create_responses_router":
        try:
            from .fastapi import create_responses_router

            return create_responses_router
        except ImportError as e:
            raise ImportError(
                "FastAPI integration requires FastAPI to be installed. "
                "Install with: pip install langgraph-responses-gateway[web]"
            ) from e
    elif name == "ResponsesAPIConfig":
        try:
            from .fastapi import ResponsesAPIConfig

            return ResponsesAPIConfig
        except ImportError as e:
            raise ImportError(
                "FastAPI integration requires FastAPI to be installed. "
                "Install with: pip install langgraph-responses-gateway[web]"
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
