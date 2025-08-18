"""Framework integrations for langgraph-responses-gateway."""

from typing import TYPE_CHECKING

# Lazy imports to avoid requiring FastAPI when not needed
if TYPE_CHECKING:
    from .fastapi import ResponsesAPIConfig, create_responses_router

__all__ = ["create_responses_router", "ResponsesAPIConfig"]


def __getattr__(name):
    """Lazy import for optional dependencies."""
    if name in ["create_responses_router", "ResponsesAPIConfig"]:
        try:
            from .fastapi import ResponsesAPIConfig, create_responses_router

            return locals()[name]
        except ImportError as e:
            raise ImportError(
                "FastAPI integration requires FastAPI to be installed. "
                "Install with: pip install langgraph-responses-gateway[web]"
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
