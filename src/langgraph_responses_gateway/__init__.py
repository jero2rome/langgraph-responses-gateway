"""LangGraph Responses Gateway - Pure service library for OpenAI Responses API.

This package provides a framework-agnostic service for exposing LangGraph agents
through the OpenAI Responses API format, without any web framework dependencies.

Author: Jerome Mohanan
License: MIT
"""

from .service import ResponsesGatewayService, ResponsesRequest
from .version import __version__

__all__ = [
    "ResponsesGatewayService",
    "ResponsesRequest",
    "__version__",
]


# Lazy loading for optional web framework integrations
def __getattr__(name: str):
    """Lazy import for optional dependencies."""
    if name == "integrations":
        try:
            from . import integrations

            return integrations
        except ImportError as e:
            raise ImportError(
                "Web framework integrations require optional dependencies. "
                "Install with: pip install langgraph-responses-gateway[web]"
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
