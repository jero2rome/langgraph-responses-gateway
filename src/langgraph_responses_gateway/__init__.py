"""LangGraph Responses Gateway - Pure service library for OpenAI Responses API.

This package provides a framework-agnostic service for exposing LangGraph agents
through the OpenAI Responses API format, without any web framework dependencies.

Author: Jerome Mohanan
License: MIT
"""

# Export integrations for framework-specific conveniences
from . import integrations
from .service import ResponsesGatewayService, ResponsesRequest
from .version import __version__

__all__ = [
    "ResponsesGatewayService",
    "ResponsesRequest",
    "integrations",
    "__version__",
]
