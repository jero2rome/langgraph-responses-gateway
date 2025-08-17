"""LangGraph Responses Gateway - Bridge LangGraph agents to OpenAI Responses API.

This package provides a simple gateway to expose any LangGraph agent or multi-agent
system as an OpenAI Responses API endpoint, making it compatible with modern AI
clients like Vercel AI SDK.

Author: Jerome Mohanan
License: MIT
"""

from .gateway import ResponsesGateway
from .version import __version__

__all__ = ["ResponsesGateway", "__version__"]
