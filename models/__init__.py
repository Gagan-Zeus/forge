"""Model client package."""

from models.copilot_client import CopilotClient, CopilotAPIError, CopilotAuthError
from models.opencode_client import OpenCodeClient, OpenCodeAPIError, OpenCodeAuthError
from models.unified_client import (
    UnifiedModelClient,
    ModelClient,
    ModelAPIError,
    ModelAuthError,
)

__all__ = [
    "CopilotClient", "CopilotAPIError", "CopilotAuthError",
    "OpenCodeClient", "OpenCodeAPIError", "OpenCodeAuthError",
    "UnifiedModelClient", "ModelClient", "ModelAPIError", "ModelAuthError",
]
