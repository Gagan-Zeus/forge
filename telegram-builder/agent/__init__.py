"""Autonomous build agent package."""

__all__ = ["BuildOrchestrator", "BuildResult"]


def __getattr__(name: str):
	if name in {"BuildOrchestrator", "BuildResult"}:
		from agent.orchestrator import BuildOrchestrator, BuildResult

		exports = {
			"BuildOrchestrator": BuildOrchestrator,
			"BuildResult": BuildResult,
		}
		return exports[name]
	raise AttributeError(f"module 'agent' has no attribute {name!r}")
