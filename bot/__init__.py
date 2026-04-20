"""Telegram bot package."""

__all__ = ["run_bot"]


def __getattr__(name: str):
	if name == "run_bot":
		from bot.handlers import run_bot

		return run_bot
	raise AttributeError(f"module 'bot' has no attribute {name!r}")
