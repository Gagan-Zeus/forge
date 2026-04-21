# telegram-builder

`telegram-builder` is a Telegram-triggered autonomous coding agent that turns a project idea into a locally generated codebase, validates it, retries fixes when needed, and can optionally create and push a GitHub repository. It now uses the GitHub Copilot SDK (Python) and the local Copilot CLI session for model access.

## Prerequisites

- Python 3.11+
- A Telegram bot token from BotFather
- GitHub Copilot CLI installed and available in `PATH`
- Git installed and available in `PATH`
- (Optional) GitHub account with permissions to create repositories

## Setup

1. Clone the repository and enter it:
   ```bash
   git clone <your-repo-url>
   cd forge
   ```
2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure environment variables:
   ```bash
   cp .env.example .env
   ```
   Fill in:
   - `TELEGRAM_BOT_TOKEN`
   - `GITHUB_USERNAME`
   - `GITHUB_TOKEN` (recommended for GitHub push; PAT with repo permissions)
   - `PROJECTS_DIR` (default: `./generated_projects`)

## How To Run

```bash
python run.py
```

## First Run Auth (Copilot SDK + CLI)

Authenticate your local Copilot CLI once:

```bash
copilot auth login
```

Then run the bot and send `/start`. The bot checks CLI auth state through the SDK and enters chatbot mode.

## Example Conversation (ASCII)

```text
You: /start
Bot: Copilot SDK connected. Chatbot mode is active.
Bot: Default model: GPT-5 Mini
Bot: Commands:
Bot: /model - change model
Bot: /status - show current state
Bot: /cancel - cancel running build
Bot: /reset - reset chat state
You: /model
Bot: Current model: GPT-5 Mini
Bot: Choose a model: [buttons]
```

## Copilot Models

Only these models are available through `/model`, with default `gpt-5-mini`:

- `gpt-5.3-codex`
- `gpt-5.2-codex`
- `gpt-5.2`
- `gpt-5.4-mini`
- `gpt-5-mini`
- `gpt-4.1`
- `claude-haiku-4.5`

## Troubleshooting

### Copilot authentication issues

- Ensure Copilot CLI is installed and available in `PATH`.
- Run `copilot auth login` and confirm success.
- Send `/start` again.

### Build failed

- Use `/status` to inspect progress.
- Review the error summary sent by the bot.
- If generated code still fails after retries, refine requirements and restart the chat with `/reset`.
- If a specific model fails, switch model with `/model` and retry.

### GitHub push failed

- Set `GITHUB_TOKEN` in `.env` to a PAT with repository create/push permissions.
- Confirm the token owner can create repos under `GITHUB_USERNAME` (user or org).
- Ensure local Git is installed and configured.
- Try with `Push to GitHub = no` to validate local generation first.
