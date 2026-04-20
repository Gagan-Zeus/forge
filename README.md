# telegram-builder

`telegram-builder` is a Telegram-triggered autonomous coding agent that turns a project idea into a locally generated codebase, validates it, retries fixes when needed, and can optionally create and push a GitHub repository. It uses GitHub Copilot Device Flow authentication and then calls Copilot chat models to plan and generate the project files.

## Prerequisites

- Python 3.11+
- A Telegram bot token from BotFather
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

## First Run Auth (GitHub Copilot Device Flow)

On first `/start`, if `auth/tokens.json` is missing or invalid, the bot begins GitHub Device Flow:

1. Bot sends a message with `https://github.com/login/device` and a user code.
2. You manually open the URL and enter the code.
3. Bot polls GitHub until authorization completes.
4. Bot stores tokens in `auth/tokens.json` and confirms connection.

Token behavior:
- Long-lived `access_token` is stored once from OAuth device flow.
- Short-lived Copilot token is refreshed automatically before expiry.

## Example Conversation (ASCII)

```text
You: /start
Bot: To connect GitHub Copilot, go to:
     https://github.com/login/device
     Enter code: XXXX-XXXX
     Waiting for you to authorize...
Bot: Copilot connected! All models are now available.
     Send me your project idea to get started.
Bot: Step 1 - What project do you want to build? Describe it in detail.
You: Build a FastAPI task manager with JWT auth and SQLite.
Bot: Step 2 - Which language/stack? (e.g. Python, Node.js, React, FastAPI)
You: Python + FastAPI
Bot: Step 3 - Which AI model?
You: [tap GPT-4o]
Bot: Step 4 - Any special requirements? (libraries, constraints, architecture)
You: Use SQLAlchemy async and Alembic.
Bot: Step 5 - Push to GitHub when done? (yes / no)
You: yes
Bot: Repo name?
You: fastapi-task-manager
Bot: Public or private?
You: private
Bot: Step 6 - Summary ... Ready to build? (yes / no)
You: yes
Bot: Planning your project...
Bot: Got the plan - building 11 files...
Bot: Writing app/main.py... (1/11)
...
Bot: Running validation...
Bot: Pushing to GitHub...
Bot: Done! Repo: https://github.com/<user>/<repo>
```

## Adding New Copilot Models

Model support is centralized in `models/copilot_client.py`:

1. Add the model identifier to `CopilotClient.SUPPORTED_MODELS`.
2. Add a display label + value pair in `bot/handlers.py` under `MODEL_OPTIONS`.
3. Restart the bot.

## Troubleshooting

### Token expired / authentication issues

- Delete `auth/tokens.json` and send `/start` again.
- Ensure you complete device flow on the same GitHub account intended for Copilot.
- If you see HTTP 403 for `copilot_internal/v2/token`, verify that account has active Copilot access (individual or organization entitlement).

### Build failed

- Use `/status` to inspect progress.
- Review the error summary sent by the bot.
- If generated code still fails after retries, refine requirements and restart with `/reset`.

### GitHub push failed

- Set `GITHUB_TOKEN` in `.env` to a PAT with repository create/push permissions.
- Confirm the token owner can create repos under `GITHUB_USERNAME` (user or org).
- Ensure local Git is installed and configured.
- Try with `Push to GitHub = no` to validate local generation first.
