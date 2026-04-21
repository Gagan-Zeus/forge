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

Then run the bot and send `/start`. The bot checks CLI auth state through the SDK and loads available models dynamically.

## Example Conversation (ASCII)

```text
You: /start
Bot: Copilot SDK connected. Send me your project idea to get started.
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

## Copilot Models

Available models are loaded from the Copilot SDK (`models.list`) at runtime. The Step 3 keyboard is refreshed from the SDK model list on `/start`, and you can also type any model id manually.

## Troubleshooting

### Copilot authentication issues

- Ensure Copilot CLI is installed and available in `PATH`.
- Run `copilot auth login` and confirm success.
- Send `/start` again.

### Build failed

- Use `/status` to inspect progress.
- Review the error summary sent by the bot.
- If generated code still fails after retries, refine requirements and restart with `/reset`.
- If a specific model fails, try another model id from Step 3 or type a model id manually.

### GitHub push failed

- Set `GITHUB_TOKEN` in `.env` to a PAT with repository create/push permissions.
- Confirm the token owner can create repos under `GITHUB_USERNAME` (user or org).
- Ensure local Git is installed and configured.
- Try with `Push to GitHub = no` to validate local generation first.
