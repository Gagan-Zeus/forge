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
You: /model
Bot: Current model: GPT-5 Mini
Bot: Choose a model: [buttons]
You: /project
Bot: Project workflow started. Model locked to: GPT-5 Mini
Bot: Step 1 - What project do you want to build? Describe it in detail.
You: Build a FastAPI task manager with JWT auth and SQLite.
Bot: Step 2 - Which language/stack? (e.g. Python, Node.js, React, FastAPI)
You: Python + FastAPI
Bot: Step 3 - Any special requirements? (libraries, constraints, architecture)
You: Use SQLAlchemy async and Alembic.
Bot: Step 4 - Push to GitHub when done? (yes / no)
You: yes
Bot: Repo name?
You: fastapi-task-manager
Bot: Public or private?
You: private
Bot: Step 5 - Summary ... Ready to build? (yes / no)
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

Only these models are available through `/model`, with default `gpt-5-mini`:

- `gpt-5.3-codex`
- `gpt-5.2-codex`
- `gpt-5.2`
- `gpt-5.4-mini`
- `gpt-5-mini`
- `gpt-4.1`
- `claude-haiku-4.5`

Project generation runs only when you send `/project`.

## Troubleshooting

### Copilot authentication issues

- Ensure Copilot CLI is installed and available in `PATH`.
- Run `copilot auth login` and confirm success.
- Send `/start` again.

### Build failed

- Use `/status` to inspect progress.
- Review the error summary sent by the bot.
- If generated code still fails after retries, refine requirements and restart with `/project`.
- If a specific model fails, switch model with `/model` and retry `/project`.

### GitHub push failed

- Set `GITHUB_TOKEN` in `.env` to a PAT with repository create/push permissions.
- Confirm the token owner can create repos under `GITHUB_USERNAME` (user or org).
- Ensure local Git is installed and configured.
- Try with `Push to GitHub = no` to validate local generation first.
