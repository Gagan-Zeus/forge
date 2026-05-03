# Forge

An Autonomous End to End Project Builder

Forge is a Telegram-triggered autonomous coding agent that turns a project idea into a locally generated codebase, validates it, retries fixes when needed, and can optionally create and push a GitHub repository. It now uses the GitHub Copilot SDK (Python) and the local Copilot CLI session for model access.

## Prerequisites

- Python 3.11+
- A Telegram bot token from BotFather
- GitHub Copilot CLI installed and available in `PATH`
- Git installed and available in `PATH`
- (Optional) GitHub account with permissions to create repositories

## Setup

1. Clone the repository and enter it:
   ```bash
   git clone https://github.com/Gagan-Zeus/forge.git
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
    - `PROJECTS_DIR` (Change to your Preffered Directory. default: `./generated_projects`)
    - `SYSTEM_PROMPT_PATH` (optional, default: `./system-prompt.txt`; loaded as hidden base system prompt)

## How To Run

```bash
python run.py
```

## Easier Daily Run (Background)

If you do not want to keep a terminal tab occupied, use the helper script:

```bash
./scripts/forgectl.sh start
```

Useful commands:

```bash
./scripts/forgectl.sh status
./scripts/forgectl.sh logs
./scripts/forgectl.sh stop
```

Notes:
- The script uses `.venv/bin/python` automatically when available.
- Runtime files are stored in `.forge/` (`forge.pid`, `forge.log`).

## First Run Auth (Copilot SDK + CLI)

Authenticate your local Copilot CLI once:

```bash
copilot auth login
```

Then run the bot and send `/start`. The bot checks CLI auth state through the SDK and enters chatbot mode.

## Copilot Models

Only these models are available through `/model`, with default `gpt-5-mini`:

- `gpt-5.3-codex`
- `gpt-5.2-codex`
- `gpt-5.2`
- `gpt-5.4-mini`
- `gpt-5-mini`
- `gpt-4.1`
- `claude-haiku-4.5`

## Project Build Command

Project generation runs **only** through `/create`.

- Example: `/create build a hello world html page`
- Regular chat messages (even if they mention "create", "build", or similar terms) stay in normal Copilot chat mode.
- The builder now runs in explicit phases with Telegram status updates: **PLAN -> BUILD -> README -> FINAL**.
- Files are generated incrementally and written to disk immediately (not as one giant final response).
- `/create` focuses on project creation and does not run post-generation tests/validation.
- Dependency manifests (`requirements.txt`, `package.json`) are refreshed to current registry versions during generation.

## GitHub Push Command

The `/github` command allows you to push the active project to a GitHub repository.

- Usage: `/github <repo_name> [--branch <branch_name>]`
- Example: `/github my-awesome-project --branch develop`
- If no branch is specified, it defaults to `main`.
- The repository name is required for the first push; subsequent pushes to the same repo will use the saved name if not provided.

## Project Selection Command

The `/project` command allows you to select an existing project from your directory.

- Usage: `/project`
- Telegram will display inline buttons for each project directory found in your `PROJECTS_DIR`.
- Selecting a project sets it as active, allowing you to use `/update` or `/github` immediately.

## Install Command

The `/install` command analyzes your active project, automatically installs dependencies, validates the project, and **intelligently fixes issues** if install or validation fails.

- Usage: `/install`
- Automatically detects the project type based on dependency files (package.json, requirements.txt, etc.)
- For Python projects, creates/uses `.venv` and runs checks through the venv Python
- After install, runs safe validation checks such as Python compile/startup checks or npm build/test scripts
- If validation fails, fixes affected files file-by-file and retries
- Runs the appropriate install command(s):
  - **Node.js**: `npm install` (detects package.json)
  - **Python pip**: `pip install -r requirements.txt` (with automatic .venv detection/creation)
  - **Python Poetry**: `poetry install` (detects and uses poetry.lock)
  - **Python PDM**: `pdm install` (detects pdm.lock)
  - **Python uv**: `uv sync` or `uv pip install` (fastest Python package manager)
  - **Python Hatch**: `hatch env create` (detects Hatch projects)
  - **Python pipenv**: `pipenv install`
  - **Rust**: `cargo build`
  - **Go**: `go mod download`
  - **Ruby**: `bundle install`
  - **PHP**: `composer install`
  - **Java Maven**: `mvn install`
  - **Java Gradle**: `gradle build`

### Python Virtual Environment Support

For Python projects, the `/install` command automatically:
- **Detects existing virtual environments** (`.venv`, `venv`, `env`, `.env`)
- **Creates a new `.venv`** if none exists
- **Uses the venv's Python/pip** for all install commands
- **Supports all modern Python tools**:
  - **uv** - Ultra-fast Python package installer (preferred when available)
  - **PDM** - Modern Python package manager
  - **Poetry** - Dependency management and packaging
  - **Hatch** - Modern extensible Python project manager
  - **pipenv** - Python dev workflow for humans
  - **pip** - Standard package installer (fallback)

### Smart Error Recovery

If installation fails, the command automatically attempts to fix common issues:

**Automatic Fixes for Node.js:**
- Clears `node_modules` and lock files if corrupted
- Runs `npm audit fix` for security vulnerabilities

**Automatic Fixes for Python:**
- **Creates virtual environment** (`.venv`) if missing
- **Detects and uses** existing virtual environments automatically
- **Supports modern tools**: uv, PDM, Hatch, Poetry, pipenv
- Removes problematic packages that can't be found
- Upgrades pip if needed
- Regenerates lock files (poetry.lock, pdm.lock, uv.lock)
- Installs uv as a fast alternative if pip fails

**Automatic Fixes for Go:**
- Runs `go mod tidy` to clean up dependencies

**AI-Powered Fixes:**
- Uses Copilot to analyze complex errors
- Suggests version updates and dependency corrections
- Applies syntax fixes to dependency files

### Post-Install Validation

After dependencies install successfully, `/install` verifies the project is actually runnable:

- **Python**: runs `python -m compileall -q .` through `.venv`, then probes common entrypoints like `app.py` or `main.py` with a timeout so Flask/FastAPI apps do not hang the bot
- **npm projects**: runs `npm run build` when available and meaningful test scripts when present
- **Go/Rust**: runs standard test/build checks
- **Static/code issues**: detects important runtime warnings like deprecated naive UTC datetime usage and fixes the source file

If checks fail, Copilot receives the error output and relevant files, then `/install` writes minimal file-by-file fixes and runs validation again.

### Example Outputs

**Successful Install:**
```
✅ Dependencies installed successfully!

✅ Node.js/npm: Installed successfully using `npm install`
✅ Python pip: Installed successfully using `pip install -r requirements.txt`

Commands executed:
• npm install
• pip install -r requirements.txt
```

**Install with Fixes:**
```
✅ Dependencies installed successfully!

✅ Node.js/npm: Installed successfully after fixes using `npm install`

🔧 Fixes applied:
• Cleared node_modules and lock file
• Ran npm audit fix

Commands executed:
• npm install
• rm -rf node_modules package-lock.json
• npm audit fix --force
• npm install (after fix)
```

**Failed Install (with attempted fixes):**
```
❌ Installation failed:

❌ Python pip: Failed to install dependencies
   Error: Could not find a version that satisfies the requirement xyz
   Attempted fixes: Removed problematic packages: xyz, Removed syntax errors

🔧 Attempted fixes:
• Removed problematic packages: xyz
• Fixed syntax in requirements.txt
```

## Chatbot Tool Execution

The chatbot mode now supports tool execution! When you chat with the bot, it can:

- **Execute shell commands** - Run commands in your project directory (e.g., "run npm test", "show git status")
- **Read files** - Read contents of any file in your projects (e.g., "read README.md", "show me package.json")
- **List directories** - Show directory contents (e.g., "list the files", "what's in the src folder")
- **Get file info** - See file size, type, and modification date
- **Count files** - Get total file counts in directories
- **Search files** - Find files by pattern (e.g., "find all .js files")
- **Get project structure** - Display a tree view of your project

The chatbot will automatically use these tools when you ask it to perform actions like:
- "Summarize this project" - Reads files and provides a summary
- "Show me the project structure" - Displays a tree view
- "Run the tests" - Executes test commands
- "Check git status" - Runs git commands
- "What files are in this project?" - Lists all files

Note: Commands are sandboxed to your `PROJECTS_DIR` for security. Dangerous commands like `sudo`, `rm -rf /`, `curl`, `wget`, and `eval` are blocked.

## Troubleshooting

### Copilot authentication issues

- Ensure Copilot CLI is installed and available in `PATH`.
- Run `copilot auth login` and confirm success.
- Send `/start` again.

### Build failed

- Use `/status` to inspect progress.
- Review the error summary sent by the bot.
- If generated code needs fixes, refine requirements and restart the chat with `/reset`.
- If a specific model fails, switch model with `/model` and retry.

### GitHub push failed

- Set `GITHUB_TOKEN` in `.env` to a PAT with repository create/push permissions.
- Confirm the token owner can create repos under `GITHUB_USERNAME` (user or org).
- Ensure local Git is installed and configured.
- Try with `Push to GitHub = no` to validate local generation first.
