import logging
import os

from dotenv import load_dotenv

from bot.handlers import run_bot

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

if __name__ == "__main__":
    run_bot(
        telegram_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        github_username=os.getenv("GITHUB_USERNAME", ""),
        projects_dir=os.getenv("PROJECTS_DIR", "./generated_projects"),
    )
