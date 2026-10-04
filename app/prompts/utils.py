from __future__ import annotations

from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent

# Read a prompt template from app/prompts/ by file name.
def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / name

    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")

    return path.read_text(encoding="utf-8").strip()


# Fill a loaded template's {placeholders}.
def render_prompt(template: str, **values: str) -> str:
    return template.format(**values)
