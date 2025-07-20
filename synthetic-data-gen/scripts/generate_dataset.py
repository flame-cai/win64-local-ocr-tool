# This script is a user-friendly wrapper around the CLI defined in src/cli.py.
# It makes the project structure more intuitive for end-users.

import sys
from pathlib import Path

# Add the project root to the Python path to allow imports from `src`
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cli import app

if __name__ == "__main__":
    # Call the Typer application
    app()