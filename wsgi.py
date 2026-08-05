"""WSGI entry point for production servers (gunicorn, etc.).

dashboard/app.py uses bare imports (`import auth`, `import chatbot`, ...)
that assume dashboard/ is on sys.path, matching how it's run locally
(`python3 dashboard/app.py`). gunicorn instead loads modules relative to
the repo root, so this shim puts dashboard/ on sys.path first.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "dashboard"))

from app import app  # noqa: E402

if __name__ == "__main__":
    app.run()
