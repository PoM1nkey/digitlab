"""Streamlit Community Cloud entrypoint. Select this file when deploying."""
import os
import runpy

from digitlab.cloud import ensure_api
from digitlab.model import ROOT

if os.getenv("API_URL") in (None, "http://127.0.0.1:8000"):
    os.environ["API_URL"] = ensure_api()
runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
