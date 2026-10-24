"""Start a private FastAPI process in a Streamlit Community Cloud container."""
import atexit
import subprocess
import sys
import threading
import time

import requests

from digitlab.model import ROOT
from scripts.download_model import main as download_model

_processes = {}
_lock = threading.Lock()


def _healthy(url):
    try:
        response = requests.get(url + "/health", timeout=1)
        return response.ok and response.json().get("model") == "mnist-12"
    except (requests.RequestException, ValueError):
        return False


def ensure_api(port=8000):
    """Reuse the API across Streamlit reruns and concurrent browser sessions."""
    url = f"http://127.0.0.1:{port}"
    with _lock:
        if _healthy(url):
            return url
        if not (ROOT / "artifacts/mnist-12.onnx").exists():
            download_model()
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "digitlab.api:app",
            "--host", "127.0.0.1", "--port", str(port),
        ], cwd=ROOT)
        _processes[port] = process
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("FastAPI не смог запуститься. Проверьте журнал сервиса.")
            if _healthy(url):
                return url
            time.sleep(0.1)
        process.terminate()
        raise RuntimeError("FastAPI не успел запуститься за 20 секунд.")


@atexit.register
def stop_apis():
    for process in _processes.values():
        if process.poll() is None:
            process.terminate()
