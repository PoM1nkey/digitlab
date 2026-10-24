"""Run both services in a single container, stop if either exits."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

os.chdir(Path(__file__).resolve().parents[1])
processes = []


def stop(*_):
    for process in processes:
        process.terminate()


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
processes.append(subprocess.Popen([sys.executable, "-m", "uvicorn", "digitlab.api:app", "--host", "0.0.0.0", "--port", "8000"]))
processes.append(subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", os.getenv("PORT", "7860"), "--server.headless", "true", "--browser.gatherUsageStats", "false"]))
try:
    while all(p.poll() is None for p in processes):
        time.sleep(0.5)
finally:
    stop()
    for process in processes:
        process.wait()
