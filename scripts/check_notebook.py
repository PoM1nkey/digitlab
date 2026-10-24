"""Execute the demo's Python cells locally; this does not claim a Colab run."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
notebook = json.loads((ROOT / "notebooks/demo.ipynb").read_text(encoding="utf-8"))
build = ROOT / ".build/notebook"
build.mkdir(parents=True, exist_ok=True)
os.chdir(build)
scope = {"display": lambda image: print("Displayed image", image.size)}
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if source.startswith("%pip"):
            continue
        exec(compile(source, "demo.ipynb", "exec"), scope)
print("Notebook Python cells passed locally")
