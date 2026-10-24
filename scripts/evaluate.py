"""Measure raw model accuracy and complete image pipeline separately."""
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from digitlab.model import session, predict, ROOT
from scripts.prepare_samples import load_test

if __name__ == "__main__":
    images, labels = load_test()
    runtime = session()
    name = runtime.get_inputs()[0].name
    raw_predictions = []
    pipeline_predictions = []
    for image in images[:1000]:
        tensor = image.astype(np.float32).reshape(1, 1, 28, 28) / 255
        raw_predictions.append(int(runtime.run(None, {name: tensor})[0].argmax()))
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="PNG")
        pipeline_predictions.append(predict(buffer.getvalue())["digit"])
    result = {"dataset": "MNIST test split, first 1000 examples in original order",
              "n": 1000,
              "raw_model_accuracy": float(np.mean(np.array(raw_predictions) == labels[:1000])),
              "image_pipeline_accuracy": float(np.mean(np.array(pipeline_predictions) == labels[:1000])),
              "note": "A fixed test subset, not a guarantee for photos or arbitrary handwriting."}
    (ROOT / "artifacts/evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
