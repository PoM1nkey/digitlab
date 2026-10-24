"""Image validation, MNIST preprocessing and ONNX inference."""
import io
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 16_000_000


def preprocess(data: bytes) -> np.ndarray:
    if not data or len(data) > MAX_BYTES:
        raise ValueError("Изображение должно занимать от 1 байта до 5 МБ.")
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.format not in {"PNG", "JPEG", "WEBP"}:
                raise ValueError("Поддерживаются PNG, JPEG и WEBP.")
            if source.width * source.height > MAX_PIXELS:
                raise ValueError("Изображение слишком большое: максимум 16 мегапикселей.")
            image = ImageOps.exif_transpose(source).convert("RGBA")
            background = Image.new("RGBA", image.size, "white")
            background.alpha_composite(image)
            gray = np.asarray(background.convert("L"), dtype=np.float32)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Не удалось прочитать изображение.") from exc
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    if float(np.median(border)) > 127:
        gray = 255 - gray
    gray = np.maximum(gray - float(np.median(np.concatenate(
        (gray[0], gray[-1], gray[:, 0], gray[:, -1])))), 0)
    if float(gray.max()) < 20:
        raise ValueError("На изображении не видна цифра.")
    mask = gray > max(20, float(gray.max()) * 0.2)
    ys, xs = np.nonzero(mask)
    crop = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    image = Image.fromarray(crop.clip(0, 255).astype(np.uint8))
    image.thumbnail((20, 20), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(image, ((28 - image.width) // 2, (28 - image.height) // 2))
    pixels = np.asarray(canvas, dtype=np.float32) / 255.0
    # MNIST centres the digit using its centre of mass.
    yy, xx = np.indices(pixels.shape)
    mass = float(pixels.sum())
    dx = round(13.5 - float((xx * pixels).sum()) / mass)
    dy = round(13.5 - float((yy * pixels).sum()) / mass)
    shifted = Image.new("L", (28, 28), 0)
    shifted.paste(canvas, (dx, dy))
    return (np.asarray(shifted, dtype=np.float32) / 255).reshape(1, 1, 28, 28)


@lru_cache(maxsize=1)
def session() -> ort.InferenceSession:
    path = Path(os.getenv("MODEL_PATH", str(ROOT / "artifacts/mnist-12.onnx")))
    if not path.is_file():
        raise RuntimeError("Модель отсутствует. Выполните python scripts/download_model.py")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options,
                               providers=["CPUExecutionProvider"])


def predict(data: bytes) -> dict:
    tensor = preprocess(data)
    runtime = session()
    logits = runtime.run(None, {runtime.get_inputs()[0].name: tensor})[0].reshape(-1)
    scores = np.exp(logits - logits.max())
    probabilities = scores / scores.sum()
    digit = int(probabilities.argmax())
    return {"digit": digit, "confidence": float(probabilities[digit]),
            "probabilities": [float(p) for p in probabilities],
            "model": "ONNX Model Zoo MNIST-12",
            "warning": "Оценка softmax не гарантирует правильность распознавания."}
