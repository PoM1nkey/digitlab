"""Download pretrained MNIST weights from a pinned Hugging Face revision."""
import hashlib
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URL = "https://huggingface.co/onnxmodelzoo/mnist-12/resolve/1cd752f4a3d818fa215aa621c78b8a10a4a1a3a5/mnist-12.onnx"


def main():
    destination = ROOT / "artifacts/mnist-12.onnx"
    destination.parent.mkdir(exist_ok=True)
    with urllib.request.urlopen(URL, timeout=60) as response:
        data = response.read()
    if len(data) < 1000 or data.startswith(b"version https://git-lfs"):
        raise RuntimeError("Сервер не вернул ONNX модель")
    checksum_file = ROOT / "artifacts/model.sha256"
    digest = hashlib.sha256(data).hexdigest()
    if checksum_file.exists() and checksum_file.read_text().split()[0] != digest:
        raise RuntimeError("SHA256 модели отличается от закреплённого значения")
    destination.write_bytes(data)
    print(f"Downloaded {len(data)} bytes, SHA256 {digest}")


if __name__ == "__main__":
    main()
