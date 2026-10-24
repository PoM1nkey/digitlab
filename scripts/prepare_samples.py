"""Download the MNIST test split and select one example of each digit."""
import gzip
from pathlib import Path
import urllib.request

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"


def load_test():
    root = ROOT / "artifacts"
    root.mkdir(exist_ok=True)
    arrays = []
    for name, offset in [("t10k-images-idx3-ubyte.gz", 16), ("t10k-labels-idx1-ubyte.gz", 8)]:
        target = root / name
        if not target.exists():
            urllib.request.urlretrieve(BASE + name, target)
        arrays.append(np.frombuffer(gzip.decompress(target.read_bytes()), dtype=np.uint8, offset=offset))
    return arrays[0].reshape(-1, 28, 28), arrays[1]


if __name__ == "__main__":
    images, labels = load_test()
    samples = ROOT / "samples"
    samples.mkdir(exist_ok=True)
    for digit in range(10):
        index = int(np.flatnonzero(labels == digit)[0])
        Image.fromarray(images[index]).save(samples / f"digit_{digit}_mnist_{index}.png")
    print("Saved ten MNIST test examples")
