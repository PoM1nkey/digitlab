import json
from pathlib import Path

cells = []


def markdown(text):
    cells.append(dict(id=f"cell-{len(cells)}", cell_type="markdown", metadata={}, source=text.splitlines(True)))


def code(text):
    cells.append(dict(id=f"cell-{len(cells)}", cell_type="code", metadata={}, execution_count=None,
                      outputs=[], source=text.splitlines(True)))


markdown("""# DigitLab — готовая модель с Hugging Face
Автор: Егор Пескишев. Модель не обучается.
В Google Colab выберите «Среда выполнения → Выполнить все».
Источник: https://huggingface.co/onnxmodelzoo/mnist-12
""")
code("%pip -q install onnxruntime pillow numpy")
code('''import urllib.request, hashlib, gzip
import numpy as np
from PIL import Image
import onnxruntime as ort
url = "https://huggingface.co/onnxmodelzoo/mnist-12/resolve/1cd752f4a3d818fa215aa621c78b8a10a4a1a3a5/mnist-12.onnx"
weights = urllib.request.urlopen(url).read()
assert hashlib.sha256(weights).hexdigest() == "5c688690f8bacf667d4c2074af5ad0646ca328d7ab03eccf944a65b320171bdd"
with open("mnist-12.onnx", "wb") as f:
    f.write(weights)
runtime = ort.InferenceSession("mnist-12.onnx", providers=["CPUExecutionProvider"])
print("Форма входа:", runtime.get_inputs()[0].shape)
''')
markdown("## Инференс на фиксированном примере тестовой выборки MNIST")
code('''raw = urllib.request.urlopen("https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz").read()
images = np.frombuffer(gzip.decompress(raw), dtype=np.uint8, offset=16).reshape(-1,28,28)
image = images[0]
display(Image.fromarray(image).resize((224,224)))
tensor = image.astype(np.float32).reshape(1,1,28,28)/255
logits = runtime.run(None, {runtime.get_inputs()[0].name:tensor})[0].reshape(-1)
probabilities = np.exp(logits-logits.max())
probabilities /= probabilities.sum()
print("Результат:", int(probabilities.argmax()), "Ожидаемая метка: 7")
print("Вероятности:", probabilities)
assert int(probabilities.argmax()) == 7
''')
markdown("""## Вывод
Готовая модель загружена с Hugging Face, контрольная сумма проверена.
Инференс выполнен без обучения. Для первой цифры MNIST ожидается класс 7.
Модель распознаёт одну цифру. Softmax не гарантирует правильность результата.
Для произвольных фотографий нужна отдельная проверка качества.
""")
root = Path(__file__).resolve().parents[1] / "notebooks"
root.mkdir(exist_ok=True)
(root / "demo.ipynb").write_text(json.dumps(dict(cells=cells,
    metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    nbformat=4, nbformat_minor=5), ensure_ascii=False, indent=2), encoding="utf-8")
