---
title: DigitLab
emoji: ✍️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# DigitLab — распознавание рукописных цифр

Учебный проект по дисциплине «Программная инженерия», КГЭУ.
Streamlit отправляет изображение в FastAPI, API выполняет инференс готовой CNN
MNIST-12 с Hugging Face через ONNX Runtime. Собственная модель не обучается.
Автор: Егор Пескишев, GitHub [PoM1nkey](https://github.com/PoM1nkey).

## Быстрый запуск

Нужен Python 3.11 или 3.12. Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
.venv/Scripts/python.exe scripts/download_model.py
.venv/Scripts/python.exe scripts/serve.py
```

Откройте http://localhost:7860. Swagger API: http://localhost:8000/docs.
Остановка: Ctrl+C. `start.ps1` выполняет установку и запуск автоматически.

Linux / macOS:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python scripts/download_model.py
.venv/bin/python scripts/serve.py
```

Для раздельного запуска: `python -m uvicorn digitlab.api:app --port 8000`
и в другом терминале `python -m streamlit run app.py`. Если API расположен
на другом адресе, задайте серверную переменную окружения `API_URL`.

## Использование

Выберите готовый пример или загрузите PNG/JPEG/WEBP до 5 МБ.
На изображении должна быть одна цифра на однотонном фоне. Приложение
нормализует полярность, выделяет цифру, вписывает её в поле 20×20 внутри
28×28 и центрирует по центру масс. API возвращает класс 0–9 и десять
оценок softmax. Оценки не являются калиброванной уверенностью.

```bash
curl -X POST http://localhost:8000/predict -F file=@samples/digit_7_mnist_0.png
```

`GET /health` проверяет загрузку модели. `POST /predict` возвращает 422 для
повреждённого или пустого изображения, 413 при превышении 5 МБ.
Изображения обрабатываются в памяти без постоянного хранения.

## Модель и данные

- [Модель на Hugging Face](https://huggingface.co/onnxmodelzoo/mnist-12).
- Загрузчик использует фиксированную ревизию `1cd752f4a3d818fa215aa621c78b8a10a4a1a3a5`.
- Модель `mnist-12.onnx`, 26 143 байта, opset 12. В метаданных Hugging Face
  указана Apache-2.0, в тексте карточки и исходном ONNX Model Zoo — MIT.
  Это расхождение сохранено в THIRD_PARTY_NOTICES.md.
- Вход `float32[1,1,28,28]`, выход — десять логитов.
- SHA256 закреплён в `artifacts/model.sha256`; загрузчик отвергает другое содержимое.
- [Тестовая выборка MNIST](https://storage.googleapis.com/cvdf-datasets/mnist/).
  В `samples/` лежит первый пример каждого класса из тестовой выборки,
  выбранный по метке без отбора по результату модели.
- `python scripts/evaluate.py` воспроизводит оценку на первых 1000 тестовых
  примерах и сохраняет фактические значения в `artifacts/evaluation.json`.
  Оценка исходной модели отделена от оценки всего конвейера обработки.

## Тесты и CI

```bash
python -m pytest -q
```

Проверяются реальный инференс, десять примеров, формат вероятностей,
нормализация фона, пустые/повреждённые файлы, отсутствие файла и лимит размера.
`.github/workflows/ci.yml` запускает тесты при каждом push и pull request.
Модель загружается и проверяется перед тестами. Отчёт JUnit сохраняется
как артефакт Actions. Workflow готов локально; факт успешного запуска в GitHub
можно подтвердить только после публикации репозитория.

## Развёртывание

```bash
docker build -t digitlab .
docker run --rm -p 7860:7860 -p 8000:8000 digitlab
```

Подробный порядок публикации: [docs/DEPLOY.md](docs/DEPLOY.md).
Docker запускает два процесса. Streamlit доступен на 7860, FastAPI на 8000.
В Docker Space Hugging Face наружу выходит только 7860, UI обращается к API
внутри контейнера. Для публичного Swagger/API нужен хостинг, который публикует
8000, или отдельный API сервис.

## Материалы практикума

- `notebooks/demo.ipynb` — демонстрация готовой модели в Google Colab.
- `docs/REQUIREMENTS.md` — требования курса и соответствие файлам проекта.
- `docs/DEFENSE.md` — сценарий демонстрации и ответы на вопросы.
- Отчёт и презентация будут дополнены ссылками на опубликованные сервисы
  после фактической публикации.

## Ограничения

Модель предназначена для отдельных цифр MNIST. Она не распознаёт слова,
несколько цифр и показания приборов целиком. Нет детектора «не цифра»:
для произвольного рисунка модель тоже может вернуть высокий softmax.
Нет авторизации, глобального ограничения запросов и промышленного мониторинга.
Перед публичным промышленным использованием нужны эти меры и отдельная
проверка качества на целевых изображениях.

Исходный код приложения — MIT. Условия сторонней модели приведены в
`THIRD_PARTY_NOTICES.md`.
