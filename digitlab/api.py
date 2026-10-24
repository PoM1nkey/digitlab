"""HTTP interface. No images are stored on disk."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from digitlab.model import MAX_BYTES, predict, session


@asynccontextmanager
async def lifespan(app):
    session()
    yield


app = FastAPI(title="DigitLab API", version="1.0.0", lifespan=lifespan,
              description="Распознавание одной рукописной цифры от 0 до 9")


@app.get("/health")
def health():
    session()
    return {"status": "ok", "model": "mnist-12"}


@app.post("/predict")
async def recognize(file: UploadFile = File(...)):
    data = await file.read(MAX_BYTES + 1)
    await file.close()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Максимальный размер файла — 5 МБ.")
    try:
        return await run_in_threadpool(predict, data)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
