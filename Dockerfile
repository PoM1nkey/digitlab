FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python scripts/download_model.py
EXPOSE 7860
CMD ["python", "scripts/serve.py"]
