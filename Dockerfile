FROM python:3.13-slim-bookworm

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serving.txt ./
RUN python -m pip install --no-cache-dir -r requirements-serving.txt

COPY predict.py ./
COPY artifacts/lightgbm_model.txt artifacts/model_metadata.json ./artifacts/

EXPOSE 8080
CMD ["python", "-m", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8080"]
