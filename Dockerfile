FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 && rm -rf /var/lib/apt/lists/*

COPY serve/requirements-serve.txt serve/requirements-serve.txt
RUN pip install --no-cache-dir -r serve/requirements-serve.txt

# Model artifacts: SavedModel dir, scaler, feature order (+ optional tuned thresholds)
COPY artifacts/saved_model ./artifacts/saved_model
COPY artifacts/scaler.pkl ./artifacts/scaler.pkl
COPY artifacts/selected_features.json ./artifacts/selected_features.json
COPY artifacts/thresholds*.json ./artifacts/
# common.py is needed to unpickle scaler.pkl (signed_log1p)
COPY common.py ./serve/common.py
COPY serve/app.py ./serve/app.py

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
