FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 && rm -rf /var/lib/apt/lists/*

COPY serve/requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Model artifacts: SavedModel dir, scaler, feature order
COPY artifacts/saved_model            ./artifacts/saved_model
COPY artifacts/scaler.pkl             ./artifacts/scaler.pkl
COPY artifacts/selected_features.json ./artifacts/selected_features.json
COPY serve/app.py                     ./serve/app.py

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
