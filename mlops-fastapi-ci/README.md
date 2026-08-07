# MLOps FastAPI + MLflow + Docker

Small but complete **train → track → serve → containerize** demo.

Useful as a clean GitHub pin to show MLOps fundamentals without shipping a huge research repo.

## Pipeline

```
train.py  →  MLflow run + model.pkl
                ↓
            FastAPI /predict
                ↓
            Docker image (uvicorn)
```

## Tech stack

scikit-learn · MLflow · FastAPI · Uvicorn · Docker · (optional) GitHub Actions

## Install & run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python train.py                 # trains, logs to MLflow, writes model.pkl
uvicorn app:app --reload        # http://127.0.0.1:8000/docs
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"x": 5.0}'
```

## Docker

```bash
python train.py                 # ensure model.pkl exists
docker build -t mlops-api .
docker run --rm -p 8000:8000 mlops-api
```

## MLflow UI

```bash
mlflow ui --backend-store-uri ./mlruns
```

## Suggested CI (GitHub Actions)

```yaml
# .github/workflows/ci.yml
name: ci
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt
      - run: python train.py
      - run: python -c "from app import app; print('import ok')"
```

## Scope

This is intentionally a **toy regression** (y ≈ 2x). The value is the packaging pattern, not the model accuracy.
