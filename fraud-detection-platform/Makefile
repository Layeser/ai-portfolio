PY ?= python
export PYTHONPATH := src
.PHONY: install data train drift serve test test-llm lint agent evaluate-agent docker all

install:        ## dependencies into the active (shared) venv
	$(PY) -m pip install -r requirements.txt
data:           ## simulate transactions (training period + drifted production period)
	$(PY) -m fraudguard.data.simulate
train:          ## train rules / logistic / LightGBM, log to MLflow, write figures
	$(PY) -m fraudguard.train
drift:          ## PSI drift report on the simulated production period
	$(PY) -m fraudguard.monitoring.drift
serve:          ## scoring API on http://127.0.0.1:8000/docs
	uvicorn fraudguard.api.main:app --reload
agent:          ## investigate the 3 highest-risk alerts (needs GROQ_API_KEY)
	$(PY) -m fraudguard.agent.run --top 3
evaluate-agent: ## compare agent verdicts with ground truth on 16 alerts
	$(PY) -m fraudguard.agent.run --evaluate 16
test:           ## offline test suite (no API key needed)
	$(PY) -m pytest -m "not llm"
test-llm:       ## end-to-end agent test with a real LLM
	$(PY) -m pytest -m llm
lint:
	ruff check src tests && ruff format --check src tests
mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db
docker:
	docker compose up --build
all: data train drift test
