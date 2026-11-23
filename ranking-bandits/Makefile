PY ?= python
export PYTHONPATH := src
.PHONY: install benchmark serve test lint
install:        ## dependencies into the active (shared) venv
	$(PY) -m pip install -r requirements.txt
benchmark:      ## both scenarios, figures + reports/benchmark.json
	$(PY) -m rankbandit.simulate
serve:          ## online ranking API on http://127.0.0.1:8000/docs
	uvicorn rankbandit.api:app --reload
test:
	$(PY) -m pytest -q
lint:
	ruff check src tests
