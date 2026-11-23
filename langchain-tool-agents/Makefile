PY ?= python
export PYTHONPATH := src
.PHONY: install run test
install:        ## dependencies into the active (shared) venv
	$(PY) -m pip install -r requirements.txt
run:     ## interactive agent (GROQ_API_KEY in .env)
	$(PY) -m toolagent.agent
test:
	$(PY) -m pytest -q
