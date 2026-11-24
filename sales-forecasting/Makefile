PY ?= python
export PYTHONPATH := src
.PHONY: install data validate test-fold figures serve test

install:          ## dependencies into the active (shared) venv
	$(PY) -m pip install -r requirements.txt
data:             ## download M5 (public mirror, ~50 MB) into data/raw
	mkdir -p data/raw && curl -sL -o data/raw/m5.zip https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip \
	  && cd data/raw && unzip -o -q m5.zip calendar.csv sell_prices.csv sales_train_evaluation.csv sales_test_evaluation.csv && rm m5.zip
validate:         ## every modelling choice is made on the validation fold (d_1914-1941)
	$(PY) -m salesfc.lgbm --fold validation --rounds 600 --train-days 450
	$(PY) -m salesfc.tide --fold validation --epochs 30
test-fold:        ## the M5 evaluation period (d_1942-1969), once, with the settings chosen on validation
	$(PY) -m salesfc.lgbm --fold test --rounds 450 --train-days 450
	$(PY) -m salesfc.tide --fold test --epochs 22
	$(PY) scripts/score_test.py
figures:
	$(PY) scripts/make_figures.py
serve:            ## forecast API
	uvicorn salesfc.api:app
test:
	$(PY) -m pytest -q
