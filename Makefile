PY ?= python3

.PHONY: install test smoke quick full figures clean
install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .
test:
	$(PY) -m pytest -q
smoke:
	$(PY) scripts/run_all.py --scale smoke --out results/scratch
quick:
	$(PY) scripts/run_all.py --scale quick --out results/preliminary
full:
	$(PY) scripts/run_all.py --scale full --out results/full
figures:
	$(PY) scripts/make_figures.py --in results/preliminary --out results/preliminary
clean:
	rm -rf results/scratch .pytest_cache
