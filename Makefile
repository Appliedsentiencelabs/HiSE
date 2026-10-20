PYTHON ?= python

.PHONY: test validate-notebooks

test:
	$(PYTHON) -m pytest -q hise/tests

validate-notebooks:
	$(PYTHON) hise/scripts/validate_notebooks.py
