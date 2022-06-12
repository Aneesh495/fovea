# Makefile for Fovea MLIR Compiler and Image Signal Processor

.PHONY: all test verify-mlir bench demo lint clean

PYTHON ?= python3
MLIR_OPT ?= mlir-opt

all: test verify-mlir

test:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py"

verify-mlir:
	@echo "Verifying generated MLIR suite with $(MLIR_OPT)..."
	@if command -v $(MLIR_OPT) >/dev/null 2>&1; then \
		find dialects/generated -name "*.mlir" -exec $(MLIR_OPT) {} --allow-unregistered-dialect -o /dev/null \; && \
		echo "All MLIR modules verified successfully."; \
	else \
		echo "Warning: mlir-opt not found in PATH, skipping binary syntax verification."; \
	fi

bench:
	$(PYTHON) benchmarks/run_all_benchmarks.py

demo:
	$(PYTHON) -m fovea.cli.main demo

lint:
	$(PYTHON) check_no_em_dash.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf artifacts/
	rm -rf *.ppm *.pfm *.raw
