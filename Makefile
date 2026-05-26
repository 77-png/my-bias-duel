SHELL := /usr/bin/bash
ifeq ($(OS), Windows_NT)
    PYTHON := .venv/Scripts/python.exe
else
    PYTHON := .venv/bin/python.exe
endif
export PYTHONPATH=./
export PYTHONIOENCODING=utf-8

LOGDIR = logs
$(LOGDIR):
	mkdir -p "$(LOGDIR)"

ifeq ($(OS), Windows_NT)
    TIME := $(shell powershell -Command "Get-Date -Format 'MMdd_HHmm'")
else
    TIME := $(shell date +%m%d_%H%M)
endif

WRITE_TO_LOG := 2>&1 | tee -a $(LOGDIR)/run_$(TIME).log

# =======================================
# Run Experiments
# =======================================
RESULT_DIR := results
$(RESULT_DIR):
	mkdir -p "$(RESULT_DIR)"

EX_PATH := ./scripts_for_experiments

compare_bias: $(LOGDIR) $(RESULT_DIR)
	@echo "Compare bias estimation"
	$(PYTHON) $(EX_PATH)/compare_bias_estimation.py $(WRITE_TO_LOG)

compare_preference: $(LOGDIR) $(RESULT_DIR)
	@echo "Compare bias estimation"
	$(PYTHON) $(EX_PATH)/compare_preference_estimation.py $(WRITE_TO_LOG)

main_plot: $(LOGDIR) $(RESULT_DIR)
	@echo "Plot main figure"
	$(PYTHON) $(EX_PATH)/main_experiments.py --run_figure $(WRITE_TO_LOG)

main_table: $(LOGDIR) $(RESULT_DIR)
	@echo "Compute main table"
	$(PYTHON) $(EX_PATH)/main_experiments.py --run_table $(WRITE_TO_LOG)

main_supplement: $(LOGDIR) $(RESULT_DIR)
	@echo "Compute main supplement"
	$(PYTHON) $(EX_PATH)/main_supplement.py --large_scale $(WRITE_TO_LOG)

.PHONY: compare_bias compare_preference main_plot main_table

# =======================================
# Aux utils
# =======================================
clean_result_dir: $(RESULT_DIR)
	rm -f $(RESULT_DIR)/*

run_all: compare_bias compare_preference main_plot main_table
rerun_all: clean_result_dir run_all

.PHONY: clean_result_dir run_all rerun_all
