#!/usr/bin/bash
export PYTHONPATH=./
export PYTHONIOENCODING=utf-8
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    PYTHON=.venv/Scripts/python.exe
else
    PYTHON=.venv/bin/python.exe
fi

LOGDIR=logs
RESULT_DIR=results
EX_PATH=./scripts_for_experiments

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    TIME=$(powershell -Command "Get-Date -Format 'MMdd_HHmm'")
else
    TIME=$(date +%m%d_%H%M)
fi

WRITE_TO_LOG="2>&1 | tee -a $LOGDIR/run_$TIME.log"

mkdir -p "$LOGDIR"
mkdir -p "$RESULT_DIR"

# Run Experiments -----------------------
echo "Compare bias estimation"
$PYTHON $EX_PATH/compare_bias_estimation.py $WRITE_TO_LOG

echo "Compare preference estimation"
$PYTHON $EX_PATH/compare_preference_estimation.py $WRITE_TO_LOG

echo "Plot main figure"
$PYTHON $EX_PATH/main_experiments.py --run_figure $WRITE_TO_LOG

echo "Compute main table"
$PYTHON $EX_PATH/main_experiments.py --run_table $WRITE_TO_LOG
