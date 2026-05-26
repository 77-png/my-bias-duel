================
BiasDuel
================

Welcome to the BiasDuel repository. This repository facilitates research on dueling bandit algorithms designed to tackle biased evaluators. It is designed for easy replication of all numerical experiments in our paper.

Repository Structure
================

.
├── duel_bias/                # Core implementation
├── duelpy/                   # A library for dueling bandits
├── scripts_for_experiments/  # Scripts for easy replication
└── demo.py                   # Demo script for usage

The repository is organized into several main components:

* duel_bias/: Extends the duelpy library to specifically address evaluator bias in dueling bandits. It includes various bias-sensitive algorithms and mechanisms for bias estimation and handling.
* duelpy/: An existing Python library (duelpy - https://gitlab.com/duelpy/duelpy.git) dedicated to solving dueling bandit problems.
* scripts_for_experiments/: Includes scripts for running benchmark experiments, comparing algorithms, and generating plots/results.
* demo.py: A script to demonstrate the basic usage of the biased dueling bandit algorithms.

Installation
================

Using uv
----------------

Install uv (if not already present) using:
  $ pip install uv

Synchronize Python virtual environment using:
  $ uv sync

Using pip
----------------

Ensure you have Python 3.12 or newer. Then, create and activate a Python virtual environment:
  $ python -m venv .venv

Finally, install the dependencies using:
  $ pip install -r requirements.txt

Running Experiments
================

One-Click Replication
----------------

For easy replication of all numerical experiments, we provide convenient one-click run scripts. To run all experiments, execute either the shell command:
  $ sh run_all_experiments.sh

or using make:
  $ make run_all

Either command will automatically run all the experiments and save the results to the results/ directory.

Individual Experiment Commands
----------------

Alternatively, individual experiments can be reproduced using the following make commands:

* $ make main_table: Generate comparison tables of different algorithms' performance.
* $ make main_plot: Create figures comparing different algorithms.
* $ make compare_bias: Run ablation experiments for different bias estimation methods.
* $ make compare_preference: Run ablation experiments for different preference estimation methods.
* $ make main_supplement: Reproduce other supplementary experiments in the appendix.
