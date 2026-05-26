"""Supplementary experiments

This script contains the supplementary experiments for the paper.
"""
from typing import Union
import numpy as np
import argparse
from copy import deepcopy

from duelpy.algorithms import DoubleThompsonSampling
from duelpy.algorithms import RelativeConfidenceSampling
from duelpy.algorithms import RelativeUCB
from duel_bias.algorithms import UCB
from duel_bias.algorithms import ModifiedBeatTheWinner
from duel_bias.algorithms import DoubleThompsonSamplingBias
from duel_bias.algorithms import RelativeConfidenceSamplingBias
from duel_bias.algorithms import RelativeUCBBias
from duel_bias.algorithms import BiasSensitiveUCB

from scripts_for_experiments.benchmark_utils import benchmark
from scripts_for_experiments.benchmark_utils.examples import generate_task_normal
from scripts_for_experiments.main_experiments import batch_run_table

def run_table_num_arms_evaluators(algos_list, algos_extra_list, task_variance, params_user_bias: dict = None):
    """Run the experiments varying the number of arms and evaluators.

    Args:
        algos_list (list): List of algorithms to run.
        algos_extra_list (list): List of algorithms, which only to presented in table, not in figure.
        task_variance (float): Variance of the task.
        params_user_bias (dict): parameters of beta distribution for simulating evaluators bias.
    """
    if params_user_bias is None:
        params_user_bias = {"alpha": 2.0, "beta": 1.0}
    
    # Result path
    path_to_result = "results/supplement-table-num_arms-arm_variance_" \
                     "{var}-bias_beta_{alpha}_{beta}".format(var=task_variance, **params_user_bias)
    
    # Task parameters (num_arms, num_evaluators, rounds)
    task_params_list = [
        (10, 10, 40000),
        (20, 20, 40000),
        (50, 50, 40000),
        (100, 100, 40000),
    ]

    settings_list = [
        (
            f"supplement-num_arms_{num_arms}",
            generate_task_normal(num_arms, scale=np.sqrt(task_variance), seed=42),
            {"user_num": num_evaluators, **params_user_bias},
            {'time_horizon': rounds, 'run_times': 20}
        )
        for num_arms, num_evaluators, rounds  in task_params_list
    ]
    # Run
    column_name = 'Number of arms and evaluators'
    column_values = [f"{num_arms, num_evaluators}" for num_arms, num_evaluators, _ in task_params_list]
    result = batch_run_table(
        algorithm_dicts=algos_list + algos_extra_list,
        settings=settings_list,
        run_id=path_to_result,
        column_name=column_name,
        column_values=column_values
    )

    # Export
    benchmark.export_table(
        data=result,
        column_name=column_name,
        output_name_prefix=path_to_result,
    )
    benchmark.export_evaluator_bias(
        data=result,
        extra_header=column_name,
        output_name_prefix=path_to_result
    )

    # Split result DataFrame based on column_name values to export figures
    split_results = {}

    for value in column_values:
        split_results[value] = result[(result[column_name] == value) & (~result["algorithm"].isin([p['name'] for _, p in algos_extra_list]))]
    
    for idx, df in enumerate(split_results.values()):
        benchmark.export_figures(
            data=df,
            output_name_prefix=f"{path_to_result}-arm_{task_params_list[idx][0]}"
                          f"-evaluators_{task_params_list[idx][1]}",
            variance=False,
            color_style="main_figure",
        )

def run_speed_testing(algos_list):
    """Run the experiments varying the number of arms and evaluators to test running time.

    Args:
        algos_list (list): List of algorithms to run.
    """
    params_user_bias = {"alpha": 2.0, "beta": 1.0}

    # Remove comparative_user_bias from algorithm parameters
    algos_list_copy = deepcopy(algos_list)
    for algo in algos_list_copy:
        if 'comparative_user_bias' in algo[1]:
            del algo[1]['comparative_user_bias']

    # Result path
    path_to_result = "results/supplement-table-run-speed"

    # Task parameters (num_arms, num_evaluators, rounds)
    task_params_list = [
        (20, 20, 40000),
        (40, 40, 40000),
        (80, 80, 40000),
        (160, 160, 40000),
    ]

    settings_list = [
        (
            f"speed-num_arms_{num_arms}",
            generate_task_normal(num_arms, scale=np.sqrt(4.0), seed=42),
            {"user_num": num_evaluators, **params_user_bias},
            {'time_horizon': rounds, 'run_times': 5, 'is_parallel': False}
        )
        for num_arms, num_evaluators, rounds in task_params_list
    ]

    # Run
    column_name = 'Number of arms and evaluators'
    column_values = [f"{num_arms, num_evaluators}" for num_arms, num_evaluators, _ in task_params_list]
    result = batch_run_table(
        algorithm_dicts=algos_list,
        settings=settings_list,
        run_id=path_to_result,
        column_name=column_name,
        column_values=column_values
    )

    # Export
    benchmark.export_time_table(
        data=result,
        column_name=column_name,
        output_name_prefix=path_to_result,
        aggregation="mean"
    )

def run_bias_checkpoints():
    """Run the experiments of BS-UN to export bias error at different checkpoints.
    """
    # Result path
    path_to_result = "results/supplement-table-bias-checkpoints"

    # Task parameters (num_arms, num_evaluators, task_variance, param_beta, rounds)
    total_rounds = 40000
    task_params_list = [
        # Beta(4,1)
        (10, 10, 4.0, {"alpha": 4.0, "beta": 1.0}, total_rounds),
        (50, 50, 4.0, {"alpha": 4.0, "beta": 1.0}, total_rounds),
        (10, 10, 2.0, {"alpha": 4.0, "beta": 1.0}, total_rounds),
        (50, 50, 2.0, {"alpha": 4.0, "beta": 1.0}, total_rounds),
        (10, 10, 1.0, {"alpha": 4.0, "beta": 1.0}, total_rounds),
        (50, 50, 1.0, {"alpha": 4.0, "beta": 1.0}, total_rounds),
        # Beta(2,1)
        (10, 10, 4.0, {"alpha": 2.0, "beta": 1.0}, total_rounds),
        (50, 50, 4.0, {"alpha": 2.0, "beta": 1.0}, total_rounds),
        (10, 10, 2.0, {"alpha": 2.0, "beta": 1.0}, total_rounds),
        (50, 50, 2.0, {"alpha": 2.0, "beta": 1.0}, total_rounds),
        (10, 10, 1.0, {"alpha": 2.0, "beta": 1.0}, total_rounds),
        (50, 50, 1.0, {"alpha": 2.0, "beta": 1.0}, total_rounds),
    ]

    settings_list = [
        (
            f"supplement-num_arms_{num_arms}",
            generate_task_normal(num_arms, scale=np.sqrt(task_variance), seed=42),
            {"user_num": num_evaluators, **params_user_bias},
            {'time_horizon': rounds, 'run_times': 20}
        )
        for num_arms, num_evaluators, task_variance, params_user_bias, rounds in task_params_list
    ]

    # Run
    column_name = 'BC-Params'
    column_values = [f"{num_arms, num_evaluators, task_variance, params_user_bias["alpha"], params_user_bias["beta"]}"
                     for num_arms, num_evaluators, task_variance, params_user_bias, _ in task_params_list]
    result = batch_run_table(
        algorithm_dicts=[(BiasSensitiveUCB, {'name': 'BS-UN(*)', 'comparative_user_bias': None})],
        settings=settings_list,
        run_id=path_to_result,
        column_name=column_name,
        column_values=column_values
    )

    # Export
    benchmark.export_table_checkpoints(
        data=result,
        extra_header=column_name,
        output_name_prefix=path_to_result,
    )

TASK_SCALE = "task-large-scale"
TASK_SPEED = "task-test-speed"
TASK_BIAS_CHECKPOINTS = "task-bias-checkpoints"

def run_main_table(tasks: Union[list[str], str] = None):
    """Run the main table experiments.
    """

    if tasks is None:
        tasks = [TASK_SCALE, TASK_SPEED]
    elif isinstance(tasks, str):
        tasks = [tasks]

    algos_list = [
        (DoubleThompsonSampling, {'name': 'DT'}),
        (RelativeConfidenceSampling, {'name': 'RC'}),
        (RelativeUCB, {'name': 'RUCB'}),
        (UCB, {'name': 'UCB'}),
        (ModifiedBeatTheWinner, {'name': 'MBTW'}),
        (DoubleThompsonSamplingBias, {'name': 'DT-B(*)', 'comparative_user_bias': None}),
        (RelativeConfidenceSamplingBias, {'name': 'RC-B(*)', 'comparative_user_bias': None}),
        (RelativeUCBBias, {'name': 'RUCB-B(*)', 'comparative_user_bias': None}),
        (BiasSensitiveUCB, {'name': 'BS-UN(*)', 'comparative_user_bias': None}),
        (BiasSensitiveUCB, {'name': 'BS-K(*)', 'assigned_user_bias': None}),
    ]

    # The algorithms only presented in table
    algos_extra_list = [
        # (BiasSensitiveUCB, {'name': 'BS-UN(*) (Exploit)', 'comparative_user_bias': None, 'is_exploit': True}),
        # (BiasSensitiveUCB, {'name': 'BS-K(*) (Exploit)', 'assigned_user_bias': None, 'is_exploit': True}),
    ]

    if TASK_SPEED in tasks:
        print("Test run speed.")
        run_speed_testing(algos_list + algos_extra_list)

    if TASK_SCALE in tasks:
        print("Comparison of different numbers of arms and evaluators.")
        run_table_num_arms_evaluators(algos_list, algos_extra_list, task_variance=4.0)
        run_table_num_arms_evaluators(algos_list, algos_extra_list, task_variance=2.0)
        run_table_num_arms_evaluators(algos_list, algos_extra_list, task_variance=1.0)

    if TASK_BIAS_CHECKPOINTS in tasks:
        run_bias_checkpoints()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", action='store_true', default=False,
                        help="To run the task of testing speed")
    parser.add_argument("--large_scale", action='store_true', default=False,
                        help="To run the task of large scale.")
    parser.add_argument("--bias", action='store_true', default=False,
                        help="To run the task of bias checkpoints.")
    parser.add_argument("--all", action='store_true', default=False,
                        help="To run the all tasks.")
    args = parser.parse_args()

    task_list = []
    if args.large_scale:
        task_list.append(TASK_SCALE)
    if args.speed:
        task_list.append(TASK_SPEED)
    if args.bias:
        task_list.append(TASK_BIAS_CHECKPOINTS)
    if args.all:
        task_list = [TASK_SPEED, TASK_SCALE, TASK_BIAS_CHECKPOINTS]

    if task_list:
        run_main_table(task_list)
