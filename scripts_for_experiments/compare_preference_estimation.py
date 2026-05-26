import os
from contextlib import contextmanager
import numpy as np
import pandas as pd
import inspect

from duel_bias.algorithms import BiasSensitiveUCB
from scripts_for_experiments.benchmark_utils.bradley_terry_luce_model import BradleyTerryLuceModelBias
from scripts_for_experiments.benchmark_utils import benchmark
from scripts_for_experiments.benchmark_utils.examples import generate_task_normal
from scripts_for_experiments.benchmark_utils.examples import get_user_bias_beta
from scripts_for_experiments.benchmark_utils.examples import print_info_problem

import duel_bias.config.preference_estimation as config_preference_estimation

def get_classes_with_type(module):
    objs = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ == module.__name__:
            objs.append(obj)
    return objs

@contextmanager
def switch_config_value(new_config_value):
    original_config = config_preference_estimation.Config
    config_preference_estimation.Config = new_config_value
    try:
        yield
    finally:
        config_preference_estimation.Config = original_config

def run(task_config):
    # Settings
    num_arms = 10
    task_variance = 4.0
    user_num = 10
    params_user_bias = (2.0, 1.0)
    random_state = np.random.default_rng(42)

    # Generate task
    task = generate_task_normal(num_arms, scale=np.sqrt(task_variance))

    # Generate user bias
    alpha, beta = params_user_bias
    user_bias = get_user_bias_beta(user_num=user_num, alpha=alpha, beta=beta)

    time_horizon = 2000
    run_times = 100

    task_name = task_config.name if hasattr(task_config, 'name') else task_config.__name__
    if task_config.__name__ == 'WeightIdentityN': # Mark the best one
        comparison_list = [
            (BiasSensitiveUCB, {'name': f"{task_name}-UN(*)"}),
            (BiasSensitiveUCB, {'name': f"{task_name}-K(*)", 'assigned_user_bias': user_bias}),
        ]
    else:
        comparison_list = [
            (BiasSensitiveUCB, {'name': f"{task_name}-UN"}),
            (BiasSensitiveUCB, {'name': f"{task_name}-K", 'assigned_user_bias': user_bias}),
        ]
    algorithms, parameters_list = zip(*comparison_list)

    env = BradleyTerryLuceModelBias(num_arms=num_arms, random_state=random_state, skill_vector=task, user_bias=user_bias)
    print_info_problem(env.preference_matrix.preferences, user_bias)
    result = benchmark.run_experiment(algorithms=algorithms, time_horizon=time_horizon, num_arms=num_arms,
                                      sample_interval=1, environment_class=lambda **kwargs: env,
                                      runs=run_times, list_optional_parameters=parameters_list)
    return result

def batch_run(identifier):
    saved_filename = identifier + '.pk1'

    if os.path.exists(saved_filename):
        print("Load previously saved data: {filename}".format(filename=saved_filename))
        collected_data = pd.read_pickle(saved_filename)
    else:
        collected_data = pd.DataFrame()

        # config_list = get_classes_with_type(config_preference_estimation)
        config_list = [
            config_preference_estimation.WeightBiasInv,
            config_preference_estimation.WeightIdentityN
        ]
        for config in config_list:
            with switch_config_value(config):
                print(
                    "Consider config:",
                    config.name if hasattr(config, 'name') else config.__name__
                )
                single_data = run(config)
                collected_data = pd.concat([collected_data, single_data], ignore_index=True)
        collected_data.to_pickle(identifier + '.pk1')
    return collected_data

if __name__ == '__main__':
    path_to_result = 'results/compare-preference_estimation'
    data = batch_run(path_to_result)
    benchmark.export_figures(data, output_name_prefix=path_to_result, color_style='preference')
    benchmark.export_table(data, column_name=None, output_name_prefix=path_to_result)
