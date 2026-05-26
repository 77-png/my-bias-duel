import argparse
import os
import numpy as np
import pandas as pd
from numpy import ndarray

from scripts_for_experiments.benchmark_utils.bradley_terry_luce_model import BradleyTerryLuceModelBias
from scripts_for_experiments.benchmark_utils import benchmark
from scripts_for_experiments.benchmark_utils.examples import generate_task_normal
from scripts_for_experiments.benchmark_utils.examples import get_user_bias_beta
from scripts_for_experiments.benchmark_utils.examples import print_info_problem

from duelpy.algorithms.interfaces import CondorcetProducer
from duelpy.algorithms import DoubleThompsonSampling
from duelpy.algorithms import RelativeConfidenceSampling
from duelpy.algorithms import RelativeUCB
from duel_bias.algorithms import UCB
from duel_bias.algorithms import ModifiedBeatTheWinner
from duel_bias.algorithms import DoubleThompsonSamplingBias
from duel_bias.algorithms import RelativeConfidenceSamplingBias
from duel_bias.algorithms import RelativeUCBBias
from duel_bias.algorithms import BiasSensitiveUCB
from duel_bias.algorithms.my_bt_bias_algo import MyBTBiasAlgo
from duel_bias.algorithms.my_group_bias_ucb import MyGroupBiasUCB
from duel_bias.algorithms.my_thompson_beta_plus import MyThompsonBetaPlus

def run(task: ndarray, bias: ndarray, settings: list[tuple[type[CondorcetProducer], dict]], options: dict = None) -> pd.DataFrame:
    """Run experiments according to a list of settings.

    Parameters
    --------
    task
        The example to benchmark different algorithms.
    bias
        The bias of users who give feedback at each round.
    settings
        A list of settings, each item is the (algorithm, dictionary of optional parameters) tuple
    options
        Additional parameters
    """
    # Get settings
    algorithms, parameters_list = zip(*settings)

    for optional_parameters in parameters_list:
        if 'assigned_user_bias' in optional_parameters:
            optional_parameters['assigned_user_bias'] = bias
        
        if 'comparative_user_bias' in optional_parameters:
            optional_parameters['comparative_user_bias'] = bias

    # Generate example
    number_arms = task.shape[0]
    random_state = np.random.default_rng(42)

    # Set parameters
    time_horizon = options['time_horizon'] if 'time_horizon' in options else 4000
    run_times = options['run_times'] if 'run_times' in options else 50
    n_jobs = 1 if 'is_parallel' in options and options['is_parallel'] is False else -2

    # Run
    env = BradleyTerryLuceModelBias(num_arms=number_arms, random_state=random_state, skill_vector=task, user_bias=bias)
    print_info_problem(env.preference_matrix.preferences, bias)
    data = benchmark.run_experiment(algorithms=algorithms, time_horizon=time_horizon, num_arms=number_arms,
                                    sample_interval=1, environment_class=lambda **kwargs: env,
                                    runs=run_times, n_jobs=n_jobs, base_random_seed=0,
                                    list_optional_parameters=parameters_list)
    return data

def batch_run_figure(tasks, bias):
    data_list = []
    for (task_id, task, settings, options) in tasks:
        saved_filename = task_id + '.pk1'
        if os.path.exists(saved_filename):
            print("Load previously saved data: {filename}".format(filename=saved_filename))
            data = pd.read_pickle(saved_filename)
        else:
            print("Running {task}".format(task=task_id))
            data = run(task, bias, settings, options)
            data.to_pickle(saved_filename)
        data_list.append(data)
    return data_list

def batch_run_table(algorithm_dicts, settings, run_id, column_name, column_values):
    results = pd.DataFrame()
    saved_filename = run_id + '.pk1'

    try:
        results = pd.read_pickle(saved_filename)
        print("Successfully load previously saved data: {filename}".format(filename=saved_filename))

        return results
    except EOFError as e:
        print(f"Error loading {saved_filename}: {e}. Deleting and rerunning the experiment.")
        os.remove(saved_filename)
    except FileNotFoundError:
        print("Running the experiment.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

    for index in range(len(settings)): # setting == (path_to_results, task, params_bias, additional params)
        task_id, task, params_bias, options = settings[index]
        print("Running {task}".format(task=task_id))
        user_bias = get_user_bias_beta(**params_bias)
        result = run(task, user_bias, algorithm_dicts, options)
        result[column_name] = column_values[index]
        results = pd.concat([results, result], ignore_index=True)

    results.to_pickle(run_id + ".pk1")
    return results

def run_main_table_user_bias(algos_list, task_variance: float = 4.0):
    # Result path
    path_to_result = f"results/table-user_bias-arm_variance_{task_variance}"

    # Set params
    num_arms = 10
    num_users = 10
    seed = 42

    # Different user bias ---------------
    task = generate_task_normal(num_arms, scale=np.sqrt(task_variance), seed=seed)
    user_bias_settings = [
        ("beta41", {"alpha": 4.0, "beta": 1.0}),
        ("beta31", {"alpha": 3.0, "beta": 1.0}),
        ("beta21", {"alpha": 2.0, "beta": 1.0}),
        ("beta11", {"alpha": 1.0, "beta": 1.0}),
        ("No-Bias", {"alpha": 1.0, "beta": 0.1, "center": 1.0}),
    ]

    # setting == (task_id, task, params_bias, additional params)
    if task_variance > 2.0:
        rounds = 4000
    else:
        rounds = 10000

    settings_list = [
        (
            f"{name}-task-variance{task_variance}",
            task,
            {"user_num": num_users, **bias_setting},
            {'time_horizon': rounds, 'run_times': 100}
        )
        for name, bias_setting in user_bias_settings
    ]

    # Run
    column_name = 'user_bias'
    column_values = ["Beta({alpha}, {beta})".format(**params) for _, params in user_bias_settings]
    column_values[-1] = "No-Bias"
    result = batch_run_table(
        algorithm_dicts=algos_list,
        settings=settings_list,
        run_id=path_to_result,
        column_name=column_name,
        column_values=column_values,
    )

    # Export
    benchmark.export_table(
        data=result,
        column_name=column_name,
        output_name_prefix=path_to_result,
    )

def run_main_table_normal_task(algos_list, params_user_bias: dict = None, task_variance_list: list[float] = None):
    """

    Parameters
    --------
    algos_list
        A list of settings, each item is the (algorithm, dictionary of optional parameters) tuple
    params_user_bias
        Parameters of beta distribution
    task_variance_list
        Different variance of arm score
    """
    if params_user_bias is None:
        params_user_bias = {"alpha": 2.0, "beta": 1.0}

    # Default task list
    if task_variance_list is None:
        task_variance_list = [4.0, 3.0, 2.0, 1.0]

    # Result path
    path_to_result = "results/table-arm_variance-user_bias_beta_{alpha}_{beta}".format(**params_user_bias)

    # Set params
    num_arms = 10
    num_users = 10
    seed = 42

    # setting == (task_id, task, params_bias, additional params)
    settings_list = [
        (
            f"comparison-preference_{variance}",
            generate_task_normal(num_arms, scale=np.sqrt(variance), seed=seed),
            {"user_num": num_users, **params_user_bias},
            {'time_horizon': 10000, 'run_times': 100}
        )
        for variance in task_variance_list
    ]

    # Run
    column_name = 'arm_variance'
    column_values = ["Variance = {}".format(task_variance) for task_variance in task_variance_list]
    result = batch_run_table(
        algorithm_dicts=algos_list,
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

def run_main_table_num_arms(algos_list, task_variance: float = 4.0, params_user_bias: dict = None, num_arms_list: list[int] = None):
    if params_user_bias is None:
        params_user_bias = {"alpha": 2.0, "beta": 1.0}

    # Different numbers of arms
    if num_arms_list is None:
        # num_arms_list = [5, 15, 20]
        num_arms_list = [10, 15, 20, 50]

    # Result path
    path_to_result = "results/table-num_arms-arm_variance_{var}-user_bias_beta_{alpha}_{beta}".format(var=task_variance, **params_user_bias)

    # Set params
    num_users = 10
    seed = 42

    # setting == (task_id, task, params_bias, additional params)
    if task_variance > 2.0:
        rounds = 4000
    else:
        rounds = 10000

    settings_list = [
        (
            f"comparison-num_arms_{num_arms}",
            generate_task_normal(num_arms, scale=np.sqrt(task_variance), seed=seed),
            {"user_num": num_users, **params_user_bias},
            {'time_horizon': rounds, 'run_times': 100}
        )
        for num_arms in num_arms_list
    ]

    # Run
    column_name = 'num_arms'
    column_values = ["Num_Arms = {}".format(num_arms) for num_arms in num_arms_list]
    result = batch_run_table(
        algorithm_dicts=algos_list,
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

def run_main_table_num_users(algos_list, task_variance: float = 4.0, params_user_bias: dict = None, num_users_list: list[int] = None):
    if params_user_bias is None:
        params_user_bias = {"alpha": 2.0, "beta": 1.0}

    # Different numbers of users
    if num_users_list is None:
        num_users_list = [5, 15, 20]

    # Result path
    path_to_result = "results/table-num_users-arm_variance_{var}-user_bias_beta_{alpha}_{beta}".format(var=task_variance, **params_user_bias)

    # Set params
    num_arms = 10
    seed = 42

    # setting == (task_id, task, params_bias, additional params)
    if task_variance > 2.0:
        rounds = 4000
    else:
        rounds = 10000

    task = generate_task_normal(num_arms, scale=np.sqrt(task_variance), seed=seed)
    settings_list = [
        (
            f"comparison-num_users_{num_users}",
            task,
            {"user_num": num_users, **params_user_bias},
            {'time_horizon': rounds, 'run_times': 100}
        )
        for num_users in num_users_list
    ]

    # Run
    column_name = 'num_users'
    column_values = ["Num_Users = {}".format(num_users) for num_users in num_users_list]
    result = batch_run_table(
        algorithm_dicts=algos_list,
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

def run_main_figure(is_show: bool = False, is_save: bool = True):
    # Result path
    path_to_result = 'results/comparison-algos'

    # Set params
    num_arms = 10
    num_users = 10
    params_bias = (2.0, 1.0)  # alpha, beta
    seed = 42

    # Generate user bias
    alpha, beta = params_bias
    user_bias = get_user_bias_beta(user_num=num_users, alpha=alpha, beta=beta)

    # Algorithms
    comparison_list = [
        (DoubleThompsonSampling, {'name': 'DT'}),
        (RelativeConfidenceSampling, {'name': 'RC'}),
        (RelativeUCB, {'name': 'RUCB'}),
        (UCB, {'name': 'UCB'}),
        (ModifiedBeatTheWinner, {'name': 'MBTW'}),
        (DoubleThompsonSamplingBias, {'name': 'DT-B(*)'}),
        (RelativeConfidenceSamplingBias, {'name': 'RC-B(*)'}),
        (RelativeUCBBias, {'name': 'RUCB-B(*)'}),
        (BiasSensitiveUCB, {'name': 'BS-UN(*)'}),
        (BiasSensitiveUCB, {'name': 'BS-K(*)', 'assigned_user_bias': user_bias}),
        (MyBTBiasAlgo, {'name': 'MyBT', 'lr': 0.1, 'est_steps': 10}),
        (MyGroupBiasUCB, {'name': 'MyGroup', 'lr': 0.01, 'momentum': 0.9}),
    ]

    # Experiments
    task_list = [
        # (path_to_results, task, algo_settings, additional_settings)
        (f"{path_to_result}-task-variance4", generate_task_normal(num_arms, seed=seed, scale=np.sqrt(4)), comparison_list, {'time_horizon': 4000, 'run_times': 100}),
        (f"{path_to_result}-task-variance2", generate_task_normal(num_arms, seed=seed, scale=np.sqrt(2)), comparison_list, {'time_horizon': 10000, 'run_times': 100}),
        (f"{path_to_result}-task-variance1", generate_task_normal(num_arms, seed=seed, scale=np.sqrt(1)), comparison_list, {'time_horizon': 10000, 'run_times': 100}),
    ]

    # Run
    result_list = batch_run_figure(task_list, user_bias)

    # Plot
    for index in range(len(result_list)):
        if is_show:
            benchmark.plot_results(
                data=result_list[index],
                title=f"Task {index}"
            )
        if is_save:
            benchmark.export_figures(
                data=result_list[index],
                output_name_prefix=task_list[index][0],
                color_style="main_figure"
            )

def run_main_table():
    # Algorithms for comparison
    algos_list = [
        (DoubleThompsonSampling, {'name': 'DT'}),
        (RelativeConfidenceSampling, {'name': 'RC'}),
        (RelativeUCB, {'name': 'RUCB'}),
        (UCB, {'name': 'UCB'}),
        (ModifiedBeatTheWinner, {'name': 'MBTW'}),
        (DoubleThompsonSamplingBias, {'name': 'DT-B(*)'}),
        (RelativeConfidenceSamplingBias, {'name': 'RC-B(*)'}),
        (RelativeUCBBias, {'name': 'RUCB-B(*)'}),
        (BiasSensitiveUCB, {'name': 'BS-UN(*)'}),
        (BiasSensitiveUCB, {'name': 'BS-K(*)', 'assigned_user_bias': None}),
        (MyBTBiasAlgo, {'name': 'MyBT', 'lr': 0.1, 'est_steps': 10}),
        # (MyLightBiasUCB, {'name': 'MyLight', 'c_base': 0.3}),
        (MyGroupBiasUCB, {'name': 'MyGroup', 'lr': 0.01, 'momentum': 0.9}),
        # (MyThompsonBetaPlus, {'name': 'MyThompson+', 'c_base': 0.25}),
    ]

    # print("=== 臂异质性测试 ===")
    # run_main_table_normal_task(algos_list, params_user_bias={"alpha":2.0, "beta":1.0}, task_variance_list=[1.0,2.0,4.0])

    # print("=== 偏置集中度测试 ===")
    # run_main_table_user_bias(algos_list, task_variance=2.0)

    # print("=== 评估者数量测试 ===")
    # run_main_table_num_users(algos_list, task_variance=2.0, params_user_bias={"alpha":2.0,"beta":1.0}, num_users_list=[5,15,20])

    print("=== 臂数量测试 ===")
    run_main_table_num_arms(algos_list, task_variance=2.0, params_user_bias={"alpha":2.0,"beta":1.0}, num_arms_list=[10,15,20,50])
    # print("Comparison of different user bias")
    # run_main_table_user_bias(algos_list, task_variance=4.0)
    # run_main_table_user_bias(algos_list, task_variance=2.0)
    # run_main_table_user_bias(algos_list, task_variance=1.0)

    # print("Comparison of different variance of arm score")
    # run_main_table_normal_task(algos_list, task_variance_list=[4.0, 3.0, 2.0, 1.0])

    # print("Comparison of different numbers of arms")
    # run_main_table_num_arms(algos_list, task_variance=4.0, num_arms_list=[5, 15, 20])
    # run_main_table_num_arms(algos_list, task_variance=2.0, num_arms_list=[5, 10, 15, 20, 30, 50])
    # run_main_table_num_arms(algos_list, task_variance=1.0, num_arms_list=[5, 15, 20])

    # print("Comparison of different numbers of users who give feedback")
    # run_main_table_num_users(algos_list, task_variance=4.0, num_users_list = [5, 15, 20])
    # run_main_table_num_users(algos_list, task_variance=2.0, num_users_list = [5, 15, 20])
    # run_main_table_num_users(algos_list, task_variance=1.0, num_users_list = [5, 15, 20])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--run_figure", action='store_true', default=False,
                        help="To plot the main figures.")
    parser.add_argument("-t", "--run_table", action='store_true', default=False,
                        help="To plot the main tables.")
    args = parser.parse_args()

    if args.run_figure:
        run_main_figure()
    elif args.run_table:
        run_main_table()
    else:
        print("Leave without running any experiments.")