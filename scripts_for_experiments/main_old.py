import argparse
import os
import numpy as np
import pandas as pd
from numpy import ndarray

from scripts_for_experiments.benchmark_utils.bradley_terry_luce_model import BradleyTerryLuceModelBias
from scripts_for_experiments.benchmark_utils import benchmark
from scripts_for_experiments.benchmark_utils.examples import (
    generate_task_normal,
    get_user_bias_beta,
    print_info_problem,
    generate_task_barp,
    get_user_bias_barp,
    generate_group_membership,
    print_info_problem_barp,
)

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

def run(task: ndarray, bias: ndarray, settings: list, options: dict = None) -> pd.DataFrame:
    algorithms, parameters_list = zip(*settings)
    parameters_list = list(parameters_list)

    for params in parameters_list:
        if 'assigned_user_bias' in params:
            params['assigned_user_bias'] = bias
        if 'comparative_user_bias' in params:
            params['comparative_user_bias'] = bias

    number_arms = task.shape[0]
    random_state = np.random.default_rng(42)

    time_horizon = options.get('time_horizon', 4000) if options else 4000
    run_times = options.get('run_times', 50) if options else 50
    n_jobs = 1 if options and options.get('is_parallel') is False else -2

    env = BradleyTerryLuceModelBias(
        num_arms=number_arms,
        random_state=random_state,
        skill_vector=task,
        user_bias=bias,
        use_barp_model=False
    )
    print_info_problem(env.preference_matrix.preferences, bias)
    
    data = benchmark.run_experiment(
        algorithms=algorithms,
        time_horizon=time_horizon,
        num_arms=number_arms,
        sample_interval=1,
        environment_class=lambda **kwargs: env,
        runs=run_times,
        n_jobs=n_jobs,
        base_random_seed=0,
        list_optional_parameters=parameters_list
    )
    return data


def run_barp(
    task: ndarray,
    user_bias: ndarray,
    group_membership: ndarray,
    settings: list,
    options: dict = None
) -> pd.DataFrame:
    algorithms, parameters_list = zip(*settings)
    parameters_list = list(parameters_list)

    for params in parameters_list:
        if 'assigned_user_bias' in params:
            params['assigned_user_bias'] = user_bias
        if 'comparative_user_bias' in params:
            params['comparative_user_bias'] = user_bias

    number_arms = len(task)
    random_state = np.random.default_rng(42)

    time_horizon = options.get('time_horizon', 10000) if options else 10000
    run_times = options.get('run_times', 100) if options else 100
    n_jobs = 1 if options and options.get('is_parallel') is False else -2

    env = BradleyTerryLuceModelBias(
        num_arms=number_arms,
        random_state=random_state,
        skill_vector=task,
        user_bias=user_bias,
        group_membership=group_membership,
        use_barp_model=True,
        max_duels=time_horizon,
    )
    
    print_info_problem_barp(
        env.preference_matrix.preferences,
        user_bias,
        group_membership,
        task
    )
    
    data = benchmark.run_experiment(
        algorithms=algorithms,
        time_horizon=time_horizon,
        num_arms=number_arms,
        sample_interval=1,
        environment_class=lambda **kwargs: env,
        runs=run_times,
        n_jobs=n_jobs,
        base_random_seed=0,
        list_optional_parameters=parameters_list
    )
    return data


def run_barp_bias_mean(algos_list, task_variance: float = 2.0):
    path_to_result = f"results/barp-bias_mean-variance_{task_variance}"
    
    num_arms, num_users, seed = 10, 10, 42
    task = generate_task_barp(num_arms, score_std=np.sqrt(task_variance), seed=seed)
    group_membership = generate_group_membership(num_arms, 0.7, seed)
    
    bias_mean_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    
    results = pd.DataFrame()
    for bias_mean in bias_mean_values:
        print(f"Running bias_mean={bias_mean}")
        user_bias = get_user_bias_barp(num_users, 'normal', mean=bias_mean, std=1.0, seed=seed)
        result = run_barp(task, user_bias, group_membership, algos_list,
                         {'time_horizon': 10000, 'run_times': 100})
        result['bias_mean'] = f"mean={bias_mean}"
        results = pd.concat([results, result], ignore_index=True)
    
    results.to_pickle(path_to_result + ".pk1")
    benchmark.export_table(results, 'bias_mean', path_to_result)
    return results


def run_barp_bias_std(algos_list, task_variance: float = 2.0):
    path_to_result = f"results/barp-bias_std-variance_{task_variance}"
    
    num_arms, num_users, seed = 10, 10, 42
    task = generate_task_barp(num_arms, score_std=np.sqrt(task_variance), seed=seed)
    group_membership = generate_group_membership(num_arms, 0.7, seed)
    
    bias_std_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    
    results = pd.DataFrame()
    for bias_std in bias_std_values:
        print(f"Running bias_std={bias_std}")
        user_bias = get_user_bias_barp(num_users, 'normal', mean=0.0, std=bias_std, seed=seed)
        result = run_barp(task, user_bias, group_membership, algos_list,
                         {'time_horizon': 10000, 'run_times': 100})
        result['bias_std'] = f"std={bias_std}"
        results = pd.concat([results, result], ignore_index=True)
    
    results.to_pickle(path_to_result + ".pk1")
    benchmark.export_table(results, 'bias_std', path_to_result)
    return results


def run_barp_bias_distribution(algos_list, task_variance: float = 2.0):
    path_to_result = f"results/barp-bias_distribution-variance_{task_variance}"
    
    num_arms, num_users, seed = 10, 10, 42
    task = generate_task_barp(num_arms, score_std=np.sqrt(task_variance), seed=seed)
    group_membership = generate_group_membership(num_arms, 0.7, seed)
    
    distributions = [
        ('uniform', {'distribution': 'uniform', 'min_val': -5.0, 'max_val': 5.0}),
        ('normal', {'distribution': 'normal', 'mean': 0.0, 'std': 2.5}),
        ('exponential', {'distribution': 'exponential', 'std': 2.0}),
    ]
    
    results = pd.DataFrame()
    for dist_name, dist_params in distributions:
        print(f"Running distribution={dist_name}")
        user_bias = get_user_bias_barp(num_users, seed=seed, **dist_params)
        result = run_barp(task, user_bias, group_membership, algos_list,
                         {'time_horizon': 10000, 'run_times': 100})
        result['distribution'] = dist_name
        results = pd.concat([results, result], ignore_index=True)
    
    results.to_pickle(path_to_result + ".pk1")
    benchmark.export_table(results, 'distribution', path_to_result)
    return results


def run_barp_arm_variance(algos_list, bias_mean: float = 2.0, bias_std: float = 1.0):
    path_to_result = f"results/barp-arm_variance-bias_mean_{bias_mean}"
    
    num_arms, num_users, seed = 10, 10, 42
    variance_list = [1.0, 2.0, 4.0]
    
    results = pd.DataFrame()
    for variance in variance_list:
        print(f"Running variance={variance}")
        task = generate_task_barp(num_arms, score_std=np.sqrt(variance), seed=seed)
        group_membership = generate_group_membership(num_arms, 0.7, seed)
        user_bias = get_user_bias_barp(num_users, 'normal', mean=bias_mean, std=bias_std, seed=seed)
        
        time_horizon = 4000 if variance > 2.0 else 10000
        result = run_barp(task, user_bias, group_membership, algos_list,
                         {'time_horizon': time_horizon, 'run_times': 100})
        result['arm_variance'] = f"var={variance}" 
        results = pd.concat([results, result], ignore_index=True)
    
    results.to_pickle(path_to_result + ".pk1")
    benchmark.export_table(results, 'arm_variance', path_to_result)
    return results


def run_barp_num_arms(algos_list, task_variance: float = 2.0, bias_mean: float = 2.0, bias_std: float = 1.0):
    path_to_result = f"results/barp-num_arms-variance_{task_variance}"
    
    num_users, seed = 10, 42
    num_arms_list = [10, 15, 20, 50]
    
    results = pd.DataFrame()
    for num_arms in num_arms_list:
        print(f"Running num_arms={num_arms}")
        task = generate_task_barp(num_arms, score_std=np.sqrt(task_variance), seed=seed)
        group_membership = generate_group_membership(num_arms, 0.7, seed)
        user_bias = get_user_bias_barp(num_users, 'normal', mean=bias_mean, std=bias_std, seed=seed)
        
        time_horizon = 4000 if task_variance > 2.0 else 10000
        result = run_barp(task, user_bias, group_membership, algos_list,
                         {'time_horizon': time_horizon, 'run_times': 100})
        result['num_arms'] = f"K={num_arms}"
        results = pd.concat([results, result], ignore_index=True)
    
    results.to_pickle(path_to_result + ".pk1")
    benchmark.export_table(results, 'num_arms', path_to_result)
    return results


def run_barp_num_users(algos_list, task_variance: float = 2.0, bias_mean: float = 2.0, bias_std: float = 1.0):
    path_to_result = f"results/barp-num_users-variance_{task_variance}"
    
    num_arms, seed = 10, 42
    num_users_list = [5, 10, 15, 20]
    
    task = generate_task_barp(num_arms, score_std=np.sqrt(task_variance), seed=seed)
    group_membership = generate_group_membership(num_arms, 0.7, seed)
    
    results = pd.DataFrame()
    for num_users in num_users_list:
        print(f"Running num_users={num_users}")
        user_bias = get_user_bias_barp(num_users, 'normal', mean=bias_mean, std=bias_std, seed=seed)
        
        time_horizon = 4000 if task_variance > 2.0 else 10000
        result = run_barp(task, user_bias, group_membership, algos_list,
                         {'time_horizon': time_horizon, 'run_times': 100})
        result['num_users'] = f"M={num_users}"
        results = pd.concat([results, result], ignore_index=True)
    
    results.to_pickle(path_to_result + ".pk1")
    benchmark.export_table(results, 'num_users', path_to_result)
    return results


def get_algos_list():
    return [
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
        # (MyBTBiasAlgo, {'name': 'MyBT', 'lr': 0.1, 'est_steps': 10}),
        (MyGroupBiasUCB, {'name': 'MyGroup', 'lr': 0.01, 'momentum': 0.9}),
    ]


def run_all_barp_experiments():
    algos_list = get_algos_list()
    
    print("\n" + "="*60)
    print("BARP: 偏见均值实验 (Figure 2 Left)")
    print("="*60)
    run_barp_bias_mean(algos_list, task_variance=2.0)

    print("\n" + "="*60)
    print("BARP: 偏见方差实验 (Figure 2 Right)")
    print("="*60)
    run_barp_bias_std(algos_list, task_variance=2.0)

    print("\n" + "="*60)
    print("BARP: 偏见分布实验 (Figure 5)")
    print("="*60)
    run_barp_bias_distribution(algos_list, task_variance=2.0)

    print("\n" + "="*60)
    print("BARP: Arm异质性实验")
    print("="*60)
    run_barp_arm_variance(algos_list)

    print("\n" + "="*60)
    print("BARP: Arm数量实验")
    print("="*60)
    run_barp_num_arms(algos_list)

    print("\n" + "="*60)
    print("BARP: 用户数量实验")
    print("="*60)
    run_barp_num_users(algos_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--run_barp", action='store_true', help="Run all BARP experiments")
    parser.add_argument("--bias_mean", action='store_true', help="BARP: bias mean experiment")
    parser.add_argument("--bias_std", action='store_true', help="BARP: bias std experiment")
    parser.add_argument("--bias_dist", action='store_true', help="BARP: bias distribution experiment")
    parser.add_argument("--arm_var", action='store_true', help="BARP: arm variance experiment")
    parser.add_argument("--num_arms", action='store_true', help="BARP: num arms experiment")
    parser.add_argument("--num_users", action='store_true', help="BARP: num users experiment")
    args = parser.parse_args()

    algos_list = get_algos_list()

    if args.run_barp:
        run_all_barp_experiments()
    elif args.bias_mean:
        run_barp_bias_mean(algos_list)
    elif args.bias_std:
        run_barp_bias_std(algos_list)
    elif args.bias_dist:
        run_barp_bias_distribution(algos_list)
    elif args.arm_var:
        run_barp_arm_variance(algos_list)
    elif args.num_arms:
        run_barp_num_arms(algos_list)
    elif args.num_users:
        run_barp_num_users(algos_list)
    else:
        print("Usage:")
        print("  python run_benchmark.py -b           # Run all BARP experiments")
        print("  python run_benchmark.py --bias_mean  # Bias mean experiment")
        print("  python run_benchmark.py --bias_std   # Bias std experiment")
        print("  python run_benchmark.py --bias_dist  # Bias distribution experiment")
        print("  python run_benchmark.py --arm_var    # Arm variance experiment")
        print("  python run_benchmark.py --num_arms   # Num arms experiment")
        print("  python run_benchmark.py --num_users  # Num users experiment")