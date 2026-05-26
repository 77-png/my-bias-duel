# Load necessary algorithms
from duel_bias.algorithms import DoubleThompsonSamplingBias
from duel_bias.algorithms import BiasSensitiveUCB

# Load custom biased preference model
from scripts_for_experiments.benchmark_utils.group_model import PreferenceGroupModelBias

# Benchmark utilities
from scripts_for_experiments.benchmark_utils import benchmark
from scripts_for_experiments.benchmark_utils.examples import print_info_problem
from scripts_for_experiments.benchmark_utils.examples import get_user_bias_beta

import sys
import numpy as np

# Utility function for detecting debug mode
def get_n_parallel():
    in_debug = False
    get_trace = getattr(sys, 'gettrace', None)
    if get_trace is not None and get_trace():
        in_debug = True

    import os
    if not in_debug and 'PYDEVD_LOAD_VALUES_ASYNC' in os.environ:
        in_debug = True

    return 1 if in_debug else -2

if __name__ == "__main__":
    random_state = np.random.default_rng(42)
    time_horizon = 10000
    run_times = 20
    num_arms = 10
    user_num = 10
    arm_variance = 4.0

    # ======= 构造模型参数 =======
    s = random_state.normal(0, np.sqrt(arm_variance), size=num_arms)  # 技能值 s_i
    g = random_state.integers(0, 2, size=num_arms)  # 偏好类别 g_i ∈ {0,1}
    r = get_user_bias_beta(user_num=user_num, alpha=2.0, beta=1.0)  # 用户偏好 r_k

    mean_user_bias = r
    n_parallel = get_n_parallel()

    # ======= 算法配置 =======
    comparison_list = [
        (DoubleThompsonSamplingBias, {
            'name': 'DT-B(*)',
            'comparative_user_bias': mean_user_bias,
            'is_track_residual': True
        }),
        (BiasSensitiveUCB, {
            'name': 'BS-UN(*)',
            'comparative_user_bias': mean_user_bias,
            'is_track_residual': False  # track_residual 部分还未实现
        }),
    ]
    algorithms, parameters_list = zip(*comparison_list)

    # ======= 初始化自定义环境 =======
    env = PreferenceGroupModelBias(g=g, s=s, r=r, random_state=random_state)

    print_info_problem(env.preference_matrix.preferences, r)

    # ======= 运行实验 =======
    result = benchmark.run_experiment(
        algorithms=algorithms,
        time_horizon=time_horizon,
        num_arms=num_arms,
        sample_interval=1,
        environment_class=lambda **kwargs: env,
        runs=run_times,
        n_jobs=n_parallel,
        base_random_seed=0,
        list_optional_parameters=parameters_list
    )

    # ======= 可视化与输出 =======
    benchmark.plot_results(result, title="PreferenceGroupModelBias (Arm Variance {:.1f})".format(arm_variance))
    # benchmark.export_evaluator_bias(result, output_name_prefix="results/group_bias")
    # benchmark.export_figures(result, output_name_prefix="results/group_bias")
    # benchmark.export_table_checkpoints(result, label="bias_error", output_name_prefix="results/group_bias")
