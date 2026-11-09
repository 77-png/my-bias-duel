HORIZON      = 3000
RUNS         = 20
N_INSTANCES  = 20
N_JOBS       = -1

# Tuned parameters selected by preliminary search
MYGROUP_ALPHA = 0.51
MYGROUP_EPSILON = 0.1
MYGROUP_LAMBDA_Q = 0.01
MYGROUP_EM_REFIT_FREQ = 200

DIAGNOSTIC_SEED_GROUPS = {
    "old_42_46": [42, 43, 44, 45, 46],
    "clean_100_104": [100, 101, 102, 103, 104],
}

# Evaluation protocol: freeze after tuning/diagnostics
TUNING_SEEDS = [42, 43, 44]
FINAL_SEEDS = list(range(200, 220))


import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from scripts_for_experiments.benchmark_utils.bradley_terry_luce_model import (
    BradleyTerryLuceModelBias,
)
from scripts_for_experiments.benchmark_utils.examples import (
    generate_task_barp,
    get_user_bias_barp,
    generate_group_membership,
)

from duelpy.algorithms import DoubleThompsonSampling
from duelpy.algorithms import RelativeConfidenceSampling
from duelpy.algorithms import RelativeUCB

from duel_bias.algorithms import UCB
from duel_bias.algorithms import ModifiedBeatTheWinner
from duel_bias.algorithms import DoubleThompsonSamplingBias
from duel_bias.algorithms import RelativeConfidenceSamplingBias
from duel_bias.algorithms import RelativeUCBBias
from duel_bias.algorithms import BiasSensitiveUCB
from duel_bias.algorithms.my_group_bias_ucb import MyGroupBiasUCB


RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

HORIZON      = 3000
RUNS         = 20
N_INSTANCES  = 20
N_JOBS       = -1
CHECKPOINTS  = [500, 1000, 1500, 2000, 2500, 3000]

# Long-horizon weak-regret experiment only. Main/final accuracy experiments stay at T=3000.
WEAK_REGRET_HORIZON = 10000

# Frozen final-evaluation protocol. Do not change after inspecting results.
if len(FINAL_SEEDS) != N_INSTANCES:
    raise ValueError(
        f"FINAL_SEEDS has {len(FINAL_SEEDS)} seeds but N_INSTANCES={N_INSTANCES}"
    )
if set(TUNING_SEEDS) & set(FINAL_SEEDS):
    raise ValueError("TUNING_SEEDS and FINAL_SEEDS must be disjoint.")

# Independent problem instances used only for diagnostic checking.
DIAGNOSTIC_SEEDS = [100, 101, 102, 103, 104]



def inject_adversarial_setup(task, group, delta=1.5):
    task  = task.copy()
    group = group.copy()
    idx   = np.argsort(task)
    best_idx, decoy_idx = int(idx[-1]), int(idx[-2])
    base_max = np.max(np.delete(task, [best_idx, decoy_idx]))
    task[decoy_idx]  = base_max + 0.5
    task[best_idx]   = task[decoy_idx] + delta
    group[best_idx]  = 0
    group[decoy_idx] = 1
    return task, group


def get_algorithms():
    return {
        "DT":      DoubleThompsonSampling,
        "RC":      RelativeConfidenceSampling,
        "RUCB":    RelativeUCB,
        "UCB":     UCB,
        "MBTW":    ModifiedBeatTheWinner,
        "DT-B":    DoubleThompsonSamplingBias,
        "RC-B":    RelativeConfidenceSamplingBias,
        "RUCB-B":  RelativeUCBBias,
        "BS-UN":   BiasSensitiveUCB,
        "MyGroup": MyGroupBiasUCB,
    }

def make_mygroup_factory(
    alpha=MYGROUP_ALPHA,
    epsilon=MYGROUP_EPSILON,
    lambda_q=MYGROUP_LAMBDA_Q,
    em_refit_freq=MYGROUP_EM_REFIT_FREQ,
):
    def factory(env, **kw):

        return MyGroupBiasUCB(
            env,
            time_horizon=kw.get(
                "time_horizon",
                HORIZON
            ),
            random_state=kw.get(
                "random_state",
                None
            ),
            alpha=alpha,
            epsilon=epsilon,
            lambda_q=lambda_q,
            em_refit_freq=em_refit_freq,
        )

    return factory

def search_mygroup_params_stage1():

    print("\n=== MyGroup Hyperparameter Search: Stage 1 ===")

    # 搜索阶段先少跑一点
    search_runs = 20
    search_instances = 3

    num_arms = 10
    num_users = 10

    # ==============================
    # 48组参数
    # ==============================

    alpha_list = [
        0.30,
        0.51,
        0.70,
        1.00,
    ]

    epsilon_list = [
        0.10,
        0.30,
        0.40,
        0.50,
    ]

    lambda_list = [
        0.001,
        0.005,
        0.010,
    ]

    results = []

    total = (
        len(alpha_list)
        * len(epsilon_list)
        * len(lambda_list)
    )

    current = 0

    # ==============================
    # 开始遍历48组
    # ==============================

    for alpha in alpha_list:

        for epsilon in epsilon_list:

            for lambda_q in lambda_list:

                current += 1

                print(
                    f"\n[{current}/{total}] "
                    f"alpha={alpha}, "
                    f"epsilon={epsilon}, "
                    f"lambda_q={lambda_q}"
                )

                algo = make_mygroup_factory(
                    alpha=alpha,
                    epsilon=epsilon,
                    lambda_q=lambda_q,
                    em_refit_freq=200,
                )

                regrets = []

                # ==============================
                # 3个instance
                # ==============================

                for inst in range(search_instances):

                    seed = 42 + inst

                    task = generate_task_barp(
                        num_arms,
                        score_std=0.5,
                        seed=seed,
                    )

                    group = generate_group_membership(
                        num_arms,
                        0.7,
                        seed,
                    )

                    task, group = inject_adversarial_setup(
                        task,
                        group,
                    )

                    user_bias = get_user_bias_barp(
                        num_users,
                        "normal",
                        mean=2.0,
                        std=2.5,
                        seed=seed,
                    )

                    # ==============================
                    # 只跑MyGroup
                    # ==============================

                    df_final, _, df_regret = run_experiment(
                        task,
                        user_bias,
                        group,
                        algos={
                            "MyGroup": algo
                        },
                        collect_regret=True,
                    )

                    final_regret = df_regret[
                        df_regret["timestep"] == HORIZON
                    ]["cumulative_weak_regret"].iloc[0]

                    regrets.append(final_regret)

                mean_regret = float(
                    np.mean(regrets)
                )

                results.append({
                    "alpha": alpha,
                    "epsilon": epsilon,
                    "lambda_q": lambda_q,
                    "em_refit_freq": 200,
                    "cumulative_weak_regret": mean_regret,
                })

                print(
                    f"    -> weak regret={mean_regret:.4f}"
                )

    # ==============================
    # 保存结果
    # ==============================

    df = pd.DataFrame(results)

    df = df.sort_values(
        "cumulative_weak_regret",
        ascending=True,
    )

    path = (
        f"{RESULT_DIR}/"
        f"mygroup_search_stage1.csv"
    )

    df.to_csv(
        path,
        index=False,
    )

    print(
        "\n================================"
    )
    print(
        "TOP 10 PARAMETERS"
    )
    print(
        "================================"
    )

    print(
        df.head(10).to_string(
            index=False
        )
    )

    print(
        f"\nSaved: {path}"
    )

    return df

def get_offline_group_ucb_factory():
    return lambda env, **kw: MyGroupBiasUCB(
        env,
        time_horizon=kw.get("time_horizon", HORIZON),
        random_state=kw.get("random_state", None),
        em_refit_freq=float('inf'),
    )

def get_ablation_algorithms():

    return {

        "Full": lambda env, **kw:
            MyGroupBiasUCB(
                env,
                time_horizon=kw.get(
                    "time_horizon",
                    HORIZON
                ),
                random_state=kw.get(
                    "random_state",
                    None
                ),
                cross_group_prob=0.1,
            ),

        "noWarmup": lambda env, **kw:
            MyGroupBiasUCB(
                env,
                time_horizon=kw.get(
                    "time_horizon",
                    HORIZON
                ),
                random_state=kw.get(
                    "random_state",
                    None
                ),
                warmup_obs_per_user_pair=0,
                warmup_max_frac=0.0,
                cross_group_prob=0.1,
            ),

        "noGdGate": lambda env, **kw:
            MyGroupBiasUCB(
                env,
                time_horizon=kw.get(
                    "time_horizon",
                    HORIZON
                ),
                random_state=kw.get(
                    "random_state",
                    None
                ),
                gd_stability_gate=0,
                cross_group_prob=0.1,
            ),

        "noDecay": lambda env, **kw:
            MyGroupBiasUCB(
                env,
                time_horizon=kw.get(
                    "time_horizon",
                    HORIZON
                ),
                random_state=kw.get(
                    "random_state",
                    None
                ),
                recent_decay=1.0,
                cross_group_prob=0.1,
            ),

        "noXGroup": lambda env, **kw:
            MyGroupBiasUCB(
                env,
                time_horizon=kw.get(
                    "time_horizon",
                    HORIZON
                ),
                random_state=kw.get(
                    "random_state",
                    None
                ),
                cross_group_prob=0.0,
            ),
    }

def get_cross_prob_algorithms():

    probs=[
        0.0,
        0.05,
        0.1,
        0.2,
        0.5,
        1.0
    ]

    return {
        f"cross_prob={p}":
        lambda env,p=p,**kw:
        MyGroupBiasUCB(
            env,
            time_horizon=
                kw.get(
                    "time_horizon",
                    HORIZON
                ),
            random_state=
                kw.get(
                    "random_state",
                    None
                ),
            cross_group_prob=p,
        )

        for p in probs
    }

def get_sensitivity_algorithms():
    return {
        f"λ={lam:.4f}": lambda env, lam=lam, **kw: MyGroupBiasUCB(
            env,
            time_horizon=kw.get("time_horizon", HORIZON),
            random_state=kw.get("random_state", None),
            lambda_q=lam,
        )
        for lam in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    }

def get_cross_prob_sensitivity_algorithms():

    probs = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]

    algos = {}

    for p in probs:

        algos[f"cross_prob={p}"] = (
            lambda env, p=p, **kw:
            MyGroupBiasUCB(
                env,
                time_horizon=kw.get(
                    "time_horizon",
                    HORIZON
                ),
                random_state=kw.get(
                    "random_state",
                    None
                ),

                cross_group_correction=True,
                cross_group_prob=p,
            )
        )

    return algos

def experiment_cross_group_prob():

    print("\n=== Cross Group Probability Sensitivity ===")


    num_arms = 10
    num_users = 10


    algos=get_cross_prob_algorithms()


    all_results = []


    for inst, seed in enumerate(FINAL_SEEDS):


        print(
            f"Instance {inst+1}/{N_INSTANCES}"
        )


        task = generate_task_barp(
            num_arms,
            score_std=0.5,
            seed=seed,
        )


        group = generate_group_membership(
            num_arms,
            0.7,
            seed,
        )


        task, group = inject_adversarial_setup(
            task,
            group,
        )


        user_bias = get_user_bias_barp(
            num_users,
            "normal",
            mean=2,
            std=2.5,
            seed=seed,
        )


        _, df_anytime, _ = run_experiment(
            task,
            user_bias,
            group,
            algos=algos,
            with_checkpoints=True
        )

        df_anytime["instance"] = inst
        all_results.append(df_anytime)



    combined = pd.concat(
        all_results,
        ignore_index=True,
    )

    agg = (
        combined
        .groupby(["algorithm", "timestep"])
        .agg(
            accuracy=("accuracy", "mean"),
            ci=(
                "accuracy",
                lambda x:
                1.96 * x.std() / np.sqrt(len(x))
            )
        )
        .reset_index()
    )


    print(
        agg.pivot(
            index="timestep",
            columns="algorithm",
            values="accuracy"
        )
    )


    agg.to_csv(
        f"{RESULT_DIR}/cross_group_prob_sensitivity.csv",
        index=False,
    )


    return agg

def experiment_sensitivity_lambda():
    print("\n=== Sensitivity Analysis: λ (Adversarial Bias, μ_β=2, σ_β=2.5) ===")
    algos = get_sensitivity_algorithms()
    all_finals = []
    for inst, seed in enumerate(FINAL_SEEDS):
        task  = generate_task_barp(10, score_std=0.5, seed=seed)
        group = generate_group_membership(10, 0.7, seed)
        task, group = inject_adversarial_setup(task, group)
        user_bias = get_user_bias_barp(10, "normal", mean=2, std=2.5, seed=seed)
        df_f, _, _ = run_experiment(task, user_bias, group, algos=algos)
        df_f["instance"] = inst
        all_finals.append(df_f)

    combined = pd.concat(all_finals)
    agg = combined.groupby("algorithm").agg(
        accuracy=("accuracy", "mean"),
        ci=("accuracy", lambda x: 1.96 * x.std() / max(np.sqrt(len(x)), 1))
    ).reset_index()
    agg.to_csv(f"{RESULT_DIR}/sensitivity_lambda.csv", index=False)
    print("\nLambda sensitivity results:")
    print(agg.to_string(index=False))

def compute_true_borda_scores(skill_vector):
    """
    Compute the true unbiased Borda score of each arm.

    theta_i = 1 / (K - 1) * sum_{j != i} sigmoid(s_i - s_j)
    """
    skills = np.asarray(skill_vector, dtype=float)
    K = len(skills)

    # diff[i, j] = s_i - s_j
    diff = skills[:, None] - skills[None, :]

    # Unbiased pairwise preference probabilities
    pairwise_prob = 1.0 / (
        1.0 + np.exp(-np.clip(diff, -30, 30))
    )

    # Exclude self-comparisons
    np.fill_diagonal(pairwise_prob, 0.0)

    return pairwise_prob.sum(axis=1) / (K - 1)

def run_single(
    task,
    user_bias,
    group_membership,
    algo_factory,
    seed,
    checkpoints=None,
    collect_regret=False,
    horizon=None,
):
    if horizon is None:
        horizon = HORIZON

    env = BradleyTerryLuceModelBias(
        num_arms=len(task),
        num_users=len(user_bias),
        random_state=np.random.RandomState(seed),
        skill_vector=task,
        user_bias=user_bias,
        group_membership=group_membership,
    )

    rng = np.random.RandomState(seed)

    if callable(algo_factory) and not isinstance(algo_factory, type):
        algo = algo_factory(
            env,
            time_horizon=horizon,
            random_state=rng,
        )
    else:
        try:
            algo = algo_factory(
                env,
                time_horizon=horizon,
                random_state=rng,
            )
        except TypeError:
            try:
                algo = algo_factory(
                    env,
                    time_horizon=horizon,
                )
            except TypeError:
                algo = algo_factory(env)

    true_borda = compute_true_borda_scores(env.skill_vector)
    best_borda = float(np.max(true_borda))

    cumulative_regret = 0.0
    regret_history = []

    checkpoint_set = set(checkpoints) if checkpoints else set()
    checkpoint_preds = {}

    for t in range(1, horizon + 1):

        # 执行 step 前已有的 duel 数量
        history_length_before = len(env.duel_history)

        if hasattr(algo, "step"):
            algo.step()

        if collect_regret:

            new_duels = env.duel_history[
                history_length_before:
            ]

            new_regret = 0.0

            for arm_i, arm_j, _ in new_duels:
                # if algo.__class__.__name__.startswith("MyGroup") and t > 1500:
                #     print(t, arm_i, arm_j)

                instantaneous_regret = (
                    best_borda
                    - max(
                        true_borda[arm_i],
                        true_borda[arm_j],
                    )
                )

                instantaneous_regret = max(
                    float(instantaneous_regret),
                    0.0,
                )

                cumulative_regret += instantaneous_regret
                new_regret += instantaneous_regret

            # if t in [500, 1000, 1500, 2000, 2500, 3000]:
            #     print(
            #         f"[PAIR] seed={seed}, "
            #         f"t={t}, "
            #         f"pair=({arm_i}, {arm_j}), "
            #         f"best_arm={int(np.argmax(true_borda))}, "
            #         f"regret={instantaneous_regret:.6f}"
            #     )

            regret_history.append(
                cumulative_regret
            )

        if t in checkpoint_set:

            checkpoint_preds[t] = _extract_prediction(
                algo,
                task,
                seed,
            )

    final_pred = _extract_prediction(
        algo,
        task,
        seed,
    )

    stats = {}

    if hasattr(algo, "get_cross_group_stats"):
        stats = algo.get_cross_group_stats()

    if collect_regret:

        return (
            final_pred,
            checkpoint_preds,
            np.asarray(
                regret_history,
                dtype=float,
            ),
            stats,
        )

    return final_pred, checkpoint_preds, stats


def _extract_prediction(algo, task, seed):
    if hasattr(algo, "get_condorcet_winner"):
        try:
            return int(algo.get_condorcet_winner())
        except Exception:
            pass
    if hasattr(algo, "s"):
        return int(np.argmax(algo.s))
    if hasattr(algo, "preference_estimate"):
        est = algo.preference_estimate
        mat = (est.get_mean_estimate_matrix()
               if hasattr(est, "get_mean_estimate_matrix") else est)
        if isinstance(mat, np.ndarray):
            return int(np.argmax(np.mean(mat, axis=1)))
    return int(np.random.default_rng(seed).integers(len(task)))


def run_experiment(
    task,
    user_bias,
    group_membership,
    algos=None,
    with_checkpoints=False,
    collect_regret=False,
    horizon=None,
):
    if horizon is None:
        horizon = HORIZON

    if algos is None:
        algos = get_algorithms()

    true_best = int(np.argmax(task))

    results_final = []
    results_anytime = []
    results_regret = []

    for name, algo in algos.items():

        print(f"    algorithm: {name}")

        ckpts = CHECKPOINTS if with_checkpoints else None

        t0 = time.perf_counter()

        outputs = Parallel(n_jobs=N_JOBS)(
            delayed(run_single)(
                task=task,
                user_bias=user_bias,
                group_membership=group_membership,
                algo_factory=algo,
                seed=r,
                checkpoints=ckpts,
                collect_regret=collect_regret,
                horizon=horizon,
            )
            for r in range(RUNS)
        )

        wall_time = time.perf_counter() - t0

        preds = [o[0] for o in outputs]

        if collect_regret:
            cross_stats = [
                o[3]
                for o in outputs
            ]
        else:
            cross_stats = [
                o[2]
                for o in outputs
            ]

        cross_group_ratios = [
            s.get("cross_group_ratio", 0.0)
            for s in cross_stats
        ]

        forced_cross_group = [
            s.get("forced_cross_group_duels", 0)
            for s in cross_stats
        ]

        em_refit_counts = [
            s.get("em_refit_count", 0)
            for s in cross_stats
        ]

        warmup_duels = [
            s.get("warmup_duels", 0)
            for s in cross_stats
        ]

        cross_group_correction_counts = [
            s.get("cross_group_correction_count", 0)
            for s in cross_stats
        ]

        acc_list = [
            1 if p == true_best else 0
            for p in preds
        ]

        results_final.append({
            "algorithm": name,

            "accuracy": float(np.mean(acc_list)),

            "ci": (
                1.96
                * float(np.std(acc_list))
                / np.sqrt(RUNS)
            ),

            "wall_sec": round(wall_time, 1),

            "cross_group_ratio": float(
                np.mean(cross_group_ratios)
            ),

            "forced_cross_group_duels": float(
                np.mean(forced_cross_group)
            ),

            "em_refit_count": float(
                np.mean(em_refit_counts)
            ),

            "warmup_duels": float(
                np.mean(warmup_duels)
            ),

            "cross_group_correction_count": float(
                np.mean(cross_group_correction_counts)
            ),
        })

        if with_checkpoints:

            for cp in CHECKPOINTS:

                cp_preds = [
                    o[1].get(cp, -1)
                    for o in outputs
                ]

                cp_acc = [
                    1 if p == true_best else 0
                    for p in cp_preds
                ]

                results_anytime.append({
                    "algorithm": name,
                    "timestep": cp,
                    "accuracy": float(np.mean(cp_acc)),
                    "ci": (
                        1.96
                        * float(np.std(cp_acc))
                        / np.sqrt(RUNS)
                    ),
                })

        if collect_regret:

            regret_matrix = np.vstack(
                [o[2] for o in outputs]
            )

            regret_mean = np.mean(
                regret_matrix,
                axis=0,
            )

            regret_ci = (
                1.96
                * np.std(
                    regret_matrix,
                    axis=0,
                )
                / np.sqrt(RUNS)
            )

            for t in range(horizon):

                results_regret.append({
                    "algorithm": name,
                    "timestep": t + 1,
                    "cumulative_weak_regret": (
                        float(regret_mean[t])
                    ),
                    "ci": float(regret_ci[t]),
                })

    df_final = pd.DataFrame(results_final)

    df_anytime = (
        pd.DataFrame(results_anytime)
        if with_checkpoints
        else None
    )

    df_regret = (
        pd.DataFrame(results_regret)
        if collect_regret
        else None
    )

    return (
        df_final,
        df_anytime,
        df_regret,
    )

def make_tuned_mygroup_factory(**extra_kwargs):
    def factory(env, **kw):
        return MyGroupBiasUCB(
            env,
            time_horizon=kw.get("time_horizon", HORIZON),
            random_state=kw.get("random_state", None),

            alpha=MYGROUP_ALPHA,
            epsilon=MYGROUP_EPSILON,
            lambda_q=MYGROUP_LAMBDA_Q,
            em_refit_freq=MYGROUP_EM_REFIT_FREQ,

            **extra_kwargs,
        )

    return factory

def run_multi_instance(experiment_fn, seeds=None, **kwargs):
    seeds = FINAL_SEEDS if seeds is None else list(seeds)
    all_finals = []
    for inst, seed in enumerate(seeds):
        print(f"  Instance {inst+1}/{len(seeds)} (seed={seed})")
        df_f, _ = experiment_fn(seed=seed, **kwargs)
        df_f["instance"] = inst
        all_finals.append(df_f)

    combined = pd.concat(all_finals)
    agg = (combined.groupby("algorithm")
           .agg(accuracy=("accuracy", "mean"),
                ci=("accuracy", lambda x: 1.96 * x.std() / np.sqrt(len(x))),
                wall_sec=("wall_sec", "mean"))
           .reset_index())
    return agg, combined


# ────────────────────────────────────────────────────────────────────────────
# Plot utilities — error bars clipped to [0, 1]
# ────────────────────────────────────────────────────────────────────────────

def plot_results(df, xcol, filename, title_suffix=""):
    plt.figure(figsize=(10, 6))
    for algo in df.algorithm.unique():
        d    = df[df.algorithm == algo]
        mine = algo.startswith("MyGroup")

        acc = d.accuracy.values
        ci  = d.ci.values
        yerr_lower = np.minimum(ci, acc)          # don't go below 0
        yerr_upper = np.minimum(ci, 1.0 - acc)    # don't go above 1

        plt.errorbar(d[xcol], acc, yerr=[yerr_lower, yerr_upper],
                     marker="D" if mine else "o",
                     linewidth=3.5 if mine else 1.5,
                     alpha=1.0 if mine else 0.6,
                     label=algo)
    plt.xlabel(xcol, fontsize=12)
    plt.ylabel("Optimal Arm Selection Accuracy", fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.title(f"Performance under Adversarial Bias (T={HORIZON}) "
              f"{title_suffix}", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_anytime(df_anytime, filename):
    if df_anytime is None or df_anytime.empty:
        return
    plt.figure(figsize=(10, 6))
    for algo in df_anytime.algorithm.unique():
        d = df_anytime[df_anytime.algorithm == algo]
        mine = algo.startswith("MyGroup")

        acc = d.accuracy.values
        ci  = d.ci.values
        yerr_lower = np.minimum(ci, acc)
        yerr_upper = np.minimum(ci, 1.0 - acc)

        plt.errorbar(d.timestep, acc, yerr=[yerr_lower, yerr_upper],
                     marker="D" if mine else "o",
                     linewidth=3.5 if mine else 1.5,
                     alpha=1.0 if mine else 0.6,
                     label=algo)
    plt.xlabel("Timestep", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.title("Anytime Accuracy (convergence speed)", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def plot_weak_regret(df_regret, filename):

    if df_regret is None or df_regret.empty:
        return

    plt.figure(figsize=(10, 6))

    for algo in df_regret.algorithm.unique():

        d = df_regret[
            df_regret.algorithm == algo
        ].sort_values(
            "timestep"
        )

        mine = algo.startswith(
            "MyGroup"
        )

        regret = (
            d.cumulative_weak_regret
            .values
        )

        plt.plot(
            d.timestep,
            regret,
            marker="D" if mine else None,
            markevery=300 if mine else None,
            linewidth=3.5 if mine else 1.5,
            alpha=1.0 if mine else 0.7,
            label=algo,
        )

    plt.xlabel(
        "Number of Duels",
        fontsize=12,
    )

    plt.ylabel(
        "Cumulative Weak Regret",
        fontsize=12,
    )

    plt.title(
        "Cumulative Weak Regret under Adversarial Bias",
        fontsize=14,
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.5,
    )

    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


# ────────────────────────────────────────────────────────────────────────────
# Experiments
# ────────────────────────────────────────────────────────────────────────────

def _single_bias_mean_instance(bias_mean, seed=42):
    num_arms, num_users = 10, 10
    task  = generate_task_barp(num_arms, score_std=0.5, seed=seed)
    group = generate_group_membership(num_arms, 0.7, seed)
    task, group = inject_adversarial_setup(task, group)
    user_bias = get_user_bias_barp(
        num_users, "normal", mean=bias_mean, std=2.5, seed=seed)
    df_f, _, _ = run_experiment(task, user_bias, group)
    df_f["bias_mean"] = bias_mean
    return df_f


def experiment_bias_mean():
    print("\n=== Bias Mean Sweep (σ_β = 2.5, multi-instance) ===")
    all_results = []
    for b in [0, 1, 2, 3, 4, 5]:
        print(f"bias_mean = {b}")
        instance_results = []
        for inst, seed in enumerate(FINAL_SEEDS):
            df = _single_bias_mean_instance(b, seed=seed)
            df["instance"] = inst
            instance_results.append(df)
        combined = pd.concat(instance_results)
        agg = (combined.groupby(["algorithm", "bias_mean"])
               .agg(accuracy=("accuracy", "mean"),
                    ci=("accuracy",
                        lambda x: 1.96 * x.std() / max(np.sqrt(len(x)), 1)),
                    wall_sec=("wall_sec", "mean"))
               .reset_index())
        all_results.append(agg)

    results = pd.concat(all_results)
    results.to_csv(f"{RESULT_DIR}/bias_mean_accuracy.csv", index=False)
    plot_results(results, "bias_mean", f"{RESULT_DIR}/bias_mean_plot.png",
                 f"(avg over {N_INSTANCES} instances)")


def experiment_bias_std():
    print("\n=== Bias Std Sweep (μ_β = 2, multi-instance) ===")
    all_results = []
    for s in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        print(f"bias_std = {s}")
        instance_results = []
        for inst, seed in enumerate(FINAL_SEEDS):
            num_arms, num_users = 10, 10
            task  = generate_task_barp(num_arms, score_std=0.5, seed=seed)
            group = generate_group_membership(num_arms, 0.7, seed)
            task, group = inject_adversarial_setup(task, group)
            user_bias = get_user_bias_barp(
                num_users, "normal", mean=2, std=s, seed=seed)
            df_f, _, _ = run_experiment(task, user_bias, group)
            df_f["bias_std"] = s
            df_f["instance"] = inst
            instance_results.append(df_f)
        combined = pd.concat(instance_results)
        agg = (combined.groupby(["algorithm", "bias_std"])
               .agg(accuracy=("accuracy", "mean"),
                    ci=("accuracy",
                        lambda x: 1.96 * x.std() / max(np.sqrt(len(x)), 1)),
                    wall_sec=("wall_sec", "mean"))
               .reset_index())
        all_results.append(agg)

    results = pd.concat(all_results)
    results.to_csv(f"{RESULT_DIR}/bias_std_accuracy.csv", index=False)
    plot_results(results, "bias_std", f"{RESULT_DIR}/bias_std_plot.png",
                 f"(avg over {N_INSTANCES} instances)")


def experiment_num_arms():
    print("\n=== Num Arms Sweep (multi-instance) ===")
    all_results = []
    for k in [10, 20, 30, 50]:
        print(f"num_arms = {k}")
        instance_results = []
        for inst, seed in enumerate(FINAL_SEEDS):
            num_users = 10
            task  = generate_task_barp(k, score_std=0.5, seed=seed)
            group = generate_group_membership(k, 0.7, seed)
            task, group = inject_adversarial_setup(task, group)
            user_bias = get_user_bias_barp(
                num_users, "normal", mean=2, std=2.5, seed=seed)
            df_f, _, _ = run_experiment(task, user_bias, group)
            df_f["num_arms"] = k
            df_f["instance"] = inst
            instance_results.append(df_f)
        combined = pd.concat(instance_results)
        agg = (combined.groupby(["algorithm", "num_arms"])
               .agg(accuracy=("accuracy", "mean"),
                    ci=("accuracy",
                        lambda x: 1.96 * x.std() / max(np.sqrt(len(x)), 1)),
                    wall_sec=("wall_sec", "mean"))
               .reset_index())
        all_results.append(agg)

    results = pd.concat(all_results)
    results.to_csv(f"{RESULT_DIR}/num_arms_accuracy.csv", index=False)
    plot_results(results, "num_arms", f"{RESULT_DIR}/num_arms_plot.png",
                 f"(avg over {N_INSTANCES} instances)")


def experiment_num_users():
    print("\n=== Num Users Sweep (multi-instance) ===")
    all_results = []
    for m in [5, 10, 20, 30]:
        print(f"num_users = {m}")
        instance_results = []
        for inst, seed in enumerate(FINAL_SEEDS):
            num_arms = 10
            task  = generate_task_barp(num_arms, score_std=0.5, seed=seed)
            group = generate_group_membership(num_arms, 0.7, seed)
            task, group = inject_adversarial_setup(task, group)
            user_bias = get_user_bias_barp(
                m, "normal", mean=2, std=2.5, seed=seed)
            df_f, _, _ = run_experiment(task, user_bias, group)
            df_f["num_users"] = m
            df_f["instance"] = inst
            instance_results.append(df_f)
        combined = pd.concat(instance_results)
        agg = (combined.groupby(["algorithm", "num_users"])
               .agg(accuracy=("accuracy", "mean"),
                    ci=("accuracy",
                        lambda x: 1.96 * x.std() / max(np.sqrt(len(x)), 1)),
                    wall_sec=("wall_sec", "mean"))
               .reset_index())
        all_results.append(agg)

    results = pd.concat(all_results)
    results.to_csv(f"{RESULT_DIR}/num_users_accuracy.csv", index=False)
    plot_results(results, "num_users", f"{RESULT_DIR}/num_users_plot.png",
                 f"(avg over {N_INSTANCES} instances)")


def experiment_anytime():
    """Formal multi-algorithm convergence figure over all frozen final instances."""
    print("\n=== Anytime Accuracy / Convergence (multi-instance, all algorithms) ===")
    num_arms, num_users = 10, 10
    all_anytime = []

    for inst, seed in enumerate(FINAL_SEEDS):
        print(f"  Instance {inst + 1}/{N_INSTANCES} (seed={seed})")

        task = generate_task_barp(num_arms, score_std=0.5, seed=seed)
        group = generate_group_membership(num_arms, 0.7, seed)
        task, group = inject_adversarial_setup(task, group)
        user_bias = get_user_bias_barp(
            num_users, "normal", mean=2.0, std=2.5, seed=seed
        )

        _, df_anytime, _ = run_experiment(
            task,
            user_bias,
            group,
            algos=get_algorithms(),
            with_checkpoints=True,
            collect_regret=False,
        )

        if df_anytime is not None and not df_anytime.empty:
            df_anytime["instance"] = inst
            df_anytime["problem_seed"] = seed
            all_anytime.append(df_anytime)

    if not all_anytime:
        print("No anytime results collected.")
        return None

    combined = pd.concat(all_anytime, ignore_index=True)
    combined.to_csv(
        f"{RESULT_DIR}/anytime_accuracy_per_instance.csv", index=False
    )

    # IMPORTANT: CI is computed across problem instances, matching the final
    # evaluation protocol, rather than treating stochastic runs as independent
    # problem instances.
    agg = (
        combined
        .groupby(["algorithm", "timestep"])
        .agg(
            accuracy=("accuracy", "mean"),
            ci=(
                "accuracy",
                lambda x: 1.96 * x.std() / max(np.sqrt(len(x)), 1),
            ),
        )
        .reset_index()
    )

    csv_path = f"{RESULT_DIR}/anytime_accuracy.csv"
    fig_path = f"{RESULT_DIR}/anytime_convergence_all_algorithms.png"
    agg.to_csv(csv_path, index=False)
    plot_anytime(agg, fig_path)

    print(f"Saved convergence CSV : {csv_path}")
    print(f"Saved convergence plot: {fig_path}")
    return agg


def experiment_weak_regret():

    print(
        "\n=== Cumulative Weak Regret "
        "(MyGroup, Tuned Parameters) ==="
    )

    num_arms = 10
    num_users = 10

    bias_mean = 2.0
    bias_std = 2.5

    # ==========================================
    # Tuned parameters
    # ==========================================
    alpha = 0.51
    epsilon = 0.1
    lambda_q = 0.010
    em_refit_freq = 200

    print(
        "\nUsing tuned parameters:"
    )
    print(
        f"alpha={alpha}, "
        f"epsilon={epsilon}, "
        f"lambda_q={lambda_q}, "
        f"em_refit_freq={em_refit_freq}"
    )

    print(f"Long-horizon weak regret: T={WEAK_REGRET_HORIZON}")

    all_regrets = []

    for inst, seed in enumerate(FINAL_SEEDS):

        print(
            f"  Instance "
            f"{inst + 1}/{N_INSTANCES} "
            f"(seed={seed})"
        )

        # ==========================================
        # Generate task
        # ==========================================
        task = generate_task_barp(
            num_arms,
            score_std=0.5,
            seed=seed,
        )

        # ==========================================
        # Generate group membership
        # ==========================================
        group = generate_group_membership(
            num_arms,
            0.7,
            seed,
        )

        task, group = inject_adversarial_setup(
            task,
            group,
        )

        user_bias = get_user_bias_barp(
            num_users,
            "normal",
            mean=bias_mean,
            std=bias_std,
            seed=seed,
        )

        algos = get_algorithms()

        _, _, df_regret = run_experiment(
            task,
            user_bias,
            group,
            algos=algos,
            collect_regret=True,
            horizon=WEAK_REGRET_HORIZON,
        )

        if df_regret is not None:

            df_regret["instance"] = inst

            all_regrets.append(
                df_regret
            )

    # ==========================================
    # Check results
    # ==========================================
    if not all_regrets:

        print(
            "No weak-regret results collected."
        )

        return

    combined = pd.concat(
        all_regrets,
        ignore_index=True,
    )

    # ==========================================
    # Aggregate across instances
    # ==========================================
    agg = (
        combined
        .groupby(
            ["algorithm", "timestep"]
        )
        .agg(
            cumulative_weak_regret=(
                "cumulative_weak_regret",
                "mean",
            ),
            ci=(
                "cumulative_weak_regret",
                lambda x:
                    1.96
                    * x.std()
                    / max(
                        np.sqrt(len(x)),
                        1,
                    ),
            ),
        )
        .reset_index()
    )

    # ==========================================
    # Save results
    # ==========================================
    csv_path = (
        f"{RESULT_DIR}/"
        f"mygroup_tuned_weak_regret_T{WEAK_REGRET_HORIZON}.csv"
    )

    fig_path = (
        f"{RESULT_DIR}/"
        f"mygroup_tuned_weak_regret_T{WEAK_REGRET_HORIZON}.png"
    )

    agg.to_csv(
        csv_path,
        index=False,
    )

    plot_weak_regret(
        agg,
        fig_path,
    )

    print(
        "\nMyGroup tuned weak-regret "
        "experiment complete."
    )

    print(
        f"Parameters: "
        f"alpha={alpha}, "
        f"epsilon={epsilon}, "
        f"lambda_q={lambda_q}, "
        f"em_refit_freq={em_refit_freq}"
    )

    print(
        f"Saved CSV: {csv_path}"
    )

    print(
        f"Saved figure: {fig_path}"
    )

def experiment_ablation():
    print("\n=== Ablation Study (Fixed) ===")
    num_arms, num_users = 10, 10
    ablation_algos = get_ablation_algorithms()
 
    conditions = [
        {"bias_mean": 2, "bias_std": 2.5, "label": "standard"},
        {"bias_mean": 2, "bias_std": 0.5, "label": "low-heterogeneity"},
        {"bias_mean": 1, "bias_std": 1.5, "label": "weak-bias"},
    ]
 
    for cond in conditions:
        print(f"\n  Condition: {cond['label']} "
              f"(μ_β={cond['bias_mean']}, σ_β={cond['bias_std']})")
 
        all_results = []
        for inst, seed in enumerate(FINAL_SEEDS):
            task  = generate_task_barp(num_arms, score_std=0.5, seed=seed)
            group = generate_group_membership(num_arms, 0.7, seed)
            task, group = inject_adversarial_setup(task, group)
            user_bias = get_user_bias_barp(
                num_users, "normal",
                mean=cond["bias_mean"],
                std=cond["bias_std"],
                seed=seed)
 
            df_f, _, _ = run_experiment(
                task, user_bias, group, algos=ablation_algos)
            df_f["instance"] = inst
            df_f["condition"] = cond["label"]
            all_results.append(df_f)
 
        results = pd.concat(all_results)
        agg = (
            results.groupby("algorithm")
            .agg(
                accuracy=("accuracy", "mean"),

                ci=("accuracy",
                    lambda x:
                    1.96 * x.std() / max(np.sqrt(len(x)), 1)),

                wall_sec=("wall_sec", "mean"),

                cross_group_ratio=(
                    "cross_group_ratio",
                    "mean"
                ),

                forced_cross_group_duels=(
                    "forced_cross_group_duels",
                    "mean"
                ),
            )
            .reset_index()
        )
        agg["condition"] = cond["label"]
 
        fname = f"{RESULT_DIR}/ablation_{cond['label']}_accuracy.csv"
        agg.to_csv(fname, index=False)
        print(f"\n  Results ({cond['label']}):")
        print(agg.to_string(index=False))
 
    print("\n  Ablation study complete. Check results/ for CSV files.")


def report_warmup_budget():
    print("\n=== Warmup Budget Report ===")
    print(f"{'K':>4} {'M':>4} {'pairs':>6} {'raw':>7} {'warmup':>7} "
          f"{'%horizon':>9} {'capped':>7}")
    for K in [10, 20, 30, 50]:
        for M in [5, 10, 20, 30]:
            pairs = K * (K - 1) // 2
            raw = pairs * M * 5
            capped = min(raw, int(HORIZON * 0.5))
            print(f"{K:>4} {M:>4} {pairs:>6} {raw:>7} {capped:>7} "
                  f"{100*capped/HORIZON:>8.1f}% "
                  f"{'yes' if capped < raw else 'no':>7}")

def generate_balanced_setup(num_arms, num_users, bias_mean, bias_std, seed):
    task = generate_task_barp(num_arms, score_std=0.5, seed=seed)
    group = generate_group_membership(num_arms, 0.7, seed)
    user_bias = get_user_bias_barp(
        num_users, "normal", mean=bias_mean, std=bias_std, seed=seed
    )
    return task, group, user_bias


def _single_balanced_instance(bias_mean=2.0, bias_std=2.5, seed=42):
    num_arms, num_users = 10, 10
    task, group, user_bias = generate_balanced_setup(
        num_arms, num_users, bias_mean, bias_std, seed
    )
    df_f, _, _ = run_experiment(task, user_bias, group)
    return df_f


def experiment_balanced():
    print("\n=== Balanced Environment (μ_β=2, σ_β=2.5, multi-instance) ===")
    all_finals = []
    for inst, seed in enumerate(FINAL_SEEDS):
        print(f"  Instance {inst+1}/{N_INSTANCES} (seed={seed})")
        df = _single_balanced_instance(seed=seed)
        df["instance"] = inst
        all_finals.append(df)

    combined = pd.concat(all_finals)
    agg = (combined.groupby("algorithm")
           .agg(accuracy=("accuracy", "mean"),
                ci=("accuracy", lambda x: 1.96 * x.std() / np.sqrt(len(x))),
                wall_sec=("wall_sec", "mean"))
           .reset_index())

    print("\nBalanced Environment Results:")
    print(agg.to_string(index=False))
    agg.to_csv(f"{RESULT_DIR}/balanced_accuracy.csv", index=False)
    return agg


def get_offline_comparison_algorithms():
    """Return online vs offline variants of MyGroupBiasUCB."""
    return {
        "MyGroup-online": lambda env, **kw: MyGroupBiasUCB(
            env,
            time_horizon=kw.get("time_horizon", HORIZON),
            random_state=kw.get("random_state", None),
        ),
        "MyGroup-offline": lambda env, **kw: MyGroupBiasUCB(
            env,
            time_horizon=kw.get("time_horizon", HORIZON),
            random_state=kw.get("random_state", None),
            em_refit_freq=float('inf'),
        ),
    }


def plot_offline_vs_online(df_anytime, filename):
    plt.figure(figsize=(8, 5))
    colors = {"MyGroup-online": "C0", "MyGroup-offline": "C1"}
    labels = {"MyGroup-online": "Online (periodic refit)",
              "MyGroup-offline": "Offline (single estimation)"}

    for algo in df_anytime.algorithm.unique():
        d = df_anytime[df_anytime.algorithm == algo]
        acc = d.accuracy.values
        ci = d.ci.values
        yerr_lower = np.minimum(ci, acc)
        yerr_upper = np.minimum(ci, 1.0 - acc)

        plt.errorbar(d.timestep, acc, yerr=[yerr_lower, yerr_upper],
                     marker='o', linewidth=2,
                     color=colors.get(algo, None),
                     label=labels.get(algo, algo))

    plt.xlabel("Timestep", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.title("Online vs Offline EM Refits (Adversarial Bias)", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved offline vs online plot to {filename}")


def experiment_offline_vs_online():
    print("\n=== Offline vs Online Comparison (Adversarial Bias) ===")
    num_arms, num_users = 10, 10
    bias_mean, bias_std = 2.0, 2.5
    algos = get_offline_comparison_algorithms()

    all_anytime = []
    for inst, seed in enumerate(FINAL_SEEDS):
        task = generate_task_barp(num_arms, score_std=0.5, seed=seed)
        group = generate_group_membership(num_arms, 0.7, seed)
        task, group = inject_adversarial_setup(task, group)
        user_bias = get_user_bias_barp(
            num_users, "normal", mean=bias_mean, std=bias_std, seed=seed)

        _, df_any, _ = run_experiment(
            task, user_bias, group,
            algos=algos,
            with_checkpoints=True
        )
        if df_any is not None:
            df_any["instance"] = inst
            all_anytime.append(df_any)
        else:
            print(f"  Warning: Instance {inst+1} returned no anytime data")

    if all_anytime:
        combined = pd.concat(all_anytime)
        agg = (combined.groupby(["algorithm", "timestep"])
               .agg(accuracy=("accuracy", "mean"),
                    ci=("accuracy", lambda x: 1.96 * x.std() / np.sqrt(len(x))))
               .reset_index())
        agg.to_csv(f"{RESULT_DIR}/offline_vs_online_anytime.csv", index=False)
        plot_offline_vs_online(agg, f"{RESULT_DIR}/offline_vs_online_plot.png")
        print("\nOffline vs Online summary:")
        print(agg.pivot(index="timestep", columns="algorithm", values="accuracy"))
    else:
        print("No anytime data collected.")

def experiment_main_results():
    print("\n" + "=" * 60)
    print("MAIN RESULTS")
    print("=" * 60)

    num_arms = 10
    num_users = 10

    bias_mean = 2.0
    bias_std = 2.5

    all_results = []

    if len(FINAL_SEEDS) != N_INSTANCES:
        raise ValueError(
            f"FINAL_SEEDS has {len(FINAL_SEEDS)} seeds but N_INSTANCES={N_INSTANCES}"
        )

    print(f"Final untouched problem seeds: {FINAL_SEEDS}")
    print(f"Stochastic runs per instance: {RUNS}")

    for inst, seed in enumerate(FINAL_SEEDS):

        print(
            f"\nInstance {inst + 1}/{N_INSTANCES} "
            f"(seed={seed})"
        )

        # --------------------------------------------------
        # Generate environment
        # --------------------------------------------------
        task = generate_task_barp(
            num_arms,
            score_std=0.5,
            seed=seed,
        )

        group = generate_group_membership(
            num_arms,
            0.7,
            seed,
        )

        # Adversarial bias trap
        task, group = inject_adversarial_setup(
            task,
            group,
        )

        user_bias = get_user_bias_barp(
            num_users,
            "normal",
            mean=bias_mean,
            std=bias_std,
            seed=seed,
        )

        # --------------------------------------------------
        # Run all baselines + proposed method
        # --------------------------------------------------
        df_final, _, _ = run_experiment(
            task,
            user_bias,
            group,
            algos=get_algorithms(),
            with_checkpoints=False,
            collect_regret=False,
        )

        df_final["instance"] = inst
        df_final["problem_seed"] = seed

        all_results.append(df_final)

    # ------------------------------------------------------
    # Aggregate over instances
    # ------------------------------------------------------
    combined = pd.concat(
        all_results,
        ignore_index=True,
    )

    # Save the per-instance accuracies as an audit trail.
    per_instance_path = (
        f"{RESULT_DIR}/main_results_per_instance.csv"
    )
    combined.to_csv(per_instance_path, index=False)

    # CI below is computed across the 20 problem-instance accuracies,
    # not across the pooled stochastic runs.

    results = (
        combined
        .groupby("algorithm")
        .agg(
            accuracy=("accuracy", "mean"),
            ci=(
                "accuracy",
                lambda x:
                    1.96
                    * x.std()
                    / max(np.sqrt(len(x)), 1),
            ),
            wall_sec=("wall_sec", "mean"),
        )
        .reset_index()
    )

    # Sort by accuracy
    results = results.sort_values(
        "accuracy",
        ascending=False,
    )

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------
    csv_path = (
        f"{RESULT_DIR}/"
        f"main_results.csv"
    )

    fig_path = (
        f"{RESULT_DIR}/"
        f"main_results.png"
    )

    results.to_csv(
        csv_path,
        index=False,
    )

    plot_main_results(
        results,
        fig_path,
    )

    # ------------------------------------------------------
    # Print
    # ------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL MAIN RESULTS")
    print("=" * 60)

    print(
        results[
            ["algorithm", "accuracy", "ci", "wall_sec"]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(f"  CSV (aggregate)   : {csv_path}")
    print(f"  CSV (per-instance): {per_instance_path}")
    print(f"  Plot              : {fig_path}")

    return results

def _aligned_group_accuracy(est_g, true_g):
    """Binary latent-group accuracy, invariant to label swapping (0<->1)."""
    if est_g is None:
        return np.nan
    est_g = np.asarray(est_g).astype(int).reshape(-1)
    true_g = np.asarray(true_g).astype(int).reshape(-1)
    if len(est_g) != len(true_g):
        return np.nan
    direct = np.mean(est_g == true_g)
    flipped = np.mean((1 - est_g) == true_g)
    return float(max(direct, flipped))


def _safe_corr(a, b):
    if a is None or b is None:
        return np.nan
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if len(a) != len(b) or len(a) < 2:
        return np.nan
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def run_single_mygroup_diagnostic(
    task,
    user_bias,
    group_membership,
    run_seed,
):
    """
    Run one stochastic MyGroup trial and collect diagnostic quantities.
    """

    env = BradleyTerryLuceModelBias(
        num_arms=len(task),
        num_users=len(user_bias),
        random_state=np.random.RandomState(run_seed),
        skill_vector=task,
        user_bias=user_bias,
        group_membership=group_membership,
    )

    rng = np.random.RandomState(run_seed)

    algo = MyGroupBiasUCB(
        env,
        time_horizon=HORIZON,
        random_state=rng,
        alpha=MYGROUP_ALPHA,
        epsilon=MYGROUP_EPSILON,
        lambda_q=MYGROUP_LAMBDA_Q,
        em_refit_freq=MYGROUP_EM_REFIT_FREQ,
    )

    for _ in range(HORIZON):
        algo.step()

    # --------------------------------------------------
    # True environment
    # --------------------------------------------------
    task_arr = np.asarray(task, dtype=float)
    true_group = np.asarray(group_membership, dtype=int)
    true_beta = np.asarray(user_bias, dtype=float)

    true_best = int(np.argmax(task_arr))

    order = np.argsort(task_arr)
    true_decoy = int(order[-2])

    # --------------------------------------------------
    # Final prediction
    # --------------------------------------------------
    final_pred = _extract_prediction(
        algo,
        task,
        run_seed,
    )

    correct = int(final_pred == true_best)

    # --------------------------------------------------
    # Estimated intrinsic score
    # Prefer s because your algorithm uses it for ranking.
    # Fall back to q if needed.
    # --------------------------------------------------
    if hasattr(algo, "s"):
        estimated_q = np.asarray(algo.s, dtype=float).copy()
    elif hasattr(algo, "q"):
        estimated_q = np.asarray(algo.q, dtype=float).copy()
    else:
        estimated_q = np.full(len(task_arr), np.nan)

    if np.all(np.isnan(estimated_q)):
        estimated_best = -1
        estimated_q_true_best = np.nan
        estimated_q_true_decoy = np.nan
        estimated_q_max_other = np.nan
        estimated_max_other_arm = -1
        q_best_minus_decoy = np.nan
        q_gap_truebest_vs_best_other = np.nan
    else:
        estimated_best = int(np.nanargmax(estimated_q))

        estimated_q_true_best = float(
            estimated_q[true_best]
        )

        estimated_q_true_decoy = float(
            estimated_q[true_decoy]
        )

        other_mask = np.ones(
            len(estimated_q),
            dtype=bool,
        )
        other_mask[true_best] = False

        other_indices = np.arange(
            len(estimated_q)
        )[other_mask]

        estimated_max_other_arm = int(
            other_indices[
                np.nanargmax(
                    estimated_q[other_mask]
                )
            ]
        )

        estimated_q_max_other = float(
            estimated_q[
                estimated_max_other_arm
            ]
        )

        q_best_minus_decoy = float(
            estimated_q_true_best
            - estimated_q_true_decoy
        )

        q_gap_truebest_vs_best_other = float(
            estimated_q_true_best
            - estimated_q_max_other
        )

    # --------------------------------------------------
    # Estimated latent group
    # binary labels have label-switch ambiguity,
    # so score both orientations and keep the better one.
    # --------------------------------------------------
    if hasattr(algo, "g"):
        estimated_group = np.asarray(
            algo.g,
            dtype=int,
        ).copy()

        direct_acc = np.mean(
            estimated_group == true_group
        )

        flipped_acc = np.mean(
            (1 - estimated_group) == true_group
        )

        if flipped_acc > direct_acc:
            aligned_group = 1 - estimated_group
            group_recovery = float(flipped_acc)
            group_was_flipped = 1
        else:
            aligned_group = estimated_group
            group_recovery = float(direct_acc)
            group_was_flipped = 0
    else:
        estimated_group = np.full(
            len(true_group),
            -1,
            dtype=int,
        )
        aligned_group = estimated_group.copy()
        group_recovery = np.nan
        group_was_flipped = 0

    # --------------------------------------------------
    # Estimated evaluator bias
    # --------------------------------------------------
    if hasattr(algo, "beta"):
        estimated_beta = np.asarray(
            algo.beta,
            dtype=float,
        ).copy()

        if (
            len(estimated_beta) == len(true_beta)
            and np.std(estimated_beta) > 1e-12
            and np.std(true_beta) > 1e-12
        ):
            beta_corr = float(
                np.corrcoef(
                    estimated_beta,
                    true_beta,
                )[0, 1]
            )
        else:
            beta_corr = np.nan
    else:
        estimated_beta = np.full(
            len(true_beta),
            np.nan,
        )
        beta_corr = np.nan

    # --------------------------------------------------
    # Is the mistaken winner specifically the decoy?
    # --------------------------------------------------
    predicted_decoy = int(
        final_pred == true_decoy
    )

    estimated_best_is_decoy = int(
        estimated_best == true_decoy
    )

    estimated_best_is_true_best = int(
        estimated_best == true_best
    )

    return {
        "run_seed": run_seed,

        "correct": correct,
        "final_pred": int(final_pred),

        "true_best": true_best,
        "true_decoy": true_decoy,

        "estimated_best": estimated_best,
        "estimated_best_is_true_best":
            estimated_best_is_true_best,
        "estimated_best_is_decoy":
            estimated_best_is_decoy,
        "predicted_decoy":
            predicted_decoy,

        "group_recovery":
            group_recovery,
        "group_was_flipped":
            group_was_flipped,

        "beta_corr":
            beta_corr,

        "estimated_q_true_best":
            estimated_q_true_best,
        "estimated_q_true_decoy":
            estimated_q_true_decoy,
        "estimated_q_max_other":
            estimated_q_max_other,
        "estimated_max_other_arm":
            estimated_max_other_arm,

        "q_best_minus_decoy":
            q_best_minus_decoy,
        "q_gap_truebest_vs_best_other":
            q_gap_truebest_vs_best_other,

        # Save vectors for detailed inspection
        "estimated_q_vector":
            np.array2string(
                estimated_q,
                precision=4,
                separator=";",
            ),

        "estimated_group_vector":
            np.array2string(
                estimated_group,
                separator=";",
            ),

        "aligned_group_vector":
            np.array2string(
                aligned_group,
                separator=";",
            ),

        "estimated_beta_vector":
            np.array2string(
                estimated_beta,
                precision=4,
                separator=";",
            ),
    }

def experiment_mygroup_instance_diagnostics(
    seed_groups=DIAGNOSTIC_SEED_GROUPS,
):
    """
    Compare MyGroup diagnostics between:

      old_42_46:
          seeds 42, 43, 44, 45, 46

      clean_100_104:
          seeds 100, 101, 102, 103, 104

    IMPORTANT:
      Parameters are fixed.
      This function performs NO hyperparameter selection.

    Saves:
      1. mygroup_seed_comparison_summary.csv
      2. mygroup_seed_comparison_runs.csv
      3. mygroup_seed_comparison_environment.csv
      4. mygroup_seed_group_summary.csv
    """

    print("\n" + "=" * 80)
    print("MYGROUP SEED-GROUP DIAGNOSTICS")
    print("=" * 80)

    print(
        f"Tuned parameters fixed at: "
        f"alpha={MYGROUP_ALPHA}, "
        f"epsilon={MYGROUP_EPSILON}, "
        f"lambda_q={MYGROUP_LAMBDA_Q}, "
        f"refit={MYGROUP_EM_REFIT_FREQ}"
    )

    for cohort, seeds in seed_groups.items():
        print(
            f"{cohort}: {list(seeds)}"
        )

    num_arms = 10
    num_users = 10

    all_runs = []
    env_rows = []

    # ==================================================
    # Run both seed groups
    # ==================================================
    for cohort, seeds in seed_groups.items():

        print("\n" + "-" * 80)
        print(f"COHORT: {cohort}")
        print("-" * 80)

        for inst, problem_seed in enumerate(seeds):

            print(
                f"\n{cohort} | "
                f"instance {inst + 1}/{len(seeds)} "
                f"| seed={problem_seed}"
            )

            # ------------------------------------------
            # Generate exactly the same environment
            # protocol as the main experiment
            # ------------------------------------------
            task = generate_task_barp(
                num_arms,
                score_std=0.5,
                seed=problem_seed,
            )

            group = generate_group_membership(
                num_arms,
                0.7,
                problem_seed,
            )

            task, group = inject_adversarial_setup(
                task,
                group,
            )

            user_bias = get_user_bias_barp(
                num_users,
                "normal",
                mean=2.0,
                std=2.5,
                seed=problem_seed,
            )

            # ------------------------------------------
            # Ground truth
            # ------------------------------------------
            order = np.argsort(task)

            true_best = int(
                order[-1]
            )

            true_decoy = int(
                order[-2]
            )

            env_rows.append({
                "cohort":
                    cohort,

                "problem_seed":
                    problem_seed,

                "true_best":
                    true_best,

                "true_decoy":
                    true_decoy,

                "true_best_group":
                    int(group[true_best]),

                "true_decoy_group":
                    int(group[true_decoy]),

                "task_gap_best_decoy":
                    float(
                        task[true_best]
                        - task[true_decoy]
                    ),

                "task_std":
                    float(np.std(task)),

                "bias_sample_mean":
                    float(np.mean(user_bias)),

                "bias_sample_std":
                    float(np.std(user_bias)),

                "bias_min":
                    float(np.min(user_bias)),

                "bias_max":
                    float(np.max(user_bias)),

                "group1_fraction":
                    float(np.mean(group)),

                "task_vector":
                    np.array2string(
                        np.asarray(task),
                        precision=4,
                        separator=";",
                    ),

                "group_vector":
                    np.array2string(
                        np.asarray(group),
                        separator=";",
                    ),

                "bias_vector":
                    np.array2string(
                        np.asarray(user_bias),
                        precision=4,
                        separator=";",
                    ),
            })

            # ------------------------------------------
            # 100 stochastic runs
            # ------------------------------------------
            outputs = Parallel(
                n_jobs=N_JOBS
            )(
                delayed(
                    run_single_mygroup_diagnostic
                )(
                    task,
                    user_bias,
                    group,
                    run_seed=r,
                )
                for r in range(RUNS)
            )

            # ------------------------------------------
            # Add problem-instance identifiers
            # ------------------------------------------
            for row in outputs:
                row["cohort"] = cohort
                row["problem_seed"] = problem_seed
                row["instance"] = inst

            all_runs.extend(outputs)

            df_inst = pd.DataFrame(
                outputs
            )

            print(
                f"  accuracy="
                f"{df_inst['correct'].mean():.3f}"
            )

            print(
                f"  group recovery="
                f"{df_inst['group_recovery'].mean():.3f}"
            )

            print(
                f"  beta corr="
                f"{df_inst['beta_corr'].mean():.3f}"
            )

            print(
                f"  q(best-maxOther)="
                f"{df_inst['q_gap_truebest_vs_best_other'].mean():.3f}"
            )

            print(
                f"  q(best-decoy)="
                f"{df_inst['q_best_minus_decoy'].mean():.3f}"
            )

            print(
                f"  estimated-best-is-decoy="
                f"{df_inst['estimated_best_is_decoy'].mean():.3f}"
            )

    # ==================================================
    # DataFrames
    # ==================================================
    runs_df = pd.DataFrame(
        all_runs
    )

    env_df = pd.DataFrame(
        env_rows
    )

    # ==================================================
    # Per-problem-seed summary
    # ==================================================
    summary = (
        runs_df
        .groupby(
            ["cohort", "problem_seed"]
        )
        .agg(
            accuracy=(
                "correct",
                "mean",
            ),

            accuracy_run_ci=(
                "correct",
                lambda x:
                    1.96
                    * x.std(ddof=1)
                    / np.sqrt(len(x)),
            ),

            group_recovery=(
                "group_recovery",
                "mean",
            ),

            group_recovery_std=(
                "group_recovery",
                "std",
            ),

            beta_corr=(
                "beta_corr",
                "mean",
            ),

            beta_corr_std=(
                "beta_corr",
                "std",
            ),

            estimated_best_true_rate=(
                "estimated_best_is_true_best",
                "mean",
            ),

            estimated_best_decoy_rate=(
                "estimated_best_is_decoy",
                "mean",
            ),

            predicted_decoy_rate=(
                "predicted_decoy",
                "mean",
            ),

            estimated_q_true_best=(
                "estimated_q_true_best",
                "mean",
            ),

            estimated_q_true_decoy=(
                "estimated_q_true_decoy",
                "mean",
            ),

            estimated_q_max_other=(
                "estimated_q_max_other",
                "mean",
            ),

            q_best_minus_decoy=(
                "q_best_minus_decoy",
                "mean",
            ),

            q_best_minus_decoy_std=(
                "q_best_minus_decoy",
                "std",
            ),

            q_gap=(
                "q_gap_truebest_vs_best_other",
                "mean",
            ),

            q_gap_std=(
                "q_gap_truebest_vs_best_other",
                "std",
            ),
        )
        .reset_index()
    )

    # ------------------------------------------
    # Merge environment-level quantities
    # ------------------------------------------
    env_small = env_df.drop(
        columns=[
            "task_vector",
            "group_vector",
            "bias_vector",
        ]
    )

    summary = summary.merge(
        env_small,
        on=[
            "cohort",
            "problem_seed",
        ],
        how="left",
    )

    # ==================================================
    # Cohort-level comparison
    #
    # IMPORTANT:
    # CI here is across the five problem instances,
    # not across the 500 stochastic runs.
    # ==================================================
    cohort_summary = (
        summary
        .groupby("cohort")
        .agg(
            mean_accuracy=(
                "accuracy",
                "mean",
            ),

            instance_accuracy_std=(
                "accuracy",
                "std",
            ),

            instance_accuracy_ci=(
                "accuracy",
                lambda x:
                    1.96
                    * x.std(ddof=1)
                    / np.sqrt(len(x)),
            ),

            mean_group_recovery=(
                "group_recovery",
                "mean",
            ),

            mean_beta_corr=(
                "beta_corr",
                "mean",
            ),

            mean_q_gap=(
                "q_gap",
                "mean",
            ),

            mean_q_best_minus_decoy=(
                "q_best_minus_decoy",
                "mean",
            ),

            decoy_failure_rate=(
                "estimated_best_decoy_rate",
                "mean",
            ),
        )
        .reset_index()
    )

    # ==================================================
    # Save
    # ==================================================
    summary_path = (
        f"{RESULT_DIR}/"
        "mygroup_seed_comparison_summary.csv"
    )

    runs_path = (
        f"{RESULT_DIR}/"
        "mygroup_seed_comparison_runs.csv"
    )

    env_path = (
        f"{RESULT_DIR}/"
        "mygroup_seed_comparison_environment.csv"
    )

    cohort_path = (
        f"{RESULT_DIR}/"
        "mygroup_seed_group_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    runs_df.to_csv(
        runs_path,
        index=False,
    )

    env_df.to_csv(
        env_path,
        index=False,
    )

    cohort_summary.to_csv(
        cohort_path,
        index=False,
    )

    # ==================================================
    # Print
    # ==================================================
    print("\n" + "=" * 80)
    print("PER-INSTANCE SUMMARY")
    print("=" * 80)

    display_cols = [
        "cohort",
        "problem_seed",
        "accuracy",
        "group_recovery",
        "beta_corr",
        "estimated_best_decoy_rate",
        "q_best_minus_decoy",
        "q_gap",
    ]

    print(
        summary[
            display_cols
        ].to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)
    print("COHORT COMPARISON")
    print("=" * 80)

    print(
        cohort_summary.to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(f"  {summary_path}")
    print(f"  {runs_path}")
    print(f"  {env_path}")
    print(f"  {cohort_path}")

    print("\nInterpretation guide:")
    print(
        "  group_recovery ~= 1 but q_gap < 0 "
        "-> groups are recovered, but intrinsic ranking fails"
    )
    print(
        "  q_best_minus_decoy < 0 "
        "-> the favored-group decoy outranks the true best intrinsically"
    )
    print(
        "  estimated_best_decoy_rate ~= 1 "
        "-> failure is specifically caused by the injected decoy"
    )
    print(
        "  old_42_46 >> clean_100_104 "
        "-> strong problem-instance sensitivity / selection effect"
    )

    return (
        summary,
        runs_df,
        env_df,
        cohort_summary,
    )


def plot_main_results(df, filename):
    if df is None or df.empty:
        return

    plt.figure(figsize=(10, 6))

    x = np.arange(len(df))

    bars = plt.bar(
        x,
        df["accuracy"].values,
        yerr=df["ci"].values,
        capsize=4,
    )

    plt.xticks(
        x,
        df["algorithm"].values,
        rotation=45,
        ha="right",
    )

    plt.ylabel(
        "Optimal Arm Selection Accuracy",
        fontsize=12,
    )

    plt.xlabel(
        "Algorithm",
        fontsize=12,
    )

    plt.ylim(0, 1.05)

    plt.title(
        "Main Results under Adversarial Bias",
        fontsize=14,
    )

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.5,
    )

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=150,
    )

    plt.close()


# ────────────────────────────────────────────────────────────────────────────

def run_all():
    # Default: only the frozen final Main Table.
    experiment_main_results()


def run_paper_experiments():
    # Formal paper evaluation under one frozen protocol.
    # All experiments below use FINAL_SEEDS = 200..219 and RUNS = 20.
    experiment_main_results()
    experiment_bias_mean()
    experiment_bias_std()
    experiment_balanced()
    experiment_ablation()
    experiment_anytime()
    experiment_weak_regret()


if __name__ == "__main__":
    import multiprocessing
    import argparse
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-all", action="store_true",
                        help="Run all experiments used in the final paper.")
    parser.add_argument("--main-only", action="store_true",
                        help="Run only the final Main Table.")
    parser.add_argument("--bias-only", action="store_true",
                        help="Run bias-mean and bias-std robustness experiments.")
    parser.add_argument("--balanced-only", action="store_true")
    parser.add_argument("--ablation-only", action="store_true")
    parser.add_argument("--weak-regret-only", action="store_true")
    parser.add_argument("--anytime-only", action="store_true")
    parser.add_argument("--offline-compare", action="store_true")
    parser.add_argument("--diagnose-mygroup", action="store_true")
    args = parser.parse_args()

    print("Frozen protocol:")
    print(f"  tuning seeds : {TUNING_SEEDS}")
    print(f"  final seeds  : {FINAL_SEEDS}")
    print(f"  runs/instance: {RUNS}")
    print(f"  horizon      : {HORIZON}")

    if args.diagnose_mygroup:
        experiment_mygroup_instance_diagnostics()
    elif args.offline_compare:
        experiment_offline_vs_online()
    elif args.ablation_only:
        experiment_ablation()
    elif args.anytime_only:
        experiment_anytime()
    elif args.weak_regret_only:
        experiment_weak_regret()
    elif args.balanced_only:
        experiment_balanced()
    elif args.bias_only:
        experiment_bias_mean()
        experiment_bias_std()
    elif args.paper_all:
        run_paper_experiments()
    else:
        # Safe default: do not accidentally launch every expensive experiment.
        # experiment_main_results()
        experiment_num_arms()
        experiment_num_users()
