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
RUNS         = 100
N_INSTANCES  = 5
N_JOBS       = -1
CHECKPOINTS  = [500, 1000, 1500, 2000, 2500, 3000]



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

def get_offline_group_ucb_factory():
    return lambda env, **kw: MyGroupBiasUCB(
        env,
        time_horizon=kw.get("time_horizon", HORIZON),
        random_state=kw.get("random_state", None),
        em_refit_freq=float('inf'),
    )

def get_ablation_algorithms():
    return {
        "Full": lambda env, **kw: MyGroupBiasUCB(
            env,
            time_horizon=kw.get("time_horizon", HORIZON),
            random_state=kw.get("random_state", None),
        ),

        "noWarmup": lambda env, **kw: MyGroupBiasUCB(
            env,
            time_horizon=kw.get("time_horizon", HORIZON),
            random_state=kw.get("random_state", None),
            warmup_obs_per_user_pair=0,
            warmup_max_frac=0.0,
        ),

        "noInit": lambda env, **kw: MyGroupBiasUCB(
            env,
            time_horizon=kw.get("time_horizon", HORIZON),
            random_state=kw.get("random_state", None),
            use_spectral_init=False,
            use_svd_init=False,
            em_restarts=3,
        ),

        "noXGroup": lambda env, **kw: MyGroupBiasUCB(
            env,
            time_horizon=kw.get("time_horizon", HORIZON),
            random_state=kw.get("random_state", None),
            cross_group_correction=False,
        ),

        "noGdGate": lambda env, **kw: MyGroupBiasUCB(
            env,
            time_horizon=kw.get("time_horizon", HORIZON),
            random_state=kw.get("random_state", None),
            gd_stability_gate=0.0,
        ),

        "noDecay": lambda env, **kw: MyGroupBiasUCB(
            env,
            time_horizon=kw.get("time_horizon", HORIZON),
            random_state=kw.get("random_state", None),
            recent_decay=1.0,
        ),
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

def experiment_sensitivity_lambda():
    print("\n=== Sensitivity Analysis: λ (Adversarial Bias, μ_β=2, σ_β=2.5) ===")
    algos = get_sensitivity_algorithms()
    all_finals = []
    for inst in range(N_INSTANCES):
        seed = 42 + inst
        task  = generate_task_barp(10, score_std=0.5, seed=seed)
        group = generate_group_membership(10, 0.7, seed)
        task, group = inject_adversarial_setup(task, group)
        user_bias = get_user_bias_barp(10, "normal", mean=2, std=2.5, seed=seed)
        df_f, _ = run_experiment(task, user_bias, group, algos=algos)
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

def run_single(task, user_bias, group_membership, algo_factory, seed,
               checkpoints=None):
    env = BradleyTerryLuceModelBias(
        num_arms=len(task),
        num_users=len(user_bias),
        random_state=np.random.RandomState(seed),
        skill_vector=task,
        user_bias=user_bias,
        group_membership=group_membership,
    )
    rng = np.random.RandomState(seed)
    horizon = HORIZON

    if callable(algo_factory) and not isinstance(algo_factory, type):
        algo = algo_factory(env, time_horizon=horizon, random_state=rng)
    else:
        try:
            algo = algo_factory(env, time_horizon=horizon, random_state=rng)
        except TypeError:
            try:
                algo = algo_factory(env, time_horizon=horizon)
            except TypeError:
                algo = algo_factory(env)

    checkpoint_set = set(checkpoints) if checkpoints else set()
    checkpoint_preds = {}

    for t in range(1, horizon + 1):
        if hasattr(algo, "step"):
            algo.step()

        if t in checkpoint_set:
            checkpoint_preds[t] = _extract_prediction(algo, task, seed)

    final_pred = _extract_prediction(algo, task, seed)
    return final_pred, checkpoint_preds


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


def run_experiment(task, user_bias, group_membership, algos=None,
                   with_checkpoints=False):
    if algos is None:
        algos = get_algorithms()
    true_best = int(np.argmax(task))
    results_final   = []
    results_anytime = []

    for name, algo in algos.items():
        print(f"    algorithm: {name}")
        ckpts = CHECKPOINTS if with_checkpoints else None

        t0 = time.perf_counter()
        outputs = Parallel(n_jobs=N_JOBS)(
            delayed(run_single)(
                task, user_bias, group_membership, algo, r, ckpts)
            for r in range(RUNS)
        )
        wall_time = time.perf_counter() - t0

        preds = [o[0] for o in outputs]
        acc_list = [1 if p == true_best else 0 for p in preds]
        results_final.append({
            "algorithm":  name,
            "accuracy":   float(np.mean(acc_list)),
            "ci":         1.96 * float(np.std(acc_list)) / np.sqrt(RUNS),
            "wall_sec":   round(wall_time, 1),
        })

        if with_checkpoints:
            for cp in CHECKPOINTS:
                cp_preds = [o[1].get(cp, -1) for o in outputs]
                cp_acc = [1 if p == true_best else 0 for p in cp_preds]
                results_anytime.append({
                    "algorithm": name,
                    "timestep":  cp,
                    "accuracy":  float(np.mean(cp_acc)),
                    "ci":        1.96 * float(np.std(cp_acc)) / np.sqrt(RUNS),
                })

    df_final = pd.DataFrame(results_final)
    df_anytime = pd.DataFrame(results_anytime) if with_checkpoints else None
    return df_final, df_anytime


def run_multi_instance(experiment_fn, n_instances=N_INSTANCES, **kwargs):
    all_finals = []
    for inst in range(n_instances):
        print(f"  Instance {inst+1}/{n_instances} (seed={42 + inst})")
        df_f, _ = experiment_fn(seed=42 + inst, **kwargs)
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
    df_f, _ = run_experiment(task, user_bias, group)
    df_f["bias_mean"] = bias_mean
    return df_f


def experiment_bias_mean():
    print("\n=== Bias Mean Sweep (σ_β = 2.5, multi-instance) ===")
    all_results = []
    for b in [0, 1, 2, 3, 4, 5]:
        print(f"bias_mean = {b}")
        instance_results = []
        for inst in range(N_INSTANCES):
            df = _single_bias_mean_instance(b, seed=42 + inst)
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
        for inst in range(N_INSTANCES):
            seed = 42 + inst
            num_arms, num_users = 10, 10
            task  = generate_task_barp(num_arms, score_std=0.5, seed=seed)
            group = generate_group_membership(num_arms, 0.7, seed)
            task, group = inject_adversarial_setup(task, group)
            user_bias = get_user_bias_barp(
                num_users, "normal", mean=2, std=s, seed=seed)
            df_f, _ = run_experiment(task, user_bias, group)
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
        for inst in range(N_INSTANCES):
            seed = 42 + inst
            num_users = 10
            task  = generate_task_barp(k, score_std=0.5, seed=seed)
            group = generate_group_membership(k, 0.7, seed)
            task, group = inject_adversarial_setup(task, group)
            user_bias = get_user_bias_barp(
                num_users, "normal", mean=2, std=2.5, seed=seed)
            df_f, _ = run_experiment(task, user_bias, group)
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
        for inst in range(N_INSTANCES):
            seed = 42 + inst
            num_arms = 10
            task  = generate_task_barp(num_arms, score_std=0.5, seed=seed)
            group = generate_group_membership(num_arms, 0.7, seed)
            task, group = inject_adversarial_setup(task, group)
            user_bias = get_user_bias_barp(
                m, "normal", mean=2, std=2.5, seed=seed)
            df_f, _ = run_experiment(task, user_bias, group)
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
    print("\n=== Anytime Accuracy ===")
    num_arms, num_users, seed = 10, 10, 42
    task  = generate_task_barp(num_arms, score_std=0.5, seed=seed)
    group = generate_group_membership(num_arms, 0.7, seed)
    task, group = inject_adversarial_setup(task, group)
    user_bias = get_user_bias_barp(
        num_users, "normal", mean=2, std=2.5, seed=seed)

    _, df_anytime = run_experiment(
        task, user_bias, group, with_checkpoints=True)

    if df_anytime is not None:
        df_anytime.to_csv(
            f"{RESULT_DIR}/anytime_accuracy.csv", index=False)
        plot_anytime(df_anytime, f"{RESULT_DIR}/anytime_plot.png")


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
        for inst in range(N_INSTANCES):
            seed = 42 + inst
            task  = generate_task_barp(num_arms, score_std=0.5, seed=seed)
            group = generate_group_membership(num_arms, 0.7, seed)
            task, group = inject_adversarial_setup(task, group)
            user_bias = get_user_bias_barp(
                num_users, "normal",
                mean=cond["bias_mean"],
                std=cond["bias_std"],
                seed=seed)
 
            df_f, _ = run_experiment(
                task, user_bias, group, algos=ablation_algos)
            df_f["instance"] = inst
            df_f["condition"] = cond["label"]
            all_results.append(df_f)
 
        results = pd.concat(all_results)
        agg = (results.groupby("algorithm")
               .agg(accuracy=("accuracy", "mean"),
                    ci=("accuracy",
                        lambda x: 1.96 * x.std() / max(np.sqrt(len(x)), 1)),
                    wall_sec=("wall_sec", "mean"))
               .reset_index())
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
    df_f, _ = run_experiment(task, user_bias, group)
    return df_f


def experiment_balanced():
    print("\n=== Balanced Environment (μ_β=2, σ_β=2.5, multi-instance) ===")
    all_finals = []
    for inst in range(N_INSTANCES):
        print(f"  Instance {inst+1}/{N_INSTANCES} (seed={42 + inst})")
        df = _single_balanced_instance(seed=42 + inst)
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
    for inst in range(N_INSTANCES):
        seed = 42 + inst
        task = generate_task_barp(num_arms, score_std=0.5, seed=seed)
        group = generate_group_membership(num_arms, 0.7, seed)
        task, group = inject_adversarial_setup(task, group)
        user_bias = get_user_bias_barp(
            num_users, "normal", mean=bias_mean, std=bias_std, seed=seed)

        _, df_any = run_experiment(
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


# ────────────────────────────────────────────────────────────────────────────

def run_all():
    # report_warmup_budget()
    # experiment_balanced()
    # experiment_bias_mean()
    # experiment_bias_std()
    # experiment_num_arms()
    # experiment_num_users()
    # experiment_anytime()
    experiment_ablation()
    experiment_sensitivity_lambda()
    # experiment_offline_vs_online() 


if __name__ == "__main__":
    import multiprocessing
    import argparse
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--run_barp", action="store_true", default=True)
    parser.add_argument("--ablation-only", action="store_true")
    parser.add_argument("--anytime-only", action="store_true")
    parser.add_argument("--offline-compare", action="store_true")
    args = parser.parse_args()

    if args.offline_compare:
        experiment_offline_vs_online()
    elif args.ablation_only:
        experiment_ablation()
    elif args.anytime_only:
        experiment_anytime()
    elif args.run_barp:
        run_all()