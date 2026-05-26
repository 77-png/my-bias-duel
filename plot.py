import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import FuncFormatter

# ======================
# 配置参数
# ======================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'figure.figsize': (10, 6),
    'figure.dpi': 300
})

# ======================
# 可视化函数 (已修复列名问题)
# ======================
def format_regret(x, pos):
    if x >= 1e5:
        return f'{x/1e5:.0f}e5'
    elif x >= 1e4:
        return f'{x/1e4:.0f}e4'
    return f'{x:.0f}'

def plot_experiment(results, title, x_label, y_label, group_col, 
                   hue_order=None, style_order=None, save_name=None):
    """修复列名问题的可视化函数"""
    print("Available columns in results:", results.columns.tolist())
    
    # 1. 修复时间列 (已处理)
    time_col = 'time'
    if 'time' not in results.columns:
        possible_time_cols = ['time_step', 'timestep', 'time_index', 'step']
        for col in possible_time_cols:
            if col in results.columns:
                time_col = col
                print(f"Using '{time_col}' as time column")
                break
    
    # 2. 修复后悔列 (关键修复！)
    regret_col = 'cumulative_regret'
    possible_regret_cols = [
        'cumulative_regret', 
        'Cumulative Average Regret', 
        'Cumulative Weak Regret',
        'Cumulative Regret'
    ]
    for col in possible_regret_cols:
        if col in results.columns:
            regret_col = col
            print(f"Using '{regret_col}' as regret column")
            break
    
    # 3. 处理分组列 (保持原逻辑)
    if group_col == 'bias_mean':
        results[group_col] = results[group_col].str.replace('mean=', '').astype(float)
    elif group_col == 'bias_std':
        results[group_col] = results[group_col].str.replace('std=', '').astype(float)
    elif group_col == 'arm_variance':
        results[group_col] = results[group_col].str.replace('var=', '').astype(float)
    elif group_col == 'num_arms':
        results[group_col] = results[group_col].str.replace('K=', '').astype(int)
    elif group_col == 'num_users':
        results[group_col] = results[group_col].str.replace('M=', '').astype(int)
    
    # 4. 组合数据 (使用修复后的列名)
    grouped = results.groupby([time_col, group_col, 'algorithm'])[regret_col].mean().reset_index()
    
    # 5. 绘制图表
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    sns.lineplot(
        data=grouped,
        x=time_col,
        y=regret_col,
        hue=group_col,
        style='algorithm',
        markers=True,
        dashes=False,
        linewidth=2.5,
        markersize=8,
        alpha=0.9
    )
    
    plt.title(title, pad=20)
    plt.xlabel(x_label, labelpad=10)
    plt.ylabel(y_label, labelpad=10)
    
    ax.yaxis.set_major_formatter(FuncFormatter(format_regret))
    plt.xscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if group_col == 'distribution':
        legend = plt.legend(title=group_col.title(), loc='upper left', bbox_to_anchor=(1.05, 1))
    else:
        legend = plt.legend(title=group_col.title(), loc='upper left', bbox_to_anchor=(1.05, 1))
    
    # 6. 保存/显示
    if save_name:
        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig(f"results/{save_name}.png", bbox_inches='tight')
        plt.close()
        print(f"✅ 保存图表: results/{save_name}.png")
    else:
        plt.show()

# ======================
# 执行所有实验可视化
# ======================
def visualize_all_experiments():
    # 确保结果目录存在
    os.makedirs("results", exist_ok=True)
    
    # 1. 偏见均值实验
    bias_mean_file = "barp-bias_mean-variance_2.0.pk1"
    if os.path.exists(f"{bias_mean_file}"):
        results = pd.read_pickle(bias_mean_file)
        print(f"\nLoaded {bias_mean_file}, columns: {results.columns.tolist()}")
        plot_experiment(
            results,
            title="Effect of Bias Mean on Cumulative Regret (Figure 2 Left)",
            x_label="Time Steps (log scale)",
            y_label="Cumulative Regret",
            group_col="bias_mean",
            save_name="barp_bias_mean_plot"
        )
    
    # 2. 偏见方差实验
    bias_std_file = "barp-bias_std-variance_2.0.pk1"
    if os.path.exists(f"{bias_std_file}"):
        results = pd.read_pickle(bias_std_file)
        print(f"\nLoaded {bias_std_file}, columns: {results.columns.tolist()}")
        plot_experiment(
            results,
            title="Effect of Bias Standard Deviation on Cumulative Regret (Figure 2 Right)",
            x_label="Time Steps (log scale)",
            y_label="Cumulative Regret",
            group_col="bias_std",
            save_name="barp_bias_std_plot"
        )
    
    # 3. 偏见分布实验
    bias_dist_file = "barp-bias_distribution-variance_2.0.pk1"
    if os.path.exists(f"{bias_dist_file}"):
        results = pd.read_pickle(bias_dist_file)
        print(f"\nLoaded {bias_dist_file}, columns: {results.columns.tolist()}")
        plot_experiment(
            results,
            title="Performance under Different Bias Distributions (Figure 5)",
            x_label="Time Steps (log scale)",
            y_label="Cumulative Regret",
            group_col="distribution",
            hue_order=['uniform', 'normal', 'exponential'],
            style_order=['DT', 'RC', 'RUCB', 'UCB', 'MBTW', 'DT-B(*)', 'RC-B(*)', 'RUCB-B(*)', 'BS-UN(*)', 'BS-K(*)', 'MyBT', 'MyGroup'],
            save_name="barp_bias_distribution_plot"
        )
    
    # 4. Arm方差实验
    arm_var_file = "barp-arm_variance-bias_mean_2.0.pk1"
    if os.path.exists(f"{arm_var_file}"):
        results = pd.read_pickle(f"{arm_var_file}")
        print(f"Loaded {arm_var_file}, columns: {results.columns.tolist()}")
        plot_experiment(
            results,
            title="Effect of Arm Score Variance on Algorithm Performance",
            x_label="Time Steps (log scale)",
            y_label="Cumulative Regret",
            group_col="arm_variance",
            save_name="barp_arm_variance_plot"
        )
    
    # 5. Arm数量实验
    num_arms_file = "barp-num_arms-variance_2.0.pk1"
    if os.path.exists(f"{num_arms_file}"):
        results = pd.read_pickle(f"{num_arms_file}")
        print(f"Loaded {num_arms_file}, columns: {results.columns.tolist()}")
        plot_experiment(
            results,
            title="Effect of Number of Arms on Algorithm Performance",
            x_label="Time Steps (log scale)",
            y_label="Cumulative Regret",
            group_col="num_arms",
            save_name="barp_num_arms_plot"
        )
    
    # 6. 用户数量实验
    num_users_file = "barp-num_users-variance_2.0.pk1"
    if os.path.exists(f"{num_users_file}"):
        results = pd.read_pickle(f"{num_users_file}")
        print(f"Loaded {num_users_file}, columns: {results.columns.tolist()}")
        plot_experiment(
            results,
            title="Effect of Number of Users on Algorithm Performance",
            x_label="Time Steps (log scale)",
            y_label="Cumulative Regret",
            group_col="num_users",
            save_name="barp_num_users_plot"
        )
    
    print("\n🎉 所有图表已成功生成！请查看 results/ 目录下的PNG文件")

# ======================
# 运行可视化
# ======================
if __name__ == "__main__":
    visualize_all_experiments()