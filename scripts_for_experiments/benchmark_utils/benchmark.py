"""Run experiments on the command line.
"""

import inspect
import time
import re
from typing import Dict, Union
from typing import Generator, Callable
from typing import List, Type, Optional
from copy import deepcopy
from pathlib import Path

from joblib import delayed
from joblib import Parallel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from duelpy.stats.metrics import AverageRegret, WeakRegret
from duelpy.stats.metrics import BestArmRate
from duelpy.stats.metrics import Cumulative
from duelpy.stats.metrics import ExponentialMovingAverage
from duelpy.stats.metrics import Metric

from duel_bias.algorithms.BiasAlgorithm import BiasAlgorithm
from duel_bias.utils.metrics import HasBestArmRate
from duel_bias.utils.metrics import RoundTime
from duel_bias.feedback.bias_feedback_decorators import MetricKeepingFeedbackMechanism
from duel_bias.feedback.bias_tracker import BiasTracker

from scripts_for_experiments.benchmark_utils.bradley_terry_luce_model import BradleyTerryLuceModel
from scripts_for_experiments.benchmark_utils.bradley_terry_luce_model import BradleyTerryLuceModelBias
Type_BTLModel = Callable[..., Union[BradleyTerryLuceModelBias, BradleyTerryLuceModel]]

def run_single_algorithm(
    task_random_state: np.random.default_rng,
    num_arms: int,
    algorithm_class: Type[BiasAlgorithm],
    environment_class: Type_BTLModel,
    parameters: Dict,
    sample_interval: int,
    run_id: int,
    optional_parameters: Optional[Dict] = None,
) -> pd.DataFrame:
    """Execute one algorithm for one problem setting and return the results.
    """
    environment_parameters = {
        "num_arms": num_arms,
        "random_state": task_random_state,
    }

    environment_parameters = {
        k: v
        for (k, v) in environment_parameters.items()
        if k in inspect.getfullargspec(environment_class.__init__)[0]
    }

    feedback_mechanism = environment_class(**environment_parameters)
    metrics: Dict[str, Metric] = {
        "Cumulative Weak Regret": Cumulative(
            WeakRegret(feedback_mechanism.preference_matrix)
        ),
        "Rate of Rounds with Best Arm": ExponentialMovingAverage(
            HasBestArmRate(feedback_mechanism.preference_matrix.get_condorcet_winner()),
            alpha=0.01,
        ),
        "Cumulative Average Regret": Cumulative(
            AverageRegret(feedback_mechanism.preference_matrix)
        ),
        "Best Arm Rate": ExponentialMovingAverage(
            BestArmRate(feedback_mechanism.preference_matrix.get_condorcet_winner()),
            alpha=0.01,
        ),
        "Time of Rounds": RoundTime(),
    }
    wrapped_feedback = MetricKeepingFeedbackMechanism(
        feedback_mechanism, metrics=metrics, sample_interval=sample_interval
    )

    # Parameters.
    parameters["random_state"] = task_random_state
    if optional_parameters is not None:
        parameters.update(optional_parameters)

    # Filter accepted parameters.
    parameters_to_pass = dict()
    for parameter in parameters.keys():
        if parameter in inspect.getfullargspec(algorithm_class.__init__)[0]:
            parameters_to_pass[parameter] = parameters[parameter]

    # Run
    instance = algorithm_class(wrapped_feedback, **parameters_to_pass)
    instance.run()

    num_evaluators = len(feedback_mechanism.user_bias)

    if hasattr(instance, "get_bias") and instance.get_bias() is not None:
        evaluator_bias = instance.get_bias()
    else:
        evaluator_bias = [None,] * num_evaluators

    if hasattr(instance, "get_comparative_bias") and instance.get_comparative_bias() is not None:
        comparative_bias = instance.get_comparative_bias()
    else:
        comparative_bias = [None,] * num_evaluators

    bias_error, bcd_residual = None, None
    if hasattr(instance, "bias_tracker") \
            and isinstance(instance.bias_tracker, BiasTracker):
        bias_error, bcd_residual = instance.bias_tracker.get_error()

    len_array = len(wrapped_feedback.results["time_step"])

    if bias_error:
        if len(bias_error) < len_array:
            bias_error.extend([None, ] * (len_array - len(bias_error)))
        elif len(bias_error) > len_array:
            raise ValueError("Error of Evaluator Bias has length larger than time_step")
        wrapped_feedback.results["Error of Evaluator Bias"] = bias_error
    else:
        wrapped_feedback.results["Error of Evaluator Bias"] = [None, ] * len_array

    if bcd_residual:
        if len(bcd_residual) < len_array:
            bcd_residual.extend([None, ] * (len_array - len(bcd_residual)))
        elif len(bcd_residual) > len_array:
            raise ValueError("Error of Evaluator Bias has length larger than time_step")
        wrapped_feedback.results["BCD Residual"] = bcd_residual
    else:
        wrapped_feedback.results["BCD Residual"] = [None, ] * len_array

    # Save results
    data_frame = pd.DataFrame(wrapped_feedback.results)
    data_frame["algorithm"] = optional_parameters["name"] if "name" in optional_parameters else algorithm_class.__name__
    data_frame["run_id"] = run_id
    #   Convert ndarray to string representation for storage
    data_frame["bias"] = transform_array_to_string(evaluator_bias)
    data_frame["comparative_bias"] = transform_array_to_string(comparative_bias)

    return data_frame

def run_experiment(
    algorithms: List[Type[BiasAlgorithm]],
    environment_class: Type_BTLModel,
    time_horizon: int,
    sample_interval: int,
    num_arms: int,
    runs: int,
    n_jobs: int = 1,
    base_random_seed: int = 0,
    list_optional_parameters: Optional[List[Dict]] = None,
) -> pd.DataFrame:
    """Run the experiment.
    """

    start_time = time.time()
    failure_probability = (
        1 / time_horizon
    )
    parameters = {
        "time_horizon": time_horizon,
        "failure_probability": failure_probability,
    }

    def job_producer() -> Generator:
        for index, algorithm_class in enumerate(algorithms):
            algorithm_name = algorithm_class.__name__
            for run_id in range(runs):
                random_state = np.random.RandomState(
                    (base_random_seed + hash(algorithm_name) + run_id) % 2 ** 32
                )
                yield delayed(run_single_algorithm)(
                    random_state,
                    num_arms,
                    algorithm_class,
                    deepcopy(environment_class),
                    parameters,
                    sample_interval,
                    run_id,
                    deepcopy(list_optional_parameters[index]),
                )

    jobs = list(job_producer())
    result = Parallel(n_jobs=n_jobs, verbose=10)(jobs)
    runtime = time.time() - start_time
    print(f"Experiments took {round(runtime)}s.")

    # For keeping all arrays in element of results has the same length, we have filled with None in some columns
    # Here we drop these columns in df which is all None for df in results
    for df in result:
        for key in ["BCD Residual", "Error of Evaluator Bias"]:
            if key in df.columns and df[key].isna().all():
                df.drop(columns=[key], inplace=True)
    
    return pd.concat(result)

COLOR_PALETTE_USER_BIAS = [
    "#D9534F", "#41AAAA", "#3A74BF", "#826890", "#5A757C", "#E99A3B", "#7AC377"
]
COLOR_PALETTE_PREFERENCE = [
    "#3A74BF", "#E99A3B", "#7AC377", "#D9534F"
]
COLOR_PALETTE_MAIN_FIGURE = [
    "#E99A3B", "#3A74BF", "#8D9191", "#7AC377", "#386E6E",
    "#E99A3B", "#3A74BF", "#8D9191", "#7AC377", "#D9534F"
]

def export_figures(data: pd.DataFrame, output_name_prefix: str = None, variance: bool = False, color_style: str = None) -> None:
    """Export experiment results using seaborn.
    """
    # Settings
    title_fontsize = 26
    content_fontsize = 24
    legend_fontsize = 24
    label_spacing = 0.3
    alpha = 0.85

    algorithms = data["algorithm"].unique()

    if not variance:
        data_change = data.groupby(['algorithm', 'time_step']).mean(numeric_only=True).reset_index()
    else:
        data_change = data

    dashes2 = None
    if color_style == "user_bias":
        color_palette = COLOR_PALETTE_USER_BIAS
        colors = sns.color_palette(color_palette, n_colors=len(algorithms))
        colors = [(*color, alpha) for color in colors]
        dashes = [""] * len(algorithms)
        label_spacing = 0.2
        legend_fontsize = 23.25
    elif color_style == "preference":
        color_palette = COLOR_PALETTE_PREFERENCE
        colors = sns.color_palette(color_palette, n_colors=len(algorithms))
        colors = [(*color, alpha) for color in colors]
        dashes = [""] * len(algorithms)
    elif color_style == "main_figure":
        label_spacing = 0.2
        color_palette = COLOR_PALETTE_MAIN_FIGURE
        colors = sns.color_palette(color_palette, n_colors=len(algorithms))
        colors = [(*color, alpha) for color in colors]
        value_no_dash = ""
        value_ye_dash = (1.8, 0.8)
        dashes = [
            value_ye_dash, value_ye_dash, value_ye_dash, value_ye_dash, value_ye_dash,
            value_no_dash, value_no_dash, value_no_dash, value_no_dash, value_no_dash
        ]
        value_no_dash = ""
        value_ye_dash = (3, 2)
        dashes2 = [
            value_ye_dash, value_ye_dash, value_ye_dash, value_ye_dash, value_ye_dash,
            value_no_dash, value_no_dash, value_no_dash, value_no_dash, value_no_dash
        ]
    else:
        color_palette = COLOR_PALETTE_USER_BIAS
        colors = sns.color_palette(color_palette, n_colors=len(algorithms))
        dashes = [""] * len(algorithms)
        # raise ValueError("Unknown color style")

    color_mapping = dict(zip(algorithms, colors))
    dash_mapping = dict(zip(algorithms, dashes))

    if dashes2 is None:
        dash_mapping2 = dash_mapping
    else:
        dash_mapping2 = dict(zip(algorithms, dashes2))

    sns.set_theme(
        style='ticks',
        rc={'xtick.labelsize': content_fontsize, 'ytick.labelsize': content_fontsize}
    )

    def export_single_figure(data_plot, key_of_data, hue_order, label, name, colors, dashes = None, show_legend: bool = True, plot_type: str = 'line'):
        plt.figure(figsize=(9, 6))

        def plot_line_plot():
            ax_line = sns.lineplot(
                data=data_plot,
                x="time_step",
                y=key_of_data,
                hue="algorithm",
                hue_order=hue_order,
                palette=colors,
                style="algorithm",
                dashes=dashes,
                linewidth=4,
                legend=show_legend,
            )
            return ax_line

        def plot_scatter_plot():
            ax_scatter = sns.scatterplot(
                data=sample_data(data_plot, target_size=1000),
                x="time_step",
                y=key_of_data,
                hue="algorithm",
                hue_order=hue_order,
                palette=colors,
                style="algorithm",
                s=50,
                legend=show_legend,
            )
            return ax_scatter

        if plot_type == 'line':
            ax = plot_line_plot()
        elif plot_type == 'scatter':
            ax = plot_scatter_plot()
        else:
            raise ValueError('Unknown type')

        ax.set_xlabel('Rounds', fontsize=title_fontsize)
        ax.set_ylabel(label, fontsize=title_fontsize)

        lgd_setting = {
            "fontsize": legend_fontsize,
            "ncol": None,
            "handlelength": 0.7,
            "handletextpad": 0.2,
            "columnspacing": 0.4,
            "labelspacing": label_spacing,
            "borderaxespad": 0.0,
            "frameon": False,
        }

        if show_legend and len(hue_order) < 5:
            lgd_setting["ncol"] = 1
            ax.legend(**lgd_setting)
        elif show_legend:
            n_col = 2
            n_entries = len(hue_order)
            n_rows = (n_entries // n_col + 1)

            lgd_setting["ncol"] = n_col
            ax.legend(**lgd_setting)

            if n_entries % 2 != 0:
                legend = ax.legend_
                legend_handles = legend.legend_handles
                legend_labels = [text.get_text() for text in legend.texts]
                legend_handles.insert(n_rows, plt.Line2D([], [], visible=False))
                legend_labels.insert(n_rows, "")

                ax.legend(handles=legend_handles, labels=legend_labels, **lgd_setting)

        # Modify legend
        # if show_legend:
        #     if color_style == 'main_figure':
        #         legend = ax.legend_
        #         legend_handles = legend.legend_handles
        #         legend_labels = [text.get_text() for text in legend.texts]
        #         for idx, label in enumerate(legend_labels):
        #             if idx >= 5:
        #                 legend_labels[idx] = label + "(*)"
        #         ax.legend(handles=legend_handles, labels=legend_labels, **lgd_setting)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(4)

        file_name = Path(f"{output_name_prefix}-{name}.pdf")
        file_name.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(file_name, dpi=600, transparent=True, bbox_inches='tight', pad_inches=0)
        # plt.show()
        plt.close()

    # Weak Regret
    export_single_figure(
        data_plot=data_change,
        key_of_data='Cumulative Weak Regret',
        hue_order=algorithms,
        label='Cumulative Weak Regret',
        name='weak-regret',
        colors=color_mapping,
        dashes=dash_mapping,
        show_legend=True,
        plot_type='line'
    )

    # Percentage of Best Arm
    export_single_figure(
        data_plot=data_change,
        key_of_data='Rate of Rounds with Best Arm',
        hue_order=algorithms,
        label='Rounds with Best Arm',
        name='best-arm',
        colors=color_mapping,
        dashes=dash_mapping2,
        show_legend=False,
        plot_type='line'
    )

    # Average Regret
    export_single_figure(
        data_plot=data_change,
        key_of_data='Cumulative Average Regret',
        hue_order=algorithms,
        label='Cumulative Average Regret',
        name='average-regret',
        colors=color_mapping,
        dashes=dash_mapping,
        show_legend=True,
        plot_type='line'
    )

    # Error of evaluator bias
    error_bias_label = 'Error of Evaluator Bias'

    if error_bias_label in data_change.columns:
        valid_algorithms = [
            alg for alg in algorithms
            if not data_change[data_change['algorithm'] == alg][error_bias_label].isna().all()
        ]
    else:
        valid_algorithms = []

    if valid_algorithms:
        valid_data = data_change[data_change['algorithm'].isin(valid_algorithms)]
        export_single_figure(
            data_plot=valid_data,
            key_of_data=error_bias_label,
            hue_order=valid_algorithms,
            label='Error of Evaluator Bias',
            name='evaluator-bias',
            show_legend=True,
            colors={alg: color_mapping[alg] for alg in valid_algorithms},
            dashes={alg: dash_mapping[alg] for alg in valid_algorithms},
            plot_type='line'
        )

def extract_number(col_name):
    match = re.search(r'\d+', col_name)
    return int(match.group()) if match else 0

def export_evaluator_bias(data: pd.DataFrame, extra_header: str = None, output_name_prefix: str = None):
    if output_name_prefix is not None:
        if output_name_prefix.endswith('.csv'):
            output_name_prefix = output_name_prefix[:-4] + '-'
        elif not output_name_prefix.endswith('-'):
            output_name_prefix = output_name_prefix + '-'
    else:
        output_name_prefix = ""

    header = ['algorithm', extra_header] if extra_header is not None else ['algorithm']
    last_steps = data.groupby(header + ['run_id']).apply(
        lambda group: group.nlargest(1, 'time_step').drop(columns=['time_step'])
    ).reset_index(drop=True)

    if extra_header is not None:
        last_steps[f'algorithm-{extra_header}'] = last_steps.apply(
            lambda row: f"{row['algorithm']}-{row[extra_header]}",
            axis=1
        )
        header = [f'algorithm-{extra_header}']
    
    last_steps.groupby(header)
    last_steps.reset_index()

    def export_bias_to_csv(bias_type, bias_name = None):
        """Export bias data to CSV files.

        Parameters
        ----------
        bias_type : str
            The type of bias to export, either 'bias' or 'comparative_bias'.
        bias_name : str, optional
            The name of the bias to export, used for the file name.
        """
        if bias_name is None:
            bias_name = bias_type

        bias_data = {}
        index_col = f'algorithm-{extra_header}' if extra_header is not None else 'algorithm'
        for index_value in last_steps[index_col].unique():
            algorithm_data = last_steps[last_steps[index_col] == index_value]

            for _, row in algorithm_data.iterrows():
                bias_array = transform_string_to_array(row[bias_type])
                if bias_array is None or len(bias_array) == 0 or all(x is None for x in bias_array):
                    continue
                if index_value not in bias_data:
                    bias_data[index_value] = []
                bias_data[index_value].append(bias_array)

        # Calculate average bias
        avg_bias_data = {}
        for algorithm, bias_arrays in bias_data.items():
            if bias_arrays:
                avg_bias = np.mean(bias_arrays, axis=0)
                avg_bias_data[algorithm] = avg_bias

        # CSV export
        if avg_bias_data:
            max_evaluators = max(len(bias) for bias in avg_bias_data.values())
            columns = [f'evaluator_{i}' for i in range(max_evaluators)]
            df = pd.DataFrame(index=list(avg_bias_data.keys()), columns=columns)

            for algorithm, bias in avg_bias_data.items():
                for i, value in enumerate(bias):
                    df.loc[algorithm, f'evaluator_{i}'] = value

            file_name = Path(f"{output_name_prefix}{bias_name}.csv")
            file_name.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(file_name)
            print(f"Exported {bias_type} data to {file_name}")

    # Export estimated bias and exact bias to CSV
    export_bias_to_csv('bias', bias_name='estimated_bias')
    export_bias_to_csv('comparative_bias', bias_name='exact_bias')

def export_time_table(data: pd.DataFrame, column_name: str, output_name_prefix: str = None, aggregation: str = "mean") -> None:
    def export_single_table_all_rounds(values_label, name, agg_func='mean'):
        all_steps = data.groupby(['algorithm', column_name]).mean(numeric_only=True).reset_index()
        pivot_data = all_steps.pivot_table(
            index='algorithm',
            columns=column_name,
            values=values_label,
            aggfunc=agg_func,
        )
        pivot_data = pivot_data[sorted(pivot_data.columns, key=extract_number)]
        file_name = Path(f"{output_name_prefix}-{name}.xlsx")
        file_name.parent.mkdir(parents=True, exist_ok=True)
        pivot_data.to_excel(file_name)

    if aggregation == "mean":
        export_single_table_all_rounds(
            values_label='Time of Rounds', name='average-run-time', agg_func='mean'
        )
    elif aggregation == "sum":
        export_single_table_all_rounds(
            values_label='Time of Rounds', name='total-run-time', agg_func='sum'
        )

def export_table(data: pd.DataFrame, column_name: str = None, output_name_prefix: str = None) -> None:
    # Define headers based on column_name
    if column_name is None:
        headers = ['algorithm']
    else:
        headers = ['algorithm', column_name]

    # First, get the last 100 steps for each run
    last_100_steps = data.groupby([*headers, 'run_id']).apply(
        lambda group: group.nlargest(100, 'time_step')
    ).reset_index(drop=True)

    # Calculate mean for last 100 steps within each run
    run_means = last_100_steps.groupby([*headers, 'run_id']).mean(numeric_only=True).reset_index()

    def export_single_table_metrics(values_label, name, data_to_process):
        # Calculate overall mean across runs
        pivot_mean = data_to_process.pivot_table(
            index='algorithm',
            columns=column_name,
            values=values_label,
            aggfunc='mean'
        )
        pivot_mean = pivot_mean[sorted(pivot_mean.columns, key=extract_number)]

        # Calculate standard deviation of run means
        pivot_std = data_to_process.pivot_table(
            index='algorithm',
            columns=column_name,
            values=values_label,
            aggfunc=lambda x: x.std() / np.sqrt(len(x))
        )
        pivot_std = pivot_std[sorted(pivot_std.columns, key=extract_number)]

        # Export results to Excel
        file_name = Path(f"{output_name_prefix}-{name}.xlsx")
        file_name.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(file_name) as writer:
            pivot_mean.to_excel(writer, sheet_name='Mean')
            pivot_std.to_excel(writer, sheet_name='SE')

    def export_metrics_for_valid_algorithms(values_label, name):
        # Filter algorithms with valid data
        valid_algorithms = [
            alg for alg in data['algorithm'].unique()
            if not run_means[run_means['algorithm'] == alg][values_label].isna().all()
        ]
        if valid_algorithms:
            valid_data = run_means[run_means['algorithm'].isin(valid_algorithms)]
            export_single_table_metrics(values_label, name, valid_data)

    # Export tables for each metric
    metrics = {
        'Cumulative Weak Regret': 'weak-regret',
        'Rate of Rounds with Best Arm': 'best-arm',
        'Cumulative Average Regret': 'average-regret'
    }

    # Process standard metrics
    for metric, file_name in metrics.items():
        if metric in run_means.columns:
            export_single_table_metrics(metric, file_name, run_means)

    # Process evaluator bias separately
    if 'Error of Evaluator Bias' in run_means.columns:
        export_metrics_for_valid_algorithms('Error of Evaluator Bias', 'error-evaluator-bias')

def export_table_checkpoints(data: pd.DataFrame, extra_header: str = None, label: str = None, output_name_prefix: str = None) -> None:
    if output_name_prefix is not None:
        if output_name_prefix.endswith('.xlsx'):
            output_name_prefix = output_name_prefix[:-4] + '-'
        elif not output_name_prefix.endswith('-'):
            output_name_prefix = output_name_prefix + '-'
    else:
        output_name_prefix = ""

    if label is None or label == 'bias_error':
        metric_label = 'Error of Evaluator Bias'

    # Verify that the label exists in the data
    if metric_label not in data.columns:
        print(f"Warning: {metric_label} not found in data columns")
        return

    header = ['algorithm', extra_header] if extra_header is not None else ['algorithm']
    # Remove None from header if extra_header is None
    header = [h for h in header if h is not None]
    grouped_data = data.groupby(header + ['run_id'])

    # Define checkpoints of time_step
    checkpoints = [
        0, 10, 100, 1000, 5000, 10000, 20000, 40000
    ]
    
    # Process data for each checkpoint
    checkpoint_data = {}
    valid_data_found = False
    
    for checkpoint in checkpoints:
        checkpoint_values = []
        for name, group in grouped_data:
            if checkpoint == -1:
                # For the last checkpoint, get the last time step
                if not group[metric_label].isna().all():
                    last_value = group.nlargest(1, 'time_step')[metric_label].values[0]
                    checkpoint_values.append((name, last_value))
            else:
                # For other checkpoints, get the closest time step to the checkpoint
                valid_data = group[group['time_step'] <= checkpoint]
                if not valid_data.empty and not valid_data[metric_label].isna().all():
                    closest_value = valid_data.nlargest(1, 'time_step')[metric_label].values[0]
                    checkpoint_values.append((name, closest_value))
        
        if checkpoint_values:
            valid_data_found = True
            # Convert to series with proper index
            if len(header) > 1:
                # For multi-index case, ensure the values are tuples
                index_values = [x[0] if isinstance(x[0], tuple) else tuple([x[0]]) for x in checkpoint_values]
                index = pd.MultiIndex.from_tuples(index_values, names=header)
            else:
                # For single index case, extract single values
                index_values = [x[0][0] if isinstance(x[0], tuple) else x[0] for x in checkpoint_values]
                index = pd.Index(index_values, name=header[0])
            checkpoint_data[checkpoint] = pd.Series([x[1] for x in checkpoint_values], index=index)
    
    if not valid_data_found:
        print(f"No valid data found for {metric_label} at any checkpoint")
        return

    # Convert to DataFrame
    result_df = pd.DataFrame(checkpoint_data)
    
    # Rename columns to be more descriptive
    column_mapping = {checkpoint: f'Round {checkpoint}' for checkpoint in checkpoints}
    result_df = result_df.rename(columns=column_mapping)
    
    # Calculate mean across runs
    if extra_header is not None:
        if isinstance(result_df.index, pd.MultiIndex):
            result_df = result_df.reset_index()
        
        result_df['combined_index'] = result_df.apply(
            lambda row: f"{row['algorithm']}-{row[extra_header]}",
            axis=1
        )

        result_df = result_df.drop(
            ['algorithm', extra_header, 'run_id'],
            axis=1
        ).groupby('combined_index').mean(numeric_only=True)
    else:
        result_df = result_df.groupby('algorithm').mean(numeric_only=True)

    file_name = Path(f"{output_name_prefix}{label.lower().replace(' ', '-')}-checkpoints.xlsx")
    file_name.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_excel(file_name)
    print(f"Exported {metric_label} checkpoints data to {file_name}")


def plot_results(data: pd.DataFrame, title: str = None, variance: bool = False) -> None:
    """Plot experiment results using seaborn.

    Parameters
    ----------
    data
        A pandas dataframe with columns time_step, Cumulative Average Regret and
        wall_clock.
    variance
        whether plot variance (take a long time) or not.
    """
    algorithms = data["algorithm"].unique()

    if not variance:
        data_change = data.groupby(['algorithm', 'time_step']).mean(numeric_only=True).reset_index()
    else:
        data_change = data

    # Explicitly set color and dash mappings to ensure that they are consistent
    # between the different plots. We could also rely on seaborn's
    # deterministic behavior when generating the mappings, but I think it is
    # better to specify it explicitly.

    # colors = sns.color_palette(n_colors=len(algorithms) // 2)
    # colors = [color for color in colors for _ in range(2)]

    colors = sns.color_palette(n_colors=len(algorithms))
    color_mapping = dict(zip(algorithms, colors))

    # dashes = ["", (5, 3)] * len(algorithms)
    # dash_mapping = dict(zip(algorithms, dashes))
    dashes = [""] * len(algorithms)
    dash_mapping = dict(zip(algorithms, dashes))

    _fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    _fig.suptitle(title, fontsize=16)
    (ax1, ax2, ax3, ax4) = axes.flat
    sns.lineplot(
        data=data_change,
        x="time_step",
        y="Cumulative Weak Regret",
        hue="algorithm",
        hue_order=algorithms,
        palette=color_mapping,
        style="algorithm",
        dashes=dash_mapping,
        linewidth=2,
        ax=ax1,
        legend=False,
    )
    sns.lineplot(
        data=data_change,
        x="time_step",
        y="Rate of Rounds with Best Arm",
        hue="algorithm",
        hue_order=algorithms,
        palette=color_mapping,
        style="algorithm",
        dashes=dash_mapping,
        linewidth=2,
        ax=ax2,
        legend=False,
    )
    sns.lineplot(
        data=data_change,
        x="time_step",
        y="Cumulative Average Regret",
        hue="algorithm",
        hue_order=algorithms,
        palette=color_mapping,
        style="algorithm",
        dashes=dash_mapping,
        linewidth=2,
        ax=ax3,
        legend=True,
    )
    sns.lineplot(
        data=data_change,
        x="time_step",
        y="Best Arm Rate",
        hue="algorithm",
        hue_order=algorithms,
        palette=color_mapping,
        style="algorithm",
        dashes=dash_mapping,
        linewidth=2,
        ax=ax4,
        legend=False,
    )
    plt.show()

    # ==== The second graph
    plot_configs = [
        {
            'column': 'Time of Rounds',
            'title': 'Time of Rounds'
        },
        {
            'column': 'Error of Evaluator Bias',
            'title': 'Error of Evaluator Bias'
        },
        {
            'column': 'BCD Residual',
            'title': 'BCD Residual'
        }
    ]

    valid_plots = [
        config for config in plot_configs
        if config['column'] in data_change.columns
    ]
    
    num_plots = len(valid_plots)
    
    if num_plots <= 2:
        n_rows, n_cols = 1, num_plots
    else:  # num_plots == 3
        n_rows, n_cols = 2, 2
        
    _fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(6*n_cols, 6*n_rows))
    _fig.suptitle(title, fontsize=16)
    
    # Arrange axes
    if num_plots == 1:
        axes = [axes]
    elif num_plots == 3:
        axes_list = [axes[0,0], axes[1,0], axes[1,1]]
        axes_list[0].set_position([0.2, 0.55, 0.6, 0.35])  
        axes_list[1].set_position([0.1, 0.1, 0.35, 0.35])
        axes_list[2].set_position([0.55, 0.1, 0.35, 0.35])
        axes[0,1].set_visible(False)
        axes = axes_list
    
    # Plot
    for idx, (ax, plot_config) in enumerate(zip(axes, valid_plots)):
        column = plot_config['column']
        
        if column in ['Error of Evaluator Bias', 'BCD Residual']:
            valid_algorithms = [
                alg for alg in algorithms
                if not data_change[data_change['algorithm'] == alg][column].isna().all()
            ]
            
            if valid_algorithms:
                valid_data = data_change[data_change['algorithm'].isin(valid_algorithms)]
                sampled_valid_data = sample_data(valid_data, target_size=1000)
                sns.lineplot(
                    data=sampled_valid_data,
                    x="time_step",
                    y=column,
                    hue="algorithm",
                    hue_order=valid_algorithms,
                    palette={alg: color_mapping[alg] for alg in valid_algorithms},
                    style="algorithm",
                    dashes={alg: dash_mapping[alg] for alg in valid_algorithms},
                    linewidth=2,
                    ax=ax
                )
        else:
            sampled_data_change = sample_data(data_change, target_size=1000)
            sns.lineplot(
                data=sampled_data_change,
                x="time_step",
                y=column,
                hue="algorithm",
                hue_order=algorithms,
                palette=color_mapping,
                style="algorithm",
                dashes=dash_mapping,
                linewidth=2,
                ax=ax
            )
        
        ax.set_title(plot_config['title'])

    plt.show()


def sample_data(data: pd.DataFrame, target_size: int = 1000, move_average: bool = True) -> pd.DataFrame:
    """Sample data points to reduce to target size while preserving characteristics.

    Parameters
    ----------
    data : pd.DataFrame including column "algorithm"
        Input dataframe to be sampled
    target_size : int, optional
        Target number of samples per algorithm, by default 1000
    move_average : bool, optional
        If True, use moving average of each step window instead of single sample

    Returns
    -------
    pd.DataFrame
        Sampled dataframe
    """
    algorithms = data["algorithm"].unique()
    sampled_data = []

    for alg in algorithms:
        alg_data = data[data['algorithm'] == alg].copy()
        alg_data = alg_data.reset_index(drop=True)
        
        if len(alg_data) > target_size:
            step = len(alg_data) // target_size
            if move_average:
                sampled_rows = []
                for i in range(0, min(len(alg_data) - step + 1, target_size * step), step):
                    window = alg_data.iloc[i:i + step]
                    mean_row = window.select_dtypes(include=[np.number]).mean()
                    non_numeric = window.select_dtypes(exclude=[np.number]).iloc[0]
                    sampled_rows.append(pd.concat([mean_row, non_numeric]))
                alg_data = pd.DataFrame(sampled_rows).reset_index(drop=True)
            else:
                sampled_indices = range(0, len(alg_data), step)[:target_size]
                alg_data = alg_data.iloc[sampled_indices].reset_index(drop=True)
        sampled_data.append(alg_data)

    return pd.concat(sampled_data, ignore_index=True)

def transform_array_to_string(array: Union[np.ndarray, list[float]]) -> str:
    return ",".join([str(item) for item in array])

def transform_string_to_array(string: str) -> np.ndarray:
    return np.array([None if item.strip() == 'None' else float(item) for item in string.split(',')])
