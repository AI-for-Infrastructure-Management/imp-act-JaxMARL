import os

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

# Configuration
DATA_DIR = "evaluation/plotting/data"
FIGURES_DIR = "evaluation/plotting/figures"
HEURISTIC_RESULTS_FILE = os.path.join(DATA_DIR, "best_heuristic_results.yaml")
EVALUATION_RESULTS_FILE = os.path.join(DATA_DIR, "combined_evaluation_results.csv")


ALGORITHM_NAMES = {
    "vdn_rnn": "VDN",
    "qmix_rnn": "QMIX",
    "pqn_rnn": "PQN-VDN",
    "ippo_rnn": "IPPO",
    "mappo_rnn": "MAPPO",
    "vdn_ba_rnn": "VDN-BA",
}

ENV_PLOT_TITLES = {
    "ToyExample-v2": "ToyExample\n(12 agents)",
    "Cologne-v1": "Cologne\n(60 agents)",
    "CologneBonnDusseldorf-v1": "CologneBonnDusseldorf\n(178 agents)",
    "ToyExample-v2-unconstrained": "ToyExample Unconstrained\n(12 agents)",
    "Cologne-v1-unconstrained": "Cologne Unconstrained\n(60 agents)",
    "CologneBonnDusseldorf-v1-unconstrained": "CologneBonnDusseldorf Unconstrained\n(178 agents)",
    "Cologne-v1-critical-budget": "Cologne Critical Budget\n(60 agents)",
    "Cologne-v1-moderate-budget": "Cologne Moderate Budget\n(60 agents)",
}

PLOT_PARAMS = {
    "font.size": 8,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "text.usetex": False,
}


def load_data():
    """Loads heuristic results and evaluation data."""
    with open(HEURISTIC_RESULTS_FILE) as f:
        heuristic_results = yaml.load(f, Loader=yaml.FullLoader)

    baseline_heuristic_values = {
        env: results["segment_based_heuristic"]["mean_reward"]
        for env, results in heuristic_results.items()
    }

    data = pd.read_csv(EVALUATION_RESULTS_FILE)
    return data, baseline_heuristic_values


def get_best_checkpoints(data, env_name, baseline_heuristic_values):
    """
    Extracts best checkpoints for each algorithm and normalizes rewards.
    """
    algorithms = data["algorithm"].unique()
    heuristic_score = baseline_heuristic_values[env_name]

    best_checkpoint_per_algorithm_norm = {}

    for alg in algorithms:
        alg_data = data[data["algorithm"] == alg]
        runs = alg_data["WANDB_RUN_ID"].unique()

        best_runs = {}
        for run in runs:
            run_max = alg_data[alg_data["WANDB_RUN_ID"] == run]["mean"].max()
            # Normalize
            best_runs[run] = (run_max - heuristic_score) / abs(heuristic_score)

        best_checkpoint_per_algorithm_norm[alg] = best_runs

    return best_checkpoint_per_algorithm_norm


def prepare_plot_data(best_checkpoints, algorithms_to_plot):
    """Converts dictionary of best checkpoints to DataFrame for plotting."""
    plot_data_norm = []
    for alg, runs in best_checkpoints.items():
        if alg not in algorithms_to_plot:
            continue
        for value in runs.values():
            if pd.notna(value) and np.isfinite(value):
                plot_data_norm.append(
                    {"Algorithm": alg, "Performance Improvement (%)": value * 100}
                )
    df = pd.DataFrame(plot_data_norm)
    if df.empty and "Algorithm" not in df.columns:
        return pd.DataFrame(columns=["Algorithm", "Performance Improvement (%)"])
    return df


def custom_y_labels(y, pos, y_min):
    if y == 0:
        return "0%"
    if y == y_min:
        return f"≤ {y:.0f}%"
    return f"{y:+.1f}%"


def add_jittered_scatter(
    ax, plot_df, algorithms_to_plot, y_min, y_max, show_percentages=True
):
    """Adds jittered scatter points and annotations for best/worst values."""
    out_of_range_points = {alg: [] for alg in algorithms_to_plot}

    for j, alg in enumerate(algorithms_to_plot):
        alg_data = plot_df[plot_df["Algorithm"] == alg]["Performance Improvement (%)"]
        if len(alg_data) == 0:
            continue

        y_vals = alg_data.values
        # Sort for better jitter calculation
        sorted_indices = np.argsort(y_vals)
        y_vals = y_vals[sorted_indices]

        x_jittered = np.zeros(len(y_vals))
        in_range_mask = np.ones(len(y_vals), dtype=bool)

        for k, y_val in enumerate(y_vals):
            # Simple jitter logic based on density
            indices = list(range(len(y_vals)))
            indices.remove(k)
            if len(indices) > 0:
                dist = abs(y_val - y_vals[indices])
                close_points = (dist < 1).sum()
            else:
                close_points = 0

            x_jittered[k] = j
            if close_points > 0:
                # Deterministic jitter based on value and index
                jitter_amount = (
                    np.sin(
                        3 * np.pi * (y_val + k * np.sqrt(2) + np.random.normal(0, 1))
                    )
                    * 0.3
                )
                x_jittered[k] = j + jitter_amount

            if y_val < y_min:
                out_of_range_points[alg].append((x_jittered[k], y_val))
                in_range_mask[k] = False

        # Plot in-range points
        ax.scatter(
            x_jittered[in_range_mask],
            y_vals[in_range_mask],
            color="black",
            edgecolors="none",
            alpha=0.6,
            s=10,
        )

        if show_percentages:
            annotate_best_worst(ax, alg_data, j, y_min, y_max)

    return out_of_range_points


def annotate_best_worst(ax, alg_data, x_pos, y_min, y_max):
    """Annotates the best and worst values for an algorithm."""
    best_val = alg_data.max()
    worst_val = alg_data.min()

    best_text = f"{best_val:+.1f}%"
    worst_text = f"{worst_val:+.1f}%"

    # Best value annotation
    if best_val <= y_max:
        ax.annotate(
            best_text,
            xy=(x_pos, best_val),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6,
            fontweight="bold",
            color="green",
        )
    else:
        ax.text(
            x_pos,
            y_max - 5,
            f"↑\n{best_text}",
            ha="center",
            va="center",
            fontsize=6,
            fontweight="bold",
            color="green",
        )

    # Worst value annotation
    if worst_val >= y_min:
        ax.annotate(
            worst_text,
            xy=(x_pos, worst_val),
            xytext=(0, -5),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=6,
            fontweight="bold",
            color="red",
        )
    else:
        ax.text(
            x_pos,
            y_min + 5,
            f"{worst_text}\n↓",
            ha="center",
            va="center",
            fontsize=6,
            fontweight="bold",
            color="red",
        )


def plot_baseline_line(ax, baseline_value):
    """Adds the baseline heuristic line and label."""
    ax.axhline(
        y=0,
        color="red",
        linestyle="-",
        linewidth=1,
        label="Heuristic $\\text{H}_\\text{PS}$",
    )

    if abs(baseline_value) >= 1e9:
        baseline_text = f"$\\text{{H}}_\\text{{PS}}$ = {baseline_value/1e9:.1f}B"
    else:
        baseline_text = f"$\\text{{H}}_\\text{{PS}}$ = {baseline_value/1e6:.1f}M"

    # Plot in top left corner
    ax.text(
        0.02,
        0.98,
        baseline_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="red",
        fontsize=7,
        fontweight="bold",
    )


def boxplot_legend_entries(axes, full_legend=False):
    """Collects legend entries from the panels and removes any per-panel legends.

    With full_legend, also adds entries explaining what the box, median and whiskers show.
    """
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        if ax.get_legend():
            ax.get_legend().remove()
    if full_legend:
        handles += [
            mpatches.Patch(facecolor="white", edgecolor="0.25", linewidth=1),
            mlines.Line2D([], [], color="0.25", linewidth=1),
            mlines.Line2D([], [], color="0.25", linewidth=1, marker="|", markersize=5),
        ]
        labels += ["Box: 25th-75th percentile", "Line: median", "Whiskers: min / max"]
    return handles, labels


def add_boxplot_legend(axes, legend_ax, loc, full_legend=False):
    """Adds a single legend to legend_ax."""
    handles, labels = boxplot_legend_entries(axes, full_legend=full_legend)
    legend_ax.legend(handles, labels, loc=loc, fontsize=7)


def create_plots(
    best_checkpoints_all,
    baseline_heuristic_values,
    envs_to_plot,
    algorithms_to_plot,
    filename,
    title_dict=None,
    show_percentages=True,
    show_scatter=True,
    show_baseline=True,
    random_seed=45081,
    figsize=(5.5, 3.8),
    y_lims=(-55, 30),
    x_margin=0.75,
    env_colors=None,
    full_legend=False,
):
    """Main plotting function."""
    plt.rcParams.update(PLOT_PARAMS)
    np.random.seed(random_seed)

    num_envs = len(envs_to_plot)
    fig, axes = plt.subplots(1, num_envs, figsize=figsize, sharey=True)

    if num_envs == 1:
        axes = [axes]

    title_dict = title_dict if title_dict else ENV_PLOT_TITLES

    for i, (ax, env_name) in enumerate(zip(axes, envs_to_plot, strict=True)):
        title = title_dict.get(env_name)
        norm_data = best_checkpoints_all[env_name]
        # Only algorithms with runs on this map (e.g. no VDN-BA without a budget)
        panel_algorithms = [alg for alg in algorithms_to_plot if alg in norm_data]
        plot_df = prepare_plot_data(norm_data, panel_algorithms)

        # Boxplot
        boxplot_kwargs = {
            "x": "Algorithm",
            "y": "Performance Improvement (%)",
            "data": plot_df,
            "ax": ax,
            "whis": (0, 100),
            "showfliers": True,
            "flierprops": {"marker": "o", "markersize": 3, "alpha": 0.6},
            "boxprops": {"linewidth": 1},
            "medianprops": {"linewidth": 1},
            "whiskerprops": {"linewidth": 1},
            "capprops": {"linewidth": 1},
            "order": panel_algorithms,
        }

        if env_colors and env_name in env_colors:
            boxplot_kwargs["color"] = env_colors[env_name]
        else:
            boxplot_kwargs["palette"] = "viridis"

        sns.boxplot(**boxplot_kwargs)

        ax.grid(True, axis="both", alpha=0.3)
        ax.set_axisbelow(True)

        # X-axis formatting
        ax.set_xticks(range(len(panel_algorithms)))
        ax.set_xticklabels([ALGORITHM_NAMES[alg] for alg in panel_algorithms])
        ax.tick_params(axis="x", rotation=45)
        ax.set_xlim(-x_margin, len(panel_algorithms) - 1 + x_margin)

        # Y-axis formatting
        ax.set_ylim(y_lims)
        y_min, y_max = ax.get_ylim()

        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(
                lambda y, pos, y_min=y_min: custom_y_labels(y, pos, y_min)
            )
        )

        # Add scatter points and annotations
        out_of_range_points = {}
        if show_scatter:
            out_of_range_points = add_jittered_scatter(
                ax,
                plot_df,
                panel_algorithms,
                y_min,
                y_max,
                show_percentages=show_percentages,
            )

        # Add baseline
        if show_baseline:
            plot_baseline_line(ax, baseline_heuristic_values[env_name])

        # Handle out-of-range points visualization
        if i == 0:
            yticks = list(ax.get_yticks())
            if y_min not in yticks:
                yticks.append(y_min)
            ax.set_yticks(yticks)

        for points in out_of_range_points.values():
            for x_pos, _ in points:
                ax.scatter(
                    x_pos,
                    y_min,
                    color="red",
                    alpha=0.7,
                    s=15,
                    marker="v",
                    edgecolors="none",
                )

        # Titles and labels
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("", fontsize=9)
        if i == 0:
            ax.set_ylabel("Relative Performance (%)", fontsize=8)
        else:
            ax.set_ylabel("")

    add_boxplot_legend(axes, axes[-1], "lower right", full_legend=full_legend)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.0)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, f"{filename}.pdf"), bbox_inches="tight")
    plt.savefig(
        os.path.join(FIGURES_DIR, f"{filename}.png"), bbox_inches="tight", dpi=300
    )
    print(f"Figures saved to {FIGURES_DIR}")


def create_plots_combined(
    best_checkpoints_all,
    base_envs,
    algorithms_to_plot,
    filename,
    title_dict=None,
    random_seed=45081,
    figsize=(5.5, 3.8),
    y_lims=(-55, 30),
    x_margin=0.75,
    show_baseline=True,
    baseline_heuristic_values=None,
):
    """Plots combined constrained and unconstrained results."""
    plt.rcParams.update(PLOT_PARAMS)
    np.random.seed(random_seed)

    num_envs = len(base_envs)
    fig, axes = plt.subplots(1, num_envs, figsize=figsize, sharey=True)

    if num_envs == 1:
        axes = [axes]

    for i, (ax, base_env) in enumerate(zip(axes, base_envs, strict=True)):
        title = (
            title_dict.get(base_env, ENV_PLOT_TITLES.get(base_env, base_env))
            if title_dict
            else ENV_PLOT_TITLES.get(base_env, base_env)
        )
        # Prepare data
        norm_data_constrained = best_checkpoints_all[base_env]
        df_constrained = prepare_plot_data(norm_data_constrained, algorithms_to_plot)
        df_constrained["Scenario"] = "Constrained"
        df_constrained["PlotOrder"] = df_constrained["Algorithm"]

        unconstrained_env = f"{base_env}-unconstrained"
        norm_data_unconstrained = best_checkpoints_all[unconstrained_env]
        df_unconstrained = prepare_plot_data(
            norm_data_unconstrained, algorithms_to_plot
        )
        df_unconstrained["Scenario"] = "Unconstrained"
        df_unconstrained["PlotOrder"] = df_unconstrained["Algorithm"] + "_unconstrained"

        plot_df = pd.concat([df_constrained, df_unconstrained])

        # Define order: Constrained first, then Unconstrained
        # Filter out algorithms that don't have data in the respective scenario
        available_orders = set(plot_df["PlotOrder"].unique())
        full_order = algorithms_to_plot + [
            alg + "_unconstrained" for alg in algorithms_to_plot
        ]
        order = [o for o in full_order if o in available_orders]

        num_constrained_present = sum(
            1 for o in order if not o.endswith("_unconstrained")
        )

        # Boxplot with hue='Scenario' to color them, but x='PlotOrder' to position them
        sns.boxplot(
            x="PlotOrder",
            y="Performance Improvement (%)",
            hue="Scenario",
            data=plot_df,
            palette="viridis",
            ax=ax,
            whis=(0, 100),
            showfliers=True,
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.6},
            boxprops={"linewidth": 1},
            medianprops={"linewidth": 1},
            whiskerprops={"linewidth": 1},
            capprops={"linewidth": 1},
            order=order,
            width=0.6,
            dodge=False,
        )

        ax.grid(True, axis="both", alpha=0.3)
        ax.set_axisbelow(True)

        # X-axis formatting
        labels = []
        for o in order:
            if o.endswith("_unconstrained"):
                alg_key = o.replace("_unconstrained", "")
                labels.append(ALGORITHM_NAMES.get(alg_key, alg_key))
            else:
                labels.append(ALGORITHM_NAMES.get(o, o))

        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labels)
        ax.tick_params(axis="x", rotation=90)
        ax.set_xlim(-x_margin, len(order) - 1 + x_margin)

        # Vertical separator
        if num_constrained_present > 0 and num_constrained_present < len(order):
            ax.axvline(
                x=num_constrained_present - 0.5,
                color="black",
                linestyle="--",
                linewidth=0.8,
                alpha=0.5,
            )

        # Y-axis formatting
        ax.set_ylim(y_lims)
        y_min, y_max = ax.get_ylim()
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(
                lambda y, pos, y_min=y_min: custom_y_labels(y, pos, y_min)
            )
        )

        # Baseline line (y=0)
        ax.axhline(
            y=0,
            color="red",
            linestyle="-",
            linewidth=1,
            label="Prioritized Heuristic $\\text{H}_\\text{PS}$",
        )

        if show_baseline and baseline_heuristic_values:
            # Constrained baseline (left)
            val_constrained = baseline_heuristic_values[base_env]
            if abs(val_constrained) >= 1e9:
                text_c = f"$\\text{{H}}_\\text{{PS}}$ = {val_constrained/1e9:.1f}B"
            else:
                text_c = f"$\\text{{H}}_\\text{{PS}}$ = {val_constrained/1e6:.1f}M"

            ax.text(
                0.02,
                0.98,
                text_c,
                transform=ax.transAxes,
                ha="left",
                va="top",
                color="red",
                fontsize=7,
                fontweight="bold",
            )

            # Unconstrained baseline (right)
            val_unconstrained = baseline_heuristic_values[unconstrained_env]
            if abs(val_unconstrained) >= 1e9:
                text_u = f"$\\text{{H}}_\\text{{PS}}$ = {val_unconstrained/1e9:.1f}B"
            else:
                text_u = f"$\\text{{H}}_\\text{{PS}}$ = {val_unconstrained/1e6:.1f}M"

            # Calculate position for unconstrained text
            x_min = -x_margin
            x_max = len(order) - 1 + x_margin
            span = x_max - x_min

            if num_constrained_present > 0:
                sep_x = num_constrained_present - 0.5
                x_rel = (sep_x - x_min) / span
                text_x_pos = x_rel + 0.02
            else:
                text_x_pos = 0.02

            ax.text(
                text_x_pos,
                0.98,
                text_u,
                transform=ax.transAxes,
                ha="left",
                va="top",
                color="red",
                fontsize=7,
                fontweight="bold",
            )

        # Titles and labels
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("", fontsize=9)

        if i == 0:
            ax.set_ylabel("Relative Performance (%)", fontsize=8)
        else:
            ax.set_ylabel("")
            if ax.get_legend():
                ax.legend_.remove()

    # Full legend below the panels, inside the figure so its size is unchanged
    handles, labels = boxplot_legend_entries(axes, full_legend=True)
    plt.tight_layout(rect=(0, 0.07, 1, 0.965))
    plt.subplots_adjust(wspace=0.0)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        fontsize=7,
        frameon=False,
        labelspacing=0.3,
        borderaxespad=0.0,
    )

    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, f"{filename}.pdf"), bbox_inches="tight")
    plt.savefig(
        os.path.join(FIGURES_DIR, f"{filename}.png"), bbox_inches="tight", dpi=300
    )
    print(f"Figures saved to {FIGURES_DIR}")


def main():
    data, baseline_heuristic_values = load_data()

    environments = data["map_name"].unique()
    best_checkpoints_all = {
        env: get_best_checkpoints(
            data[data["map_name"] == env], env, baseline_heuristic_values
        )
        for env in environments
    }

    envs_to_plot = ["ToyExample-v2", "Cologne-v1", "CologneBonnDusseldorf-v1"]
    algorithms_to_plot = [
        "vdn_rnn",
        "vdn_ba_rnn",
        "qmix_rnn",
        "pqn_rnn",
        "mappo_rnn",
        "ippo_rnn",
    ]

    create_plots(
        best_checkpoints_all,
        baseline_heuristic_values,
        envs_to_plot=envs_to_plot,
        algorithms_to_plot=algorithms_to_plot,
        filename="Extra_Figure_Envs",
    )

    envs_to_plot = [f"{env}-unconstrained" for env in envs_to_plot]

    create_plots(
        best_checkpoints_all,
        baseline_heuristic_values,
        envs_to_plot=envs_to_plot,
        algorithms_to_plot=algorithms_to_plot,
        filename="Extra_Figure_Envs_Unconstrained",
    )

    envs_to_plot = [
        f"Cologne-v1{env_suffix}"
        for env_suffix in ["-unconstrained", "-moderate-budget", "", "-critical-budget"]
    ]

    title_dict = {
        "Cologne-v1-unconstrained": "Cologne Unconstrained\n(Infinite budget)",
        "Cologne-v1-moderate-budget": "Cologne Moderate\n(300M budget)",
        "Cologne-v1": "Cologne Limited \n(200M budget)",
        "Cologne-v1-critical-budget": "Cologne Critical\n(150M budget)",
    }

    env_colors = {
        "Cologne-v1-unconstrained": "#414487",
        "Cologne-v1-moderate-budget": "#2a788e",
        "Cologne-v1": "#22a884",
        "Cologne-v1-critical-budget": "#7ad151",
    }

    create_plots(
        best_checkpoints_all,
        baseline_heuristic_values,
        envs_to_plot=envs_to_plot,
        algorithms_to_plot=algorithms_to_plot,
        filename="Figure_4_Cologne_Budgets",
        title_dict=title_dict,
        show_baseline=True,
        show_scatter=False,
        show_percentages=False,
        y_lims=(-40, 30),
        env_colors=env_colors,
        full_legend=True,
    )

    create_plots(
        best_checkpoints_all,
        baseline_heuristic_values,
        envs_to_plot=["ToyExample-v2", "Cologne-v1", "CologneBonnDusseldorf-v1"],
        algorithms_to_plot=["vdn_rnn", "vdn_ba_rnn"],
        filename="Extra_Figure_VDN_vs_VDN-BA",
        show_baseline=True,
        y_lims=(-5, 30),
    )

    create_plots_combined(
        best_checkpoints_all,
        base_envs=["ToyExample-v2", "Cologne-v1", "CologneBonnDusseldorf-v1"],
        algorithms_to_plot=[
            "vdn_rnn",
            "vdn_ba_rnn",
            "qmix_rnn",
            "pqn_rnn",
            "mappo_rnn",
            "ippo_rnn",
        ],
        filename="Figure_3_Envs_Constrained_vs_Unconstrained",
        y_lims=(-40, 30),
        show_baseline=True,
        baseline_heuristic_values=baseline_heuristic_values,
    )


if __name__ == "__main__":
    main()
