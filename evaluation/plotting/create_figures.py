import os

import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy.stats import trim_mean

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

def tt(map_name):
    """Map name in typewriter font, as \\texttt in the paper."""
    return rf"$\mathtt{{{map_name}}}$"


# Budgets of the constrained scenarios from the imp-act presets (budget_amount)
ENV_BUDGETS = {
    "ToyExample-v2": "6.5M",
    "Cologne-v1": "200M",
    "CologneBonnDusseldorf-v1": "800M",
}

ENV_PLOT_TITLES = {
    "ToyExample-v2": f"{tt('ToyExample')}\n(12 agents)",
    "Cologne-v1": f"{tt('Cologne')}\n(60 agents)",
    "CologneBonnDusseldorf-v1": f"{tt('CologneBonnDusseldorf')}\n(178 agents)",
    "ToyExample-v2-unconstrained": f"{tt('ToyExample')} Unconstrained\n(12 agents)",
    "Cologne-v1-unconstrained": f"{tt('Cologne')} Unconstrained\n(60 agents)",
    "CologneBonnDusseldorf-v1-unconstrained": f"{tt('CologneBonnDusseldorf')} Unconstrained\n(178 agents)",
    "Cologne-v1-critical-budget": f"{tt('Cologne')} Critical\n(60 agents, 150M budget)",
    "Cologne-v1-moderate-budget": f"{tt('Cologne')} Moderate\n(60 agents, 300M budget)",
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
    # Math in Times, and \mathtt in LaTeX's typewriter font (the default DejaVu Sans Mono is
    # wider and heavier than the surrounding Times text)
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "mathtext.tt": "CMU Typewriter Text",
}

# Y-axis label of the relative performance plots
RELATIVE_PERFORMANCE_LABEL = r"Normalized Expected Return ($\Delta \bar{R}^{\pi}_0$ %)"
ABSOLUTE_RETURN_LABEL = r"Expected Return ($\mathbb{E}\left[R_0^\pi\right]$)"


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
    title_fontsize=9,
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
        ax.set_title(title, fontsize=title_fontsize)
        ax.set_xlabel("", fontsize=9)
        if i == 0:
            ax.set_ylabel(RELATIVE_PERFORMANCE_LABEL, fontsize=8)
        else:
            ax.set_ylabel("")

    add_boxplot_legend(axes, axes[-1], "lower right", full_legend=full_legend)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.0)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, f"{filename}.pdf"), bbox_inches="tight")
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
            # Budget applies to the constrained (left) half only, below its H_PS
            ax.text(
                0.02,
                0.92,
                f"Budget = {ENV_BUDGETS[base_env]}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                color="0.25",
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
            ax.set_ylabel(RELATIVE_PERFORMANCE_LABEL, fontsize=8)
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
    print(f"Figures saved to {FIGURES_DIR}")


# Environment steps per update (NUM_ENVS * NUM_STEPS) of the final runs, from
# experiments/config/final_runs on the final-run-configs branch (50M timesteps for every run).
# PPO checkpoints are indexed from update 0, the Q-learning ones from update 1.
STEPS_PER_UPDATE = {
    "vdn_rnn": 32 * 51,
    "vdn_ba_rnn": 32 * 51,
    "qmix_rnn": 32 * 51,
    "pqn_rnn": 128 * 51,
    "ippo_rnn": 64 * 128,
    "mappo_rnn": 64 * 256,
}
UPDATE_INDEX_OFFSET = {"ippo_rnn": 1, "mappo_rnn": 1}
TOTAL_TIMESTEPS = 50_000_000


def training_curve_stats(env_data, alg):
    """IQM and interquartile range over seeds of the evaluated return per environment step."""
    alg_data = env_data[env_data["algorithm"] == alg].copy()
    alg_data["env_steps"] = (
        alg_data["checkpoint_step"] + UPDATE_INDEX_OFFSET.get(alg, 0)
    ) * STEPS_PER_UPDATE[alg]
    returns = alg_data.pivot(index="env_steps", columns="WANDB_RUN_ID", values="mean")
    iqm = returns.apply(lambda row: trim_mean(row.dropna(), 0.25), axis=1)
    q25 = returns.quantile(0.25, axis=1)
    q75 = returns.quantile(0.75, axis=1)
    return returns.index.values, iqm.values, q25.values, q75.values


def format_return(value, pos=None):
    """Formats an absolute return in billions (B) or millions (M)."""
    if abs(value) >= 1e9:
        return f"{value/1e9:.2f}B"
    return f"{value/1e6:.0f}M"


def create_training_curves(
    data,
    baseline_heuristic_values,
    env_grid,
    algorithms_to_plot,
    filename,
    title_dict=None,
    figsize=(7.0, 7.5),
    y_lims=(-40, 30),
    row_labels=None,
):
    """
    Training curves (IQM over seeds, IQR shaded) of the re-evaluated checkpoints against
    environment steps, one panel
    per environment. env_grid is a list of rows, each a list of environments (None leaves the
    panel empty). Returns are shown relative to H_PS (left axis, shared limits y_lims) with the
    corresponding absolute return on a linked right axis; H_PS is the dashed line at 0%.
    """
    plt.rcParams.update(PLOT_PARAMS)
    title_dict = title_dict if title_dict else {}

    n_rows, n_cols = len(env_grid), max(len(row) for row in env_grid)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    # Full viridis range (dark purple to yellow), as in the paper's training curves.
    # VDN-BA shares VDN's colour with a dash-dot line, so the other colours stay fixed.
    base_algorithms = [alg for alg in algorithms_to_plot if alg != "vdn_ba_rnn"]
    colors = dict(
        zip(
            base_algorithms,
            plt.cm.viridis(np.linspace(0, 1, len(base_algorithms))),
            strict=True,
        )
    )
    colors["vdn_ba_rnn"] = colors.get("vdn_rnn", "black")
    line_styles = {"vdn_ba_rnn": "-."}

    for i, row in enumerate(env_grid):
        for j in range(n_cols):
            ax = axes[i, j]
            env_name = row[j] if j < len(row) else None
            if env_name is None:
                ax.axis("off")
                continue

            heuristic_score = baseline_heuristic_values[env_name]

            def to_pct(r, h=heuristic_score):
                return (r - h) / abs(h) * 100

            def to_return(p, h=heuristic_score):
                return h + p / 100 * abs(h)

            env_data = data[data["map_name"] == env_name]
            for alg in algorithms_to_plot:
                if alg not in env_data["algorithm"].unique():
                    continue
                x, iqm, q25, q75 = training_curve_stats(env_data, alg)
                ax.plot(
                    x,
                    to_pct(iqm),
                    color=colors[alg],
                    linestyle=line_styles.get(alg, "-"),
                    linewidth=1.5,
                    label=f"{ALGORITHM_NAMES.get(alg, alg)} (IQM)",
                )
                ax.fill_between(
                    x,
                    to_pct(q25),
                    to_pct(q75),
                    color=colors[alg],
                    alpha=0.2,
                    linewidth=0,
                )

            ax.axhline(
                y=0,
                color="red",
                linestyle="--",
                linewidth=1,
                label="$\\text{H}_\\text{PS}$",
            )
            ax.text(
                0.02,
                0.97,
                f"$\\text{{H}}_\\text{{PS}}$ = {format_return(heuristic_score)}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                color="red",
                fontsize=7,
                fontweight="bold",
            )

            ax.set_ylim(y_lims)
            y_min = ax.get_ylim()[0]
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(
                    lambda y, pos, y_min=y_min: custom_y_labels(y, pos, y_min)
                )
            )
            # Linked axis: the same curves read as absolute expected return
            abs_ax = ax.secondary_yaxis("right", functions=(to_return, to_pct))
            abs_ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_return))
            abs_ax.tick_params(axis="y", colors="0.5", labelsize=6)
            abs_ax.spines["right"].set_color("0.5")

            ax.set_title(
                title_dict.get(env_name, env_name), fontsize=8
            )
            ax.set_xlim(0, TOTAL_TIMESTEPS)
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, pos: f"{x/1e6:.0f}M")
            )
            ax.grid(True, alpha=0.3)
            ax.set_axisbelow(True)
            ax.set_xlabel("Environment Steps" if i == n_rows - 1 else "")
            ax.set_ylabel(RELATIVE_PERFORMANCE_LABEL if j == 0 else "")
            # Every panel has its own absolute scale, so every right axis gets the label
            abs_ax.set_ylabel(ABSOLUTE_RETURN_LABEL, color="0.5")
            # Row label (e.g. the map) left of the first column's y-axis label
            if j == 0 and row_labels:
                ax.annotate(
                    row_labels[i],
                    xy=(0, 0.5),
                    xycoords="axes fraction",
                    xytext=(-48, 0),
                    textcoords="offset points",
                    rotation=90,
                    ha="right",
                    va="center",
                    fontsize=9,
                )

    # One legend for all panels, below the grid. Collected over all panels since not every
    # algorithm runs on every map (no VDN-BA without a budget), in the order of algorithms_to_plot.
    legend_entries = {}
    for ax in axes.flat:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            legend_entries.setdefault(label, handle)
    labels = [
        f"{ALGORITHM_NAMES.get(alg, alg)} (IQM)" for alg in algorithms_to_plot
    ] + ["$\\text{H}_\\text{PS}$"]
    labels = [label for label in labels if label in legend_entries]
    handles = [legend_entries[label] for label in labels]
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(labels),
        fontsize=7,
        frameon=False,
        borderaxespad=0.0,
    )

    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, f"{filename}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Figures saved to {FIGURES_DIR}")


# Wall-clock time (s) for 1,000 rollout episodes (Appendix D, Table "Speedup comparison")
ROLLOUT_TIMES = {
    "ToyExample-v2": {"numpy": 77.1, "numpy_mp": 7.8, "jax": 7.0},
    "Cologne-v1": {"numpy": 540.0, "numpy_mp": 47.2, "jax": 7.0},
    "CologneBonnDusseldorf-v1": {"numpy": 6301.3, "numpy_mp": 481.8, "jax": 26.4},
}

# End-to-end VDN training on CologneBonnDusseldorf (Appendix D): time (s) to 58,752,000 steps
VDN_TRAINING_TIMES = {"numpy_epymarl": 1_987_397, "jax_jaxmarl": 21_952}


# VDN hyperparameter sensitivity (appendix). Cached W&B runs from
# evaluation/vdn_cbd_analysis.ipynb and evaluation/vdn_cologne_analysis.ipynb.
VDN_CACHE_DIRS = {
    "Cologne-v1": "evaluation/data/vdn_cologne",
    "Cologne-v1-unconstrained": "evaluation/data/vdn_cologne",
    "CologneBonnDusseldorf-v1": "evaluation/data/vdn_cbd_unconstrained",
    "CologneBonnDusseldorf-v1-unconstrained": "evaluation/data/vdn_cbd_unconstrained",
}
# Rank of each configuration in the VDN tuning sweep on Cologne (1 = selected)
VDN_COMBO_RANKS = {"C0003": 2, "C0023": 3, "C0004": 4, "C0043": 5}
VDN_SINGLE_CHANGES = {"vdn_lr": "LR only", "vdn_hidden": "Hidden only"}


def iter_vdn_runs(env_name):
    """(label, config, run_dir) of every labelled VDN configuration run on `env_name`."""
    cache = VDN_CACHE_DIRS[env_name]
    projects = {
        f"imp-act-jaxMARL-final-runs_{env_name}": "final",
        f"imp-act-jaxMARL-final-runs_{env_name}__vdn_rnn_100M": "100M",
        f"imp-act-jaxMARL-vdn-ablation_{env_name}": "ablation",
    }
    for project, kind in projects.items():
        project_dir = os.path.join(cache, project)
        if not os.path.isdir(project_dir):
            continue
        for run_id in sorted(os.listdir(project_dir)):
            run_dir = os.path.join(project_dir, run_id)
            with open(os.path.join(run_dir, "config.yaml")) as f:
                cfg = yaml.safe_load(f)
            if cfg.get("ALG_NAME") != "vdn_rnn":
                continue

            if kind == "final":
                label = "Rank 1" if cfg["TOTAL_TIMESTEPS"] == 50_000_000 else None
            elif kind == "100M":
                label = "100M steps"
            else:
                folder = cfg.get("HYDRA_PATH", "").split("/")[-2]
                combo = folder.removeprefix("vdn_").upper()
                if folder in VDN_SINGLE_CHANGES:
                    label = VDN_SINGLE_CHANGES[folder]
                elif combo in VDN_COMBO_RANKS:
                    label = f"Rank {VDN_COMBO_RANKS[combo]}"
                else:
                    label = None
            if label is not None:
                yield label, cfg, run_dir


def load_vdn_variants(env_name, heuristic_value):
    """Best test return per seed (% over H_PS) for every VDN configuration run on `env_name`.

    Only seeds that reached at least 95% of TOTAL_TIMESTEPS are kept.
    """
    results = {}
    for label, cfg, run_dir in iter_vdn_runs(env_name):
        history_file = os.path.join(run_dir, "history.csv")
        if not os.path.exists(history_file):
            continue
        history = pd.read_csv(history_file)[["env_step", "test_returned_episode_returns"]].dropna()
        if history.empty or history["env_step"].max() < 0.95 * cfg["TOTAL_TIMESTEPS"]:
            continue
        best = history["test_returned_episode_returns"].max()
        results.setdefault(label, []).append(
            (best - heuristic_value) / abs(heuristic_value) * 100
        )
    return results


# Order of the configurations in the CologneBonnDusseldorf-unconstrained figure and table:
# next-best ranks first, then single changes and longer training
VDN_CBD_ORDER = ["Rank 1", "Rank 2", "Rank 3", "Rank 4", "Rank 5", "LR only", "Hidden only", "100M steps"]


# Colors of the VDN appendix figures: stops of the viridis palette used in the other figures
VDN_COLORS = {
    "Rank 1": "#440154",  # dark purple
    "alternative": "#21918c",  # teal, ranks 2-5
    "lower rank": "#addc30",  # light green, ranks 6-10 (tuning figure only)
    "single": "#3b528b",  # blue, single changes of rank 1 (CBD figure only)
    "100M steps": "#fde725",  # yellow
}


def median_color(face_color):
    """Median line color that stays visible on the box fill (white on dark fills)."""
    r, g, b = mcolors.to_rgb(face_color)
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.45 else "#3a3a3a"
VDN_TUNING_DIR = "evaluation/data/hyper_parameter_tuning/hyper_parameter_tuning_Cologne-v1"


def load_vdn_tuning_ranking(heuristic_value, n_bins=100):
    """VDN tuning sweep on Cologne, ranked by mean best test return over seeds (1 = selected).

    Per combination: best test return of each seed and the learning curve of each seed
    (last test return in each of `n_bins` equal env-step bins), both as % over H_PS.
    """
    seeds, curves = {}, {}
    total_steps = None
    for run_id in sorted(os.listdir(VDN_TUNING_DIR)):
        run_dir = os.path.join(VDN_TUNING_DIR, run_id)
        with open(os.path.join(run_dir, "config.yaml")) as f:
            cfg = yaml.safe_load(f)
        if cfg.get("ALG_NAME") != "vdn_rnn":
            continue
        history = pd.read_csv(os.path.join(run_dir, "history.csv"))
        history = history[["env_step", "test_returned_episode_returns"]].dropna()
        pct = (history["test_returned_episode_returns"] - heuristic_value) / abs(heuristic_value) * 100
        bins = np.minimum((history["env_step"] / cfg["TOTAL_TIMESTEPS"] * n_bins).astype(int), n_bins - 1)
        combo = cfg["COMBO_ID"]
        total_steps = cfg["TOTAL_TIMESTEPS"]
        seeds.setdefault(combo, []).append(pct.max())
        curve = pct.groupby(bins).last().reindex(range(n_bins))
        # Light smoothing over neighbouring bins, as test returns are noisy
        curves.setdefault(combo, []).append(curve.rolling(5, min_periods=1).mean().values)
    steps = (np.arange(n_bins) + 1) / n_bins * total_steps
    ranked = sorted(seeds, key=lambda c: np.mean(seeds[c]), reverse=True)
    return steps, [(combo, seeds[combo], np.array(curves[combo])) for combo in ranked]


def draw_min_max_boxes(ax, groups, positions, colors, y_floor, width=0.6):
    """Boxes (25th-75th percentile, median line, min/max whiskers) as in Figure 3."""
    for values, pos, color in zip(groups, positions, colors):
        clipped = np.maximum(values, y_floor)
        ax.boxplot(
            [clipped],
            positions=[pos],
            widths=width,
            whis=(0, 100),
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": color, "edgecolor": "#3a3a3a", "linewidth": 0.8},
            medianprops={"color": median_color(color), "linewidth": 1.0},
            whiskerprops={"color": "#3a3a3a", "linewidth": 0.8},
            capprops={"color": "#3a3a3a", "linewidth": 0.8},
        )



def style_vdn_axis(ax, y_floor, y_top, y_ticks):
    """Axis style of the box-plot figures: solid H_PS line, full frame, light grid."""
    ax.axhline(0, color="red", linewidth=1, zorder=0)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_yticks(y_ticks)
    ax.set_ylim(y_floor, y_top)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, pos: custom_y_labels(y, pos, y_floor))
    )
    ax.set_ylabel(RELATIVE_PERFORMANCE_LABEL)


def box_legend(ax, loc, ncol=2):
    """Same box-plot legend as Figure 3."""
    ax.legend(
        handles=[
            mlines.Line2D([], [], color="red", linewidth=1),
            mpatches.Patch(facecolor="white", edgecolor="0.25", linewidth=1),
            mlines.Line2D([], [], color="0.25", linewidth=1),
            mlines.Line2D([], [], color="0.25", linewidth=1, marker="|", markersize=5),
        ],
        labels=[
            "Heuristic $\\text{H}_\\text{PS}$",
            "Box: 25th-75th percentile",
            "Line: median",
            "Whiskers: min / max",
        ],
        loc=loc,
        ncol=ncol,
        frameon=False,
        fontsize=6,
        handlelength=1.4,
        columnspacing=1.5,
    )


def save_vdn_figure(fig, filename):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, f"{filename}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {os.path.join(FIGURES_DIR, filename)}.pdf")


def create_vdn_tuning_figure(
    baseline_heuristic_values, filename="Figure_8_VDN_Tuning_Cologne", top_n=10, figsize=(7.0, 2.8)
):
    """Top `top_n` configurations of the VDN tuning sweep on budget-constrained Cologne:
    (a) mean learning curve over the sweep seeds, styled as the training curves;
    (b) highest test return of each seed, styled as the box plots."""
    plt.rcParams.update(PLOT_PARAMS)
    heuristic_score = baseline_heuristic_values["Cologne-v1"]
    steps, ranking = load_vdn_tuning_ranking(heuristic_score)
    ranking = ranking[:top_n]
    fig, (ax_curves, ax_box) = plt.subplots(1, 2, figsize=figsize)

    def rank_color(rank):
        return (
            VDN_COLORS["Rank 1"] if rank == 1
            else VDN_COLORS["alternative"] if rank <= 5
            else VDN_COLORS["lower rank"]
        )

    # (a) Learning curves as in the training-curve figures: dashed H_PS line, its value in the
    # corner, and the absolute expected return on a linked right axis.
    # Drawn worst-first so rank 1 ends up on top.
    for rank, (_, _, curves) in reversed(list(enumerate(ranking, start=1))):
        ax_curves.plot(
            steps, np.nanmean(curves, axis=0), color=rank_color(rank),
            linewidth=1.5 if rank == 1 else 1.0, zorder=3 if rank == 1 else 2,
        )
        if rank == 1:
            ax_curves.fill_between(
                steps, np.nanmin(curves, axis=0), np.nanmax(curves, axis=0),
                color=rank_color(rank), alpha=0.2, linewidth=0, zorder=1,
            )
    ax_curves.axhline(y=0, color="red", linestyle="--", linewidth=1)
    ax_curves.text(
        0.02, 0.97, f"$\\text{{H}}_\\text{{PS}}$ = {format_return(heuristic_score)}",
        transform=ax_curves.transAxes, ha="left", va="top", color="red", fontsize=7,
        fontweight="bold",
    )
    y_lims = (-40, 30)
    ax_curves.set_ylim(y_lims)
    ax_curves.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, pos: custom_y_labels(y, pos, y_lims[0]))
    )
    abs_ax = ax_curves.secondary_yaxis(
        "right",
        functions=(
            lambda p: heuristic_score + p / 100 * abs(heuristic_score),
            lambda r: (r - heuristic_score) / abs(heuristic_score) * 100,
        ),
    )
    abs_ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_return))
    abs_ax.tick_params(axis="y", colors="0.5", labelsize=6)
    abs_ax.spines["right"].set_color("0.5")
    abs_ax.set_ylabel(ABSOLUTE_RETURN_LABEL, color="0.5")
    ax_curves.set_xlim(0, steps[-1])
    ax_curves.xaxis.set_major_locator(mticker.MultipleLocator(5e6))
    ax_curves.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{x/1e6:.0f}M"))
    ax_curves.set_xlabel("Environment Steps")
    ax_curves.set_ylabel(RELATIVE_PERFORMANCE_LABEL)
    ax_curves.set_title("(a) Learning curves (mean over seeds)", fontsize=8)
    ax_curves.legend(
        handles=[
            mlines.Line2D([], [], color=VDN_COLORS["Rank 1"], linewidth=1.5),
            mpatches.Patch(facecolor=VDN_COLORS["Rank 1"], alpha=0.2),
            mlines.Line2D([], [], color=VDN_COLORS["alternative"], linewidth=1.0),
            mlines.Line2D([], [], color=VDN_COLORS["lower rank"], linewidth=1.0),
            mlines.Line2D([], [], color="red", linestyle="--", linewidth=1),
        ],
        labels=[
            "Rank 1 (selected)",
            "Rank 1, min / max over seeds",
            "Ranks 2-5",
            "Ranks 6-10",
            "$\\text{H}_\\text{PS}$",
        ],
        loc="lower right",
        frameon=False,
        fontsize=6,
        handlelength=1.4,
    )

    # (b) Highest expected return per seed as box plots: solid H_PS line and its value in the corner
    draw_min_max_boxes(
        ax_box,
        [values for _, values, _ in ranking],
        range(len(ranking)),
        [rank_color(rank) for rank in range(1, len(ranking) + 1)],
        y_floor=-np.inf,
    )
    rng = np.random.default_rng(0)
    for x, (_, values, _) in enumerate(ranking):
        ax_box.scatter(
            x + rng.uniform(-0.12, 0.12, len(values)), values,
            s=8, color="#3a3a3a", edgecolor="white", linewidth=0.4, zorder=4,
        )
    ax_box.axhline(y=0, color="red", linestyle="-", linewidth=1)
    ax_box.text(
        0.02, 0.97, f"$\\text{{H}}_\\text{{PS}}$ = {format_return(heuristic_score)}",
        transform=ax_box.transAxes, ha="left", va="top", color="red", fontsize=7,
        fontweight="bold",
    )
    ax_box.set_xticks(range(len(ranking)))
    ax_box.set_xticklabels(
        ["1\n(selected)" if r == 1 else str(r) for r in range(1, len(ranking) + 1)]
    )
    ax_box.set_xlabel("Rank in hyperparameter search")
    ax_box.set_xlim(-0.6, len(ranking) - 0.4)
    ax_box.set_ylim(-2, 20)  # room below 0 so the H_PS line is visible
    ax_box.set_yticks(range(0, 21, 5))
    ax_box.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, pos: custom_y_labels(y, pos, None))
    )
    ax_box.set_title("(b) Highest expected return per seed", fontsize=8)
    ax_box.legend(
        handles=[
            mpatches.Patch(facecolor="white", edgecolor="0.25", linewidth=1),
            mlines.Line2D([], [], color="0.25", linewidth=1),
            mlines.Line2D([], [], color="0.25", linewidth=1, marker="|", markersize=5),
            mlines.Line2D([], [], linestyle="", marker="o", markersize=2.5, color="#3a3a3a"),
            mlines.Line2D([], [], color="red", linewidth=1),
        ],
        labels=[
            "Box: 25th-75th percentile",
            "Line: median",
            "Whiskers: min / max",
            "Seed",
            "Heuristic $\\text{H}_\\text{PS}$",
        ],
        loc="lower left",
        bbox_to_anchor=(0, 0.06),
        ncol=2,
        frameon=False,
        fontsize=6,
        handlelength=1.4,
        columnspacing=1.5,
    )

    for ax in (ax_curves, ax_box):
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
    fig.suptitle(
        f"Hyperparameter tuning environment: {tt('Cologne')} "
        f"(60 agents, {ENV_BUDGETS['Cologne-v1']} budget constraint)",
        fontsize=9,
    )
    plt.tight_layout()
    save_vdn_figure(fig, filename)


def create_vdn_transfer_figure(
    baseline_heuristic_values, filename="Figure_9_VDN_Rank1_vs_Rank5", figsize=(3.8, 2.8)
):
    """Selected configuration (rank 1) vs. rank 5 on Cologne and CologneBonnDusseldorf,
    with and without the budget constraint. One panel per map, map name on top as in Figure 4."""
    plt.rcParams.update(PLOT_PARAMS)
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    y_floor, y_top = -40, 25
    width = 0.32
    maps = [
        (axes[0], "Cologne-v1", ENV_PLOT_TITLES["Cologne-v1"]),
        (axes[1], "CologneBonnDusseldorf-v1", ENV_PLOT_TITLES["CologneBonnDusseldorf-v1"]),
    ]
    for ax, env, title in maps:
        for x, scenario in enumerate([env, f"{env}-unconstrained"]):
            variants = load_vdn_variants(scenario, baseline_heuristic_values[scenario])
            draw_min_max_boxes(
                ax,
                [variants["Rank 1"], variants["Rank 5"]],
                [x - width / 2 - 0.02, x + width / 2 + 0.02],
                [VDN_COLORS["Rank 1"], VDN_COLORS["alternative"]],
                y_floor,
                width=width,
            )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Constrained", "Unconstrained"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(title, fontsize=8)
        style_vdn_axis(ax, y_floor, y_top, [-40, -30, -20, -10, 0, 10, 20])
    axes[1].set_ylabel("")
    # Box-plot elements are explained in the legend of the tuning figure; only the ranks here
    axes[0].legend(
        handles=[
            mpatches.Patch(facecolor=VDN_COLORS["Rank 1"], edgecolor="#3a3a3a", label="Rank 1 (selected)"),
            mpatches.Patch(facecolor=VDN_COLORS["alternative"], edgecolor="#3a3a3a", label="Rank 5"),
        ],
        loc="lower left",
        frameon=False,
        fontsize=6,
        handlelength=1.0,
    )
    plt.tight_layout(w_pad=0.5)
    save_vdn_figure(fig, filename)


def create_vdn_cbd_unconstrained_figure(
    baseline_heuristic_values, filename="Figure_10_VDN_CBD_Unconstrained", figsize=(3.6, 2.35)
):
    """Next-best tuning ranks and variants of the selected VDN configuration on
    unconstrained CologneBonnDusseldorf. Hyperparameters are in a LaTeX table next to it."""
    plt.rcParams.update(PLOT_PARAMS)
    fig, ax = plt.subplots(figsize=figsize)
    y_floor, y_top = -40, 5
    env = "CologneBonnDusseldorf-v1-unconstrained"
    variants = load_vdn_variants(env, baseline_heuristic_values[env])
    order = [label for label in VDN_CBD_ORDER if label in variants]
    colors = [
        VDN_COLORS["Rank 1"] if label == "Rank 1"
        else VDN_COLORS["single"] if label in VDN_SINGLE_CHANGES.values()
        else VDN_COLORS["100M steps"] if label == "100M steps"
        else VDN_COLORS["alternative"]
        for label in order
    ]
    draw_min_max_boxes(ax, [variants[l] for l in order], range(len(order)), colors, y_floor)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(
        [f"{l}\n(selected)" if l == "Rank 1" else l.replace(" only", "\nonly").replace(" steps", "\nsteps")
         for l in order],
        fontsize=6,
    )
    style_vdn_axis(ax, y_floor, y_top, [-40, -30, -20, -10, 0])
    # Map and scenario on top, as in Figure 2 and the main box-plot figures
    ax.set_title(ENV_PLOT_TITLES[env], fontsize=8)
    save_vdn_figure(fig, filename)


def format_duration(seconds):
    """Compact wall-clock label, e.g. 26 s, 8 min, 6.1 h, 23 d."""
    if seconds < 100:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds / 60:.0f} min"
    if seconds < 86400:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.0f} d"


BAR_EDGE = "#2b2b2b"  # outline of the bars in the speedup figure


def create_speedup_figure(filename="Figure_3_JAX_Speedup", figsize=(3.0, 2.0)):
    """Wall-clock times of NumPy and JAX, with arrows marking the JAX speedup."""
    plt.rcParams.update(PLOT_PARAMS)
    fig, ax = plt.subplots(figsize=figsize)

    envs = list(ROLLOUT_TIMES)
    group_spacing = 1.55  # room for the speedup arrow and label between groups
    x = np.arange(len(envs)) * group_spacing
    width = 0.36  # wide enough for the time labels on top of the bars
    methods = {
        "numpy": ("NumPy, AMD EPYC 7642 CPU (1 core)", "#afafaf"),
        "numpy_mp": ("NumPy MP, AMD EPYC 7642 CPU (12 cores)", "#28557c"),
        "jax": ("JAX, NVIDIA RTX 4090 GPU", "#369a61"),
    }
    offsets = {method: (i - 1) * width for i, method in enumerate(methods)}

    def draw_speedup_arrow(x_from, t_from, x_to, t_to, label=None):
        """Dotted level from the NumPy bar top, then a vertical drop onto the JAX bar."""
        label = label or f"{t_from / t_to:.0f}"
        x_arrow = x_to + width / 2 + 0.17  # clear of the time label on the JAX bar
        ax.plot(
            [x_from + width / 2, x_arrow],
            [t_from, t_from],
            color="black",
            linestyle=":",
            linewidth=0.6,
        )
        ax.plot(
            [x_to + width / 2, x_arrow],
            [t_to, t_to],
            color="black",
            linestyle=":",
            linewidth=0.6,
        )
        ax.annotate(
            "",
            xy=(x_arrow, t_to),
            xytext=(x_arrow, t_from),
            arrowprops=dict(
                arrowstyle="-|>",
                color="black",
                lw=0.7,
                mutation_scale=5,
                shrinkA=0,
                shrinkB=0,
            ),
        )
        # Label resting on the dotted level, ending at the arrow
        ax.text(
            x_arrow + 0.02,
            t_from * 1.08,
            f"{label}×",
            ha="right",
            va="bottom",
            fontsize=6,
            fontweight="bold",
        )

    # Rollouts: wall-clock time for 1,000 episodes
    for method, (label, color) in methods.items():
        times = [ROLLOUT_TIMES[env][method] for env in envs]
        bars = ax.bar(
            x + offsets[method], times, 0.85 * width, label=label, color=color,
            edgecolor=BAR_EDGE, linewidth=0.6,
        )
        ax.bar_label(bars, labels=[format_duration(t) for t in times], padding=1, fontsize=5)
    for i, env in enumerate(envs):
        draw_speedup_arrow(
            x[i] + offsets["numpy"],
            ROLLOUT_TIMES[env]["numpy"],
            x[i] + offsets["jax"],
            ROLLOUT_TIMES[env]["jax"],
        )

    # Training: end-to-end VDN on CologneBonnDusseldorf, NumPy/EPyMARL vs JAX/JaxMARL
    x_split = x[-1] + 1.15
    x_train = x_split + 0.6
    # Hatched and unfilled: same NumPy/JAX colours, marking training rather than rollouts
    train_bars = [
        (x_train - width / 2, VDN_TRAINING_TIMES["numpy_epymarl"], methods["numpy"][1],
         "EPyMARL + NumPy IMP-act"),
        (x_train + width / 2, VDN_TRAINING_TIMES["jax_jaxmarl"], methods["jax"][1],
         "JaxMARL + JAX IMP-act"),
    ]
    for x_bar, t, color, framework in train_bars:
        # Hatch in the bar colour, outline dark as for the rollout bars
        ax.bar(x_bar, t, 0.85 * width, facecolor="white", edgecolor=color, hatch="//////",
               linewidth=0)
        bars = ax.bar(x_bar, t, 0.85 * width, facecolor="none", edgecolor=BAR_EDGE, linewidth=0.6)
        ax.bar_label(bars, labels=[format_duration(t)], padding=1, fontsize=5)
        ax.text(
            x_bar,
            1.5,
            framework,
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=5,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6),
        )
    draw_speedup_arrow(
        train_bars[0][0],
        train_bars[0][1],
        train_bars[1][0],
        train_bars[1][1],
        label="90",  # 90.5, reported as ~90 in the paper
    )

    # Partition rollouts and training
    ax.axvline(x_split, color="gray", linestyle="--", linewidth=0.8)
    for x_text, text in [
        ((x[0] + x[-1]) / 2, "Environment rollouts (1,000 episodes)"),
        (x_train, "Training (VDN)"),
    ]:
        ax.text(
            x_text,
            1.02,
            text,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.set_yscale("log")
    ax.set_ylim(1, 3e7)
    ax.set_xlim(x[0] - 0.65, x_train + 0.75)
    # Human-readable durations instead of powers of ten
    duration_ticks = {1: "1 s", 10: "10 s", 60: "1 min", 600: "10 min", 3600: "1 h", 86400: "1 day", 864000: "10 days"}
    ax.set_yticks(list(duration_ticks))
    ax.set_yticklabels(list(duration_ticks.values()), fontsize=6)
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    cbd_label = f"{tt('CologneBonn')}\n{tt('Dusseldorf')}\n(178 agents)"
    ax.set_xticks(list(x) + [x_train])
    ax.set_xticklabels(
        [
            f"{tt('ToyExample')}\n(12 agents)",
            f"{tt('Cologne')}\n(60 agents)",
            cbd_label,
            cbd_label,
        ],
        fontsize=6,
    )
    ax.grid(axis="y", which="major", color="#e0e0e0", linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        mpatches.Patch(facecolor="white", edgecolor=BAR_EDGE, hatch="//////", linewidth=0.6)
    )
    labels.append("Training, NVIDIA RTX 4090 GPU (Appendix D)")
    ax.legend(
        handles,
        labels,
        loc="upper left",
        frameon=False,
        handlelength=1.6,  # wide enough to show the training hatch
        handleheight=1.0,
        handletextpad=0.4,
        borderaxespad=0.3,
        fontsize=5,
    )

    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, f"{filename}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Figures saved to {FIGURES_DIR}")


def main():
    create_speedup_figure()

    data, baseline_heuristic_values = load_data()
    create_vdn_tuning_figure(baseline_heuristic_values)
    create_vdn_transfer_figure(baseline_heuristic_values)
    create_vdn_cbd_unconstrained_figure(baseline_heuristic_values)

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
        "Cologne-v1-unconstrained": f"{tt('Cologne')} Unconstrained\n(Infinite budget)",
        "Cologne-v1-moderate-budget": f"{tt('Cologne')} Moderate\n(300M budget)",
        "Cologne-v1": f"{tt('Cologne')} Limited\n(200M budget)",
        "Cologne-v1-critical-budget": f"{tt('Cologne')} Critical\n(150M budget)",
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
        filename="Figure_5_Cologne_Budgets",
        title_dict=title_dict,
        show_baseline=True,
        show_scatter=False,
        show_percentages=False,
        y_lims=(-40, 30),
        env_colors=env_colors,
        full_legend=True,        title_fontsize=8,  # fits the typewriter "Cologne" in the four titles
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
        filename="Figure_4_Envs_Constrained_vs_Unconstrained",
        y_lims=(-40, 35),  # headroom for the H_PS and budget labels
        show_baseline=True,
        baseline_heuristic_values=baseline_heuristic_values,
    )

    # Training curves: constrained vs unconstrained per map, and the Cologne budget scenarios
    curve_algorithms = [
        "vdn_rnn",
        "vdn_ba_rnn",
        "qmix_rnn",
        "pqn_rnn",
        "mappo_rnn",
        "ippo_rnn",
    ]
    # Budgets from the imp-act presets (budget_amount)
    # Map (with agents) per row, scenario per panel
    curve_maps = {
        "ToyExample-v2": f"{tt('ToyExample')} (12 agents)",
        "Cologne-v1": f"{tt('Cologne')} (60 agents)",
        "CologneBonnDusseldorf-v1": f"{tt('CologneBonnDusseldorf')} (178 agents)",
    }
    curve_titles = {}
    for env in curve_maps:
        curve_titles[env] = f"Constrained ({ENV_BUDGETS[env]} budget)"
        curve_titles[f"{env}-unconstrained"] = "Unconstrained"
    create_training_curves(
        data,
        baseline_heuristic_values,
        env_grid=[
            [env, f"{env}-unconstrained"]
            for env in ["ToyExample-v2", "Cologne-v1", "CologneBonnDusseldorf-v1"]
        ],
        algorithms_to_plot=curve_algorithms,
        filename="Figure_6_Training_Curves_Envs_Constrained_vs_Unconstrained",
        title_dict=curve_titles,
        row_labels=list(curve_maps.values()),
    )

    budget_titles = {
        "Cologne-v1-unconstrained": f"{tt('Cologne')} Unconstrained",
        "Cologne-v1-moderate-budget": f"{tt('Cologne')} Moderate (300M budget)",
        "Cologne-v1": f"{tt('Cologne')} Limited (200M budget)",
        "Cologne-v1-critical-budget": f"{tt('Cologne')} Critical (150M budget)",
    }
    create_training_curves(
        data,
        baseline_heuristic_values,
        env_grid=[
            ["Cologne-v1-unconstrained", "Cologne-v1-moderate-budget"],
            ["Cologne-v1", "Cologne-v1-critical-budget"],
        ],
        algorithms_to_plot=curve_algorithms,
        filename="Figure_7_Training_Curves_Cologne_Budgets",
        title_dict=budget_titles,
        figsize=(7.0, 5.2),
    )


if __name__ == "__main__":
    main()
