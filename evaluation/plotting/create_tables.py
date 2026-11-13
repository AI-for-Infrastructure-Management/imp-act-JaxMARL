import os

import numpy as np
import pandas as pd
import yaml

from create_figures import VDN_CBD_ORDER, iter_vdn_runs

# Configuration
DATA_DIR = "evaluation/plotting/data"
TABLES_DIR = "evaluation/plotting/tables"
HEURISTIC_RESULTS_FILE = os.path.join(DATA_DIR, "best_heuristic_results.yaml")
EVALUATION_RESULTS_FILE = os.path.join(DATA_DIR, "combined_evaluation_results.csv")

ALGORITHM_NAMES = {
    "vdn_rnn": "VDN",
    "qmix_rnn": "QMIX",
    "pqn_rnn": "PQN-VDN",
    "ippo_rnn": "IPPO",
    "mappo_rnn": "MAPPO",
    "vdn_ba_rnn": r"$\text{VDN}_{\text{BA}}$",
}

ENV_DISPLAY_NAMES = {
    "ToyExample-v2": r"\textbf{Toy-} \\ \textbf{Example}",
    "Cologne-v1": r"\textbf{Cologne}",
    "CologneBonnDusseldorf-v1": r"\textbf{CologneBonn-} \\ \textbf{Dusseldorf}",
    "ToyExample-v2-unconstrained": r"\textbf{ToyExample} \\ \textbf{Unconstrained}",
    "Cologne-v1-unconstrained": r"\textbf{Cologne} \\ \textbf{Unconstrained}",
    "CologneBonnDusseldorf-v1-unconstrained": r"\textbf{CologneBonn-} \\ \textbf{Dusseldorf} \\ \textbf{Unconstrained}",
    "Cologne-v1-moderate-budget": r"\textbf{Cologne} \\ \textbf{Moderate}",
    "Cologne-v1-critical-budget": r"\textbf{Cologne} \\ \textbf{Critical}",
}


def load_data():
    """Loads heuristic results and evaluation data."""
    with open(HEURISTIC_RESULTS_FILE) as f:
        heuristic_results = yaml.load(f, Loader=yaml.FullLoader)

    baseline_heuristic_values = {}
    for env, results in heuristic_results.items():
        if "segment_based_heuristic" in results:
            baseline_heuristic_values[env] = results["segment_based_heuristic"][
                "mean_reward"
            ]

    data = pd.read_csv(EVALUATION_RESULTS_FILE)
    return data, baseline_heuristic_values


def get_best_checkpoints(data, env_name, baseline_heuristic_values):
    """
    Extracts best checkpoints for each algorithm and normalizes rewards.
    """
    algorithms = data["algorithm"].unique()
    heuristic_score = baseline_heuristic_values[env_name]

    best_checkpoint_per_algorithm = {}

    for alg in algorithms:
        alg_data = data[data["algorithm"] == alg]
        runs = alg_data["WANDB_RUN_ID"].unique()

        best_runs = []
        for run in runs:
            run_data = alg_data[alg_data["WANDB_RUN_ID"] == run]
            best_idx = run_data["mean"].idxmax()
            best_row = run_data.loc[best_idx]

            # Normalize
            norm_mean = (best_row["mean"] - heuristic_score) / abs(heuristic_score)
            norm_lower = (best_row["lower_ci"] - heuristic_score) / abs(heuristic_score)
            norm_upper = (best_row["upper_ci"] - heuristic_score) / abs(heuristic_score)

            best_runs.append(
                {"mean": norm_mean, "lower_ci": norm_lower, "upper_ci": norm_upper}
            )

        best_checkpoint_per_algorithm[alg] = best_runs

    return best_checkpoint_per_algorithm


def format_heuristic_value(val):
    """Formats heuristic value for display."""
    if abs(val) >= 1e9:
        return f"{val/1e9:.2f}\\text{{B}}"
    return f"{val/1e6:.2f}\\text{{M}}"


def wrap_table(latex_lines, caption, label, placement, arraystretch=1.25):
    """Wraps a tabular in a table environment (shrunk to the text width if wider), with shared font and spacing."""
    return [
        r"\begin{table}[" + placement + "]",
        r"  \caption{" + caption + "}",
        *([r"  \label{" + label + "}"] if label else []),
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \renewcommand{\arraystretch}{" + str(arraystretch) + "}",
        # Shrink to the text width if wider, never enlarge
        r"  \resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%",
        *latex_lines,
        r"  }",
        r"\end{table}",
    ]


def generate_latex_table(
    best_checkpoints_all,
    baseline_heuristic_values,
    envs_to_plot=None,
    output_filename="results_table.tex",
):
    """Generates LaTeX table."""
    if envs_to_plot is None:
        envs_to_plot = ["ToyExample-v2", "Cologne-v1", "CologneBonnDusseldorf-v1"]

    main_algorithms = ["vdn_rnn", "qmix_rnn", "pqn_rnn", "mappo_rnn", "ippo_rnn"]
    baseline_alg = "vdn_ba_rnn"

    latex_lines = []
    latex_lines.append(r"  \begin{tabular}{llrc}")
    latex_lines.append(r"    \toprule")
    latex_lines.append(r"    \thead{\textbf{Environment}} & \thead{\textbf{Algorithm}}")
    latex_lines.append(r"    & \thead{\textbf{${\Delta \bar{R}^{\pi}_0}$}} ")
    latex_lines.append(r"    & \thead{\textbf{95\% CI}} \\")
    latex_lines.append(r"    \midrule")

    for env in envs_to_plot:
        # Calculate stats for all algorithms
        alg_stats = {}
        for alg in main_algorithms + [baseline_alg]:
            if alg in best_checkpoints_all[env]:
                runs = best_checkpoints_all[env][alg]
                # Find the single best checkpoint across all runs
                best_run = max(runs, key=lambda x: x["mean"])

                val_mean = best_run["mean"] * 100
                val_lower = best_run["lower_ci"] * 100
                val_upper = best_run["upper_ci"] * 100

                alg_stats[alg] = {"mean": val_mean, "ci": (val_lower, val_upper)}
            else:
                alg_stats[alg] = {"mean": np.nan, "ci": (np.nan, np.nan)}

        # Find best main algorithm
        best_main_alg = None
        best_main_mean = -np.inf
        best_main_ci = None

        for alg in main_algorithms:
            if (
                not np.isnan(alg_stats[alg]["mean"])
                and alg_stats[alg]["mean"] > best_main_mean
            ):
                best_main_mean = alg_stats[alg]["mean"]
                best_main_alg = alg
                best_main_ci = alg_stats[alg]["ci"]

        # Format environment column
        h_val = baseline_heuristic_values[env]
        h_str = format_heuristic_value(h_val)
        env_display = ENV_DISPLAY_NAMES.get(env, env.replace("_", "\\_"))
        multirow_cmd = (
            r"\multirow{6}{*}{{\shortstack{"
            + env_display
            + r" \\ $\text{H}_\text{PS}="
            + h_str
            + r"$}}}"
        )

        # Print main algorithms
        for i, alg in enumerate(main_algorithms):
            row_str = ""
            if i == 0:
                row_str += f"    {multirow_cmd} \n    & "
            else:
                row_str += "    & "

            alg_name = ALGORITHM_NAMES[alg]
            mean = alg_stats[alg]["mean"]
            ci = alg_stats[alg]["ci"]

            # Formatting mean
            if np.isnan(mean):
                mean_str = "--"
                ci_str = "--"
            else:
                mean_str = f"{mean:+.2f}\\%"
                if alg == best_main_alg:
                    mean_str = r"\textbf{" + mean_str + "}"

                # Check for overlap with best
                is_star = False
                if alg != best_main_alg and not np.isnan(mean):
                    # Check if CI overlaps with best algorithm's CI
                    overlap = max(ci[0], best_main_ci[0]) < min(ci[1], best_main_ci[1])
                    if overlap:
                        is_star = True

                if is_star:
                    mean_str = "*" + mean_str

                ci_str = f"{{[{ci[0]:.2f}\\,, {ci[1]:.2f}]}}"

            row_str += f"{alg_name:<8} & {mean_str:<16} & {ci_str:<18} \\\\"
            latex_lines.append(row_str)

        # Separator
        latex_lines.append(r"    \cmidrule(lr){2-4}")

        # Print baseline algorithm
        alg = baseline_alg
        alg_name = ALGORITHM_NAMES[alg]
        mean = alg_stats[alg]["mean"]
        ci = alg_stats[alg]["ci"]

        if np.isnan(mean):
            mean_str = "--"
            ci_str = "--"
        else:
            mean_str = f"{mean:+.2f}\\%"
            ci_str = f"{{[{ci[0]:.2f}\\,, {ci[1]:.2f}]}}"

        latex_lines.append(
            f"    & {alg_name:<8} & {mean_str:<16} & {ci_str:<18}   \\\\"
        )

        if env != envs_to_plot[-1]:
            latex_lines.append(r"    \midrule")
        else:
            latex_lines.append(r"    \bottomrule")

    latex_lines.append(r"  \end{tabular}")

    os.makedirs(TABLES_DIR, exist_ok=True)
    output_path = os.path.join(TABLES_DIR, output_filename)
    with open(output_path, "w") as f:
        f.write("\n".join(latex_lines))
    print(f"Table saved to {output_path}")


# Extra algorithms: shown in the main results table with an asterisk, but not considered
# when marking the best result and the results tied with it
EXTRA_ALGORITHMS = {"vdn_ba_rnn"}


def best_model_stats(data, baseline_heuristic_values, heuristic_results, env):
    """
    Best checkpoint across all seeds per algorithm on env, as % vs H_PS, with that checkpoint's
    own 95% CI. H_PS is best if no algorithm beats it; results whose CI overlaps the best
    one's are marked tied. EXTRA_ALGORITHMS are never best or tied.
    """
    h = heuristic_results[env]["segment_based_heuristic"]
    h_mean = baseline_heuristic_values[env]

    def to_pct(v):
        return (v - h_mean) / abs(h_mean) * 100

    alg_stats = {}
    env_data = data[data["map_name"] == env]
    for alg, alg_data in env_data.groupby("algorithm"):
        best = alg_data.loc[alg_data["mean"].idxmax()]
        alg_stats[alg] = {
            "mean": to_pct(best["mean"]),
            "ci": (to_pct(best["lower_ci"]), to_pct(best["upper_ci"])),
        }

    best_alg = max(
        (a for a in alg_stats if a not in EXTRA_ALGORITHMS),
        key=lambda a: alg_stats[a]["mean"],
    )
    heuristic_wins = alg_stats[best_alg]["mean"] <= 0
    h_stats = {
        "mean": h_mean,
        "ci": (to_pct(h["lower_95ci"]), to_pct(h["upper_95ci"])),
        "is_best": heuristic_wins,
    }
    ref_ci = h_stats["ci"] if heuristic_wins else alg_stats[best_alg]["ci"]
    for alg, s in alg_stats.items():
        s["is_best"] = not heuristic_wins and alg == best_alg
        s["tied"] = (
            not s["is_best"]
            and alg not in EXTRA_ALGORITHMS
            and s["ci"][0] <= ref_ci[1]
            and ref_ci[0] <= s["ci"][1]
        )
    return alg_stats, h_stats


def format_ci(s):
    return f"{{[{s['ci'][0]:.2f}\\,, {s['ci'][1]:.2f}]}}"


def format_alg_mean(alg_stats, alg):
    """% vs H_PS; bold if best, underlined if its CI overlaps the best one's."""
    if alg not in alg_stats:
        return "--"
    s = alg_stats[alg]
    mean_str = f"{s['mean']:+.2f}\\%"
    if s["is_best"]:
        mean_str = r"\textbf{" + mean_str + "}"
    elif s["tied"]:
        mean_str = r"\underline{" + mean_str + "}"
    return mean_str


def format_heuristic_mean(h_stats):
    """H_PS is the reference, so 0% on the same scale as the algorithms; bold if best."""
    mean_str = r"0.00\%"
    if h_stats["is_best"]:
        mean_str = r"\textbf{" + mean_str + "}"
    return mean_str


def generate_latex_table_scenarios(
    data,
    baseline_heuristic_values,
    blocks,
    output_filename,
    scenario_header="Scenario",
    caption=None,
    label=None,
    placement="h",
):
    """
    Generates LaTeX table of the best model per algorithm, with algorithms in columns (like the
    per-seed table) and one value row plus one CI row per scenario. Values and marking as in
    best_model_stats.

    blocks: list of (display name, [(env, scenario label), ...]), one Environment cell per block.
    """
    algorithms = [
        "vdn_ba_rnn",
        "vdn_rnn",
        "qmix_rnn",
        "pqn_rnn",
        "mappo_rnn",
        "ippo_rnn",
    ]  # VDN_BA next to H_PS
    last_col = 3 + len(algorithms)

    with open(HEURISTIC_RESULTS_FILE) as f:
        heuristic_results = yaml.load(f, Loader=yaml.FullLoader)

    latex_lines = []
    latex_lines.append(
        r"  \begin{tabular}{@{}cl" + "c" * (1 + len(algorithms)) + "@{}}"
    )
    latex_lines.append(r"    \toprule")
    latex_lines.append(
        r"    \thead{\textbf{Environment}} & \thead{\textbf{"
        + scenario_header
        + r"}} & \thead{\textbf{$\text{H}_\text{PS}$}} & "
        + " & ".join(
            r"\thead{\textbf{"
            + ALGORITHM_NAMES[alg]
            + ("$^*$" if alg in EXTRA_ALGORITHMS else "")
            + "}}"
            for alg in algorithms
        )
        + r" \\"
    )
    latex_lines.append(r"    \midrule")

    for k, (display_name, rows) in enumerate(blocks):
        latex_lines.append(
            r"    \multirow{"
            + str(2 * len(rows))
            + r"}{*}{{\shortstack[c]{"
            + display_name
            + r"}}}"
        )
        for i, (env, env_label) in enumerate(rows):
            alg_stats, h_stats = best_model_stats(
                data, baseline_heuristic_values, heuristic_results, env
            )
            # Value row, then its 95% CI row; H_PS's absolute return goes under the scenario
            # label, in the CI row so both rows keep the same height
            latex_lines.append(f"    % {env}")
            latex_lines.append(
                r"    & "
                + env_label
                + " & "
                + format_heuristic_mean(h_stats)
                + " & "
                + " & ".join(format_alg_mean(alg_stats, alg) for alg in algorithms)
                + r" \\"
            )
            latex_lines.append(
                r"    & {\footnotesize\color{gray} $\text{H}_\text{PS}$: $"
                + format_heuristic_value(h_stats["mean"])
                + "$} & "
                + format_ci(h_stats)
                + " & "
                + " & ".join(
                    format_ci(alg_stats[alg]) if alg in alg_stats else "--"
                    for alg in algorithms
                )
                + r" \\"
            )
            # Rule between scenarios, with the same gap above and below
            if i != len(rows) - 1:
                latex_lines.append(r"    \addlinespace[2pt]")
                latex_lines.append(f"    \\cmidrule(lr){{2-{last_col}}}")
                latex_lines.append(r"    \addlinespace[2pt]")

        if k != len(blocks) - 1:
            latex_lines.append(r"    \addlinespace[2pt]")
            latex_lines.append(r"    \midrule")
            latex_lines.append(r"    \addlinespace[2pt]")

    latex_lines.append(r"    \bottomrule")
    latex_lines.append(r"  \end{tabular}")

    if caption is not None:
        latex_lines = wrap_table(
            latex_lines, caption, label, placement, arraystretch=1.15
        )

    os.makedirs(TABLES_DIR, exist_ok=True)
    output_path = os.path.join(TABLES_DIR, output_filename)
    with open(output_path, "w") as f:
        f.write("\n".join(latex_lines))
    print(f"Table saved to {output_path}")


def get_per_seed_results(data, env_name, baseline_heuristic_values):
    """
    Extracts the best checkpoint per seed for each algorithm and normalizes rewards.
    Returns a dict: {algorithm: [{seed: int, mean: float}, ...]}
    """
    algorithms = data["algorithm"].unique()
    heuristic_score = baseline_heuristic_values[env_name]

    per_seed_results = {}

    for alg in algorithms:
        alg_data = data[data["algorithm"] == alg]
        runs = alg_data["WANDB_RUN_ID"].unique()

        seed_results = []
        for i, run in enumerate(runs):
            run_data = alg_data[alg_data["WANDB_RUN_ID"] == run]
            best_idx = run_data["mean"].idxmax()
            best_row = run_data.loc[best_idx]

            # Normalize
            norm_mean = (
                (best_row["mean"] - heuristic_score) / abs(heuristic_score) * 100
            )

            seed_results.append({"seed": i + 1, "mean": norm_mean})

        # Sort by mean (descending)
        seed_results.sort(key=lambda x: x["mean"], reverse=True)
        per_seed_results[alg] = seed_results

    return per_seed_results


def generate_latex_table_per_seed(
    per_seed_results_all,
    baseline_heuristic_values,
    blocks,
    output_filename,
    caption=None,
    label=None,
):
    """
    Generates LaTeX table showing per-seed results with IQR, two scenarios side by side.

    blocks: list of (display name, left env, right env, (left label, right label)). A block
    whose labels differ from the previous block's gets its own header row. Consecutive blocks
    with the same display name share one Environment cell.
    Each side lists the algorithms with runs on its maps (VDN_BA first, next to H_PS, as in the
    budget table), so VDN_BA only appears for the constrained scenarios.
    Rows are ranks: each algorithm's seeds are sorted by performance independently.
    With a caption, the tabular is wrapped in a full-page table (see wrap_table).
    """
    with open(HEURISTIC_RESULTS_FILE) as f:
        heuristic_results = yaml.load(f, Loader=yaml.FullLoader)

    all_algorithms = [
        "vdn_ba_rnn",
        "vdn_rnn",
        "qmix_rnn",
        "pqn_rnn",
        "mappo_rnn",
        "ippo_rnn",
    ]
    side_algorithms = [
        [
            alg
            for alg in all_algorithms
            if any(alg in per_seed_results_all[block[1 + side]] for block in blocks)
        ]
        for side in (0, 1)
    ]
    n_left, n_right = len(side_algorithms[0]), len(side_algorithms[1])
    last_col = 2 + n_left + n_right

    def num_seeds_of(env):
        # Not from the first column's algorithm, which may have no runs on env (VDN_BA)
        return max(len(seeds) for seeds in per_seed_results_all[env].values())

    def alg_headers(algs):
        return " & ".join(
            r"\thead{\textbf{" + ALGORITHM_NAMES[alg] + "}}" for alg in algs
        )

    def group_header(labels):
        return [
            r"    & & \multicolumn{"
            + str(n_left)
            + r"}{c}{\textbf{"
            + labels[0]
            + r"}} & \multicolumn{"
            + str(n_right)
            + r"}{c}{\textbf{"
            + labels[1]
            + r"}} \\",
            f"    \\cmidrule(lr){{3-{2 + n_left}}} \\cmidrule(lr){{{3 + n_left}-{last_col}}}",
        ]

    latex_lines = []
    # Extra gap between the left and right scenario groups
    latex_lines.append(
        r"  \begin{tabular}{@{}cr"
        + "r" * n_left
        + r"@{\hspace{16pt}}"
        + "r" * n_right
        + "@{}}"
    )
    latex_lines.append(r"    \toprule")
    latex_lines += group_header(blocks[0][3])
    latex_lines.append(
        r"    \thead{\textbf{Environment}} & \thead{\textbf{Rank}} & "
        + alg_headers(side_algorithms[0])
        + " & "
        + alg_headers(side_algorithms[1])
        + r" \\"
    )
    latex_lines.append(r"    \midrule")

    prev_labels = blocks[0][3]
    for k, (display_name, left_env, right_env, labels) in enumerate(blocks):
        if labels != prev_labels:
            latex_lines += group_header(labels)
            prev_labels = labels

        sides = [per_seed_results_all[left_env], per_seed_results_all[right_env]]
        num_seeds = num_seeds_of(left_env)
        # H_PS as the 0 reference with its 95% CI, both in % relative to its mean, and its
        # absolute return in grey (as in the main table)
        h_strs = []
        for env in (left_env, right_env):
            h_mean = baseline_heuristic_values[env]
            h = heuristic_results[env]["segment_based_heuristic"]
            lo, hi = (
                (v - h_mean) / abs(h_mean) * 100
                for v in (h["lower_95ci"], h["upper_95ci"])
            )
            h_strs.append(
                f"0.00\\% \\; {{[{lo:.2f}\\,, {hi:.2f}]}} \\quad "
                f"{{\\footnotesize\\color{{gray}} ${format_heuristic_value(h_mean)}$}}"
            )

        same_as_prev = k > 0 and blocks[k - 1][0] == display_name
        same_as_next = k < len(blocks) - 1 and blocks[k + 1][0] == display_name

        # Multirow over the H_PS, seed and IQR rows of this block and of any following blocks
        # with the same name (plus their header rows)
        latex_lines.append(f"    % {left_env} | {right_env}")
        if not same_as_prev:
            span, j = 0, k
            while j < len(blocks) and blocks[j][0] == display_name:
                span += num_seeds_of(blocks[j][1]) + 2
                if j > k and blocks[j][3] != blocks[j - 1][3]:
                    span += 1
                j += 1
            latex_lines.append(
                r"    \multirow{"
                + str(span)
                + r"}{*}{{\shortstack[c]{"
                + display_name
                + r"}}}"
            )
        latex_lines.append(
            r"    & $\text{H}_\text{PS}$ & \multicolumn{"
            + str(n_left)
            + r"}{c}{"
            + h_strs[0]
            + r"} & \multicolumn{"
            + str(n_right)
            + r"}{c}{"
            + h_strs[1]
            + r"} \\"
        )
        latex_lines.append(f"    \\cmidrule(lr){{2-{last_col}}}")

        for seed_idx in range(num_seeds):
            cells = []
            for side, algs in zip(sides, side_algorithms, strict=True):
                for alg in algs:
                    if alg in side and seed_idx < len(side[alg]):
                        cells.append(f"{side[alg][seed_idx]['mean']:+.2f}")
                    else:
                        cells.append("--")
            latex_lines.append(
                f"    & {seed_idx + 1:<2} & " + " & ".join(cells) + r" \\"
            )

        latex_lines.append(f"    \\cmidrule(lr){{2-{last_col}}}")

        iqr_cells = []
        for side, algs in zip(sides, side_algorithms, strict=True):
            for alg in algs:
                if alg in side and len(side[alg]) > 0:
                    means = [r["mean"] for r in side[alg]]
                    iqr = np.percentile(means, 75) - np.percentile(means, 25)
                    iqr_cells.append(f"\\textbf{{{iqr:.2f}}}")
                else:
                    iqr_cells.append("--")
        latex_lines.append(r"    & \textbf{IQR} & " + " & ".join(iqr_cells) + r" \\")

        # Rule between blocks with the same gap above and below, as in the main table
        if same_as_next:
            latex_lines.append(r"    \addlinespace[2pt]")
            latex_lines.append(f"    \\cmidrule(lr){{2-{last_col}}}")
            latex_lines.append(r"    \addlinespace[2pt]")
        elif k != len(blocks) - 1:
            latex_lines.append(r"    \addlinespace[2pt]")
            latex_lines.append(r"    \midrule")
            latex_lines.append(r"    \addlinespace[2pt]")
        else:
            latex_lines.append(r"    \bottomrule")

    latex_lines.append(r"  \end{tabular}")

    if caption is not None:
        latex_lines = wrap_table(latex_lines, caption, label, "h", arraystretch=1.15)

    os.makedirs(TABLES_DIR, exist_ok=True)
    output_path = os.path.join(TABLES_DIR, output_filename)
    with open(output_path, "w") as f:
        f.write("\n".join(latex_lines))
    print(f"Per-seed table saved to {output_path}")


def generate_heuristic_comparison_table(
    blocks, output_filename, caption=None, label=None
):
    """
    Generates LaTeX table comparing the heuristics' expected return (10^9) with the standard
    deviation in parentheses, one row per budget scenario; the best heuristic per row is bold.

    blocks: list of (display name, [(env, budget label), ...]), one Environment cell per block.
    """
    with open(HEURISTIC_RESULTS_FILE) as f:
        heuristic_results = yaml.load(f, Loader=yaml.FullLoader)

    heuristic_order = [
        "do_nothing",
        "random",
        "always_repair",
        "segment_based_heuristic",
    ]
    heuristic_headers = [
        r"\thead{\textbf{Always routine} \\ \textbf{inspections}}",
        r"\thead{\textbf{Random} \\ \textbf{actions}}",
        r"\thead{\textbf{Always minor} \\ \textbf{repair}}",
        r"\thead{\textbf{Prioritized} \\ \textbf{segment-based} \\ \textbf{($\text{H}_\text{PS}$)}}",
    ]

    latex_lines = []
    latex_lines.append(r"  \begin{tabular}{@{}cl" + "r" * len(heuristic_order) + "@{}}")
    latex_lines.append(r"    \toprule")
    latex_lines.append(
        r"    \thead{\textbf{Environment}} & \thead{\textbf{Budget}} & "
        + " & ".join(heuristic_headers)
        + r" \\"
    )
    latex_lines.append(r"    \midrule")

    for k, (display_name, rows) in enumerate(blocks):
        latex_lines.append(
            r"    \multirow{"
            + str(len(rows))
            + r"}{*}{{\shortstack[c]{"
            + display_name
            + r"}}}"
        )
        for env, budget_label in rows:
            results = heuristic_results[env]
            best = max(heuristic_order, key=lambda h: results[h]["mean_reward"])
            cells = []
            for h in heuristic_order:
                mean = f"{results[h]['mean_reward'] / 1e9:.2f}"
                mean = f"$\\mathbf{{{mean}}}$" if h == best else f"${mean}$"
                cells.append(f"{mean} ({results[h]['std_reward'] / 1e9:.2f})")
            latex_lines.append(f"    & {budget_label} & " + " & ".join(cells) + r" \\")
        if k != len(blocks) - 1:
            latex_lines.append(r"    \midrule")
            latex_lines.append(r"    \addlinespace[4pt]")
        else:
            latex_lines.append(r"    \bottomrule")

    latex_lines.append(r"  \end{tabular}")

    if caption is not None:
        latex_lines = wrap_table(latex_lines, caption, label, "h", arraystretch=1.2)

    os.makedirs(TABLES_DIR, exist_ok=True)
    output_path = os.path.join(TABLES_DIR, output_filename)
    with open(output_path, "w") as f:
        f.write("\n".join(latex_lines))
    print(f"Heuristic comparison table saved to {output_path}")


def generate_heuristic_parameters_table(
    blocks, output_filename="heuristic_parameters.tex", caption=None, label=None
):
    """
    Generates LaTeX table of the best segment_based_heuristic's parameters, laid out like the
    heuristic comparison table (same blocks: one Environment cell, one row per budget scenario).
    """
    with open(HEURISTIC_RESULTS_FILE) as f:
        heuristic_results = yaml.load(f, Loader=yaml.FullLoader)

    prioritization_names = {
        ("volumes", "positive"): "highest base traffic volume",
        ("volumes", "negative"): "lowest base traffic volume",
        ("segment_lengths", "positive"): "highest segment length",
        ("segment_lengths", "negative"): "lowest segment length",
        ("costs", "positive"): "highest action cost",
        ("costs", "negative"): "lowest action cost",
    }

    latex_lines = []
    latex_lines.append(r"  \begin{tabular}{@{}clcclc@{}}")
    latex_lines.append(r"    \toprule")
    latex_lines.append(
        r"    \thead{\textbf{Environment}} & \thead{\textbf{Budget}} & \thead{$T^{\text{insp}}$} & "
        r"\thead{$i_{\text{thres}}$} & \thead{\textbf{Prioritization} $\xi$} & \thead{$k$} \\"
    )
    latex_lines.append(r"    \midrule")

    for b_idx, (display_name, rows) in enumerate(blocks):
        latex_lines.append(
            r"    \multirow{"
            + str(len(rows))
            + r"}{*}{{\shortstack[c]{"
            + display_name
            + r"}}}"
        )
        for env, budget_label in rows:
            params = heuristic_results[env]["segment_based_heuristic"]["parameters"]
            if params.get("prioritization_enabled", False):
                key = (
                    "costs"
                    if params.get("prioritization_key") == "cost"
                    else params.get("prioritization_key")
                )
                sign = params.get("prioritization_sign")
                xi = prioritization_names.get(
                    (key, sign), f"{sign} {key}".replace("_", " ")
                )
                k = str(params.get("top_k", "--"))
            else:
                xi, k = "--", "--"
            latex_lines.append(
                f"    & {budget_label} & {params.get('inspection_interval', '--')} & "
                f"{params.get('repair_threshold', '--')} & {xi} & {k} \\\\"
            )
        if b_idx != len(blocks) - 1:
            latex_lines.append(r"    \midrule")
            latex_lines.append(r"    \addlinespace[4pt]")
        else:
            latex_lines.append(r"    \bottomrule")

    latex_lines.append(r"  \end{tabular}")

    if caption is not None:
        latex_lines = wrap_table(latex_lines, caption, label, "h", arraystretch=1.2)

    os.makedirs(TABLES_DIR, exist_ok=True)
    output_path = os.path.join(TABLES_DIR, output_filename)
    with open(output_path, "w") as f:
        f.write("\n".join(latex_lines))
    print(f"Heuristic parameters table saved to {output_path}")


def format_lr_latex(lr):
    mantissa, exponent = f"{lr:.0e}".split("e")
    power = f"10^{{{int(exponent)}}}"
    return f"${power}$" if mantissa == "1" else f"${mantissa}\\times{power}$"


def generate_vdn_cbd_configs_table(
    env="CologneBonnDusseldorf-v1-unconstrained",
    output_filename="vdn_cbd_unconstrained_configs.tex",
):
    """Hyperparameters of the VDN configurations in the unconstrained CologneBonnDusseldorf
    appendix figure, read from the run configs. Only a tabular: the appendix sets it next to
    the figure with \\captionof. "--" marks a value equal to rank 1 (bold)."""
    columns = {
        r"\thead{LR}": lambda c: format_lr_latex(c["LR"]),
        r"\thead{Hidden\\size}": lambda c: str(c["HIDDEN_SIZE"]),
        r"\thead{Target\\update}": lambda c: str(c["TARGET_UPDATE_INTERVAL"]),
        # Environment steps over which epsilon decays
        r"\thead{Exploration\\steps}": lambda c: f"{c['EPS_DECAY'] * c['TOTAL_TIMESTEPS'] / 1e6:g}M",
        r"\thead{Steps}": lambda c: f"{c['TOTAL_TIMESTEPS'] / 1e6:g}M",
    }
    configs = {}
    for label, cfg, _ in iter_vdn_runs(env):
        values = [fmt(cfg) for fmt in columns.values()]
        if configs.setdefault(label, values) != values:
            raise ValueError(f"Seeds of {label} on {env} differ in hyperparameters")
    reference = configs["Rank 1"]

    latex_lines = [
        r"\begin{tabular}{l" + "c" * len(columns) + "}",
        r"    \toprule",
        r"    \thead{Configuration} & " + " & ".join(columns) + r" \\",
        r"    \midrule",
    ]
    for label in VDN_CBD_ORDER:
        if label not in configs:
            continue
        if label == "Rank 1":
            cells = [r"\textbf{Rank 1}"] + [
                r"$\mathbf{" + v.strip("$") + "}$" if v.startswith("$") else r"\textbf{" + v + "}"
                for v in reference
            ]
        else:
            # Single changes start a new block after the next-best ranks
            if label == "LR only":
                latex_lines.append(r"    \midrule")
            cells = [label] + [
                "--" if v == ref else v for v, ref in zip(configs[label], reference)
            ]
        latex_lines.append("    " + " & ".join(cells) + r" \\")
    latex_lines += [r"    \bottomrule", r"\end{tabular}"]

    os.makedirs(TABLES_DIR, exist_ok=True)
    output_path = os.path.join(TABLES_DIR, output_filename)
    with open(output_path, "w") as f:
        f.write("\n".join(latex_lines) + "\n")
    print(f"VDN configurations table saved to {output_path}")


def main():
    data, baseline_heuristic_values = load_data()

    # Appendix: VDN hyperparameter sensitivity, next to Figure_VDN_CBD_Unconstrained
    generate_vdn_cbd_configs_table()

    environments = data["map_name"].unique()
    best_checkpoints_all = {
        env: get_best_checkpoints(
            data[data["map_name"] == env], env, baseline_heuristic_values
        )
        for env in environments
    }

    # Original table
    generate_latex_table(best_checkpoints_all, baseline_heuristic_values)

    # Main results: constrained and unconstrained per map, algorithms in columns
    generate_latex_table_scenarios(
        data,
        baseline_heuristic_values,
        blocks=[
            (
                r"\textbf{ToyExample}",
                [
                    ("ToyExample-v2", "Constrained"),
                    ("ToyExample-v2-unconstrained", "Unconstrained"),
                ],
            ),
            # Cologne with all budget scenarios, constrained first as in the other blocks
            (
                r"\textbf{Cologne}",
                [
                    ("Cologne-v1-moderate-budget", "Moderate (300M)"),
                    ("Cologne-v1", "Limited (200M)"),
                    ("Cologne-v1-critical-budget", "Critical (150M)"),
                    ("Cologne-v1-unconstrained", "Unconstrained"),
                ],
            ),
            (
                ENV_DISPLAY_NAMES["CologneBonnDusseldorf-v1"],
                [
                    ("CologneBonnDusseldorf-v1", "Constrained"),
                    ("CologneBonnDusseldorf-v1-unconstrained", "Unconstrained"),
                ],
            ),
        ],
        output_filename="best_results_main.tex",
        caption=(
            r"Normalized expected return ($\Delta \bar{R}^\pi_0\%$) with 95\% confidence "
            r"intervals (CI) for each algorithm and environment configuration, reporting the "
            r"highest value achieved across all training seeds and evaluation checkpoints. The "
            r"algorithm with the highest expected return in each environment is highlighted in "
            r"bold, while algorithms whose 95\% CI overlaps with that of the highest-performing "
            r"algorithm are underlined. $\text{H}_\text{PS}$ is the reference (0\%); its expected "
            r"return is given under each scenario. Note that $\text{VDN}_{\text{BA}}$ is a modified VDN "
            r"policy described in Section~\ref{subsec:open_challenge}."
        ),
        label="tab:best_model_returns",
        placement="!t",
    )

    # Best checkpoint per seed, for the per-seed tables below
    per_seed_results_all = {
        env: get_per_seed_results(
            data[data["map_name"] == env], env, baseline_heuristic_values
        )
        for env in environments
    }
    # Cologne budget scenarios in Figure 4's order, all seeds (the best seed is in Table 1)
    generate_latex_table_per_seed(
        per_seed_results_all,
        baseline_heuristic_values,
        blocks=[
            (
                r"\textbf{Cologne}",
                "Cologne-v1-unconstrained",
                "Cologne-v1-moderate-budget",
                ("Unconstrained", "Moderate (300M)"),
            ),
            (
                r"\textbf{Cologne}",
                "Cologne-v1",
                "Cologne-v1-critical-budget",
                ("Limited (200M)", "Critical (150M)"),
            ),
        ],
        output_filename="best_model_cologne_budget_scenarios.tex",
        caption=(
            r"Best checkpoint performance per seed according to normalized expected return "
            r"($\Delta \bar{R}^\pi_0$ \%) and IQR, by algorithm, for different \texttt{Cologne} "
            r"budget scenarios. $\text{H}_\text{PS}$ is the reference (0\%) and is given "
            r"with its 95\% CI; its expected return is shown in grey. Within each column, seeds are ranked by performance, so a row does not "
            r"correspond to the same seed across algorithms. $\text{VDN}_{\text{BA}}$ is a modified "
            r"VDN policy discussed in Section~\ref{subsec:open_challenge}."
        ),
        label="tab:best_model_cologne_budget_scenarios",
    )

    # Per-seed results table: every map once, two scenarios side by side
    constrained_vs_unconstrained = ("Constrained", "Unconstrained")
    generate_latex_table_per_seed(
        per_seed_results_all,
        baseline_heuristic_values,
        blocks=[
            (
                r"\textbf{ToyExample}",
                "ToyExample-v2",
                "ToyExample-v2-unconstrained",
                constrained_vs_unconstrained,
            ),
            (
                r"\textbf{Cologne}",
                "Cologne-v1",
                "Cologne-v1-unconstrained",
                constrained_vs_unconstrained,
            ),
            (
                ENV_DISPLAY_NAMES["CologneBonnDusseldorf-v1"],
                "CologneBonnDusseldorf-v1",
                "CologneBonnDusseldorf-v1-unconstrained",
                constrained_vs_unconstrained,
            ),
        ],
        output_filename="best_checkpoints_all.tex",
        caption=(
            r"Best checkpoint performance per seed according to normalized expected return "
            r"($\Delta \bar{R}^\pi_0$ \%) and IQR, by environment and algorithm. Constrained "
            r"and unconstrained scenarios are shown side by side. $\text{H}_\text{PS}$ is the reference "
            r"(0\%) and is given with its 95\% CI; its expected return is shown in grey. Within each column, seeds are "
            r"ranked by performance, so a row does not correspond to the same seed across "
            r"algorithms."
        ),
        label="tab:best_checkpoints",
    )

    # Heuristic tables: every map once, one row per budget scenario
    heuristic_blocks = [
        (
            r"\textbf{ToyExample}",
            [
                ("ToyExample-v2-unconstrained", "unconstrained"),
                ("ToyExample-v2", "constrained"),
            ],
        ),
        (
            r"\textbf{Cologne}",
            [
                ("Cologne-v1-unconstrained", "unconstrained"),
                ("Cologne-v1-moderate-budget", "moderate (300M)"),
                ("Cologne-v1", "limited (200M)"),
                ("Cologne-v1-critical-budget", "critical (150M)"),
            ],
        ),
        (
            ENV_DISPLAY_NAMES["CologneBonnDusseldorf-v1"],
            [
                ("CologneBonnDusseldorf-v1-unconstrained", "unconstrained"),
                ("CologneBonnDusseldorf-v1", "constrained"),
            ],
        ),
    ]
    generate_heuristic_comparison_table(
        blocks=heuristic_blocks,
        output_filename="heuristic_results_all.tex",
        caption=(
            r"Expected return ($10^9$) of the expert-based heuristic policies, with standard deviation "
            r"in parentheses. The best heuristic per environment is in bold. Always routine inspections "
            r"is unaffected by the budget; always minor repair beats it only when the budget truncates "
            r"its over-spending."
        ),
        label="tab:heuristic_results",
    )

    generate_heuristic_parameters_table(
        blocks=heuristic_blocks,
        caption=(
            r"Parameters of the best prioritized segment-based heuristic ($\text{H}_\text{PS}$) per "
            r"environment and budget: inspection interval $T^{\text{insp}}$, repair threshold "
            r"$i_{\text{thres}}$, prioritization criterion $\xi$ and number of prioritized segments $k$. "
            r"Without a budget, no prioritization is used (--)."
        ),
        label="tab:heuristic_parameters",
    )


if __name__ == "__main__":
    main()
