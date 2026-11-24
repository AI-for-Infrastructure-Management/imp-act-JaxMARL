"""Build the LaTeX tables of the paper, written to tables/.

Results (Tables 1, 20-24): best checkpoint returns relative to the heuristic H_PS, per
seed and per scenario, and the heuristic baselines, from results.csv, heuristics.csv and
vdn_variants.csv.

Hyperparameters (Tables 14-18), straight from the repo's YAML configs. For each algorithm
(VDN, QMIX, PQN-VDN, MAPPO, IPPO) this pulls together three stages of a config -- the
convergence pilot, the Cologne-v1 tuning-sweep search space, and the selected final-run
config -- into one four-column table (Hyperparameter | Pilot | Search space | Selected),
written as a LaTeX booktabs table to tables/Table_14_18_Hyperparameters.tex. No number is
hardcoded here, so re-running this after a config change regenerates correct tables. The
console preview shows the same values as the .tex file, rendered as readable Unicode math
(e.g. "5 x 10^-5") instead of the raw LaTeX source.

Rollout speedup (Table 12), from the timings in common.py.

Compute usage (Table 19). Reads data/compute_usage.csv, which is released on Zenodo and
downloaded when missing -- this script never talks to wandb. The CSV is exported by the
local nb_compute_accounting.ipynb, the only place that does.

The CSV already holds the filtered WORKING SET, not every raw fetched run: the notebook
keeps only the 50M-timestep budget for VDN/QMIX final runs (their 20M runs are an
earlier, superseded budget -- tuning keeps its own 20M budget untouched), and only
finished runs. This script just loads it and computes GPU hours = wall-clock runtime
(summary["_runtime"], fetched by the notebook) x GPU count, summed per cell.

The checks and the table always print to the console as plain markdown -- there is no
flag that silences this, and the raw LaTeX itself is never dumped to the console. The
LaTeX is written to tables/Table_19_Compute_Usage.tex.

The released data files are downloaded from Zenodo when missing.

Usage:
    python results/create_tables.py
    python results/create_tables.py --only results
    python results/create_tables.py --only hyperparameters --show none
    python results/create_tables.py --only compute --csv path/to/other.csv
"""

import argparse
import glob
import json
import re
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from common import (
    ALGORITHM_NAMES,
    BASELINE_HEURISTIC,
    REPO_ROOT,
    ROLLOUT_TIMES,
    TABLES_DIR,
    baseline_values,
    best_per_run,
    data_file,
    heuristics,
    load_results,
    normalize,
)
from create_figures import VDN_CBD_ORDER


def render_table(df):
    """Markdown table for the console (plain text if tabulate is missing)."""
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_string(index=False)


# ===========================================================================
# Results tables (Tables 1, 20-24)
# ===========================================================================
LATEX_ALGORITHM_NAMES = {**ALGORITHM_NAMES, "vdn_ba_rnn": r"$\text{VDN}_{\text{BA}}$"}

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


def format_heuristic_value(val):
    """Formats heuristic value for display."""
    if abs(val) >= 1e9:
        return f"{val/1e9:.2f}\\text{{B}}"
    return f"{val/1e6:.2f}\\text{{M}}"


def paper_repair_threshold(params, num_damage_states=5):
    """Repair threshold in the paper's numbering (5 = very good, ..., 1 = poor).

    The heuristic repairs when the observed index (0 = very good) is >= the stored threshold,
    i.e. when the paper's damage state is <= num_damage_states - threshold.
    """
    if "repair_threshold" not in params:
        return "--"
    return num_damage_states - int(params["repair_threshold"])


def wrap_table(latex_lines, caption, label, placement, arraystretch=1.25, fill_width=False):
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
        (
            r"  \resizebox{\textwidth}{!}{%"
            if fill_width
            else r"  \resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%"
        ),
        *latex_lines,
        r"  }",
        r"\end{table}",
    ]


# Extra algorithms: shown in the main results table with an asterisk, but not considered
# when marking the best result and the results tied with it
EXTRA_ALGORITHMS = {"vdn_ba_rnn"}


def best_model_stats(data, baseline_heuristic_values, env):
    """
    Best checkpoint across all seeds per algorithm on env, as % vs H_PS, with that checkpoint's
    own 95% CI. H_PS is best if no algorithm beats it; results whose CI overlaps the best
    one's are marked tied. EXTRA_ALGORITHMS are never best or tied.
    """
    h = heuristics().loc[(env, BASELINE_HEURISTIC)]
    h_mean = baseline_heuristic_values[env]

    def to_pct(v):
        return normalize(v, h_mean)

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
    env_header="Environment",
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

    latex_lines = []
    latex_lines.append(
        r"  \begin{tabular}{@{}cl" + "c" * (1 + len(algorithms)) + "@{}}"
    )
    latex_lines.append(r"    \toprule")
    latex_lines.append(
        r"    \thead{\textbf{" + env_header + r"}} & \thead{\textbf{"
        + scenario_header
        + r"}} & \thead{\textbf{$\text{H}_\text{PS}$}} & "
        + " & ".join(
            r"\thead{\textbf{"
            + LATEX_ALGORITHM_NAMES[alg]
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
                data, baseline_heuristic_values, env
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

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TABLES_DIR / output_filename
    with open(output_path, "w") as f:
        f.write("\n".join(latex_lines))
    print(f"Table saved to {output_path}")


def get_per_seed_results(data, env_name):
    """
    Best checkpoint of every seed per algorithm, as % over H_PS, sorted best first.
    Returns a dict: {algorithm: [{seed: int, mean: float}, ...]}
    """
    return {
        alg: sorted(
            ({"seed": i + 1, "mean": mean} for i, mean in enumerate(runs["mean_pct"])),
            key=lambda x: x["mean"],
            reverse=True,
        )
        for alg, runs in best_per_run(data, env_name).groupby("algorithm", sort=False)
    }


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
            r"\thead{\textbf{" + LATEX_ALGORITHM_NAMES[alg] + "}}" for alg in algs
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
            h = heuristics().loc[(env, BASELINE_HEURISTIC)]
            lo, hi = (normalize(v, h_mean) for v in (h["lower_95ci"], h["upper_95ci"]))
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

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TABLES_DIR / output_filename
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
            results = heuristics().loc[env]
            best = results.loc[heuristic_order, "mean_reward"].idxmax()
            cells = []
            for h in heuristic_order:
                mean = f"{results.loc[h, 'mean_reward'] / 1e9:.2f}"
                mean = f"$\\mathbf{{{mean}}}$" if h == best else f"${mean}$"
                cells.append(f"{mean} ({results.loc[h, 'std_reward'] / 1e9:.2f})")
            latex_lines.append(f"    & {budget_label} & " + " & ".join(cells) + r" \\")
        if k != len(blocks) - 1:
            latex_lines.append(r"    \midrule")
            latex_lines.append(r"    \addlinespace[4pt]")
        else:
            latex_lines.append(r"    \bottomrule")

    latex_lines.append(r"  \end{tabular}")

    if caption is not None:
        latex_lines = wrap_table(latex_lines, caption, label, "h", arraystretch=1.2, fill_width=True)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TABLES_DIR / output_filename
    with open(output_path, "w") as f:
        f.write("\n".join(latex_lines))
    print(f"Heuristic comparison table saved to {output_path}")


def generate_heuristic_parameters_table(
    blocks, output_filename="Table_24_Heuristic_Parameters.tex", caption=None, label=None
):
    """
    Generates LaTeX table of the best segment_based_heuristic's parameters, laid out like the
    heuristic comparison table (same blocks: one Environment cell, one row per budget scenario).
    """
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
            params = json.loads(heuristics().loc[(env, BASELINE_HEURISTIC), "parameters"])
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
                f"{paper_repair_threshold(params)} & {xi} & {k} \\\\"
            )
        if b_idx != len(blocks) - 1:
            latex_lines.append(r"    \midrule")
            latex_lines.append(r"    \addlinespace[4pt]")
        else:
            latex_lines.append(r"    \bottomrule")

    latex_lines.append(r"  \end{tabular}")

    if caption is not None:
        latex_lines = wrap_table(latex_lines, caption, label, "h", arraystretch=1.2)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TABLES_DIR / output_filename
    with open(output_path, "w") as f:
        f.write("\n".join(latex_lines))
    print(f"Heuristic parameters table saved to {output_path}")


def format_lr_latex(lr):
    mantissa, exponent = f"{lr:.0e}".split("e")
    power = f"10^{{{int(exponent)}}}"
    return f"${power}$" if mantissa == "1" else f"${mantissa}\\times{power}$"


def generate_vdn_cbd_configs_table(
    env="CologneBonnDusseldorf-v1-unconstrained",
    output_filename="Table_22_VDN_CBD_Unconstrained_Configs.tex",
):
    """Hyperparameters of the VDN configurations in the unconstrained CologneBonnDusseldorf
    appendix figure, read from vdn_variants.csv. Only a tabular: the appendix sets it next to
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
    runs = pd.read_csv(data_file("vdn_variants.csv"))
    for cfg in runs[runs["map_name"] == env].to_dict("records"):
        label = cfg["configuration"]
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

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TABLES_DIR / output_filename
    with open(output_path, "w") as f:
        f.write("\n".join(latex_lines) + "\n")
    print(f"VDN configurations table saved to {output_path}")


def write_results_tables():
    """Tables 1 and 20-24, from results.csv, heuristics.csv and vdn_variants.csv."""
    data = load_results()
    baseline_heuristic_values = baseline_values()

    # Appendix: VDN hyperparameter sensitivity, next to Figure_VDN_CBD_Unconstrained
    generate_vdn_cbd_configs_table()

    environments = data["map_name"].unique()

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
        output_filename="Table_1_Best_Results.tex",
        env_header="Map",
        scenario_header="Environment",
        caption=(
            r"Normalized expected return ($\Delta \bar{R}^\pi_0\%$) with 95\% confidence "
            r"intervals (CI) for each algorithm and environment configuration, reporting the "
            r"highest value achieved across all training seeds and evaluation checkpoints. The "
            r"algorithm with the highest expected return in each environment is highlighted in "
            r"bold, while algorithms whose 95\% CI overlaps with that of the highest-performing "
            r"algorithm are underlined. $\text{H}_\text{PS}$ is the reference (0\%); its expected "
            r"return is given under each scenario. $\text{VDN}_{\text{BA}}$$^*$ is a modified VDN "
            r"policy described in Section~\ref{subsec:open_challenge}."
        ),
        label="tab:best_model_returns",
        placement="!t",
    )

    # Best checkpoint per seed, for the per-seed tables below
    per_seed_results_all = {env: get_per_seed_results(data, env) for env in environments}
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
        output_filename="Table_21_Best_Checkpoints_Cologne_Budgets.tex",
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
        output_filename="Table_20_Best_Checkpoints_Envs.tex",
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
        output_filename="Table_23_Heuristic_Results.tex",
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
            r"$i_{\text{thres}}$ (Equation~\ref{eq:insp_based_minrep_insp_dn}), prioritization criterion $\xi$ and number of prioritized segments $k$. "
            r"Without a budget, no prioritization is used (--)."
        ),
        label="tab:heuristic_parameters",
    )


# ===========================================================================
# Hyperparameter tables (Tables 14-18)
# ===========================================================================
# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------
# PyYAML's default float resolver doesn't recognise unquoted scientific
# notation without a decimal point (e.g. "1e-4"), so it leaves those as
# strings instead of floats. We patch the resolver so every numeric literal
# in the configs comes back as a real number (underscored ints like
# 50_000_000 already work out of the box).
class ConfigLoader(yaml.SafeLoader):
    pass


_FLOAT_RE = (
    r"^[-+]?(?:[0-9][0-9_]*\.[0-9_]*(?:[eE][-+]?[0-9]+)?"
    r"|\.[0-9_]+(?:[eE][-+]?[0-9]+)?"
    r"|[0-9][0-9_]*[eE][-+]?[0-9]+"
    r"|\.(?:inf|Inf|INF)|nan|NaN|NAN)$"
)
ConfigLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float", re.compile(_FLOAT_RE), list("-+0123456789.")
)


def load_yaml(path):
    with open(path) as f:
        return yaml.load(f, Loader=ConfigLoader)


CFG_DIR = REPO_ROOT / "experiments" / "config"
PILOT_DIR = CFG_DIR / "pilot_runs"
SWEEP_DIR = CFG_DIR / "hyper_parameter_tuning" / "cologne_v1"
FINAL_DIR = CFG_DIR / "final_runs"
HYPERPARAMETER_LATEX_PATH = TABLES_DIR / "Table_14_18_Hyperparameters.tex"


# ---------------------------------------------------------------------------
# What counts as a hyperparameter
# ---------------------------------------------------------------------------
# Bookkeeping / infra keys that show up in the configs but aren't actually
# hyperparameters, so they don't get a row.
EXCLUDED_EXACT = {
    "ENV_NAME",
    "ALG_NAME",
    "SEED",
    "NUM_SEEDS",
    "HYP_TUNE",
    "ENTITY",
    "PROJECT",
    "DOUBLE_PRECISION_MODE",
    "hydra",
    "command",
    "program",
    "method",
    "metric",
    "name",
    "SWEEP_SEEDS",
    "STORE_EVAL_RETURNS",
    "ENV_KWARGS.map_name",
}
EXCLUDED_PREFIXES = ("WANDB_", "SAVE_CHECKPOINTS", "TEST_")


def is_excluded(key):
    return key in EXCLUDED_EXACT or key.startswith(EXCLUDED_PREFIXES)


# EPS_DECAY is stored as a fraction of updates, but the pilot/sweep/final
# stages don't share a training budget, so the raw fractions aren't
# comparable across columns. We convert it to absolute env steps instead.
FRACTION_OF_UPDATES_KEYS = {"EPS_DECAY"}


def label_for(key):
    if key in FRACTION_OF_UPDATES_KEYS:
        return key + " (env. steps)"
    return key


# PQN-VDN goes last, and QMIX/IPPO each close out a page (see write_hyperparameter_latex).
ALG_ORDER = ["vdn_rnn", "qmix_rnn", "mappo_rnn", "ippo_rnn", "pqn_rnn"]
CLEARPAGE_AFTER = {"ippo_rnn"}
DASH = "—"


def slug(alg):
    return alg.replace("_rnn", "")


def flatten_config(cfg):
    """Turn the nested ENV_KWARGS block into dotted keys, everything else stays flat."""
    flat = OrderedDict()
    for k, v in cfg.items():
        if k == "ENV_KWARGS" and isinstance(v, dict):
            for kk, vv in v.items():
                if kk == "include_extra_observations" and isinstance(vv, dict):
                    for kkk, vvv in vv.items():
                        flat["ENV_KWARGS.include_extra_observations." + kkk] = vvv
                else:
                    flat["ENV_KWARGS." + kk] = vv
        else:
            flat[k] = v
    return flat


# ---------------------------------------------------------------------------
# Number formatting -- always LaTeX math, since the console/text preview is
# meant to show exactly what ends up in the .tex file, not a simplified form.
# ---------------------------------------------------------------------------
def _fmt_sci(v):
    """Normalized scientific notation as inline math, e.g. 5e-05 -> "$5 \\times 10^{-5}$"."""
    mant, exp = f"{float(v):.6e}".split("e")
    mant = mant.rstrip("0").rstrip(".")
    exp = int(exp)
    return rf"${mant} \times 10^{{{exp}}}$"


def _fmt_millions(v):
    """Large step counts on a fixed 1e6 base so they read as millions: "$50 \\times 10^{6}$"."""
    mant = f"{float(v) / 1e6:.6f}".rstrip("0").rstrip(".")
    return rf"${mant} \times 10^{{6}}$"


def fmt_value(v):
    # Every value >= 1000 gets scientific notation (budgets keep the 1e6
    # "millions" base), and so do small values below 1e-2; plain counts
    # below 1000 (hidden size, num envs, epochs, ...) stay as-is.
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, int):
        if abs(v) >= 1_000_000:
            return _fmt_millions(v)
        if abs(v) >= 1000:
            return _fmt_sci(v)
        return str(v)
    if isinstance(v, float):
        if v == 0:
            return "0"
        if abs(v) >= 1_000_000:
            return _fmt_millions(v)
        if abs(v) >= 1000 or abs(v) < 1e-2:
            return _fmt_sci(v)
        return f"{v:.6f}".rstrip("0").rstrip(".")
    return str(v)


def _math_safe(part):
    return (part.startswith("$") and part.endswith("$")) or re.match(
        r"^-?[0-9][0-9.]*$", part
    ) is not None


def _mathify(part):
    return part[1:-1] if part.startswith("$") and part.endswith("$") else part


def fmt_list(key, vals, tt):
    if key in FRACTION_OF_UPDATES_KEYS and tt is not None:
        parts = [_fmt_millions(float(x) * float(tt)) for x in vals]
    else:
        parts = [fmt_value(x) for x in vals]
    # A plain "[a, b]" doesn't size its brackets to tall 10^x superscripts,
    # so swept lists that contain that kind of math get wrapped as one
    # \left[ ... \right] expression instead.
    if all(_math_safe(p) for p in parts) and any("times" in p for p in parts):
        joined = ",\\ ".join(_mathify(p) for p in parts)
        return rf"$\left[ {joined} \right]$"
    return "[" + ", ".join(parts) + "]"


_LATEX_TEXT_MAP = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_"}


def latex_escape(s):
    """Escape LaTeX specials in plain text, but leave inline math ($...$) untouched."""
    s = str(s).replace("—", "---")
    out = []
    for seg in re.split(r"(\$[^$]*\$)", s):
        if len(seg) >= 2 and seg[0] == "$" and seg[-1] == "$":
            out.append(seg)
        else:
            out.append(re.sub(r"[&%#_]", lambda m: _LATEX_TEXT_MAP[m.group(0)], seg))
    return "".join(out)


_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
_TIMES10_RE = re.compile(r"([0-9.]+) \\times 10\^\{(-?[0-9]+)\}")


def _times10_to_unicode(match):
    mant, exp = match.group(1), match.group(2)
    return f"{mant} × 10{exp.translate(_SUPERSCRIPT)}"


def render_math(s):
    """Turn one of our own generated LaTeX cell strings into readable Unicode text.

    The .tex file keeps the raw LaTeX; this is only for the console/text
    preview, where "$5 \\times 10^{-5}$" is a lot harder to read than "5 x 10^-5".
    """
    s = s.replace(r"\left[", "[").replace(r"\right]", "]")
    s = s.replace("[ ", "[").replace(" ]", "]")
    s = _TIMES10_RE.sub(_times10_to_unicode, s)
    s = s.replace("$", "").replace(",\\ ", ", ")
    s = s.replace(r"\qquad ", "    ").replace(r"\quad ", "  ")
    s = (
        s.replace(r"\_", "_")
        .replace(r"\&", "&")
        .replace(r"\%", "%")
        .replace(r"\#", "#")
    )
    return s.replace("---", "—")


# ---------------------------------------------------------------------------
# Discovery -- find the pilot / sweep / base / final config for each algorithm
# ---------------------------------------------------------------------------
def index_by_alg_name(dir_path):
    """Index every non-sweep *.yaml in a directory by its ALG_NAME field."""
    idx = {}
    for p in sorted(glob.glob(str(Path(dir_path) / "*.yaml"))):
        if p.endswith("_sweep.yaml"):
            continue
        cfg = load_yaml(p)
        if isinstance(cfg, dict) and "ALG_NAME" in cfg:
            idx[cfg["ALG_NAME"]] = p
    return idx


def discover():
    """Locate the pilot/base/final configs and sweep files, keyed by ALG_NAME.

    We match by ALG_NAME rather than filename because the filenames aren't
    consistent -- e.g. PQN-VDN's pilot/final files are named "pqn_vdn_*" but
    its sweep file is "pqn_sweep.yaml".
    """
    pilot_idx = index_by_alg_name(PILOT_DIR)
    base_idx = index_by_alg_name(SWEEP_DIR)  # the hydra configs the sweeps overlay
    final_idx = index_by_alg_name(FINAL_DIR)

    sweeps = {}
    for sweep_file in sorted(glob.glob(str(SWEEP_DIR / "*_sweep.yaml"))):
        sweep = load_yaml(sweep_file)
        alg = re.match(r"^([a-z0-9]+_rnn)", sweep.get("name", "")).group(1)
        sweeps[alg] = {"file": sweep_file, "params": sweep.get("parameters", {})}

    missing = [
        a
        for a in ALG_ORDER
        if not (a in pilot_idx and a in base_idx and a in final_idx and a in sweeps)
    ]
    if missing:
        raise RuntimeError(f"Missing pilot/sweep/base/final config for: {missing}")

    print("Found configs for:", ", ".join(ALGORITHM_NAMES[a] for a in ALG_ORDER))
    for a in ALG_ORDER:
        name = ALGORITHM_NAMES[a]
        pilot_rel = Path(pilot_idx[a]).relative_to(REPO_ROOT)
        sweep_rel = Path(sweeps[a]["file"]).relative_to(REPO_ROOT)
        final_rel = Path(final_idx[a]).relative_to(REPO_ROOT)
        print(f"  {name:<9} pilot={pilot_rel} sweep={sweep_rel} final={final_rel}")
    return pilot_idx, base_idx, final_idx, sweeps


def parse_algorithm(a, pilot_idx, base_idx, final_idx, sweeps):
    pilot = flatten_config(load_yaml(pilot_idx[a]))
    base = flatten_config(load_yaml(base_idx[a]))
    final = flatten_config(load_yaml(final_idx[a]))
    params = sweeps[a]["params"]
    # The sweep file sets its own budget for VDN/QMIX/PQN-VDN, but MAPPO/IPPO
    # inherit TOTAL_TIMESTEPS from the base config it overlays.
    if "TOTAL_TIMESTEPS" in params and "value" in params["TOTAL_TIMESTEPS"]:
        sw_tt = params["TOTAL_TIMESTEPS"]["value"]
    else:
        sw_tt = base.get("TOTAL_TIMESTEPS")
    return {
        "pilot": pilot,
        "base": base,
        "final": final,
        "params": params,
        "sw_tt": sw_tt,
        "pilot_tt": pilot.get("TOTAL_TIMESTEPS"),
        "final_tt": final.get("TOTAL_TIMESTEPS"),
    }


# ---------------------------------------------------------------------------
# Row building -- rows are built once, already LaTeX-escaped, and reused
# verbatim for the console preview, the text file, and the .tex file.
# ---------------------------------------------------------------------------
def build_rows(a, data):
    """One table's worth of rows, in the order the final config declares its keys.

    Following the final config's own order (rather than the sweep file's)
    makes it easy to cross-check a row against the file it came from.
    """
    d = data[a]
    pilot, base, final, params = d["pilot"], d["base"], d["final"], d["params"]
    sw_tt, pilot_tt, final_tt = d["sw_tt"], d["pilot_tt"], d["final_tt"]
    env_prefix = "ENV_KWARGS."
    indent_str = {1: r"\quad ", 2: r"\qquad "}

    # QMIX's sweep samples MIXER_EMBEDDING_DIM and MIXER_HYPERNET_HIDDEN_DIM
    # jointly (only two combos are ever tried), so we recover each sub-key's
    # searched set from that coupled group instead of treating them as
    # independently searched.
    coupled = {}
    for pkey, spec in params.items():
        values = spec.get("values") if isinstance(spec, dict) else None
        if (
            pkey.startswith("_")
            and isinstance(values, list)
            and values
            and isinstance(values[0], dict)
        ):
            for sub_key in values[0]:
                coupled[sub_key] = [combo[sub_key] for combo in values]

    def search_cell(k):
        if k in coupled:
            return fmt_list(k, coupled[k], sw_tt)
        spec = params.get(k)
        if spec is None:
            return fmt_value(base[k]) if k in base else DASH
        if "values" in spec:
            return fmt_list(k, spec["values"], sw_tt)
        if "value" in spec:
            v = spec["value"]
            if k in FRACTION_OF_UPDATES_KEYS:
                return _fmt_millions(float(v) * float(sw_tt))
            return fmt_value(v)
        return DASH

    def cell(k, cfg, tt):
        if k not in cfg:
            return DASH
        v = cfg[k]
        if k in FRACTION_OF_UPDATES_KEYS and tt is not None:
            return _fmt_millions(float(v) * float(tt))
        return fmt_value(v)

    def make_row(k, label=None, indent=0):
        text = indent_str.get(indent, "") + (
            label if label is not None else label_for(k)
        )
        return {
            "label": latex_escape(text),
            "pilot": latex_escape(cell(k, pilot, pilot_tt)),
            "search": latex_escape(search_cell(k)),
            "selected": latex_escape(cell(k, final, final_tt)),
        }

    def make_header(label, indent=0):
        return {
            "label": latex_escape(indent_str.get(indent, "") + label),
            "pilot": "",
            "search": "",
            "selected": "",
        }

    rows, placed, env_done = [], set(), [False]

    def emit_env_block():
        env_done[0] = True
        rows.append(make_header("ENV_KWARGS"))
        enc = env_prefix + "encoding_type"
        if any(enc in c for c in (final, base, pilot)):
            rows.append(make_row(enc, "encoding_type", 1))
        inc_prefix = env_prefix + "include_extra_observations."
        inc_keys = [k for k in final if k.startswith(inc_prefix)]
        for src in (base, pilot):
            inc_keys += [
                k for k in src if k.startswith(inc_prefix) and k not in inc_keys
            ]
        if inc_keys:
            rows.append(make_header("include_extra_observations", 1))
            for k in inc_keys:
                rows.append(make_row(k, k.rsplit(".", 1)[1], 2))
        for src in (
            final,
            base,
            pilot,
        ):  # mark every ENV key placed, incl. excluded map_name
            placed.update(k for k in src if k.startswith(env_prefix))

    for k in final:
        if k in placed:
            continue
        if is_excluded(k):
            placed.add(k)
            continue
        if k.startswith(env_prefix):
            emit_env_block()
            continue
        placed.add(k)
        rows.append(make_row(k))

    # Nothing should be silently dropped: pick up anything the sweep/base/
    # pilot mention that the final config happens not to.
    for k in list(params) + list(base) + list(pilot):
        if k in placed:
            continue
        placed.add(k)
        if k.startswith(("_", env_prefix)) or is_excluded(k):
            continue
        rows.append(make_row(k))

    if not env_done[0] and any(
        k.startswith(env_prefix) and not is_excluded(k)
        for src in (base, pilot)
        for k in src
    ):
        emit_env_block()

    return rows


def rows_to_frame(rows):
    return pd.DataFrame(
        [[r["label"], r["pilot"], r["search"], r["selected"]] for r in rows],
        columns=["Hyperparameter", "Pilot", "Search space", "Selected"],
    )


def rows_to_preview_frame(rows):
    """Same rows as rows_to_frame, but with the LaTeX rendered to readable Unicode."""
    return pd.DataFrame(
        [
            [render_math(r[c]) for c in ("label", "pilot", "search", "selected")]
            for r in rows
        ],
        columns=["Hyperparameter", "Pilot", "Search space", "Selected"],
    )


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------
def to_latex(a, rows):
    name = ALGORITHM_NAMES[a]
    caption = (
        rf"{name} hyperparameters: pilot runs, hyperparameter tuning search space "
        r"(on \texttt{Cologne}), and selected final-run configuration."
    )
    tex = rows_to_frame(rows).to_latex(
        index=False,
        escape=False,
        column_format="@{}llll@{}",
        caption=caption,
        label=f"tab:{slug(a)}_hypers",
        position="H",
    )
    tex = tex.replace("\\begin{table}[H]\n", "\\begin{table}[H]\n\\centering\n", 1)
    # `[H]` (float package) plus a compact font keeps these from each landing
    # on their own float page and lets two share a page.
    tex = tex.replace("\\begin{tabular}", "\\footnotesize\n\\begin{tabular}", 1)
    return tex


def write_hyperparameter_latex(all_rows):
    sections = ["% \\usepackage{booktabs,float}"]
    for a in ALG_ORDER:
        tex_body = to_latex(a, all_rows[a]).rstrip("\n")
        if a == ALG_ORDER[0]:
            # The first table may float to the top of the appendix page
            tex_body = tex_body.replace("\\begin{table}[H]", "\\begin{table}[h]", 1)
        sections.append(f"%% {ALGORITHM_NAMES[a]} %%\n{tex_body}")
        if a in CLEARPAGE_AFTER:
            sections.append("\\clearpage")
    HYPERPARAMETER_LATEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    HYPERPARAMETER_LATEX_PATH.write_text("\n\n".join(sections) + "\n")
    print("Wrote", HYPERPARAMETER_LATEX_PATH)


# ---------------------------------------------------------------------------
# Console preview -- same values as the .tex file, rendered readable
# ---------------------------------------------------------------------------
def show_hyperparameter_console(all_rows):
    for a in ALG_ORDER:
        print(f"\n### {ALGORITHM_NAMES[a]} ###")
        print(render_table(rows_to_preview_frame(all_rows[a])))


def write_hyperparameter_tables(show=True):
    """Tables 14-18, from the YAML configs in experiments/config."""
    pilot_idx, base_idx, final_idx, sweeps = discover()
    data = {a: parse_algorithm(a, pilot_idx, base_idx, final_idx, sweeps) for a in ALG_ORDER}
    all_rows = {a: build_rows(a, data) for a in ALG_ORDER}

    if show:
        show_hyperparameter_console(all_rows)

    write_hyperparameter_latex(all_rows)


# ===========================================================================
# Compute usage table (Table 19)
# ===========================================================================
COMPUTE_LATEX_PATH = TABLES_DIR / "Table_19_Compute_Usage.tex"

# ---------------------------------------------------------------------------
# Table layout -- single label mapping per concept, used by both renderings
# ---------------------------------------------------------------------------
ALG_COL_ORDER = [
    "pqn_rnn",
    "vdn_rnn",
    "qmix_rnn",
    "mappo_rnn",
    "ippo_rnn",
]
# Final-run environment configurations, in paper/figure order. Paper display names
# double as the \texttt{} labels in the table.
ENV_ORDER = [
    "ToyExample-v2",
    "ToyExample-v2-unconstrained",
    "Cologne-v1",
    "Cologne-v1-critical-budget",
    "Cologne-v1-moderate-budget",
    "Cologne-v1-unconstrained",
    "CologneBonnDusseldorf-v1",
    "CologneBonnDusseldorf-v1-unconstrained",
]
ENV_PAPER = {
    "ToyExample-v2": "ToyExample",
    "ToyExample-v2-unconstrained": "ToyExample-unconstrained",
    "Cologne-v1": "Cologne",
    "Cologne-v1-critical-budget": "Cologne-critical",
    "Cologne-v1-moderate-budget": "Cologne-moderate",
    "Cologne-v1-unconstrained": "Cologne-unconstrained",
    "CologneBonnDusseldorf-v1": "CologneBonnDusseldorf",
    "CologneBonnDusseldorf-v1-unconstrained": "CologneBonnDusseldorf-unconstrained",
}

# Experimental design shown in each stage's section-header row (per algorithm).
STAGE_DESIGN = {
    "pilot": "3 envs × 3 seeds/alg",
    "tuning": "50 configs × 3 seeds/alg",
    "final": "10 seeds/env/alg",
}
ROW_PILOT, ROW_TUNING = "Pilot runs", "Hyperparameter tuning (Cologne)"

# VDN/QMIX final runs exist at both 20M and 50M budgets; keep the 50M ones only.
KEEP_50M_ALGS = {"vdn_rnn", "qmix_rnn"}
FINAL_TT_KEEP = 50_000_000

# Cell markers -- must render distinctly, they mean different things.
MARK_NORUN = "—"  # em dash: no matching run found

# Internal cell-token sentinels (distinct from the display markers above).
NORUN, BLANK = "NORUN", "BLANK"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_runs(csv_path):
    """Load the CSV (already the filtered working set) and compute GPU hours.

    Re-checks the two filters the notebook applies before writing the CSV, and warns
    rather than silently re-filtering -- a mismatch means the CSV is stale or was hand-edited.
    """
    df = pd.read_csv(csv_path)
    df["runtime_h"] = df["runtime_s"] / 3600.0
    df["gpu_count_eff"] = df["gpu_count"].fillna(1).astype(int)
    df["gpu_hours"] = (
        df["runtime_h"] * df["gpu_count_eff"]
    )  # GPU hours = runtime x count

    not_finished = int((df["state"] != "finished").sum())
    if not_finished:
        print(
            f"!! {not_finished} rows are not 'finished' -- expected the CSV to be pre-filtered"
        )
    bad_budget = int(
        (
            (df["stage"] == "final")
            & df["alg"].isin(KEEP_50M_ALGS)
            & (df["total_timesteps"] != FINAL_TT_KEEP)
        ).sum()
    )
    if bad_budget:
        print(
            f"!! {bad_budget} final VDN/QMIX rows are not at the 50M budget -- "
            "expected the CSV to be pre-filtered"
        )
    print(f"loaded {len(df)} runs from {csv_path}")
    return df


def row_key_of(rec):
    if rec["stage"] == "pilot":
        return ROW_PILOT
    if rec["stage"] == "tuning":
        return ROW_TUNING
    return ENV_PAPER.get(rec["map_name"])  # final -> env row (None if unmapped)


def assign_rows(work):
    work = work.copy()
    work["row_key"] = work.apply(row_key_of, axis=1)
    work["col"] = work["alg"]
    return work


# ---------------------------------------------------------------------------
# Checks (always run -- diagnostics, not gated by --show)
# ---------------------------------------------------------------------------
def check_runs_per_cell(work):
    env_rows = [ENV_PAPER[v] for v in ENV_ORDER]
    row_order = [ROW_PILOT, ROW_TUNING, *env_rows]
    present_rows = [r for r in row_order if r in set(work["row_key"])]
    count_pivot = (
        work[work["col"].isin(ALG_COL_ORDER)]
        .pivot_table(
            index="row_key",
            columns="col",
            values="stage",  # any column: counts the runs per cell
            aggfunc="count",
            fill_value=0,
        )
        .reindex(index=present_rows, columns=ALG_COL_ORDER, fill_value=0)
        .rename(columns=ALGORITHM_NAMES)
    )
    print("\n(1) runs per cell (finished)")
    print(count_pivot.to_string())
    flagged = [
        (env, alg, int(count_pivot.loc[env, alg]))
        for env in env_rows
        if env in count_pivot.index
        for alg in count_pivot.columns
        if count_pivot.loc[env, alg] not in (0, 10)
    ]
    print(
        "\nall present final cells have 10 seeds."
        if not flagged
        else "\nfinal cells != 10 seeds:"
    )
    for env, alg, n in flagged:
        print(f"   FLAG {env} / {alg}: {n} seeds")


def check_device_mix(work):
    print("\n(2) GPU hours by device")
    dev = (
        work.assign(gpu_name=work["gpu_name"].fillna("UNKNOWN"))
        .groupby("gpu_name")["gpu_hours"]
        .sum()
        .sort_values(ascending=False)
    )
    for name, hours_ in dev.items():
        print(f"   {name:<32} {hours_:10.2f} GPU-h")
    print(f"   {'TOTAL':<32} {dev.sum():10.2f} GPU-h")


def show_device_table(work):
    """Device x stage GPU-hour breakdown -- console only, internal knowledge (no .tex)."""
    stage_label = {"pilot": "Pilot", "tuning": "Tuning", "final": "Final"}
    stage_order = ["pilot", "tuning", "final"]
    dev_pivot = work.assign(gpu_name=work["gpu_name"].fillna("UNKNOWN")).pivot_table(
        index="gpu_name",
        columns="stage",
        values="gpu_hours",
        aggfunc="sum",
        fill_value=0.0,
    )
    for s in stage_order:
        if s not in dev_pivot.columns:
            dev_pivot[s] = 0.0
    dev_pivot = dev_pivot[stage_order]
    dev_pivot = dev_pivot.loc[dev_pivot.sum(axis=1).sort_values(ascending=False).index]
    dev_pivot["Total"] = dev_pivot.sum(axis=1)
    dev_pivot.loc["Total"] = dev_pivot.sum(axis=0)
    df = (
        dev_pivot.rename(columns=stage_label)
        .reset_index()
        .rename(columns={"gpu_name": "GPU device"})
    )
    for col in df.columns[1:]:
        df[col] = df[col].map(lambda v: f"{v:.2f}")
    print("\n(3) GPU hours by device x stage (internal; not written to LaTeX)")
    print(render_table(df))


# ---------------------------------------------------------------------------
# Table construction -- shared by the console and LaTeX renderings
# ---------------------------------------------------------------------------
def build_table(work):
    hours, counts = defaultdict(float), defaultdict(int)
    for _, r in work.iterrows():
        if r["col"] in ALG_COL_ORDER and r["row_key"] is not None:
            counts[(r["row_key"], r["col"])] += 1
            if pd.notna(r["gpu_hours"]):
                hours[(r["row_key"], r["col"])] += float(r["gpu_hours"])

    pilot_maps = [
        ENV_PAPER[v]
        for v in ENV_ORDER
        if v in set(work[work["stage"] == "pilot"]["map_name"])
    ]

    def data_cell(row_key, alg):
        if counts[(row_key, alg)] == 0:
            return NORUN
        return hours[(row_key, alg)]

    def row_total(row_key):
        return sum(hours[(row_key, a)] for a in ALG_COL_ORDER)

    groups = [
        ("Pilot runs", "pilot", [(pilot_maps, ROW_PILOT)]),
        ("Hyperparameter tuning", "tuning", [(["Cologne"], ROW_TUNING)]),
        ("Final runs", "final", [([ENV_PAPER[v]], ENV_PAPER[v]) for v in ENV_ORDER]),
    ]
    env_rows = [ENV_PAPER[v] for v in ENV_ORDER]
    col_tot = {
        a: sum(hours[(rk, a)] for rk in [ROW_PILOT, ROW_TUNING, *env_rows])
        for a in ALG_COL_ORDER
    }
    grand_total = sum(col_tot.values())
    final_total = work[work["stage"] == "final"]["gpu_hours"].sum()
    return {
        "groups": groups,
        "data_cell": data_cell,
        "row_total": row_total,
        "col_tot": col_tot,
        "grand_total": grand_total,
        "final_total": final_total,
    }


# ---------------------------------------------------------------------------
# Console / text rendering
# ---------------------------------------------------------------------------
def format_report(table):
    marker = {NORUN: MARK_NORUN, BLANK: ""}

    def cell(tok):
        return f"{tok:.2f}" if isinstance(tok, (int, float)) else marker[tok]

    blocks = ["(4) GPU compute-usage table"]
    for stage_label, stage_key, rows in table["groups"]:
        data = [
            [
                ", ".join(names),
                *[cell(table["data_cell"](rk, a)) for a in ALG_COL_ORDER],
                cell(table["row_total"](rk)),
            ]
            for names, rk in rows
        ]
        df = pd.DataFrame(
            data,
            columns=["Environment", *[ALGORITHM_NAMES[a] for a in ALG_COL_ORDER], "Total"],
        )
        blocks.append(f"{stage_label} · {STAGE_DESIGN[stage_key]}\n{render_table(df)}")

    tot_row = [cell(table["col_tot"][a]) for a in ALG_COL_ORDER] + [
        cell(table["grand_total"])
    ]
    tot_df = pd.DataFrame(
        [tot_row], columns=[ALGORITHM_NAMES[a] for a in ALG_COL_ORDER] + ["Total"]
    )
    blocks.append(f"Total\n{render_table(tot_df)}")

    blocks.append(
        f"final runs : {table['final_total']:.2f} GPU-h ({table['final_total'] / 24:.2f} GPU-days)\n"
        f"grand total: {table['grand_total']:.2f} GPU-h ({table['grand_total'] / 24:.2f} GPU-days)"
    )
    return "\n\n".join(blocks)


def show_compute_console(table):
    print("\n" + format_report(table))


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------
def write_compute_latex(table):
    tex_marker = {NORUN: "---", BLANK: ""}

    def tex_cell(tok):
        return f"{tok:.2f}" if isinstance(tok, (int, float)) else tex_marker[tok]

    def tex_env(names):
        return ", ".join(rf"\texttt{{{n}}}" for n in names)

    def tex_design(key):
        return STAGE_DESIGN[key].replace("×", r"$\times$")

    alg_headers = [ALGORITHM_NAMES[a] for a in ALG_COL_ORDER]
    ncols = 1 + len(ALG_COL_ORDER) + 1  # Environment + algs + Total
    colspec = r"@{}l" + "r" * (len(ALG_COL_ORDER) + 1) + r"@{}"
    grand_total = table["grand_total"]

    caption = (
        rf"GPU compute usage by stage and environment. The experiments required "
        rf"approximately {grand_total:,.0f} GPU-hours ({grand_total / 24:,.0f} GPU-days) in "
        rf"total. Because algorithms and environments were run on different GPU models, "
        rf"these values provide an approximate measure of total compute usage rather than "
        rf"directly comparable computational costs."
    )

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:compute_accounting}",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        "Environment & " + " & ".join(alg_headers) + r" & Total \\",
        r"\midrule",
    ]
    for stage_label, stage_key, rows in table["groups"]:
        section = rf"\textbf{{{stage_label}}} -- {tex_design(stage_key)}"
        lines.append(rf"\multicolumn{{{ncols}}}{{c}}{{{section}}} \\")
        for names, rk in rows:
            body = [tex_cell(table["data_cell"](rk, a)) for a in ALG_COL_ORDER] + [
                tex_cell(table["row_total"](rk))
            ]
            lines.append(tex_env(names) + " & " + " & ".join(body) + r" \\")
        lines.append(r"\midrule")

    tot_cells = [
        rf"\textbf{{{tex_cell(table['col_tot'][a])}}}" for a in ALG_COL_ORDER
    ] + [rf"\textbf{{{tex_cell(grand_total)}}}"]
    tot_cells[-1] = r"\textbf{\valuetotalgpunumber{}}"  # defined in paper/definitions.tex
    lines.append(r"\textbf{Total} & " + " & ".join(tot_cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]

    tex = "\n".join(lines) + "\n"
    COMPUTE_LATEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    COMPUTE_LATEX_PATH.write_text(tex)
    print("Wrote", COMPUTE_LATEX_PATH)
    return tex


def write_compute_usage_table(csv_path=None):
    """Table 19, from compute_usage.csv (or the CSV at csv_path)."""
    raw = load_runs(csv_path or data_file("compute_usage.csv"))
    work = assign_rows(raw)

    check_runs_per_cell(work)
    check_device_mix(work)
    show_device_table(work)

    table = build_table(work)
    show_compute_console(table)
    write_compute_latex(table)


# ---------------------------------------------------------------------------
# Rollout speedup (Table 12)
# ---------------------------------------------------------------------------
SPEEDUP_LATEX_PATH = TABLES_DIR / "Table_12_JAX_Speedup.tex"


def write_speedup_table():
    """Table 12: wall-clock time for 1,000 rollout episodes and speedup over single-core NumPy."""
    lines = [
        r"\begin{table}[ht]",
        r"  \caption{Speedup comparison for 1,000 episodes. Speedup for NumPy multi-processing (MP) "
        r"and JAX is calculated relative to the respective environment's NumPy timing for 1,000 "
        r"episodes, e.g., for JAX, Speedup = Time(NumPy) / Time(JAX).}",
        r"  \label{tab:speedup_comparison_environments}",
        r"  \centering",
        r"  \begin{tabular}{@{}l r rr rr@{}}",
        r"    \toprule",
        r"    \multirow{2}{*}{Environment} & NumPy & \multicolumn{2}{c}{NumPy (MP)} "
        r"& \multicolumn{2}{c}{JAX} \\",
        r"    \cmidrule(lr){2-2} \cmidrule(lr){3-4} \cmidrule(lr){5-6}",
        r"     & \makecell[c]{Time (s)} & \makecell[c]{Time (s)} & \makecell[c]{Speedup} "
        r"& \makecell[c]{Time (s)} & \makecell[c]{Speedup} \\",
        r"    \midrule",
    ]
    for env, t in ROLLOUT_TIMES.items():
        name = env.rsplit("-v", 1)[0]
        lines.append(
            rf"    \texttt{{{name}}} & {t['numpy']:.1f} & {t['numpy_mp']:.1f} & "
            rf"{t['numpy'] / t['numpy_mp']:.1f} & {t['jax']:.1f} & {t['numpy'] / t['jax']:.1f} \\"
        )
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    SPEEDUP_LATEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPEEDUP_LATEX_PATH.write_text("\n".join(lines) + "\n")
    print("Wrote", SPEEDUP_LATEX_PATH)


# ===========================================================================
TABLE_GROUPS = ("results", "hyperparameters", "compute")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", choices=TABLE_GROUPS,
                        help="build only this group of tables (default: all)")
    parser.add_argument("--show", choices=["console", "none"], default="console",
                        help="hyperparameter tables: print them to stdout, or stay quiet "
                             "(the .tex file is always (re)written)")
    parser.add_argument("--csv", type=Path, default=None,
                        help="compute usage table: input CSV exported by "
                             "nb_compute_accounting.ipynb (default: data/compute_usage.csv)")
    args = parser.parse_args()

    if args.only in (None, "results"):
        write_results_tables()
    if args.only in (None, "hyperparameters"):
        write_hyperparameter_tables(show=args.show == "console")
    if args.only in (None, "compute"):
        write_compute_usage_table(args.csv)
        write_speedup_table()


if __name__ == "__main__":
    main()
