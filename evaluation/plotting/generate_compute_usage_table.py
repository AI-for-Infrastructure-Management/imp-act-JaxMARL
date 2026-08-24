"""Build the GPU compute-usage table (paper appendix, Table 24) from cached wandb data.

Reads the CSV that evaluation/plotting/data/compute_accounting.ipynb exports
(evaluation/plotting/data/compute_usage.csv) -- this script never talks to wandb. That
notebook is the only place that does; re-run it to refresh the CSV as the experiments
progress, then re-run this script to regenerate the table.

The CSV already holds the filtered WORKING SET, not every raw fetched run: the notebook
keeps only the 50M-timestep budget for VDN/QMIX final runs (their 20M runs are an
earlier, superseded budget -- tuning keeps its own 20M budget untouched), and only
finished runs. This script just loads it and computes GPU hours = wall-clock runtime
(summary["_runtime"], fetched by the notebook) x GPU count, summed per cell. VDN-BA is
deferred for now and kept as a clearly marked placeholder column so the layout matches
the paper.

The checks and the table always print to the console as plain markdown -- there is no
flag that silences this, and the raw LaTeX itself is never dumped to the console. The
LaTeX is written to evaluation/plotting/tables/compute_usage.tex.

Usage:
    python evaluation/plotting/generate_compute_usage_table.py
    python evaluation/plotting/generate_compute_usage_table.py --csv path/to/other.csv
"""

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def find_repo_root(start):
    for candidate in [start, *start.parents]:
        if (candidate / "experiments" / "config" / "pilot_runs").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find experiments/config/pilot_runs above " + str(start)
    )


REPO_ROOT = find_repo_root(Path(__file__).resolve())
CSV_PATH = REPO_ROOT / "evaluation" / "plotting" / "data" / "compute_usage.csv"
LATEX_PATH = Path(__file__).resolve().parent / "tables" / "compute_usage.tex"


# ---------------------------------------------------------------------------
# Table layout -- single label mapping per concept, used by both renderings
# ---------------------------------------------------------------------------
ALG_COL_ORDER = [
    "pqn_rnn",
    "vdn_rnn",
    "vdn_ba_rnn",
    "qmix_rnn",
    "mappo_rnn",
    "ippo_rnn",
]
ALG_LABEL = {
    "pqn_rnn": "PQN-VDN",
    "vdn_rnn": "VDN",
    "vdn_ba_rnn": "VDN-BA",
    "qmix_rnn": "QMIX",
    "mappo_rnn": "MAPPO",
    "ippo_rnn": "IPPO",
}
DEFERRED_ALGS = {"vdn_ba_rnn"}  # ignored for now -> placeholder column

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
MARK_DEFERRED = "‡"  # double dagger: VDN-BA, deferred
MARK_STRUCT_NA = "n/a"  # structurally empty (e.g. VDN-BA has no pilot/tuning stage)
MARK_NORUN = "—"  # em dash: no matching run found

# Internal cell-token sentinels (distinct from the display markers above).
NORUN, NA, DEFERRED, BLANK = "NORUN", "NA", "DEFERRED", "BLANK"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_runs(csv_path):
    """Load the CSV (already the filtered working set) and compute GPU hours.

    Re-checks the two filters the notebook applies before writing the CSV, and warns
    rather than silently re-filtering -- a mismatch means the CSV is stale or was hand-edited.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found -- run evaluation/plotting/data/compute_accounting.ipynb "
            "first to fetch run data from wandb and export it."
        )
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


def render_table(df):
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_string(index=False)


# ---------------------------------------------------------------------------
# Checks (always run -- diagnostics, not gated by --show)
# ---------------------------------------------------------------------------
def check_runs_per_cell(work):
    env_rows = [ENV_PAPER[v] for v in ENV_ORDER]
    row_order = [ROW_PILOT, ROW_TUNING, *env_rows]
    present_rows = [r for r in row_order if r in set(work["row_key"])]
    train_cols = [c for c in ALG_COL_ORDER if c not in DEFERRED_ALGS]
    count_pivot = (
        work[work["col"].isin(ALG_COL_ORDER)]
        .pivot_table(
            index="row_key",
            columns="col",
            values="run_id",
            aggfunc="count",
            fill_value=0,
        )
        .reindex(index=present_rows, columns=train_cols, fill_value=0)
        .rename(columns=ALG_LABEL)
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
    train_algs = [a for a in ALG_COL_ORDER if a not in DEFERRED_ALGS]
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
        if alg in DEFERRED_ALGS:
            # VDN-BA has no pilot/tuning stage (structural n/a); its final rows are deferred.
            return NA if row_key in (ROW_PILOT, ROW_TUNING) else DEFERRED
        if counts[(row_key, alg)] == 0:
            return NORUN
        return hours[(row_key, alg)]

    def row_total(row_key):
        return sum(hours[(row_key, a)] for a in train_algs)

    groups = [
        ("Pilot runs", "pilot", [(pilot_maps, ROW_PILOT)]),
        ("Hyperparameter tuning", "tuning", [(["Cologne"], ROW_TUNING)]),
        ("Final runs", "final", [([ENV_PAPER[v]], ENV_PAPER[v]) for v in ENV_ORDER]),
    ]
    env_rows = [ENV_PAPER[v] for v in ENV_ORDER]
    col_tot = {
        a: (
            DEFERRED
            if a in DEFERRED_ALGS
            else sum(hours[(rk, a)] for rk in [ROW_PILOT, ROW_TUNING, *env_rows])
        )
        for a in ALG_COL_ORDER
    }
    grand_total = sum(v for v in col_tot.values() if isinstance(v, float))
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
    marker = {NORUN: MARK_NORUN, NA: MARK_STRUCT_NA, DEFERRED: MARK_DEFERRED, BLANK: ""}

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
            columns=["Environment", *[ALG_LABEL[a] for a in ALG_COL_ORDER], "Total"],
        )
        blocks.append(f"{stage_label} · {STAGE_DESIGN[stage_key]}\n{render_table(df)}")

    tot_row = [cell(table["col_tot"][a]) for a in ALG_COL_ORDER] + [
        cell(table["grand_total"])
    ]
    tot_df = pd.DataFrame(
        [tot_row], columns=[ALG_LABEL[a] for a in ALG_COL_ORDER] + ["Total"]
    )
    blocks.append(f"Total\n{render_table(tot_df)}")

    blocks.append(
        f"final runs : {table['final_total']:.2f} GPU-h ({table['final_total'] / 24:.2f} GPU-days)\n"
        f"grand total: {table['grand_total']:.2f} GPU-h ({table['grand_total'] / 24:.2f} GPU-days)"
        "  (excl. VDN-BA)"
    )
    return "\n\n".join(blocks)


def show_console(table):
    print("\n" + format_report(table))


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------
def write_latex(table):
    tex_marker = {NORUN: "---", NA: "n/a", DEFERRED: r"\ddag", BLANK: ""}

    def tex_cell(tok):
        return f"{tok:.2f}" if isinstance(tok, (int, float)) else tex_marker[tok]

    def tex_env(names):
        return ", ".join(rf"\texttt{{{n}}}" for n in names)

    def tex_design(key):
        return STAGE_DESIGN[key].replace("×", r"$\times$")

    alg_headers = [ALG_LABEL[a] for a in ALG_COL_ORDER]
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
        r"% Requires \usepackage{booktabs} and \usepackage{graphicx} (for \resizebox).",
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
    lines.append(r"\textbf{Total} & " + " & ".join(tot_cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]

    tex = "\n".join(lines) + "\n"
    LATEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    LATEX_PATH.write_text(tex)
    print("Wrote", LATEX_PATH.relative_to(REPO_ROOT))
    return tex


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=CSV_PATH,
        help="input CSV exported by evaluation/plotting/data/compute_accounting.ipynb "
        "(default: %(default)s)",
    )
    args = parser.parse_args()

    raw = load_runs(args.csv)
    work = assign_rows(raw)

    check_runs_per_cell(work)
    check_device_mix(work)
    show_device_table(work)

    table = build_table(work)
    show_console(table)
    write_latex(table)


if __name__ == "__main__":
    main()
