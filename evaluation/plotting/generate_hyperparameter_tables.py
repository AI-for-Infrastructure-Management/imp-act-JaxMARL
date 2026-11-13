"""Build the hyperparameter appendix tables straight from the repo's YAML configs.

For each algorithm (VDN, QMIX, PQN-VDN, MAPPO, IPPO) this pulls together three
stages of a config -- the convergence pilot, the Cologne-v1 tuning-sweep search
space, and the selected final-run config -- into one four-column table
(Hyperparameter | Pilot | Search space | Selected) and writes it out as a
LaTeX booktabs table, written to evaluation/plotting/tables/hyperparameter_tables.tex.
No number is hardcoded here, so re-running this after a config change
regenerates correct tables. The console preview shows the same values as the
.tex file, rendered as readable Unicode math (e.g. "5 x 10^-5") instead of
the raw LaTeX source.

Usage:
    python evaluation/plotting/generate_hyperparameter_tables.py
    python evaluation/plotting/generate_hyperparameter_tables.py --show none
"""

import argparse
import glob
import re
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import yaml


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


def find_repo_root(start):
    """Walk up from `start` until we hit the experiments/config directory."""
    for candidate in [start, *start.parents]:
        if (candidate / "experiments" / "config" / "pilot_runs").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find experiments/config/pilot_runs above {start}"
    )


REPO_ROOT = find_repo_root(Path(__file__).resolve())
CFG_DIR = REPO_ROOT / "experiments" / "config"
PILOT_DIR = CFG_DIR / "pilot_runs"
SWEEP_DIR = CFG_DIR / "hyper_parameter_tuning" / "cologne_v1"
FINAL_DIR = CFG_DIR / "final_runs"
LATEX_PATH = Path(__file__).resolve().parent / "tables" / "hyperparameter_tables.tex"


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


DISPLAY_NAMES = {
    "vdn_rnn": "VDN",
    "qmix_rnn": "QMIX",
    "pqn_rnn": "PQN-VDN",
    "mappo_rnn": "MAPPO",
    "ippo_rnn": "IPPO",
}
# PQN-VDN goes last, and QMIX/IPPO each close out a page (see write_latex).
ALG_ORDER = ["vdn_rnn", "qmix_rnn", "mappo_rnn", "ippo_rnn", "pqn_rnn"]
CLEARPAGE_AFTER = {"qmix_rnn", "ippo_rnn"}
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

    print("Found configs for:", ", ".join(DISPLAY_NAMES[a] for a in ALG_ORDER))
    for a in ALG_ORDER:
        name = DISPLAY_NAMES[a]
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


def render_table(df):
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_string(index=False)


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------
def to_latex(a, rows):
    name = DISPLAY_NAMES[a]
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


def write_latex(all_rows):
    sections = ["% \\usepackage{booktabs,float}"]
    for a in ALG_ORDER:
        tex_body = to_latex(a, all_rows[a]).rstrip("\n")
        sections.append(f"%% {DISPLAY_NAMES[a]} %%\n{tex_body}")
        if a in CLEARPAGE_AFTER:
            sections.append("\\clearpage")
    LATEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    LATEX_PATH.write_text("\n\n".join(sections) + "\n")
    print("Wrote", LATEX_PATH.relative_to(REPO_ROOT))


# ---------------------------------------------------------------------------
# Console preview -- same values as the .tex file, rendered readable
# ---------------------------------------------------------------------------
def show_console(all_rows):
    for a in ALG_ORDER:
        print(f"\n### {DISPLAY_NAMES[a]} ###")
        print(render_table(rows_to_preview_frame(all_rows[a])))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--show", choices=["console", "none"], default="console",
                        help="print the tables to stdout, or stay quiet "
                             "(the .tex file is always (re)written)")
    args = parser.parse_args()

    pilot_idx, base_idx, final_idx, sweeps = discover()
    data = {a: parse_algorithm(a, pilot_idx, base_idx, final_idx, sweeps) for a in ALG_ORDER}
    all_rows = {a: build_rows(a, data) for a in ALG_ORDER}

    if args.show == "console":
        show_console(all_rows)

    write_latex(all_rows)


if __name__ == "__main__":
    main()
