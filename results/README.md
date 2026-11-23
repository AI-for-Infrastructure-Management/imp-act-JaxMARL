# Results

`create_figures.py` and `create_tables.py` reproduce the figures and tables in the paper. Data
files, if missing, are automatically downloaded from [Zenodo](https://zenodo.org/records/24175731)
to `data/`. Apart from the policy rollouts, everything runs within a few seconds on a CPU.

```bash
pip install pyarrow  # reads the .parquet files

python results/create_figures.py   # figures/Figure_3 ... Figure_11, Extra_Figure_*.pdf
python results/create_tables.py    # tables/*.tex
```

Figure 11 and the budget-overrun rate require policy rollouts, which use JAX and the released
checkpoints and run considerably faster on a GPU. To reproduce the remaining figures without
the rollouts, we provide the `--skip-figure-11` flag. The notebooks are run from this folder
(`cd results`).

In addition, `create_figures.py` creates `Extra_Figure_*.pdf` and `create_tables.py` creates
`results_table.tex`, which are not used in the paper.

## Files

| File | Contents |
|---|---|
| `common.py` | Paths, Zenodo downloads (`data_file`, `best_checkpoint`), the comparison protocol, algorithm names and the plot style |
| `create_figures.py` | Figures 3–11 |
| `create_tables.py` | Tables 1 and 14–24; `--only {results,hyperparameters,compute}` builds one group |
| `get_and_plot_rollouts.py` | Rollouts of a released checkpoint (`RolloutDataGenerator`) or of the heuristic (`HeuristicRolloutDataGenerator`), and plots of an episode (`RolloutPlotter`) |
| `validate_budget_overrun_rate.py` | Share of timesteps in which the budget constraint changes the policy's actions |
| `nb_rollout_data.ipynb` | Episode costs, constraints, actions and single-episode trajectories of a policy |
| `nb_hyperparameter_tuning.ipynb` | Completeness, ranking and sensitivity of the hyperparameter sweep |
| `data/` | The Zenodo record: `results.csv`, `heuristics.csv`, `tuning_results.parquet`, `vdn_variants.csv`, `vdn_curves.parquet`, `compute_usage.csv`, and `checkpoints/` (`rankNN.tar.gz` and the run folders extracted from them) |
| `figures/`, `tables/` | The generated figures (PDF) and LaTeX tables |

## Comparing a new method

New methods can be compared with the results in the paper using the same evaluation
protocol. Specifically, for each training seed, we select the best checkpoint and normalize its
expected return with respect to the prioritized segment-based heuristic H<sub>PS</sub>. Both steps are
implemented in `common.py`:

```python
from common import baseline_values, best_per_run, load_results, normalize

results = load_results()                          # one row per evaluated checkpoint
best = best_per_run(results, "Cologne-v1")        # best checkpoint per (algorithm, run)
best[["algorithm", "run", "mean_pct", "lower_ci_pct", "upper_ci_pct"]]  # % over H_PS
normalize(my_return, baseline_values()["Cologne-v1"])  # the same for a new return
```
