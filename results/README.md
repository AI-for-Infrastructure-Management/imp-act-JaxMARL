# Results

Code behind every figure, table and claim in the paper. The scripts read the released data
of the [Zenodo record](https://zenodo.org/records/24175731), which they download to `data/`
when it is missing. Everything except the rollouts runs on a CPU in a few seconds.

```bash
pip install pyarrow  # reads the .parquet files

python results/create_figures.py   # figures/Figure_3 ... Figure_11, Extra_Figure_*.pdf
python results/create_tables.py    # tables/*.tex
```

The policy rollouts (Figure 11 and the budget-overrun rate) need JAX and the released
checkpoints, and are much faster on a GPU; `create_figures.py --skip-figure-11` builds the
other figures only. The notebooks run from this folder (`cd results`).

## Where each result comes from

| Paper | Output | Made by | Data |
|---|---|---|---|
| **Main text** | | | |
| Table 1 | `tables/best_results_main.tex` | `create_tables.py` | `results.csv`, `heuristics.csv` |
| Figure 3 | `figures/Figure_3_JAX_Speedup.pdf` | `create_figures.py` | timings in `create_figures.py` |
| Figure 4 | `figures/Figure_4_Envs_Constrained_vs_Unconstrained.pdf` | `create_figures.py` | `results.csv`, `heuristics.csv` |
| Figure 5 | `figures/Figure_5_Cologne_Budgets.pdf` | `create_figures.py` | `results.csv`, `heuristics.csv` |
| Sec. 6, budget-overrun rate | printed | `validate_budget_overrun_rate.py` | rank-1 VDN checkpoint (GPU) |
| **Appendix D** | | | |
| Tables 12, 13 | written in the paper | — | timings in `create_figures.py` (`ROLLOUT_TIMES`, `VDN_TRAINING_TIMES`) |
| **Appendix E** | | | |
| Tables 14–18 | `tables/hyperparameter_tables.tex` | `create_tables.py --only hyperparameters` | `experiments/config/` |
| Table 19 | `tables/compute_usage.tex` | `create_tables.py --only compute` | `compute_usage.csv` |
| **Appendix F** | | | |
| Table 20 | `tables/best_checkpoints_all.tex` | `create_tables.py` | `results.csv`, `heuristics.csv` |
| Table 21 | `tables/best_model_cologne_budget_scenarios.tex` | `create_tables.py` | `results.csv`, `heuristics.csv` |
| Figures 6, 7 | `figures/Figure_6_*.pdf`, `figures/Figure_7_*.pdf` | `create_figures.py` | `results.csv`, `heuristics.csv` |
| Figure 8 | `figures/Figure_8_VDN_Tuning_Cologne.pdf` | `create_figures.py` | `tuning_results.parquet` |
| Figures 9, 10 | `figures/Figure_9_*.pdf`, `figures/Figure_10_*.pdf` | `create_figures.py` | `vdn_variants.csv` |
| Table 22 | `tables/vdn_cbd_unconstrained_configs.tex` | `create_tables.py` | `vdn_variants.csv` |
| Tables 23, 24 | `tables/heuristic_results_all.tex`, `tables/heuristic_parameters.tex` | `create_tables.py` | `heuristics.csv` |
| Figure 11 | `figures/Figure_11.pdf` | `create_figures.py` | rank-1 VDN checkpoint (GPU) |

`create_figures.py` also writes `Extra_Figure_*.pdf` and `create_tables.py` writes
`results_table.tex`, which are not in the paper.

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
| `figures/`, `tables/` | Outputs |

## Comparing a new method

All returns are reported relative to the prioritized segment-based heuristic H_PS, as the
best checkpoint of each training seed. `common.py` implements this protocol:

```python
from common import baseline_values, best_per_run, load_results, normalize

results = load_results()                          # one row per evaluated checkpoint
best = best_per_run(results, "Cologne-v1")        # best checkpoint per (algorithm, run)
best[["algorithm", "run", "mean_pct", "lower_ci_pct", "upper_ci_pct"]]  # % over H_PS
normalize(my_return, baseline_values()["Cologne-v1"])  # the same for a new return
```
