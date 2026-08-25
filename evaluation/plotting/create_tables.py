import pandas as pd
import numpy as np
import yaml
import os
import scipy.stats as stats

# Configuration
DATA_DIR = 'evaluation/plotting/data'
TABLES_DIR = 'evaluation/plotting/tables'
HEURISTIC_RESULTS_FILE = os.path.join(DATA_DIR, 'best_heuristic_results.yaml')
EVALUATION_RESULTS_FILE = os.path.join(DATA_DIR, 'combined_evaluation_results.csv')

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
    with open(HEURISTIC_RESULTS_FILE, 'r') as f:
        heuristic_results = yaml.load(f, Loader=yaml.FullLoader)
    
    baseline_heuristic_values = {}
    for env, results in heuristic_results.items():
        if "segment_based_heuristic" in results:
             baseline_heuristic_values[env] = results["segment_based_heuristic"]["mean_reward"]
    
    data = pd.read_csv(EVALUATION_RESULTS_FILE)
    return data, baseline_heuristic_values

def get_best_checkpoints(data, env_name, baseline_heuristic_values):
    """
    Extracts best checkpoints for each algorithm and normalizes rewards.
    """
    algorithms = data['algorithm'].unique()
    heuristic_score = baseline_heuristic_values[env_name]
    
    best_checkpoint_per_algorithm = {}
    
    for alg in algorithms:
        alg_data = data[data['algorithm'] == alg]
        runs = alg_data['WANDB_RUN_ID'].unique()
        
        best_runs = []
        for run in runs:
            run_data = alg_data[alg_data['WANDB_RUN_ID'] == run]
            best_idx = run_data['mean'].idxmax()
            best_row = run_data.loc[best_idx]
            
            # Normalize
            norm_mean = (best_row['mean'] - heuristic_score) / abs(heuristic_score)
            norm_lower = (best_row['lower_ci'] - heuristic_score) / abs(heuristic_score)
            norm_upper = (best_row['upper_ci'] - heuristic_score) / abs(heuristic_score)
            
            best_runs.append({
                'mean': norm_mean,
                'lower_ci': norm_lower,
                'upper_ci': norm_upper
            })
            
        best_checkpoint_per_algorithm[alg] = best_runs
        
    return best_checkpoint_per_algorithm

def format_heuristic_value(val):
    """Formats heuristic value for display."""
    if abs(val) >= 1e9:
        return f"{val/1e9:.1f}\\text{{B}}"
    else:
        return f"{val/1e6:.0f}\\text{{M}}"

def generate_latex_table(best_checkpoints_all, baseline_heuristic_values, envs_to_plot=None, output_filename='results_table.tex'):
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
                best_run = max(runs, key=lambda x: x['mean'])
                
                val_mean = best_run['mean'] * 100
                val_lower = best_run['lower_ci'] * 100
                val_upper = best_run['upper_ci'] * 100
                
                alg_stats[alg] = {'mean': val_mean, 'ci': (val_lower, val_upper)}
            else:
                alg_stats[alg] = {'mean': np.nan, 'ci': (np.nan, np.nan)}
        
        # Find best main algorithm
        best_main_alg = None
        best_main_mean = -np.inf
        best_main_ci = None
        
        for alg in main_algorithms:
            if not np.isnan(alg_stats[alg]['mean']):
                if alg_stats[alg]['mean'] > best_main_mean:
                    best_main_mean = alg_stats[alg]['mean']
                    best_main_alg = alg
                    best_main_ci = alg_stats[alg]['ci']
        
        # Format environment column
        h_val = baseline_heuristic_values[env]
        h_str = format_heuristic_value(h_val)
        env_display = ENV_DISPLAY_NAMES.get(env, env.replace("_", "\\_"))
        multirow_cmd = r"\multirow{6}{*}{{\shortstack{" + env_display + r" \\ $\text{H}_\text{PS}=" + h_str + r"$}}}"
        
        # Print main algorithms
        for i, alg in enumerate(main_algorithms):
            row_str = ""
            if i == 0:
                row_str += f"    {multirow_cmd} \n    & "
            else:
                row_str += "    & "
            
            alg_name = ALGORITHM_NAMES[alg]
            mean = alg_stats[alg]['mean']
            ci = alg_stats[alg]['ci']
            
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
        mean = alg_stats[alg]['mean']
        ci = alg_stats[alg]['ci']
        
        if np.isnan(mean):
            mean_str = "--"
            ci_str = "--"
        else:
            mean_str = f"{mean:+.2f}\\%"
            ci_str = f"{{[{ci[0]:.2f}\\,, {ci[1]:.2f}]}}"
        
        latex_lines.append(f"    & {alg_name:<8} & {mean_str:<16} & {ci_str:<18}   \\\\")
        
        if env != envs_to_plot[-1]:
            latex_lines.append(r"    \midrule")
        else:
            latex_lines.append(r"    \bottomrule")

    latex_lines.append(r"  \end{tabular}")
    
    os.makedirs(TABLES_DIR, exist_ok=True)
    output_path = os.path.join(TABLES_DIR, output_filename)
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex_lines))
    print(f"Table saved to {output_path}")

def generate_latex_table_combined(best_checkpoints_all, baseline_heuristic_values):
    """Generates LaTeX table comparing Constrained and Unconstrained results."""
    envs_to_plot = ["ToyExample-v2", "Cologne-v1", "CologneBonnDusseldorf-v1"]
    main_algorithms = ["vdn_rnn", "qmix_rnn", "pqn_rnn", "mappo_rnn", "ippo_rnn"]
    baseline_alg = "vdn_ba_rnn"
    
    latex_lines = []
    latex_lines.append(r"  \begin{tabular}{llrcrc}")
    latex_lines.append(r"    \toprule")
    latex_lines.append(r"    & & \multicolumn{2}{c}{\textbf{Constrained}} & \multicolumn{2}{c}{\textbf{Unconstrained}} \\")
    latex_lines.append(r"    \cmidrule(lr){3-4} \cmidrule(lr){5-6}")
    latex_lines.append(r"    \thead{\textbf{Environment}} & \thead{\textbf{Algorithm}}")
    latex_lines.append(r"    & \thead{\textbf{${\Delta \bar{R}^{\pi}_0}$}} ")
    latex_lines.append(r"    & \thead{\textbf{95\% CI}} ")
    latex_lines.append(r"    & \thead{\textbf{${\Delta \bar{R}^{\pi}_0}$}} ")
    latex_lines.append(r"    & \thead{\textbf{95\% CI}} \\")
    latex_lines.append(r"    \midrule")
    
    for env in envs_to_plot:
        unconstrained_env = f"{env}-unconstrained"
        
        # Calculate stats for all algorithms (Constrained)
        alg_stats_constrained = {}
        for alg in main_algorithms + [baseline_alg]:
            if alg in best_checkpoints_all[env]:
                runs = best_checkpoints_all[env][alg]
                best_run = max(runs, key=lambda x: x['mean'])
                val_mean = best_run['mean'] * 100
                val_lower = best_run['lower_ci'] * 100
                val_upper = best_run['upper_ci'] * 100
                alg_stats_constrained[alg] = {'mean': val_mean, 'ci': (val_lower, val_upper)}
            else:
                alg_stats_constrained[alg] = {'mean': np.nan, 'ci': (np.nan, np.nan)}

        # Calculate stats for all algorithms (Unconstrained)
        alg_stats_unconstrained = {}
        for alg in main_algorithms + [baseline_alg]:
            if unconstrained_env in best_checkpoints_all and alg in best_checkpoints_all[unconstrained_env]:
                runs = best_checkpoints_all[unconstrained_env][alg]
                best_run = max(runs, key=lambda x: x['mean'])
                val_mean = best_run['mean'] * 100
                val_lower = best_run['lower_ci'] * 100
                val_upper = best_run['upper_ci'] * 100
                alg_stats_unconstrained[alg] = {'mean': val_mean, 'ci': (val_lower, val_upper)}
            else:
                alg_stats_unconstrained[alg] = {'mean': np.nan, 'ci': (np.nan, np.nan)}
        
        # Find best main algorithm (Constrained)
        best_main_alg_c = None
        best_main_mean_c = -np.inf
        best_main_ci_c = None
        for alg in main_algorithms:
            if not np.isnan(alg_stats_constrained[alg]['mean']):
                if alg_stats_constrained[alg]['mean'] > best_main_mean_c:
                    best_main_mean_c = alg_stats_constrained[alg]['mean']
                    best_main_alg_c = alg
                    best_main_ci_c = alg_stats_constrained[alg]['ci']

        # Find best main algorithm (Unconstrained)
        best_main_alg_u = None
        best_main_mean_u = -np.inf
        best_main_ci_u = None
        for alg in main_algorithms:
            if not np.isnan(alg_stats_unconstrained[alg]['mean']):
                if alg_stats_unconstrained[alg]['mean'] > best_main_mean_u:
                    best_main_mean_u = alg_stats_unconstrained[alg]['mean']
                    best_main_alg_u = alg
                    best_main_ci_u = alg_stats_unconstrained[alg]['ci']
        
        # Format environment column
        h_val = baseline_heuristic_values[env]
        h_str = format_heuristic_value(h_val)
        
        h_val_u = baseline_heuristic_values[unconstrained_env]
        h_str_u = format_heuristic_value(h_val_u)

        env_display = ENV_DISPLAY_NAMES[env]
        multirow_cmd = r"\multirow{7}{*}{{\shortstack{" + env_display + r"}}}"
        
        # Heuristic row
        latex_lines.append(f"    {multirow_cmd}")
        latex_lines.append(r"    & & \multicolumn{2}{c}{$\text{H}_\text{PS}=" + h_str + r"$} & \multicolumn{2}{c}{$\text{H}_\text{PS}=" + h_str_u + r"$} \\")
        latex_lines.append(r"    \cmidrule(lr){3-6}")

        # Print main algorithms
        for i, alg in enumerate(main_algorithms):
            row_str = "    & "
            
            alg_name = ALGORITHM_NAMES[alg]
            
            # Constrained Data
            mean_c = alg_stats_constrained[alg]['mean']
            ci_c = alg_stats_constrained[alg]['ci']
            mean_str_c = f"{mean_c:+.2f}\\%"
            if alg == best_main_alg_c:
                mean_str_c = r"\textbf{" + mean_str_c + "}"
            
            is_star_c = False
            if alg != best_main_alg_c and not np.isnan(mean_c):
                overlap = max(ci_c[0], best_main_ci_c[0]) < min(ci_c[1], best_main_ci_c[1])
                if overlap:
                    is_star_c = True
            if is_star_c:
                mean_str_c = "*" + mean_str_c
            ci_str_c = f"{{[{ci_c[0]:.2f}\\,, {ci_c[1]:.2f}]}}"

            # Unconstrained Data
            mean_u = alg_stats_unconstrained[alg]['mean']
            ci_u = alg_stats_unconstrained[alg]['ci']
            if np.isnan(mean_u):
                mean_str_u = "--"
                ci_str_u = "--"
            else:
                mean_str_u = f"{mean_u:+.2f}\\%"
                if alg == best_main_alg_u:
                    mean_str_u = r"\textbf{" + mean_str_u + "}"
                
                is_star_u = False
                if alg != best_main_alg_u and not np.isnan(mean_u):
                    overlap = max(ci_u[0], best_main_ci_u[0]) < min(ci_u[1], best_main_ci_u[1])
                    if overlap:
                        is_star_u = True
                if is_star_u:
                    mean_str_u = "*" + mean_str_u
                ci_str_u = f"{{[{ci_u[0]:.2f}\\,, {ci_u[1]:.2f}]}}"
            
            row_str += f"{alg_name:<8} & {mean_str_c:<16} & {ci_str_c:<18} & {mean_str_u:<16} & {ci_str_u:<18} \\\\"
            latex_lines.append(row_str)
            
        # Separator
        latex_lines.append(r"    \cmidrule(lr){2-6}")
        
        # Print baseline algorithm
        alg = baseline_alg
        alg_name = ALGORITHM_NAMES[alg]
        
        # Constrained
        mean_c = alg_stats_constrained[alg]['mean']
        ci_c = alg_stats_constrained[alg]['ci']
        mean_str_c = f"{mean_c:+.2f}\\%"
        ci_str_c = f"{{[{ci_c[0]:.2f}\\,, {ci_c[1]:.2f}]}}"

        # Unconstrained
        mean_u = alg_stats_unconstrained[alg]['mean']
        ci_u = alg_stats_unconstrained[alg]['ci']
        if np.isnan(mean_u):
            mean_str_u = "--"
            ci_str_u = "--"
        else:
            mean_str_u = f"{mean_u:+.2f}\\%"
            ci_str_u = f"{{[{ci_u[0]:.2f}\\,, {ci_u[1]:.2f}]}}"
        
        latex_lines.append(f"    & {alg_name:<8} & {mean_str_c:<16} & {ci_str_c:<18} & {mean_str_u:<16} & {ci_str_u:<18}   \\\\")
        
        if env != envs_to_plot[-1]:
            latex_lines.append(r"    \midrule")
        else:
            latex_lines.append(r"    \bottomrule")

    latex_lines.append(r"  \end{tabular}")
    
    os.makedirs(TABLES_DIR, exist_ok=True)
    output_path = os.path.join(TABLES_DIR, 'results_table_combined.tex')
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex_lines))
    print(f"Combined table saved to {output_path}")

def get_per_seed_results(data, env_name, baseline_heuristic_values):
    """
    Extracts the best checkpoint per seed for each algorithm and normalizes rewards.
    Returns a dict: {algorithm: [{seed: int, mean: float}, ...]}
    """
    algorithms = data['algorithm'].unique()
    heuristic_score = baseline_heuristic_values[env_name]
    
    per_seed_results = {}
    
    for alg in algorithms:
        alg_data = data[data['algorithm'] == alg]
        runs = alg_data['WANDB_RUN_ID'].unique()
        
        seed_results = []
        for i, run in enumerate(runs):
            run_data = alg_data[alg_data['WANDB_RUN_ID'] == run]
            best_idx = run_data['mean'].idxmax()
            best_row = run_data.loc[best_idx]
            
            # Normalize
            norm_mean = (best_row['mean'] - heuristic_score) / abs(heuristic_score) * 100
            
            seed_results.append({
                'seed': i + 1,
                'mean': norm_mean
            })
        
        # Sort by mean (descending)
        seed_results.sort(key=lambda x: x['mean'], reverse=True)
        per_seed_results[alg] = seed_results
        
    return per_seed_results


def generate_latex_table_per_seed(per_seed_results_all, baseline_heuristic_values, 
                                   envs_to_plot=None, output_filename='results_table_per_seed.tex'):
    """
    Generates LaTeX table showing per-seed results with IQR.
    
    Table format:
    Environment | Seed | VDN | QMIX | PQN-VDN | MAPPO | IPPO
    """
    if envs_to_plot is None:
        envs_to_plot = ["ToyExample-v2", "Cologne-v1", "CologneBonnDusseldorf-v1"]
    
    main_algorithms = ["vdn_rnn", "qmix_rnn", "pqn_rnn", "mappo_rnn", "ippo_rnn"]
    
    latex_lines = []
    latex_lines.append(r"  \begin{tabular}{@{}crccccc@{}}")
    latex_lines.append(r"    \toprule")
    latex_lines.append(r"    \thead{\textbf{Environment}} & \thead{\textbf{Seed}} &")
    latex_lines.append(r"    \thead{\textbf{VDN}} & \thead{\textbf{QMIX}} & \thead{\textbf{PQN-VDN}} & ")
    latex_lines.append(r"    \thead{\textbf{MAPPO}} & \thead{\textbf{IPPO}} \\")
    latex_lines.append(r"    \midrule")
    
    for env_idx, env in enumerate(envs_to_plot):
        per_seed = per_seed_results_all[env]
        
        # Get number of seeds (assume all algorithms have the same number)
        num_seeds = len(per_seed[main_algorithms[0]])
        
        # Format environment column
        h_val = baseline_heuristic_values[env]
        h_str = format_heuristic_value(h_val)
        env_display = ENV_DISPLAY_NAMES.get(env, env.replace("_", "\\_"))
        
        # Multirow for environment name (num_seeds + 1 for IQR row)
        multirow_cmd = r"\multirow{" + str(num_seeds + 1) + r"}{*}{{\shortstack[c]{" + env_display + r" \\ $\text{H}_\text{PS} = " + h_str + r"$}}}"
        
        # Add comment for environment
        env_comment = env.replace("-v1", "").replace("-v2", "")
        latex_lines.append(f"    % {env_comment}")
        
        # Print seed rows
        for seed_idx in range(num_seeds):
            row_str = ""
            if seed_idx == 0:
                row_str += f"     {multirow_cmd}\n        "
            else:
                row_str += "        "
            
            row_str += f"&  {seed_idx + 1:<6}"
            
            for alg in main_algorithms:
                if alg in per_seed and seed_idx < len(per_seed[alg]):
                    mean = per_seed[alg][seed_idx]['mean']
                    row_str += f" & {mean:+.2f}"
                else:
                    row_str += " & --"
            
            row_str += " \\\\"
            latex_lines.append(row_str)
        
        # Separator before IQR
        latex_lines.append(r"    \cmidrule(lr){2-7}")
        
        # Calculate and print IQR row
        iqr_row = "        & \\textbf{IQR}"
        for alg in main_algorithms:
            if alg in per_seed and len(per_seed[alg]) > 0:
                means = [r['mean'] for r in per_seed[alg]]
                q75 = np.percentile(means, 75)
                q25 = np.percentile(means, 25)
                iqr = q75 - q25
                iqr_row += f" & \\textbf{{{iqr:.2f}}}"
            else:
                iqr_row += " & --"
        iqr_row += " \\\\"
        latex_lines.append(iqr_row)
        
        # Add midrule or bottomrule
        if env_idx != len(envs_to_plot) - 1:
            latex_lines.append(r"    \midrule")
        else:
            latex_lines.append(r"    \bottomrule")
    
    latex_lines.append(r"  \end{tabular}")
    
    os.makedirs(TABLES_DIR, exist_ok=True)
    output_path = os.path.join(TABLES_DIR, output_filename)
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex_lines))
    print(f"Per-seed table saved to {output_path}")

def generate_heuristic_comparison_table(envs_to_plot=None, output_filename='heuristic_comparison.tex'):
    """
    Generates LaTeX table comparing different heuristics across environments.
    """
    if envs_to_plot is None:
        envs_to_plot = ["ToyExample-v2", "Cologne-v1", "CologneBonnDusseldorf-v1"]

    with open(HEURISTIC_RESULTS_FILE, 'r') as f:
        heuristic_results = yaml.load(f, Loader=yaml.FullLoader)

    latex_lines = []
    latex_lines.append(r" \begin{tabular}{clrr}")
    latex_lines.append(r"    \toprule")
    latex_lines.append(r"    \thead{\textbf{Environment}} & \thead{\textbf{Heuristic}} & \thead{\textbf{Expected Return} ($10^9$)} & \thead{\textbf{Std. Dev.} ($10^9$)} \\")
    latex_lines.append(r"    \midrule")

    heuristics_map = {
        "do_nothing": "Always routine inspections",
        "random": "Random actions",
        "always_repair": "Always minor repair",
        "segment_based_heuristic": r"Prioritized segment-based (\(\text{H}_\text{PS}\))"
    }
    
    # Define order
    heuristic_order = ["do_nothing", "random", "always_repair", "segment_based_heuristic"]
    

    for i, env in enumerate(envs_to_plot):
        if env not in heuristic_results:
             print(f"Warning: Environment {env} not found in heuristic results.")
             continue
             
        env_results = heuristic_results[env]
        
        env_display = ENV_DISPLAY_NAMES.get(env, env.replace("_", "\\_"))
        
        available_heuristics = [h for h in heuristic_order if h in env_results]
        num_rows = len(available_heuristics)
        
        multirow_cmd = r"\multirow{" + str(num_rows) + r"}{*}{{\shortstack{" + env_display + r"}}}"

        for j, h_key in enumerate(available_heuristics):
            row_str = ""
            if j == 0:
                row_str += f"    {multirow_cmd}\n    & "
            else:
                row_str += "    & "
            
            h_name = heuristics_map.get(h_key, h_key)
            result = env_results[h_key]
            
            mean_val = result["mean_reward"] / 1e9
            std_val = result["std_reward"] / 1e9
            
            # Formatting
            mean_str = f"{mean_val:,.2f}"
            std_str = f"{std_val:,.2f}"
            
            # Bold for segment_based_heuristic (or best?)
            if h_key == "segment_based_heuristic":
                 mean_str = r"\textbf{" + mean_str + "}"

            row_str += f"{h_name:<40} & {mean_str:>10} & {std_str:>10} \\\\"
            latex_lines.append(row_str)
        
        if i != len(envs_to_plot) - 1:
            latex_lines.append(r"    \midrule")
        else:
            latex_lines.append(r"    \bottomrule")

    latex_lines.append(r"  \end{tabular}")

    os.makedirs(TABLES_DIR, exist_ok=True)
    output_path = os.path.join(TABLES_DIR, output_filename)
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex_lines))
    print(f"Heuristic comparison table saved to {output_path}")

def generate_heuristic_parameters_table(output_filename='heuristic_parameters.tex'):
    """
    Generates LaTeX table showing parameters for the best segment_based_heuristic.
    """
    with open(HEURISTIC_RESULTS_FILE, 'r') as f:
        heuristic_results = yaml.load(f, Loader=yaml.FullLoader)

    latex_lines = []
    latex_lines.append(r"  \begin{tabular}{crrcc}")
    latex_lines.append(r"    \toprule")
    latex_lines.append(r"    \thead{\textbf{Environment}} & \thead{$T^{\text{insp}}$} & \thead{$i_{\text{thres}}$} & \thead{$\xi$} & \thead{$k$} \\")
    latex_lines.append(r"    \midrule")

    # Order of environments to display
    envs_order = [
        "ToyExample-v2", "Cologne-v1", "CologneBonnDusseldorf-v1",
        "ToyExample-v2-unconstrained", "Cologne-v1-unconstrained", "CologneBonnDusseldorf-v1-unconstrained",
        "Cologne-v1-moderate-budget", "Cologne-v1-critical-budget"
    ]
    
    for i, env in enumerate(envs_order):
        if env not in heuristic_results:
            continue
            
        results = heuristic_results[env]
        if "segment_based_heuristic" not in results:
            continue
            
        params = results["segment_based_heuristic"]["parameters"]
        
        env_display = ENV_DISPLAY_NAMES.get(env, env.replace("_", "\\_"))
        
        # Parameters
        t_insp = params.get('inspection_interval', '-')
        i_thres = params.get('repair_threshold', '-')
        
        # Xi
        xi_str = "-"
        k_str = "-"
        if params.get('prioritization_enabled', False):
            key = params.get('prioritization_key')
            sign = params.get('prioritization_sign')
            k_val = params.get('top_k')
            k_str = str(k_val) if k_val is not None else "-"
            
            if key == 'volumes' and sign == 'positive':
                xi_str = "highest base traffic volumes"
            elif key == 'volumes' and sign == 'negative':
                xi_str = "lowest base traffic volumes"
            elif key == 'segment_lengths' and sign == 'positive':
                xi_str = "highest segment lengths"
            elif key == 'segment_lengths' and sign == 'negative':
                xi_str = "lowest segment lengths"
            elif key in ['cost', 'costs'] and sign == 'positive':
                 xi_str = "highest action cost"
            elif key in ['cost', 'costs'] and sign == 'negative':
                 xi_str = "lowest action cost"
            else:
                 xi_str = f"{sign} {key}".replace('_', ' ')
        
        # One row per environment
        row_str = f"    {{\\shortstack{{{env_display}}}}} & {t_insp} & {i_thres} & {xi_str} & {k_str} \\\\"
        latex_lines.append(row_str)
        
        if i != len(envs_order) - 1:
            latex_lines.append(r"    \midrule")
        else:
            latex_lines.append(r"    \bottomrule")

    latex_lines.append(r"  \end{tabular}")
    
    os.makedirs(TABLES_DIR, exist_ok=True)
    output_path = os.path.join(TABLES_DIR, output_filename)
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex_lines))
    print(f"Heuristic parameters table saved to {output_path}")

def main():
    data, baseline_heuristic_values = load_data()
    
    environments = data['map_name'].unique()
    best_checkpoints_all = {
        env: get_best_checkpoints(data[data['map_name'] == env], env, baseline_heuristic_values)
        for env in environments
    }

    # Original table
    generate_latex_table(best_checkpoints_all, baseline_heuristic_values)
    
    # Combined table
    generate_latex_table_combined(best_checkpoints_all, baseline_heuristic_values)

    # Cologne Budget Scenarios table
    cologne_budgets_envs = [
        "Cologne-v1-unconstrained",
        "Cologne-v1-moderate-budget",
        "Cologne-v1",
        "Cologne-v1-critical-budget"
    ]
    generate_latex_table(
        best_checkpoints_all, 
        baseline_heuristic_values, 
        envs_to_plot=cologne_budgets_envs, 
        output_filename='results_table_budgets.tex'
    )
    
    # Per-seed results table
    per_seed_results_all = {
        env: get_per_seed_results(data[data['map_name'] == env], env, baseline_heuristic_values)
        for env in environments
    }
    
    # Per-seed table for constrained environments
    generate_latex_table_per_seed(
        per_seed_results_all,
        baseline_heuristic_values,
        envs_to_plot=["ToyExample-v2", "Cologne-v1", "CologneBonnDusseldorf-v1"],
        output_filename='best_checkpoints.tex'
    )
    
    # Per-seed table for unconstrained environments
    generate_latex_table_per_seed(
        per_seed_results_all,
        baseline_heuristic_values,
        envs_to_plot=[
            "ToyExample-v2-unconstrained",
            "Cologne-v1-unconstrained",
            "CologneBonnDusseldorf-v1-unconstrained"
        ],
        output_filename='best_checkpoints_unconstrained.tex'
    )
    
    # Per-seed table for Cologne budget scenarios
    generate_latex_table_per_seed(
        per_seed_results_all,
        baseline_heuristic_values,
        envs_to_plot=[
            "Cologne-v1-unconstrained",
            "Cologne-v1-moderate-budget",
            "Cologne-v1",
            "Cologne-v1-critical-budget"
        ],
        output_filename='best_checkpoints_budgets.tex'
    )
    
    # Heuristic Comparison Table
    generate_heuristic_comparison_table(output_filename='heuristic_results.tex')
    
    # Heuristic Comparison Table (Cologne Budgets)
    generate_heuristic_comparison_table(envs_to_plot=cologne_budgets_envs, output_filename='heuristic_results_budgets.tex')

    # Heuristic Comparison Table (Unconstrained)
    unconstrained_envs = [
        "ToyExample-v2-unconstrained",
        "Cologne-v1-unconstrained",
        "CologneBonnDusseldorf-v1-unconstrained"
    ]
    generate_heuristic_comparison_table(envs_to_plot=unconstrained_envs, output_filename='heuristic_results_unconstrained.tex')
    
    # Heuristic Parameters Table
    generate_heuristic_parameters_table()

if __name__ == "__main__":
    main()
