import logging
import numpy as np
import jax
import jaxmarl
import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig, OmegaConf
from heuristic_rollout_data_generator import HeuristicRolloutDataGenerator


def round_to_nearest_multiple(x, base=5):
    scale = int(np.floor(np.log10(abs(x))))
    factor = 10 ** (scale - 1)
    return round(x / (base * factor)) * (base * factor)


def human_readable_scale(x):
    ax = abs(x)
    if ax >= 1e12:
        return 1e12, "T"
    if ax >= 1e9:
        return 1e9, "B"
    if ax >= 1e6:
        return 1e6, "M"
    if ax >= 1e3:
        return 1e3, "K"
    return 1, ""


def visualize_periodic_metrics(period_stats, rounded_budgets):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    labels = [
        "Routine inspections",
        "Major inspections",
        "Minor repairs",
        "Major repairs",
        "Replacements",
    ]

    ############# Avg. number of actions per period #############
    series = [np.array(period_stats[f"action{i}_per_period"]) for i in range(5)]

    # X axis (periods)
    _periods = np.arange(1, len(series[0]) + 1)
    bar_width = 0.6

    # Stacked bars
    bottom = np.zeros_like(series[0], dtype=float)
    for y, lab in zip(series, labels):
        ax[0].bar(_periods, y, width=bar_width, bottom=bottom, label=lab)
        bottom += y

    ax[0].set_xlabel("Periods")
    ax[0].set_ylabel("Average number of actions per period")
    ax[0].set_title("Average Number of Actions per Period")
    ax[0].grid(axis="y", linestyle="-", alpha=0.7, linewidth=0.5)
    # ax[0].legend()

    ############# Avg. cost per period #############
    series = [np.array(period_stats[f"action{i}_cost_per_period"]) for i in range(5)]

    # Stacked bars
    bottom = np.zeros_like(series[0], dtype=float)
    for y, lab in zip(series, labels):
        ax[1].bar(_periods, y, width=bar_width, bottom=bottom, label=lab)
        bottom += y

    # === Median line across total cycle costs ===
    cycle_costs = np.stack(series, axis=1).sum(axis=1)
    median = np.median(cycle_costs)
    norm_const, unit_label = human_readable_scale(median)

    ax[1].hlines(
        median,
        xmin=0,
        xmax=len(_periods) + 1,
        colors="red",
        linestyles="dashed",
        label=f"Median cost: {median/norm_const:.2f} {unit_label}",
    )

    # draw lines for each rounded budget
    for i, b in enumerate(rounded_budgets):
        norm_const, unit_label = human_readable_scale(b)
        val = b / norm_const  # convert to chosen unit
        ax[1].hlines(
            -b,
            xmin=0,
            xmax=len(_periods) + 1,
            colors="gray",
            linestyles="--",
            alpha=0.6,
        )
        ax[1].text(
            0,  # slightly beyond last bar
            -b,  # align with the line
            f"{val:.1f} {unit_label}",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
        )

    ax[1].set_xlabel("Periods")
    ax[1].set_ylabel("Average cost per period")
    ax[1].set_title("Average Cost per Period")
    ax[1].grid(axis="y", linestyle="-", alpha=0.7, linewidth=0.5)
    # ax[1].legend()

    plt.tight_layout(rect=[0, 0, 0.78, 1])  # leave room at bottom for legend
    # === Shared legend outside the plots ===
    handles, labels_ = ax[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_,
        loc="center left",  # or "lower center" / "center right"
        bbox_to_anchor=(0.78, 0.5),
        ncol=1,
        frameon=True,
    )

    plt.show()


@hydra.main(
    config_path="../experiments/config/heuristics/best_parameters",
    config_name="toy_example_v2_heuristic",
    version_base=None,
)
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)

    env = jaxmarl.make("road_env", map_name=cfg.map)
    rdg = HeuristicRolloutDataGenerator(config)

    print("Generating rollout data")
    key = jax.random.PRNGKey(87)
    num_rollouts = 1000
    episode_data = rdg.generate_rollout_data(key, num_rollouts, verbose=False)

    # * What is the average distribution of `applied_actions` per period?
    batched_get_rewards = jax.jit(
        jax.vmap(  # over episodes B
            jax.vmap(  # over timesteps T
                env.env._get_rewards_from_table,  # already vmaps over C
                in_axes=(0, 0, 0, None),  # dam, act, forced, seg_len
            ),
            in_axes=(0, 0, 0, None),
        )
    )
    print(f"Generated data from {num_rollouts} rollouts.")

    NUM_ACTIONS = env.action_space().n
    period = 5  #!HARDCODED
    horizon = env.env.max_timesteps
    period_stats = {
        key: []
        for key in [
            *(f"action{i}_per_period" for i in range(NUM_ACTIONS)),
            *(f"action{i}_cost_per_period" for i in range(NUM_ACTIONS)),
        ]
    }

    l = 0
    for m in range(period, horizon + period, period):
        # applied_actions     | shape: (1000, 50, 60)
        # edge_states         | shape: (1000, 51, 60)
        # forced_repair_flags | shape: (1000, 50, 60)

        for a in range(NUM_ACTIONS):

            action_mask = episode_data["applied_actions"][:, l:m, :] == a

            # number of actions per period
            num_actions = np.mean(action_mask)
            period_stats[f"action{a}_per_period"].append(num_actions)

            # cost per period
            # set all other actions to 0
            actions = episode_data["applied_actions"][:, l:m, :] * action_mask.astype(
                int
            )
            # compute rewards only for the selected actions
            _rewards = batched_get_rewards(
                episode_data["edge_states"][:, l:m, :],
                actions,
                episode_data["forced_repair_flags"][:, l:m, :],
                env.env.segment_lengths,
            )
            # since action 0 has non-zero reward for doing nothing, we mask the rewards
            _rewards *= action_mask.astype(float)
            _rewards = _rewards.sum(axis=(1, 2)).mean(axis=0)
            period_stats[f"action{a}_cost_per_period"].append(_rewards)

        l += period

    ######################### Budget Quantiles #########################
    cycle_costs = np.stack(
        [np.array(period_stats[f"action{i}_cost_per_period"]) for i in range(5)]
    ).sum(axis=0)

    quantile_levels = [0.25, 0.5, 0.75, 0.95]
    labels = ["easy", "B1-Moderate", "B2-Limited", "B3-Critical"]

    _budget_amounts = np.quantile(cycle_costs, quantile_levels)
    budget_amounts = []
    rounded_budgets = []

    # Choose normalization scale from the median value
    norm_const, unit_label = human_readable_scale(np.median(np.abs(_budget_amounts)))

    for q, label, amount in zip(quantile_levels, labels, _budget_amounts):
        amount = -amount  # make positive
        rounded = round_to_nearest_multiple(amount)
        budget_amounts.append(rounded)
        print(
            f"{label:<12} ({int(q*100):>2}th percentile) | "
            f"budget: {amount/norm_const:6.2f}{unit_label}  "
            f"(~{rounded/norm_const:6.2f}{unit_label})"
        )
        rounded_budgets.append(rounded)

    ############################# Plotting #############################
    visualize_periodic_metrics(period_stats, rounded_budgets)


if __name__ == "__main__":
    main()
