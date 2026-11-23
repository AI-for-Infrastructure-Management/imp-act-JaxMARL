"""Roll out a released checkpoint or the segment-based heuristic, and plot the episodes.

RolloutDataGenerator rolls out a MARL checkpoint, HeuristicRolloutDataGenerator the
heuristic; both return the same episode_data dict (see episode_data). RolloutPlotter plots
one episode of it.
"""

import copy
import math
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from common import DARK, EDGE, GOLD, ORANGE, PURPLE, TEAL
from matplotlib import patches
from matplotlib.colors import PowerNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
from omegaconf import OmegaConf

import jaxmarl
from jaxmarl.wrappers.baselines import CTRolloutManager, LogWrapper

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))  # *_road_env.py modules


# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------
def episode_data(
    initial_state, states, actions, rewards, infos, agents, max_timesteps,
    time_first=False, verbose=False,
):
    """Collect a batch of rollouts in one dict of arrays, with episodes on the first axis.

    initial_state and states are the IMP-act environment states before the first step and
    after every step. Per-step arrays (states, actions, rewards, infos) are batch-first
    (episodes, timesteps, ...), or time-first (timesteps, episodes, ...) with time_first.

    Keys and their shapes, with E episodes, T timesteps, N segments (agents) and S damage
    states, e.g. for 10 episodes on Cologne:

        action (10, 50, 60)
        applied_actions (10, 50, 60)
        budget_constraints_applied (10, 50)
        budget_remaining (10, 51)
        edge_beliefs (10, 51, 60, 5)
        edge_observations (10, 51, 60)
        edge_states (10, 51, 60)
        episode_cost (10, 1)
        forced_repair_flags (10, 50, 60)
        forced_replace_constraint_applied (10, 50)
        maintenance_reward (10, 50)
        reward (10, 50)
        terminal_reward (10, 50)
        timesteps (10, 51)
        total_travel_time (10, 50)
        traffic_volumes (10, 50, 60)
        travel_time_reward (10, 50)
        travel_times (10, 50, 60)
    """

    def batch_first(x):
        return jnp.swapaxes(x, 0, 1) if time_first else x

    def with_initial(field):
        """The state field before the first step and after every step: (E, T+1, ...)."""
        return jnp.concatenate(
            [getattr(initial_state, field)[:, None], batch_first(getattr(states, field))], axis=1
        )

    returns = batch_first(infos["returned_episode_returns"] * infos["returned_episode"])
    data = {
        # observations
        "timesteps": with_initial("timestep"),
        "edge_states": with_initial("damage_state"),
        "edge_observations": with_initial("observation"),
        "edge_beliefs": with_initial("belief"),
        # actions
        "action": batch_first(jnp.stack([actions[a] for a in agents], axis=-1)),
        "applied_actions": batch_first(infos["applied_actions"]),
        "forced_replace_constraint_applied": batch_first(infos["forced_replace_constraint_applied"]),
        # traffic
        "travel_times": batch_first(infos["travel_times"]),
        "traffic_volumes": batch_first(infos["traffic_volumes"]),
        "total_travel_time": batch_first(infos["total_travel_time"]),
        # budget
        "budget_remaining": with_initial("budget_remaining"),
        "budget_constraints_applied": batch_first(infos["budget_constraints_applied"]),
        "forced_repair_flags": batch_first(infos["forced_repair_flags"]),
        # rewards
        "reward": batch_first(rewards["__all__"]),
        "episode_cost": returns.mean(axis=-1).sum(axis=-1, keepdims=True),
        "travel_time_reward": batch_first(infos["reward_elements"]["travel_time_reward"]),
        "maintenance_reward": batch_first(infos["reward_elements"]["maintenance_reward"]),
        "terminal_reward": batch_first(infos["reward_elements"]["terminal_reward"]),
    }

    # The rollout must cover the whole episode
    assert (
        data["timesteps"][0, -1] == max_timesteps
    ), f"Last timestep is not the same as the env.max_timesteps: {data['timesteps'][0, -1]} != {max_timesteps}. Make sure to disable the automatic reset of the environment in the last step of the rollout."

    if verbose:
        for key in sorted(data):
            print(f"{key:<35}", data[key].shape)

    return data


def get_single_episode_data(episode_data, episode_idx=0):
    """One episode of the rollout data."""
    return jax.tree.map(lambda x: x[episode_idx], episode_data)


class RolloutDataGenerator:

    def __init__(self, env_name, alg, checkpoint_path, checkpoint_id, test_num_envs=10):
        """Roll out the checkpoint `checkpoint_id` of the run directory `checkpoint_path`.

        The run directory holds config.yaml and
        checkpoints/<vmapped_seed_key>/checkpoint_<id>.safetensors (the layout of the
        released checkpoints).
        """
        self.env_name = env_name
        self.alg = alg
        self.checkpoint_id = checkpoint_id
        vmapped_seed = 0

        # train config
        train_config = OmegaConf.to_container(
            OmegaConf.load(os.path.join(checkpoint_path, "config.yaml"))
        )
        # Saved configs omit NUM_UPDATES, which sets the zero-padding of checkpoint names
        train_config.setdefault(
            "NUM_UPDATES",
            train_config["TOTAL_TIMESTEPS"]
            // train_config["NUM_STEPS"]
            // train_config["NUM_ENVS"],
        )

        if self.alg in ["mappo_rnn", "mappo"]:

            from mappo_rnn_road_env import load_checkpoint_agent, make_get_rollout_data

            # Initialise environment
            env = jaxmarl.make(train_config["ENV_NAME"], **train_config["ENV_KWARGS"])
            _, _, loaded_params, _ = load_checkpoint_agent(
                checkpoint_path, self.checkpoint_id, vmapped_seed, train_config
            )

        elif self.alg in ["ippo_rnn", "ippo"]:
            from ippo_rnn_road_env import load_checkpoint_agent, make_get_rollout_data

            # Initialise environment
            env = jaxmarl.make(train_config["ENV_NAME"], **train_config["ENV_KWARGS"])

            _, loaded_params, _ = load_checkpoint_agent(
                checkpoint_path, self.checkpoint_id, vmapped_seed, train_config
            )

        elif self.alg in ["vdn_rnn", "vdn"]:

            from vdn_rnn_road_env import (
                load_checkpoint_agent,
                make_get_rollout_data,
                env_from_config,
            )

            # Initialise environment
            env, _ = env_from_config(copy.deepcopy(train_config))
            env = CTRolloutManager(env, batch_size=1, preprocess_obs=False)

            train_config["MAX_ACTION_SPACE"] = env.max_action_space
            loaded_params, _, _ = load_checkpoint_agent(
                checkpoint_path, self.checkpoint_id, vmapped_seed, train_config
            )

        elif self.alg in ["qmix_rnn", "qmix"]:
            from qmix_rnn_road_env import (
                load_checkpoint_agent,
                make_get_rollout_data,
                env_from_config,
            )

            # Initialise environment
            env, _ = env_from_config(copy.deepcopy(train_config))
            env = CTRolloutManager(env, batch_size=1, preprocess_obs=False)

            train_config["MAX_ACTION_SPACE"] = env.max_action_space
            loaded_params, _, _, _ = load_checkpoint_agent(
                checkpoint_path, self.checkpoint_id, vmapped_seed, train_config
            )

        elif self.alg in ["pqn_rnn", "pqn"]:
            from pqn_rnn_road_env import (
                load_checkpoint_agent,
                make_get_rollout_data,
                env_from_config,
            )

            # Initialise environment
            env, _ = env_from_config(copy.deepcopy(train_config))
            env = CTRolloutManager(env, batch_size=1, preprocess_obs=False)

            train_config["MAX_ACTION_SPACE"] = env.max_action_space
            loaded_params, _ = load_checkpoint_agent(
                checkpoint_path, self.checkpoint_id, vmapped_seed, train_config
            )

        else:
            raise NotImplementedError(f"Algorithm {self.alg} not supported.")

        self.jit_get_rollout_data = jax.jit(
            make_get_rollout_data(train_config, test_num_envs)
        )
        self.loaded_params = loaded_params
        self.env = env
        self.road_env = env.env

    def generate_rollout_data(self, key, verbose=False):
        """Roll out test_num_envs episodes; see episode_data for the keys and shapes."""
        initial_env_state, env_state, env_act, rewards, infos = self.jit_get_rollout_data(
            key, self.loaded_params
        )
        return episode_data(
            initial_env_state.env_state, env_state.env_state, env_act, rewards, infos,
            self.env.agents, self.road_env.max_timesteps, time_first=True, verbose=verbose,
        )

    get_single_episode_data = staticmethod(get_single_episode_data)


class HeuristicRolloutDataGenerator:

    def __init__(self, config):
        """Roll out the segment-based heuristic with the parameters in `config`."""
        from heuristic_evaluation_road_env import make_get_rollout_data

        # Environment
        env = jaxmarl.make("road_env", map_name=config["map"], record_rollout=True)
        env = LogWrapper(env)
        self.env = env
        self.road_env = env.env

        # Rollout data generator
        self.get_rollout_data = make_get_rollout_data(config)

    def generate_rollout_data(self, key, num_rollouts, verbose=False):
        """Roll out num_rollouts episodes; see episode_data for the keys and shapes."""
        rollout_keys = jax.random.split(key, num_rollouts)
        initial_env_state, env_state, actions, rewards, infos = self.get_rollout_data(
            rollout_keys
        )
        return episode_data(
            initial_env_state.env_state, env_state.env_state, actions, rewards, infos,
            self.env.agents, self.road_env.max_timesteps, verbose=verbose,
        )

    get_single_episode_data = staticmethod(get_single_episode_data)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
# action id: (label, marker style). Dot / ring: low- / high-fidelity inspection (a lens);
# hollow / filled plus: minor / major repair; star: replaced, as new.
ACTIONS = {
    0: ("Low-fidelity inspection", {"marker": ".", "color": "0.5", "markersize": 4}),
    1: (
        "High-fidelity inspection",
        {
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": EDGE,
            "markeredgewidth": 0.6,
            "markersize": 4,
        },
    ),
    2: (
        "Minor repair",
        {
            "marker": "P",
            "fillstyle": "none",
            "markeredgecolor": PURPLE,
            "markeredgewidth": 0.8,
            "markersize": 5.5,
        },
    ),
    3: (
        "Major repair",
        {
            "marker": "P",
            "markerfacecolor": DARK,
            "markeredgecolor": EDGE,
            "markeredgewidth": 0.4,
            "markersize": 5.5,
        },
    ),
    4: (
        "Replace",
        {
            "marker": "*",
            "markerfacecolor": ORANGE,
            "markeredgecolor": EDGE,
            "markeredgewidth": 0.4,
            "markersize": 7,
        },
    ),
}


class RolloutPlotter:
    """
    This function returns a template to fill the episode data.
    The plotting functions use this stencil to create the data for plotting.
    (This is a replacement for the recorder class in NumPy environment)

    N: number of components
    T: max timesteps
    S: number of damage states
    E: number of edges

    # stencil for the plotting data
    plot_data = {
        "edge_states": [],  #! shape: (T+1, N)
        "edge_observations": [],  #! shape: (T+1, N)
        "edge_beliefs": [],  #! shape: (T+1, S, N)
        "action": [],  # shape: (T, N)
        "applied_actions": [],  # shape: (T, N)
        "total_travel_time": [],  # shape: (T,)
        "travel_times": [],  # shape: (T, E)
        "reward": [],  # shape: (T,)
        "travel_time_reward": [],  # shape: (T,)
        "maintenance_reward": [],  # shape: (T,)
        "terminal_reward": [],  # shape: (T,)
        "budget_remaining": [],  #! shape: (T+1,)
        "budget_constraints_applied": [],  # shape: (T,)
        "forced_replace_constraint_applied": [],  # shape: (T,)
        "traffic_volumes": [],  # shape: (T, E)
        "episode_cost": 0,
    }
    """

    def __init__(self, env):
        self.env = env.env
        self.num_components = env.num_agents
        self.num_damage_states = env.env.num_damage_states
        self.max_timesteps = env.env.max_timesteps
        self.base_total_travel_time = env.env.base_total_travel_time

    def _segment_length_ranks(self):
        """Rank 1 is the longest segment."""
        lengths = np.asarray(self.env.segment_lengths)
        order = np.argsort(-lengths)
        ranks = np.empty(len(lengths), dtype=int)
        ranks[order] = np.arange(1, len(lengths) + 1)
        return ranks

    def _plot_deterioration(
        self,
        plot_data,
        components_to_show="all",
        save_kwargs=None,
        fig_width=13,
        ncols=2,
    ):
        """Per segment, the deterioration and action panels of Figure 11 (f).

        Top: belief (grey), true damage state (black), observation (gold) and unsafe states
        (red dashed). Bottom: proposed and executed actions, with overridden proposals shaded
        red. Budget renewals are dashed grey lines; the terminal state is hatched.
        """
        if components_to_show == "all":
            components_to_show = range(self.num_components)
        components_to_show = list(components_to_show)

        T = self.max_timesteps
        S = self.num_damage_states
        t_state = np.arange(T + 1)
        renewal = self.env.budget_renewal_interval
        length_rank = self._segment_length_ranks()
        rejected = np.asarray(plot_data["action"]) != np.asarray(plot_data["applied_actions"])

        def style_axis(ax, xlabel):
            ax.set_xlim(-0.5, T + 0.5)
            ax.set_xticks(np.arange(0, T + 1, 5))
            ax.tick_params(labelbottom=xlabel)
            ax.grid(True, alpha=0.3)
            ax.set_axisbelow(True)
            for t in range(0, T, renewal):
                ax.axvline(t, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, zorder=0)

        def mark_actions(ax, actions, y, **kwargs):
            for a, (_, style) in ACTIONS.items():
                t = np.where(actions == a)[0]
                ax.plot(t, np.full(len(t), y), linestyle="", **style, **kwargs)

        nrows = math.ceil(len(components_to_show) / ncols)
        row_height = 1.45  # inches per segment: deterioration panel, action strip, title
        fig = plt.figure(figsize=(fig_width, nrows * row_height + 1.0))
        outer = fig.add_gridspec(nrows, ncols, hspace=0.35, wspace=0.12,
                                 top=1 - 0.15 / fig.get_figheight(),
                                 bottom=0.85 / fig.get_figheight())

        for i, seg in enumerate(components_to_show):
            row, col = divmod(i, ncols)
            inner = outer[row, col].subgridspec(2, 1, height_ratios=[0.75, 0.35], hspace=0.12)
            ax_det = fig.add_subplot(inner[0])
            ax_act = fig.add_subplot(inner[1], sharex=ax_det)
            last_row = row == nrows - 1 or i + ncols >= len(components_to_show)

            # Deterioration
            ax_det.pcolormesh(t_state, np.arange(S), np.asarray(plot_data["edge_beliefs"])[:, seg, :].T,
                              shading="nearest", cmap="Greys", vmin=0, vmax=1, alpha=0.35,
                              edgecolors="none")
            states = np.asarray(plot_data["edge_states"])[:, seg]
            ax_det.plot(t_state, states, color="black", linewidth=1.4)
            ax_det.plot(t_state, np.asarray(plot_data["edge_observations"])[:, seg], "-o",
                        color=GOLD, markersize=2, linewidth=0.7)
            for t in np.where(states == S - 1)[0]:
                ax_det.axvline(t, color="red", linestyle="--", linewidth=1.0)
            ax_det.axvspan(T - 0.5, T + 0.5, facecolor="none", edgecolor="0.7", hatch="////",
                           linewidth=0)
            # Damage states as in the paper: 1 (worst) to 5 (very good), best at the top
            ax_det.set_ylim(S - 0.5, -0.5)
            ax_det.set_yticks(np.arange(S), S - np.arange(S))
            if col == 0:
                ax_det.set_ylabel("Damage State")
            style_axis(ax_det, xlabel=False)
            ax_det.set_title(f"Segment {seg} (length rank {length_rank[seg]}/{len(length_rank)})")

            # Proposed and executed actions
            mark_actions(ax_act, np.asarray(plot_data["action"])[:, seg], 1, alpha=0.4)
            mark_actions(ax_act, np.asarray(plot_data["applied_actions"])[:, seg], 0)
            for t in np.where(rejected[:, seg])[0]:
                ax_act.axvspan(t - 0.5, t + 0.5, color="red", alpha=0.12, linewidth=0)
            ax_act.set_ylim(-0.6, 1.6)
            ax_act.set_yticks([0, 1], ["Executed", "Proposed"] if col == 0 else ["", ""])
            ax_act.tick_params(axis="y", length=0)
            style_axis(ax_act, xlabel=last_row)
            ax_act.grid(False, axis="y")
            if last_row:
                ax_act.set_xlabel("Timestep")

        fig.legend(
            handles=[
                Line2D([], [], color="black", linewidth=1.4, label="True damage state"),
                Line2D([], [], color=GOLD, marker="o", markersize=2, linewidth=0.7,
                       label="Observation"),
                patches.Patch(facecolor="0.6", alpha=0.35, label="Belief"),
                Line2D([], [], color="red", linestyle="--", label="Unsafe state"),
                patches.Patch(facecolor="red", alpha=0.12, label="Proposal overridden"),
                patches.Patch(facecolor="none", edgecolor="0.7", hatch="////", linewidth=0,
                              label="Terminal state"),
                Line2D([], [], color="gray", linestyle="--", linewidth=0.8, label="Budget renewal"),
                *[Line2D([], [], linestyle="", label=label, **style)
                  for label, style in ACTIONS.values()],
            ],
            loc="lower center", bbox_to_anchor=(0.5, 0), ncol=6, frameon=False,
            columnspacing=0.8, handlelength=1.4, handletextpad=0.4,
        )

        if save_kwargs is not None:
            fig.savefig(**save_kwargs)
        plt.show()

    def _plot_budget(self, plot_data, segment=None, save_kwargs=None, fig_width=6.5, fig_height=1.9):
        """Network budget usage over the episode, as in Figure 11 (c).

        Budget used (%), forced replacements per timestep (red stars with their count) and,
        on the right axis, the proposals the budget rejected across all segments. With
        ``segment``, the timesteps at which that segment's proposal was rejected are marked.
        """
        T = self.max_timesteps
        t_state, t_act = np.arange(T + 1), np.arange(T)
        actions = np.asarray(plot_data["action"])
        applied = np.asarray(plot_data["applied_actions"])
        forced = np.asarray(plot_data["forced_repair_flags"]).astype(bool)
        rejected = actions != applied

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), layout="constrained")
        used = 100 * (1 - np.asarray(plot_data["budget_remaining"]) / float(self.env.budget_amount))
        ax.plot(t_state, used, "-o", color=DARK, linewidth=1.2, markersize=2.5,
                markerfacecolor="white", markeredgewidth=0.8)
        for t, count in enumerate(np.asarray(plot_data["forced_replace_constraint_applied"])):
            if count:
                ax.plot(t, 112, "*", color="red", markersize=6)
                ax.text(t, 119, str(int(count)), color="red", fontsize=6, fontweight="bold",
                        ha="center", va="bottom")
        ax.set_ylim(-5, 135)
        ax.set_yticks([0, 50, 100])
        ax.set_ylabel("Budget Used (%)")
        ax.set_xlim(-0.5, T + 0.5)
        ax.set_xticks(np.arange(0, T + 1, 5))
        ax.set_xlabel("Timestep")
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        for t in range(0, T, self.env.budget_renewal_interval):
            ax.axvline(t, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, zorder=0)

        # Proposals rejected by the budget across all segments (forced replacements excluded)
        budget_rejected = (rejected & ~forced).sum(axis=1)
        ax_rej = ax.twinx()
        ax_rej.bar(t_act, budget_rejected, width=0.7, color="red", alpha=0.25, zorder=0)
        ax_rej.set_ylim(0, 3 * max(budget_rejected.max(), 1))
        ax_rej.set_ylabel("Rejected\nProposals", color="red", alpha=0.7)
        ax_rej.tick_params(axis="y", colors="red", labelsize=6)
        ax.set_zorder(ax_rej.get_zorder() + 1)  # budget line on top of the bars
        ax.patch.set_visible(False)

        handles = [
            Line2D([], [], color=DARK, marker="o", markersize=2.5, markerfacecolor="white",
                   label="Budget used"),
            Line2D([], [], color="gray", linestyle="--", linewidth=0.8, label="Budget renewal"),
            Line2D([], [], color="red", marker="*", linestyle="", markersize=6,
                   label="Forced replacements (count)"),
            patches.Patch(facecolor="red", alpha=0.25, label="Rejected proposals (count)"),
        ]
        if segment is not None:
            seg_rejected = np.where(rejected[:, segment] & ~forced[:, segment])[0]
            ax_rej.plot(seg_rejected, budget_rejected[seg_rejected] + 0.06 * ax_rej.get_ylim()[1],
                        "v", color="black", markersize=3)
            handles.append(Line2D([], [], marker="v", color="black", linestyle="", markersize=3,
                                  label=f"Segment {segment} rejected"))
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=3,
                  frameon=False, columnspacing=0.8, handlelength=1.4)
        ax.set_title("Network budget usage", pad=30)

        if save_kwargs is not None:
            fig.savefig(**save_kwargs)
        plt.show()

    def _plot_traffic_volume_and_travel_times(
        self, plot_data, components_to_show="all", save_kwargs=None, fig_width=10
    ):
        """How maintenance disturbs the traffic of each segment over the episode.

        Heatmaps (segments x timesteps) of the change from the segment's typical (median)
        traffic volume and travel time, so that each segment is compared with itself.
        Executed repairs and replacements are marked: they cut the segment's capacity, which
        raises its travel time and pushes traffic onto other segments. The strip above (a)
        shows how much traffic was rerouted at each timestep.
        """
        if components_to_show == "all":
            components_to_show = range(self.num_components)
        segments = np.asarray(list(components_to_show))

        volumes = np.asarray(plot_data["traffic_volumes"], dtype=float)[:, segments]
        travel_times = np.asarray(plot_data["travel_times"], dtype=float)[:, segments]
        typical_volume = np.median(volumes, axis=0)
        volume_change = 100 * (volumes / typical_volume - 1)
        travel_time_rise = np.clip(100 * (travel_times / np.median(travel_times, axis=0) - 1), 0, None)
        # Rerouted traffic: the volume that moved between segments, as a share of all traffic
        rerouted = 100 * 0.5 * np.abs(volumes - typical_volume).sum(axis=1) / typical_volume.sum()

        applied = np.asarray(plot_data["applied_actions"])[:, segments]
        T, n = applied.shape
        step = 10 if n > 20 else 1
        renewal = self.env.budget_renewal_interval

        heat_height = 0.042 * n
        fig = plt.figure(figsize=(fig_width, 1.4 + 0.55 + 2 * heat_height), layout="constrained")
        grid = fig.add_gridspec(3, 2, height_ratios=[0.55, heat_height, heat_height],
                                width_ratios=[1, 0.015])
        ax_shift = fig.add_subplot(grid[0, 0])
        ax_vol = fig.add_subplot(grid[1, 0], sharex=ax_shift)
        ax_tt = fig.add_subplot(grid[2, 0], sharex=ax_shift)

        def mark_maintenance(ax):
            for a in (2, 3, 4):  # repairs and replacements
                t, seg = np.where(applied == a)
                style = {**ACTIONS[a][1]}
                style["markersize"] = style["markersize"] * 0.5
                ax.plot(t + 0.5, seg + 0.5, linestyle="", **style)

        def mark_renewals(ax):
            for t in range(renewal, T, renewal):
                ax.axvline(t, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)

        # Rerouting strip
        ax_shift.bar(np.arange(T) + 0.5, rerouted, width=0.8, color=DARK)
        ax_shift.set_ylabel("Rerouted\n(% of traffic)")
        ax_shift.set_title("(a) Traffic volume and rerouting", loc="left")
        ax_shift.grid(True, axis="y", alpha=0.3)
        ax_shift.set_axisbelow(True)
        ax_shift.tick_params(labelbottom=False, length=0)
        for side in ("top", "right"):
            ax_shift.spines[side].set_visible(False)
        mark_renewals(ax_shift)

        # Heatmaps
        limit = max(np.abs(volume_change).max(), 1e-9)
        heatmaps = [
            (ax_vol, volume_change, "RdBu_r", TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit),
             "Volume change (%)", None),
            (ax_tt, travel_time_rise, "Reds",
             PowerNorm(gamma=0.5, vmin=0, vmax=max(travel_time_rise.max(), 1e-9)),
             "Travel time rise (%)", "(b) Travel time"),
        ]
        for row, (ax, values, cmap, norm, label, title) in enumerate(heatmaps, start=1):
            mesh = ax.pcolormesh(values.T, cmap=cmap, norm=norm)
            mark_maintenance(ax)
            mark_renewals(ax)
            ax.set_yticks(np.arange(0, n, step) + 0.5, segments[::step])
            ax.set_ylim(n, 0)
            ax.tick_params(length=0)
            ax.set_ylabel("Segment")
            if title:
                ax.set_title(title, loc="left")
            cbar = fig.colorbar(mesh, cax=fig.add_subplot(grid[row, 1]))
            cbar.outline.set_visible(False)
            cbar.ax.tick_params(length=0)
            cbar.set_label(label)
        ax_vol.tick_params(labelbottom=False)
        ax_tt.set_xlim(0, T)
        ax_tt.set_xticks(np.arange(0, T + 1, 5))
        ax_tt.set_xlabel("Timestep")
        ax_shift.legend(
            handles=[
                *[Line2D([], [], linestyle="", label=ACTIONS[a][0], **ACTIONS[a][1])
                  for a in (2, 3, 4)],
                Line2D([], [], color="gray", linestyle="--", linewidth=0.6, label="Budget renewal"),
            ],
            loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=4, frameon=False,
            borderaxespad=0.2, columnspacing=0.8, handletextpad=0.3,
        )

        if save_kwargs is not None:
            fig.savefig(**save_kwargs)
        plt.show()

    def _plot_travel_time_and_rewards(self, plot_data, save_kwargs=None, fig_width=6.5, fig_height=2.4):
        """Network travel time and the cost components over the episode.

        (a) Total travel time relative to the base travel time of the network. (b) Cost per
        timestep, stacked by component. (c) Share of each component in the episode's cost.
        """
        T = self.max_timesteps
        t_act = np.arange(T)
        components = [
            ("Travel time", -np.asarray(plot_data["travel_time_reward"]), DARK),
            ("Maintenance", -np.asarray(plot_data["maintenance_reward"]), TEAL),
            ("Terminal", -np.asarray(plot_data["terminal_reward"]), ORANGE),
        ]

        fig, (ax_tt, ax_cost, ax_share) = plt.subplots(
            1, 3, figsize=(fig_width, fig_height), layout="constrained",
            gridspec_kw={"width_ratios": [1, 1, 0.32]},
        )

        def style_time_axis(ax):
            ax.set_xlim(-0.5, T + 0.5)
            ax.set_xticks(np.arange(0, T + 1, 10))
            ax.set_xlabel("Timestep")
            ax.grid(True, alpha=0.3)
            ax.set_axisbelow(True)
            for t in range(0, T, self.env.budget_renewal_interval):
                ax.axvline(t, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, zorder=0)

        # (a) Travel time
        travel_time = 100 * np.asarray(plot_data["total_travel_time"]) / float(self.base_total_travel_time)
        ax_tt.axhline(100, color="black", linestyle=":", linewidth=0.8)
        ax_tt.plot(t_act, travel_time, "-o", color=DARK, linewidth=1.2, markersize=2.5,
                   markerfacecolor="white", markeredgewidth=0.8)
        ax_tt.set_ylabel("Travel Time (% of base)")
        ax_tt.set_title("(a) Network travel time")
        style_time_axis(ax_tt)

        # (b) Cost per timestep, stacked by component
        bottom = np.zeros(T)
        for label, cost, color in components:
            ax_cost.bar(t_act, cost / 1e9, bottom=bottom, width=0.8, color=color, label=label)
            bottom += cost / 1e9
        ax_cost.set_ylabel(r"Cost ($\times 10^9$)")
        ax_cost.set_title("(b) Cost per timestep")
        style_time_axis(ax_cost)
        ax_cost.legend(frameon=False, loc="upper left")

        # (c) Share of the episode's cost
        totals = np.array([cost.sum() for _, cost, _ in components])
        shares = 100 * totals / totals.sum()
        bottom = 0.0
        for (label, _, color), share in zip(components, shares):
            ax_share.bar(0, share, bottom=bottom, width=0.6, color=color)
            if share >= 4:
                ax_share.text(0, bottom + share / 2, f"{share:.0f}%", ha="center", va="center",
                              color="white", fontsize=7)
            bottom += share
        ax_share.set_xlim(-0.5, 0.5)
        ax_share.set_ylim(0, 100)
        ax_share.set_xticks([])
        ax_share.set_ylabel("Share of Episode Cost (%)")
        ax_share.set_title("(c) Cost share")
        for side in ("top", "right"):
            ax_share.spines[side].set_visible(False)

        if save_kwargs is not None:
            fig.savefig(**save_kwargs)
        plt.show()
