import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as patches


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
        self.num_component_actions = len(env.env.action_map)
        self.max_timesteps = env.env.max_timesteps
        self.base_total_travel_time = env.env.base_total_travel_time
        self.initial_edge_volumes = None

    def _plot_deterioration(
        self,
        plot_data,
        components_to_show="all",
        show_proposed_actions=True,
        save_kwargs=None,
        fig_width=14,
        fig_height=None,
    ):

        if components_to_show == "all":
            components_to_show = range(self.num_components)
        elif isinstance(components_to_show, int):
            components_to_show = [components_to_show]
        elif isinstance(components_to_show, list):
            components_to_show = [int(c) for c in components_to_show]
            if any(c >= self.num_components for c in components_to_show):
                raise ValueError(f"components must be less than {self.num_components}")
        else:
            raise ValueError(
                "components must be 'all' or an integer or a list of integers"
            )

        if len(components_to_show) > 2:
            nrows = math.ceil(len(components_to_show) / 2)
            if fig_height is None:
                fig_height = 0.75 * len(components_to_show)
            fig, _ax = plt.subplots(
                nrows, 2, figsize=(fig_width, fig_height), sharex=True, sharey=True
            )
            _ax = np.atleast_2d(_ax)

        else:
            if fig_height is None:
                fig_height = 2.5
            fig, _ax = plt.subplots(
                1, len(components_to_show), figsize=(fig_width, fig_height)
            )
            _ax = np.array([_ax])

        # ticks and labels: actions
        time_horizon_ticks = np.arange(0, self.max_timesteps + 1, 10)
        action_markers = [".", "s", "<", ">", "^"]
        action_labels = [
            "routine inspection",
            "high-fidelity inspection",
            "minor repair",
            "major repair",
            "replace",
        ]
        action_colors = ["gray", "orange", "blue", "dodgerblue", "darkviolet"]
        action_markersize = 7

        # compute components failure timepoints
        plot_data["component_failures"] = np.zeros(
            (self.max_timesteps + 1, self.num_components)
        )
        for t in range(self.max_timesteps + 1):
            # if damage state is num_damage_states-1, then component has failed
            plot_data["component_failures"][t, :] = plot_data["edge_states"][t, :] == (
                self.num_damage_states - 1
            )

        for c, comp in enumerate(components_to_show):

            if len(components_to_show) > 2:
                ax = _ax[c // 2, c % 2]
            else:
                ax = _ax[c]

            # state
            (h_true_state,) = ax.plot(
                plot_data["timesteps"],
                plot_data["edge_states"][:, comp],
                "-",
                label="true state",
                color="tab:green",
                markersize=2,
                alpha=0.5,
            )

            # observation
            (h_obs,) = ax.plot(
                plot_data["timesteps"],
                plot_data["edge_observations"][:, comp],
                "-o",
                label="observation",
                color="tab:blue",
                markersize=2,
                alpha=0.8,
            )

            # belief
            ax.pcolormesh(
                plot_data["timesteps"],
                np.arange(self.num_damage_states),
                plot_data["edge_beliefs"][:, comp, :].T,
                shading="nearest",
                cmap="binary",  # _r for reversed
                alpha=0.2,
                vmin=0,
                vmax=1,
                edgecolors="face",
            )

            # draw vertical lines when component fails
            if plot_data["component_failures"][:, comp].any():
                for t in np.where(plot_data["component_failures"][:, comp])[0]:
                    ax.axvline(t, color="red", linestyle="--", alpha=0.5)

            # Highlight the last timestep with hatching
            last_timestep_start = plot_data["timesteps"][-1] - 0.5
            last_timestep_width = (
                plot_data["timesteps"][-1] - plot_data["timesteps"][-2]
            )
            rect = patches.Rectangle(
                (last_timestep_start, -0.5),  # Lower left corner of the rectangle
                last_timestep_width,  # Width of the rectangle (covers last timestep)
                self.num_damage_states + 1,  # Height of the rectangle
                facecolor="none",  # No fill color
                hatch="\\" * 8,  # Hatching pattern
                edgecolor="black",  # Edge color
                alpha=0.1,  # Transparency
                label="terminal state",
            )
            ax.add_patch(rect)

            ## Plot agent actions
            for a in range(self.num_component_actions):
                # applied actions
                _x = np.where(plot_data["applied_actions"][:, comp] == a)
                ax.plot(
                    _x,
                    2,
                    action_markers[a],
                    markersize=action_markersize,
                    label=action_labels[a],
                    color=action_colors[a],
                )

                if show_proposed_actions:
                    # proposed actions
                    _x = np.where(plot_data["action"][:, comp] == a)
                    ax.plot(
                        _x,
                        3,
                        action_markers[a],
                        markersize=action_markersize,
                        color=action_colors[a],
                        alpha=0.3,
                    )

            ax.set_xlim([-0.5, self.max_timesteps + 0.5])
            ax.set_ylim([-0.5, self.num_damage_states - 0.5])
            ax.set_xticks(time_horizon_ticks)
            ax.set_yticks(np.arange(self.num_damage_states))
            ax.set_xlabel("timestep", fontsize=12)
            ax.set_ylabel("damage state", fontsize=10)
            ax.set_title(
                f"Deterioration, inspection and maintenance | segment: {comp}",
                fontsize=12,
                weight="bold",
            )

            # create legend handles
            legend_handles = []
            for a in range(self.num_component_actions):
                legend_handles += [
                    Line2D(
                        [],
                        [],
                        marker=action_markers[a],
                        markersize=action_markersize,
                        label=action_labels[a],
                        color=action_colors[a],
                        linestyle="None",
                    )
                ]

        legend_handles += [h_true_state, h_obs, rect]
        pcolormesh_proxy = patches.Patch(
            facecolor="gray", alpha=0.2, label="Edge Beliefs"
        )
        legend_handles += [pcolormesh_proxy]
        if plot_data["component_failures"].any():
            legend_handles += [
                Line2D([], [], color="red", linestyle="--", label="unsafe state")
            ]

        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.17),  # slightly lower anchor for better spacing
            ncol=7,  # increase columns to prevent crowding
            # frameon=False,  # cleaner look without box
        )

        # fig.suptitle("Deterioration, Inspection and Maintenance", fontsize=14, weight="bold", y=1.01)
        fig.tight_layout()
        plt.show()

        if save_kwargs is not None:
            fig.savefig(**save_kwargs)

    def _plot_budget(
        self,
        plot_data,
        save_kwargs=None,
        fig_width=8,
        fig_height=4,
    ):

        time = plot_data["timesteps"]

        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height), sharex=True)

        percent_remaining = plot_data["budget_remaining"] / self.env.budget_amount * 100
        percent_used = 100 - percent_remaining

        ax.plot(time, percent_used, "-o", color="blue", alpha=0.5)

        for t in plot_data["timesteps"][:-1]:
            # draw vertical lines for budget renewals
            if t % self.env.budget_renewal_interval == 0:
                ax.axvline(t, color="green", linestyle="--", alpha=0.6)

            # add a cross marker for forced replace constraint
            if plot_data["forced_replace_constraint_applied"][t]:
                ax.plot(t, 50, "x", color="red")

        ax.set_title("Budget Usage", fontsize=12)
        ax.set_ylabel("% budget used", fontsize=12)
        ax.set_ylim([-2, 105])
        # ax.set_yticks(np.arange(0, 101, 10))
        ax.set_xlabel("time", fontsize=12)
        ax.set_xlim([-0.5, self.max_timesteps + 0.5])
        ax.set_xticks(np.arange(0, self.max_timesteps + 1, 10))

        # make a custom legend
        custom_lines = [
            Line2D([], [], color="blue", alpha=0.5, label="budget used"),
            Line2D(
                [], [], color="green", linestyle="--", alpha=0.6, label="budget renewal"
            ),
            Line2D([], [], color="red", marker="x", label="forced replace constraint"),
        ]

        fig.legend(
            handles=custom_lines,
            loc="center right",
            fontsize=10,
            bbox_to_anchor=(0.42, 0.8),
        )
        ax.grid(alpha=0.7)

        if save_kwargs is not None:
            fig.savefig(**save_kwargs)

    def _plot_traffic_volume_and_travel_times(
        self, plot_data, components_to_show="all", save_kwargs=None
    ):

        time = plot_data["timesteps"][:-1]

        fig, _ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

        # Define labels for edges
        if components_to_show == "all":
            components_to_show = range(self.env.graph.ecount())
        elif isinstance(components_to_show, int):
            components_to_show = [components_to_show]
        elif isinstance(components_to_show, list):
            components_to_show = [int(c) for c in components_to_show]
            if any(c >= self.env.graph.ecount() for c in components_to_show):
                raise ValueError(
                    f"components must be less than {self.env.graph.ecount()}"
                )
        else:
            raise ValueError(
                "components must be 'all' or an integer or a list of integers"
            )

        # ticks and labels: actions
        time_horizon_ticks = np.arange(0, self.max_timesteps + 1, 10)
        action_markers = [".", "s", "<", ">", "^"]
        action_labels = [
            "routine inspection",
            "high-fidelity inspection",
            "minor repair",
            "major repair",
            "replace",
        ]
        action_colors = ["gray", "orange", "blue", "dodgerblue", "darkviolet"]
        action_markersize = 7

        labels = [f"edge {i}" for i in components_to_show]
        colors = plt.cm.get_cmap("tab10", len(components_to_show))

        for c, comp in enumerate(components_to_show):
            _ax[0].plot(
                time,
                plot_data["traffic_volumes"][:, comp],
                label=labels[c],
                color=colors(c),  # Use distinct color for each line
                marker="o",  # Optional: add markers to each line
                linestyle="-",  # Keep lines solid
                linewidth=1.5,  # Thicker lines for better visibility
            )
            ## Plot agent actions
            ymin, ymax = _ax[0].get_ylim()
            y_middle = (ymax + ymin) / 2
            for a in range(self.num_component_actions):
                # applied actions
                _x = np.where(plot_data["applied_actions"][:, comp] == a)
                _ax[0].plot(
                    _x,
                    y_middle,
                    action_markers[a],
                    markersize=action_markersize,
                    label=action_labels[a],
                    color=action_colors[a],
                )

            _ax[1].plot(
                time,
                plot_data["travel_times"][:, comp],
                color=colors(c),  # Use distinct color for each line
                marker="x",  # Optional: add markers to each line
                linestyle="-",  # Keep lines solid
                linewidth=1.5,  # Thicker lines for better visibility
            )
            ## Plot agent actions
            ymin, ymax = _ax[1].get_ylim()
            y_middle = (ymax + ymin) / 2
            for a in range(self.num_component_actions):
                # applied actions
                _x = np.where(plot_data["applied_actions"][:, comp] == a)
                _ax[1].plot(
                    _x,
                    y_middle,
                    action_markers[a],
                    markersize=action_markersize,
                    color=action_colors[a],
                )

        _ax[0].set_ylabel("Traffic Volume", fontsize=12)
        _ax[1].set_ylabel("Travel Time", fontsize=12)

        for ax in _ax:
            ax.set_xlim([-0.5, self.max_timesteps + 0.5])
            # ax.set_xticks(np.arange(0, self.max_timesteps + 1, 1))
            ax.grid(True, which="both", linewidth=0.5)
            ax.set_xlabel("timestep", fontsize=12)

        fig.suptitle("Traffic Data | segment: 13 ", fontsize=12, weight="bold")
        # fig.legend(loc="center right", bbox_to_anchor=(1.1, 0.5), fontsize=10)
        fig.tight_layout()

        if save_kwargs:
            fig.savefig(**save_kwargs)

        plt.show()

    def _plot_travel_time_and_rewards(self, plot_data, save_kwargs=None):

        time = plot_data["timesteps"][:-1]

        fig, _ax = plt.subplots(1, 3, figsize=(18, 5))

        # plot total travel time
        ax = _ax[0]
        ax.plot(
            time,
            plot_data["total_travel_time"],
            label="total travel time",
            color="black",
        )
        ax.axhline(
            self.base_total_travel_time,
            linestyle="--",
            color="red",
            label="base travel time",
        )
        ax.set_ylabel("total travel time", fontsize=12)
        ax.set_title("Total Travel Time", fontsize=14)

        # plot rewards
        ax = _ax[1]
        ax.plot(time, plot_data["reward"], label="total reward")
        ax.plot(
            time,
            plot_data["travel_time_reward"],
            label="travel time reward",
        )
        ax.plot(
            time,
            plot_data["maintenance_reward"],
            label="maintenance reward",
        )
        ax.plot(
            time,
            plot_data["terminal_reward"],
            label="terminal reward",
        )
        ax.set_ylabel("reward", fontsize=12)
        ax.set_title("Reward Components", fontsize=14)

        for ax in _ax[:-1]:
            ax.set_xlabel("time", fontsize=12)
            ax.set_xlim([-0.5, self.max_timesteps + 0.5])
            ax.set_xticks(np.arange(0, self.max_timesteps + 1, 10))
            ax.grid()
            ax.legend()

        # plot reward pie chart
        ax = _ax[2]
        x1 = -plot_data["travel_time_reward"].sum()
        x2 = -plot_data["maintenance_reward"].sum()
        x3 = -plot_data["terminal_reward"].sum()
        ax.pie(
            [x1, x2, x3],
            labels=["travel time", "maintenance", "terminal"],
            autopct="%1.1f%%",
            startangle=90,
        )
        ax.set_title("Reward Distribution", fontsize=14)

        plt.show()

        if save_kwargs is not None:
            fig.savefig(**save_kwargs)


def plot_action_stats(action_stats_data, save_kwargs=None):
    """
    Plot the action statistics for each agent.

    Parameters
    ----------
    action_stats_data : np.ndarray
    shape: (NUM_EPISODES, NUM_TIMESTEPS, NUM_AGENTS)
    action_stats_data[i, j, k] = action taken by agent k at timestep j in episode i

    """

    NUM_EPISODES, NUM_TIMESTEPS, NUM_AGENTS = action_stats_data.shape

    nrows = math.ceil(NUM_AGENTS / 2)
    fig, _ax = plt.subplots(
        nrows, 2, figsize=(10, NUM_AGENTS * 1), sharex=True, sharey=True
    )

    # make bins so that the histogram markers align with the action space
    _bins = range(
        min(action_stats_data.flatten()), max(action_stats_data.flatten()) + 2
    )

    for c in range(NUM_AGENTS):
        ax = _ax[c // 2, c % 2]

        ax.hist(
            action_stats_data[:, :, c].flatten(),
            bins=_bins,
            density=True,
            align="left",
            rwidth=0.4,
            orientation="horizontal",
            color="dodgerblue",
        )
        # put text on top of the bars
        for i in range(len(_bins) - 1):
            count = np.sum(action_stats_data[:, :, c] == i)
            _density = count / (NUM_TIMESTEPS * NUM_EPISODES)
            ax.text(
                _density,
                _bins[i],
                f"{_density:.6f}",
                ha="left",
                va="center",
                fontsize=10,
            )

        ax.set_title(f"Component {c}", fontsize=12, weight="bold")
        ax.set_yticks(
            ticks=np.arange(0, 5),
            labels=[
                "do-nothing",
                "inspect",
                "minor-repair",
                "major-repair",
                "replace",
            ],
        )
        ax.grid(axis="x", linestyle="--", alpha=0.7)

    fig.suptitle(
        f"Agent-wise Action Density ({NUM_EPISODES} rollouts)",
        fontsize=14,
        weight="bold",
        y=1,
    )
    fig.tight_layout()
    plt.show()

    if save_kwargs is not None:
        fig.savefig(**save_kwargs)
