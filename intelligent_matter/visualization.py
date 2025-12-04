
from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation

from intelligent_matter.env import SwarmEnv, AgentType


def animate_episode(
    env: SwarmEnv,
    policy,
    steps: int = 300,
    interval_ms: int = 50,
    show: bool = True,
    save_path: Optional[str] = None,
):
    """
    Animate one episode of the swarm environment.

    Parameters
    ----------
    env : SwarmEnv
        The environment instance (will be stepped inside this function).
    policy :
        Object with method `act(obs) -> np.ndarray[int]` of shape (N,).
    steps : int
        Number of environment steps to animate.
    interval_ms : int
        Delay between frames in milliseconds (controls playback speed).
    show : bool
        Whether to call plt.show() at the end.
    save_path : str or None
        If given, save the animation to this path (e.g. "swarm.mp4").
        Requires ffmpeg or similar installed for mp4.
    """
    cfg = env.config
    N = env.n_agents

    # Reset env and get initial observations
    obs = env.reset()

    # Precompute static visual information
    # Agent colors: sighted vs blind, will later fade / change for dead agents.
    agent_types = env.agent_types
    base_colors = np.zeros((N, 4), dtype=float)  # RGBA

    # Sighted: blue, Blind: orange
    sighted_mask = agent_types == AgentType.SIGHTED
    blind_mask = agent_types == AgentType.BLIND

    base_colors[sighted_mask] = np.array([0.2, 0.3, 0.9, 1.0])
    base_colors[blind_mask] = np.array([0.9, 0.5, 0.1, 1.0])

    # Figure setup
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0.0, cfg.box_size)
    ax.set_ylim(0.0, cfg.box_size)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Swarm Foraging Simulation")

    # Scatter plot for agents
    scat = ax.scatter(
        env.positions[:, 0],
        env.positions[:, 1],
        s=40,
        c=base_colors,
        edgecolors="k",
        linewidths=0.5,
    )

    # Food patch as a circle
    food_circle = patches.Circle(
        env.food_center,
        radius=cfg.food_radius,
        facecolor="green",
        alpha=0.3,
        edgecolor="none",
    )
    ax.add_patch(food_circle)

    # Text overlay for step counter
    text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )

    # We'll keep some state in a dict to avoid using globals
    state = {
        "obs": obs,
        "cum_reward": np.zeros(N, dtype=float),
        "step": 0,
    }

    def init():
        """Initialize animation frame."""
        scat.set_offsets(env.positions)
        scat.set_facecolors(base_colors)
        food_circle.center = env.food_center
        text.set_text("step = 0")
        return scat, food_circle, text

    def update(frame_idx):
        """Animation update: one environment step."""
        # Unpack state
        obs = state["obs"]

        # Policy action
        actions = policy.act(obs)  # (N,)
        # Step environment
        new_obs, rewards, dones, info = env.step(actions)

        state["obs"] = new_obs
        state["cum_reward"] += rewards
        state["step"] += 1

        # Update agent positions
        scat.set_offsets(env.positions)

        # Update colors: fade dead agents
        colors = base_colors.copy()
        dead = info.get("dead", None)
        if dead is not None and np.any(dead):
            # fade to gray + transparent
            colors[dead] = np.array([0.5, 0.5, 0.5, 0.3])
        scat.set_facecolors(colors)

        # Update food patch position, if you ever relocate it inside env
        food_center = info.get("food_center", None)
        if food_center is not None:
            food_circle.center = food_center

        # Update text
        text.set_text(f"step = {state['step']}")

        return scat, food_circle, text

    ani = FuncAnimation(
        fig,
        update,
        frames=steps,
        init_func=init,
        interval=interval_ms,
        blit=True,
    )

    if save_path is not None:
        # e.g. save_path="swarm.mp4"
        ani.save(save_path, fps=1000.0 / interval_ms)

    if show:
        plt.show()

    return ani