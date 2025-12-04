from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple

import numpy as np


class AgentType(Enum):
    SIGHTED = auto()
    BLIND = auto()


@dataclass
class SwarmConfig:
    n_agents: int = 32
    frac_sighted: float = 0.25  # fraction of sighted agents
    box_size: float = 10.0
    dt: float = 0.1
    speed: float = 0.3
    turn_angle: float = np.pi / 8  # per step for left/right
    food_radius: float = 1.5
    neighbor_radius_sighted: float = 2.0
    neighbor_radius_blind: float = 1.0
    food_sight_range: float = 6.0  # max distance at which food is "seen" by sighted
    energy_per_step: float = 0.01
    energy_per_reward: float = 0.1
    initial_energy: float = 1.0
    periodic_boundary: bool = True
    rng_seed: int | None = None


class SwarmEnv:
    """
    Minimal multi-agent swarm foraging environment.

    - 2D periodic box.
    - Agents move at constant speed with discrete turn actions {-1, 0, +1}.
    - One circular food patch.
    - Each agent gets *individual* reward: 1 if inside the patch, else 0.
    - Observations contain:
        - food bearing + distance (only for sighted agents within sight range),
        - local neighbor density and mean heading,
        - agent type flag (0 = blind, 1 = sighted).
    """

    def __init__(self, config: SwarmConfig | None = None) -> None:
        self.config = config or SwarmConfig()
        self.rng = np.random.default_rng(self.config.rng_seed)

        self.n_agents = self.config.n_agents
        self.agent_types: np.ndarray  # shape (N,), values in {AgentType.SIGHTED, AgentType.BLIND}
        self.positions: np.ndarray  # shape (N, 2)
        self.headings: np.ndarray  # shape (N,)
        self.energy: np.ndarray  # shape (N,)

        self.food_center: np.ndarray  # shape (2,)
        self.t: int = 0

        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def obs_dim(self) -> int:
        # [food_dx, food_dy, food_dist_norm,
        #  local_density, mean_cos, mean_sin,
        #  type_flag]
        return 7

    @property
    def action_dim(self) -> int:
        # actions: -1 (left), 0 (straight), +1 (right)
        return 3

    def reset(self) -> np.ndarray:
        """Reset environment and return initial observations."""
        cfg = self.config

        # Random positions and headings
        self.positions = self.rng.uniform(0.0, cfg.box_size, size=(self.n_agents, 2))
        self.headings = self.rng.uniform(-np.pi, np.pi, size=(self.n_agents,))
        self.energy = np.full(self.n_agents, cfg.initial_energy, dtype=float)

        # Assign types
        n_sighted = int(round(cfg.frac_sighted * self.n_agents))
        types = np.array([AgentType.SIGHTED] * n_sighted +
                         [AgentType.BLIND] * (self.n_agents - n_sighted))
        self.rng.shuffle(types)
        self.agent_types = types

        # Place food patch
        self.food_center = self._sample_food_position()

        self.t = 0
        return self._compute_observations()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Step the environment.

        Parameters
        ----------
        actions : np.ndarray, shape (N,)
            Each action in {-1, 0, +1}: left, straight, right.

        Returns
        -------
        obs : np.ndarray, shape (N, obs_dim)
        rewards : np.ndarray, shape (N,)
        dones : np.ndarray, shape (N,)
        info : dict
        """
        cfg = self.config
        assert actions.shape == (self.n_agents,)

        # Update headings
        turn = (actions.astype(float) - 1.0)  # {0,1,2} -> {-1,0,+1}
        self.headings += turn * cfg.turn_angle

        # Update positions
        disp = cfg.speed * cfg.dt * np.stack(
            (np.cos(self.headings), np.sin(self.headings)), axis=-1
        )
        self.positions += disp

        # Boundary conditions
        if cfg.periodic_boundary:
            self.positions %= cfg.box_size
        else:
            # simple reflecting boundaries
            for i in range(2):
                over = self.positions[:, i] > cfg.box_size
                under = self.positions[:, i] < 0.0
                self.positions[over, i] = 2 * cfg.box_size - self.positions[over, i]
                self.positions[under, i] = -self.positions[under, i]
                # flip heading component
                self.headings[over | under] = np.pi - self.headings[over | under]

        # Compute rewards (inside food patch)
        dpos = self.positions - self.food_center
        if cfg.periodic_boundary:
            dpos = self._apply_min_image(dpos)
        dist2 = np.sum(dpos**2, axis=-1)
        in_food = dist2 < cfg.food_radius**2
        rewards = in_food.astype(float)

        # Energy update and "death"
        self.energy -= cfg.energy_per_step
        self.energy += cfg.energy_per_reward * rewards
        dead = self.energy <= 0.0

        # For now, dead agents just get zeroed observations and no movement;
        # you can later respawn or remove them.
        self.energy[dead] = 0.0

        obs = self._compute_observations()
        dones = dead.astype(bool)

        self.t += 1
        info = {
            "in_food": in_food,
            "dead": dead,
            "food_center": self.food_center.copy(),
        }
        return obs, rewards, dones, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_food_position(self) -> np.ndarray:
        # Keep it away from boundaries a bit
        margin = self.config.food_radius + 0.5
        return self.rng.uniform(
            margin,
            self.config.box_size - margin,
            size=(2,),
        )

    def _apply_min_image(self, dpos: np.ndarray) -> np.ndarray:
        """Apply minimum-image convention for periodic box."""
        L = self.config.box_size
        return (dpos + 0.5 * L) % L - 0.5 * L

    def _compute_observations(self) -> np.ndarray:
        cfg = self.config
        N = self.n_agents

        obs = np.zeros((N, self.obs_dim), dtype=float)

        # Food vector and distance
        dpos = self.positions - self.food_center
        if cfg.periodic_boundary:
            dpos = self._apply_min_image(dpos)
        dist = np.linalg.norm(dpos, axis=-1)

        # For sighted agents within range: normalized direction + normalized distance
        food_visible = (self.agent_types == AgentType.SIGHTED) & (dist < cfg.food_sight_range)
        # Avoid division by zero
        safe_dist = np.maximum(dist, 1e-6)
        food_dir = dpos / safe_dist[:, None]  # (N,2)

        # fill for sighted & visible
        obs[food_visible, 0:2] = food_dir[food_visible]
        obs[food_visible, 2] = dist[food_visible] / cfg.food_sight_range

        # Blind agents (or out-of-range sighted) keep zeros in first 3 components

        # Neighbor statistics (local density + mean heading)
        pos = self.positions
        headings = self.headings
        for i in range(N):
            if self.agent_types[i] == AgentType.SIGHTED:
                R = cfg.neighbor_radius_sighted
            else:
                R = cfg.neighbor_radius_blind

            d = pos - pos[i]
            if cfg.periodic_boundary:
                d = self._apply_min_image(d)
            dist_i = np.linalg.norm(d, axis=-1)
            mask = (dist_i < R) & (dist_i > 0.0)

            if np.any(mask):
                density = float(mask.sum()) / (np.pi * R**2)  # rough density estimate
                mean_cos = np.cos(headings[mask]).mean()
                mean_sin = np.sin(headings[mask]).mean()
            else:
                density = 0.0
                mean_cos = 0.0
                mean_sin = 0.0

            obs[i, 3] = density
            obs[i, 4] = mean_cos
            obs[i, 5] = mean_sin

        # Type flag
        type_flag = (self.agent_types == AgentType.SIGHTED).astype(float)
        obs[:, 6] = type_flag

        return obs
