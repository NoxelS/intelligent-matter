
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class RandomPolicy:
    action_dim: int
    rng_seed: int | None = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.rng_seed)

    def act(self, obs: np.ndarray) -> np.ndarray:
        """
        obs: (N, obs_dim)
        returns actions: (N,) in {0,1,2} for now.
        """
        N = obs.shape[0]
        return self.rng.integers(0, self.action_dim, size=N, dtype=int)


@dataclass
class MLPPolicy:
    """
    Simple 2-layer MLP policy with tanh activations and categorical output.

    The parameters are provided as a single flat vector `theta`.
    This makes it easy to use with evolutionary algorithms.
    """

    input_dim: int
    hidden_dim: int
    output_dim: int
    theta: np.ndarray  # flat parameter vector

    def __post_init__(self) -> None:
        self.theta = np.asarray(self.theta, dtype=float)
        assert self.theta.ndim == 1
        expected = self.num_params(self.input_dim, self.hidden_dim, self.output_dim)
        if self.theta.size != expected:
            raise ValueError(
                f"theta has wrong size {self.theta.size}, expected {expected}"
            )

    @staticmethod
    def num_params(input_dim: int, hidden_dim: int, output_dim: int) -> int:
        # W1 (in x h) + b1 (h) + W2 (h x out) + b2 (out)
        return input_dim * hidden_dim + hidden_dim + hidden_dim * output_dim + output_dim

    @staticmethod
    def _unpack_theta(
        theta: np.ndarray, input_dim: int, hidden_dim: int, output_dim: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Unpack flat theta into (W1, b1, W2, b2)."""
        idx = 0
        w1_size = input_dim * hidden_dim
        W1 = theta[idx : idx + w1_size].reshape(input_dim, hidden_dim)
        idx += w1_size

        b1 = theta[idx : idx + hidden_dim]
        idx += hidden_dim

        w2_size = hidden_dim * output_dim
        W2 = theta[idx : idx + w2_size].reshape(hidden_dim, output_dim)
        idx += w2_size

        b2 = theta[idx : idx + output_dim]
        return W1, b1, W2, b2

    def act(self, obs: np.ndarray) -> np.ndarray:
        """
        obs: (N, input_dim)
        returns actions: (N,) in {0,1,2,..., output_dim-1}
        """
        W1, b1, W2, b2 = self._unpack_theta(
            self.theta, self.input_dim, self.hidden_dim, self.output_dim
        )

        x = obs  # (N, input_dim)
        h = np.tanh(x @ W1 + b1)        # (N, hidden_dim)
        logits = h @ W2 + b2           # (N, output_dim)

        # Categorical sampling with softmax
        # For evolutionary evaluation, argmax is often fine; but we can add a bit of stochasticity.
        # Here: greedy (argmax) for simplicity.
        actions = np.argmax(logits, axis=-1)
        return actions.astype(int)


@dataclass
class EvoPolicy:
    """
    Thin wrapper to make an evolved theta usable as a 'policy' with .act().
    """

    theta: np.ndarray
    input_dim: int
    hidden_dim: int
    output_dim: int

    def act(self, obs: np.ndarray) -> np.ndarray:
        mlp = MLPPolicy(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            theta=self.theta,
        )
        return mlp.act(obs)