# intelligent_matter/evolutionary.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List

import numpy as np

from intelligent_matter.env import SwarmEnv, SwarmConfig
from intelligent_matter.policies import MLPPolicy


@dataclass
class GAConfig:
    population_size: int = 50
    generations: int = 100
    hidden_dim: int = 16
    episodes_per_individual: int = 3
    steps_per_episode: int = 200
    elite_fraction: float = 0.2   # fraction of top individuals kept as elites
    mutation_std: float = 0.1     # std of Gaussian noise added to params
    crossover_rate: float = 0.5   # probability to take gene from parent A vs B
    rng_seed: int | None = 0


def evaluate_individual(
    theta: np.ndarray,
    env_config: SwarmConfig,
    ga_cfg: GAConfig,
    rng: np.random.Generator,
) -> float:
    """
    Evaluate a single individual (policy parameters) by running several episodes
    and returning average reward per agent per step.

    Fitness is defined as:
        fitness = total_reward / (episodes * steps_per_episode * n_agents)
    """
    # Local copy of config with new seed per individual (avoid correlations)
    env_cfg = SwarmConfig(**vars(env_config))
    env_cfg.rng_seed = rng.integers(0, 2**31 - 1)

    env = SwarmEnv(env_cfg)

    policy = MLPPolicy(
        input_dim=env.obs_dim,
        hidden_dim=ga_cfg.hidden_dim,
        output_dim=env.action_dim,
        theta=theta,
    )

    total_reward = 0.0
    n_agents = env.n_agents

    for ep in range(ga_cfg.episodes_per_individual):
        obs = env.reset()
        for t in range(ga_cfg.steps_per_episode):
            actions = policy.act(obs)
            obs, rewards, dones, info = env.step(actions)
            total_reward += rewards.sum()

    denom = ga_cfg.episodes_per_individual * ga_cfg.steps_per_episode * n_agents
    fitness = total_reward / denom
    return float(fitness)


def crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator,
    rate: float,
) -> np.ndarray:
    """
    Simple uniform crossover: each gene is taken from parent A with probability `rate`,
    otherwise from parent B.
    """
    assert parent_a.shape == parent_b.shape
    mask = rng.random(size=parent_a.shape) < rate
    child = np.where(mask, parent_a, parent_b)
    return child


def mutate(theta: np.ndarray, rng: np.random.Generator, std: float) -> np.ndarray:
    """
    Add Gaussian noise with standard deviation `std`.
    """
    noise = rng.normal(loc=0.0, scale=std, size=theta.shape)
    return theta + noise


def run_ga_training(
    env_config: SwarmConfig,
    ga_cfg: GAConfig,
) -> Tuple[np.ndarray, List[float]]:
    """
    Run a simple genetic algorithm to evolve a shared MLP policy.

    Returns
    -------
    best_theta : np.ndarray
        Parameters of the best individual found.
    history : list[float]
        Best fitness per generation.
    """
    rng = np.random.default_rng(ga_cfg.rng_seed)

    # Create a temporary env to know obs_dim and action_dim
    tmp_env = SwarmEnv(env_config)
    param_dim = MLPPolicy.num_params(
        input_dim=tmp_env.obs_dim,
        hidden_dim=ga_cfg.hidden_dim,
        output_dim=tmp_env.action_dim,
    )

    # Initialize population with small Gaussian random weights
    population = rng.normal(
        loc=0.0, scale=0.5, size=(ga_cfg.population_size, param_dim)
    )

    n_elite = max(1, int(ga_cfg.elite_fraction * ga_cfg.population_size))

    history: List[float] = []

    for gen in range(ga_cfg.generations):
        # Evaluate population
        fitnesses = np.zeros(ga_cfg.population_size, dtype=float)
        for i in range(ga_cfg.population_size):
            fitnesses[i] = evaluate_individual(
                theta=population[i],
                env_config=env_config,
                ga_cfg=ga_cfg,
                rng=rng,
            )

        # Sort by fitness (descending)
        idx_sorted = np.argsort(fitnesses)[::-1]
        population = population[idx_sorted]
        fitnesses = fitnesses[idx_sorted]

        best_fitness = float(fitnesses[0])
        history.append(best_fitness)
        print(f"[Gen {gen+1:03d}] best fitness = {best_fitness:.4f}")

        # Elitism: keep top n_elite unchanged
        elites = population[:n_elite].copy()

        # Create new population
        new_population = [elites[0]]  # keep very best
        # Fill the rest of the population
        while len(new_population) < ga_cfg.population_size:
            # Select two parents (tournament or roulette; here: random among elites)
            parents_idx = rng.integers(0, n_elite, size=2)
            pa = elites[parents_idx[0]]
            pb = elites[parents_idx[1]]

            child = crossover(pa, pb, rng, rate=ga_cfg.crossover_rate)
            child = mutate(child, rng, std=ga_cfg.mutation_std)
            new_population.append(child)

        population = np.stack(new_population, axis=0)

    # Final evaluation to ensure we return the true best
    fitnesses = np.zeros(ga_cfg.population_size, dtype=float)
    for i in range(ga_cfg.population_size):
        fitnesses[i] = evaluate_individual(
            theta=population[i],
            env_config=env_config,
            ga_cfg=ga_cfg,
            rng=rng,
        )
    best_idx = int(np.argmax(fitnesses))
    best_theta = population[best_idx].copy()
    best_final = float(fitnesses[best_idx])
    print(f"Final best fitness = {best_final:.4f}")

    return best_theta, history
