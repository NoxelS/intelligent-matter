# intelligent_matter/run_simulation.py

from __future__ import annotations

import numpy as np

from intelligent_matter.env import SwarmConfig, SwarmEnv, AgentType
from intelligent_matter.policies import RandomPolicy, EvoPolicy
from intelligent_matter.visualization import animate_episode
from intelligent_matter.evolutionary import GAConfig, run_ga_training


def run_one_episode_numeric(steps: int = 200) -> None:
    cfg = SwarmConfig(
        n_agents=32,
        frac_sighted=0.25,
        rng_seed=42,
    )
    env = SwarmEnv(cfg)
    policy = RandomPolicy(action_dim=env.action_dim, rng_seed=0)

    obs = env.reset()
    cum_reward = np.zeros(cfg.n_agents)

    for t in range(steps):
        actions = policy.act(obs)  # shape (N,)
        obs, rewards, dones, info = env.step(actions)
        cum_reward += rewards

    print("Average reward per agent:", cum_reward.mean())

    sighted_mask = env.agent_types == AgentType.SIGHTED
    blind_mask = env.agent_types == AgentType.BLIND

    if np.any(sighted_mask):
        print("Average reward sighted:", cum_reward[sighted_mask].mean())
    if np.any(blind_mask):
        print("Average reward blind:", cum_reward[blind_mask].mean())

    print("Finished numeric episode.")


def train_evo_and_animate() -> None:
    # Environment config for training
    env_cfg = SwarmConfig(
        n_agents=64,
        frac_sighted=0.25,
        rng_seed=123,
    )

    # GA hyperparameters (tune as needed)
    ga_cfg = GAConfig(
        population_size=64,
        generations=200,
        hidden_dim=32,
        episodes_per_individual=3,
        steps_per_episode=200,
        elite_fraction=0.10,
        mutation_std=0.075,
        crossover_rate=0.5,
        rng_seed=77,
    )

    # Train
    best_theta, history = run_ga_training(env_cfg, ga_cfg)

    # Show final fitness curve
    print("Fitness history:", history)

    # Create environment and policy for visualization
    env = SwarmEnv(env_cfg)
    policy = EvoPolicy(
        theta=best_theta,
        input_dim=env.obs_dim,
        hidden_dim=ga_cfg.hidden_dim,
        output_dim=env.action_dim,
    )

    # Animate a single episode with the evolved policy
    animate_episode(env, policy, steps=400, interval_ms=40, show=True, save_path=None)


def run_with_animation_random(steps: int = 300) -> None:
    cfg = SwarmConfig(
        n_agents=32,
        frac_sighted=0.25,
        rng_seed=123,
    )
    env = SwarmEnv(cfg)
    policy = RandomPolicy(action_dim=env.action_dim, rng_seed=1)

    animate_episode(env, policy, steps=steps, interval_ms=50, show=True, save_path=None)


if __name__ == "__main__":
    # 1) quick numeric test with random policy
    run_one_episode_numeric(steps=200)

    # 2) random-policy animation (for debugging)
    # run_with_animation_random(steps=300)

    # 3) evolutionary training + animation of best individual
    train_evo_and_animate()
