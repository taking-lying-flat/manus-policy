# https://spinningup.openai.com/en/latest/algorithms/trpo.html
# https://en.wikipedia.org/wiki/Conjugate_gradient_method

from __future__ import annotations

import argparse
from itertools import count
from typing import Callable

import gymnasium as gym
import numpy as np
import scipy.optimize
import torch
import torch.nn.functional as F

from models import CategoricalPolicy, GaussPolicy, Value
from replay import Memory
from state import ZFilter
from trpo import trpo_step
from utils import (
    categorical_log_density,
    get_flat_grad_from,
    get_flat_params_from,
    normal_log_density,
    set_flat_params_to,
)

_ENV_LIST = ["Hopper-v5", "HalfCheetah-v5", "CartPole-v1", "LunarLander-v3"]
UpdateInfo = dict[str, float]


# =============================================================================
# CLI
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {e}" for e in sorted(_ENV_LIST)),
    )
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--env-name", default="Hopper-v5")
    parser.add_argument("--tau", type=float, default=0.97)
    parser.add_argument("--l2-reg", type=float, default=1e-3)
    parser.add_argument("--max-kl", type=float, default=1e-2)
    parser.add_argument("--damping", type=float, default=1e-1)
    parser.add_argument("--batch-size", type=int, default=15000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--cuda", action="store_true")
    return parser


# =============================================================================
# Environment helpers
# =============================================================================
def _is_discrete(env: gym.Env) -> bool:
    return isinstance(env.action_space, gym.spaces.Discrete)


def _build_policy(env: gym.Env) -> tuple[torch.nn.Module, Value]:
    obs = env.observation_space
    n = int(np.prod(obs.shape))
    if _is_discrete(env):
        return CategoricalPolicy(n, int(env.action_space.n)), Value(n)
    return GaussPolicy(n, int(np.prod(env.action_space.shape))), Value(n)


def _select_action(
    state: np.ndarray,
    policy_net: torch.nn.Module,
    *,
    discrete: bool,
) -> np.ndarray:
    param = next(policy_net.parameters())
    state = torch.as_tensor(
        state,
        dtype=param.dtype,
        device=param.device,
    ).view(1, -1)

    with torch.no_grad():
        if discrete:
            logits = policy_net(state)
            probs = torch.softmax(logits, dim=-1)
            return probs.multinomial(num_samples=1).cpu().numpy().reshape(-1)

        mean, _, std = policy_net(state)
        return torch.normal(mean, std).cpu().numpy().reshape(-1)


# =============================================================================
# GAE + value optimisation
# =============================================================================
def _compute_gae(
    rewards: torch.Tensor,
    masks: torch.Tensor,
    gae_masks: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    gamma: float,
    tau: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros((), dtype=rewards.dtype, device=rewards.device)

    for i in reversed(range(rewards.size(0))):
        delta = rewards[i] + gamma * masks[i] * next_values[i] - values[i]
        gae = delta + gamma * tau * gae_masks[i] * gae
        advantages[i] = gae

    returns = advantages + values
    return returns.detach(), advantages.detach()


def _optimize_value(
    value_net: Value,
    states: torch.Tensor,
    returns: torch.Tensor,
    l2_reg: float,
) -> None:
    device = next(value_net.parameters()).device

    def fn(flat_params: np.ndarray) -> tuple[float, np.ndarray]:
        flat_tensor = torch.from_numpy(flat_params).to(device=device)
        set_flat_params_to(value_net, flat_tensor)
        for p in value_net.parameters():
            if p.grad is not None:
                p.grad.zero_()
        loss = (value_net(states) - returns).pow(2).mean()
        l2 = sum(p.pow(2).sum() for p in value_net.parameters()) * l2_reg
        loss = loss + l2
        loss.backward()
        return (
            loss.item(),
            get_flat_grad_from(value_net).cpu().numpy(),
        )

    flat, _, _ = scipy.optimize.fmin_l_bfgs_b(
        func=fn,
        x0=get_flat_params_from(value_net).cpu().numpy(),
        maxiter=25,
    )
    set_flat_params_to(value_net, torch.from_numpy(flat).to(device=device))


# =============================================================================
# Surrogate loss + KL (discrete / continuous)
# =============================================================================
def _build_loss_kl(
    policy_net: torch.nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    fixed_log_prob: torch.Tensor,
    fixed_dist: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    discrete: bool,
) -> tuple[Callable[[bool], torch.Tensor], Callable[[], torch.Tensor]]:
    if discrete:
        def get_loss(eval_mode: bool = False) -> torch.Tensor:
            if eval_mode:
                with torch.no_grad():
                    logits = policy_net(states)
            else:
                logits = policy_net(states)
            return (
                -advantages
                * torch.exp(categorical_log_density(actions, logits) - fixed_log_prob)
            ).mean()

        def get_kl() -> torch.Tensor:
            logits1 = policy_net(states)
            logits0 = fixed_dist
            probs0 = F.softmax(logits0, dim=-1)
            logp0 = F.log_softmax(logits0, dim=-1)
            logp1 = F.log_softmax(logits1, dim=-1)
            return (probs0 * (logp0 - logp1)).sum(-1, keepdim=True)
    else:
        def get_loss(eval_mode: bool = False) -> torch.Tensor:
            if eval_mode:
                with torch.no_grad():
                    m, ls, s = policy_net(states)
            else:
                m, ls, s = policy_net(states)
            return (
                -advantages
                * torch.exp(normal_log_density(actions, m, ls, s) - fixed_log_prob)
            ).mean()

        def get_kl() -> torch.Tensor:
            m1, ls1, s1 = policy_net(states)
            m0, ls0, s0 = fixed_dist
            kl = ls1 - ls0 + (s0.pow(2) + (m0 - m1).pow(2)) / (2.0 * s1.pow(2)) - 0.5
            return kl.sum(1, keepdim=True)
    return get_loss, get_kl


def update_params(
    batch,
    policy_net: torch.nn.Module,
    value_net: Value,
    *,
    discrete: bool,
    gamma: float,
    tau: float,
    l2_reg: float,
    max_kl: float,
    damping: float,
) -> UpdateInfo:
    dtype = torch.get_default_dtype()
    device = next(policy_net.parameters()).device

    # --- tensors on device (NN forward passes) ---
    states = torch.as_tensor(batch.state, dtype=dtype, device=device).view(len(batch.reward), -1)
    next_states = torch.as_tensor(batch.next_state, dtype=dtype, device=device).view(len(batch.reward), -1)
    actions = (
        torch.as_tensor(batch.action, dtype=torch.long, device=device)
        if discrete
        else torch.as_tensor(batch.action, dtype=dtype, device=device)
    )

    # --- tensors on CPU (GAE sequential loop) ---
    rewards = torch.as_tensor(batch.reward, dtype=dtype)
    masks = torch.as_tensor(batch.mask, dtype=dtype)
    gae_masks = torch.as_tensor(batch.gae_mask, dtype=dtype)

    with torch.no_grad():
        values = value_net(states)
        next_values = value_net(next_states)
    returns_cpu, advantages_cpu = _compute_gae(
        rewards=rewards,
        masks=masks,
        gae_masks=gae_masks,
        values=values.cpu(),
        next_values=next_values.cpu(),
        gamma=gamma,
        tau=tau,
    )
    returns = returns_cpu.to(device)

    with torch.no_grad():
        value_loss_before = (value_net(states) - returns).pow(2).mean().item()

    _optimize_value(
        value_net=value_net,
        states=states,
        returns=returns,
        l2_reg=l2_reg,
    )

    with torch.no_grad():
        value_loss_after = (value_net(states) - returns).pow(2).mean().item()

    advantages_cpu = (advantages_cpu - advantages_cpu.mean()) / (advantages_cpu.std(unbiased=False) + 1e-8)
    advantages = advantages_cpu.to(device)

    with torch.no_grad():
        if discrete:
            fixed_logits = policy_net(states).clone()
            fixed_log_prob = categorical_log_density(actions, fixed_logits).clone()
        else:
            fixed_mean, fixed_log_std, fixed_std = policy_net(states)
            fixed_mean = fixed_mean.clone()
            fixed_log_std = fixed_log_std.clone()
            fixed_std = fixed_std.clone()
            fixed_log_prob = normal_log_density(
                x=actions,
                mean=fixed_mean,
                log_std=fixed_log_std,
                std=fixed_std,
            ).clone()

    fixed_dist = fixed_logits if discrete else (fixed_mean, fixed_log_std, fixed_std)
    get_loss, get_kl = _build_loss_kl(
        policy_net=policy_net,
        states=states,
        actions=actions,
        advantages=advantages,
        fixed_log_prob=fixed_log_prob,
        fixed_dist=fixed_dist,
        discrete=discrete,
    )
    _, trpo_info = trpo_step(
        model=policy_net,
        get_loss=get_loss,
        get_kl=get_kl,
        max_kl=max_kl,
        damping=damping,
    )

    info: UpdateInfo = {
        "value_loss_before": value_loss_before,
        "value_loss_after": value_loss_after,
        **trpo_info,
    }
    return info


# =============================================================================
# Main training loop
# =============================================================================
def _print_training_info(
    iteration: int,
    num_steps: int,
    num_episodes: int,
    last_reward: float,
    avg_reward: float,
    update_info: UpdateInfo,
) -> None:
    mean_len = num_steps / max(num_episodes, 1)

    print(
        f"\n[iter] {iteration}"
        f" | steps: {num_steps}"
        f" | episodes: {num_episodes}"
        f" | mean_len: {mean_len:.1f}"
    )
    print(
        "[reward]"
        f" last_episode: {last_reward:.2f}"
        f" | avg_episode: {avg_reward:.2f}"
    )
    print(
        "[update]"
        f" value_loss: {update_info['value_loss_before']:.6f}"
        f" -> {update_info['value_loss_after']:.6f}"
        f" | policy_improve: {update_info['policy_improve']:.6f}"
        f" | kl: {update_info['kl']:.6f}"
        f" | step_frac: {update_info['line_search_step_frac']:.3f}"
    )


def main() -> None:
    args = build_parser().parse_args()
    torch.set_default_dtype(torch.float64)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    env_kwargs = {"render_mode": "human"} if args.render else {}
    env = gym.make(args.env_name, **env_kwargs)
    try:
        discrete = _is_discrete(env)
        policy_net, value_net = _build_policy(env)
        policy_net = policy_net.to(device)
        value_net = value_net.to(device)
        running_state = ZFilter(env.observation_space.shape, clip=5)

        for iteration in count(1):
            memory = Memory()
            num_steps, reward_batch, num_episodes = 0, 0.0, 0

            # collect rollout
            while num_steps < args.batch_size:
                state, _ = env.reset()
                state = running_state(state)
                reward_sum = 0.0

                for step in range(10000):
                    action_np = _select_action(
                        state=state,
                        policy_net=policy_net,
                        discrete=discrete,
                    )
                    if discrete:
                        step_action = int(action_np.reshape(-1)[0])
                    else:
                        step_action = action_np.reshape(-1).astype(np.float64)
                    action_stored = step_action

                    next_state, reward, terminated, truncated, _ = env.step(step_action)
                    reward_sum += float(reward)
                    next_state = running_state(next_state)
                    bootstrap_mask = 0.0 if terminated else 1.0
                    gae_mask = 0.0 if terminated or truncated else 1.0
                    memory.push(
                        state=state,
                        action=action_stored,
                        mask=bootstrap_mask,
                        next_state=next_state,
                        reward=reward,
                        gae_mask=gae_mask,
                    )
                    if args.render:
                        env.render()
                    if terminated or truncated:
                        break
                    state = next_state

                num_steps += step + 1
                num_episodes += 1
                reward_batch += reward_sum

            reward_batch /= num_episodes

            update_info = update_params(
                batch=memory.get_all(),
                policy_net=policy_net,
                value_net=value_net,
                discrete=discrete,
                gamma=args.gamma,
                tau=args.tau,
                l2_reg=args.l2_reg,
                max_kl=args.max_kl,
                damping=args.damping,
            )

            if iteration % args.log_interval == 0:
                _print_training_info(
                    iteration=iteration,
                    num_steps=num_steps,
                    num_episodes=num_episodes,
                    last_reward=reward_sum,
                    avg_reward=reward_batch,
                    update_info=update_info,
                )

            if args.max_iterations > 0 and iteration >= args.max_iterations:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
