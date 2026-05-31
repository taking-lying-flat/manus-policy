import argparse
import os
from itertools import count

import gymnasium as gym
import numpy as np
import scipy.optimize
import torch
import torch.nn.functional as F

from models import CategoricalPolicy, GaussPolicy, Value
from replay import Memory
from sim import select_action
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


# =============================================================================
# CLI
# =============================================================================
def build_parser():
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
    parser.add_argument("--seed", type=int, default=543)
    parser.add_argument("--batch-size", type=int, default=15000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--max-episodes", type=int, default=0)
    return parser


# =============================================================================
# Environment helpers
# =============================================================================
def _is_discrete(env):
    return isinstance(env.action_space, gym.spaces.Discrete)


def _build_policy(env):
    obs = env.observation_space
    if _is_discrete(env):
        n = int(np.prod(obs.shape))
        return CategoricalPolicy(n, int(env.action_space.n)), Value(n)
    n = int(np.prod(obs.shape))
    return GaussPolicy(n, int(np.prod(env.action_space.shape))), Value(n)


# =============================================================================
# GAE + value optimisation
# =============================================================================
def _compute_gae(rewards, masks, values, gamma, tau):
    T = len(rewards)
    returns = torch.zeros(T, 1)
    advantages = torch.zeros(T, 1)
    prev_return, prev_value, prev_adv = 0.0, 0.0, 0.0
    for i in reversed(range(T)):
        returns[i] = rewards[i] + gamma * prev_return * masks[i]
        delta = rewards[i] + gamma * prev_value * masks[i] - values[i].detach()
        advantages[i] = delta + gamma * tau * prev_adv * masks[i]
        prev_return = returns[i, 0].item()
        prev_value = values[i, 0].detach().item()
        prev_adv = advantages[i, 0].item()
    return returns, advantages


def _optimize_value(value_net, states, returns, l2_reg):
    def fn(flat_params):
        set_flat_params_to(value_net, torch.Tensor(flat_params))
        for p in value_net.parameters():
            if p.grad is not None:
                p.grad.data.fill_(0)
        loss = (value_net(states) - returns).pow(2).mean()
        for p in value_net.parameters():
            loss = loss + p.pow(2).sum() * l2_reg
        loss.backward()
        return (loss.data.double().numpy().item(),
                get_flat_grad_from(value_net).data.double().numpy())
    flat, _, _ = scipy.optimize.fmin_l_bfgs_b(
        fn, get_flat_params_from(value_net).double().numpy(), maxiter=25)
    set_flat_params_to(value_net, torch.Tensor(flat))


# =============================================================================
# Surrogate loss + KL (discrete / continuous)
# =============================================================================
def _build_loss_kl(policy_net, states, actions, advantages, fixed_log_prob, discrete):
    if discrete:
        def get_loss(volatile=False):
            if volatile:
                with torch.no_grad():
                    logits = policy_net(states)
            else:
                logits = policy_net(states)
            return (-advantages * torch.exp(categorical_log_density(actions, logits) - fixed_log_prob)).mean()

        def get_kl():
            logits1 = policy_net(states)
            logits0 = logits1.detach()
            probs0 = F.softmax(logits0, dim=-1)
            logp0 = F.log_softmax(logits0, dim=-1)
            logp1 = F.log_softmax(logits1, dim=-1)
            return (probs0 * (logp0 - logp1)).sum(-1, keepdim=True)
    else:
        def get_loss(volatile=False):
            if volatile:
                with torch.no_grad():
                    m, ls, s = policy_net(states)
            else:
                m, ls, s = policy_net(states)
            return (-advantages * torch.exp(normal_log_density(actions, m, ls, s) - fixed_log_prob)).mean()

        def get_kl():
            m1, ls1, s1 = policy_net(states)
            m0, ls0, s0 = m1.detach(), ls1.detach(), s1.detach()
            kl = ls1 - ls0 + (s0.pow(2) + (m0 - m1).pow(2)) / (2.0 * s1.pow(2)) - 0.5
            return kl.sum(1, keepdim=True)
    return get_loss, get_kl


def update_params(batch, policy_net, value_net, *, discrete, gamma, tau, l2_reg, max_kl, damping):
    rewards = torch.from_numpy(batch.reward)
    masks = torch.from_numpy(batch.mask)
    states = torch.from_numpy(batch.state)
    actions = torch.from_numpy(batch.action).long() if discrete else torch.from_numpy(batch.action)

    returns, advantages = _compute_gae(rewards, masks, value_net(states), gamma, tau)
    _optimize_value(value_net, states, returns, l2_reg)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    with torch.no_grad():
        if discrete:
            fixed_log_prob = categorical_log_density(actions, policy_net(states)).clone()
        else:
            m, ls, s = policy_net(states)
            fixed_log_prob = normal_log_density(actions, m, ls, s).clone()

    get_loss, get_kl = _build_loss_kl(policy_net, states, actions, advantages, fixed_log_prob, discrete)
    trpo_step(policy_net, get_loss, get_kl, max_kl, damping)


# =============================================================================
# Checkpoint
# =============================================================================
def _save(path, policy_net, value_net, running_state, episode, best):
    torch.save({
        "policy": policy_net.state_dict(),
        "value": value_net.state_dict(),
        "running_mean": running_state._stat._mean.copy(),
        "running_std": running_state._stat.std.copy(),
        "running_count": running_state._stat._count,
        "episode": episode,
        "best_reward": best,
    }, path)


# =============================================================================
# Main training loop
# =============================================================================
def main():
    args = build_parser().parse_args()
    torch.set_default_dtype(torch.float64)
    os.makedirs(args.env_name, exist_ok=True)
    save_path = os.path.join(args.env_name, "best.pt")

    env = gym.make(args.env_name)
    discrete = _is_discrete(env)
    policy_net, value_net = _build_policy(env)
    env.reset(seed=args.seed)
    torch.manual_seed(args.seed)
    running_state = ZFilter(env.observation_space.shape, clip=5)
    best_avg = -float("inf")

    for episode in count(1):
        memory = Memory()
        num_steps, reward_batch, num_episodes = 0, 0.0, 0

        # collect rollout
        while num_steps < args.batch_size:
            state, _ = env.reset()
            state = running_state(state)
            reward_sum = 0.0

            for step in range(10000):
                _, action_np = select_action(state, policy_net, discrete=discrete)
                if discrete:
                    step_action = int(action_np.item())
                    action_stored = action_np.squeeze()
                    if action_stored.ndim == 0:
                        action_stored = np.array([action_stored])
                else:
                    step_action = action_np.squeeze(0)
                    action_stored = action_np

                next_state, reward, terminated, truncated, _ = env.step(step_action)
                reward_sum += float(reward)
                next_state = running_state(next_state)
                memory.push(state, action_stored, 0 if terminated or truncated else 1, next_state, reward)
                if args.render:
                    env.render()
                if terminated or truncated:
                    break
                state = next_state

            num_steps += step + 1
            num_episodes += 1
            reward_batch += reward_sum

        reward_batch /= num_episodes

        # TRPO update
        update_params(
            memory.get_all(), policy_net, value_net,
            discrete=discrete, gamma=args.gamma, tau=args.tau,
            l2_reg=args.l2_reg, max_kl=args.max_kl, damping=args.damping,
        )

        # log
        if episode % args.log_interval == 0:
            print(f"\nEpisode {episode}\t reward: {reward_sum:.2f}\t avg: {reward_batch:.2f}")

        # save best
        if reward_batch > best_avg:
            best_avg = reward_batch
            _save(save_path, policy_net, value_net, running_state, episode, best_avg)

        if args.max_episodes > 0 and episode >= args.max_episodes:
            break

    env.close()


if __name__ == "__main__":
    main()
