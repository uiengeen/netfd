"""
PPO trainer adapted from CleanRL (https://github.com/vwxyzjn/cleanrl).

Adaptations
-----------
- A list-of-environments rollout loop (the AFD env is custom and lightweight,
  so we don't need gym.vector.SyncVectorEnv)
- No wandb / tensorboard / tyro CLI; everything goes through a `PPOConfig`
  dataclass.
- Otherwise standard CleanRL PPO: GAE, clipped surrogate, value clipping,
  entropy bonus, learning-rate annealing, gradient clipping, advantage
  normalization.
"""

import os
import time
from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


# =============================================================================
# Config
# =============================================================================

@dataclass
class PPOConfig:
    # Training budget
    total_timesteps: int = 200_000
    num_envs: int = 8
    num_steps: int = 64               # rollout length per env per iteration

    # Optimization
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    num_minibatches: int = 4
    update_epochs: int = 4
    max_grad_norm: float = 0.5

    # PPO loss
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    norm_advantages: bool = True
    clip_vloss: bool = True

    # Network
    hidden: int = 128

    # Reproducibility & logging
    seed: int = 1
    verbose: bool = True
    log_every_n_iters: int = 0        # 0 = auto (~25 prints over training)


# =============================================================================
# Agent network: shared input, separate actor/critic heads
# =============================================================================

def _layer_init(layer, std=np.sqrt(2.0), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.critic = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            _layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            _layer_init(nn.Linear(hidden, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            _layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            _layer_init(nn.Linear(hidden, n_actions), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(x)


# =============================================================================
# Trainer
# =============================================================================

def train(env_factory: Callable[[int], "AFDEnv"],
          cfg: PPOConfig = None) -> "tuple[Agent, dict]":
    """Train PPO on the env produced by `env_factory(seed)`.

    Returns
    -------
    agent : trained `Agent`
    log   : dict with per-iteration metrics (step, ep_return, ep_accuracy,
            ep_final_entropy, policy_loss, value_loss, entropy, approx_kl,
            clipfrac).
    """
    if cfg is None:
        cfg = PPOConfig()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    envs = [env_factory(cfg.seed + 17 * i) for i in range(cfg.num_envs)]
    obs_dim = envs[0].observation_space.shape[0]
    n_actions = envs[0].action_space.n

    agent = Agent(obs_dim, n_actions, hidden=cfg.hidden).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    # Rollout storage
    obs_buf = torch.zeros((cfg.num_steps, cfg.num_envs, obs_dim)).to(device)
    actions_buf = torch.zeros((cfg.num_steps, cfg.num_envs), dtype=torch.long).to(device)
    logprobs_buf = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    rewards_buf = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    dones_buf = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    values_buf = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)

    # Initial reset
    next_obs = np.zeros((cfg.num_envs, obs_dim), dtype=np.float32)
    for i, env in enumerate(envs):
        o, _ = env.reset(seed=cfg.seed + 100 * i)
        next_obs[i] = o
    next_obs_t = torch.from_numpy(next_obs).to(device)
    next_done = torch.zeros(cfg.num_envs).to(device)

    batch_size = cfg.num_envs * cfg.num_steps
    minibatch_size = batch_size // cfg.num_minibatches
    num_iterations = max(1, cfg.total_timesteps // batch_size)

    log = {"step": [], "ep_return": [], "ep_accuracy": [], "ep_final_entropy": [],
           "policy_loss": [], "value_loss": [], "entropy": [],
           "approx_kl": [], "clipfrac": []}

    ep_acc_window: List[int] = []
    ep_ent_window: List[float] = []
    ep_ret_window: List[float] = []
    cur_returns = np.zeros(cfg.num_envs, dtype=np.float32)

    log_every = cfg.log_every_n_iters
    if log_every <= 0:
        log_every = max(1, num_iterations // 25)

    global_step = 0
    start = time.time()

    for it in range(1, num_iterations + 1):
        if cfg.anneal_lr:
            frac = 1.0 - (it - 1) / num_iterations
            optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate

        # ----- Rollout -----
        for step in range(cfg.num_steps):
            global_step += cfg.num_envs
            obs_buf[step] = next_obs_t
            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs_t)
                values_buf[step] = value.flatten()
            actions_buf[step] = action
            logprobs_buf[step] = logprob

            new_obs = np.zeros((cfg.num_envs, obs_dim), dtype=np.float32)
            new_rewards = np.zeros(cfg.num_envs, dtype=np.float32)
            new_dones = np.zeros(cfg.num_envs, dtype=np.float32)
            actions_np = action.cpu().numpy()

            for i, env in enumerate(envs):
                o, r, term, trunc, info = env.step(int(actions_np[i]))
                done = bool(term or trunc)
                new_rewards[i] = r
                cur_returns[i] += r
                if done:
                    correct = int(info["predicted_idx"] == info["true_fault_idx"])
                    ep_acc_window.append(correct)
                    ep_ent_window.append(float(info["entropy"]))
                    ep_ret_window.append(float(cur_returns[i]))
                    cur_returns[i] = 0.0
                    o, _ = env.reset(seed=cfg.seed + 100 * i + global_step + it * 7919)
                new_obs[i] = o
                new_dones[i] = float(done)

            rewards_buf[step] = torch.from_numpy(new_rewards).to(device)
            next_obs_t = torch.from_numpy(new_obs).to(device)
            next_done = torch.from_numpy(new_dones).to(device)

        # ----- GAE -----
        with torch.no_grad():
            next_value = agent.get_value(next_obs_t).reshape(1, -1)
            advantages = torch.zeros_like(rewards_buf).to(device)
            lastgaelam = 0
            for t in reversed(range(cfg.num_steps)):
                if t == cfg.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = rewards_buf[t] + cfg.gamma * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = (
                    delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values_buf

        # Flatten
        b_obs = obs_buf.reshape(-1, obs_dim)
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        # ----- PPO update -----
        b_inds = np.arange(batch_size)
        clipfracs = []
        approx_kl_v = 0.0

        for _ in range(cfg.update_epochs):
            np.random.shuffle(b_inds)
            for s in range(0, batch_size, minibatch_size):
                e = s + minibatch_size
                mb = b_inds[s:e]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb], b_actions[mb])
                logratio = newlogprob - b_logprobs[mb]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl_v = float(((ratio - 1) - logratio).mean().item())
                    clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coef)
                                     .float().mean().item())

                mb_adv = b_advantages[mb]
                if cfg.norm_advantages:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if cfg.clip_vloss:
                    v_unclipped = (newvalue - b_returns[mb]) ** 2
                    v_clipped = b_values[mb] + torch.clamp(
                        newvalue - b_values[mb], -cfg.clip_coef, cfg.clip_coef)
                    v_clipped = (v_clipped - b_returns[mb]) ** 2
                    v_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

        # ----- Log -----
        log["step"].append(global_step)
        log["ep_return"].append(float(np.mean(ep_ret_window[-200:])) if ep_ret_window else 0.0)
        log["ep_accuracy"].append(float(np.mean(ep_acc_window[-200:])) if ep_acc_window else 0.0)
        log["ep_final_entropy"].append(float(np.mean(ep_ent_window[-200:])) if ep_ent_window else 0.0)
        log["policy_loss"].append(float(pg_loss.item()))
        log["value_loss"].append(float(v_loss.item()))
        log["entropy"].append(float(entropy_loss.item()))
        log["approx_kl"].append(approx_kl_v)
        log["clipfrac"].append(float(np.mean(clipfracs)))

        if cfg.verbose and (it % log_every == 0 or it == 1 or it == num_iterations):
            print(f"  it {it:4d}/{num_iterations}  step={global_step:7d}  "
                  f"return={log['ep_return'][-1]:5.2f}  "
                  f"acc={log['ep_accuracy'][-1]:.3f}  "
                  f"H_final={log['ep_final_entropy'][-1]:.3f}  "
                  f"pi_loss={pg_loss.item():+.3f}  "
                  f"vf={v_loss.item():.3f}  "
                  f"KL={approx_kl_v:.4f}")

    if cfg.verbose:
        print(f"  done in {time.time() - start:.1f}s, total steps = {global_step}")
    return agent, log


# =============================================================================
# Policy wrapper for evaluation
# =============================================================================

class PPOPolicy:
    def __init__(self, agent: Agent, deterministic: bool = True):
        self.agent = agent
        self.device = next(agent.parameters()).device
        self.deterministic = deterministic

    def reset(self):
        pass

    def act(self, obs):
        x = torch.from_numpy(
            np.asarray(obs, dtype=np.float32)
        ).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.agent.actor(x)
            if self.deterministic:
                a = int(logits.argmax(dim=-1).item())
            else:
                a = int(Categorical(logits=logits).sample().item())
        return a


# =============================================================================
# Save / load
# =============================================================================

def save_agent(agent: Agent, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(agent.state_dict(), path)


def load_agent(path: str, obs_dim: int, n_actions: int,
               hidden: int = 128) -> Agent:
    agent = Agent(obs_dim, n_actions, hidden=hidden)
    agent.load_state_dict(torch.load(path, map_location="cpu"))
    agent.eval()
    return agent
