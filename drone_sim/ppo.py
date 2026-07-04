"""PPO with GAE for the batched quadrotor environment.

Everything stays on one torch device (MPS on Apple silicon), so rollouts and
updates never round-trip through numpy.
"""

import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from drone_sim.env import ACT_DIM, OBS_DIM, QuadrotorEnv


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM, hidden: int = 128):
        super().__init__()

        def mlp():
            return nn.Sequential(
                nn.Linear(obs_dim, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
            )

        self.actor_body = mlp()
        self.mu = nn.Linear(hidden, act_dim)
        self.critic = nn.Sequential(mlp(), nn.Linear(hidden, 1))
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.7))
        nn.init.uniform_(self.mu.weight, -0.01, 0.01)
        nn.init.zeros_(self.mu.bias)

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mu = torch.tanh(self.mu(self.actor_body(obs)))
        return torch.distributions.Normal(mu, self.log_std.clamp(-4.0, 1.0).exp())

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False):
        d = self.dist(obs)
        a = d.mean if deterministic else d.sample()
        return a, d.log_prob(a).sum(-1), self.value(obs)


@dataclass
class PPOConfig:
    rollout_steps: int = 64
    updates: int = 300
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    epochs: int = 5
    minibatches: int = 8
    value_coef: float = 0.5
    entropy_coef: float = 1e-3
    max_grad_norm: float = 0.5


class PPOTrainer:
    def __init__(self, env: QuadrotorEnv, config: PPOConfig | None = None,
                 device: str = "cpu"):
        self.env = env
        self.cfg = config or PPOConfig()
        self.device = torch.device(device)
        self.net = ActorCritic().to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)
        self.log: list[dict] = []

    def train(self, log_every: int = 10, on_update=None):
        cfg, env, net = self.cfg, self.env, self.net
        n, T = env.n, cfg.rollout_steps
        dev = self.device

        obs_buf = torch.zeros(T, n, OBS_DIM, device=dev)
        act_buf = torch.zeros(T, n, ACT_DIM, device=dev)
        logp_buf = torch.zeros(T, n, device=dev)
        rew_buf = torch.zeros(T, n, device=dev)
        val_buf = torch.zeros(T, n, device=dev)
        done_buf = torch.zeros(T, n, device=dev)

        obs = env.reset()
        # Running episode-return bookkeeping for logging.
        ep_ret = torch.zeros(n, device=dev)
        ep_len = torch.zeros(n, device=dev)
        recent_returns, recent_lengths, recent_dists = [], [], []
        t_start = time.time()

        for update in range(1, cfg.updates + 1):
            for t in range(T):
                action, logp, value = net.act(obs)
                obs_buf[t], act_buf[t] = obs, action
                logp_buf[t], val_buf[t] = logp, value

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated | truncated

                # Bootstrap truncated (time-limit) episodes with V(final_obs).
                if truncated.any():
                    with torch.no_grad():
                        vf = net.value(info["final_obs"])
                    reward = reward + cfg.gamma * vf * truncated.float()

                rew_buf[t] = reward
                done_buf[t] = done.float()

                ep_ret += reward
                ep_len += 1
                if done.any():
                    ids = done.nonzero(as_tuple=True)[0]
                    recent_returns += ep_ret[ids].tolist()
                    recent_lengths += ep_len[ids].tolist()
                    recent_dists += info["dist"][ids].tolist()
                    ep_ret[ids] = 0.0
                    ep_len[ids] = 0.0

            # ---------------------------------------------------------- GAE
            with torch.no_grad():
                next_value = net.value(obs)
            adv = torch.zeros(T, n, device=dev)
            lastgae = torch.zeros(n, device=dev)
            for t in reversed(range(T)):
                nv = next_value if t == T - 1 else val_buf[t + 1]
                # No bootstrapping across autoreset boundaries: terminal states
                # get V=0, truncated ones were already bootstrapped into reward.
                nv = nv * (1.0 - done_buf[t])
                delta = rew_buf[t] + cfg.gamma * nv - val_buf[t]
                lastgae = delta + cfg.gamma * cfg.gae_lambda * (1.0 - done_buf[t]) * lastgae
                adv[t] = lastgae
            ret = adv + val_buf

            b_obs = obs_buf.reshape(-1, OBS_DIM)
            b_act = act_buf.reshape(-1, ACT_DIM)
            b_logp = logp_buf.reshape(-1)
            b_adv = adv.reshape(-1)
            b_ret = ret.reshape(-1)
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

            # -------------------------------------------------------- update
            batch = T * n
            mb = batch // cfg.minibatches
            for _ in range(cfg.epochs):
                perm = torch.randperm(batch, device=dev)
                for i in range(cfg.minibatches):
                    idx = perm[i * mb:(i + 1) * mb]
                    d = net.dist(b_obs[idx])
                    logp = d.log_prob(b_act[idx]).sum(-1)
                    ratio = (logp - b_logp[idx]).exp()
                    surr1 = ratio * b_adv[idx]
                    surr2 = ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * b_adv[idx]
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = (net.value(b_obs[idx]) - b_ret[idx]).pow(2).mean()
                    entropy = d.entropy().sum(-1).mean()
                    loss = (policy_loss + cfg.value_coef * value_loss
                            - cfg.entropy_coef * entropy)
                    self.opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                    self.opt.step()

            # ----------------------------------------------------------- log
            if recent_returns:
                mean_ret = sum(recent_returns) / len(recent_returns)
                mean_len = sum(recent_lengths) / len(recent_lengths)
                mean_dist = sum(recent_dists) / len(recent_dists)
            else:
                mean_ret = mean_len = mean_dist = float("nan")
            entry = {
                "update": update,
                "steps": update * T * n,
                "mean_return": mean_ret,
                "mean_ep_len": mean_len,
                "mean_final_dist": mean_dist,
                "value_loss": value_loss.item(),
                "policy_loss": policy_loss.item(),
            }
            self.log.append(entry)
            recent_returns, recent_lengths, recent_dists = [], [], []

            if update % log_every == 0 or update == 1:
                sps = entry["steps"] / (time.time() - t_start)
                print(
                    f"upd {update:4d} | steps {entry['steps']:>9,} | "
                    f"return {mean_ret:8.1f} | ep_len {mean_len:6.1f} | "
                    f"dist {mean_dist:5.2f} m | {sps:,.0f} steps/s",
                    flush=True,
                )
            if on_update is not None:
                on_update(self, entry)

        return self.log

    def save(self, path: str):
        torch.save({"model": self.net.state_dict(), "log": self.log}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.net.load_state_dict(ckpt["model"])
        self.log = ckpt.get("log", [])
