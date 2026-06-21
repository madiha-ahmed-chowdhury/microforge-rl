import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from online_rl.config import PPO_CONFIG, N_CPU, N_MEMORY, N_TIMEOUT, CPU_BINS, MEMORY_BINS, TIMEOUT_BINS


class _RolloutBuffer:
    def __init__(self, rollout_steps: int):
        self.rollout_steps = rollout_steps
        self.clear()

    def push(self, state, cpu_idx: int, mem_idx: int, tms_idx: int,
             log_prob: float, value: float, reward: float, done: bool):
        self.states.append(np.array(state, dtype=np.float32))
        self.cpu_acts.append(int(cpu_idx))
        self.mem_acts.append(int(mem_idx))
        self.tms_acts.append(int(tms_idx))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def is_ready(self) -> bool:
        return len(self.states) >= self.rollout_steps

    def clear(self):
        self.states    = []
        self.cpu_acts  = []
        self.mem_acts  = []
        self.tms_acts  = []
        self.log_probs = []
        self.values    = []
        self.rewards   = []
        self.dones     = []

    def __len__(self) -> int:
        return len(self.states)


class _PPOActor(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.trunk_cpu = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128),       nn.ReLU(),
        )
        self.trunk_mem = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128),       nn.ReLU(),
        )
        self.trunk_tms = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128),       nn.ReLU(),
        )
        self.cpu_head     = nn.Linear(128, N_CPU)
        self.mem_head     = nn.Linear(128, N_MEMORY)
        self.timeout_head = nn.Linear(128, N_TIMEOUT)

    def forward(self, state):
        cpu_probs = F.softmax(self.cpu_head(self.trunk_cpu(state)), dim=-1)
        mem_probs = F.softmax(self.mem_head(self.trunk_mem(state)), dim=-1)
        tms_probs = F.softmax(self.timeout_head(self.trunk_tms(state)), dim=-1)
        return cpu_probs, mem_probs, tms_probs


class _PPOCritic(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),       nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)


class PPOAgent:
    def __init__(self, cfg: dict):
        sd = cfg["state_dim"]
        self.actor  = _PPOActor(sd)
        self.critic = _PPOCritic(sd)

        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=cfg["lr_actor"])
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg["lr_critic"])

        self.gamma        = cfg["gamma"]
        self.gae_lambda   = cfg["gae_lambda"]
        self.clip_epsilon = cfg["clip_epsilon"]
        self.ppo_epochs   = cfg["ppo_epochs"]
        self.entropy_coef = cfg["entropy_coef"]

        self._rollout = _RolloutBuffer(cfg["rollout_steps"])

    # ── Public interface ──────────────────────────────────────────────────────

    def select_action(self, state, deterministic: bool = False) -> dict:
        s = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            cpu_p, mem_p, tms_p = self.actor(s)
            value = self.critic(s).item()

        if deterministic:
            ci = int(cpu_p.argmax(dim=-1).item())
            mi = int(mem_p.argmax(dim=-1).item())
            ti = int(tms_p.argmax(dim=-1).item())
            log_prob = 0.0
        else:
            ci = int(torch.multinomial(cpu_p, 1).item())
            mi = int(torch.multinomial(mem_p, 1).item())
            ti = int(torch.multinomial(tms_p, 1).item())
            eps = 1e-8
            log_prob = float(
                torch.log(cpu_p[0, ci] + eps) +
                torch.log(mem_p[0, mi] + eps) +
                torch.log(tms_p[0, ti] + eps)
            )

        return {
            "cpu_idx":        ci,
            "mem_idx":        mi,
            "timeout_idx":    ti,
            "cpu_millicores": CPU_BINS[ci],
            "memory_mb":      MEMORY_BINS[mi],
            "timeout_ms":     TIMEOUT_BINS[ti],
            "log_prob":       log_prob,
            "value":          value,
        }

    def store(self, state, cpu_idx: int, mem_idx: int, tms_idx: int,
              log_prob: float, value: float, reward: float, done: bool):
        self._rollout.push(state, cpu_idx, mem_idx, tms_idx, log_prob, value, reward, done)

    def ready_to_update(self) -> bool:
        return self._rollout.is_ready()

    def update(self) -> dict:
        buf = self._rollout
        advantages, returns = self._compute_gae(buf.rewards, buf.values, buf.dones)

        states      = torch.FloatTensor(np.array(buf.states))
        cpu_acts    = torch.LongTensor(buf.cpu_acts)
        mem_acts    = torch.LongTensor(buf.mem_acts)
        tms_acts    = torch.LongTensor(buf.tms_acts)
        old_lp      = torch.FloatTensor(buf.log_probs)
        adv         = torch.FloatTensor(advantages)
        ret         = torch.FloatTensor(returns)

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        agg = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

        for _ in range(self.ppo_epochs):
            new_lp, entropy = self._log_probs_entropy(states, cpu_acts, mem_acts, tms_acts)

            ratio   = torch.exp(new_lp - old_lp.detach())
            clipped = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            actor_loss  = -torch.min(ratio * adv, clipped * adv).mean()
            critic_loss = F.mse_loss(self.critic(states), ret)
            entropy_bonus = -self.entropy_coef * entropy
            total_loss = actor_loss + 0.5 * critic_loss + entropy_bonus

            self.actor_opt.zero_grad()
            self.critic_opt.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(),  0.5)
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.actor_opt.step()
            self.critic_opt.step()

            agg["actor_loss"]  += actor_loss.item()  / self.ppo_epochs
            agg["critic_loss"] += critic_loss.item() / self.ppo_epochs
            agg["entropy"]     += entropy.item()     / self.ppo_epochs

        self._rollout.clear()
        return agg

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "actor":      self.actor.state_dict(),
            "critic":     self.critic.state_dict(),
            "actor_opt":  self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
        }, path)

    def load(self, path: str, cfg: dict = None) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        if "actor_opt" in ckpt:
            self.actor_opt.load_state_dict(ckpt["actor_opt"])
        if "critic_opt" in ckpt:
            self.critic_opt.load_state_dict(ckpt["critic_opt"])
        if cfg is not None:
            for pg in self.actor_opt.param_groups:
                pg["lr"] = cfg["lr_actor"]
            for pg in self.critic_opt.param_groups:
                pg["lr"] = cfg["lr_critic"]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            next_val = values[t + 1] if t + 1 < len(values) else 0
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae   = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns

    def _log_probs_entropy(self, states, cpu_acts, mem_acts, tms_acts):
        cpu_p, mem_p, tms_p = self.actor(states)
        eps = 1e-8
        cpu_lp = torch.log(cpu_p.gather(1, cpu_acts.unsqueeze(1)) + eps).squeeze(1)
        mem_lp = torch.log(mem_p.gather(1, mem_acts.unsqueeze(1)) + eps).squeeze(1)
        tms_lp = torch.log(tms_p.gather(1, tms_acts.unsqueeze(1)) + eps).squeeze(1)
        log_prob = cpu_lp + mem_lp + tms_lp

        cpu_ent = -(cpu_p * torch.log(cpu_p + eps)).sum(dim=1)
        mem_ent = -(mem_p * torch.log(mem_p + eps)).sum(dim=1)
        tms_ent = -(tms_p * torch.log(tms_p + eps)).sum(dim=1)
        entropy = (cpu_ent + mem_ent + tms_ent).mean()
        return log_prob, entropy
