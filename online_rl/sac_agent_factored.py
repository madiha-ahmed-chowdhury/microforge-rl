import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from online_rl.config import SAC_CONFIG, N_CPU, N_MEMORY, N_TIMEOUT, CPU_BINS, MEMORY_BINS, TIMEOUT_BINS


# ── Discrete SAC with separate per-action trunks ──────────────────────────────
#
# Difference from sac_agent.py: instead of one shared trunk feeding all three
# action heads, each head gets its own 256→128 trunk so CPU, memory, and
# timeout can learn independent representations without competing for capacity.

class _FactoredActor(nn.Module):
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

    def sample(self, state):
        cpu_p, mem_p, tms_p = self(state)
        cpu_idx = torch.multinomial(cpu_p, 1).squeeze(1)
        mem_idx = torch.multinomial(mem_p, 1).squeeze(1)
        tms_idx = torch.multinomial(tms_p, 1).squeeze(1)
        eps = 1e-8
        log_prob = (
            torch.log(cpu_p.gather(1, cpu_idx.unsqueeze(1)) + eps).squeeze(1) +
            torch.log(mem_p.gather(1, mem_idx.unsqueeze(1)) + eps).squeeze(1) +
            torch.log(tms_p.gather(1, tms_idx.unsqueeze(1)) + eps).squeeze(1)
        )
        return (cpu_idx, mem_idx, tms_idx), log_prob

    def deterministic(self, state):
        cpu_p, mem_p, tms_p = self(state)
        return cpu_p.argmax(dim=-1), mem_p.argmax(dim=-1), tms_p.argmax(dim=-1)


class _FactoredCritic(nn.Module):
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
        return (
            self.cpu_head(self.trunk_cpu(state)),
            self.mem_head(self.trunk_mem(state)),
            self.timeout_head(self.trunk_tms(state)),
        )


class DiscreteSACAgentFactored:
    def __init__(self, cfg: dict):
        sd = cfg["state_dim"]

        self.actor   = _FactoredActor(sd)
        self.critic1 = _FactoredCritic(sd)
        self.critic2 = _FactoredCritic(sd)
        self.target1 = _FactoredCritic(sd)
        self.target2 = _FactoredCritic(sd)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())

        self.log_alpha      = torch.zeros(1, requires_grad=True)
        self.target_entropy = cfg["target_entropy"]
        self.tau            = cfg["tau"]
        self.gamma          = cfg["gamma"]

        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=cfg["lr_actor"])
        self.critic_opt = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=cfg["lr_critic"],
        )
        self.alpha_opt = optim.Adam([self.log_alpha], lr=cfg["lr_alpha"])

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, deterministic: bool = False) -> dict:
        s = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                cpu_i, mem_i, tms_i = self.actor.deterministic(s)
            else:
                (cpu_i, mem_i, tms_i), _ = self.actor.sample(s)
        ci = int(cpu_i.item())
        mi = int(mem_i.item())
        ti = int(tms_i.item())
        return {
            "cpu_idx":        ci,
            "mem_idx":        mi,
            "timeout_idx":    ti,
            "cpu_millicores": CPU_BINS[ci],
            "memory_mb":      MEMORY_BINS[mi],
            "timeout_ms":     TIMEOUT_BINS[ti],
        }

    def update(self, batch) -> dict:
        states, cpu_acts, mem_acts, tms_acts, rewards, next_states, dones = batch

        # ── Critic update ────────────────────────────────────────────────────
        with torch.no_grad():
            cpu_np, mem_np, tms_np = self.actor(next_states)
            eps = 1e-8
            cpu_tq1, mem_tq1, tms_tq1 = self.target1(next_states)
            cpu_tq2, mem_tq2, tms_tq2 = self.target2(next_states)
            cpu_v  = (cpu_np * (torch.min(cpu_tq1, cpu_tq2) - self.alpha * torch.log(cpu_np + eps))).sum(dim=1, keepdim=True)
            mem_v  = (mem_np * (torch.min(mem_tq1, mem_tq2) - self.alpha * torch.log(mem_np + eps))).sum(dim=1, keepdim=True)
            tms_v  = (tms_np * (torch.min(tms_tq1, tms_tq2) - self.alpha * torch.log(tms_np + eps))).sum(dim=1, keepdim=True)
            target = rewards + self.gamma * (1 - dones) * (cpu_v + mem_v + tms_v) / 3

        cpu_q1, mem_q1, tms_q1 = self.critic1(states)
        cpu_q2, mem_q2, tms_q2 = self.critic2(states)

        cpu_q1_taken = cpu_q1.gather(1, cpu_acts.unsqueeze(1))
        mem_q1_taken = mem_q1.gather(1, mem_acts.unsqueeze(1))
        tms_q1_taken = tms_q1.gather(1, tms_acts.unsqueeze(1))
        cpu_q2_taken = cpu_q2.gather(1, cpu_acts.unsqueeze(1))
        mem_q2_taken = mem_q2.gather(1, mem_acts.unsqueeze(1))
        tms_q2_taken = tms_q2.gather(1, tms_acts.unsqueeze(1))

        critic_loss = (
            F.mse_loss(cpu_q1_taken, target) + F.mse_loss(cpu_q2_taken, target) +
            F.mse_loss(mem_q1_taken, target) + F.mse_loss(mem_q2_taken, target) +
            F.mse_loss(tms_q1_taken, target) + F.mse_loss(tms_q2_taken, target)
        )

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), 1.0)
        self.critic_opt.step()

        # ── Actor update ─────────────────────────────────────────────────────
        with torch.no_grad():
            cpu_q1a, mem_q1a, tms_q1a = self.critic1(states)
            cpu_q2a, mem_q2a, tms_q2a = self.critic2(states)
            cpu_q = torch.min(cpu_q1a, cpu_q2a)
            mem_q = torch.min(mem_q1a, mem_q2a)
            tms_q = torch.min(tms_q1a, tms_q2a)

        cpu_p, mem_p, tms_p = self.actor(states)
        eps = 1e-8
        cpu_lp = torch.log(cpu_p + eps)
        mem_lp = torch.log(mem_p + eps)
        tms_lp = torch.log(tms_p + eps)

        actor_loss = (
            (cpu_p * (self.alpha.detach() * cpu_lp - cpu_q)).sum(dim=1).mean() +
            (mem_p * (self.alpha.detach() * mem_lp - mem_q)).sum(dim=1).mean() +
            (tms_p * (self.alpha.detach() * tms_lp - tms_q)).sum(dim=1).mean()
        )

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # ── Alpha update ─────────────────────────────────────────────────────
        log_prob = (
            (cpu_p.detach() * cpu_lp.detach()).sum(dim=1) +
            (mem_p.detach() * mem_lp.detach()).sum(dim=1) +
            (tms_p.detach() * tms_lp.detach()).sum(dim=1)
        ).mean()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach())
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        nn.utils.clip_grad_norm_([self.log_alpha], 1.0)
        self.alpha_opt.step()

        # ── Soft target update ───────────────────────────────────────────────
        for target, source in [(self.target1, self.critic1), (self.target2, self.critic2)]:
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
            "alpha_loss":  alpha_loss.item(),
            "alpha":       self.alpha.item(),
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "actor":      self.actor.state_dict(),
            "critic1":    self.critic1.state_dict(),
            "critic2":    self.critic2.state_dict(),
            "target1":    self.target1.state_dict(),
            "target2":    self.target2.state_dict(),
            "log_alpha":  self.log_alpha,
            "actor_opt":  self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "alpha_opt":  self.alpha_opt.state_dict(),
        }, path)

    def load(self, path: str, cfg: dict = None) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self.actor.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        self.target1.load_state_dict(ckpt["target1"])
        self.target2.load_state_dict(ckpt["target2"])
        self.log_alpha = ckpt["log_alpha"]

        if "actor_opt" in ckpt:
            self.actor_opt.load_state_dict(ckpt["actor_opt"])
        if "critic_opt" in ckpt:
            self.critic_opt.load_state_dict(ckpt["critic_opt"])
        if "alpha_opt" in ckpt:
            self.alpha_opt.load_state_dict(ckpt["alpha_opt"])

        if cfg is not None:
            for pg in self.actor_opt.param_groups:
                pg["lr"] = cfg["lr_actor"]
            for pg in self.critic_opt.param_groups:
                pg["lr"] = cfg["lr_critic"]
            for pg in self.alpha_opt.param_groups:
                pg["lr"] = cfg["lr_alpha"]
