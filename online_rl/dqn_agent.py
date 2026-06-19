import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from online_rl.config import DQN_CONFIG, N_CPU, N_MEMORY, N_TIMEOUT, CPU_BINS, MEMORY_BINS, TIMEOUT_BINS


class _QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128),       nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, state):
        return self.net(state)


class DQNAgent:
    def __init__(self, cfg: dict):
        sd = cfg["state_dim"]

        self.cpu_q      = _QNetwork(sd, N_CPU)
        self.mem_q      = _QNetwork(sd, N_MEMORY)
        self.tms_q      = _QNetwork(sd, N_TIMEOUT)
        self.cpu_target = _QNetwork(sd, N_CPU)
        self.mem_target = _QNetwork(sd, N_MEMORY)
        self.tms_target = _QNetwork(sd, N_TIMEOUT)

        self.cpu_target.load_state_dict(self.cpu_q.state_dict())
        self.mem_target.load_state_dict(self.mem_q.state_dict())
        self.tms_target.load_state_dict(self.tms_q.state_dict())

        lr = cfg["lr"]
        self.cpu_opt = optim.Adam(self.cpu_q.parameters(), lr=lr)
        self.mem_opt = optim.Adam(self.mem_q.parameters(), lr=lr)
        self.tms_opt = optim.Adam(self.tms_q.parameters(), lr=lr)

        self.gamma     = cfg["gamma"]
        self.tau       = cfg["tau"]
        self.epsilon   = 1.0
        self._eps_min  = 0.05
        self._eps_decay = 0.995

    def decay_epsilon(self) -> None:
        self.epsilon = max(self._eps_min, self.epsilon * self._eps_decay)

    def select_action(self, state, deterministic: bool = False) -> dict:
        if not deterministic and random.random() < self.epsilon:
            ci = random.randrange(N_CPU)
            mi = random.randrange(N_MEMORY)
            ti = random.randrange(N_TIMEOUT)
        else:
            s = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                ci = int(self.cpu_q(s).argmax(dim=1).item())
                mi = int(self.mem_q(s).argmax(dim=1).item())
                ti = int(self.tms_q(s).argmax(dim=1).item())
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
        rewards = rewards.squeeze(1)
        dones   = dones.squeeze(1)

        losses = {}
        for name, q_net, target_net, acts, opt in [
            ("cpu", self.cpu_q, self.cpu_target, cpu_acts, self.cpu_opt),
            ("mem", self.mem_q, self.mem_target, mem_acts, self.mem_opt),
            ("tms", self.tms_q, self.tms_target, tms_acts, self.tms_opt),
        ]:
            with torch.no_grad():
                next_q = target_net(next_states).max(dim=1).values
                target = rewards + self.gamma * (1 - dones) * next_q

            current_q = q_net(states).gather(1, acts.unsqueeze(1)).squeeze(1)
            loss = F.mse_loss(current_q, target)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
            opt.step()
            losses[f"{name}_loss"] = loss.item()

        for q_net, target_net in [
            (self.cpu_q, self.cpu_target),
            (self.mem_q, self.mem_target),
            (self.tms_q, self.tms_target),
        ]:
            for tp, sp in zip(target_net.parameters(), q_net.parameters()):
                tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

        losses["epsilon"] = self.epsilon
        return losses

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "cpu_q":      self.cpu_q.state_dict(),
            "mem_q":      self.mem_q.state_dict(),
            "tms_q":      self.tms_q.state_dict(),
            "cpu_target": self.cpu_target.state_dict(),
            "mem_target": self.mem_target.state_dict(),
            "tms_target": self.tms_target.state_dict(),
            "epsilon":    self.epsilon,
            "cpu_opt":    self.cpu_opt.state_dict(),
            "mem_opt":    self.mem_opt.state_dict(),
            "tms_opt":    self.tms_opt.state_dict(),
        }, path)

    def load(self, path: str, cfg: dict = None) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self.cpu_q.load_state_dict(ckpt["cpu_q"])
        self.mem_q.load_state_dict(ckpt["mem_q"])
        self.tms_q.load_state_dict(ckpt["tms_q"])
        self.cpu_target.load_state_dict(ckpt["cpu_target"])
        self.mem_target.load_state_dict(ckpt["mem_target"])
        self.tms_target.load_state_dict(ckpt["tms_target"])
        if "epsilon" in ckpt:
            self.epsilon = ckpt["epsilon"]
        if "cpu_opt" in ckpt:
            self.cpu_opt.load_state_dict(ckpt["cpu_opt"])
        if "mem_opt" in ckpt:
            self.mem_opt.load_state_dict(ckpt["mem_opt"])
        if "tms_opt" in ckpt:
            self.tms_opt.load_state_dict(ckpt["tms_opt"])
        if cfg is not None:
            for opt in [self.cpu_opt, self.mem_opt, self.tms_opt]:
                for pg in opt.param_groups:
                    pg["lr"] = cfg["lr"]
