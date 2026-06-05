import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from online_rl.config import SAC_CONFIG


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),       nn.ReLU(),
        )
        self.mean_head    = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)

    def forward(self, state):
        x       = self.shared(state)
        mean    = self.mean_head(x)
        log_std = self.log_std_head(x).clamp(-20, 2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self(state)
        std  = log_std.exp()
        noise  = torch.randn_like(mean)
        z      = mean + std * noise
        action = torch.tanh(z)
        log_prob = (
            torch.distributions.Normal(mean, std).log_prob(z)
            - torch.log(1 - action.pow(2) + 1e-6)
        ).sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic(self, state):
        mean, _ = self(state)
        return torch.tanh(mean)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),                    nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


class SACAgent:
    def __init__(self, cfg: dict = None):
        cfg = cfg or SAC_CONFIG
        sd  = cfg["state_dim"]
        ad  = cfg["action_dim"]

        self.actor    = Actor(sd, ad)
        self.critic1  = Critic(sd, ad)
        self.critic2  = Critic(sd, ad)
        self.target1  = Critic(sd, ad)
        self.target2  = Critic(sd, ad)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())

        self.log_alpha     = torch.zeros(1, requires_grad=True)
        self.target_entropy = -ad * cfg["target_entropy_ratio"]

        lr = cfg["lr"]
        self.actor_opt  = optim.Adam(self.actor.parameters(),    lr=lr)
        self.critic_opt = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr
        )
        self.alpha_opt  = optim.Adam([self.log_alpha], lr=lr)

        self.gamma = cfg["gamma"]
        self.tau   = cfg["tau"]

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        s = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic(s)
            else:
                action, _ = self.actor.sample(s)
        return action.squeeze(0).numpy()

    def update(self, batch) -> dict:
        states, actions, rewards, next_states, dones = batch

        with torch.no_grad():
            next_actions, next_log_pi = self.actor.sample(next_states)
            q1_next = self.target1(next_states, next_actions)
            q2_next = self.target2(next_states, next_actions)
            q_next  = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            q_target = rewards + self.gamma * (1 - dones) * q_next

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        new_actions, log_pi = self.actor.sample(states)
        actor_loss = (self.alpha * log_pi - torch.min(
            self.critic1(states, new_actions),
            self.critic2(states, new_actions),
        )).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

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
            "actor":     self.actor.state_dict(),
            "critic1":   self.critic1.state_dict(),
            "critic2":   self.critic2.state_dict(),
            "target1":   self.target1.state_dict(),
            "target2":   self.target2.state_dict(),
            "log_alpha": self.log_alpha,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self.actor.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        self.target1.load_state_dict(ckpt["target1"])
        self.target2.load_state_dict(ckpt["target2"])
        self.log_alpha = ckpt["log_alpha"]
