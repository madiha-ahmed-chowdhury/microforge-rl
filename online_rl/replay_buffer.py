import random
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self.buffer   = []
        self.pos      = 0

    def push(self, state, action, reward, next_state, done):
        entry = (
            np.array(state,      dtype=np.float32),
            np.array(action,     dtype=np.float32),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        )
        if len(self.buffer) < self.capacity:
            self.buffer.append(entry)
        else:
            self.buffer[self.pos] = entry
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(rewards).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)

    def is_ready(self, n: int) -> bool:
        return len(self.buffer) >= n


class DiscreteReplayBuffer:
    """Replay buffer for DiscreteSACAgent — stores (cpu_idx, mem_idx, timeout_idx) as ints."""

    def __init__(self, capacity: int = 20000):
        self.capacity = capacity
        self.buffer   = []
        self.pos      = 0

    def push(self, state, cpu_idx: int, mem_idx: int, timeout_idx: int,
             reward: float, next_state, done: bool):
        entry = (
            np.array(state,      dtype=np.float32),
            int(cpu_idx),
            int(mem_idx),
            int(timeout_idx),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        )
        if len(self.buffer) < self.capacity:
            self.buffer.append(entry)
        else:
            self.buffer[self.pos] = entry
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, cpu_acts, mem_acts, tms_acts, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(cpu_acts),
            torch.LongTensor(mem_acts),
            torch.LongTensor(tms_acts),
            torch.FloatTensor(rewards).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)

    def is_ready(self, n: int) -> bool:
        return len(self.buffer) >= n
