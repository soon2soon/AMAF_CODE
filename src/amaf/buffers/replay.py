from __future__ import annotations

import random
from collections import deque, namedtuple
from typing import Any

import numpy as np
import torch


class ReplayBuffer:
    """Preallocated replay buffer for vector-state Gymnasium tasks."""

    def __init__(self, state_dim: int, action_dim: int, capacity: int, device: torch.device):
        self.capacity = int(capacity)
        self.device = device
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.next_state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((self.capacity, 1), dtype=np.float32)
        self.done = np.zeros((self.capacity, 1), dtype=np.float32)

    def add(self, state, action, next_state, reward, done) -> None:
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=int(batch_size))
        return tuple(
            torch.as_tensor(x[idx], dtype=torch.float32, device=self.device)
            for x in (self.state, self.action, self.next_state, self.reward, self.done)
        )

    def __len__(self) -> int:
        return self.size


class DiscreteReplayBuffer:
    def __init__(self, state_dim: int, capacity: int, device: torch.device):
        self.capacity = int(capacity)
        self.device = device
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.next_state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((self.capacity, 1), dtype=np.int64)
        self.reward = np.zeros((self.capacity, 1), dtype=np.float32)
        self.done = np.zeros((self.capacity, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done) -> None:
        self.state[self.ptr] = state
        self.action[self.ptr, 0] = int(action)
        self.reward[self.ptr, 0] = float(reward)
        self.next_state[self.ptr] = next_state
        self.done[self.ptr, 0] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=int(batch_size))
        return (
            torch.as_tensor(self.state[idx], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.action[idx], dtype=torch.long, device=self.device),
            torch.as_tensor(self.reward[idx], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.next_state[idx], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.done[idx], dtype=torch.float32, device=self.device),
        )

    def __len__(self) -> int:
        return self.size


CropExperience = namedtuple("CropExperience", ["state", "action", "reward", "next_state", "done"])


class CropReplayBuffer:
    """Object replay buffer preserving raw crop state arrays until LM tokenization."""

    def __init__(self, capacity: int):
        self.memory: deque[CropExperience] = deque(maxlen=int(capacity))

    def add(self, state, action, reward, next_state, done) -> None:
        self.memory.append(CropExperience(np.asarray(state).copy(), int(action), float(reward), np.asarray(next_state).copy(), bool(done)))

    def sample(self, batch_size: int) -> list[CropExperience]:
        return random.sample(self.memory, k=int(batch_size))

    def __len__(self) -> int:
        return len(self.memory)


class BootstrappedCropReplayBuffer:
    def __init__(self, capacity: int, n_heads: int, mask_prob: float = 0.5):
        self.capacity = int(capacity)
        self.n_heads = int(n_heads)
        self.mask_prob = float(mask_prob)
        self.memory: deque[Any] = deque(maxlen=self.capacity)

    def add(self, state, action, reward, next_state, done) -> None:
        mask = np.random.binomial(1, self.mask_prob, size=self.n_heads).astype(np.float32)
        if mask.sum() == 0:
            mask[np.random.randint(self.n_heads)] = 1.0
        self.memory.append((np.asarray(state).copy(), int(action), float(reward), np.asarray(next_state).copy(), bool(done), mask))

    def sample(self, batch_size: int):
        return random.sample(self.memory, k=int(batch_size))

    def __len__(self) -> int:
        return len(self.memory)
