---
sidebar_position: 3
title: Chapter 11 - Reinforcement Learning for Robot Control
---

# Chapter 11 - Reinforcement Learning for Robot Control

## Learning Objectives

By the end of this chapter, you will be able to:
- Implement reinforcement learning algorithms for robot control tasks
- Design reward functions for robotic manipulation and navigation
- Create simulation environments for RL training in robotics
- Apply deep reinforcement learning techniques to robot control
- Integrate learned policies with real-time robot control systems
- Optimize RL training for sample efficiency and safety
- Evaluate and validate learned robot control policies
- Transfer learned policies from simulation to real robots (sim-to-real)

## Prerequisites

Before starting this chapter, you should have:
- Understanding of machine learning fundamentals
- Knowledge of reinforcement learning concepts (MDPs, value functions, policies)
- Experience with deep learning frameworks (PyTorch/TensorFlow)
- Understanding of robot kinematics and dynamics
- Completion of Module 1 and Module 3 Chapter 10 content
- Familiarity with Isaac Sim and perception systems

## Introduction to Reinforcement Learning in Robotics

### Why Reinforcement Learning for Robotics?

Reinforcement Learning (RL) has emerged as a powerful approach for robotic control, offering several advantages over traditional control methods:

**Adaptability**: RL agents can learn to adapt to new environments and changing conditions without explicit programming.

**Complex Task Learning**: RL can learn complex behaviors that are difficult to program manually, such as dexterous manipulation or navigation in cluttered environments.

**Optimization**: RL naturally optimizes for task-specific objectives through reward maximization.

**Generalization**: Well-designed RL systems can generalize to unseen situations within their training distribution.

### RL Framework for Robotics

The RL framework consists of:
- **Agent**: The robot or control system
- **Environment**: The physical or simulated world
- **State**: Robot and environment state (positions, velocities, sensor readings)
- **Action**: Control commands sent to the robot
- **Reward**: Scalar feedback signal indicating task success
- **Policy**: Mapping from states to actions

### Key Challenges in Robot RL

**Sample Efficiency**: Real robots are expensive to operate; RL algorithms need to learn with minimal samples.

**Safety**: Learning processes must ensure robot and human safety during training.

**Continuous Action Spaces**: Robots often have continuous control inputs, requiring specialized RL algorithms.

**Real-time Constraints**: Robot control systems must respond within strict timing requirements.

## Deep Reinforcement Learning Algorithms for Robotics

### Deep Q-Network (DQN) for Discrete Actions

DQN is suitable for robotic tasks with discrete action spaces, such as navigation with predefined waypoints.

```python
# dqn_robot_control.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import gym
from typing import List, Tuple

class DQN(nn.Module):
    """
    Deep Q-Network for discrete action spaces in robotics.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(DQN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

class DQNAgent:
    """
    DQN agent for robotic control with experience replay and target network.
    """
    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-4,
                 gamma: float = 0.99, epsilon: float = 1.0,
                 epsilon_decay: float = 0.995, epsilon_min: float = 0.01,
                 memory_size: int = 100000, batch_size: int = 32):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Network components
        self.q_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network = DQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Hyperparameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        # Replay memory
        self.memory = deque(maxlen=memory_size)

        # Update target network
        self.update_target_network()

    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy."""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_dim)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())

    def replay(self):
        """Train on batch of experiences from replay memory."""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filepath: str):
        """Save model weights."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)

    def load(self, filepath: str):
        """Load model weights."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
```

### Deep Deterministic Policy Gradient (DDPG) for Continuous Control

DDPG is ideal for continuous action spaces typical in robot control.

```python
# ddpg_robot_control.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class Actor(nn.Module):
    """
    Actor network that maps states to actions.
    """
    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 hidden_dim: int = 256):
        super(Actor, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.network(state)

class Critic(nn.Module):
    """
    Critic network that estimates Q-values.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Critic, self).__init__()

        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        return q1, q2

    def Q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], 1)
        return self.q1(sa)

class DDPGAgent:
    """
    DDPG agent for continuous control in robotics.
    """
    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 lr_actor: float = 1e-4, lr_critic: float = 1e-3,
                 gamma: float = 0.99, tau: float = 0.005,
                 noise_std: float = 0.2, noise_clip: float = 0.5,
                 memory_size: int = 100000, batch_size: int = 100):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Networks
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Initialize target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.noise_clip = noise_clip
        self.batch_size = batch_size

        # Replay memory
        self.memory = deque(maxlen=memory_size)

    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Select action with optional noise for exploration."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state_tensor).cpu().data.numpy().flatten()

        if add_noise:
            noise = np.random.normal(0, self.noise_std, size=len(action))
            action = action + noise
            action = np.clip(action, -1, 1)  # Clip to valid range

        return action

    def remember(self, state: np.ndarray, action: np.ndarray,
                 reward: float, next_state: np.ndarray, done: bool):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        """Train the agent on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.FloatTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)

        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_Q1, target_Q2 = self.critic_target(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (self.gamma * target_Q * ~dones)

        current_Q1, current_Q2 = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_loss = -self.critic.Q1(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filepath: str):
        """Save model weights."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
        }, filepath)

    def load(self, filepath: str):
        """Load model weights."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
```

### Twin Delayed DDPG (TD3) for Improved Stability

TD3 addresses overestimation bias in DDPG, making it more stable for robotic control.

```python
# td3_robot_control.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class TD3Agent:
    """
    Twin Delayed DDPG agent for stable continuous control in robotics.
    """
    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 lr_actor: float = 1e-4, lr_critic: float = 1e-3,
                 gamma: float = 0.99, tau: float = 0.005,
                 policy_noise: float = 0.2, noise_clip: float = 0.5,
                 policy_freq: int = 2,
                 memory_size: int = 100000, batch_size: int = 100):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Networks (same as DDPG but with twin critics)
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic_1 = Critic(state_dim, action_dim).to(self.device)
        self.critic_1_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr_critic)

        self.critic_2 = Critic(state_dim, action_dim).to(self.device)
        self.critic_2_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr_critic)

        # Initialize target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        # Hyperparameters
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.batch_size = batch_size

        # Training step counter
        self.total_it = 0

        # Replay memory
        self.memory = deque(maxlen=memory_size)

    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Select action with optional target policy smoothing."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state_tensor).cpu().data.numpy().flatten()

        if add_noise:
            noise = np.random.normal(0, self.policy_noise, size=len(action))
            noise = np.clip(noise, -self.noise_clip, self.noise_clip)
            action = action + noise
            action = np.clip(action, -self.max_action, self.max_action)

        return action

    def remember(self, state: np.ndarray, action: np.ndarray,
                 reward: float, next_state: np.ndarray, done: bool):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        """Train the agent with TD3 algorithm."""
        if len(self.memory) < self.batch_size:
            return

        self.total_it += 1

        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.FloatTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)

        # Select action according to policy and add clipped noise
        noise = torch.FloatTensor(actions).data.normal_(0, self.policy_noise).to(self.device)
        noise = noise.clamp(-self.noise_clip, self.noise_clip)
        next_actions = (self.actor_target(next_states) + noise).clamp(-self.max_action, self.max_action)

        # Compute target Q-value
        target_Q1, target_Q2 = self.critic_1_target(next_states, next_actions)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = rewards + (self.gamma * target_Q * ~dones)

        # Critic loss
        current_Q1, current_Q2 = self.critic_1(states, actions), self.critic_2(states, actions)

        critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)

        # Optimize Critics
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic_1.Q1(states, self.actor(states)).mean()

            # Optimize Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target networks
            for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filepath: str):
        """Save model weights."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_1_state_dict': self.critic_1.state_dict(),
            'critic_1_target_state_dict': self.critic_1_target.state_dict(),
            'critic_2_state_dict': self.critic_2.state_dict(),
            'critic_2_target_state_dict': self.critic_2_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_1_optimizer_state_dict': self.critic_1_optimizer.state_dict(),
            'critic_2_optimizer_state_dict': self.critic_2_optimizer.state_dict(),
            'total_it': self.total_it
        }, filepath)

    def load(self, filepath: str):
        """Load model weights."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
        self.critic_1_target.load_state_dict(checkpoint['critic_1_target_state_dict'])
        self.critic_2.load_state_dict(checkpoint['critic_2_state_dict'])
        self.critic_2_target.load_state_dict(checkpoint['critic_2_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_1_optimizer.load_state_dict(checkpoint['critic_1_optimizer_state_dict'])
        self.critic_2_optimizer.load_state_dict(checkpoint['critic_2_optimizer_state_dict'])
        self.total_it = checkpoint['total_it']
```

### Soft Actor-Critic (SAC) for Sample Efficiency

SAC is particularly effective for robotic control due to its sample efficiency and stability.

```python
# sac_robot_control.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class GaussianPolicy(nn.Module):
    """
    Stochastic policy for SAC with reparameterization trick.
    """
    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 hidden_dim: int = 256, log_std_min: float = -20,
                 log_std_max: float = 2):
        super(GaussianPolicy, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to get mean and log std."""
        features = self.network(state)
        mean = self.mean_linear(features)
        log_std = self.log_std_linear(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action with reparameterization trick."""
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action

        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob, torch.tanh(mean * self.max_action)

class QNetwork(nn.Module):
    """
    Q-network for SAC.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(QNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass to get Q-value."""
        sa = torch.cat([state, action], 1)
        return self.network(sa)

class SACAgent:
    """
    Soft Actor-Critic agent for robotic control.
    """
    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 lr_actor: float = 1e-4, lr_critic: float = 1e-3,
                 lr_alpha: float = 1e-4, gamma: float = 0.99,
                 tau: float = 0.005, alpha: float = 0.2,
                 target_update_interval: int = 1,
                 automatic_entropy_tuning: bool = True,
                 memory_size: int = 100000, batch_size: int = 256):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Networks
        self.actor = GaussianPolicy(state_dim, action_dim, max_action).to(self.device)
        self.critic_1 = QNetwork(state_dim, action_dim).to(self.device)
        self.critic_1_target = QNetwork(state_dim, action_dim).to(self.device)
        self.critic_2 = QNetwork(state_dim, action_dim).to(self.device)
        self.critic_2_target = QNetwork(state_dim, action_dim).to(self.device)

        # Initialize target networks
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr_critic)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr_critic)

        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning

        # Auto-tune alpha
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(action_dim).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

        # Replay memory
        self.memory = deque(maxlen=memory_size)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Select action from policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if evaluate:
            _, _, action = self.actor.sample(state_tensor)
        else:
            action, _, _ = self.actor.sample(state_tensor)

        return action.cpu().data.numpy().flatten()

    def remember(self, state: np.ndarray, action: np.ndarray,
                 reward: float, next_state: np.ndarray, done: bool):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        """Train the agent with SAC algorithm."""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.FloatTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)

        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_states)
            target_q_min = torch.min(
                self.critic_1_target(next_states, next_action),
                self.critic_2_target(next_states, next_action)
            )
            target_q = rewards + (1 - dones) * self.gamma * (target_q_min - self.alpha * next_log_prob)

        # Critic Loss
        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)

        critic_1_loss = nn.MSELoss()(q1, target_q)
        critic_2_loss = nn.MSELoss()(q2, target_q)

        # Optimize Critic
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # Compute actor loss
        pi, log_pi, _ = self.actor.sample(states)

        q1_pi = self.critic_1(states, pi)
        q2_pi = self.critic_2(states, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = ((self.alpha * log_pi) - min_q_pi).mean()

        # Optimize Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Auto tune alpha
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()

        # Update target networks
        if self.total_it % self.target_update_interval == 0:
            for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filepath: str):
        """Save model weights."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_1_state_dict': self.critic_1.state_dict(),
            'critic_1_target_state_dict': self.critic_1_target.state_dict(),
            'critic_2_state_dict': self.critic_2.state_dict(),
            'critic_2_target_state_dict': self.critic_2_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_1_optimizer_state_dict': self.critic_1_optimizer.state_dict(),
            'critic_2_optimizer_state_dict': self.critic_2_optimizer.state_dict(),
            'alpha': self.alpha,
            'log_alpha': self.log_alpha,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict() if self.automatic_entropy_tuning else None
        }, filepath)

    def load(self, filepath: str):
        """Load model weights."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
        self.critic_1_target.load_state_dict(checkpoint['critic_1_target_state_dict'])
        self.critic_2.load_state_dict(checkpoint['critic_2_state_dict'])
        self.critic_2_target.load_state_dict(checkpoint['critic_2_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_1_optimizer.load_state_dict(checkpoint['critic_1_optimizer_state_dict'])
        self.critic_2_optimizer.load_state_dict(checkpoint['critic_2_optimizer_state_dict'])
        self.alpha = checkpoint['alpha']
        self.log_alpha = checkpoint['log_alpha']

        if self.automatic_entropy_tuning:
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-4)
            if checkpoint['alpha_optimizer_state_dict'] is not None:
                self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
```

## Simulation Environments for RL Training

### Custom Robot Environment

Creating custom environments is crucial for training RL agents on specific robotic tasks.

```python
# robot_environment.py
import gym
from gym import spaces
import numpy as np
import pygame
from typing import Dict, Tuple, Optional
import math

class RobotArmEnv(gym.Env):
    """
    Custom environment for robot arm control using RL.
    """
    def __init__(self,
                 max_episode_steps: int = 500,
                 goal_tolerance: float = 0.1,
                 control_frequency: float = 100.0):
        super(RobotArmEnv, self).__init__()

        # Environment parameters
        self.max_episode_steps = max_episode_steps
        self.goal_tolerance = goal_tolerance
        self.control_frequency = control_frequency
        self.dt = 1.0 / control_frequency

        # Robot parameters (2-DOF planar arm)
        self.link_lengths = [0.5, 0.5]  # meters
        self.max_velocity = 2.0  # rad/s

        # Action space: joint velocity commands
        self.action_space = spaces.Box(
            low=-self.max_velocity,
            high=self.max_velocity,
            shape=(2,),
            dtype=np.float32
        )

        # Observation space: [joint_angles, joint_velocities, goal_position, current_end_effector_pos]
        obs_dim = 2 + 2 + 2 + 2  # 8 total dimensions
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Initialize state
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        # Random initial joint angles
        self.joint_angles = np.random.uniform(-np.pi, np.pi, 2)
        self.joint_velocities = np.zeros(2)

        # Random goal position (within workspace)
        goal_radius = np.random.uniform(0.3, 0.9)  # 30-90% of max reach
        goal_angle = np.random.uniform(0, 2*np.pi)
        self.goal_position = np.array([
            goal_radius * np.cos(goal_angle),
            goal_radius * np.sin(goal_angle)
        ])

        # Episode step counter
        self.current_step = 0

        # Compute initial end effector position
        self.end_effector_pos = self.forward_kinematics(self.joint_angles)

        return self.get_observation()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step of the environment."""
        # Clip action
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Update joint velocities
        self.joint_velocities = action

        # Update joint angles (Euler integration)
        self.joint_angles += self.joint_velocities * self.dt
        self.joint_angles = np.arctan2(
            np.sin(self.joint_angles),
            np.cos(self.joint_angles)
        )  # Normalize angles to [-pi, pi]

        # Compute new end effector position
        self.end_effector_pos = self.forward_kinematics(self.joint_angles)

        # Calculate reward
        reward = self.calculate_reward()

        # Check termination conditions
        done = self.check_termination()

        # Increment step counter
        self.current_step += 1

        # Info dictionary
        info = {
            'distance_to_goal': self.compute_distance_to_goal(),
            'end_effector_pos': self.end_effector_pos,
            'goal_pos': self.goal_position,
            'joint_angles': self.joint_angles
        }

        return self.get_observation(), reward, done, info

    def get_observation(self) -> np.ndarray:
        """Get current observation."""
        obs = np.concatenate([
            self.joint_angles,
            self.joint_velocities,
            self.goal_position,
            self.end_effector_pos
        ])
        return obs

    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """Compute end effector position from joint angles."""
        q1, q2 = joint_angles

        # First link end
        x1 = self.link_lengths[0] * np.cos(q1)
        y1 = self.link_lengths[0] * np.sin(q1)

        # Second link end (end effector)
        x2 = x1 + self.link_lengths[1] * np.cos(q1 + q2)
        y2 = y1 + self.link_lengths[1] * np.sin(q1 + q2)

        return np.array([x2, y2])

    def compute_distance_to_goal(self) -> float:
        """Compute distance from end effector to goal."""
        return np.linalg.norm(self.end_effector_pos - self.goal_position)

    def calculate_reward(self) -> float:
        """Calculate reward based on current state."""
        distance = self.compute_distance_to_goal()

        # Dense reward based on distance reduction
        reward = -distance  # Negative distance encourages getting closer

        # Bonus for reaching goal
        if distance < self.goal_tolerance:
            reward += 100.0  # Large bonus for reaching goal

        # Penalty for joint limits (if near limits)
        joint_limit_penalty = 0.0
        for angle in self.joint_angles:
            if abs(abs(angle) - np.pi) < 0.2:  # Within 0.2 rad of joint limit
                joint_limit_penalty -= 1.0

        reward += joint_limit_penalty

        return reward

    def check_termination(self) -> bool:
        """Check if episode should terminate."""
        # Terminate if goal reached
        if self.compute_distance_to_goal() < self.goal_tolerance:
            return True

        # Terminate if max steps reached
        if self.current_step >= self.max_episode_steps:
            return True

        # Terminate if joints are in dangerous positions
        # (This is a simplified check - real robots would have more complex safety checks)
        for angle in self.joint_angles:
            if abs(angle) > np.pi * 0.95:  # Near joint limits
                return True

        return False

    def render(self, mode='human'):
        """Render the environment (for visualization)."""
        if mode == 'human':
            # Initialize pygame for visualization
            if not hasattr(self, 'screen'):
                pygame.init()
                self.screen = pygame.display.set_mode((800, 600))
                pygame.display.set_caption('Robot Arm Environment')
                self.clock = pygame.time.Clock()

            # Clear screen
            self.screen.fill((255, 255, 255))

            # Scale factor for drawing
            scale = 200
            center = np.array([400, 300])

            # Draw goal
            goal_screen = center + self.goal_position * scale
            pygame.draw.circle(self.screen, (0, 255, 0), goal_screen.astype(int), 10)

            # Draw robot arm
            # Base of arm
            base_screen = center
            pygame.draw.circle(self.screen, (0, 0, 0), base_screen.astype(int), 5)

            # First link
            q1, q2 = self.joint_angles
            elbow_pos = base_screen + np.array([
                self.link_lengths[0] * np.cos(q1),
                self.link_lengths[0] * np.sin(q1)
            ]) * scale

            pygame.draw.line(self.screen, (0, 0, 255), base_screen, elbow_pos, 5)

            # Second link
            end_effector_screen = elbow_pos + np.array([
                self.link_lengths[1] * np.cos(q1 + q2),
                self.link_lengths[1] * np.sin(q1 + q2)
            ]) * scale

            pygame.draw.line(self.screen, (0, 0, 255), elbow_pos, end_effector_screen, 5)

            # Draw end effector
            pygame.draw.circle(self.screen, (255, 0, 0), end_effector_screen.astype(int), 8)

            # Update display
            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS

class NavigationEnv(gym.Env):
    """
    Navigation environment for mobile robot RL.
    """
    def __init__(self,
                 map_size: Tuple[float, float] = (10.0, 10.0),
                 max_episode_steps: int = 1000,
                 robot_radius: float = 0.3,
                 goal_tolerance: float = 0.5):
        super(NavigationEnv, self).__init__()

        self.map_size = map_size
        self.max_episode_steps = max_episode_steps
        self.robot_radius = robot_radius
        self.goal_tolerance = goal_tolerance

        # Action space: [linear_velocity, angular_velocity]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -np.pi]),  # max linear vel, max angular vel
            high=np.array([1.0, np.pi]),
            dtype=np.float32
        )

        # Observation space: [robot_x, robot_y, robot_theta, goal_x, goal_y, obstacles_relative_pos]
        obs_dim = 2 + 1 + 2 + 10 * 2  # 25 total (assuming 10 obstacles)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Initialize state
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        # Random robot position (away from borders)
        margin = self.robot_radius + 1.0
        self.robot_pos = np.array([
            np.random.uniform(margin, self.map_size[0] - margin),
            np.random.uniform(margin, self.map_size[1] - margin)
        ])

        # Random robot orientation
        self.robot_theta = np.random.uniform(-np.pi, np.pi)

        # Random goal position (away from robot and borders)
        while True:
            self.goal_pos = np.array([
                np.random.uniform(margin, self.map_size[0] - margin),
                np.random.uniform(margin, self.map_size[1] - margin)
            ])
            # Ensure goal is not too close to robot
            if np.linalg.norm(self.goal_pos - self.robot_pos) > 3.0:
                break

        # Generate random obstacles
        self.obstacles = []
        num_obstacles = np.random.randint(3, 8)
        for _ in range(num_obstacles):
            while True:
                obs_pos = np.array([
                    np.random.uniform(margin, self.map_size[0] - margin),
                    np.random.uniform(margin, self.map_size[1] - margin)
                ])

                # Check that obstacle is not too close to robot or goal
                if (np.linalg.norm(obs_pos - self.robot_pos) > 1.5 and
                    np.linalg.norm(obs_pos - self.goal_pos) > 1.5):
                    self.obstacles.append(obs_pos)
                    break

        self.current_step = 0

        return self.get_observation()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step of the environment."""
        # Clip action
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Extract action components
        linear_vel, angular_vel = action

        # Update robot state (simple differential drive model)
        dt = 0.1  # Time step
        self.robot_theta += angular_vel * dt
        self.robot_theta = np.arctan2(
            np.sin(self.robot_theta),
            np.cos(self.robot_theta)
        )  # Normalize angle

        dx = linear_vel * np.cos(self.robot_theta) * dt
        dy = linear_vel * np.sin(self.robot_theta) * dt

        new_pos = self.robot_pos + np.array([dx, dy])

        # Check collision with environment boundaries
        if (new_pos[0] < self.robot_radius or
            new_pos[0] > self.map_size[0] - self.robot_radius or
            new_pos[1] < self.robot_radius or
            new_pos[1] > self.map_size[1] - self.robot_radius):
            # Boundary collision - don't update position
            pass
        else:
            # Check collision with obstacles
            collision = False
            for obs_pos in self.obstacles:
                if np.linalg.norm(new_pos - obs_pos) < self.robot_radius + 0.3:
                    collision = True
                    break

            if not collision:
                self.robot_pos = new_pos

        # Calculate reward
        reward = self.calculate_reward()

        # Check termination
        done = self.check_termination()

        # Increment step
        self.current_step += 1

        # Info
        info = {
            'distance_to_goal': np.linalg.norm(self.robot_pos - self.goal_pos),
            'robot_pos': self.robot_pos,
            'goal_pos': self.goal_pos,
            'collisions': self.check_collision_with_obstacles()
        }

        return self.get_observation(), reward, done, info

    def get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Robot state
        robot_state = np.concatenate([self.robot_pos, [self.robot_theta]])

        # Goal state
        goal_state = self.goal_pos

        # Obstacle relative positions (up to 10, pad with zeros if fewer)
        obstacle_states = []
        for i in range(10):
            if i < len(self.obstacles):
                # Relative position to robot
                rel_pos = self.obstacles[i] - self.robot_pos
                obstacle_states.extend(rel_pos)
            else:
                # Padding
                obstacle_states.extend([0.0, 0.0])

        obs = np.concatenate([robot_state, goal_state, obstacle_states])
        return obs

    def calculate_reward(self) -> float:
        """Calculate reward."""
        # Distance to goal
        dist_to_goal = np.linalg.norm(self.robot_pos - self.goal_pos)

        # Base reward: negative distance
        reward = -dist_to_goal * 0.1

        # Bonus for getting closer to goal
        prev_dist = getattr(self, '_prev_dist_to_goal', dist_to_goal)
        if dist_to_goal < prev_dist:
            reward += 0.5  # Small bonus for progress
        self._prev_dist_to_goal = dist_to_goal

        # Large bonus for reaching goal
        if dist_to_goal < self.goal_tolerance:
            reward += 100.0

        # Penalty for collisions
        if self.check_collision_with_obstacles():
            reward -= 10.0

        return reward

    def check_collision_with_obstacles(self) -> bool:
        """Check if robot collides with any obstacle."""
        for obs_pos in self.obstacles:
            if np.linalg.norm(self.robot_pos - obs_pos) < self.robot_radius + 0.3:
                return True
        return False

    def check_termination(self) -> bool:
        """Check if episode should terminate."""
        # Reach goal
        if np.linalg.norm(self.robot_pos - self.goal_pos) < self.goal_tolerance:
            return True

        # Max steps
        if self.current_step >= self.max_episode_steps:
            return True

        return False
```

## Reward Function Design for Robotics

Designing appropriate reward functions is critical for successful RL in robotics.

```python
# reward_design.py
import numpy as np
from typing import Dict, Any, Callable

class RewardDesigner:
    """
    Utility class for designing reward functions for robotic tasks.
    """
    @staticmethod
    def reacher_reward(
        end_effector_pos: np.ndarray,
        goal_pos: np.ndarray,
        joint_angles: np.ndarray,
        joint_velocities: np.ndarray,
        goal_tolerance: float = 0.1,
        joint_limit_penalty: float = 1.0
    ) -> float:
        """
        Reward function for reaching tasks.
        """
        # Distance to goal (negative for minimization)
        distance = np.linalg.norm(end_effector_pos - goal_pos)
        distance_reward = -distance

        # Goal achievement bonus
        goal_bonus = 0.0
        if distance < goal_tolerance:
            goal_bonus = 100.0

        # Joint limit penalty
        limit_penalty = 0.0
        for angle in joint_angles:
            # Penalize being close to joint limits (±π)
            if abs(abs(angle) - np.pi) < 0.3:
                limit_penalty -= joint_limit_penalty

        # Smoothness penalty (encourage smooth motions)
        smoothness_penalty = -0.01 * np.sum(np.abs(joint_velocities))

        total_reward = distance_reward + goal_bonus + limit_penalty + smoothness_penalty
        return total_reward

    @staticmethod
    def manipulation_reward(
        object_pos: np.ndarray,
        target_pos: np.ndarray,
        gripper_pos: np.ndarray,
        object_lifted: bool,
        grasp_success: bool,
        manipulation_progress: float
    ) -> Dict[str, float]:
        """
        Reward function for manipulation tasks.
        """
        rewards = {}

        # Primary task: move object to target
        object_to_target_dist = np.linalg.norm(object_pos - target_pos)
        rewards['object_to_target'] = -object_to_target_dist

        # Grasp reward
        if grasp_success:
            rewards['grasp'] = 20.0
        else:
            # Encourage gripper to approach object
            gripper_to_object_dist = np.linalg.norm(gripper_pos - object_pos)
            rewards['approach_object'] = -gripper_to_object_dist * 0.5

        # Lift reward (bonus for lifting object)
        if object_lifted:
            rewards['lift'] = 10.0

        # Progress reward (encourage task completion)
        rewards['progress'] = manipulation_progress * 50.0

        # Safety penalties
        rewards['safety'] = -1.0  # Small penalty for each step (encourage efficiency)

        return rewards

    @staticmethod
    def navigation_reward(
        robot_pos: np.ndarray,
        goal_pos: np.ndarray,
        prev_robot_pos: np.ndarray,
        obstacles: np.ndarray,
        collision_detected: bool,
        progress_threshold: float = 0.1
    ) -> Dict[str, float]:
        """
        Reward function for navigation tasks.
        """
        rewards = {}

        # Distance to goal
        current_dist = np.linalg.norm(robot_pos - goal_pos)
        rewards['distance'] = -current_dist * 0.1

        # Progress reward (positive reward for moving toward goal)
        if 'prev_dist' not in globals():
            prev_dist = current_dist
        else:
            prev_dist = globals()['prev_dist']

        if current_dist < prev_dist - progress_threshold:
            rewards['progress'] = 5.0
        globals()['prev_dist'] = current_dist

        # Collision penalty
        if collision_detected:
            rewards['collision'] = -50.0

        # Obstacle avoidance
        obstacle_proximity_penalty = 0.0
        for obs_pos in obstacles:
            dist_to_obs = np.linalg.norm(robot_pos - obs_pos)
            if dist_to_obs < 1.0:  # Within 1m of obstacle
                obstacle_proximity_penalty -= (1.0 - dist_to_obs) * 10.0

        rewards['obstacle_avoidance'] = obstacle_proximity_penalty

        # Success bonus
        if current_dist < 0.5:  # Close to goal
            rewards['success'] = 100.0

        return rewards

class AdaptiveRewardShaping:
    """
    Adaptive reward shaping that adjusts based on learning progress.
    """
    def __init__(self, initial_weights: Dict[str, float],
                 adaptation_rate: float = 0.01):
        self.weights = initial_weights.copy()
        self.adaptation_rate = adaptation_rate
        self.episode_rewards = []
        self.reward_components_history = []

    def update_weights(self, episode_reward: float,
                      reward_components: Dict[str, float]):
        """Update reward weights based on learning progress."""
        self.episode_rewards.append(episode_reward)
        self.reward_components_history.append(reward_components.copy())

        # If we have enough episodes, adapt weights
        if len(self.episode_rewards) > 10:
            # Calculate recent performance trends
            recent_rewards = self.episode_rewards[-10:]
            performance_improvement = np.mean(recent_rewards[-5:]) - np.mean(recent_rewards[:-5])

            # If performance is stagnating, adjust weights to emphasize harder components
            if abs(performance_improvement) < 0.1:  # No significant improvement
                # Find components with consistently low values
                for comp_name in reward_components.keys():
                    comp_values = [comp.get(comp_name, 0) for comp in self.reward_components_history[-10:]]
                    avg_comp_value = np.mean(comp_values)

                    # If component is consistently low, increase its weight
                    if avg_comp_value < 0 and abs(avg_comp_value) > 5:  # Large negative component
                        self.weights[comp_name] *= (1 + self.adaptation_rate)
                    elif avg_comp_value > 0 and avg_comp_value < 1:  # Low positive component
                        self.weights[comp_name] *= (1 + self.adaptation_rate * 0.5)

    def get_adaptive_reward(self, reward_components: Dict[str, float]) -> float:
        """Calculate weighted reward based on current weights."""
        total_reward = 0.0
        for comp_name, value in reward_components.items():
            weight = self.weights.get(comp_name, 1.0)
            total_reward += weight * value

        return total_reward

class CurriculumLearningScheduler:
    """
    Scheduler for curriculum learning in robotic RL.
    Gradually increases task difficulty.
    """
    def __init__(self,
                 initial_difficulty: float = 0.1,
                 max_difficulty: float = 1.0,
                 difficulty_growth_rate: float = 0.01,
                 success_threshold: float = 0.8):
        self.current_difficulty = initial_difficulty
        self.max_difficulty = max_difficulty
        self.difficulty_growth_rate = difficulty_growth_rate
        self.success_threshold = success_threshold

        self.episode_successes = []
        self.current_curriculum_stage = 0

    def update_curriculum(self, episode_success: bool):
        """Update curriculum based on episode success."""
        self.episode_successes.append(episode_success)

        # Calculate recent success rate
        if len(self.episode_successes) > 20:
            recent_success_rate = np.mean(self.episode_successes[-20:])

            # If success rate is high enough, increase difficulty
            if recent_success_rate >= self.success_threshold:
                self.current_difficulty = min(
                    self.current_difficulty + self.difficulty_growth_rate,
                    self.max_difficulty
                )

                # Reset success history when increasing difficulty
                self.episode_successes = []

    def get_task_parameters(self) -> Dict[str, Any]:
        """Get current task parameters based on curriculum stage."""
        params = {}

        # Adjust task difficulty based on current level
        params['goal_tolerance'] = 0.5 * (1 - self.current_difficulty) + 0.05
        params['obstacle_density'] = self.current_difficulty * 0.5
        params['max_episode_steps'] = int(1000 * (1 - self.current_difficulty) + 500)

        return params
```

## Safety and Sample Efficiency

### Safe Exploration Techniques

```python
# safety_exploration.py
import numpy as np
from typing import Tuple, Optional
import warnings

class SafeExplorationWrapper:
    """
    Wrapper that adds safety constraints to exploration.
    """
    def __init__(self,
                 action_space: 'spaces.Box',
                 safety_constraints: dict,
                 exploration_noise_scale: float = 0.1):
        self.action_space = action_space
        self.safety_constraints = safety_constraints
        self.exploration_noise_scale = exploration_noise_scale

        # Initialize safety parameters
        self.safety_buffer = safety_constraints.get('buffer', 0.1)
        self.max_action_change = safety_constraints.get('max_action_change', 0.2)
        self.prev_action = None

    def safe_action(self, action: np.ndarray, state: np.ndarray) -> np.ndarray:
        """
        Modify action to ensure safety constraints are met.
        """
        # Clip action to action space bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Limit action changes from previous action
        if self.prev_action is not None:
            action_change = action - self.prev_action
            action_change_norm = np.linalg.norm(action_change)

            if action_change_norm > self.max_action_change:
                # Scale action change to be within limits
                scaling_factor = self.max_action_change / action_change_norm
                action = self.prev_action + action_change * scaling_factor

        # Add safety checks based on state
        action = self.state_dependent_safety_check(action, state)

        self.prev_action = action.copy()
        return action

    def state_dependent_safety_check(self, action: np.ndarray, state: np.ndarray) -> np.ndarray:
        """
        Apply state-dependent safety checks.
        """
        # Example: If close to joint limits, reduce action magnitude
        if len(state) >= 4:  # Assuming state has joint angles at beginning
            joint_angles = state[:2]  # First 2 elements are joint angles

            for i, angle in enumerate(joint_angles):
                # Check if close to joint limits
                if abs(abs(angle) - np.pi) < 0.3:  # Within 0.3 rad of limit
                    # Reduce action magnitude for this joint
                    action[i] *= 0.5

        return action

class HindsightExperienceReplay:
    """
    Hindsight Experience Replay (HER) for sparse reward environments.
    """
    def __init__(self,
                 reward_func: Callable,
                 strategy: str = 'future',
                 k: int = 4):
        self.reward_func = reward_func
        self.strategy = strategy  # 'future', 'final', 'episode'
        self.k = k  # Number of additional goals to sample per transition
        self.episode_buffer = []

    def add_to_episode(self, transition: Tuple):
        """Add transition to episode buffer."""
        self.episode_buffer.append(transition)

    def sample_her_transitions(self) -> list:
        """Sample HER transitions from episode buffer."""
        her_transitions = []

        for i, (state, action, reward, next_state, done) in enumerate(self.episode_buffer):
            # Original transition
            her_transitions.append((state, action, reward, next_state, done))

            # Sample k additional goals
            for _ in range(self.k):
                if self.strategy == 'future':
                    # Sample goal from future states in episode
                    future_idx = np.random.randint(i, len(self.episode_buffer))
                    future_state = self.episode_buffer[future_idx][0]  # state
                    new_goal = future_state[-2:]  # Assuming last 2 dims are goal
                elif self.strategy == 'final':
                    # Use final state as goal
                    final_state = self.episode_buffer[-1][0]
                    new_goal = final_state[-2:]
                elif self.strategy == 'episode':
                    # Use random state from episode as goal
                    random_idx = np.random.randint(0, len(self.episode_buffer))
                    random_state = self.episode_buffer[random_idx][0]
                    new_goal = random_state[-2:]
                else:
                    raise ValueError(f"Unknown strategy: {self.strategy}")

                # Recompute reward with new goal
                new_reward = self.reward_func(next_state, new_goal)

                # Create new transition with modified reward
                her_transitions.append((state, action, new_reward, next_state, done))

        # Clear episode buffer
        self.episode_buffer = []
        return her_transitions

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer for more efficient learning.
    """
    def __init__(self, capacity: int, alpha: float = 0.6, beta_start: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization is used
        self.beta_start = beta_start  # Importance sampling weight

        # Sum tree for efficient priority sampling
        self.tree = SumTree(capacity)

        # Additional buffers
        self.data_buffer = []
        self.pos = 0
        self.full = False

    def push(self, state: np.ndarray, action: np.ndarray,
             reward: float, next_state: np.ndarray, done: bool,
             priority: Optional[float] = None):
        """Add experience to buffer."""
        if priority is None:
            priority = 1.0  # Default priority

        # Add to sum tree
        idx = self.tree.add(priority)

        # Add to data buffer
        if len(self.data_buffer) < self.capacity:
            self.data_buffer.append((state, action, reward, next_state, done))
        else:
            self.data_buffer[self.pos] = (state, action, reward, next_state, done)

        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True

    def sample(self, batch_size: int, current_beta: float) -> Tuple:
        """Sample batch with priority weights."""
        batch = []
        idxs = []
        priorities = []

        segment = self.tree.total() / batch_size
        current_beta_by_step = min(1.0, self.beta_start + current_beta * (1.0 - self.beta_start))

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = np.random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(priority)

        # Compute importance sampling weights
        sampling_probs = np.array(priorities) / self.tree.total()
        weights = np.power(len(self.data_buffer) * sampling_probs, -current_beta_by_step)
        weights /= weights.max()

        states = np.array([transition[0] for transition in batch])
        actions = np.array([transition[1] for transition in batch])
        rewards = np.array([transition[2] for transition in batch])
        next_states = np.array([transition[3] for transition in batch])
        dones = np.array([transition[4] for transition in batch])

        return (states, actions, rewards, next_states, dones,
                weights, idxs)

    def update_priorities(self, idxs: list, priorities: list):
        """Update priorities of sampled transitions."""
        for idx, priority in zip(idxs, priorities):
            self.tree.update(idx, priority)

class SumTree:
    """
    Sum tree data structure for Prioritized Experience Replay.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.length = 0

    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Retrieve sample from tree with value s."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """Return total priority."""
        return self.tree[0]

    def add(self, p: float, data: any) -> int:
        """Add priority and data."""
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write = (self.write + 1) % self.capacity
        if self.length < self.capacity:
            self.length += 1

        return idx

    def update(self, idx: int, p: float):
        """Update priority."""
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, any]:
        """Get sample with value s."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1

        return idx, self.tree[idx], self.data[data_idx]
```

## Sim-to-Real Transfer Techniques

### Domain Randomization

```python
# domain_randomization.py
import numpy as np
import torch
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class DomainParams:
    """Container for domain randomization parameters."""
    mass_range: tuple = (0.8, 1.2)  # Factor to multiply base mass
    friction_range: tuple = (0.5, 1.5)  # Factor to multiply base friction
    damping_range: tuple = (0.8, 1.2)  # Factor to multiply base damping
    color_range: tuple = (0.0, 1.0)  # Range for color randomization
    texture_options: List[str] = None  # Available textures
    lighting_range: tuple = (0.5, 2.0)  # Range for lighting intensity

class DomainRandomizer:
    """
    Domain randomization for improving sim-to-real transfer.
    """
    def __init__(self, domain_params: DomainParams):
        self.params = domain_params
        if self.params.texture_options is None:
            self.params.texture_options = ['metal', 'wood', 'plastic', 'fabric']

    def randomize_robot_dynamics(self) -> Dict[str, float]:
        """Randomize robot dynamics parameters."""
        randomized_params = {}

        # Randomize masses
        randomized_params['link_mass_multipliers'] = np.random.uniform(
            self.params.mass_range[0], self.params.mass_range[1], size=7  # Assuming 7-DOF robot
        )

        # Randomize joint friction
        randomized_params['friction_multipliers'] = np.random.uniform(
            self.params.friction_range[0], self.params.friction_range[1], size=7
        )

        # Randomize joint damping
        randomized_params['damping_multipliers'] = np.random.uniform(
            self.params.damping_range[0], self.params.damping_range[1], size=7
        )

        return randomized_params

    def randomize_sensor_noise(self) -> Dict[str, float]:
        """Randomize sensor noise characteristics."""
        sensor_noise = {}

        # IMU noise
        sensor_noise['imu_acc_noise'] = np.random.uniform(0.001, 0.01)
        sensor_noise['imu_gyro_noise'] = np.random.uniform(0.0001, 0.001)

        # Camera noise
        sensor_noise['camera_noise_factor'] = np.random.uniform(0.0, 0.1)

        # Joint position noise
        sensor_noise['joint_pos_noise'] = np.random.uniform(0.001, 0.01)

        return sensor_noise

    def randomize_visual_appearance(self) -> Dict[str, Any]:
        """Randomize visual appearance parameters."""
        visual_params = {}

        # Randomize colors
        visual_params['robot_color'] = np.random.uniform(
            self.params.color_range[0], self.params.color_range[1], size=3
        )

        # Randomize textures
        visual_params['robot_texture'] = np.random.choice(self.params.texture_options)

        # Randomize lighting
        visual_params['light_intensity'] = np.random.uniform(
            self.params.lighting_range[0], self.params.lighting_range[1]
        )

        # Randomize camera parameters
        visual_params['camera_exposure'] = np.random.uniform(0.5, 1.5)
        visual_params['camera_contrast'] = np.random.uniform(0.8, 1.2)

        return visual_params

    def apply_randomization(self, env):
        """Apply randomization to the environment."""
        # Randomize dynamics
        dyn_params = self.randomize_robot_dynamics()
        self._apply_dynamics_randomization(env, dyn_params)

        # Randomize sensors
        sensor_params = self.randomize_sensor_noise()
        self._apply_sensor_randomization(env, sensor_params)

        # Randomize visuals
        visual_params = self.randomize_visual_appearance()
        self._apply_visual_randomization(env, visual_params)

    def _apply_dynamics_randomization(self, env, params: Dict):
        """Apply dynamics randomization to environment."""
        # This would modify the robot's physical properties in simulation
        # Implementation depends on the simulation engine (Isaac Sim, PyBullet, etc.)
        pass

    def _apply_sensor_randomization(self, env, params: Dict):
        """Apply sensor randomization to environment."""
        # Add noise to sensor readings
        pass

    def _apply_visual_randomization(self, env, params: Dict):
        """Apply visual randomization to environment."""
        # Change colors, textures, lighting in simulation
        pass

class Sim2RealTransferTrainer:
    """
    Trainer for sim-to-real transfer with domain randomization.
    """
    def __init__(self,
                 agent,
                 sim_env,
                 real_env=None,
                 domain_randomizer: DomainRandomizer = None,
                 adaptation_steps: int = 1000):
        self.agent = agent
        self.sim_env = sim_env
        self.real_env = real_env
        self.domain_randomizer = domain_randomizer
        self.adaptation_steps = adaptation_steps

        # Training statistics
        self.sim_performance = []
        self.real_performance = []

    def train_in_simulation(self, episodes: int = 1000):
        """Train agent in simulation with domain randomization."""
        for episode in range(episodes):
            if self.domain_randomizer:
                # Apply domain randomization
                self.domain_randomizer.apply_randomization(self.sim_env)

            # Run episode in simulation
            state = self.sim_env.reset()
            total_reward = 0

            done = False
            while not done:
                action = self.agent.select_action(state)

                if self.domain_randomizer:
                    # Add sensor noise during simulation
                    noise_params = self.domain_randomizer.randomize_sensor_noise()
                    # Apply noise to observations if needed
                    pass

                next_state, reward, done, info = self.sim_env.step(action)

                # Store experience
                self.agent.remember(state, action, reward, next_state, done)

                # Train agent
                self.agent.train()

                state = next_state
                total_reward += reward

            self.sim_performance.append(total_reward)

            if episode % 100 == 0:
                print(f"Episode {episode}, Average Reward: {np.mean(self.sim_performance[-100:]):.2f}")

    def adapt_to_real_world(self):
        """Adapt the policy to real-world conditions."""
        if self.real_env is None:
            print("No real environment provided for adaptation")
            return

        print("Starting real-world adaptation...")

        for step in range(self.adaptation_steps):
            state = self.real_env.reset()
            total_reward = 0

            done = False
            while not done:
                # Use trained policy with potential modifications for real world
                action = self.agent.select_action(state, add_noise=True)

                next_state, reward, done, info = self.real_env.step(action)

                # Store experience for potential fine-tuning
                # (Be careful with safety in real world!)

                state = next_state
                total_reward += reward

            self.real_performance.append(total_reward)

            if step % 100 == 0:
                print(f"Adaptation step {step}, Reward: {total_reward:.2f}")

    def evaluate_transfer(self):
        """Evaluate the effectiveness of sim-to-real transfer."""
        if self.real_env is None:
            print("No real environment for evaluation")
            return

        print("Evaluating transfer performance...")

        sim_rewards = []
        real_rewards = []

        # Evaluate in simulation
        for _ in range(10):
            state = self.sim_env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.agent.select_action(state, evaluate=True)
                state, reward, done, _ = self.sim_env.step(action)
                total_reward += reward
            sim_rewards.append(total_reward)

        # Evaluate in real world
        for _ in range(10):
            state = self.real_env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.agent.select_action(state, evaluate=True)
                state, reward, done, _ = self.real_env.step(action)
                total_reward += reward
            real_rewards.append(total_reward)

        print(f"Simulation Performance: Mean={np.mean(sim_rewards):.2f}, Std={np.std(sim_rewards):.2f}")
        print(f"Real Performance: Mean={np.mean(real_rewards):.2f}, Std={np.std(real_rewards):.2f}")
        print(f"Transfer Gap: {np.mean(sim_rewards) - np.mean(real_rewards):.2f}")
```

## Code Examples with Explanations

### Complete Robot RL System

```python
# complete_robot_rl_system.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import time
import os

class CompleteRobotRLSystem:
    """
    Complete system integrating RL algorithms, environments, and real-world deployment.
    """
    def __init__(self,
                 robot_type: str = 'manipulator',
                 algorithm: str = 'sac',
                 use_domain_randomization: bool = True):

        self.robot_type = robot_type
        self.algorithm = algorithm
        self.use_domain_randomization = use_domain_randomization

        # Initialize components
        self.env = self._create_environment()
        self.agent = self._create_agent()
        self.domain_randomizer = self._create_domain_randomizer() if use_domain_randomization else None

        # Training statistics
        self.training_rewards = []
        self.episode_lengths = []
        self.loss_history = []

        # Performance tracking
        self.start_time = None
        self.total_steps = 0

        print(f"Complete Robot RL System initialized:")
        print(f"  Robot Type: {robot_type}")
        print(f"  Algorithm: {algorithm}")
        print(f"  Domain Randomization: {use_domain_randomization}")

    def _create_environment(self):
        """Create appropriate environment based on robot type."""
        if self.robot_type == 'manipulator':
            return RobotArmEnv()
        elif self.robot_type == 'mobile':
            return NavigationEnv()
        else:
            raise ValueError(f"Unknown robot type: {self.robot_type}")

    def _create_agent(self):
        """Create appropriate RL agent based on algorithm."""
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        if self.algorithm == 'sac':
            max_action = float(self.env.action_space.high[0])
            return SACAgent(state_dim, action_dim, max_action)
        elif self.algorithm == 'td3':
            max_action = float(self.env.action_space.high[0])
            return TD3Agent(state_dim, action_dim, max_action)
        elif self.algorithm == 'ddpg':
            max_action = float(self.env.action_space.high[0])
            return DDPGAgent(state_dim, action_dim, max_action)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def _create_domain_randomizer(self):
        """Create domain randomizer."""
        params = DomainParams(
            mass_range=(0.9, 1.1),
            friction_range=(0.8, 1.2),
            damping_range=(0.9, 1.1),
            color_range=(0.3, 0.7)
        )
        return DomainRandomizer(params)

    def train(self, total_steps: int = 100000, eval_freq: int = 5000):
        """Train the RL agent."""
        self.start_time = time.time()

        state = self.env.reset()
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0

        for t in range(total_steps):
            # Select action
            action = self.agent.select_action(state)

            # Apply domain randomization if enabled
            if self.domain_randomizer and t % 100 == 0:  # Randomize every 100 steps
                self.domain_randomizer.apply_randomization(self.env)

            # Perform action
            next_state, reward, done, info = self.env.step(action)
            done_bool = float(done) if episode_timesteps < self.env.max_episode_steps else 0

            # Store data in replay buffer
            self.agent.remember(state, action, reward, next_state, done_bool)

            # Train agent
            self.agent.train()

            # Update counters
            state = next_state
            episode_reward += reward
            episode_timesteps += 1
            self.total_steps += 1

            # End of episode
            if done:
                print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")

                # Store episode statistics
                self.training_rewards.append(episode_reward)
                self.episode_lengths.append(episode_timesteps)

                # Reset environment
                state, done = self.env.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # Evaluation
            if (t + 1) % eval_freq == 0:
                avg_reward = self.evaluate(eval_episodes=10)
                print(f"Evaluation: Total Steps: {t+1}, Average Reward: {avg_reward:.3f}")

        training_time = time.time() - self.start_time
        print(f"Training completed in {training_time:.2f}s")

    def evaluate(self, eval_episodes: int = 10) -> float:
        """Evaluate the trained agent."""
        avg_reward = 0.
        for _ in range(eval_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = self.agent.select_action(state, evaluate=True)
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward

            avg_reward += episode_reward

        avg_reward /= eval_episodes
        return avg_reward

    def plot_training_progress(self):
        """Plot training progress."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot rewards
        axes[0, 0].plot(self.training_rewards)
        axes[0, 0].set_title('Training Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')

        # Plot episode lengths
        axes[0, 1].plot(self.episode_lengths)
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')

        # Plot average rewards over windows
        window_size = 100
        if len(self.training_rewards) >= window_size:
            avg_rewards = [np.mean(self.training_rewards[i:i+window_size])
                          for i in range(0, len(self.training_rewards) - window_size + 1, window_size)]
            axes[1, 0].plot(range(0, len(self.training_rewards), window_size)[:len(avg_rewards)],
                           avg_rewards)
            axes[1, 0].set_title(f'Average Rewards (Window Size: {window_size})')
            axes[1, 0].set_xlabel('Steps')
            axes[1, 0].set_ylabel('Average Reward')

        # Plot performance over time
        if self.start_time:
            elapsed_time = time.time() - self.start_time
            axes[1, 1].plot(np.arange(len(self.training_rewards)),
                           [r/elapsed_time for r in np.cumsum(self.training_rewards)])
            axes[1, 1].set_title('Cumulative Reward Rate')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Reward/Time')

        plt.tight_layout()
        plt.show()

    def save_model(self, filepath: str):
        """Save the trained model."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.agent.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a trained model."""
        self.agent.load(filepath)
        print(f"Model loaded from {filepath}")

    def deploy_to_robot(self, robot_interface):
        """Deploy the trained policy to a real robot."""
        print("Deploying policy to real robot...")

        # This would involve interfacing with the actual robot hardware
        # The trained policy would be used to generate actions based on sensor readings
        # Safety checks and real-time constraints would be critical here

        state = robot_interface.get_sensor_data()

        try:
            while True:  # Real-time control loop
                # Get action from policy
                action = self.agent.select_action(state, evaluate=True)

                # Send action to robot
                robot_interface.send_action(action)

                # Get new state
                state = robot_interface.get_sensor_data()

                # Check for termination conditions
                if robot_interface.should_terminate():
                    break

        except KeyboardInterrupt:
            print("Deployment interrupted by user")
        finally:
            robot_interface.emergency_stop()

# Example usage and demonstration
def main():
    """Demonstrate the complete RL system."""
    print("Initializing Complete Robot RL System...")

    # Create system
    rl_system = CompleteRobotRLSystem(
        robot_type='manipulator',
        algorithm='sac',
        use_domain_randomization=True
    )

    # Train the system (for demonstration, using fewer steps)
    print("\nStarting training...")
    rl_system.train(total_steps=5000, eval_freq=1000)  # Reduced steps for demo

    # Evaluate
    print("\nEvaluating trained policy...")
    final_reward = rl_system.evaluate(eval_episodes=5)
    print(f"Final evaluation reward: {final_reward:.3f}")

    # Plot results
    print("\nPlotting training progress...")
    rl_system.plot_training_progress()

    # Save model
    model_path = "./models/robot_rl_model.pth"
    rl_system.save_model(model_path)

    print("\nTraining and evaluation completed!")
    print(f"Average training reward: {np.mean(rl_system.training_rewards):.3f}")
    print(f"Training episodes: {len(rl_system.training_rewards)}")

if __name__ == "__main__":
    main()
```

**Explanation**:
- Complete RL system integrating multiple algorithms (SAC, TD3, DDPG)
- Custom environments for different robot types (manipulator, mobile)
- Domain randomization for sim-to-real transfer
- Safety mechanisms including safe exploration and prioritized replay
- Performance evaluation and visualization tools
- Model saving/loading functionality
- Deployment interface for real robots
- Comprehensive training and evaluation framework

## Hands-on Exercises

### Exercise 1: Implement Custom RL Environment

**Objective**: Create a custom robotic environment for a specific task.

**Requirements**:
1. Define a robotic task (e.g., door opening, object sorting, navigation)
2. Implement the environment with proper state/action spaces
3. Design appropriate reward function
4. Test with different RL algorithms
5. Evaluate performance and adjust design

**Implementation Steps**:
1. Choose a robotic task and define success criteria
2. Design state representation (sensor readings, goal positions, etc.)
3. Define action space (joint velocities, Cartesian velocities, etc.)
4. Implement reward function considering task objectives
5. Create termination conditions
6. Test with different RL algorithms (DQN, DDPG, SAC)
7. Analyze and optimize reward function
8. Document environment design decisions

**Expected Outcome**: A functional robotic environment that can be used for RL training with well-designed reward function.

### Exercise 2: Domain Randomization Implementation

**Objective**: Implement domain randomization to improve sim-to-real transfer.

**Requirements**:
1. Create simulation environment with randomizable parameters
2. Implement domain randomization for dynamics, sensors, and visuals
3. Train policy with domain randomization
4. Evaluate robustness to domain shifts
5. Compare with baseline without randomization

**Implementation Steps**:
1. Identify parameters to randomize (masses, frictions, colors, etc.)
2. Implement randomization functions for each parameter type
3. Integrate randomization into training loop
4. Train policies with and without domain randomization
5. Test robustness to parameter changes
6. Evaluate sim-to-real transfer capability
7. Analyze the impact of different randomization strategies

**Expected Outcome**: A robust policy that can handle variations in robot dynamics and environmental conditions.

### Exercise 3: Safe RL for Physical Robots

**Objective**: Implement safe RL techniques for deployment on physical robots.

**Requirements**:
1. Implement safe exploration mechanisms
2. Add constraint checking during training
3. Create safety-aware reward functions
4. Test with hardware-in-the-loop simulation
5. Evaluate safety metrics alongside performance

**Implementation Steps**:
1. Design safety constraints for the robot (joint limits, collision avoidance, etc.)
2. Implement safe exploration wrapper
3. Add constraint checking to environment
4. Create safety-aware reward shaping
5. Test with prioritized experience replay
6. Evaluate safety metrics (constraint violations, safe exploration rate)
7. Compare performance with and without safety constraints

**Expected Outcome**: A safe RL system that maintains good performance while respecting safety constraints.

## Summary

Chapter 11 has provided comprehensive coverage of reinforcement learning for robot control, including various algorithms (DQN, DDPG, TD3, SAC), custom environment creation, reward function design, safety mechanisms, and sim-to-real transfer techniques. You've learned about sample efficiency methods, domain randomization, and practical deployment considerations. The examples demonstrate how to create complete RL systems for robotic applications. The exercises will help you apply these concepts to build robust RL systems for real robotic tasks.

## Further Reading

1. Levine, S., et al. (2018). "Learning Hand-Eye Coordination for Robotic Grasping with Deep Learning and Large-Scale Data Collection." *International Journal of Robotics Research*, 37(4-5), 421-436.
2. Rajeswaran, A., et al. (2017). "Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations." *Conference on Robot Learning*.
3. Pinto, L., & Gupta, A. (2017). "Supersizing Self-supervision: Learning to Grasp from 50K Tries and 700 Robot Hours." *IEEE International Conference on Robotics and Automation*.
4. OpenAI et al. (2019). "Solving Rubik's Cube with a Robot Hand." *arXiv preprint arXiv:1910.07113*.

## Navigation

[Previous: Chapter 10 - AI-Powered Perception and Manipulation](/docs/module3/chapter10) | [Next: Chapter 12 - Sim-to-Real Transfer Techniques](/docs/module3/chapter12)