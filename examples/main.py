from gymnasium_gomokurs.domain.gymnasium_gomokurs.service import GomokursEnv
from gymnasium_gomokurs.adapters.manager_interfaces.tcp.manager_interface import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import logging

logging.getLogger("gymnasium-gomokurs").setLevel(logging.DEBUG)

# ==========================
# 1. The Deep Q-Network (DQN)
# ==========================

class DQNNet(nn.Module):
    def __init__(self, board_size, num_channels=3, num_actions=None):
        """
        A convolutional neural network that takes a state (stack of three channels)
        and outputs Q-values for each possible action.
        """
        super(DQNNet, self).__init__()
        if num_actions is None:
            num_actions = board_size * board_size

        self.board_size = board_size

        # A relatively deep conv network for the board.
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # After convolution, flatten and use two fully connected layers.
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * board_size * board_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        # x should have shape (batch_size, 3, board_size, board_size)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc_layers(x)
        return x

# ==========================
# 2. Replay Buffer
# ==========================

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_availables):
        """
        Store a transition in the buffer.
        - state, next_state: torch.Tensor of shape (3, board_size, board_size)
        - action: int
        - reward: float
        - done: bool
        - next_availables: numpy array indicating available moves in the next state.
        """
        self.buffer.append((state, action, reward, next_state, done, next_availables))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, next_availables = zip(*batch)
        return state, action, reward, next_state, done, next_availables

    def __len__(self):
        return len(self.buffer)

# ==========================
# 3. Preprocessing Function
# ==========================

def preprocess_observation(obs):
    """
    Convert an observation (a dictionary with keys "availables", "player", "opponent")
    into a torch tensor of shape (3, board_size, board_size).
    """
    board = np.stack([obs["availables"], obs["player"], obs["opponent"]], axis=0)
    return torch.tensor(board, dtype=torch.float32)

# ==========================
# 4. The DQN Agent
# ==========================

class DQNAgent:
    def __init__(self, board_size, lr=1e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_final=0.1, epsilon_decay=5000,
                 buffer_capacity=10000, batch_size=32, target_update=1000):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        self.board_size = board_size
        self.num_actions = board_size * board_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update

        # Epsilon for epsilon-greedy policy
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        # Policy network and target network
        self.policy_net = DQNNet(board_size).to(self.device)
        self.target_net = DQNNet(board_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def select_action(self, state, availables):
        """
        Given the current state and available moves (as a 2D numpy array), select an action.
        state: torch tensor of shape (1, 3, board_size, board_size)
        availables: numpy array (of shape (board_size, board_size) or flattened) with 1 for valid moves.
        """
        sample = random.random()
        epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                  np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        # Ensure availables is a flat 1D array.
        availables = np.array(availables).flatten()
        valid_indices = np.where(availables == 1)[0]

        if sample < epsilon:
            # Randomly choose among valid moves.
            action = int(np.random.choice(valid_indices))
            return action
        else:
            with torch.no_grad():
                q_values = self.policy_net(state.to(self.device))  # shape: (1, num_actions)
                q_values = q_values.cpu().data.numpy().flatten()
            # Mask out invalid actions by setting their Q-values to -infinity.
            q_values_masked = np.full_like(q_values, -np.inf)
            q_values_masked[valid_indices] = q_values[valid_indices]
            action = int(np.argmax(q_values_masked))
            return action

    def optimize_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return None  # Not enough samples yet.

        states, actions, rewards, next_states, dones, next_availables = self.replay_buffer.sample(self.batch_size)

        # Create tensors for the batch.
        state_batch = torch.stack(states).to(self.device)  # shape: (batch, 3, board_size, board_size)
        action_batch = torch.tensor(actions, dtype=torch.long).to(self.device)  # shape: (batch,)
        reward_batch = torch.tensor(rewards, dtype=torch.float32).to(self.device)  # shape: (batch,)
        next_state_batch = torch.stack(next_states).to(self.device)
        done_batch = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Compute Q(s,a) from the policy network.
        q_values = self.policy_net(state_batch)  # shape: (batch, num_actions)
        state_action_values = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # Compute the next state maximum Q-values from the target network.
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch)  # shape: (batch, num_actions)
            # We will apply the available-action mask per sample.
            next_q_values = next_q_values.detach()
            next_q_values_cpu = next_q_values.cpu().numpy()

            max_next_q_values = []
            for i in range(self.batch_size):
                avail = np.array(next_availables[i]).flatten()
                valid_indices = np.where(avail == 1)[0]
                if len(valid_indices) > 0:
                    max_q = np.max(next_q_values_cpu[i, valid_indices])
                else:
                    max_q = 0.0
                max_next_q_values.append(max_q)
            max_next_q_values = torch.tensor(max_next_q_values, dtype=torch.float32).to(self.device)

        # Compute the expected Q values using the Bellman equation.
        expected_state_action_values = reward_batch + (1 - done_batch) * self.gamma * max_next_q_values

        # Compute loss (mean squared error).
        loss = nn.MSELoss()(state_action_values, expected_state_action_values)

        # Optimize the model.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

# ==========================
# 5. Training Loop
# ==========================

def train(num_episodes=1000):
    # Create the TCP interface and environment.
    tcp_interface_manager = create_tcp_manager_interface_from_active_connection()
    env = GomokursEnv(tcp_interface_manager)

    board_size = 20  # Assuming the environment exposes the board size.
    agent = DQNAgent(board_size)

    total_steps = 0

    for episode in range(num_episodes):
        observation, info = env.reset()
        # Preprocess the observation into a tensor.
        state = preprocess_observation(observation)
        done = False
        episode_reward = 0

        while not done:
            # Get the available moves (a 2D array, as in your observation).
            availables = observation["availables"]

            # Add batch dimension for the network (shape: (1, 3, board_size, board_size)).
            state_tensor = state.unsqueeze(0)
            action = agent.select_action(state_tensor, availables)

            # Step the environment.
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated

            if truncated:
                env.close()
                return
            
            episode_reward += reward

            # Preprocess the next observation.
            next_state = preprocess_observation(next_observation)

            # Save transition in the replay buffer.
            agent.replay_buffer.push(state, action, reward, next_state,
                                       done, next_observation["availables"])

            # Move to the next state.
            state = next_state
            observation = next_observation

            # Optimize the model.
            loss = agent.optimize_model()
            total_steps += 1

            # Update the target network periodically.
            if total_steps % agent.target_update == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())

        print(f"Episode {episode} Reward: {episode_reward}")

if __name__ == "__main__":
    train(num_episodes=10)