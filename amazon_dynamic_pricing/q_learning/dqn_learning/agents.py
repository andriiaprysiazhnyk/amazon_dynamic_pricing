import os
import torch
import numpy as np
from copy import deepcopy
from collections import deque

from amazon_dynamic_pricing.q_learning.dqn_learning.approximators import get_network, QNetwork
from amazon_dynamic_pricing.q_learning.dqn_learning.experience_buffer import ExperienceBuffer


class DQNAgent:
    def __init__(self, config):
        self.config = config
        self.min_price = config["min_price"]
        self.max_price = config["max_price"]
        self.bins_number = config["bins_number"]
        self.division = (self.max_price - self.min_price) / self.bins_number

        self.active_network = get_network(14, self.bins_number, config["layers"])
        self._set_target_network()
        self.experience_buffer = ExperienceBuffer(14, **config["replay_buffer"])
        self.criterion = self._get_loss()
        self.optimizer = self._get_optimizer()

        self.discount_factor = config["discount_factor"]
        self.copy_timestamps = config["copy_timestamps"]
        self.learning_starts = config["learning_starts"]

        self.deque = deque(maxlen=config["consecutive_episodes"])
        self.best_reward = -float("inf")
        self.episode_number = -1
        self.best_network = None

        self.exploration_rate = config["exploration"]["exploration_rate"]
        self.exploration_decay_rate = config["exploration"]["exploration_decay_rate"]

        self.state = None
        self.action = None
        self.counter = 0

    def _get_loss(self):
        name = self.config["loss"]["name"]
        if name == "mse":
            return torch.nn.MSELoss()
        else:
            raise TypeError("Unknown loss name {}".format(name))

    def _get_optimizer(self):
        name = self.config["optimizer"]["name"]
        if name == "adam":
            return torch.optim.Adam(self.active_network.parameters(), lr=self.config["optimizer"]["lr"])
        else:
            raise TypeError("Unknown optimizer name {}".format(name))

    def _get_price(self, action):
        return (action + 1) * self.division + self.min_price

    def _set_target_network(self):
        self.target_network = deepcopy(self.active_network)
        self.target_network.eval()

    def _get_target(self, reward, state, done):
        y = reward

        if not done:
            optimal_action = self.active_network.get_optimal_action(state)
            max_q = self.target_network.get_action_value(state, optimal_action)
            y += self.discount_factor * max_q

        return y

    def begin_episode(self, observation):
        self.state = torch.FloatTensor(observation)
        self.exploration_rate *= self.exploration_decay_rate

        enable_exploration = (1 - self.exploration_rate) <= np.random.uniform(0, 1)
        self.action = np.random.randint(0, self.bins_number) if enable_exploration else\
            self.target_network.get_optimal_action(self.state)

        return self._get_price(self.action)

    def act(self, observation, reward, done):
        next_state = torch.FloatTensor(observation)

        enable_exploration = (1 - self.exploration_rate) <= np.random.uniform(0, 1)
        next_action = np.random.randint(0, self.bins_number) if enable_exploration else\
            self.target_network.get_optimal_action(next_state)

        y = self._get_target(reward, next_state, done)
        self.experience_buffer.add_sample(self.state, self.action, y)

        if self.learning_starts > 0:
            self.learning_starts -= 1
        else:
            x, actions, y = self.experience_buffer.sample()
            y_hat = self.active_network(x)
            y_hat = torch.gather(y_hat, 1, actions).squeeze()
            loss = self.criterion(y_hat, y)
            self.active_network.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.counter += 1
            if self.counter % self.copy_timestamps == 0:
                self._set_target_network()

        self.state = next_state
        self.action = next_action
        return self._get_price(self.action)

    def register_reward(self, reward, episode):
        self.deque.append(reward)

        if len(self.deque) == self.deque.maxlen:
            avg_reward = sum(self.deque) / len(self.deque)

            if avg_reward > self.best_reward:
                self.best_reward = avg_reward
                self.episode_number = episode
                self.best_network = deepcopy(self.target_network)

    def get_exploiting_agent(self):
        print(f"Best network - avr_reward = {self.best_reward}; episode = {self.episode_number}")
        return ExploitingDQNAgent(self.min_price, self.max_price, self.bins_number, self.best_network)


class ExploitingDQNAgent:
    def __init__(self, min_price, max_price, bins_number, q_network):
        self.min_price = min_price
        self.max_price = max_price
        self.bins_number = bins_number
        self.division = (max_price - min_price) / self.bins_number
        self.q_network = q_network

    def _get_price(self, action):
        return (action + 1) * self.division + self.min_price

    @staticmethod
    def load(path):
        state_dict = torch.load(path)
        q_network = QNetwork(state_dict["fc_layers"])
        q_network.load_state_dict(state_dict["weights"])
        del state_dict["fc_layers"]
        del state_dict["weights"]
        return ExploitingDQNAgent(q_network=q_network, **state_dict)

    def act(self, observation):
        state = torch.FloatTensor(observation)
        action = self.q_network.get_optimal_action(state)
        return self._get_price(action)

    def save(self, path):
        state_dict = self.q_network.get_state_dict()
        state_dict["min_price"] = self.min_price
        state_dict["max_price"] = self.max_price
        state_dict["bins_number"] = self.bins_number
        torch.save(state_dict, os.path.join(path, "agent.pth"))
