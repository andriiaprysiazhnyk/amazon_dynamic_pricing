import torch
import torch.nn as nn
from torch.distributions import Categorical, Beta
from collections import OrderedDict


def create_mlp(layers_size, last_linear=False):
    layers = [("layer{}".format(i), nn.Sequential(nn.Linear(layers_size[i - 1], layers_size[i]), nn.ReLU())) for i in
              range(1, len(layers_size))]
    if last_linear:
        last_index = len(layers_size) - 1
        layers[-1] = (f"layer{last_index}", nn.Linear(layers_size[last_index - 1], layers_size[last_index]))

    return nn.Sequential(OrderedDict(layers))


def load_agent(path):
    state_dict = torch.load(path)

    agent_type = state_dict["type"]
    weights = state_dict["weights"]

    del state_dict["type"]
    del state_dict["weights"]

    if agent_type == "discrete":
        agent = DiscreteAgent(**state_dict)
    elif agent_type == "continuous":
        agent = ContiniousAgent(**state_dict)
    else:
        raise TypeError(f"Unknown agent type: {agent_type}")

    agent.load_state_dict(weights)
    return agent


class DiscreteAgent(nn.Module):
    def __init__(self, observation_space_size, min_price, max_price, bins_number, common_hidden_layers,
                 policy_hidden_layers, value_hidden_layers):
        super(DiscreteAgent, self).__init__()

        self.observation_space_size = observation_space_size
        self.min_price = min_price
        self.max_price = max_price
        self.bins_number = bins_number
        self.division = (self.max_price - self.min_price) / self.bins_number
        self.common_hidden_layers = common_hidden_layers
        self.policy_hidden_layers = policy_hidden_layers
        self.value_hidden_layers = value_hidden_layers

        is_common_net = len(common_hidden_layers) > 0
        self.net = create_mlp([observation_space_size] + common_hidden_layers) if is_common_net else None

        input_size = common_hidden_layers[-1] if is_common_net else observation_space_size
        self.policy_head = create_mlp([input_size] + policy_hidden_layers + [bins_number], last_linear=True)
        self.state_value_head = create_mlp([input_size] + value_hidden_layers + [1], last_linear=True)

    def _get_price(self, action):
        return (action + 1) * self.division + self.min_price

    def forward(self, x):
        if self.net:
            x = self.net(x)

        logits, state_value = self.policy_head(x), self.state_value_head(x)
        return logits, state_value

    def train_act(self, x):
        action_logits, state_value = self(x)
        action = Categorical(logits=action_logits).sample()
        return action_logits, state_value, action, self._get_price(action.item())

    def act(self, observation):
        x = torch.tensor(observation).float().unsqueeze(dim=0)
        with torch.no_grad():
            action_logits, _ = self(x)
        action = Categorical(logits=action_logits).sample()
        return self._get_price(action.item())

    def get_state_dict(self):
        return {
            "type": "discrete",
            "observation_space_size": self.observation_space_size,
            "min_price": self.min_price,
            "max_price": self.max_price,
            "bins_number": self.bins_number,
            "common_hidden_layers": self.common_hidden_layers,
            "policy_hidden_layers": self.policy_hidden_layers,
            "value_hidden_layers": self.value_hidden_layers,
            "weights": self.state_dict()
        }


class ContiniousAgent(nn.Module):
    def __init__(self, observation_space_size, min_price, max_price, common_hidden_layers, policy_hidden_layers,
                 value_hidden_layers):
        super(ContiniousAgent, self).__init__()

        self.observation_space_size = observation_space_size
        self.min_price = min_price
        self.max_price = max_price
        self.diff = self.max_price - self.min_price
        self.common_hidden_layers = common_hidden_layers
        self.policy_hidden_layers = policy_hidden_layers
        self.value_hidden_layers = value_hidden_layers

        is_common_net = len(common_hidden_layers) > 0
        self.net = create_mlp([observation_space_size] + common_hidden_layers) if is_common_net else None

        input_size = common_hidden_layers[-1] if is_common_net else observation_space_size
        self.policy_head = create_mlp([input_size] + policy_hidden_layers + [2], last_linear=True)
        self.state_value_head = create_mlp([input_size] + value_hidden_layers + [1], last_linear=True)

    def _get_price(self, action):
        return action * self.diff + self.min_price

    def forward(self, x):
        epsilon = 1

        if self.net:
            x = self.net(x)

        parameters, state_value = self.policy_head(x), self.state_value_head(x)
        parameters = torch.log(1 + torch.exp(parameters)) + epsilon
        return parameters, state_value

    def train_act(self, x):
        parameters, state_value = self(x)
        dist = Beta(parameters[:, 0], parameters[:, 1])
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return log_prob, state_value, self._get_price(action.item()), parameters[:, 0].detach(), \
               parameters[:, 1].detach(), dist.entropy()

    def act(self, observation):
        x = torch.tensor(observation).float().unsqueeze(dim=0)
        with torch.no_grad():
            parameters, _ = self(x)
        action = Beta(parameters[:, 0], parameters[:, 1]).sample()
        return self._get_price(action.item())

    def get_state_dict(self):
        return {
            "type": "continuous",
            "observation_space_size": self.observation_space_size,
            "min_price": self.min_price,
            "max_price": self.max_price,
            "common_hidden_layers": self.common_hidden_layers,
            "policy_hidden_layers": self.policy_hidden_layers,
            "value_hidden_layers": self.value_hidden_layers,
            "weights": self.state_dict()
        }
