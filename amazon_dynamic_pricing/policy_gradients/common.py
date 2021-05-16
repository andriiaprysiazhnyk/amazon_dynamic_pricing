import torch
import numpy as np
import torch.optim as optim

from amazon_dynamic_pricing.policy_gradients.agents import DiscreteAgent, ContiniousAgent


def get_discounted_rewards(rewards, gamma):
    n = rewards.shape[0]
    discounted_rewards = np.empty_like(rewards, dtype=np.float)
    powers = np.linspace(0, n - 1, n)
    discounts = np.array([gamma] * n) ** powers

    discounted_rewards[0] = np.sum(rewards * discounts)
    for i in range(1, n):
        discounted_reward = np.sum(rewards[i:] * discounts[:-i])
        discounted_rewards[i] = discounted_reward

    return discounted_rewards


def calculate_loss(weighted_log_probs, epoch_advantages):
    policy_loss = -1 * torch.mean(weighted_log_probs)
    state_value_loss = torch.mean(epoch_advantages)

    return policy_loss, state_value_loss


def create_optimizer(agent, config):
    if config["name"] == "adam":
        return optim.Adam(params=agent.parameters(), lr=config["lr"])

    raise TypeError(f"Unknown optimizer name: {config['name']}")


def create_agent(config):
    config = config.copy()
    agent_type = config["type"]

    if agent_type == "discrete":
        return DiscreteAgent(14, config["min_price"], config["max_price"], config["bins_number"],
                             config["common_hidden_layers"], config["policy_hidden_layers"],
                             config["value_hidden_layers"])
    elif agent_type == "continuous":
        return ContiniousAgent(14, config["min_price"], config["max_price"], config["common_hidden_layers"],
                               config["policy_hidden_layers"], config["value_hidden_layers"])

    raise TypeError(f"Unknown agent type: {agent_type}")
