import numpy as np
from datetime import datetime

from amazon_dynamic_pricing.q_learning.tabular_learning.agents import TabularQLearningAgent
from amazon_dynamic_pricing.q_learning.dqn_learning.agents import DQNAgent
from amazon_dynamic_pricing.common.helper import create_sub_folder


def create_agent(config):
    config = config.copy()
    if config["type"] == "tabular":
        del config["type"]
        return TabularQLearningAgent(**config)
    else:
        return DQNAgent(config)


def create_logs_folders(config):
    experiment_name = f"{config['agent']['type']}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    return create_sub_folder(config["training_logs_path"], experiment_name)


def calculate_episodes_number(last_epsilon, exploration_rate, exploration_decay_rate):
    return int((np.log(last_epsilon) - np.log(exploration_rate)) / np.log(exploration_decay_rate))


if __name__ == "__main__":
    last_epsilon = 0.05
    exploration_rate = 0.99
    exploration_decay_rate = 0.99995

    print(f"Number of episodes = {calculate_episodes_number(last_epsilon, exploration_rate, exploration_decay_rate)}")
