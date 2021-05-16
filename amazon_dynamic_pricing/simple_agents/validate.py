import os
import yaml
import random
import pickle
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from amazon_dynamic_pricing.common.environment_creator import create_environment
from amazon_dynamic_pricing.common.helper import create_sub_folder
from amazon_dynamic_pricing.common.helper import copy_config
from amazon_dynamic_pricing.common.agent_validation import validate
from amazon_dynamic_pricing.simple_agents.agents import RandomAgent, GreedyAgent, ConstantAgent
from amazon_dynamic_pricing.q_learning.dqn_learning.agents import ExploitingDQNAgent
from amazon_dynamic_pricing.q_learning.tabular_learning.agents import ExploitingTabularAgent
from amazon_dynamic_pricing.policy_gradients.agents import load_agent


def create_agent(config):
    config_copy = config.copy()
    del config_copy["type"]
    if config["type"] == "greedy":
        demand_model = pickle.load(open(config["demand_model_path"], "rb"))
        price_rank_coefficients = np.load(config["price_rank_curve_coefficients_path"])
        return GreedyAgent(config["min_price"], config["max_price"], demand_model, price_rank_coefficients)
    elif config["type"] == "random":
        return RandomAgent(config["min_price"], config["max_price"])
    elif config["type"] == "min":
        return ConstantAgent(config["min_price"])
    elif config["type"] == "max":
        return ConstantAgent(config["max_price"])
    elif config["type"] == "dqn":
        return ExploitingDQNAgent.load(config["path"])
    elif config["type"] == "tabular":
        return ExploitingTabularAgent.load(config["path"])
    elif config["type"] == "policy_gradient":
        return load_agent(config["path"])

    raise TypeError(f"Unknown type of agent {config['type']}")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # read environment config
    with open(os.path.join(os.path.dirname(__file__), "..", "market_config.yaml")) as config_file:
        env_config = yaml.full_load(config_file)

    # read agent config
    with open(os.path.join(os.path.dirname(__file__), "config.yaml")) as config_file:
        experiment_config = yaml.full_load(config_file)
        experiment_config["agent"]["min_price"] = env_config["min_price"]
        experiment_config["agent"]["max_price"] = env_config["max_price"]

    env = create_environment(env_config, experiment_config["agent"]["type"] not in ["greedy", "tabular"])
    print("Create environment")
    agent = create_agent(experiment_config["agent"])
    print("Create agent\n")

    experiment_name = f"{experiment_config['agent']['type']}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    logs_path = create_sub_folder(experiment_config["evaluation_logs_path"], experiment_name)
    print("Create log folder")
    copy_config(experiment_config["agent"], logs_path)
    print("Copy agent parameters\n")

    writer = SummaryWriter(logs_path)

    print("Start Validation:")
    validate(env, agent, writer, experiment_config["validation_episodes"])
