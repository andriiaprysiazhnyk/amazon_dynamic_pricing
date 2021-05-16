import os
import yaml
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from amazon_dynamic_pricing.common.environment_creator import create_environment
from amazon_dynamic_pricing.common.helper import copy_config
from amazon_dynamic_pricing.q_learning.helper import create_agent, create_logs_folders, calculate_episodes_number


def train(env, agent, summary_writer, n_episodes, print_every=500):
    for i in range(n_episodes):
        current_reward_sum = 0

        observation = env.reset()  # get initial observation
        action = agent.begin_episode(observation)
        done = False

        while not done:
            observation, reward, done = env.step(action)
            action = agent.act(observation, reward, done)
            current_reward_sum += reward  # accumulate rewards

        agent.register_reward(current_reward_sum, i)

        if i % print_every == 0:
            print(f"Episode {i + 1}: cummulative reward = {current_reward_sum}, "
                  f"exploration rate = {agent.exploration_rate}")

        # write to TensorBoard
        summary_writer.add_scalar("Cummulative reward", current_reward_sum, i + 1)
        summary_writer.add_scalar("Exploration rate", agent.exploration_rate, i + 1)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # read environment config
    with open(os.path.join(os.path.dirname(__file__), "..", "market_config.yaml")) as config_file:
        env_config = yaml.full_load(config_file)

    # read agent config
    config_name = "tabular_learning_config.yaml"
    # config_name = "dqn_config.yaml"
    with open(os.path.join(os.path.dirname(__file__), config_name)) as config_file:
        experiment_config = yaml.full_load(config_file)
        experiment_config["agent"]["min_price"] = env_config["min_price"]
        experiment_config["agent"]["max_price"] = env_config["max_price"]

    env = create_environment(env_config, experiment_config["agent"]["type"] == "dqn")
    print("Create environment")
    agent = create_agent(experiment_config["agent"])
    print("Create agent\n")

    training_logs_path = create_logs_folders(experiment_config)
    print("Create log folder")
    copy_config(experiment_config["agent"], training_logs_path)
    print("Copy agent parameters\n")

    train_episodes_number = calculate_episodes_number(experiment_config["last_epsilon"], agent.exploration_rate,
                                                      agent.exploration_decay_rate)
    print(f"Number of training episodes = {train_episodes_number}")

    train_writer = SummaryWriter(training_logs_path)

    print("Start Training:")
    train(env, agent, train_writer, train_episodes_number)
    exploiting_agent = agent.get_exploiting_agent()
    exploiting_agent.save(training_logs_path)
