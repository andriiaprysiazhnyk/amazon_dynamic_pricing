import os
import random
import yaml
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from datetime import datetime

from amazon_dynamic_pricing.policy_gradients.common import get_discounted_rewards, calculate_loss, create_optimizer, \
    create_agent

from amazon_dynamic_pricing.common.environment_creator import create_environment
from amazon_dynamic_pricing.common.helper import copy_config, create_sub_folder


def play_episode(env, agent, training_config):
    state = env.reset()

    episode_log_probs = torch.empty(size=(0,), dtype=torch.float)
    episode_rewards = np.empty(shape=(0,), dtype=np.float)
    episode_state_values = torch.empty(size=(0,), dtype=torch.float)
    a_parameters = torch.empty(size=(0,), dtype=torch.float)
    b_parameters = torch.empty(size=(0,), dtype=torch.float)
    entropy_arr = torch.empty(size=(0,), dtype=torch.float)

    while True:
        log_prob, state_value, action, a, b, entropy = agent.train_act(torch.tensor(state).float().unsqueeze(dim=0))

        episode_log_probs = torch.cat((episode_log_probs, log_prob), dim=0)
        episode_state_values = torch.cat((episode_state_values, state_value.squeeze(dim=0)), dim=0)
        a_parameters = torch.cat((a_parameters, a))
        b_parameters = torch.cat((b_parameters, b))
        entropy_arr = torch.cat((entropy_arr, entropy))

        state, reward, done = env.step(action)
        episode_rewards = np.concatenate((episode_rewards, np.array([reward])), axis=0)

        if done:
            discounted_rewards_to_go = get_discounted_rewards(rewards=episode_rewards,
                                                              gamma=training_config["agent"]["discount_factor"])
            discounted_rewards_to_go = torch.tensor(discounted_rewards_to_go).float()
            discounted_rewards_to_go = discounted_rewards_to_go - episode_state_values

            sum_of_rewards = np.sum(episode_rewards)

            episode_weighted_log_probs = episode_log_probs * \
                                         discounted_rewards_to_go.detach()

            sum_weighted_log_probs = torch.mean(episode_weighted_log_probs).unsqueeze(dim=0)

            return sum_weighted_log_probs, sum_of_rewards, discounted_rewards_to_go, a_parameters.mean().item(), \
                   b_parameters.mean().item(), entropy_arr.mean().unsqueeze(dim=0)


def train(env, agent, training_config, opt, writer):
    episode = 0
    epoch = 0
    best_reward_sum = -float("inf")
    total_rewards = deque([], maxlen=100)

    epoch_weighted_log_probs = torch.empty(size=(0,), dtype=torch.float)
    epoch_advantages = torch.empty(size=(0,), dtype=torch.float)
    entropy_arr = torch.empty(size=(0,), dtype=torch.float)

    a, b = [], []

    for i in range(training_config["num_epochs"] * training_config["batch_size"]):
        (episode_weighted_log_prob_trajectory,
         sum_of_episode_rewards,
         advantages,
         mean_a,
         mean_b,
         entropy) = play_episode(env, agent, training_config)
        episode += 1

        total_rewards.append(sum_of_episode_rewards)
        a.append(mean_a)
        b.append(mean_b)

        epoch_weighted_log_probs = torch.cat((epoch_weighted_log_probs, episode_weighted_log_prob_trajectory),
                                             dim=0)
        entropy_arr = torch.cat((entropy_arr, entropy))
        advantages = torch.mean(advantages ** 2).unsqueeze(dim=0)
        epoch_advantages = torch.cat((epoch_advantages, advantages), dim=0)

        writer.add_scalar(tag='Cummulative reward',
                          scalar_value=sum_of_episode_rewards,
                          global_step=i)

        if episode >= training_config["batch_size"]:
            episode = 0
            epoch += 1

            avr_reward = np.mean(total_rewards)
            if avr_reward > best_reward_sum:
                best_reward_sum = avr_reward
                torch.save(agent.get_state_dict(), os.path.join(training_config["training_logs_path"], "agent.pth"))

            policy_loss, state_value_loss = calculate_loss(weighted_log_probs=epoch_weighted_log_probs,
                                                           epoch_advantages=epoch_advantages
                                                           )
            mean_entropy = entropy_arr.mean()
            total_loss = policy_loss + training_config["alpha"] * state_value_loss - training_config[
                "beta"] * mean_entropy
            opt.zero_grad()

            total_loss.backward()

            opt.step()

            print("\r", f"Epoch: {epoch}, Avg Return per Epoch: {np.mean(total_rewards):.3f}\n", end="", flush=True)

            writer.add_scalar(tag="State Value loss", scalar_value=state_value_loss, global_step=epoch)
            writer.add_scalar(tag="Policy loss", scalar_value=policy_loss, global_step=epoch)
            writer.add_scalar(tag="Mean 'a' parameter", scalar_value=sum(a) / len(a), global_step=epoch)
            writer.add_scalar(tag="Mean 'b' parameter", scalar_value=sum(b) / len(b), global_step=epoch)
            writer.add_scalar(tag="Entropy", scalar_value=mean_entropy, global_step=epoch)

            epoch_weighted_log_probs = torch.empty(size=(0,), dtype=torch.float)
            epoch_advantages = torch.empty(size=(0,), dtype=torch.float)
            entropy_arr = torch.empty(size=(0,), dtype=torch.float)
            a, b = [], []

    writer.close()


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # read environment config
    with open(os.path.join(os.path.dirname(__file__), "..", "market_config.yaml")) as config_file:
        env_config = yaml.full_load(config_file)

    # read agent config
    config_name = "continuous_agent_config.yaml"
    with open(os.path.join(os.path.dirname(__file__), config_name)) as config_file:
        experiment_config = yaml.full_load(config_file)
        experiment_config["agent"]["min_price"] = env_config["min_price"]
        experiment_config["agent"]["max_price"] = env_config["max_price"]

    env = create_environment(env_config)
    print("Create environment")
    agent = create_agent(experiment_config["agent"])
    print("Create agent")
    optimizer = create_optimizer(agent, experiment_config["optimizer"])
    print("Create optimizer\n")

    experiment_name = f"{experiment_config['agent']['type']}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    experiment_config["training_logs_path"] = create_sub_folder(experiment_config["training_logs_path"],
                                                                experiment_name)
    print("Create log folder")
    copy_config(experiment_config, experiment_config["training_logs_path"])
    print("Copy agent parameters\n")

    writer = SummaryWriter(experiment_config["training_logs_path"])

    print("Start Training:")
    train(env, agent, experiment_config, optimizer, writer)
