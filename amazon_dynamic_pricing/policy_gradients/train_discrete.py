import os
import random
import yaml
import numpy as np
import torch
from torch.nn.functional import one_hot, log_softmax, softmax
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from datetime import datetime

from amazon_dynamic_pricing.policy_gradients.common import get_discounted_rewards, create_optimizer, create_agent
from amazon_dynamic_pricing.policy_gradients.common import calculate_loss as loss

from amazon_dynamic_pricing.common.environment_creator import create_environment
from amazon_dynamic_pricing.common.helper import copy_config, create_sub_folder


def play_episode(env, agent, training_config):
    state = env.reset()

    episode_actions = torch.empty(size=(0,), dtype=torch.long)
    episode_logits = torch.empty(size=(0, training_config["agent"]["bins_number"]))
    episode_rewards = np.empty(shape=(0,), dtype=np.float32)
    episode_state_values = torch.empty(size=(0,), dtype=torch.float)

    while True:
        action_logits, state_value, action, price = agent.train_act(torch.tensor(state).float().unsqueeze(dim=0))
        episode_logits = torch.cat((episode_logits, action_logits), dim=0)
        episode_state_values = torch.cat((episode_state_values, state_value.squeeze(dim=0)), dim=0)
        episode_actions = torch.cat((episode_actions, action), dim=0)

        state, reward, done = env.step(price)
        episode_rewards = np.concatenate((episode_rewards, np.array([reward])), axis=0)

        if done:
            discounted_rewards_to_go = get_discounted_rewards(rewards=episode_rewards,
                                                              gamma=training_config["agent"]["discount_factor"])

            discounted_rewards_to_go = torch.tensor(discounted_rewards_to_go).float()
            discounted_rewards_to_go = discounted_rewards_to_go - episode_state_values
            sum_of_rewards = np.sum(episode_rewards)

            mask = one_hot(episode_actions, num_classes=training_config["agent"]["bins_number"])
            episode_log_probs = torch.mean(mask.float() * log_softmax(episode_logits, dim=1), dim=1)
            episode_weighted_log_probs = episode_log_probs * discounted_rewards_to_go.detach()

            sum_weighted_log_probs = torch.mean(episode_weighted_log_probs).unsqueeze(dim=0)

            return sum_weighted_log_probs, episode_logits, sum_of_rewards, discounted_rewards_to_go


def calculate_loss(epoch_logits, weighted_log_probs, epoch_advantages):
    p = softmax(epoch_logits, dim=1)
    log_p = log_softmax(epoch_logits, dim=1)
    entropy = -1 * torch.mean(torch.sum(p * log_p, dim=1), dim=0)

    policy_loss, state_value_loss = loss(weighted_log_probs, epoch_advantages)
    return policy_loss, state_value_loss, entropy


def train(env, agent, training_config, opt, writer):
    episode = 0
    epoch = 0
    best_reward_sum = -float("inf")
    total_rewards = deque([], maxlen=training_config["agent"]["consecutive_episodes"])

    epoch_logits = torch.empty(size=(0, training_config["agent"]["bins_number"]))
    epoch_weighted_log_probs = torch.empty(size=(0,), dtype=torch.float)

    epoch_advantages = torch.empty(size=(0,), dtype=torch.float)

    for i in range(training_config["num_epochs"] * training_config["batch_size"]):
        (episode_weighted_log_prob_trajectory,
         episode_logits,
         sum_of_episode_rewards,
         advantages) = play_episode(env, agent, training_config)
        episode += 1

        total_rewards.append(sum_of_episode_rewards)

        epoch_weighted_log_probs = torch.cat((epoch_weighted_log_probs, episode_weighted_log_prob_trajectory),
                                             dim=0)

        advantages = torch.sum(advantages ** 2).unsqueeze(dim=0)
        epoch_advantages = torch.cat((epoch_advantages, advantages), dim=0)

        epoch_logits = torch.cat((epoch_logits, episode_logits), dim=0)

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

            policy_loss, state_value_loss, entropy = calculate_loss(epoch_logits=epoch_logits,
                                                                    weighted_log_probs=epoch_weighted_log_probs,
                                                                    epoch_advantages=epoch_advantages
                                                                    )

            total_loss = policy_loss + training_config["alpha"] * state_value_loss - training_config["beta"] * entropy
            opt.zero_grad()

            total_loss.backward()

            opt.step()

            print("\r", f"Epoch: {epoch}, Avg Return per Epoch: {avr_reward:.3f}\n", end="", flush=True)

            writer.add_scalar(tag="Entropy", scalar_value=entropy, global_step=epoch)
            writer.add_scalar(tag="State Value loss", scalar_value=state_value_loss, global_step=epoch)
            writer.add_scalar(tag="Policy loss", scalar_value=policy_loss, global_step=epoch)

            epoch_logits = torch.empty(size=(0, training_config["agent"]["bins_number"]))
            epoch_weighted_log_probs = torch.empty(size=(0,), dtype=torch.float)
            epoch_advantages = torch.empty(size=(0,), dtype=torch.float)

    writer.close()


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # read environment config
    with open(os.path.join(os.path.dirname(__file__), "..", "market_config.yaml")) as config_file:
        env_config = yaml.full_load(config_file)

    # read agent config
    config_name = "discrete_agent_config.yaml"
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
    experiment_config["training_logs_path"] = create_sub_folder(experiment_config["training_logs_path"], experiment_name)
    print("Create log folder")
    copy_config(experiment_config, experiment_config["training_logs_path"])
    print("Copy agent parameters\n")

    writer = SummaryWriter(experiment_config["training_logs_path"])

    print("Start Training:")
    train(env, agent, experiment_config, optimizer, writer)
