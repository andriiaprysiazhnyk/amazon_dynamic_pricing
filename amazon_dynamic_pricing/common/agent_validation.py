def validate(env, agent, summary_writer, n_episodes, print_every=20):
    for i in range(n_episodes):
        current_reward_sum = 0

        observation = env.reset()  # get initial observation
        action = agent.act(observation)

        while True:
            observation, reward, done = env.step(action)

            if done:
                break

            action = agent.act(observation)
            current_reward_sum += reward  # accumulate reward

        if i % print_every == 0:
            print(f"Episode {i + 1}: cumulative reward = {current_reward_sum}")

        summary_writer.add_scalar("Cumulative reward", current_reward_sum, i + 1)  # write to TensorBoard
