import random

import keras.utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import custom modules
from catch_environment import CatchEnv
from models import dqn


class ReplayBuffer():

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []

    def add_sample(self, states, actions, rewards):
        episode = {"states": states, "actions": actions, "rewards": rewards, "summed_rewards": sum(rewards)}
        self.buffer.append(episode)

    def sort(self):
        # sort buffer
        self.buffer = sorted(self.buffer, key=lambda i: i["summed_rewards"], reverse=True)
        # keep the max buffer size
        self.buffer = self.buffer[:self.max_size]

    def get_random_samples(self, batch_size):
        self.sort()
        idxs = np.random.randint(0, len(self.buffer), batch_size)
        batch = [self.buffer[idx] for idx in idxs]
        return batch

    def get_n_best(self, n):
        self.sort()
        return self.buffer[:n]

    def __len__(self):
        return len(self.buffer)


class CatchAgent:
    def __init__(self):
        self.environment = CatchEnv()
        self.state_shape = (84, 84, 4)
        self.action_space = 3
        self.warm_up_episodes = 50
        self.visualize = False
        self.memory = ReplayBuffer(700)
        self.batch_size = 256
        self.last_few = 50
        self.command_size = 2
        self.desired_return = 1
        self.desired_horizon = 1
        self.horizon_scale = 0.02
        self.return_scale = 0.02
        self.rewards = []
        self.losses = []

        self.network = dqn(self.state_shape, self.action_space)
        self.warm_up_buffer()

    def warm_up_buffer(self):
        print("warming up the buffer")
        states = []
        actions = []
        rewards = []
        desired_return = self.desired_return
        desired_horizon = self.desired_horizon

        for e in range(self.warm_up_episodes):

            # terminal = False
            self.environment.reset()

            # for i in range(random.randint(1, 30)):
            state, reward, terminal = self.environment.step(1)

            while not terminal:
                # append a stack of 4 states
                states.append(state)
                command = np.asarray([desired_return * self.return_scale, desired_horizon * self.horizon_scale], dtype=np.float32)
                command = np.reshape(command, [1, len(command)])

                # take an action
                action = self.get_action(state, command)
                actions.append(action)
                # take a step
                state, reward, terminal = self.environment.step(action)
                # append the reward
                rewards.append(reward)

                desired_return -= reward
                desired_horizon -= 1
                desired_horizon = np.maximum(desired_horizon, 1)

            self.memory.add_sample(states, actions, rewards)

    def get_action(self, state, command):
        state = np.expand_dims(state, axis=0)  # Add an extra dimension for batch size
        action_probabilities = self.network.predict([state, command], verbose=0)
        action = np.random.choice(np.arange(0, self.action_space), p=action_probabilities[0])

        return action

    def get_greedy_action(self, state, command):
        state = np.expand_dims(state, axis=0)  # Add an extra dimension for batch size
        action_probabilities = self.network.predict([state, command], verbose=0)
        action = np.argmax(action_probabilities)

        return action

    def train_network(self):

        # sample a batch of random episodes
        random_episodes = self.memory.get_random_samples(self.batch_size)

        states = np.zeros((self.batch_size, self.state_shape[0], self.state_shape[1], self.state_shape[2]))
        commands = np.zeros((self.batch_size, 2))

        y = []

        for idx, episode in enumerate(random_episodes):
            T = len(episode['states'])
            t1 = np.random.randint(0, T - 1)
            t2 = np.random.randint(t1 + 1, T)

            state = np.float32(episode['states'][t1])
            desired_return = sum(episode['rewards'][t1:t2])
            desired_horizon = t2 - t1

            target = episode['actions'][t1]

            states[idx] = state[0]
            commands[idx] = np.asarray([desired_return * self.return_scale, desired_horizon * self.horizon_scale])
            y.append(target)

        _y = keras.utils.to_categorical(y, num_classes=self.action_space)

        self.network.fit([states, commands], _y, verbose=0)

    def sample_exploratory_commands(self):
        best_episodes = self.memory.get_n_best(self.last_few)
        exploratory_desired_horizon = np.mean([len(i["states"]) for i in best_episodes])

        returns = [i["summed_rewards"] for i in best_episodes]
        exploratory_desired_returns = np.random.uniform(np.mean(returns), np.mean(returns) + np.std(returns))

        return [exploratory_desired_returns, exploratory_desired_horizon]

    def generate_episode(self, e, desired_return, desired_horizon, testing):

        env = CatchEnv()
        tot_rewards = []

        states = []
        actions = []
        rewards = []

        score = 0

        env.reset()
        # for i in range(random.randint(1, 30)):
        state, _, terminal = env.step(1)

        while not terminal:
            states.append(state)

            command = np.asarray([desired_return * self.return_scale, desired_horizon * self.horizon_scale], dtype=np.float32)
            command = np.reshape(command, [1, len(command)])

            if testing:
                # always take the greedy action
                action = self.get_greedy_action(state, command)
            else:
                action = self.get_action(state, command)
                actions.append(action)

            next_state, reward, terminal = env.step(action)

            # TODO: check if this clipping makes send
            clipped_reward = np.clip(reward, -1, 1)
            rewards.append(clipped_reward)

            score += reward

            desired_return -= reward
            desired_horizon -= 1
            desired_horizon = np.maximum(desired_horizon, 1)

        self.memory.add_sample(states, actions, rewards)

        self.rewards.append(score)

        return score


def run_experiment():
    episodes = 200
    returns = []

    agent = CatchAgent()

    for episode in range(episodes):
        print("Episode {}".format(episode))

        for i in range(100):
            agent.train_network()

        print("finished training")

        for i in range(15):
            tmp_r = []
            exploratory_commands = agent.sample_exploratory_commands()
            desired_return = exploratory_commands[0]
            desired_horizon = exploratory_commands[1]
            r = agent.generate_episode(episode, desired_return, desired_horizon, False)
            tmp_r.append(r)

        print(np.mean(tmp_r))
        returns.append(np.mean(tmp_r))

        exploratory_commands = agent.sample_exploratory_commands()

    plt.plot(returns)
    pd.DataFrame(returns).to_csv("performances/returns.csv")


run_experiment()

# env = CatchEnv()
#
# env.reset()
# for i in range(10):
#     env.step(1)
