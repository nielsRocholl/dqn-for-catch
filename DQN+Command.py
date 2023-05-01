import random

import pandas as pd
import numpy as np

from collections import deque

from keras.layers import Dense, Conv2D, Flatten, Input, Multiply
from keras.optimizers import Adam, RMSprop
from keras.models import Sequential, Model
from matplotlib import pyplot as plt

from catch_environment import CatchEnv


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 32
        self.train_start = 1000
        self.memory = deque(maxlen=2000)
        self.model = self.build_model()
        self.target_model = self.build_model()

        self.update_target_model()

    def build_model(self):
        input_state = Input(shape=self.state_size)

        first_conv = Conv2D(
            32, (8, 8), strides=(4, 4), activation='relu')(input_state)
        second_conv = Conv2D(
            64, (4, 4), strides=(2, 2), activation='relu')(first_conv)
        third_conv = Conv2D(
            64, (3, 3), strides=(1, 1), activation='relu')(second_conv)

        flattened = Flatten()(third_conv)
        dense_layer = Dense(512, activation='relu')(flattened)

        command_input = Input(shape=(2,))
        sigmoidal_layer = Dense(512, activation='sigmoid')(command_input)

        multiplied_layer = Multiply()([dense_layer, sigmoidal_layer])
        final_layer = Dense(256, activation='relu')(multiplied_layer)

        action_layer = Dense(self.action_size, activation='softmax')(final_layer)

        model = Model(inputs=[input_state, command_input], outputs=action_layer)
        model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.001, rho=0.95, epsilon=0.01))

        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state, command):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            action_probabilities = self.model.predict([state, command], verbose=0)
            return np.argmax(action_probabilities[0])

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train_model(self, command):
        if len(self.memory) < self.train_start:
            return

        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, *self.state_size))
        update_target = np.zeros((batch_size, *self.state_size))

        action, reward, done = [], [], []

        for i in range(batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        # Create command input with the same batch size as update_input
        command_input = np.tile(command, (batch_size, 1))

        target = self.model.predict([update_input, command_input], verbose=0)

        target_val = self.target_model.predict([update_target, command_input], verbose=0)

        for i in range(self.batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * (
                    np.amax(target_val[i]))

        self.model.fit([update_input, command_input], target, batch_size=self.batch_size,
                       epochs=1, verbose=0)


def run_DQN():
    episodes = 2000
    results = []

    env = CatchEnv()

    state_size = (84, 84, 4)
    action_size = 3

    agent = DQNAgent(state_size, action_size)
    queue = deque(maxlen=40)

    for e in range(episodes):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1] + list(state_size))

        while not done:
            command = np.array([[1.0, 0.0]])  # move this line inside the loop
            action = agent.get_action(state, command)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1] + list(state_size))

            agent.append_sample(state, action, reward, next_state, done)
            agent.train_model(command)

            score += reward
            state = next_state

            if done:
                queue.append(score)
                agent.update_target_model()

        if e > 89:
            results.append(score)
        # round to 2 decimal places
        print(f'Episode {e}, Score: {np.mean(queue):.2f}, Epsilon: {agent.epsilon:.2f}')

    # utils.save_trained_model(game, seed, 'DQN', agent.model)
    pd.DataFrame(results).to_csv('results.csv')



import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_running_average(results, window_size=50):
    cumulative_sum = np.cumsum(results)
    running_average = (cumulative_sum[window_size - 1:] - np.concatenate(
        ([0], cumulative_sum[:-window_size]))) / window_size
    print("running average:", running_average)

    sns.set(style="darkgrid")
    plt.plot(running_average)
    plt.xlabel('Episode')
    plt.ylabel('Running Average (Window Size = {})'.format(window_size))
    plt.title('Running Average of Scores')
    plt.show()


# results = pd.read_csv('results.csv', header=None)
# plot_running_average(results[1])


run_DQN()
