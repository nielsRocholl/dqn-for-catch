import os
import time
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from collections import deque
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Input, Concatenate
from keras.optimizers import RMSprop

from catch_environment import CatchEnv
from prioritized_replay_buffer import PrioritizedReplayBuffer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DQNAgent:
    def __init__(self, prioritized_memory=True):
        # define environment
        self.env = CatchEnv()

        # define hyperparameters
        self.state_shape = (84, 84, 4)
        self.relative_xy_shape = (2,)
        self.action_space = 3
        self.warm_up_episodes = 50
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 32
        self.warm_up_episodes = self.batch_size * 2
        self.memory_size = 2000
        self.prioritized_memory = prioritized_memory
        self.memory = PrioritizedReplayBuffer(self.memory_size) if prioritized_memory else deque(maxlen=2000)

        # define models
        self.model = self.build_model()
        self.target_model = self.build_model()

        # define performance datastructure
        self.performance = {
            "score": [],
            "loss": []
        }
        self.running_average = deque(maxlen=40)

        # warm up buffer
        self.warm_up_memory_buffer()

    def save_data(self):
        df = pd.DataFrame(self.performance['score'], columns=['score'])
        df.to_csv("performance_{}.csv".format(time.time()))
        self.model.save("model_{}.h5".format(time.time()))

    def plot_running_average(self, window_size=50):
        cumulative_sum = np.cumsum(self.performance['score'])
        running_average = (cumulative_sum[window_size - 1:] - np.concatenate(
            ([0], cumulative_sum[:-window_size]))) / window_size
        sns.set(style="darkgrid")
        plt.plot(running_average)
        plt.xlabel('Episode')
        plt.ylabel('Running Average (Window Size = {})'.format(window_size))
        plt.title('Running Average of Scores')
        plt.show()

    def get_relative_distance(self, state):
        ball_y, ball_x = np.unravel_index(np.argmax(state[0, :, :, -1]), state[0, :, :, -1].shape)
        paddle_y, paddle_x = np.unravel_index(np.argmax(state[0, :, :, -2]), state[0, :, :, -2].shape)
        relative_x = (paddle_x - ball_x) / state.shape[1]
        relative_y = (paddle_y - ball_y) / state.shape[2]
        return np.array([[relative_x, relative_y]])

    def calculate_distance(self, paddle_pos, ball_pos):
        return abs(paddle_pos - ball_pos)

    def build_model(self) -> Sequential:
        """
        Define the DQN model
        :return: Sequential model
        """

        # Image input
        image_input = Input(shape=self.state_shape)
        x1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu', kernel_initializer='he_uniform')(image_input)
        x1 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu', kernel_initializer='he_uniform')(x1)
        x1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform')(x1)
        x1 = Flatten()(x1)

        # Relative x and y distances input
        relative_xy_input = Input(shape=self.relative_xy_shape)

        # Concatenate flattened Conv2D output and relative x, y distances
        x = Concatenate()([x1, relative_xy_input])

        # Dense layers
        x = Dense(512, activation='relu', kernel_initializer='he_uniform')(x)
        output = Dense(self.action_space, activation='linear', kernel_initializer='he_uniform')(x)

        model = Model(inputs=[image_input, relative_xy_input], outputs=output)

        model.compile(loss='mse', optimizer=RMSprop(learning_rate=self.learning_rate))

        return model

    def update_target_model(self):
        """
        The target model helps to stabilize the learning process by breaking the correlation between the target and
        the predicted Q-values. Additionally, it helps with the moving target problem.
        """
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        """
        Explore or exploit
        :param state: stack of 4 states
        :return: action
        """
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, 3)
        else:
            relative_distance = self.get_relative_distance(state)
            q_value = self.model.predict([state, relative_distance], verbose=0)
            return np.argmax(q_value[0])

    def append_sample(self, state, action, reward, next_state, terminal, relative_distance):
        self.memory.append((state, action, reward, next_state, terminal, relative_distance))

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train_model(self):
        if self.prioritized_memory:
            mini_batch, idxs, is_weights = self.memory.sample(self.batch_size)
        else:
            mini_batch = random.sample(self.memory, self.batch_size)
            is_weights = np.ones(self.batch_size)

        update_input = np.zeros((self.batch_size, *self.state_shape))
        update_target = np.zeros((self.batch_size, *self.state_shape))
        relative_distance_input = np.zeros((self.batch_size, *self.relative_xy_shape))

        action, reward, terminal = [], [], []

        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            terminal.append(mini_batch[i][4])
            relative_distance_input[i] = mini_batch[i][5]

        target = self.model.predict([update_input, relative_distance_input], verbose=0)
        target_val = self.target_model.predict([update_target, relative_distance_input], verbose=0)

        for i in range(self.batch_size):
            if terminal[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * (np.amax(target_val[i]))

        if self.prioritized_memory:
            history = self.model.fit([update_input, relative_distance_input], target, batch_size=self.batch_size,
                                     epochs=1, verbose=0,
                                     sample_weight=is_weights)
            abs_td_errors = np.abs(target - self.model.predict([update_input, relative_distance_input], verbose=0))
            abs_td_errors = abs_td_errors.mean(axis=1)
            self.memory.update_priorities(idxs, abs_td_errors)
        else:
            self.model.fit([update_input, relative_distance_input], target, batch_size=self.batch_size, epochs=1,
                           verbose=0)

    def warm_up_memory_buffer(self):
        """
        Populate the memory with enough sample to train the network.
        """
        print(f"Warming up the memory buffer for {self.warm_up_episodes} episodes.")
        for episode in range(self.warm_up_episodes):
            self.env.reset()
            state, reward, terminal = self.env.step(1)
            state = np.reshape(state, [1] + list(self.state_shape))

            while not terminal:
                # retrieve an action
                action = self.get_action(state)
                # take a step
                next_state, reward, terminal = self.env.step(action)
                next_state = np.reshape(next_state, [1] + list(self.state_shape))
                # calculate relative distance
                relative_distance = self.get_relative_distance(next_state)
                # append information to the memory buffer
                self.append_sample(state, action, reward, next_state, terminal, relative_distance)
                # update the current state
                state = next_state
        print("Finished warming up.")

    def run_dqn_agent(self, training_episodes=500):
        print(
            f"Training DQN agent for {training_episodes} episodes. Performance will be printed after {self.running_average.maxlen} episodes."
        )

        for episode in range(training_episodes):
            score = 0
            self.env.reset()
            state, reward, terminal = self.env.step(1)
            state = np.reshape(state, [1] + list(self.state_shape))

            while not terminal:
                # retrieve an action
                action = self.get_action(state)

                # Calculate distance before taking action
                prev_distance = self.calculate_distance(self.env.pos, self.env.ballx)

                # take a step
                next_state, reward, terminal = self.env.step(action)
                next_state = np.reshape(next_state, [1] + list(self.state_shape))

                # Calculate distance after taking action
                new_distance = self.calculate_distance(self.env.pos, self.env.ballx)

                # Calculate the distance reward
                if new_distance < prev_distance:
                    distance_reward = 0.1
                elif new_distance == prev_distance:
                    distance_reward = 0.2
                else:
                    distance_reward = -0.1

                # Add the distance reward to the original reward
                total_reward = reward + distance_reward

                # calculate relative distance
                relative_distance = self.get_relative_distance(next_state)
                # append information to memory buffer
                self.append_sample(state, action, total_reward, next_state, terminal, relative_distance)
                # decay epsilon
                self.decay_epsilon()
                # train the neural network
                self.train_model()
                # track the score
                score += reward
                # update the current state
                state = next_state
                # if the episode is over, update the target model
                if terminal:
                    self.running_average.append(score)
                    self.update_target_model()

            # track performance
            self.performance['score'].append(score)
            if episode > self.running_average.maxlen:
                print(
                    f"Episode: {episode} || Epsilon: {self.epsilon:.2f} || Score: {np.mean(self.running_average):.2f}"
                )
            # print the type of memory buffer used

        self.save_data()
        self.plot_running_average()


agent = DQNAgent(prioritized_memory=True)
agent.run_dqn_agent(training_episodes=1500)
