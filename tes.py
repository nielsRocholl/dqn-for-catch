class DQNAgent:
    def __init__(self, prioritized_memory):
        # define environment
        self.env = CatchEnv()

        # define hyperparameters
        self.state_shape = (84, 84, 4)
        self.action_space = 3
        self.discount_factor = 0.99
        self.learning_rate = 0.001  # was 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 128
        self.current_epoch = 0
        self.training_episodes = 1500
        self.warm_up_episodes = self.batch_size * 2
        self.memory_size = 5000
        self.beta_incr = (1.0 - 0.4) / self.training_episodes
        self.prioritized_memory = prioritized_memory
        self.memory = PrioritizedReplayBuffer(self.memory_size) if self.prioritized_memory else deque(maxlen=2000)
        self.beta_increment = True
        self.smart_reward = True
        self.smart_feature = True
        self.dueling = True
        self.double = True
        self.learning_rate_schedule = True
        self.gradient_clipping = True

        # define models
        self.model = self.build_model()
        self.target_model = self.build_model()

        # define performance datastructure
        self.performance = {
            "score": [],
            "loss": [],
            "test_score": []
        }
        self.running_average = deque(maxlen=40)

        # warm up buffer
        # self.warm_up_memory_buffer()

    def learning_rate_schedule_func(self):
        if self.learning_rate_schedule:
            initial_lr = self.learning_rate
            final_lr = 0.0001
            decay_rate = (initial_lr - final_lr) / self.training_episodes
            lr = initial_lr - (self.current_epoch * decay_rate)
            return lr
        else:
            return self.learning_rate

    def predict_ball_landing(self) -> int:
        .....

    def find_player_x(self):
        ....

    def save_data(self):
        ....

    def plot_running_average(self, window_size=50):
        ....

    def build_model(self) -> Sequential:
        """
        Define the model, can be either regular or dueling
        :return: Sequential model
        """

        input_layer = Input(shape=self.state_shape)
        conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu', kernel_initializer='he_uniform')(input_layer)
        conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu', kernel_initializer='he_uniform')(conv1)
        conv3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform')(conv2)
        flatten = Flatten()(conv3)

        if self.dueling:
            # Dueling DQN architecture
            state_value = Dense(1, kernel_initializer='he_uniform')(
                Dense(512, activation='relu', kernel_initializer='he_uniform')(flatten))
            action_advantage = Dense(self.action_space, kernel_initializer='he_uniform')(
                Dense(512, activation='relu', kernel_initializer='he_uniform')(flatten))
            action_mean = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(action_advantage)
            action_centered = Subtract()([action_advantage, action_mean])
            q_values = Add()([state_value, action_centered])
        else:
            # Standard DQN architecture
            q_values = Dense(self.action_space, activation='linear', kernel_initializer='he_uniform')(
                Dense(512, activation='relu', kernel_initializer='he_uniform')(flatten))

        model = Model(inputs=input_layer, outputs=q_values)
        if self.gradient_clipping:
            model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate, clipvalue=1.0))
        else:
            model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))

        return model

    def update_target_model(self):
        """
        The target model helps to stabilize the learning process by breaking the correlation between the target and
        the predicted Q-values. Additionally, it helps with the moving target problem.
        """
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state, greedy=False) -> [int]:
        """
        Explore or exploit
        :param greedy: if True, the agent will always exploit
        :param state: stack of 4 states
        :return: action
        """
        if greedy:
            q_value = self.model.predict(state, verbose=0)
            return np.argmax(q_value[0])
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, 3)
        else:
            q_value = self.model.predict(state, verbose=0)
            return np.argmax(q_value[0])

    def append_sample(self, state, action, reward, next_state, terminal):
        self.memory.append((state, action, reward, next_state, terminal))

    def decay_epsilon(self):
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def evaluate(self, episode) -> None:
        .......

    def train_model(self) -> None:
        if self.prioritized_memory:
            mini_batch, idxs, is_weights = self.memory.sample(self.batch_size)
        else:
            mini_batch = random.sample(self.memory, self.batch_size)
            is_weights = np.ones(self.batch_size)
        update_input = np.zeros((self.batch_size, *self.state_shape))
        update_target = np.zeros((self.batch_size, *self.state_shape))

        action, reward, terminal = [], [], []

        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            terminal.append(mini_batch[i][4])

        target = self.model.predict(update_input, verbose=0)
        target_next = self.model.predict(update_target, verbose=0)
        target_val = self.target_model.predict(update_target, verbose=0)

        for i in range(self.batch_size):
            if terminal[i]:
                target[i][action[i]] = reward[i]
            else:
                if self.double:
                    # Double DQN update rule
                    a = np.argmax(target_next[i])
                    target[i][action[i]] = reward[i] + self.discount_factor * target_val[i][a]
                else:
                    target[i][action[i]] = reward[i] + self.discount_factor * (np.amax(target_val[i]))

        if self.prioritized_memory:
            history = self.model.fit(update_input, target, batch_size=self.batch_size, epochs=1, verbose=0,
                                     sample_weight=is_weights)
            abs_td_errors = np.abs(target - self.model.predict(update_input, verbose=0))
            abs_td_errors = abs_td_errors.mean(axis=1)
            self.memory.update_priorities(idxs, abs_td_errors)
        else:
            self.model.fit(update_input, target, batch_size=self.batch_size, epochs=1, verbose=0)

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
                next_state, env_reward, terminal = self.env.step(action)
                next_state = np.reshape(next_state, [1] + list(self.state_shape))

                if self.smart_reward:
                    # predict the ball landing position and calculate the smart reward
                    ball_landing = self.predict_ball_landing()
                    player_x = self.find_player_x()
                    distance_to_ball_landing = abs(ball_landing - player_x)
                    # Normalize the distance to the ball landing within the desired range
                    smart_reward = 0.2 - (distance_to_ball_landing / (self.env.size - 1)) * 0.4
                    # append information to memory buffer
                self.append_sample(state, action, smart_reward + env_reward if self.smart_reward else env_reward,
                                   next_state, terminal)
                # update the current state
                state = next_state
        print("Finished warming up.")

    def test_trained_agent(self, file):
        model = keras.models.load_model(file)
        scores = []
        for episode in range(50):
            score = 0
            self.env.reset()
            state, reward, terminal = self.env.step(1)
            while not terminal:
                visualize(state)
                action = model.predict(state.reshape(1, *self.state_shape))
                state, reward, terminal = self.env.step(np.argmax(action))
                score += reward
            scores.append(score)
            print(f'Episode {episode + 1} Score: {score}')
        print(f"Average score over 20 episodes: {np.mean(scores)}")

    def run_dqn_agent(self):
        print(
            f"Training DQN agent for {self.training_episodes} episodes. Performance will be printed after {self.running_average.maxlen} episodes.")

        for episode in range(self.training_episodes):
            score = 0
            self.env.reset()
            state, reward, terminal = self.env.step(1)
            state = np.reshape(state, [1] + list(self.state_shape))

            while not terminal:
                # retrieve an action
                action = self.get_action(state)
                # take a step
                next_state, env_reward, terminal = self.env.step(action)
                next_state = np.reshape(next_state, [1] + list(self.state_shape))

                if self.smart_reward:
                    # predict the ball landing position and calculate the smart reward
                    ball_landing = self.predict_ball_landing()
                    player_x = self.find_player_x()
                    distance_to_ball_landing = abs(ball_landing - player_x)
                    # Normalize the distance to the ball landing within the desired range
                    normalized_distance = distance_to_ball_landing / (self.env.size - 1)
                    # Calculate the smart_reward by scaling the normalized distance to a smaller range, e.g. [-0.1, 0.1]
                    smart_reward = 0.1 - normalized_distance * 0.2
                # append information to memory buffer
                self.append_sample(state, action, smart_reward + env_reward if self.smart_reward else env_reward,
                                   next_state, terminal)

                # train the neural network
                self.train_model()
                # decay epsilon
                self.decay_epsilon()
                # track the score
                score += env_reward
                # update the current state
                state = next_state
                # if the episode is over, update the target model
                if terminal:
                    self.running_average.append(score)
                    self.update_target_model()

            # evaluate the agent's policy every 10 episodes
            if (episode + 1) % 10 == 0:
                self.evaluate(episode)

            # TODO: Increment the beta parameter - check if this improves performance
            if self.prioritized_memory and self.memory.beta < 1.0 and self.beta_increment:
                self.memory.beta += self.beta_incr

            # Update the learning rate according to the schedule
            lr = self.learning_rate_schedule_func()
            K.set_value(self.model.optimizer.learning_rate, lr)
            self.current_epoch += 1

        self.save_data()
        self.plot_running_average()