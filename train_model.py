import numpy as np
from collections import deque
from keras import Sequential

# import custom modules
from catch_environment import CatchEnv
from visualization import visualize, destroy


class PrioritizedReplayBuffer:
    def __init__(self, max_size, alpha):
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.alpha = alpha

    def add(self, experience, priority):
        self.buffer.append(experience)
        self.priorities.append(priority ** self.alpha)

    def sample(self, batch_size, beta):
        priorities = np.array(self.priorities)
        probabilities = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        experiences = [self.buffer[i] for i in indices]
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        return experiences, indices, weights

    def update_priorities(self, indices, priorities):
        for i, priority in zip(indices, priorities):
            self.priorities[i] = priority ** self.alpha

    def __len__(self):
        return len(self.buffer)


def train_rl_agent(params: dict, env: CatchEnv, model: Sequential):
    alpha = 0.6
    beta = 0.4
    memory = PrioritizedReplayBuffer(params['memory_size'], alpha)
    moving_avg_reward = deque(maxlen=10)
    all_rewards = []

    for ep in range(params['number_of_episodes']):
        env.reset()
        state, reward, terminal = env.step(1)
        state = np.expand_dims(state, axis=0)

        while not terminal:
            if params['visualize']:
                visualize(state)

            if np.random.rand() < params['epsilon']:
                action = np.random.randint(0, 3)
            else:
                q_values = model.predict(state)
                action = np.argmax(q_values)

            next_state, reward, terminal = env.step(action)
            next_state = np.squeeze(next_state)
            next_state = np.expand_dims(next_state, axis=0)

            initial_priority = max(1, np.max(memory.priorities)) if memory else 1
            memory.add((state, action, reward, next_state, terminal), initial_priority)

            state = next_state

            if terminal:
                moving_avg_reward.append(reward)
                all_rewards.append(reward)

            if len(memory) > params['batch_size']:
                batch, indices, weights = memory.sample(params['batch_size'], beta)
                states, actions, rewards, next_states, terminals = zip(*batch)
                states, next_states = np.concatenate(states), np.concatenate(next_states)
                q_values = model.predict(states)
                next_q_values = model.predict(next_states)

                for i, (s, a, r, ns, t) in enumerate(batch):
                    target = r + (1 - int(t)) * np.max(next_q_values[i])
                    q_values[i][a] = target
                model.train_on_batch(states, q_values)

                td_errors = np.abs(q_values - model.predict(states))
                priorities = np.max(td_errors, axis=1)
                memory.update_priorities(indices, priorities)

                beta = min(1.0, beta + 0.001)

            params['epsilon'] = max(params['epsilon_end'], params['epsilon'] * params['epsilon_decay'])
        print(f"Episode {ep + 1} completed. Moving avg reward: {sum(moving_avg_reward) / len(moving_avg_reward):.2f}")

    if params['visualize']:
        destroy()

    return model, all_rewards

# def train_rl_agent(params: dict, env: CatchEnv, model: Sequential):
#     memory = deque(maxlen=params['memory_size'])
#     moving_avg_reward = deque(maxlen=10)
#     all_rewards = []
#
#     for ep in range(params['number_of_episodes']):
#         # reset the environment
#         env.reset()
#         state, reward, terminal = env.step(1)
#         # add an extra dimension to the state
#         state = np.expand_dims(state, axis=0)
#
#         while not terminal:
#             if params['visualize']:
#                 visualize(state)
#
#             # explore or exploit
#             if np.random.rand() < params['epsilon']:
#                 # take a random action
#                 action = np.random.randint(0, 3)
#             else:
#                 # take the action with the highest q-value
#                 q_values = model.predict(state)
#                 action = np.argmax(q_values)
#
#             next_state, reward, terminal = env.step(action)
#             next_state = np.squeeze(next_state)
#             # add an extra dimension to the state
#             next_state = np.expand_dims(next_state, axis=0)
#
#             # add the experience to the memory
#             memory.append((state, action, reward, next_state, terminal))
#
#             # update the state
#             state = next_state
#
#             if terminal:
#                 moving_avg_reward.append(reward)
#                 all_rewards.append(reward)
#
#             if len(memory) > params['batch_size']:
#                 # sample a batch of experiences from the memory to break the correlation
#                 batch = random.sample(memory, params['batch_size'])
#                 states, actions, rewards, next_states, terminals = zip(*batch)
#                 states, next_states = np.concatenate(states), np.concatenate(next_states)
#                 q_values = model.predict(states)
#                 next_q_values = model.predict(next_states)
#
#                 for i, (s, a, r, ns, t) in enumerate(batch):
#                     target = r + (1 - int(t)) * np.max(next_q_values[i])
#                     q_values[i][a] = target
#                 model.train_on_batch(states, q_values)
#
#             params['epsilon'] = max(params['epsilon_end'], params['epsilon'] * params['epsilon_decay'])
#         print(f"Episode {ep + 1} completed. Moving avg reward: {sum(moving_avg_reward) / len(moving_avg_reward):.2f}")
#
#     if params['visualize']:
#         destroy()
#
#     return model, all_rewards
