from keras.models import load_model
import numpy as np

from visualization import visualize


def test_rl_agent(model_path, env, num_episodes=50):
    # Load the trained model
    model = load_model(model_path)
    reward_list = []

    for ep in range(num_episodes):
        # Reset the environment
        env.reset()
        state, reward, terminal = env.step(1)
        state = np.expand_dims(state, axis=0)

        while not terminal:
            # Visualize the current state
            visualize(state)

            # Choose the action with the highest Q-value
            q_values = model.predict(state)
            action = np.argmax(q_values)

            # Perform the action and update the state
            next_state, reward, terminal = env.step(action)
            next_state = np.expand_dims(next_state, axis=0)
            state = next_state

            # if we are not at the end of the episode, add the reward to the list
            if terminal:
                reward_list.append(reward)

        print(f"Episode {ep + 1} completed. Total reward: {sum(reward_list)/len(reward_list):.2f}")