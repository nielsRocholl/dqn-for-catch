import datetime
import pandas as pd

# import custom modules
from train_model import train_rl_agent
from catch_environment import CatchEnv
from models import build_dqn, build_dqn_hp_search
from test_model import test_rl_agent
from visualization import plot_moving_average


def hyperparameter_search(env: CatchEnv):
    params = [
        {
            "number_of_episodes": 10,
            "memory_size": 10000,
            "batch_size": 128,
            "epsilon": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.97,
            "alpha": 0.6,  # Prioritized experience replay hyperparameter
            "beta": 0.4,  # Prioritized experience replay hyperparameter
            "target_update_interval": 100,  # Target model update interval
            "lr": 1e-3,
            "dropout_rate": 0.3,
            "l1_reg": 1e-4,
            "l2_reg": 1e-4,
            "visualize": False,
        }
    ]

    # initialize a list to store the results
    results = []

    for i, hyperparams in enumerate(params):
        print(f"Running experiment {i + 1} with hyperparameters: {hyperparams}")

        # get the number of actions and input dimensions
        n_actions = env.get_num_actions()
        input_dims = env.state_shape()
        # transpose the input dimensions (because env returns (frames, height, width))
        input_dims = (input_dims[1], input_dims[2], input_dims[0])

        # build the model
        dqn = build_dqn_hp_search(
            n_actions,
            input_dims=input_dims,
            lr=hyperparams["lr"],
            dropout_rate=hyperparams["dropout_rate"],
            l1_reg=hyperparams["l1_reg"],
            l2_reg=hyperparams["l2_reg"],
        )

        # train the model
        trained_model, all_rewards = train_rl_agent(params=hyperparams, env=env, model=dqn)

        # append the rewards to the results list
        results.append(all_rewards)

        # save the rewards list, with the hyperparameters as the filename in performances/
        now = datetime.datetime.now()
        hour = now.hour
        minute = now.minute
        pd.DataFrame(all_rewards, index=range(len(all_rewards))).to_csv(f"performances/experiment{i}_{hour}:{minute}.csv")

        # save the model
        trained_model.save(f'trained_models/experiment{i}_{hour}:{minute}.h5')

    plot_moving_average(window_size=50)


def main():
    # set to True to test the model
    TEST = False

    # initialize the environment
    env = CatchEnv()

    if TEST:
        test_rl_agent(model_path='trained_models/model20230420-204115.h5', env=env)
    else:
        # train the model
        hyperparameter_search(env=env)


if __name__ == '__main__':
    main()
