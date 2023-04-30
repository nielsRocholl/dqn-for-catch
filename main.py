import datetime
import pandas as pd

# import custom modules
from train_model import train_rl_agent
from catch_environment import CatchEnv
from models import build_dqn
from test_model import test_rl_agent
from visualization import plot


def hyperparameter_search(env: CatchEnv):
    params = {
        "number_of_episodes": 400,
        "memory_size": 10000,
        "batch_size": 128,
        "epsilon": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.97,
        "alpha": 0.6,  # Prioritized experience replay hyperparameter
        "beta": 0.4,  # Prioritized experience replay hyperparameter
        "target_update_interval": 100,  # Target model update interval
        "learning_rate": 1e-3,
        "l1_reg": 1e-4,
        "l2_reg": 1e-4,
        "visualize": False,
    }

    # get the number of actions and input dimensions
    n_actions = env.get_num_actions()
    input_dims = env.state_shape()
    # transpose the input dimensions (because env returns (frames, height, width))
    input_dims = (input_dims[1], input_dims[2], input_dims[0])

    # build the model
    dqn = build_dqn(
        n_actions,
        input_dims=input_dims,
        lr=params["learning_rate"],
    )

    # train the model
    trained_model, rewards, losses = train_rl_agent(params=params, env=env, model=dqn)

    # save the rewards list, with the hyperparameters as the filename in performances/
    file_name = "experiment"
    pd.DataFrame(rewards, index=range(len(rewards))).to_csv(f"performances/reward_{file_name}.csv")
    pd.DataFrame(losses, index=range(len(losses))).to_csv(f"performances/loss_{file_name}.csv")

    # save the model
    trained_model.save(f'trained_models/{file_name}.h5')

    plot()


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
