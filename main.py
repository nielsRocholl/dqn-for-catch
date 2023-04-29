import datetime
import pandas as pd

from hyper_parameter_generator import generate_hyperparameter_configs, generate_filename
# import custom modules
from train_model import train_rl_agent
from catch_environment import CatchEnv
from models import build_dqn, build_dqn_hp_search, dqn_from_git
from test_model import test_rl_agent
from visualization import plot


def hyperparameter_search(env: CatchEnv):
    params = [{"number_of_episodes": p[0],
               "memory_size": p[1],
               "batch_size": p[2],
               "epsilon": 1.0,
               "epsilon_end": p[3],
               "epsilon_decay": p[4],
               "alpha": p[5],
               "beta": p[6],
               "gamma": p[7],
               "target_update_interval": p[8],
               "learning_rate": p[9],
               "dropout_rate": p[10],
               "l1_reg": p[11],
               "l2_reg": p[12],
               "visualize": p[13],
               "observation_steps": 128
               } for p in generate_hyperparameter_configs()]

    for i, hyperparams in enumerate(params):
        print(f"Running experiment {i + 1} with hyperparameters: {hyperparams}")

        # get the number of actions and input dimensions
        n_actions = env.get_num_actions()
        input_dims = env.state_shape()
        # transpose the input dimensions (because env returns (frames, height, width))
        input_dims = (input_dims[1], input_dims[2], input_dims[0])

        # build the model
        # dqn = build_dqn(
        #     input_dims=input_dims,
        #     n_actions=n_actions,
        #     lr=hyperparams["learning_rate"],
        # )

        dqn = build_dqn_hp_search(
            input_dims=input_dims,
            n_actions=n_actions,
            lr=hyperparams["learning_rate"],
            dropout_rate=hyperparams["dropout_rate"],
            l1_reg=hyperparams["l1_reg"],
            l2_reg=hyperparams["l2_reg"]
        )

        # train the model
        trained_model, rewards, losses, wins = train_rl_agent(params=hyperparams, env=env, model=dqn)

        # save the rewards list, with the hyperparameters as the filename in performances/
        file_name = generate_filename(hyperparams, i)
        pd.DataFrame(rewards, index=range(len(rewards))).to_csv(f"performances/reward_{file_name}.csv")
        pd.DataFrame(losses, index=range(len(losses))).to_csv(f"performances/loss_{file_name}.csv")
        pd.DataFrame(losses, index=range(len(losses))).to_csv(f"performances/loss_{file_name}.csv")
        pd.DataFrame(wins, index=range(len(wins))).to_csv(f"performances/wins_{file_name}.csv")

        # save the model
        trained_model.save(f'trained_models/{file_name}.h5')

    plot()


def main():
    # set to True to test the model
    TEST = False

    # initialize the environment
    env = CatchEnv()

    if TEST:
        test_rl_agent(model_path='trained_models/experiment0_number_of_episodes_3000_memory_size_50000_batch_size_32_epsilon_end_0.0001_epsilon_decay_0.95_alpha_0.6_beta_1.0_gamma_50_20230428-161659.h5', env=env)
    else:
        # train the model
        hyperparameter_search(env=env)


if __name__ == '__main__':
    main()
