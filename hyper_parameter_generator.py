import itertools
import datetime


def generate_hyperparameter_configs():
    number_of_episodes_options = [500]
    memory_size_options = [10000]
    batch_size_options = [128]
    epsilon_end_options = [0.01]
    epsilon_decay_options = [0.97]
    alpha_options = [0.6]
    beta_options = [0.4]
    target_update_interval_options = [100]
    gamma = [0.99]
    learning_rate_options = [1e-3]
    dropout_rate_options = [0.3]
    l1_reg_options = [1e-4]
    l2_reg_options = [1e-4]
    visualize_options = [False]

    options = [
        number_of_episodes_options,
        memory_size_options,
        batch_size_options,
        epsilon_end_options,
        epsilon_decay_options,
        alpha_options,
        beta_options,
        target_update_interval_options,
        gamma,
        learning_rate_options,
        dropout_rate_options,
        l1_reg_options,
        l2_reg_options,
        visualize_options
    ]

    return list(itertools.product(*options))


def generate_filename(hyperparams: dict, experiment_number: int) -> str:
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    filename = f"experiment{experiment_number}_"

    # Select some key hyperparameters to include in the filename
    key_hyperparams = ["number_of_episodes", "memory_size", "batch_size", "epsilon_end", "epsilon_decay", "alpha",
                        "beta", "gamma"]

    for key in key_hyperparams:
        filename += f"{key}_{hyperparams[key]}_"

    filename += timestamp

    return filename