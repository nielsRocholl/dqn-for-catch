from dqn import DQNAgent
import itertools

experiments = [
    {
        'prioritized_memory': True,
        'learning_rate_schedule': False,
        'smart_reward': False,
        'beta_increment': False,
        'dueling': False,
        'double': False,
        'gradient_clipping': False
    },
    {
        'prioritized_memory': False,
        'learning_rate_schedule': True,
        'smart_reward': False,
        'beta_increment': False,
        'dueling': False,
        'double': False,
        'gradient_clipping': False
    },
    {
        'prioritized_memory': False,
        'learning_rate_schedule': False,
        'smart_reward': True,
        'beta_increment': False,
        'dueling': False,
        'double': False,
        'gradient_clipping': False
    },
    {
        'prioritized_memory': False,
        'learning_rate_schedule': False,
        'smart_reward': False,
        'beta_increment': True,
        'dueling': False,
        'double': False,
        'gradient_clipping': False
    },
    {
        'prioritized_memory': False,
        'learning_rate_schedule': False,
        'smart_reward': False,
        'beta_increment': False,
        'dueling': True,
        'double': True,
        'gradient_clipping': False
    },
    {
        'prioritized_memory': False,
        'learning_rate_schedule': False,
        'smart_reward': False,
        'beta_increment': False,
        'dueling': False,
        'double': False,
        'gradient_clipping': True
    },
]


def run_experiments(experiments):
    print(f"running {len(experiments)} hyperparameter experiments")
    for exp in experiments:
        print(f"Running experiment with hyperparameters: {exp}")
        agent = DQNAgent(exp['prioritized_memory'])
        agent.prioritized_memory = exp['prioritized_memory']
        agent.learning_rate_schedule = exp['learning_rate_schedule']
        agent.smart_reward = exp['smart_reward']
        agent.beta_increment = exp['beta_increment']
        agent.dueling = exp['dueling']
        agent.double = exp['double']
        agent.gradient_clipping = exp['gradient_clipping']
        agent.warm_up_memory_buffer()
        agent.run_dqn_agent()


run_experiments(experiments)
