from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras.layers import Conv2D, Dense, Flatten, Dropout
from keras.regularizers import l1_l2


def build_dqn(n_actions, input_dims, lr=0.00025) -> Sequential:
    """
    Build a DQN model
    :param n_actions: number of actions
    :param input_dims: number of input dimensions
    :param lr: learning rate
    :return: compiled DQN model
    """

    # define the model
    model = Sequential([
        Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=input_dims),
        Conv2D(64, (4, 4), strides=(2, 2), activation='relu', data_format='channels_first'),
        Conv2D(64, (3, 3), strides=(1, 1), activation='relu', data_format='channels_first'),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(n_actions)
    ])

    # compile the model
    model.compile(optimizer=RMSprop(learning_rate=lr), loss='huber_loss')
    return model


def build_dqn_hp_search(n_actions, input_dims, lr=0.00025, dropout_rate=0.3, l1_reg=1e-4, l2_reg=1e-4) -> Sequential:
    """
    Build a DQN model
    :param n_actions: number of actions
    :param input_dims: number of input dimensions
    :param lr: learning rate
    :param dropout_rate: dropout rate for regularization
    :param l1_reg: L1 regularization factor
    :param l2_reg: L2 regularization factor
    :return: compiled DQN model
    """

    # define the model
    model = Sequential([
        Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=input_dims),
        Dropout(dropout_rate),
        Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
        Dropout(dropout_rate),
        Conv2D(64, (2, 2), strides=(1, 1), activation='relu'),
        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)),
        Dropout(dropout_rate),
        Dense(n_actions)
    ])

    # compile the model
    model.compile(optimizer=RMSprop(learning_rate=lr), loss='huber_loss')
    return model


def dqn_from_git(n_actions, input_dims, lr=0.00025) -> Sequential:
    model = Sequential([
        Conv2D(32, kernel_size=8, strides=4, padding="same", kernel_initializer="normal", input_shape=input_dims,
               activation="relu"),
        Conv2D(64, kernel_size=4, strides=2, kernel_initializer="normal", padding="same", activation="relu"),
        Conv2D(64, kernel_size=3, strides=1, kernel_initializer="normal", padding="same", activation="relu"),
        Flatten(),
        Dense(512, kernel_initializer="normal", activation="relu"),
        Dense(n_actions, kernel_initializer="normal")
    ])

    model.compile(optimizer=Adam(learning_rate=lr), loss="mse")
    return model
