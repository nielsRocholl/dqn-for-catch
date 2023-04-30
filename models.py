from keras.layers import Dense, Multiply, Input, Conv2D, Flatten
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop, SGD


def dqn(input_dims, n_actions) -> Sequential:
    input_state = Input(shape=input_dims)

    first_conv = Conv2D(
        32, (8, 8), strides=(4, 4), activation='relu')(input_state)
    second_conv = Conv2D(
        64, (4, 4), strides=(2, 2), activation='relu')(first_conv)
    third_conv = Conv2D(
        64, (3, 3), strides=(1, 1), activation='relu')(second_conv)

    flattened = Flatten()(third_conv)
    dense_layer = Dense(512, activation='relu')(flattened)

    command_input = Input(shape=(2,))
    sigmoidal_layer = Dense(512, activation='sigmoid')(command_input)

    multiplied_layer = Multiply()([dense_layer, sigmoidal_layer])
    final_layer = Dense(256, activation='relu')(multiplied_layer)

    action_layer = Dense(n_actions, activation='softmax')(final_layer)

    model = Model(inputs=[input_state, command_input], outputs=action_layer)
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.001, rho=0.95, epsilon=0.01))

    return model
