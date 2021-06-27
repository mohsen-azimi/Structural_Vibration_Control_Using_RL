import tensorflow as tf
from keras import Sequential

from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.optimizers import Adam, RMSprop


class NN:
    """
      Description:
      """

    def __init__(self):
        pass

    @staticmethod
    def simple_nn(n_hidden, n_units, input_shape, action_space):
        # print(input_shape)
        model = Sequential()
        for n in range(n_hidden):
            model.add(Dense(n_units, activation='relu', input_shape=input_shape))
        model.add(Dense(action_space, activation="linear"))
        model.compile(loss="mse", optimizer='adam', metrics=["accuracy"])
        # model.summary()
        return model

