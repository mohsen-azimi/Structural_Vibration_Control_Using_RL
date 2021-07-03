from keras import Sequential
from keras.layers import Input, Dense
from keras.optimizers import Adam


class NN:
    """
      Description:
      """

    def __init__(self):
        pass

    @staticmethod
    def simple_nn(lr, n_hidden, n_units, input_shape, n_actions):
        # print(input_shape)
        model = Sequential()
        for n in range(n_hidden):
            model.add(Dense(n_units, activation='relu', input_shape=input_shape))
        model.add(Dense(n_actions, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=lr), metrics=["accuracy"])
        model.summary()
        return model

