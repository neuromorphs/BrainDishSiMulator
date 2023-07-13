import tensorflow as tf
import torch
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *

import random
import numpy as np

class CustomDeepNetwork:
    def __init__(self):

        self.scale = 300.0
        self.input_size = 2
        # define online network using kera

        input_x = Input(shape=(self.input_size))
        x1 = Dense(8, activation='relu', use_bias=True, kernel_initializer="zeros", bias_initializer="zeros")(input_x)
        x2 = Dense(4, activation='relu', use_bias=True, kernel_initializer="zeros", bias_initializer="zeros")(x1)
        output_x = Dense(2, activation='linear', use_bias=True, kernel_initializer="zeros", bias_initializer="zeros")(x2)

        model = Model(inputs=input_x, outputs=output_x)
        model.compile(loss='mse', optimizer=Adam(lr=0.0))
        self.model = model

        # set all weights and bias to zeros
        zero_x1 = np.zeros((self.input_size, 8))
        zero_x1_b = np.zeros((8,))
        zero_x2 = np.zeros((8, 4))
        zero_x2_b = np.zeros((4,))
        zero_out = np.zeros((4, 2))
        zero_out_b = np.zeros((2,))

        # set single RELU Path
        zero_x1[1, 0] = 1
        zero_x1[0, 0] = -1
        zero_x2[0, 0] = 1
        zero_out[0, 0] = 1
        zero_out_b = np.asarray([0.0, 0.3])

        model.layers[1].set_weights((zero_x1, zero_x1_b))
        model.layers[2].set_weights((zero_x2, zero_x2_b))
        model.layers[3].set_weights((zero_out, zero_out_b))

        self.model = model

    def get_action(self, state):
        # convert state
        s = (np.asarray(state) - 300) / self.scale

        if self.input_size == 2:
            s = s[[0, 2]]

        out = self.model.predict(np.asarray([s]), verbose=False)
        action = np.argmax(out[0])
        return action

    def update(self, state, action, reward, next_state, done):
        pass

    def get_weights(self):
        weights = self.model.get_weights()
        weights = [torch.from_numpy(w) for w in weights]
        return weights
