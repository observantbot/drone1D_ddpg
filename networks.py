import os
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print('optimized...')

# Actor and Critic Network

class ActorCriticNet:
    def __init__(self, n_states, n_actions, chkpt_dir='models'):
        self.n_states = n_states
        self.n_actions = n_actions
        self.nodes = 64
        self.checkpoint_dir = chkpt_dir
        self.init = tf.random_uniform_initializer(minval=-0.003, 
                                                  maxval=0.003)
        

    # deterministic action - policy network
    def actor(self):
        
        # input layer
        inputs = layers.Input(shape=(self.n_states))

        # hidden layer 1
        layer1 = layers.Dense(self.nodes)(inputs)
        batch1 = tf.keras.layers.BatchNormalization()(layer1)
        activation1 = tf.keras.activations.relu(batch1)

        # hidden layer 2
        layer2 = layers.Dense(self.nodes)(activation1)
        batch2 = tf.keras.layers.BatchNormalization()(layer2)
        activation2 = tf.keras.activations.relu(batch2)

        # output layer
        outputs = layers.Dense(self.n_actions, activation="tanh", 
                                kernel_initializer=self.init,
                                bias_initializer=self.init)(activation2)

       
        model = tf.keras.Model(inputs = [inputs], outputs= [outputs])
        return model


    def critic(self):

        # State as input
        state_input = layers.Input(shape=(self.n_states))
        state_layer = layers.Dense(32, activation="relu")(state_input)
        # state_layer = layers.BatchNormalization()(state_layer)

        # Action as input
        action_input = layers.Input(shape=(self.n_actions))
        action_layer = layers.Dense(32, activation="relu")(action_input)

        # concatening state layer and action layer
        layer = layers.Concatenate()([state_layer, action_layer])
        layer = tf.keras.layers.BatchNormalization()(layer)

        # hidden layer 1 
        layer1 = layers.Dense(self.nodes)(layer)
        batch1 = tf.keras.layers.BatchNormalization()(layer1)
        activation1 = tf.keras.activations.relu(batch1)
        # hidden layer 2
        layer2 = layers.Dense(self.nodes)(activation1)
        batch2 = tf.keras.layers.BatchNormalization()(layer2)
        activation2 = tf.keras.activations.relu(batch2)

        # output layer
        outputs = layers.Dense(1, kernel_initializer=self.init,
                                    bias_initializer=self.init)(activation2)

        # output is treated as Q-value for the given state and action 
        model = tf.keras.Model([state_input, action_input], outputs)

        return model

    def checkpoint_file(self, name):
        cf = os.path.join(self.checkpoint_dir, 
                    name + '_ddpg.h5')
        return cf

    
