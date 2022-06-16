import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from networks import ActorCriticNet
from buffer import ReplayBuffer, OU_action_noise
import numpy as np

class Agent:
    def __init__(self, n_states = 2, n_actions = 1,  actor_lr=0.0001, 
                    critic_lr=0.001, tau = 0.001, std_dev_noise=0.2,
                    batch_size=64, gamma = 0.99, max_size=1000000): 

        self.buffer = ReplayBuffer(max_size=max_size,
                                   n_states=n_states, n_actions=n_actions,
                                    n_batch=batch_size)
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.noise  = OU_action_noise(std_deviation=std_dev_noise*np.ones(n_actions))
    
        self.batch_size = batch_size
        # self.action_high = 2*1.5*9.81           #2mg maximum thrust force

        network = ActorCriticNet(n_states, n_actions)
        self.actor = network.actor()
        self.critic = network.critic()
        self.target_actor = network.actor()
        self.target_critic = network.critic()

        # why did not we define the loss function here itself?
        self.actor.compile(optimizer=Adam(learning_rate=actor_lr))
        self.critic.compile(optimizer=Adam(learning_rate=critic_lr))
        self.target_actor.compile(optimizer=Adam(learning_rate=actor_lr))
        self.target_critic.compile(optimizer=Adam(learning_rate=critic_lr))

        self.cf = network.checkpoint_file
        # copy the exact parameters from actor and critic networks, 
        # initialization of actor and critic target networks
        self.update_target_network(self.target_actor.variables, 
                                        self.actor.variables, tau=1)
        self.update_target_network(self.target_critic.variables, 
                                        self.critic.variables, tau=1)


    # policy, take state as input and output an action with noise.
    def get_action(self, state, training=True):
        state = tf.convert_to_tensor([state], 
                                        dtype=tf.float32)
        action = tf.squeeze(self.actor(state)).numpy()

        if training:
            action += self.noise()
            # action += tf.random.normal(shape=[self.n_actions],
            #                              mean=0.0, stddev=self.noise)
        
        # action upper bound after addition of exploration noise
        action = np.clip(action, -1, 1)
        return [np.squeeze(action)]

    # @tf.function
    def learn(self):
        if self.buffer.buffer_cntr < self.batch_size:
            # don't learn
            print('not yet')
            return
        
        state_batch, newstate_batch, action_batch,\
            reward_batch, done_batch = self.buffer.sample()

        state_batch = tf.convert_to_tensor(state_batch, dtype=tf.float32)
        newstate_batch = tf.convert_to_tensor(newstate_batch, dtype=tf.float32)
        action_batch = tf.convert_to_tensor(action_batch, dtype=tf.float32)
        reward_batch = tf.convert_to_tensor(reward_batch, dtype=tf.float32)
        
        # critic update
        with tf.GradientTape() as t:
            target_actions = self.target_actor(newstate_batch)
            q_values_ = self.target_critic([newstate_batch,target_actions])
            targets = reward_batch + self.gamma*q_values_*(1-done_batch) 

            q_values = self.critic([state_batch, action_batch]) 
            critic_loss = keras.losses.MSE(targets, q_values)
        critic_grad = t.gradient(critic_loss, 
                            self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(
                        critic_grad, self.critic.trainable_variables))
        # print('critic_network updated--------')    

        # actor update
        with tf.GradientTape() as t:
            actions = self.actor(state_batch)
            q_values = self.critic([state_batch,actions])

            actor_loss = -tf.math.reduce_mean(q_values)
        actor_grad = t.gradient(actor_loss, 
                            self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
                        actor_grad, self.actor.trainable_variables))

        # print('actor_network updated--------')  

        # update target networks
        self.update_target_network(self.target_actor.variables, 
                                        self.actor.variables)
        self.update_target_network(self.target_critic.variables, 
                                        self.critic.variables)        

          

    
    # Update the weights of target network
    def update_target_network(self,target_weights, weights, tau=None):
        if tau is None:
            tau = self.tau
        
        '''Theta' = Theta * tau + Theta'_prev * (1-tau)
           Theta and Theta' are the parameters of actor
           and target actor networks respectively.'''
           
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))
        
        '''Phi' = Phi * tau + Phi'_prev * (1-tau)
           Phi and Phi' are the parameters of critc
           and target critic networks respectively.'''
    
    # save model
    def save_models(self, eps):
        print('... saving models ... at eps {}'.format(eps))
        self.actor.save_weights(self.cf('actor_' + str(eps)))
        self.target_actor.save_weights(self.cf('target_actor_' + str(eps)))
        self.critic.save_weights(self.cf('critic_' + str(eps)))
        self.target_critic.save_weights(self.cf('target_critic_' + str(eps)))

    # load model
    def load_models(self, eps):
        print('... loading models ...')
        self.actor.load_weights(self.cf('actor_' + str(eps)))
        self.target_actor.load_weights(self.cf('target_actor_' + str(eps)))
        self.critic.load_weights(self.cf('critic_' + str(eps)))
        self.target_critic.load_weights(self.cf('target_critic_' + str(eps)))

    # store transition
    def store_transition(self, state, action, reward, new_state, done):
        self.buffer.store(state, action, reward, new_state, done)
        
