import numpy as np

# 
class ReplayBuffer:
    def __init__(self, max_size, n_states,
                 n_actions, n_batch):
        self.buffer_size = max_size
        self.buffer_cntr = 0
        self.state_buffer = np.zeros((self.buffer_size, n_states))
        self.newstate_buffer = np.zeros((self.buffer_size, n_states))
        self.action_buffer = np.zeros((self.buffer_size, n_actions))
        self.reward_buffer = np.zeros(self.buffer_size)
        self.done_buffer = np.zeros(self.buffer_size)
        self.batch_size = n_batch
        print('batchsize: ', self.batch_size)


    def store(self, state, action, reward, new_state, done):
        index = self.buffer_cntr % self.buffer_size
        self.state_buffer[index] = state
        self.action_buffer[index] = action
        self.newstate_buffer[index] = new_state
        self.reward_buffer[index] = reward
        self.done_buffer[index] = done
        self.buffer_cntr += 1
        # print('stored..')

    def sample(self):
        max_mem = min(self.buffer_cntr, self.buffer_size)
        batch_ind = np.random.choice(max_mem, self.batch_size,
                                 replace=False)
        states = self.state_buffer[batch_ind]
        new_states = self.newstate_buffer[batch_ind]
        actions = self.action_buffer[batch_ind]
        rewards = self.reward_buffer[batch_ind]
        dones = self.done_buffer[batch_ind]
        # print('sampled...')
        return states, new_states, actions, rewards, dones



# Exploaration Action Noise - Ornstein Uhlenbeck
class OU_action_noise:
    def __init__(self, std_deviation, mean= np.array([0]), theta=0.15,
                 dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()
        print('initialized QU_action_noise....')

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


