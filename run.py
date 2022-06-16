import os
import numpy as np
import time
from pybulletsim import init_simulation, end_simulation
from env import Quadcopter
from main import Agent



'-------------------------------------'
drone, marker = init_simulation(render = True)

env = Quadcopter(drone, marker)
agent = Agent()
n_episode = 12
max_step = 3000     
# maximum number of steps per episode
'----------------------'
eps_start = 11
try:
    agent.load_models(eps_start-1)
except:
    print('..not loaded..')
'-----------------------'



def train(env, agent, n_episode, max_step):



    for eps in range(eps_start,n_episode):

        state = env.reset()
        episodic_reward = 0
        agent.noise.reset()

        for t in range(max_step):

            # take action according to our current policy with noise
            action = agent.get_action(state)
            # print('action: ',action)

            # observe reward and next state after executing action.
            new_state, reward, done, _ = env.step(action[0])
            # print('state = {}, reward = {}, done = {}'.\
            #     format(new_state, reward, done))

            # store transition in buffer
            agent.store_transition(state, action, reward, new_state, done)
            
            # train agent
            agent.learn()

            state = new_state
            episodic_reward += reward

            # Terminate episode when `done` is True
            if done:
                print('reward at done: {} and episodic reward\
                    for time {} is {}'.format(reward, t, episodic_reward))
                break

        if eps%10 == 0:
            agent.save_models(eps)
    pass



train(env, agent, n_episode, max_step)



end_simulation()





















# eps_reward_list = []
# eps_time_list = []
# eps_done_list = []
# eps_pos = []
# bestscore = 0

# for eps in range(1,n_episode):

#     state = env.reset()
#     episodic_reward = 0
#     agent.noise.reset()
#     tic = time.time()
#     for t in range(max_step):

#         # take action according to our current policy with noise
#         action = agent.get_action(state)

#         # observe reward and next state after executing action.
#         new_state, reward, done, info = env.step(action[0])

#         # store transition in buffer
#         agent.store_transition(state, action, reward, new_state, done)

#         # It is because of tf.function decorator used in agent
#         # if agent.buffer.buffer_cntr >= agent.batch_size:
#         agent.learn()

#         state = new_state
#         episodic_reward += reward
#         # Terminate episode when `done` is True
#         if done:
#             break
#         # time.sleep(0.01)
#     tac = time.time()
#     print('time taken:', tac-tic)
#     eps_reward_list.append(round(episodic_reward,3))
#     eps_time_list.append(t)
#     eps_done_list.append(info)
#     print('eps:',eps, ' completed with t',t,' and reward:', eps_reward_list[-1],'\n')

#     # evaluate and save model + csv files
#     if (eps%10==0):
#         bestscore = evaluate(env, agent, eps_reward_list, eps_time_list, eps_done_list, bestscore)
#         eps_reward_list = []
#         eps_time_list = [] 
#         eps_done_list = []