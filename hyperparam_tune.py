import os
import argparse
import numpy as np
from env import Quadcopter
from utility import plotLearning
from main import Agent
import pybullet as p
import pybullet_data
import tensorflow as tf


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print('optimized...')


physicsClient = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.81)

'------------------------------------'
# drone
drone = p.loadURDF(os.path.join(os.getcwd(),'pybullet/drone_test/drone_test.urdf'))

# marker at desired point
sphereVisualId = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                    radius = 0.05,
                                    rgbaColor= [1, 0, 0, 1])
marker = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1,
                 baseVisualShapeIndex=sphereVisualId, basePosition=[0, 0, 2],
                 useMaximalCoordinates=False)
'-------------------------------------'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    description='Command line Utility for training RL models')
    # the hyphen makes the argument optional
    parser.add_argument('-n_episode', type=int, default=500,
                        help='Number of episode to play')
    parser.add_argument('-max_step', type=int, default=10000,
                        help='Number of steps per episode')
    parser.add_argument('-actor_lr', type=float, default=0.0001,
                        help='Learning rate for actor network')
    parser.add_argument('-critic_lr', type=float, default=0.001,
                        help='Learning rate for critic network')
    parser.add_argument('-noise', type=float, default=0.2,
            help='value of OU action noise to ensure exploration')
    parser.add_argument('-gamma', type=float, default=0.99,
                        help='Discount factor for update equation.')
    parser.add_argument('-tau', type=float, default=0.1,
                        help='factor to update target network')
    # parser.add_argument('-env', type=str, default='LunarLander-v2',
    #                                     help='OpenAI gym environment for agent')
    parser.add_argument('-max_size', type=int, default=3000,
                                help='Maximum size for memory replay buffer')
    # parser.add_argument('-dims', type=int, default=8,
    #                         help='Input dimensions; matches env observation, \
    #                               must be list or tuple')
    parser.add_argument('-batch_size', type=int, default=32,
                            help='Batch size for replay memory sampling')
    # parser.add_argument('-n_actions', type=int, default=4,
    #                         help='Number of actions in discrete action space')
    args = parser.parse_args()

    env = Quadcopter(drone=drone)

    agent = Agent(actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                 tau = args.tau, noise=args.noise, batch_size=args.batch_size,
                 gamma = args.gamma, max_size= args.max_size)

    history, scores = [], []
    for i in range(1,args.n_episode):
        state = env.reset()
        agent.noise.reset()
        score = 0
        for t in range(args.max_step):
            action = agent.get_action(state)
            new_state, reward, done, info = env.step(action[0])
            score += reward
            agent.store_transition(state, action,
                                   reward, new_state, done)
            state = new_state
            agent.learn()
            if done:
                break

        history.append(args.tau)
        if done and reward>90:
            scores.append(1000)
        else:
            scores.append(min(500,t))
            
        print('episode: ', i,'t: ', t,'score: ', score)

    x = [i for i in range(1,args.n_episode)]
    # filename should reflect whatever it is you are varying to tune your
    # agent. For simplicity I'm just showing alpha and gamma, but it can be
    # the epsilons as well. You can even include parameters for the fully
    # connected layers and use them as part of the file name.
    filename = 'pybullet/ddpg/ddpg' + '_tau' + str(args.tau) + '_noise' + str(args.noise)+ \
              '.png'
    plotLearning(x, scores, history, filename)
