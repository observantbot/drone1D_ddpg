import random
import pybullet as p

# Drone Environment

class Quadcopter():
    def __init__(self, drone, marker):
        self.drone = drone
        self.marker = marker
        self.mass = 1.5         # kg
        self.gravity = 9.81     # m/s^2

        '''maximum absolute drone distance from desired point
            and maximum velocity that drone can achieve is 5 meter
            and +-5 m/s respectively'''
        self.obs_max = 5 
        self.z_des = 8.0
        self.z_dot_des = 0.0
        

    # current state--> current position, current velocity in z-direction
    def state(self):

        drone_pos, _ = p.getBasePositionAndOrientation(self.drone)
        # drone_pos stores the position of the drone (x, y, z)
        drone_vel, _ = p.getBaseVelocity(self.drone)
        # drone_vel stores the linear velocity of the 
        # drone (x_dot, y_dot, z_dot)

        # discretization and error representation of state
        state = self.abs_to_error_state(drone_pos[2], drone_vel[2])
        return state


    # reward
    def reward(self, action):
        e_z, e_z_dot = self.state()

        reward = -(10*abs(e_z) + 0.5*abs(e_z_dot) + 0.2*abs(action))

        if self.done() == 1:
            if (abs(e_z)<=0.01 / self.obs_max) and abs(e_z_dot)<=0.01 / self.obs_max:
                reward = 100
                print('----desired point achieved----')      
                
        return reward
    
    
    # whether goal is achieved or not.
    def done(self):
        e_z, e_z_dot = self.state()
        '''
        if linear velocity of drone exceeds +-5 m/s, we will
        saturate it to +-5 m/s because of the physical contraints
        of the drone.

        done=1; episodes terminates when:
          1. if e_z >= 1.5: drone is 5*1.5 m or more apart from desired pos.
          3. if abs(e_z)<=0.01/5 and abs(e_z_dot)<=0.01/5: desired condition
            is achieved.
        '''
        if (abs(e_z)>=1.5 or\
            (abs(e_z)<=0.01 / self.obs_max and\
                 abs(e_z_dot)<=0.01 / self.obs_max)):
            return True
        return False


    #info
    def info(self):
        return {}


    # step
    def step(self, action):
        # action must be a float.
        action_ = (action +1)*self.mass*self.gravity
        p.applyExternalForce(objectUniqueId=self.drone, linkIndex=-1,
                         forceObj=[0, 0 ,action_], posObj=[0,0,0], 
                         flags=p.LINK_FRAME)
        p.stepSimulation()

        state = self.state()

        # If absolute velocity of drone exceeds 5 m/s
        # then saturate it to 5 m/s or -5 m/s
        if abs(state[1]) >= 1:
            if state[1] > 0:
                linvel = [0, 0, 5]
            else:
                linvel = [0, 0, -5]
           
            angvel = [0,0,0]
            p.resetBaseVelocity(self.drone, linvel, 
                            angvel)
                            
        reward = self.reward(action)
        done = self.done()
        info = self.info()
        return state, reward, done, info


    # reset the environment
    def reset(self):
        # initializing quadcopter with random z_position and z_velocity
        droneStartPos, droneStartOrn, droneStartLinVel, droneStartAngVel\
             = self.random_state_generator()
        p.resetBasePositionAndOrientation(self.drone, droneStartPos,
                                          droneStartOrn)
        p.resetBaseVelocity(self.drone, droneStartLinVel, 
                            droneStartAngVel)
        print("\n[--------z_des: %f,    z_init: %f,     v_init: %f ---------]\n\n"
              %(self.z_des, droneStartPos[2], droneStartLinVel[2]))

        # return state
        state  = self.abs_to_error_state(droneStartPos[2], droneStartLinVel[2])
        return state


    # auto state generator
    def random_state_generator(self):

        # initialize drone's position between 3 and 13 m. (8+-5)
        z_init = random.uniform(3, 13)
        
        # initialized it with velocity in between -1 and 1 m/s.
        z_dot_init = random.uniform(-1,1)
        
        StartPos = [0,0,z_init] 
        StartOrn = p.getQuaternionFromEuler([0,0,0])
        StartLinVel = [0,0,z_dot_init]
        StartAngVel = [0,0,0]
        return StartPos, StartOrn, StartLinVel, StartAngVel


    # error representation of the state
    def abs_to_error_state(self, z, z_dot):

        e_z = (z - self.z_des) / self.obs_max
        e_z_dot = (z_dot - self.z_dot_des) / self.obs_max
    
        return [e_z, e_z_dot]





