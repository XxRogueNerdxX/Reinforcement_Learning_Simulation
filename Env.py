import numpy as np
import collections
import math
import time
from simulation import Client


class env_manager():
    def __init__(self, reward_collision = -500, reward_timestep = -5, 
    factor = 100, reward_target = 500,reward_target_collide = 50, repeat = 2,max_collision_len = 4):
        self.Rg = reward_target
        self.Rc = reward_collision
        self.Rt = reward_timestep
        self.Reward_target_collide = reward_target_collide
        self.factor = factor
        self.repeat = repeat
        self.done = False
        self.queue = collections.deque(maxlen=repeat)
        self.action_vel = [0.0,0.6]
        self.action_rotat = [-0.3, 0.0, 0.3] #kinda need to check   
        self.collision_queue  = collections.deque(maxlen=repeat)
        self.sim = Client()
        self.action_space = []
        self.target_collider_counter = 0    
        self.accuracy_limit = 0.6
        for x in self.action_vel:
            for y in self.action_rotat:
                self.action_space.append([x,y])
        self.action_space_n = len(self.action_space)
        self.observation_space = [36]


    def _call_sim(self,action,flag):
        sensor_data, velocity, angular_velocity, target_pos, collision = self.sim.step(action, flag)
        velocity = np.round_(math.sqrt(velocity[0]**2 +  velocity[1]**2), 3)
        speed = []
        speed.append(velocity)
        speed.append(angular_velocity)
        speed = np.array(speed)
        val = [[round(np.hypot(target_pos[0] , target_pos[1]), 3)]]
        state = np.concatenate((sensor_data.reshape(13,1), speed.reshape(2,1), target_pos.reshape(2,1), val))
        return state, collision



    def reset(self):
        self.queue.clear()
        self.done = False 
        state, collision = self._call_sim([0,0], True)
        for _ in range(self.repeat):
            self.queue.append(state)
            self.collision_queue.append(collision)
        return np.array(self.queue).reshape(-1)
       

    def step(self,action):
        action_ = self.action_space[action]
        state, collision = self._call_sim(action_, False)
        self.queue.append(state)
        self.collision_queue.append(collision)
        reward = self._reward()
        return np.array(self.queue).reshape(-1), reward, self.done
     
    def _reward(self):  
        reward = 0.0    
        #target 
        val_1 = math.hypot(self.queue[1][15] , self.queue[1][16]) - math.hypot(self.queue[0][15], self.queue[0][16])

        if (math.hypot(self.queue[0][15], self.queue[0][16]) < self.accuracy_limit):
            reward += self.Rg
            self._academy()
            self.done = True 
        else : 
            reward += self.factor * (math.hypot(self.queue[1][15] , self.queue[1][16]) - math.hypot(self.queue[0][15], self.queue[0][16]))
        #collision
        if(self.collision_queue[0][0]):
            reward += self.Rc
        #target_collider
       # if(self.collision_queue[0][1]):
       #     reward += self.Reward_target_collide 
       #     self.done = True 
        #reward per time step
        reward+= self.Rt

        if(self.collision_queue[1][0] and self.collision_queue[0][0]):
            self.done = True
           # self._academy()


        return round(reward, 5)
    
    def _academy(self):
      
        self.target_collider_counter += 1 
        if self.target_collider_counter < 20: 
            self.accuracy_limit = self.accuracy_limit - 0.025 if self.accuracy_limit > 0.35 else 0.35
            x = np.random.uniform(low = +1.0650e+00, high = +1.8910e+00)
            y = np.random.uniform(low = -3.9050e+00, high = -3.2680e+00)
            self.sim.target_position = [x,y,0.0] 
        else : 
            if self.target_collider_counter == 21:
                self.accuracy_limit = 0.525
            self.sim.position = [+2.6500e+00, -4.3400e+00, +8.0299e-02]
            self.accuracy_limit = self.accuracy_limit -  0.025 if self.accuracy_limit > 0.35 else 0.35
            y  = np.random.uniform(low= -3.5650e+00, high = -3.2180e+00)
            x = np.random.uniform(low=  +3.5360e+00,high =  +4.3470e+00)
            self.sim.target_position = [x,y,0.0]



    def end(self):
        self.sim.end()

'''
env = env_manager()
action_space = [i for i in range(env.action_space_n)]
for i in range(10):
    state = env.reset()
    done = False
    while not done:
        action = 0
        next_state, reward, done = env.step(action)
        print(next_state)
env.end()
'''