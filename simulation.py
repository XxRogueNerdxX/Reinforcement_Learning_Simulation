import time 
import sys 
import numpy as np 
import math 
from scipy.spatial import distance
from configparser import ConfigParser
cfg = ConfigParser()
cfg.read('config.ini')

try: 
    import sim 
except : 
    print("error")
'''
No need of create when you have def initial
'''

class Client():
    def __init__(self, velocity = 20, angular_velocity = 100, wheel_basis = 0.212
    , wheel_radius = 0.03):
        sim.simxFinish(-1)
        self.clientID = sim.simxStart('127.0.0.1',19997,True,True,5000,5)
        self.reset = True 
        self.velocity = velocity
        self.angular_velocity = angular_velocity
        self.wheel_basis = wheel_basis
        self.total_sensors = 13
        self.wheel_radius = wheel_radius 
        self.position = [+6.5000e-01,-4.4900e+00,+8.0299e-02]
        self.target_position = [+1.0900e+00,-3.5380e+00,+3.8000e-02]                               #[+3.9470e+00,+3.8440e+00,+3.8000e-02]
        self.sensor = np.zeros((self.total_sensors),dtype = np.int32)
        self.flag = False
        self.roomba_path = 'C:/Program Files/CoppeliaRobotics/CoppeliaSimEdu/models/selfmade/Roomba.ttm'

        if self.clientID != -1:
            print("Connected to API")
            sim.simxSynchronous(self.clientID, True)
            sim.simxStartSimulation(self.clientID, sim.simx_opmode_blocking)
        else:
            print("Unable to connect")

        #Loading Model 
        _ = sim.simxLoadModel(self.clientID, self.roomba_path, 0, sim.simx_opmode_blocking)
        self._intial()
        _ = sim.simxSetObjectPosition(self.clientID, self.bot, -1, self.position, sim.simx_opmode_oneshot)
        _ = sim.simxSetObjectPosition(self.clientID, self.target, -1, self.target_position, sim.simx_opmode_oneshot)


    def step(self, action, reset):
        if sys.version_info[0] == 3:
            _ = sim.simxSetObjectPosition(self.clientID, self.target, -1, self.target_position, sim.simx_opmode_oneshot)
            sensor_data = np.zeros((self.total_sensors), dtype= np.float32) 
            vel_reading =np.zeros((2), dtype = np.float32)
            angular_reading = 0 
            collision = np.zeros((2), dtype = np.float32)
            target_location = np.zeros((3), dtype = np.float32)
            target_location = np.round_(np.subtract(np.array(self.position[:2]), np.array(self.target_position[:2])), 3)

            if(reset == False):
                if(self.flag):
                    _,target_location = sim.simxGetObjectPosition(self.clientID, self.target, -1, sim.simx_opmode_buffer)
                    _,bot_location = sim.simxGetObjectPosition(self.clientID, self.bot, -1,sim.simx_opmode_buffer)
                    target_location = np.round_([bot_location[0] - target_location[0] , bot_location[1] - target_location[1]], 3)
                self.flag = True 
                
                speed = (self.velocity * action[0])/100
                turn = (self.angular_velocity * action[1])/100
                l_wheel_vel  = round((speed - self.wheel_basis * turn)/self.wheel_radius, 4)
                r_wheel_vel = round((speed + self.wheel_basis * turn)/self.wheel_radius, 4)
                _ = sim.simxSetJointTargetVelocity(self.clientID, self.left_wheel,l_wheel_vel, sim.simx_opmode_streaming)
                _ = sim.simxSetJointTargetVelocity(self.clientID, self.right_wheel, r_wheel_vel, sim.simx_opmode_streaming)

                #Collision 
                _, collision[0] = sim.simxGetIntegerSignal(self.clientID, "collision_wall", sim.simx_opmode_buffer)
                _, collision[1] = sim.simxGetIntegerSignal(self.clientID, "collision_target", sim.simx_opmode_buffer)
                sensor_data = self._readsensor_continue()
                vel_reading, angular_reading = self._get_velocity_reading_continue()
            else :
                self._create(self.position)
                
    
                #_ = sim.simxSetObjectPosition(self.clientID, self.target, self.bot, self.target_position, sim.simx_opmode_oneshot)
                #target_location = self.position - self.target_position
            sim.simxSynchronousTrigger(self.clientID)

            return sensor_data, vel_reading, angular_reading, target_location, collision
        
        else:
            raw_input('Press <enter> key to step the !')
    
    def end(self):
        sim.simxStopSimulation(self.clientID,sim.simx_opmode_blocking)
        sim.simxFinish(self.clientID)
        
    def _create(self, position):
        _ = sim.simxRemoveModel(self.clientID, self.bot, sim.simx_opmode_blocking)
        error = sim.simxLoadModel(self.clientID, self.roomba_path, 0, sim.simx_opmode_blocking)
        self._intial()
        error = sim.simxSetObjectPosition(self.clientID, self.bot, -1, position, sim.simx_opmode_oneshot)

    def _intial(self):
        #handles 
        #bot_handle 
        _, self.bot = sim.simxGetObjectHandle(self.clientID, "Bot_collider", sim.simx_opmode_blocking)
        _, _ = sim.simxGetObjectPosition(self.clientID, self.bot, -1, sim.simx_opmode_streaming)
        #wheels
        _, self.left_wheel = sim.simxGetObjectHandle(self.clientID, "Left_wheel_joint", sim.simx_opmode_blocking)
        _, self.right_wheel = sim.simxGetObjectHandle(self.clientID, "Right_wheel_joint", sim.simx_opmode_blocking)
        #Wall_collection
        _, self.wall = sim.simxGetCollectionHandle(self.clientID, "wall", sim.simx_opmode_blocking)
        #Target
        _, self.target = sim.simxGetObjectHandle(self.clientID, "target", sim.simx_opmode_blocking)
        _, _ = sim.simxGetObjectPosition(self.clientID, self.target, -1, sim.simx_opmode_streaming)
        #Sensors 
        
        for i in range(self.total_sensors):
            proxy = "proxy_"+str(i)+"_"
            _, self.sensor[i] = sim.simxGetObjectHandle(self.clientID,proxy,sim.simx_opmode_blocking)
            _, _, _, _, _ = sim.simxReadProximitySensor(self.clientID, self.sensor[i], sim.simx_opmode_streaming)
        #Sensor velocity 
        _, _, _ = sim.simxGetObjectVelocity(self.clientID, self.bot, sim.simx_opmode_streaming)
        #Collision Data Stream 
        _, _ = sim.simxGetIntegerSignal(self.clientID, "collision_wall", sim.simx_opmode_streaming)
        _, _ = sim.simxGetIntegerSignal(self.clientID, "collision_target", sim.simx_opmode_streaming)
        self.reset = False
    
    def _get_velocity_reading_continue(self):
        _, vel_reading , angular_reading = sim.simxGetObjectVelocity(self.clientID, self.bot
        , sim.simx_opmode_streaming)
        return np.round_(vel_reading[:2],3), round(angular_reading[2],3)

    def _readsensor_continue(self):
        sensor_bool = [False] * self.total_sensors
        sensor_point = np.zeros((self.total_sensors,3))
        sensor_data = np.ones((self.total_sensors,1)) 
        for i in range(self.total_sensors):
            _, sensor_bool[i], sensor_point[i], _, _ = sim.simxReadProximitySensor(self.clientID, 
            self.sensor[i],sim.simx_opmode_buffer)
            sensor_point[i] = np.round_(sensor_point[i], 3)
            if(sensor_bool[i]): 
                sensor_data[i] = np.round_(math.sqrt(sensor_point[i][0] ** 2 + sensor_point[i][1] ** 2
                +sensor_point[i][2] ** 2), 3)
        return sensor_data
    

'''
cl = Client()
sensor, velocity, angular, target, collision = cl.step([0,0], True)
print(target)
for i in range(100):
    sensor, velocity, angular, target, collision = cl.step([0,0], False)
    print(sensor)
print("done")
cl.end()
'''