import time
import sys
import numpy as np 
import math
from scipy.spatial import distance

try:
    import sim
except:
    print('error')



class Client():
    def __init__(self, limit):
        print ('Program started')
        self.max_vel = 0.03
        sim.simxFinish(-1) # just in case, close all opened connections
        self.clientID=sim.simxStart('127.0.0.1',19997,True,True,5000,5) # Connect to CoppeliaSim

        if self.clientID!=-1:
            print ('Connected to remote API server')
            sim.simxSynchronous(self.clientID,True)
            sim.simxStartSimulation(self.clientID,sim.simx_opmode_blocking)
        else: 

            print('unabl to establish connection')
        
        self._intial()
       
    def step(self):      # Now step a few times:
       
        if sys.version_info[0] == 3:
            self._get_pos()
            _ = sim.simxSetObjectPosition(self.clientID, self.target, -1, self.target_position, sim.simx_opmode_oneshot)

         
        #  error, value = sim.simxGetIntegerSignal(self.clientID, "data", sim.simx_opmode_streaming)
          #  print(value)    
        #    self.set_pos() +8.0299e-02

        else:
            raw_input('Press <enter> key to step the !')
            
        sim.simxSynchronousTrigger(self.clientID)
        
    

    def end(self):
        # stop the simulation:
        sim.simxStopSimulation(self.clientID,sim.simx_opmode_blocking)
    
        # Now close the connection to CoppeliaSim:
        sim.simxFinish(self.clientID)
    
    def _intial(self):
        _, self.target = sim.simxGetObjectHandle(self.clientID, "target", sim.simx_opmode_blocking)
        self._get_pos()
        _ = sim.simxSetObjectPosition(self.clientID, self.target, -1, self.target_position, sim.simx_opmode_oneshot)

    def _get_pos(self):
        x = np.random.uniform(low = +1.0650e+00, high = +1.8910e+00)
        y = np.random.uniform(low = -3.9050e+00, high = -3.2680e+00)
        self.target_position = [x,y,0.0]
        #sensors 
        
        



 '''   
cl = Client(10)
for i in range(500):    
    cl.step()
print('done')'''