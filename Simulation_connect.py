
# simRemoteApi.start(19999,1300,false,true)


try:
    import sim
except:
    print('unable to connect')
import time
import sys

class Client():
    def __init__(self):
        print ('Program started')
        sim.simxFinish(-1) # just in case, close all opened connections
        self.clientID=sim.simxStart('127.0.0.1',19997,True,True,5000,5) # Connect to CoppeliaSim
        if self.clientID!=-1:
            print ('Connected to remote API server')
            sim.simxSynchronous(self.clientID,True)
            sim.simxStartSimulation(self.clientID,sim.simx_opmode_blocking)

    def step(self):
        if sys.version_info[0] == 3:
            print('enter your code here')
            
        else:
            raw_input('Press <enter> key to step the simulation!')
            
        sim.simxSynchronousTrigger(self.clientID)

    def end(self):
        sim.simxStopSimulation(self.clientID,sim.simx_opmode_blocking)
        sim.simxFinish(self.clientID)

cl = Client()
for i in range(10):
    cl.step()
cl.end()