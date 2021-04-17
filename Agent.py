from NetWork_Replay import Network, ReplayMemory
import torch as T 
import numpy as np
import random 

class Agent():
    def __init__(self ,n_actions,epsilon, eps_min, eps_dec,lr,gamma,
        memory_size , name , input_dims = [34]):
            self.gamma = gamma 
            self.n_actions = n_actions 
            self.memory_size = memory_size
            self.input_dims = input_dims
            self.epsilon = epsilon 
            self.end =eps_min
            self.decay = eps_dec
            self.memory = ReplayMemory(memory_size, input_dims)
            self.policy_net = Network(lr, input_dims,n_actions, name + "policy_net.pth")
            self.target_net = Network(lr ,input_dims,n_actions, name + "Target_net.pth")
            self.action_space = [i for i in range(n_actions)]
    def select_action(self,state): 
        action = 0
        if np.random.uniform(0,1) < self.epsilon :
            action = np.random.choice(self.action_space)
            
        else : 
            with T.no_grad():
                state = T.tensor(state, dtype = T.float32).to(self.policy_net.device)
                _, action = self.policy_net.forward(state)
                action = self.action_space[T.argmax(action).item()]
          
        return action
      
    def eps_decay(self):
       # return self.end + (self.start - self.end) * math.exp(-1 * self.current_step * self.decay)
        self.epsilon = self.epsilon - self.decay if self.epsilon > self.end else self.end  
     
    def learn(self,batch_size, beta):
                
        
        states, actions, rewards, next_states, dones, weights, indicess= self.memory.sample(batch_size, beta)
        states = T.tensor(states, dtype = T.float32).to(self.policy_net.device)
        action = T.tensor(actions, dtype = T.int64).to(self.policy_net.device)
        reward = T.tensor(rewards, dtype = T.float32).to(self.policy_net.device)
        next_state = T.tensor(next_states, dtype = T.float32).to(self.policy_net.device)
        done = T.tensor(dones, dtype = T.bool).to(self.policy_net.device)
        weight = T.tensor(weights, dtype = T.float32).to(self.policy_net.device)
        
        self.policy_net.optimizer.zero_grad()
        indices = np.arange(batch_size)
        
        V_s , A_s  = self.policy_net.forward(states)
        V_ns, A_ns = self.target_net.forward(next_state)

        q_pred = T.add(V_s, (A_s - A_s.mean(dim =1, keepdim = True)))
        q_pred = q_pred[indices, action]

        q_target = T.add(V_ns, (A_ns - A_ns.mean(dim = 1, keepdim = True))).max(dim = 1)[0]
        q_target[done] = 0.0

        q_target = reward + self.gamma * q_target
        
        loss = (q_target - q_pred) ** 2 * weight 
        prios = loss + 1e-5
        loss = T.mean(loss)
        loss.backward()
        self.memory.upload_priorities(indicess, prios.detach().numpy())
        self.policy_net.optimizer.step()

'''
batch_size = 32
gamma = 0.999
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
target_update = 10 #when we update our target network
memory_size = 100
lr = 0.001
num_episodes = 10000
env = env_manager()
agent = Agent(env.action_space_n, eps_start, eps_end, eps_decay, lr, gamma, memory_size)
for i in range(10):
    state = env.reset()
    done = False
    while not done: 
        action = agent.select_action(state)
        state_,reward, done = env.step(action)
        agent.memory.push(state, action, reward, state_, done)

        if agent.memory.provide_sample(batch_size) == 1:
            agent.learn(batch_size)
            agent.target_net.save()
            agent.policy_net.save()
'''