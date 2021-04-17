import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim 
import torch as T 
from Env import env_manager
from random import choices 

class Network(nn.Module):
    def __init__(self, lr, input_dims, n_actions, file_name):
        super(). __init__()
        self.input_dims = input_dims
        self.file_name = file_name
       
        self.fc1 = nn.Linear(*input_dims, 512)
        self.bnorm1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 128)
        self.bnorm2 = nn.LayerNorm(128)
        self.V = nn.Linear(128, 1)
        self.A = nn.Linear(128, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr = lr)
        self.loss  = nn.MSELoss()
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        t = F.relu(self.fc1(state))
        t = self.bnorm1(t)
        t = F.relu(self.fc2(t))
        t = self.bnorm2(t)
        V = self.V(t)
        A = self.A(t)
        return V,A
    
    def save(self, episode, epsilon):
        print('....saving.....')
        save = {'episode': episode, 
                'epsilon' : epsilon, 
                'state_dict' : self.state_dict()}
        T.save(save, self.file_name)
    
    def load(self):
        print('.....Loading....')
        data = T.load(self.file_name)
        self.load_state_dict(data['state_dict'])
        self.eval()
        return data['episode'], data['epsilon']

class ReplayMemory():
    def __init__(self,capacity,input_dim, alpha = 0.4):
        self.capacity = capacity
        self.input_dim = input_dim 
        self.alpha = alpha 
        self.states = np.zeros((self.capacity, *self.input_dim), dtype = np.float32)
        self.actions = np.zeros((self.capacity), dtype = np.int64)
        self.rewards = np.zeros((self.capacity), dtype = np.float32)
        self.next_states = np.zeros((self.capacity, *self.input_dim), dtype = np.float32)
        self.dones = np.zeros((self.capacity), dtype = np.bool)
        self.priorities = np.ones((self.capacity), dtype = np.float32)
        self.push_count = 0
 
    
    def push(self,state,action,reward,next_state,done):
        index = self.push_count % self.capacity 
        max_prios = max(self.priorities) if self.push_count >= 1 else 1
        self.priorities[index] = max_prios
        self.states[index] = state 
        self.actions[index] = action 
        self.rewards[index] = reward 
        self.next_states[index]  = next_state
        self.dones[index] = done

        self.push_count+=1 

        
    def sample(self,batch_size, beta = 0.4): 
        batch = min(self.capacity, self.push_count)
        prios = self.priorities[:batch]
        probs = prios ** self.alpha
        probs /= sum(probs)

        indices = np.random.choice(batch, batch_size, p = probs)
        state = self.states[indices] 
        action = self.actions[indices] 
        reward = self.rewards[indices]
        next_state = self.next_states[indices]
        done = self.dones[indices]
        weights =  (batch * probs[indices]) ** (-beta)
        weights /= weights.max()
        return state, action, reward, next_state, done, weights, indices
    
    def provide_sample(self,batch_size):
        return self.push_count >=  batch_size
    
    

    def upload_priorities(self, indices, priorities):
        self.priorities[indices] = priorities
