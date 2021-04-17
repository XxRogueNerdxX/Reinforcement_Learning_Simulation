from Env import env_manager
from Agent import Agent
import numpy as np
import matplotlib.pyplot as plt 
import configparser 
cfg = configparser.ConfigParser()   
cfg.read('config.ini')

batch_size = cfg.getint('current', 'batch_size')
gamma = cfg.getfloat('current', 'gamma')
eps_start = cfg.getfloat('current', 'epsilon')
eps_end = cfg.getfloat('current', 'eps_min')
eps_decay = cfg.getfloat('current', 'eps_decay')
target_update = cfg.getint('current', 'target_update') #when we update our target network
memory_size = 2 * 10 ** 5
lr = cfg.getfloat('current', 'lr')
num_episodes = cfg.getint('current', 'num_episodes')
file_path =  'C:\\Users\\Rakesh\\Desktop\\DOcs\\Roomba\\Trio_final\\HailMary'
beta = 0.4
def beta_val(beta_number):
    beta_number = beta_number + 0.002 if beta_number < 1 else 1
    return beta_number


env = env_manager()
input_dim = env.observation_space
n_actions = env.action_space_n
agent = Agent(n_actions, eps_start, eps_end, eps_decay, lr, gamma, memory_size, name = file_path, input_dims=input_dim)
#agent.target_net.load_state_dict(file_path +'/3_'
#episode, agent.epsilon = agent.policy_net.load()
#_, _ = agent.target_net.load()
t_reward = []
avg = 0
for episode in range(num_episodes):
    beta = beta_val(beta)
    state = env.reset()
    done = False
    total_rewards = 0 
    while not done:
        
        action = agent.select_action(state) 
        state_, reward, done = env.step(action)
        agent.memory.push(state, action, reward, state_, done)
        state = state_
        total_rewards += reward
        if agent.memory.provide_sample(batch_size):
            agent.learn(batch_size, beta)
        if total_rewards <= -2500:
            done = True
    agent.eps_decay()
    t_reward.append(total_rewards)
    if episode % target_update == 0:
        reward =  np.mean(t_reward[avg:avg+target_update])
        print('episode', episode, 'avg', reward, 'epsilon', agent.epsilon)
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        agent.policy_net.save(episode, agent.epsilon)
        agent.target_net.save(episode, agent.epsilon)
        avg += target_update

    

y = np.arange(len(t_reward))
plt.plot(t_reward, y)
plt.show()
plt.savefig(file_path + 'graph.png')