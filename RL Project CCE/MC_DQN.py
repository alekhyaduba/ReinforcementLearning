import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.hidden = 100
        self.layer1 = nn.Linear(self.state_space, self.hidden, bias=False)
        self.layer2 = nn.Linear(self.hidden, self.action_space, bias=False)

    def forward(self, x):
        model = nn.Sequential(self.layer1, self.layer2)
        return model(x)


env = gym.make('MountainCar-v0')
env.seed(1)
torch.manual_seed(1)
np.random.seed(1)

steps = 2000
state = env.reset()
epsilon = 0.3
gamma = 0.99
loss_history = []
reward_history = []
episodes = 1000
max_position = -0.4
learning_rate = 0.001
successes = 0
position = []

# initialize policy

policy = Policy()
loss_fn = nn.MSELoss()
optimizer = optim.SGD(policy.parameters(), learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

for episode in range(episodes):
    episode_loss = 0
    episode_reward = 0
    state = env.reset()

    for s in range(steps):
        # env.render()
        # Get first action value function
        Q = policy(Variable(torch.from_numpy(state).type(torch.FloatTensor)))

        # Choose epsilon-greedy action
        if np.random.rand(1) < epsilon:
            action = env.action_space.sample()
        else:
            _, action = torch.max(Q, -1)
            action = action.item()

        # Step forward and receive next state and reward
        state_1, reward, done, _ = env.step(action)

        if state_1[0]:

        # Find max Q for t+1 state
        Q1 = policy(Variable(torch.from_numpy(state_1).type(torch.FloatTensor)))
        maxQ1, _ = torch.max(Q1, -1)

        # Create target Q value for training the policy
        Q_target = Q.clone()
        Q_target = Variable(Q_target.data)
        Q_target[action] = reward + torch.mul(maxQ1.detach(), gamma)

        # Calculate loss
        loss = loss_fn(Q, Q_target)

        # Update policy
        policy.zero_grad()
        loss.backward()
        optimizer.step()

        # Record history
        episode_loss += loss.item()
        episode_reward += reward

        if done:
            if state_1[0] > 0.5:
                epsilon *= 0.99
                successes += 1
                scheduler.step()


        else:
            state = state_1
        print(f"state{s} and reward{reward}")
    print(f" episode: {episode}")

torch.save(policy.state_dict(), "/Users/Alekhya/Documents/Documents/GitHub/HackerRankPractice/RL Project CCE/result.pt")
