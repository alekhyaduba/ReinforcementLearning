import gym
import torch
from point_env import PointEnv
from gym.wrappers import Monitor

import pickle
from agent import Agent
import torch
import numpy as np

from gym.wrappers.monitoring import video_recorder


# data = pickle.load(file)
device = torch.device('cuda', index=0) if torch.cuda.is_available() else torch.device('cpu')
tensor = torch.tensor


env = gym.make("Point-v0")
# vid = video_recorder.VideoRecorder(env, path='video/video.mp4')

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
file = open('Results/learned_models/Point-v0_2theta.track', 'rb')
theta = pickle.load(file)
policy_net = None
state = env.reset()
env.render()

def include_bias(x):
    # Add a constant term (1.0) to each entry in x
    return torch.cat([x, torch.ones_like(x[..., :1])], axis=-1)
    #return torch.cat(x,torch.ones(1))
def point_get_action(theta, ob):
    ob_1 = include_bias(ob)
    mean = torch.mm(theta,ob_1.view(-1,1)).view(-1,2).squeeze(0)
    return torch.normal(mean=mean, std=1.)

for i in range(500):
    env.render()
    # vid.capture_frame()

    state_var = tensor(state).unsqueeze(0)

    agent = Agent(env, 'Point-v0', device, policy_net, theta, custom_reward=None,
                      running_state=None, num_threads=1)

    action = point_get_action(theta.T, state_var).numpy()
    action = action.astype(np.float64)
    next_state, reward, done, _ = env.step(action)
    state = next_state
    # if done:
    #     break
# env.render()
