import numpy as np
import gym
import os
from time import sleep

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pong_inputdim = (1, 80, 80)
steps = 2000
pong_actions = 6
eps = np.finfo(np.float32).eps.item()

def preprocess_image(image):
    image = image[35:195]  
    image = image[::2, ::2, 0] 
    image[image == 144] = 0  
    image[image == 109] = 0  
    image[image != 0] = 1  
    return np.reshape(image, pong_inputdim)

def discount_rewards(r, gamma = 0.99):
    # take 1D float array of rewards and compute discounted reward

    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        if r[t] != 0:
            running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_r -= np.mean(discounted_r)
    discounted_r /= np.std(discounted_r) + eps
    return discounted_r

class Policy(nn.Module):
    """Pytorch CNN implementing a Policy"""
    def __init__(self):
        super(Policy, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(1568, pong_actions)

        self.saved_log_probs = []

    def forward(self, x):
        x = F.relu(self.bn1((self.conv1(x))))
        x = F.relu(self.bn2((self.conv2(x))))
        x = F.relu(self.bn3((self.conv3(x))))
        return F.softmax(self.head(x.view(x.size(0), -1)), dim=1)

    def select_action(self, state):
        state = state.float().unsqueeze(0)
        probs = self(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

def run_episode(env, policy, steps = 2000):
    observation = env.reset()
    curr_state = preprocess_image(observation)
    prev_state = None
    rewards = []
    
    for _ in range(steps):
        env.render()
        sleep(0.02)
        state = curr_state - prev_state if prev_state is not None else np.zeros(pong_inputdim)
        state = torch.tensor(state).to(device)
        action = policy.select_action(state)
        observation, reward, done, info = env.step(action)
        prev_state = curr_state
        curr_state = preprocess_image(observation)
        rewards.append(reward)
        if done:
            break
    return rewards, observation

def train(checkpoint):
    env = gym.make("Pong-v0")
    try:
        policy = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        print("Resumed checkpoint {}".format(checkpoint))
    except:
        policy = Policy()
        print("Created policy network from scratch")
    policy.to(device)
    print("device: {}".format(device))
    optimizer = optim.RMSprop(policy.parameters(), lr=1e-3, alpha = 0.99, eps = 1e-8)
    
    episode = 0
    
    while True:
        rewards, observation = run_episode(env, policy, steps)
        print("Total reward for episode {}: {}".format(episode, np.sum(rewards)))
        drewards = discount_rewards(rewards)
        policy_loss = [-log_prob * reward for log_prob, reward in zip(policy.saved_log_probs, drewards)]
        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()
        del policy.saved_log_probs[:]

        episode += 1
        # Save policy network from time to time
        if episode % 100 == 0:
            torch.save(policy, checkpoint)
        # Save animation (if requested)

if __name__ == "__main__":
    train(checkpoint = '/policygradient.pt')