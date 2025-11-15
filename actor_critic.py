# https://github.com/pytorch/examples/blob/main/reinforcement_learning/actor_critic.py

import argparse
import gymnasium as gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Cart Pole
parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N', help='random seed (default: 543)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='interval between training status logs (default: 10)')
args = parser.parse_args()

render_mode = "human" if args.render else None
env = gym.make('CartPole-v1', render_mode=render_mode)
env.reset(seed=args.seed)
torch.manual_seed(args.seed)

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class Policy(nn.Module):
    """implements both actor and critic in one model"""
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.action_head = nn.Linear(128, 2) # actor's layer
        self.value_head = nn.Linear(128, 1) # critic's layer
        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
    
    def forward(self, x):
        """forward of both actor and critic"""
        x = F.relu(self.affine1(x))

        # actor: choses action to take from state s_t by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values
    
model = Policy()
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()

def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    m = Categorical(probs) # create a categorical distribution over the list of probabilities of actions
    action = m.sample() # sample an action using the distribution
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.item() # the action to take (left or right)

def finish_episode():
    """Training code: calculate actor and critic loss and performs backprop"""
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    returns = []
    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + args.gamma * R
        returns.insert(0, R)
    
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()
        policy_losses.append(-log_prob * advantage) # actor loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]))) # value loss with smooth l1

    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum() # sum up all the values of policy_losses and value_losses
    # backprop
    loss.backward() 
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]

def main():
    running_reward = 10
    for i_episode in count(1): # infinite episodes
        # reset environment and episode reward
        state, _ = env.reset()
        ep_reward = 0
        # for each episode, only run 9999 steps so that we don't infinite loop while learning
        for t in range(1,10_000):
            action = select_action(state) # selection action from policy
            state, reward, terminated, truncated, _ = env.step(action) # take the action
            model.rewards.append(reward)
            ep_reward += reward
            if terminated or truncated: break

        running_reward = 0.05*ep_reward+(1-0.05)*running_reward # update cumulative reward
        finish_episode() # perform backprop

        # log
        if i_episode % args.log_interval == 0:
            print(f'Episode {i_episode}\tLast reward: {ep_reward:.2f}\tAverage reward: {running_reward:.2f}')

        # check if we have "solved" the cart pole problem
        if running_reward > env.spec.reward_threshold:
            print(f"Solved! Running reward is now {running_reward} and the last episode runs to {t} time steps!")
            break

if __name__ == "__main__":
    main()