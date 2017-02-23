
import os

import gym
import numpy as np



ENV_NAME = 'FrozenLake8x8-v0'
gym.undo_logger_setup()

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)

dir_path = os.path.dirname(os.path.realpath(__file__))

print("Observation space length: {}".format(env.observation_space.n))
print("Action space length: {}".format(env.action_space.n))

# Initialize table with all zeros
Q = np.ones([env.observation_space.n, env.action_space.n]) * 0
# Q = np.random.rand(env.observation_space.n, env.action_space.n)

# Set learning parameters
lr = .1
gamma = .80
num_episodes = 10000

# Create lists to contain total rewards and steps per episode
tList = []
rList = []

for i in range(num_episodes):

    # Reset environment and get first new observation
    state = env.reset()
    # env.render()
    rAll = 0
    t = 0

    # The Q-learning algorithm
    while t < 100000:
        t += 1
        #print("State:", state)

        #print(Q[state, :])

        # Choose an action by greedily (with noise) picking from Q table
        action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))

        #print("Took action {}".format(action))
        # input()

        # Get new state and reward from environment
        state1, reward, done, info = env.step(action)
        #print("Reward: {}".format(reward))

        # Update Q-Table with new knowledge
        #print("Will update with value {}".format(lr * (reward + gamma * np.max(Q[state1, :]) - Q[state, action])))
        Q[state, action] = Q[state, action] + lr * (reward + gamma * np.max(Q[state1, :]) - Q[state, action])
        rAll += reward
        state = state1

        if done:
            break

    tList.append(t)
    rList.append(rAll)
    # print("End of Episode", i, "\n")

print("Final Q table:")
print(Q)
print("Score over time: " + str(sum(rList)/num_episodes))
