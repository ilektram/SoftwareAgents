
import os

import gym
import numpy as np



ENV_NAME = 'FrozenLake8x8-v0'
gym.undo_logger_setup()

dir_path = os.path.dirname(os.path.realpath(__file__))

Q_inited = False

# Set learning parameters
lr = .9
gamma = .80
num_episodes = 100000

done = False

# Create lists to contain total rewards and steps per episode
tList = []
rList = []
i = 0

for i in range(1000):
    # Get the environment and extract the number of actions.
    env = gym.make(ENV_NAME)
    np.random.seed(123)
    env.seed(123)

    # Initialize table with all zeros
    if not Q_inited:
        print("Observation space length: {}".format(env.observation_space.n))
        print("Action space length: {}".format(env.action_space.n))
        Q = np.ones([env.observation_space.n, env.action_space.n]) * 0
        Q_inited = True

    # Reset environment and get first new observation
    state = env.reset()
    env.render()
    rAll = 0
    t = 0

    # The Q-learning algorithm
    while t < 100000:
        t += 1
        #print("State:", state)

        #print(Q[state, :])

        # Choose an action by greedily (with noise) picking from Q table
        # action = np.argmax(Q[state, :] + np.random.rand(1, env.action_space.n) * (1. / (i + 1)))
        Q_temp = Q[state, :] + np.random.rand(1, env.action_space.n) * (1. / (i + 1))
        # print(Q_temp)
        action = np.argmax(Q_temp)


        #print("Took action {}".format(action))
        # input()

        # Get new state and reward from environment
        state1, reward, done, info = env.step(action)
        env.render()

        #print("Reward: {}".format(reward))

        # Update Q-Table with new knowledge
        #print("Will update with value {}".format(lr * (reward + gamma * np.max(Q[state1, :]) - Q[state, action])))
        Q[state, action] = Q[state, action] + lr * (reward + gamma * np.max(Q[state1, :]) - Q[state, action])

        rAll += reward
        if done:
            if state1 != 63:
                print(Q[state, :])
                Q[state, action] = -1.0
            else:
                print("Solved it")
            print("-----------------------------------------------------------------------")
            break

        state = state1

    tList.append(t)
    rList.append(rAll)
    # print("End of Episode", i, "\n")

print("Final Q table:")
print(Q)
print("Score over time: " + str(sum(rList)/num_episodes))