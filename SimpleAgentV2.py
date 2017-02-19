
import os

import gym
import numpy as np


# Define environment
ENV_NAME = 'FrozenLake8x8-v0'
gym.undo_logger_setup()
games_to_play = 1000

dir_path = os.path.dirname(os.path.realpath(__file__))

Q_inited = False

# Set learning parameters
lr = .9
gamma = .6
num_episodes = 0  # number of games played during training
e = 1

done = False

# Create lists to contain total rewards and steps per episode
tList = [] # time steps in a game
rList = [] # rewards at each time step
games_won = 0 # games that agent has won
for i in range(games_to_play):
    num_episodes += 1

    # Get the environment and extract the number of actions.
    env = gym.make(ENV_NAME)
    env.seed(123)

    # Initialize table with all zeros Q not already defined
    if not Q_inited:
        print("Observation space length: {}".format(env.observation_space.n)) # documents the results of the agent's action
        print("Action space length: {}".format(env.action_space.n)) # lists the possible values agent must choose from
        Q = np.ones([env.observation_space.n, env.action_space.n]) * 0.1 # get all zeros initial Q matrix
        Q_inited = True # once initialised in a game Q must not be reset in that game

    # Reset environment and get first new observation
    state = env.reset()
    rAll = 0 # cumulative rewards
    t = 0 # time steps in a game
    epsilon = e
    # The Q-learning algorithm
    while t < 100:

        t += 1
        #print("State:", state)

        #print(Q[state, :])

        # Choose an action by greedily (with noise) picking from Q table

        # print(Q_temp)
        if (np.random.rand() < epsilon):
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])  # select optimal action


        #print("Took action {}".format(action))
        # input()

        # Get new state and reward from environment
        state1, reward, done, info = env.step(action)
        # env.render()

        #print("Reward: {}".format(reward))

        # Update Q-Table with new knowledge
        #print("Will update with value {}".format(lr * (reward + gamma * np.max(Q[state1, :]) - Q[state, action])))
        reward = reward - reward * t / 100
        Q[state, action] = Q[state, action] + lr * (reward + gamma * np.max(Q[state1, :]) - Q[state, action])

        rAll += reward
        if done:
            # if the agent has not won yet penalise that state by inserting -1 in the Q matrix
            if state1 != 63:
                Q[state, action] = 0
                print(Q[state, :])
            else:
                print("Solved it in {} steps".format(t))
                games_won += 1
            print("-----------------------------------------------------------------------")
            break
        state = state1
        if epsilon > 0:
            epsilon -= 1 / 100
    if e > .01:
        e -= 1 / games_to_play

    tList.append(num_episodes)
    rList.append(rAll)
    # print("End of Episode", i, "\n")

print("Final Q table:")
print(Q, "\n")
print("Final epsilon:", e)
print("Score over time:", str(sum(rList) / num_episodes))
print("Last score:", str(rList[-1]))
print("Percentage of games won:", games_won / num_episodes)

# Test the algorithm
# Reset environment
state = env.reset()
t = 0 # time
while t < 100:
    t += 1
    action = np.argmax(Q[state, :])  # select optimal action
    state1, reward, done, info = env.step(action)
    reward = reward - reward * t / 100
    Q[state, action] = Q[state, action] + lr * (reward + gamma * np.max(Q[state1, :]) - Q[state, action])
    env.render()
    if done:
        print("Solved in {} steps".format(t))
        break

