
import os

import gym
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers.core import Reshape
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD, Adam


# Define environment
ENV_NAME = 'FrozenLake8x8-v0'
gym.undo_logger_setup()

dir_path = os.path.dirname(os.path.realpath(__file__))

env = gym.make(ENV_NAME)
Q_inited = False

if "n" in dir(env.action_space):
    nb_actions = env.action_space.n
elif "shape" in dir(env.action_space):
    nb_actions = env.action_space.shape[0]
else:
    raise TypeError("No n or shape in action_space :-O")


if "n" in dir(env.observation_space):
    in_shape = env.observation_space.n
elif "shape" in dir(env.observation_space):
    in_shape = (1,) + env.observation_space.shape
else:
    raise TypeError("No n or shape in observation space :-O")

# Set learning parameters
lr = .5
gamma = .5
num_episodes = 0 # number of games played during training
epsilon = 0.9
sgd = SGD(lr)
adam = Adam(lr)

# Build all necessary models: V, mu, and L networks.
model = Sequential()
model.add(Dense(4, input_dim=in_shape, init='zero'))
model.add(Activation('relu'))
model.add(Dense(4, init='zero'))
model.add(Activation('relu'))
model.add(Dense(nb_actions, init='zero'))
model.add(Activation('softmax'))
print(model.summary())
model.compile(optimizer=adam,
          loss='mse',
          metrics=['accuracy'])

done = False

# Create lists to contain total rewards and steps per episode
tList = [] # time steps in a game
rList = [] # rewards at each time step
games_won = 0 # games that agent has won

for i in range(500):
    num_episodes += 1

    # Get the environment and extract the number of actions.
    #np.random.seed(123)
    env.seed(123)

    # Initialize table with all zeros Q not already defined
    if not Q_inited:
        print("Observation space length: {}".format(env.observation_space.n)) # documents the results of the agent's action
        print("Action space length: {}".format(env.action_space.n)) # lists the possible values agent must choose from
        Q_inited = True # once initialised in a game Q must not be reset in that game

    # Reset environment and get first new observation
    state = env.reset()
    # env.render()
    rAll = 0  # cumulative rewards
    Wall = []
    t = 0  # time steps in a game

    # The Q-learning algorithm
    while t < 1000:
        t += 1
        W = model.get_weights()
        Wall.append(W)

        # We can choose to either look up in the model or perform a random jump according to epsilon
        Q_temp = model.predict(np.identity(in_shape)[state].reshape(1, in_shape), batch_size=1)
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_temp)  # select optimal action

        # print("Took action {}".format(action))
        # input()

        # Get new state and reward from environment
        state1, reward, done, info = env.step(action)
        #reward = reward - reward * t / 1000 # add increased reward as episodes increase but decreased as steps in episode increase

        Q1 = model.predict(np.identity(in_shape)[state1].reshape(1, in_shape), batch_size=1)
        #env.render()

        #print("Reward: {}".format(reward))

        Qtarget = Q_temp
        # if done and state1 != 63:
        #     Qtarget[0, action] = 0
        #     Qtarget[0] = Qtarget[0] / np.sum(Qtarget[0])
        # else:
        #     Qtarget[0, action] = reward
        if done and state1 != 63:
            Qtarget[0, action] = 0

        elif done and state1 == 63:
            Qtarget[0, action] = reward

        else:
            Qtarget[0, action] += gamma * np.max(Q1 - np.mean(Q1))

        Qtarget[0] = Qtarget[0] / np.sum(Qtarget[0])
        model.fit(np.identity(in_shape)[state].reshape(1, in_shape), Qtarget, verbose=0, nb_epoch=10)

        rAll += reward
        if done:
            if state1 == 63:
                games_won += 1
                print("Solved in {} steps".format(t))
            else:
                env.render()
            epsilon /= 1.1
            print("-----------------------------------------------------------------------")
            break
        state = state1
    tList.append(num_episodes)
    rList.append(rAll)
    # print("End of Episode", i, "\n")

print("Score over time:", str(sum(rList) / num_episodes))
print("Last score:", str(rList[-1]))
print("Percentage of games won:", games_won / num_episodes)
