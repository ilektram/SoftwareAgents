import numpy as np
import gym
import tempfile

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers.core import Reshape
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD

from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy




ENV_NAME = 'FrozenLake8x8-v0'
gym.undo_logger_setup()


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)

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

# Build all necessary models: V, mu, and L networks.
model = Sequential()
model.add(Embedding(in_shape, nb_actions, input_length=1))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
model.add(Reshape((nb_actions,)))
print("Compiling the following model:")
print(model.summary(), "\n")

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=50,
               target_model_update=1e-2, policy=policy)
dqn.compile(SGD(lr=.1))

try:
    dqn.load_weights('dqn_{}_weights.h5f'.format(ENV_NAME))
    print("\n Loaded existing weights.\n")
except Exception as e:
    print(e)
    pass

temp_folder = tempfile.mkdtemp()
# env.monitor.start(temp_folder)
# env = gym.wrappers.Monitor(env, temp_folder)

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=1e5, visualize=True, verbose=2, log_interval=10000)

# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
print("Testing for 5 episodes:")
dqn.test(env, nb_episodes=5, visualize=True)

# env.monitor.close()

upload = input("Upload? (y/n)")
if upload == "y":
    gym.upload(temp_folder, api_key='YOUR_OWN_KEY')
print("DONE")
