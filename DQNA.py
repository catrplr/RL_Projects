from ale_py.roms import SpaceInvaders
import tensorflow as tf
import gym
import keras
import time
import random
import numpy as np
from keras.layers import Dense, Flatten, Conv2D
from keras.models import Sequential
from keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy, LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory


env = gym.make("")
height, width, channels = env.observation_space.shape
actions = env.action_space.n
print(actions)
# episodes = 5
# for episode in range(episodes):
#     state = env.reset()
#     done = False
#     score = 0

#     while not done:
#         action = random.choice([0, 1, 2, 3, 4, 5])
#         n_state, reward, done, info = env.step(action)
#         score += reward
#     print('Episode:{} Score:{}'.format(episode, score))
# env.close()


# def build_model(height, width, channels, actions):
#     model = Sequential()
#     model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu',
#               input_shape=(3, height, width, channels)))
#     model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
#     model.add(Conv2D(64, (3, 3), activation='relu'))
#     model.add(Flatten())
#     model.add(Dense(512, activation='relu'))
#     model.add(Dense(256, activation='relu'))
#     model.add(Dense(actions, activation='linear'))
#     return model


# model = build_model(height, width, channels, actions)
# model.summary()


# def build_agent(model, actions):
#     policy = LinearAnnealedPolicy(EpsGreedyQPolicy(
#     ), attr='eps', value_max=1., value_min=.1, value_test=.2, nb_steps=10000)
#     memory = SequentialMemory(limit=1000, window_length=3)
#     dqn = DQNAgent(model=model, memory=memory, policy=policy,
#                    enable_dueling_network=True, dueling_type='avg',
#                    nb_actions=actions, nb_steps_warmup=1000
#                    )
#     return dqn


# dqn = build_agent(model, actions)
# dqn.compile(Adam(lr=1e-4))
# dqn.fit(env, nb_steps=10000, visualize=False, verbose=2)
# scores = dqn.test(env, nb_episodes=10, visualize=True)
# print(np.mean(scores.history['episode_reward']))
# dqn.save_weights('dqn_weights.h5f')


env = gym.make('CartPole-v1')
states = env.observation_space.shape[0]
actions = env.action_space.n
episodes = 10
for episodes in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = random.choice([0, 1])
        n_state, reward, done, info = env.step(action)
        score += reward
    print('Episodes:{} Score:{}'.format(episodes, score))
env.close()


def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1, states)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


model = build_model(states, actions)
print(model.summary())


def bulid_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn


dqn = bulid_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

scores = dqn.test(env, nb_episodes=100, visualize=False)
print(np.mean(scores.history['episode_reward']))
_ = dqn.test(env, nb_episodes=5, visualize=True)
dqn.save_weights('dqn_weights.h5f', overwrite=True)

# del model
# del dqn
# del env

# env = gym.make('CartPole-v1')
# actions = env.action_space.n
# states = env.observation_space.shape[0]
# model = build_model(states, actions)
# dqn = bulid_agent(model, actions)
# dqn.compile(Adam(lr=1e-3), metrics=['mae'])
# dqn.load_weights('dqn_weights.h5f')
# dqn.test(env, nb_episodes=5, visualize=True)
