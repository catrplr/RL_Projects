import random
import numpy as np
from time import sleep
from IPython.display import clear_output
import gym
import matplotlib.pyplot as plt
env = gym.make("Taxi-v3", render_mode='ansi')
env.reset()  # reset environment to a new, random state
env.render()

print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))

# epochs = 0
# penalties, reward = 0, 0

# frames = []  # for animation

# done = False

# while not done:
#     action = env.action_space.sample()
#     state, reward, done, _, info = env.step(action)

#     if reward == -10:
#         penalties += 1

#     # Put each rendered frame into dict for animation
#     frames.append({
#         'frame': env.render(),
#         'state': state,
#         'action': action,
#         'reward': reward
#     }
#     )

#     epochs += 1


# print("Timesteps taken: {}".format(epochs))
# print("Penalties incurred: {}".format(penalties))


def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)


# print_frames(frames)

q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# For plotting metrics
all_epochs = []
all_penalties = []
episode_ = []
for i in range(1, 100001):
    state = env.reset()[0]

    epochs, penalties, reward, = 0, 0, 0
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(q_table[state])  # Exploit learned values

        next_state, reward, done, _, info = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * \
            (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1

    if i % 100 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")
    episode_.append(i)
    all_epochs.append(epochs)
    all_penalties.append(penalties)
print("Training finished.\n")
plt.plot(episode_, all_penalties)
plt.plot(episode_, all_epochs)
plt.xlabel("Episodes")
plt.legend(["Penalties", "Epochs"])
plt.show()

"""Evaluate agent's performance after Q-learning"""

total_epochs, total_penalties = 0, 0
episodes = 100
episode = []
penalty = []
epoch = []
for i in range(episodes):
    print(f"Episode: {i}")
    state = env.reset()[0]
    epochs, penalties, reward = 0, 0, 0

    done = False
    frames = []  # for animation
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, _, info = env.step(action)
        frames.append({
            'frame': env.render(),
            'state': state,
            'action': action,
            'reward': reward
        })

        if reward == -10:
            penalties += 1

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs
    episode.append(i+1)
    penalty.append(penalties)
    epoch.append(epochs)
    print_frames(frames)
    print(f"Penalty for current episode {penalties}")
    print(f"epochs for current episode {epochs}")
    print("<=============================>")

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")

plt.plot(episode, penalty)
plt.plot(episode, epoch)
plt.xlabel("Episodes")
plt.legend(["Penalties", "Epochs"])
plt.show()
