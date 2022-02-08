import numpy as np
import gym
import random
import pickle

data=open("Qlearning",'ab')


class Q_table:
    def __init__(self, low, high, x_precision, y_precision):
        self.low = low
        self.high = high
        self.x_precision = x_precision
        self.y_precision = y_precision
        self.q_table = self.form_q_table()

    def transform(self, state):
        x, y = state[0], state[1]

        x = int(x * self.x_precision)
        y = int(y * self.y_precision)

        return x, y

    def form_q_table(self):
        x_low, y_low = self.low[0], self.low[1]
        x_high, y_high = self.high[0], self.high[1]

        x_low = int(x_low * self.x_precision)
        x_high = int(x_high * self.x_precision)
        y_low = int(y_low * self.y_precision)
        y_high = int(y_high * self.y_precision)

        dict_mat = {x: {y: [random.uniform(0,1) for i in range(3)] for y in range(y_low, y_high+1)} for x in range(x_low, x_high+1)}
        for y in dict_mat[x_high]:
            for a in range(3):
                dict_mat[x_high][y][a]=0

        return dict_mat

    def choose_action_greedy(self, state):
        x, y = self.transform(state)

        return np.argmax(self.q_table[x][y])

    def value(self, state, action):
        x, y = self.transform(state)

        return self.q_table[x][y][action]

    def maxValue(self, state):
        x, y = self.transform(state)

        return np.max(self.q_table[x][y])

    def setValue(self, state, action, value):
        x, y = self.transform(state)

        self.q_table[x][y][action] = value


env = gym.make("MountainCar-v0")


q_obj = Q_table(env.observation_space.low, env.observation_space.high, 10, 100)

# constants
alpha = 0.1
gamma = 0.9
epsilon = 0.2

all_epochs = []
all_penalties = []

# For plotting metrics
all_epochs = []
# training
for i in range(10000):
    state = env.reset()

    epochs, reward = 0, 0
    done = False
    while not done:
        env.render()
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = q_obj.choose_action_greedy(state)

        new_state, reward, done, info = env.step(action)

        # find next max
        v_next = q_obj.maxValue(state)

        delta = reward + (gamma * v_next) - q_obj.value(state, action)
        delta *= alpha

        val = q_obj.value(state, action) + delta
        q_obj.setValue(state, action, val)

        state = new_state
        epochs += 1
        print(state, reward, done, info)

    print("epoch:", i)
print("Training finished.\n")
pickle.dump(q_obj.q_table,data)


"""Evaluate agent's performance after Q-learning"""

total_epochs, total_penalties = 0, 0
episodes = 100

for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0

    done = False

    while not done:
        env.render()
        action = q_obj.choose_action_greedy(state)
        state, reward, done, info = env.step(action)
        print(state,reward,done,info)

        if reward == -1:
            penalties += 1

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")

