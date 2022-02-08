import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam




ENV_NAME = "CartPole-v0"

GAMMA = 0.95
LEARNING_RATE = 0.001 #value of alpha

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0 #epsilon max
EXPLORATION_MIN = 0.1 #epsilon min
EXPLORATION_DECAY = 0.995 #epsilon decay


class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX# start with max exporation

        self.action_space = action_space #save action space
        self.memory = deque(maxlen=MEMORY_SIZE) # memory is a double ended queue which will be used to store the experience
        # initialize the model
        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu")) #observation space forms the first layer with RELU activation and outputs into24 layers
        self.model.add(Dense(24, activation="relu")) #input will be the output of previous layer and output will be another 24 units layer (basically hidden layer) with activation function as relu
        self.model.add(Dense(self.action_space, activation="linear")) # output will be the action space and input will be previous layer, activation is linear.
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))# loss function is Mean square error and optimizer is stochastic gradient descent method with learning rate of given value


    def remember(self, state, action, reward, next_state, done): # appends the values of th s,a,r,s' for later training
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state): #returns an action based for given state as per the value of epsilon
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state) # the zeroth element will be the array of size action space or basically the output of the neural network
        return np.argmax(q_values[0])

    def experience_replay(self):
        '''
        This method will feed the experience of the environment stored so far into the network. This method will only execute once the memory is full.
        done will be true for terminal or false for non terminal.?? (it can be true when the episode ends??)
        :return:
        '''
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE) #gets randomized order of the experience so that there is better learning rather than only sequential learning
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0])) #update rule for q learning
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)# what does fit do exactly??
        # self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


def cartpole():
    env = gym.make(ENV_NAME)
    # score_logger = ScoreLogger(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0
    while True:
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        dqn_solver.exploration_rate *= EXPLORATION_DECAY
        while True:
            step += 1
            env.render()
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            # reward = reward if not terminal else -reward # needs to changed for mountain car
            state_next = np.reshape(state_next, [1, observation_space]) # why reshape
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print ("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                # score_logger.add_score(step, run)
                break
            dqn_solver.experience_replay() # will only execute after the memory is full


if __name__ == "__main__":
    cartpole()