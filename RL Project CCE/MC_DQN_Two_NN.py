import os
import datetime as dtime
import gym
from keras import models
from keras import layers
from keras.optimizers import Adam
from collections import deque
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt


_MAX_MEMORY = 20000
_MODEL_DIR = "models/"
_OUTPUT_DIR = "output/"
_PLOTS_DIR = "plots/"


def setup():
    if not os.path.exists(_MODEL_DIR):
        os.makedirs(_MODEL_DIR)

    if not os.path.exists(_OUTPUT_DIR):
        os.makedirs(_OUTPUT_DIR)

    if not os.path.exists(_PLOTS_DIR):
        os.makedirs(_PLOTS_DIR)


class Tracker:
    def __init__(self):
        self.dict_values = {
            'EPISODE': [],
            'MAX_POSITION': [],
            'STEPS': [],
            'REWARDS':[]
        }

    def append_data(self, dict_values):
        for key, val in dict_values.items():
            self.dict_values[key].append(val)

    def save_track(self, filename):
        file_path = os.path.join(_OUTPUT_DIR, filename)
        with open(file_path, 'wb') as file_handler:
            pickle.dump(self.dict_values, file_handler)

    def load_track(self, filename):
        file_path = os.path.join(_OUTPUT_DIR, filename)
        with open(file_path, 'rb') as file_handler:
            self.dict_values = pickle.load(file_handler)

    def plots(self, filename):
        plot_path = os.path.join(_PLOTS_DIR, filename)


        x_axis = self.dict_values['EPISODE']
        data1 = self.dict_values['MAX_POSITION']
        data2 = self.dict_values['STEPS']

        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('EPISODE')
        ax1.set_ylabel('MAX_POSITION', color=color)
        ax1.plot(x_axis, data1, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:orange'
        ax2.set_ylabel('STEPS', color=color)
        ax2.plot(x_axis, data2, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.show()

        plt.plot(self.dict_values['EPISODE'],self.dict_values['REWARDS'],color="blue")
        plt.show()


class MountainCarTrain:

    def __init__(self, env, **kwargs):
        # constants
        self.tracker = Tracker()
        self.env = env
        self.gamma = kwargs.get('gamma', 0.99)

        # DQN Model Architecture parameters
        self.learning_rate_alpha = kwargs.get('alpha', 0.001)
        self.hidden_act_func = kwargs.get('hidden_act_func','relu')
        self.output_act_func = kwargs.get('output_act_func','linear')
        self.loss_function = kwargs.get('loss_function','mse')

        # Decaying epsilon to reduce exploration as the training progresses
        self.epsilon = kwargs.get('epsilon',1)
        self.epsilon_decay =kwargs.get('epsilon_decay',0.005)
        self.epsilon_min = kwargs.get('epsilon_min', 0.01)

        self.num_episodes = kwargs.get('num_episodes', 1000)
        self.num_iterations = kwargs.get('num_iterations', 201)
        self.reward_per_ep=0
        self.reward_summary=[]
        self.batch_size = 32

        # memory
        self.memory = deque(maxlen=_MAX_MEMORY)

        # Two networks. One for on the fly training and the other for slow training
        self.train_network = self.create_neural_network()

        # TODO: Explain how is it different from train network
        self.target_network = self.create_neural_network()

        # So, the updates made in the train network are available in the target_network
        self.target_network.set_weights(self.train_network.get_weights())

    def create_neural_network(self):
        """
        Creates the architecture of the Neural Network Model.
        :return:
        """
        model = models.Sequential()
        state_shape = self.env.observation_space.shape

        # Add hidden layers
        model.add(layers.Dense(24, activation=self.hidden_act_func, input_shape=state_shape))
        model.add(layers.Dense(48, activation=self.hidden_act_func))

        # Add output layer whose dimension is same as our action space
        model.add(layers.Dense(self.env.action_space.n, activation=self.output_act_func))
        model.compile(loss=self.loss_function, optimizer=Adam(lr=self.learning_rate_alpha))
        return model

    def get_best_action(self, state):
        """
        train_network is used for prediction.
        Select random action if generated random number is less than epsilon else best action according to the Q values
        :param state:
        :return:
        """

        self.epsilon = max(self.epsilon_min, self.epsilon)

        if np.random.rand(1) < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.train_network.predict(state)[0])

        return action


    def train_from_buffer(self):
        """
        1. Train only if batch size data is available.
        2. Samples from the past experiences randomly.
        3. Get the predictions for current state from train model and for next state from target model.
        4. Find the target value of the Q(s,a) based on reward and maxOf Q(s',a)
        5. Update the train model's Q(s,a) using Q leaning algorithm.
        :return:
        """

        if len(self.memory) < self.batch_size:
            return

        samples = random.sample(self.memory, self.batch_size)

        states = []
        new_states = []
        for sample in samples:
            state, action, reward, new_state, done = sample
            states.append(state)
            new_states.append(new_state)

        # We're reshaping so that we can train from batched experiences
        np_states = np.array(states)
        states = np_states.reshape(self.batch_size, 2)

        np_new_states = np.array(new_states)
        new_states = np_new_states.reshape(self.batch_size, 2)

        targets = self.train_network.predict(states)
        new_state_targets = self.target_network.predict(new_states)

        i = 0
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = targets[i]
            if done:
                target[action] = reward
            else:
                q_future = max(new_state_targets[i])
                target[action] = reward + q_future * self.gamma
            i += 1

        self.train_network.fit(states, targets, epochs=1, verbose=0)

    def play(self, current_state, episode):
        """
        1. Get action from train network
        2. Take the action and store the reward, next state and done info
        3. Update the max position- Only for tracking purpose
        4. Reward shaping done if position is >0.5 then reward+=10
        5. Add to the experience memory
        6. Train the train model
        7. Update current state = next state

        :param current_state:
        :param episode:
        :return:
        """
        reward_sum = 0
        max_position = -99

        iter_to_complete = 0
        for iter_to_complete in range(self.num_iterations):
            best_action = self.get_best_action(current_state)

            # self.env.render()

            next_state, reward, done, _ = self.env.step(best_action)

            next_state = next_state.reshape(1, 2)

            # # Keep track of max position
            if next_state[0][0] > max_position:
                max_position = next_state[0][0]

            # # Adjust reward for task completion
            if next_state[0][0] >= 0.5:
                reward += 10

            self.memory.append([current_state, best_action, reward, next_state, done])

            self.train_from_buffer()

            reward_sum += reward

            current_state = next_state

            if done:
                break

        if iter_to_complete >= 199:
            print("Failed to finish task in epsoide {}".format(episode))
        else:
            print("Success in epsoide {}, used {} iterations!".format(episode, iter_to_complete))
            model_path = os.path.join(_MODEL_DIR, f'trainNetworkInEPS{episode}.h5')
            self.train_network.save(model_path)

        # Sync
        self.target_network.set_weights(self.train_network.get_weights())

        print("now epsilon is {}, the reward is {} maxPosition is {}".format(max(self.epsilon_min, self.epsilon),
                                                                             reward_sum, max_position))
        self.epsilon -= self.epsilon_decay

        dict_values = {
            'EPISODE': episode,
            'MAX_POSITION': max_position,
            'STEPS': iter_to_complete+1,
            'REWARDS':reward_sum
        }
        self.tracker.append_data(dict_values)
        self.reward_summary.append(reward_sum)

    def start(self):
        for eps in range(self.num_episodes):
            current_state = self.env.reset().reshape(1, 2)
            self.play(current_state, eps)

    def save_results(self):
        str_ts = str(int(dtime.datetime.now().timestamp()))
        filename = "result_" + str_ts
        self.tracker.save_track(filename + '.track')
        self.tracker.plots(filename + '.png')

    def calculate_reward_per_episode(self):
        sum=0
        for reward in self.reward_summary:
            sum+=reward
        self.reward_summary=sum/self.num_episodes
        return self.reward_per_ep



def main():
    setup()

    env = gym.make('MountainCar-v0')
    dqn = MountainCarTrain(env=env)
    dqn.start()
    dqn.save_results()
    print(dqn.calculate_reward_per_episode())


if __name__ == '__main__':
    main()
