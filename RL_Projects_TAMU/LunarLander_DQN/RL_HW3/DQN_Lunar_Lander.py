import os
import datetime as dtime
import gym
from keras import models
from keras import layers
from collections import deque
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

_MAX_MEMORY = 200000
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
            'STEPS': [],
            'REWARDS': []
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

        series_data = pd.Series(self.dict_values['REWARDS'])
        rolling_rewards = series_data.rolling(100).mean()

        plt.xlabel("Episodes")
        plt.ylabel("Total rewards")
        plt.plot(self.dict_values['EPISODE'], self.dict_values['REWARDS'], color="blue")
        plt.plot(self.dict_values['EPISODE'], rolling_rewards, color="red")

        plt.savefig(plot_path, dpi=300)
        plt.show()



class LunarLander:

    def __init__(self, env, **kwargs):
        # constants
        self.tracker = Tracker()
        self.env = env
        self.gamma = kwargs.get('gamma', 0.99)

        # DQN Model Architecture parameters
        self.learning_rate_alpha = kwargs.get('alpha', 0.001)
        self.hidden_act_func = kwargs.get('hidden_act_func', 'relu')
        self.output_act_func = kwargs.get('output_act_func', 'linear')
        self.loss_function = kwargs.get('loss_function', 'mse')

        # Decaying epsilon to reduce exploration as the training progresses
        self.epsilon = kwargs.get('epsilon', 1)
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.005)
        self.epsilon_min = kwargs.get('epsilon_min', 0.01)

        self.num_episodes = kwargs.get('num_episodes', 1000)
        self.num_iterations = kwargs.get('num_iterations', 1000)
        self.train_after_steps = kwargs.get('num_of_steps', 1)
        self.reward_per_ep = 0
        self.reward_summary = []
        self.batch_size = 64

        # memory
        self.memory = deque(maxlen=_MAX_MEMORY)

        # Two networks. One for on the fly training and the other for slow training
        self.train_network = self.create_neural_network()

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
        model.add(layers.Dense(64, activation=self.hidden_act_func, input_shape=state_shape))
        model.add(layers.Dense(64, activation=self.hidden_act_func))
        model.add(layers.Dense(64, activation=self.hidden_act_func))

        # Add output layer whose dimension is same as our action space
        model.add(layers.Dense(self.env.action_space.n, activation=self.output_act_func))
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_alpha)
        model.compile(loss=self.loss_function, optimizer=optimizer)

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
            input = state[np.newaxis, ...]
            action = np.argmax(self.train_network.predict(input)[0])

        return action

    def get_random_action(self):
        action = self.env.action_space.sample()
        return action

    def get_q_value(self, state, network):
        """
        train_network is used for prediction.
        Select random action if generated random number is less than epsilon else best action according to the Q values
        :param network:
        :param state:
        :return:
        """

        input = state[np.newaxis, ...]
        if network == 'train':
            q_val = self.train_network.predict(input)
        elif network == 'target':
            q_val = self.target_network.predict(input)

        return q_val

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

        np_new_states = np.array(new_states)

        targets = self.train_network.predict(np_states)
        new_state_targets = self.target_network.predict(np_new_states)

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

        self.train_network.fit(np_states, targets, epochs=1, batch_size=len(targets), verbose=0)

    def play(self, current_state, episode, isPolicyRandom):
        reward_sum = 0

        iter_to_complete = 0
        for iter_to_complete in range(self.num_iterations):

            if isPolicyRandom:
                '''
                Get random action and append this action to the replay buffer memory with
                the relevant next state observed and the rewards obtained.
                This will be act as behavior policy for training the network
                '''
                random_action = self.get_random_action()
                next_state, reward, done, _ = self.env.step(random_action)
                self.memory.append([current_state, random_action, reward, next_state, done])
            else:
                '''
                if the policy is not random that is use the best action for the given state to train the network
                '''
                best_action = self.get_best_action(current_state)

                next_state, reward, done, _ = self.env.step(best_action)
                self.memory.append([current_state, best_action, reward, next_state, done])

            reward_sum += reward

            current_state = next_state

            if done:
                break

            #train from buffer after given number of steps
            if iter_to_complete % self.train_after_steps == 0:
                self.train_from_buffer()

        #save model
        if episode % 100 == 0 and episode > 0:
            model_path = os.path.join(_MODEL_DIR, f'trainNetworkInEPS{episode}.h5')
            self.train_network.save(model_path)

        #get average rewards
        if episode > 0 and episode % 50 == 0:
            print(f"average rewards {self.get_avg_rewards(last_n_elements=50)}")

        # Sync target and train networks
        self.target_network.set_weights(self.train_network.get_weights())

        #Bunch of print calls to see what is happening
        print(f"Episode: {episode}, iter:{iter_to_complete}")
        print(
            f"now epsilon is {max(self.epsilon_min, self.epsilon)}, the reward for this episode {reward_sum}")

        # Epsilon Decay
        self.epsilon -= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        # Storing the values for later analysis
        dict_values = {
            'EPISODE': episode,
            'STEPS': iter_to_complete + 1,
            'REWARDS': reward_sum
        }
        self.tracker.append_data(dict_values)
        self.reward_summary.append(reward_sum)

    def play_wo_buffer(self, current_state, episode):
        '''
        Play without buffer
        :param current_state:
        :param episode:
        :return:
        '''
        reward_sum = 0

        iter_to_complete = 0
        for iter_to_complete in range(self.num_iterations):
            best_action = self.get_best_action(current_state)

            next_state, reward, done, _ = self.env.step(best_action)

            reward_sum += reward

            # train the network
            target = self.get_q_value(current_state, network='train')
            target_new_state = self.get_q_value(next_state, network='target')

            if done:
                target[0][best_action] = reward
            else:
                q_future = max(target_new_state[0])
                target[0][best_action] = reward + q_future * self.gamma
            current_state = current_state[np.newaxis, ...]
            np_state = np.array(current_state)

            self.train_network.fit(np_state, target, epochs=1, verbose=0)
            if done:
                break

            current_state = next_state

        if episode % 100 == 0 and episode > 0:
            model_path = os.path.join(_MODEL_DIR, f'trainNetworkInEPS{episode}.h5')
            self.train_network.save(model_path)
        if episode > 0 and episode % 50 == 0:
            print(f"average rewards {self.get_avg_rewards(last_n_elements=50)}")
        # Sync
        self.target_network.set_weights(self.train_network.get_weights())
        print(f"Episode: {episode}, iter:{iter_to_complete}")
        print(
            f"now epsilon is {max(self.epsilon_min, self.epsilon)}, the reward for this episode {reward_sum}")
        self.epsilon -= self.epsilon_decay

        dict_values = {
            'EPISODE': episode,
            'STEPS': iter_to_complete + 1,
            'REWARDS': reward_sum
        }
        self.tracker.append_data(dict_values)
        self.reward_summary.append(reward_sum)

    def get_avg_rewards(self, last_n_elements=10):
        if len(self.reward_summary) >= last_n_elements:
            last_n_values = self.reward_summary[-last_n_elements:]
            avg = np.average(last_n_values)
        else:
            avg = np.average(self.reward_summary)
        return avg

    def start(self, buffer=True, isPolicyRandom=False):
        for eps in range(self.num_episodes):
            if buffer:
                self.play(self.env.reset(), eps, isPolicyRandom)
            else:
                self.play_wo_buffer(self.env.reset(), eps)

    def save_results(self):
        str_ts = str(int(dtime.datetime.now().timestamp()))
        filename = "result_" + str_ts
        self.tracker.save_track(filename + '.track')
        self.tracker.plots(filename + '.png')

    def calculate_reward_per_episode(self):
        sum = 0
        for reward in self.reward_summary:
            sum += reward
        self.reward_summary = sum / self.num_episodes
        return self.reward_per_ep


def main():
    setup()

    env = gym.make('LunarLander-v2')
    dqn = LunarLander(env=env)
    # dqn.start() #run this for Q1
    # dqn.start(buffer=False) # run this for Q4
    dqn.start(isPolicyRandom=True) # run this for Bonus question

    dqn.save_results()
    # print(dqn.calculate_reward_per_episode())


if __name__ == '__main__':
    main()
