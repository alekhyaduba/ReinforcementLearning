import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os

RESULT_DIR = 'Results/rewards/'
PLOT_DIR = "Results/plots/"

for file_name in os.listdir(RESULT_DIR):
    file_dir = RESULT_DIR + file_name.title()

    file = open(file_dir, 'rb')
    data = pickle.load(file)
    series_data = pd.Series(data['eval_reward'])
    rolling_rewards = series_data.rolling(25).mean()

    file.close()
    plt.title("Average rewards plot for " + file_name.title())
    plt.xlabel("Episodes")
    plt.ylabel("Total rewards")
    data_y = data['eval_reward']
    data_x = data['iter_num']
    series_data = pd.Series(data_y)
    rolling_rewards = series_data.rolling(25).mean()
    plt.plot(data_x, data_y, color="blue", label='Rewards')
    plt.plot(data_x, rolling_rewards, color="red", label='Average_Rewards')
    plt.legend()
    plt.savefig(PLOT_DIR + file_name.title().strip('.Track') + '.png', dpi=300)
    plt.show()
