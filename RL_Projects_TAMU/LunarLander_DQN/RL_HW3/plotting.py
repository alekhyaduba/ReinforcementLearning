import pickle
import pandas as pd
import matplotlib.pyplot as plt

# file = open('output/result_1648087734.track', 'rb')
# file = open('output/Results_Random_policy.track', 'rb')
file = open('output/result_1648201898_WO_Buffer.track', 'rb')


data = pickle.load(file)
series_data = pd.Series(data['REWARDS'])
rolling_rewards = series_data.rolling(100).mean()

file.close()
plt.xlabel("Episodes")
plt.ylabel("Total rewards")
plt.plot(data['EPISODE'], data['REWARDS'], color="blue", label='Rewards')
plt.plot(data['EPISODE'], rolling_rewards, color="red", label='Average_Rewards')
plt.legend()
# plt.show()
plt.savefig("plots/plt_DQN_WO_Buffer.png", dpi=300)
plt.show()

