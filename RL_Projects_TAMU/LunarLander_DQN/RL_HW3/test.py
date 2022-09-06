import tensorflow as tf
import gym
from tensorflow import keras
import numpy as np
from gym.wrappers.monitoring import video_recorder


model = keras.models.load_model("models/trainNetworkInEPS900.h5")
env = gym.make('LunarLander-v2')
vid = video_recorder.VideoRecorder(env, path='video/video.mp4')
model.summary()

total_reward = 0
current_state = env.reset()

while True:
    env.render()
    vid.capture_frame()
    # x = model.predict(current_state)
    input = current_state[np.newaxis, ...]
    action = np.argmax(model.predict(input)[0])
    next_state, reward, done, info = env.step(action)
    total_reward += reward
    current_state = next_state
    if done:
        break
print(total_reward)

env.close()
# show_video()
