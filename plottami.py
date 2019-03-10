import tensorflow as tf
import numpy as np
from scipy.interpolate import spline
import matplotlib.pyplot as plt
rewards = []
for e in tf.train.summary_iterator("tadaaa/output/events.out.tfevents.1551807896.Alessio-Desktop"):
    for v in e.summary.value:
        if v.tag == "episode_reward":
        	rewards.append(v.simple_value)
x = list(range(4000))
y = np.abs(rewards[:4000])
from scipy.ndimage.filters import gaussian_filter1d

ysmoothed = gaussian_filter1d(y, sigma=50)
plt.xlabel('Episode')
plt.ylabel('Total Episode Reward')
plt.plot(x, ysmoothed)
plt.show()