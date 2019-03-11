import tensorflow as tf
import numpy as np
from scipy.interpolate import spline
import matplotlib.pyplot as plt
plots = {
"avgspeed-wantedspeed":[],
"episode_reward":[],
"mean_angle":[],
"mean_trackpos":[],
}
names = {
"avgspeed-wantedspeed":"|v - v*|",
"episode_reward":"Reward",
"mean_angle":"Mean angle",
"mean_trackpos":"Mean trackPos",
}
import glob
EPISODES = 500
for e in tf.train.summary_iterator(glob.glob("output/*")[0]):
    for v in e.summary.value:
        if v.tag in plots:
        	plots[v.tag].append(v.simple_value)
x = list(range(EPISODES))
for p in plots:
	plots[p] = plots[p][:EPISODES]


from scipy.ndimage.filters import gaussian_filter1d
for p in plots:
	fig = plt.figure()
	ysmoothed = gaussian_filter1d(plots[p], sigma=5)
	ax = fig.add_subplot(111)
	ax.set_xlabel('Episode')
	ax.set_ylabel(names[p])
	ax.plot(x, ysmoothed)
plt.show()