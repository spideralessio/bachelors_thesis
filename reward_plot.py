import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
wanted = 100

def f(v, theta, trackPos):
	#return (v - (10*np.abs(v-wanted)/wanted))
	return (4*np.cos(theta) - np.abs(trackPos) - np.abs(v-wanted)/wanted)*20
def f1(v, theta, trackPos):
	res =  (np.cos(theta) - np.abs(trackPos) - np.abs(np.sin(theta)) - np.abs(v-wanted)/wanted)*5 + 10
	res[np.cos(theta) < 0] = -50
	return res
ds = [0., 0.5, 1.]

v = np.linspace(0,300,100)
theta = np.linspace(0,2*np.pi,100)

V, T = np.meshgrid(v, theta)


for i, d in enumerate(ds):
	fig = plt.figure()
	Z = f1(V, T, d)
	ax = fig.add_subplot(111, projection='3d')
	ax.set_title("d = %.1f v*=%d" % (d, wanted))
	c = np.random.standard_normal(100)
	ax.set_xlabel('Speed')
	ax.set_ylabel('Angle')
	ax.set_zlabel('Reward')
	ax.contour3D(V, T, Z, 25)
# fig = plt.figure()
# for i, d in enumerate(ds):
# 	Z = f1(V, T, d)
# 	ax = fig.add_subplot(1,len(ds), i+1,projection='3d')
# 	c = np.random.standard_normal(100)
# 	ax.set_xlabel('Speed')
# 	ax.set_ylabel('Angle')
# 	ax.set_zlabel('Reward')
# 	ax.contour3D(V, T, Z, 25)


# ax.view_init(60,35)

plt.show()

# from mpl_toolkits.mplot3d import axes3d
# import matplotlib.pyplot as plt
# from matplotlib import cm
# import numpy as np

# def V(x,y,z):
#      return np.cos(10*x) + np.cos(10*y) + np.cos(10*z) + 2*(x**2 + y**2 + z**2)

# X,Y = np.mgrid[-1:1:100j, -1:1:100j]
# Z_vals = [ -0.5, 0, 0.9 ]
# num_subplots = len( Z_vals)

# fig = plt.figure( figsize=(10,4 ) )
# for i,z in enumerate( Z_vals) :
#     ax = fig.add_subplot(1 , num_subplots , i+1, projection='3d')
#     ax.contour(X, Y, V(X,Y,z) ,cmap=cm.gnuplot)
#     ax.set_title('z = %.2f'%z,fontsize=30)
# fig.savefig('contours.png', facecolor='grey', edgecolor='none')