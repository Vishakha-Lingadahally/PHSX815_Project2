#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 00:12:42 2021

@author: vishakha

"""

from itertools import cycle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# Define parameters for the walk
dims = 2
step_n = 10000
step_set = [-1, 0, 1]
origin = np.zeros((1,dims))
# Simulate steps in 2D
step_shape = (step_n,dims)
steps = np.random.choice(a=step_set, size=step_shape)
path = np.concatenate([origin, steps]).cumsum(0)
start = path[:1]
stop = path[-1:]
# Plot the path
fig = plt.figure(figsize=(8,8),dpi=250)
ax = fig.add_subplot(111)

ax.scatter(path[:,0], path[:,1],c='blue',alpha=1,s=0.5);
ax.plot(path[:,0], path[:,1],c='blue',alpha=1,lw=0.5,ls= '-');
ax.plot(start[:,0], start[:,1],c='red', marker='+')
ax.plot(stop[:,0], stop[:,1],c='black', marker='o')
plt.title('2D Random Walk')
plt.xlabel("X-axis")
plt.ylabel("Y-label")
plt.tight_layout(pad=0)
plt.savefig('/home/vishakha/Documents/python/random_walk_2d.png',dpi=250);
plt.show()