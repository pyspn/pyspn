import csv
import pdb
import numpy as np
from numpy import genfromtxt
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = genfromtxt('activations/um_fs_uw.csv', delimiter=',')

sz = math.sqrt(len(img))
tmp = np.zeros((sz, sz, 3))

#pdb.set_trace()

for channel in range(3):
    for (i, pix) in enumerate(img[:,channel]):
        x_idx = int(i % sz)
        y_idx = int(i / sz)
        tmp[y_idx][x_idx][channel] = pix

plt.imshow(tmp)
plt.show()

