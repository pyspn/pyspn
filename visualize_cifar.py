import csv
import pdb
import numpy as np
from numpy import genfromtxt
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = genfromtxt('leaves_cifar_oneimg_2_2.csv', delimiter=',')

sz = math.sqrt(len(img))
tmp = np.zeros((sz, sz, 3))

for channel in range(3):
    for (i, pix) in enumerate(img[:,channel]):
        x_idx = i % sz
        y_idx = i / sz
        tmp[y_idx][x_idx][channel] = (pix - 0.5) * 2.7

plt.imshow(tmp)
plt.show()

