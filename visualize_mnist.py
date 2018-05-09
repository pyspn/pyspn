import csv
import pdb
import numpy as np
from numpy import genfromtxt
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = genfromtxt('leaves_cifar_oneimg_2_2.csv', delimiter=',')

sz = len(img)
img = np.repeat(img, 3).reshape((sz, sz, 3))

#img = img.reshape((sz*sz,))
#tmp = np.zeros((32, 32, 3))
#for (i, pix) in enumerate(img):
#    x_idx = int(i / 32)
#    y_idx = int(i % 32)
#    tmp[y_idx][x_idx][0] = pix
#    tmp[y_idx][x_idx ][1] = pix
#    tmp[y_idx][x_idx][2] = pix
#img=tmp

img += 0.5

plt.imshow(img)
plt.show()

