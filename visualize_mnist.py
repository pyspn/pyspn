import csv
import pdb
import numpy as np
from numpy import genfromtxt

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = genfromtxt('sample2.csv', delimiter=',')

if len(img) == 784:
    tmp = np.zeros((28, 28))
    for (i, pix) in enumerate(img):
        y_idx = i % 28
        x_idx = i / 28
        tmp[x_idx][y_idx] = pix
    img = tmp

img = img.clip(min=0, max=1)

plt.imshow(img)
plt.show()

