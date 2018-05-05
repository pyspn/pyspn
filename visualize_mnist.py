import csv
import pdb
import numpy as np
from numpy import genfromtxt
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = genfromtxt('mm_2__0_8.csv', delimiter=',')

#sz = math.sqrt(len(img))
#tmp = np.zeros((sz, sz))

#for (i, pix) in enumerate(img):
#    y_idx = i % sz
#    x_idx = i / sz
#    pdb.set_trace()
#    tmp[x_idx][y_idx] = pix
#img = tmp

img = img.clip(min=0, max=1)

plt.imshow(img)
plt.show()

