import csv
import pdb
import numpy as np
from numpy import genfromtxt

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = genfromtxt('spn_tst_activation.csv', delimiter=',')
img = img.clip(min=0, max=1)

plt.imshow(img)
plt.show()
