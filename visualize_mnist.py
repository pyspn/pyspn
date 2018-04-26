import csv
import pdb
import numpy as np
from numpy import genfromtxt

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = genfromtxt('flatspn_6_2_activation.csv', delimiter=',')

plt.imshow(img)
plt.show()
