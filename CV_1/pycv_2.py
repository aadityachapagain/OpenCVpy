#calculating image counters using pylab

from PIL import Image
import matplotlib.pyplot as plt
from pylab import array

#read image to array
im = array(Image.open('../data/archi_1.jpg').convert('L'))

#create a new figure
plt.figure()
#dont use colours
plt.gray()

plt.contour(im,origin='image')
plt.axis('equal')
plt.axis('off')

plt.figure()
plt.hist(im.flatten(),128)
plt.show()