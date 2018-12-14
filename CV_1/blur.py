from scipy.ndimage import filters
from PIL import Image
from pylab import *

im = Image.open('dota_back.jpg')

G = filters.gaussian_filter(im,8)

figure()
imshow(G)
show()

im = Image.fromarray(G)
im.save('dota_blur.jpg')