
# image Derivatives
# image intensity changes over the image is important information
# the intensity changes are generally described with x and y of a graylevel image

from pylab import *
from PIL import Image
from numpy import *
from scipy.ndimage import filters

im = array(Image.open('../data/archi_1.jpg').convert('L'))
# Sobel derivative filters
# imx = zeros(im.shape)
# filters.sobel(im,1,imx)
# imy = zeros(im.shape)
# filters.sobel(im,0,imy)
# magnitude = sqrt(imx**2+imy**2)
#
# figure()
# subplot(221)
# imshow(im)
# subplot(222)
# imshow(imx)
# subplot(223)
# imshow(imy)
# subplot(224)
# imshow(magnitude)
# axis('off')
# show()

sigma = 5 #standard deviation
imx = zeros(im.shape)
filters.gaussian_filter(im, (sigma,sigma), (0,1), imx)
imy = zeros(im.shape)
filters.gaussian_filter(im, (sigma,sigma), (1,0), imy)

figure()
subplot(221)
imshow(im)
subplot(222)
imshow(imx)
subplot(223)
imshow(imy)
subplot(224)
# imshow(magnitude)
axis('off')
show()