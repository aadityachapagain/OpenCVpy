from PIL import Image
from pylab import *
from CV_3 import wrap

# example of affine wrap of im1 to im2
im1 = np.array(Image.open('../data/beatles.jpg').convert('L'))
im2 = np.array(Image.open('../data/billboard.jpg').convert('L'))

# set to points
tp = np.array([[264,538,540,264],[40,36,605,605],[1,1,1,1]])

im3 = wrap.image_in_image(im1,im2,tp)

figure()
gray()
imshow(im3)
axis('equal')
axis('off')
show()