from PIL import Image
from pylab import *
from CV_3 import wrap, homography
from scipy import ndimage

# example of affine wrap of im1 to im2
im1 = np.array(Image.open('../data/beatles.jpg').convert('L'))
im2 = np.array(Image.open('../data/billboard.jpg').convert('L'))

tp = array([[264,538,540,264],[40,36,605,605],[1,1,1,1]])
# set from points to corners of im1
m,n = im1.shape[:2]
fp = array([[0,m,m,0],[0,0,n,n],[1,1,1,1]])
# first triangle
tp2 = tp[:,:3]
fp2 = fp[:,:3]
# compute H
H = homography.Haffine_from_points(tp2,fp2)
im1_t = ndimage.affine_transform(im1,H[:2,:2],
(H[0,2],H[1,2]),im2.shape[:2])
# alpha for triangle
alpha = wrap.alpha_for_triangle(tp2,im2.shape[0],im2.shape[1])
im3 = (1-alpha)*im2 + alpha*im1_t
# second triangle
tp2 = tp[:,[0,2,3]]
fp2 = fp[:,[0,2,3]]
# compute H
H = homography.Haffine_from_points(tp2,fp2)
im1_t = ndimage.affine_transform(im1,H[:2,:2],
(H[0,2],H[1,2]),im2.shape[:2])
# alpha for triangle
alpha = wrap.alpha_for_triangle(tp2,im2.shape[0],im2.shape[1])
im4 = (1-alpha)*im3 + alpha*im1_t
figure()
gray()
imshow(im4)
axis('equal')
axis('off')
show()

# after seeing result three points affine transformation if more accurate than 4 points affine transformation