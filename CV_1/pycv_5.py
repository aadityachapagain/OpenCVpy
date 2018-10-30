from PIL import Image
from numpy import array , ones
from scipy.ndimage import measurements,morphology
# load image and threshold to make sure it is binary
# morphology art of counting images

im = array(Image.open('../data/archi_1.jpg').convert('L'))
im = 1*(im<128)
labels, nbr_objects = measurements.label(im)
print("Number of objects:", nbr_objects)

# morphology - opening to separate objects better
im_open = morphology.binary_opening(im,ones((9,5)),iterations=2)
labels_open, nbr_objects_open = measurements.label(im_open)
print("Number of objects:", nbr_objects_open)