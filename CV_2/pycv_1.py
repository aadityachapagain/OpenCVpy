from matplotlib import pyplot as plt
import sift
import numpy as np
from PIL import Image

imname = '../data/archi_1.jpg'
im1 = np.array(Image.open(imname).convert('L'))
sift.process_image(imname,'archi.sift')
l1,d1 = sift.read_features_from_file('archi.sift')
plt.figure()
plt.gray()
sift.plot_features(im1,l1,circle=True)
plt.show()