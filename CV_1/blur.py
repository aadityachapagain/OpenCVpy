from scipy import ndimage
from PIL import Image
from pylab import *
import numpy as np
from os import listdir, getcwd ,path

files = [name.replace('.jpg','') for name in listdir(path.join(getcwd(),'img'))]


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! EXPERIMENTAL !!!!!!!!!!!!!!!!!!!!!!!!!
# normally gussain_blur in scipy blur  the image  and produce output by stacking each pixels
# over other of R, G, B color scheme and produce mean pixels of each color scheme
# which actually produce  the grayscale like output image  we can acutlly provide RGB blur
# functionality by spliting each pixel color scheme and blurring each color scheme and finally stacking
# it to the same image to produce colorful image
def blur():
    for image in files:
        im = np.array(Image.open('img/'+image+'.jpg'))

        r_pixels, g_pixels, b_pixels = np.dsplit(im, 3)

        ndimage.gaussian_filter(r_pixels, output=r_pixels, sigma=4)
        ndimage.gaussian_filter(g_pixels, output=g_pixels, sigma=4)
        ndimage.gaussian_filter(b_pixels, output=b_pixels, sigma=4)

        G = np.dstack((r_pixels, g_pixels, b_pixels))

        im = Image.fromarray(G)
        figure()
        imshow(im)
        show()
        im.save(path.join('img',image+'_blur.jpg',))


if __name__ == '__main__':
    blur()