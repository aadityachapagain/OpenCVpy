
#
# ###################################################################### #
# !!!!!!!!!!!!!!!!!!!!!!   EXPERIMENTAL   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
# ###################################################################### #
#

import numpy as np
from pylab import *
from os import listdir, getcwd ,path
import ROF
from PIL import Image

image = 'rome.jpg'
loc = '../data/'


def denoise():
    im= np.array(Image.open(loc+image))

    # split the three piexel color scheme to three variable
    r_pixels, g_pixels, b_pixels = np.dsplit(im, 3)
    print(r_pixels.dtype)

    r_pixels, g_pixels, b_pixels = r_pixels[:,:,0], g_pixels[:,:,0], b_pixels[:,:,0]

    # experimentation with np array
    # print(r_pixels.shape)
    # print(r_pixels.shape, r_pixels[:,:,0].shape, np.transpose(r_pixels).shape, np.transpose(np.transpose(r_pixels)[0]).shape)
    # print(np.array([r_pixels[:,:,0]]).shape, np.array([np.transpose(r_pixels[:,:,0])]).shape, np.transpose(np.array([np.transpose(r_pixels[:,:,0])])).shape)
    # # print(np.transpose(np.array([np.transpose(r_pixels[:,:,0])])) == r_pixels)

    r_pixels_U, r_pixels_T = ROF.denoise(r_pixels,r_pixels)
    g_pixels_U, g_pixels_T = ROF.denoise(g_pixels, g_pixels)
    b_pixels_U, b_pixels_T = ROF.denoise(b_pixels, b_pixels)

    # array inspection

    # print(r_pixels_U, r_pixels_U.dtype)
    # print(r_pixels_U.astype(np.uint8),r_pixels_U.dtype)
    #

    # print(r_pixels_U.dtype,r_pixels_U.shape)

    # re calibrate the denoised and noise pixels of each color scheme into fixed standered shape of image array
    r_pixels_U, r_pixels_T = np.transpose(np.array([np.transpose(r_pixels_U)])), np.transpose(np.array([np.transpose(r_pixels_T)]))
    g_pixels_U, g_pixels_T = np.transpose(np.array([np.transpose(g_pixels_U)])), np.transpose(np.array([np.transpose(g_pixels_T)]))
    b_pixels_U, b_pixels_T = np.transpose(np.array([np.transpose(b_pixels_U)])), np.transpose(np.array([np.transpose(b_pixels_T)]))

    # produce correct type of array
    r_pixels_U, r_pixels_T = r_pixels_U.astype(np.uint8), r_pixels_T.astype(np.uint8)
    g_pixels_U, g_pixels_T = g_pixels_U.astype(np.uint8), g_pixels_T.astype(np.uint8)
    b_pixels_U, b_pixels_T = b_pixels_U.astype(np.uint8), b_pixels_T.astype(np.uint8)

    denoised = np.dstack((r_pixels_U, g_pixels_U, b_pixels_U))
    noised = np.dstack((r_pixels_T, g_pixels_T, b_pixels_T))
    #
    # print(denoised.dtype,noised.dtype)

    # display pic in matplotlib
    figure()
    axis('off')

    subplot(221)
    title('Original PIC')
    imshow(Image.fromarray(im))

    subplot(222)
    title('Noise')
    imshow(Image.fromarray(noised))

    subplot(223)
    title('De-Noise')
    imshow(Image.fromarray(denoised))

    show()

if __name__ == '__main__':
    denoise()