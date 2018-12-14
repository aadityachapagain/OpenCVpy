from os import system
from PIL import Image
from numpy import loadtxt, savetxt, hstack, arange, cos, sin, pi,linalg,argsort, zeros, arccos,dot
import numpy as np
from matplotlib import pyplot as plt


# TO compute sift ( Scale-Invarient feature transform ) of image we use already avialable binary
# Download and install VLfeat from  http://www.vlfeat.org/  unzip the dowloaded file and add
# sift.exe inside bin folder  to path variable


def process_image(imagename, resultname, params="--edge-thresh 10 --peak-thresh 5"):
    """ Process the image and save the image in the file ."""

    if imagename[-3:] != 'pgm':
        # create a pgm file
        im = Image.open(imagename).convert('L')
        im.save('tmp.pgm')
        imagename = 'tmp.pgm'

    cmmd = str("sift " + imagename + " −−descriptors=" + resultname + " " + params)
    system(cmmd)
    print ('processed', imagename, 'to', resultname)


def read_features_from_file(filename):
    """ Read feature properties and return in matrix form. """
    f = loadtxt(filename)
    return f[:,:4],f[:,4:] # feature locations, descriptors


def write_features_to_file(filename,locs,desc):
    """ Save feature location and descriptor to file. """
    savetxt(filename,hstack((locs,desc)))


def plot_features(im,locs,circle=False):
    """
    Show image with features.
    :param im: image as array
    :param locs: (row, col, scale, orientation of each feature)
    :param circle:
    :return:
    """

    def draw_circle(c, r):
        t = arange(0, 1.01, .01) * 2 * pi
        x = r * cos(t) + c[0]
        y = r * sin(t) + c[1]
        plt.plot(x, y,'b', linewidth = 2)

    plt.imshow(im)
    if circle:
        for p in locs:
            draw_circle(p[:2], p[2])
    else:
        plt.plot(locs[:, 0], locs[:, 1],'ob')

    plt.axis('off')


def match(desc1,desc2):
    """ For each descriptor in the first image,
    select its match in the second image.
    input: desc1 (descriptors for the first image),
    desc2 (same for second image). """
    desc1 = np.array([d/linalg.norm(d) for d in desc1])
    desc2 = np.array([d/linalg.norm(d) for d in desc2])
    dist_ratio = 0.6
    desc1_size = desc1.shape
    matchscores = zeros((desc1_size[0],1),'int')
    desc2t = desc2.T # precompute matrix transpose
    for i in range(desc1_size[0]):
        dotprods = dot(desc1[i,:],desc2t) # vector of dot products
        dotprods = 0.9999*dotprods

        # inverse cosine and sort, return index for features in second image
        indx = argsort(arccos(dotprods))

        # check if nearest neighbor has angle less than dist_ratio times 2nd

        if arccos(dotprods)[indx[0]] < dist_ratio * arccos(dotprods)[indx[1]]:
            matchscores[i] = int(indx[0])
    return matchscores