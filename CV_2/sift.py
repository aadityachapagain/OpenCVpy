from os import system
from PIL import Image
from numpy import loadtxt, savetxt, hstack, arange, cos, sin, pi
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

    cmmd = str("sift " + imagename + " --output " + resultname + " " + params)
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