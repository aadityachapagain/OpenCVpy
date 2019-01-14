from CV_3 import homography
from scipy import ndimage
import numpy as np
# used to wrap one image inside another image


def image_in_image(im1: np.ndarray, im2: np.ndarray, tp):
    """Put im1 in im2 with an affine transformation such that
    Corners are as close to tp as possible.
    tp are homogenous and counterclockwise from top left"""

    # points to wrap from
    m,n = im1.shape[:2]
    fp = np.array([[0, m, m, 0], [0, 0, n, n], [1, 1, 1, 1]])

    # compute affine transform and apply
    H = homography.Haffine_from_points(tp, fp)

    im1_t = ndimage.affine_transform(im1, H[:2, :2],(H[0, 2], H[1, 2]), im2.shape[:2])
    alpha = (im1_t > 0)
    return (1 - alpha) * im2 + alpha * im1_t


def alpha_for_triangle(points,m,n):
    """ Creates alpha map of size (m,n)
    for a triangle with corners defined by points
    (given in normalized homogeneous coordinates). """
    alpha = np.zeros((m,n))
    for i in range(min(points[0]),max(points[0])):
        for j in range(min(points[1]),max(points[1])):
            x = np.linalg.solve(points,[i,j,1])
            if min(x) > 0: #all coefficients positive
                alpha[i,j] = 1
    return alpha