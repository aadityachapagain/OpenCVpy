from numpy import *
from scipy import ndimage

def normalize(points):
        """

        :param points:collection of points in the homogenous collection
        :return: collection of pints so that last row = 1
        """

        for row in points:
            row /= points[-1]
        return points


def make_homog(points):
    """
    convert the set of points to the homogenous coordinates
    """

    return vstack((points, ones((1,points.shape[1]))))


def H_from_points(fp,tp):
    """
    Find homography H, such that fp is mapped to tp using the linear
    DLT method. Points are conditioned automatically.
    :param fp:
    :param tp:
    :return:
    """

    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')

    # condition points (important for numerical reasons)
    # -- from points--
    m = mean(fp[:2],axis=1)
    maxstd = max(std(fp[:2],axis=1)) + 1e-9
    C1 = diag([1/maxstd,1/maxstd,1])
    C1[0][2] = -m[0]/maxstd
    C1[1][2] = -m[1]/maxstd
    fp = dot(C1,fp)

    # --to points--
    m = mean(tp[:2], axis=1)
    maxstd = max(std(tp[:2], axis=1)) + 1e-9
    C2 = diag([1 / maxstd, 1 / maxstd, 1])
    C2[0][2] = -m[0] / maxstd
    C2[1][2] = -m[1] / maxstd
    tp = dot(C2, tp)

    # create matrix for linear method, 2 rows for each correspondence pair
    nbr_correspondences = fp.shape[1]
    nbr_correspondences = fp.shape[1]
    A = zeros((2 * nbr_correspondences, 9))
    for i in range(nbr_correspondences):
        A[2 * i] = [-fp[0][i], -fp[1][i], -1, 0, 0, 0, tp[0][i] * fp[0][i], tp[0][i] * fp[1][i], tp[0][i]]
        A[2 * i + 1] = [0, 0, 0, -fp[0][i], -fp[1][i], -1, tp[1][i] * fp[0][i], tp[1][i] * fp[1][i], tp[1][i]]
    U, S, V = linalg.svd(A)
    H = V[8].reshape((3, 3))
    # decondition
    H = dot(linalg.inv(C2), dot(H, C1))
    # normalize and return
    return H / H[2, 2]