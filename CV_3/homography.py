from numpy import vstack, ones

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