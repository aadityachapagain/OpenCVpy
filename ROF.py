# Rudin Osher Fatemi de-noising model
from numpy import *


def denoise(im, U_init, tolerance=0.1, tau=0.125, tv_weight=100):
    '''
    An implementation of the Rudin-Osher-Fatemi (ROF) denoising model
    using the numerical procedure .
    :param im: noisy input image (grayscale)
    :param U_init: initial guess of U
    :param tolerance: tolerance for stop criterion
    :param tau: step length
    :param tv_weight: weight of TV regularizing form
    :return:
    '''

    m,n = im.shape

    # initialize
    U = U_init
    Px = im #   x-component to the duel field
    Py = im #   y-component to the duel field
    error =1

    while(error > tolerance):
        Uold = U

        #gradient of primal variable
        GradUx = roll(U, -1, axis=1)-U #  x-component of U's gradient
        GradUy = roll(U, -1, axis=0)-U #  y-component of U's gradient

        #update the dual variable
        PxNew = Px + (tau / tv_weight) * GradUx
        PyNew = Py + (tau / tv_weight) * GradUy
        NormNew = maximum(1, sqrt(PxNew ** 2 + PyNew ** 2))

        Px = PxNew / NormNew  # update of x-component (dual)
        Py = PyNew / NormNew  # update of y-component (dual)

        #update the primal variable
        RxPx = roll(Px, 1, axis=1)  # right x-translation of x-component
        RyPy = roll(Py, 1, axis=0)  # right y-translation of y-component

        DivP = (Px - RxPx) + (Py - RyPy)  # divergence of the dual field.
        U = im + tv_weight * DivP  # update of the primal variable

        #   update of error
        error = linalg.norm(U-Uold)/sqrt(n*m)

    return U,im-U # denoised image and texture residual
