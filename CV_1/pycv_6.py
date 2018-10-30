from scipy.ndimage import filters
import ROF
from PIL import Image
from pylab import *

# create synthetic image with noise
# im = zeros((500,500))
# im[100:400,100:400] = 128
# im[200:300,200:300] = 255
# im = im + 30*random.standard_normal((500,500))
im = array(Image.open('../data/archi_1.jpg').convert('L'))
U,T = ROF.denoise(im,im)
G = filters.gaussian_filter(im,10)
# save the result
# import scipy.misc
# scipy.misc.imsave('synth_rof.pdf',U)
# scipy.misc.imsave('synth_gaussian.pdf',G)

figure()
axis('off')

subplot(221)
title('orginal')
imshow(im)
subplot(222)
title('Denoised')
imshow(U)
subplot(223)
title('Noise')
imshow(T)
subplot(224)
title('Gussian Blurred')
imshow(G)

show()