from PIL import Image
from pylab import *

#read image to array
im = array(Image.open('../data/archi_1.jpg'))
#plot the image
imshow(im)

#some points
x = [400,400,3000,3000]
y = [600,2300,600,2300]

#plot the points with red star markers
plot(x,y,'r*')

#line plot connecting the first two points
plot(x[1:3],y[1:3])

#add title and show the plot
title('plotting: "architectural building" ')

# #if you want a pretier plot add it
# axis('off')

show()
