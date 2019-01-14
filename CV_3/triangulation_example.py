from pylab import *
from matplotlib.tri import Triangulation
from numpy import random

x,y = random.standard_normal((2,200))
tris = Triangulation(x,y).triangles

for t in tris:
    t_ext = [t[0], t[1], t[2], t[0]]  # add first point to end
    plot(x[t_ext],y[t_ext],'r')

plot(x,y,'*g')
axis('off')
show()