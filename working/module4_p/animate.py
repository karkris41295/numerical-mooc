from matplotlib import pyplot
from matplotlib import rcParams, cm
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
import matplotlib.animation as animation

skip = int(100) # No. of frames skipped
inter = 50 #No. of milliseconds between frames

f = Uf[::skip,:,:]
fig, ax = pyplot.subplots()
mat = ax.imshow(U, cmap = cm.RdBu)

def animate(data):
    u = data
    mat.set_data(u)
    return mat,
    
anim = animation.FuncAnimation(fig, animate, frames=f, interval=inter)
