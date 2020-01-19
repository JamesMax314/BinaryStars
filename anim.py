import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

plt.style.use('dark_background')


class twoD():
    def __init__(self, bodies, colours, dim, scale, sampling):
        self.system = bodies
        self.dimension = scale * dim
        self.border = scale * dim / 20
        self.interval = sampling
        self.scale = scale
        self.colours = colours

        lim = self.border + dim * scale
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(-lim, lim), ylim=(-lim, lim))
        self.points = self.ax.scatter([], [], cmap="jet", s=1)

        self.arrB = np.empty([len(self.system), np.shape(np.array(self.system[1].pos))[0],
                              np.shape(np.array(self.system[1].pos))[1]])
        for i in range(len(self.system)):
            self.arrB[i, :, :] = np.array(self.system[i].pos) * scale
        self.frames = int(np.shape(self.arrB)[1] / sampling)

    def init(self):
        self.points.set_offsets([])
        return self.points

    def animate(self, i):
        offsets = np.transpose([self.arrB[:, i*self.interval, 0], self.arrB[:, i*self.interval, 1]])
        self.points.set_offsets(offsets)
        self.points.set_array(self.colours)
        return self.points

    def run(self, name):
        self.anim = animation.FuncAnimation(self.fig, self.animate, init_func=self.init,
                                            frames=int(self.frames))
        plt.xlabel("x/Tm")
        plt.ylabel("y/Tm")
        self.anim.save(name, fps=30, writer='ffmpeg', dpi=300)
