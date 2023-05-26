import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation




# ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
                    # init_func=init, blit=True)
# plt.show()

class RiboVisualizer: 
    """
    TODO Visualizer for correlations/attention coeffiecients
    https://matplotlib.org/stable/tutorials/advanced/blitting.html
    """

    def __init__(self, config, initial_positions: np.array):
        self.fig, self.ax = plt.subplots()
        self.fig.patch.set_facecolor('black')
        self.ax.set_facecolor('black')
        self.ax.set_aspect('equal', 'box')
        self.ax.axis('off')

        # Generate circular RNA coordinates
        self.rna_length = 100  # You can set this to the length of your circular RNA
        theta = np.linspace(0, 2 * np.pi, self.rna_length)
        x = np.cos(theta)
        y = np.sin(theta)

        # Plot circular RNA
        self.ln_rna = self.ax.plot(x, y, 'y-', linewidth=2, alpha=0.5)

        # self.ln, = plt.plot(self.xdata, self.ydata, 'ro', animated=True)
        # self.ln, = plt.plot(self.xdata, self.ydata, 'ro', animated=True)
        # self.ln, = plt.plot(self.xdata, self.ydata, animated=True)
        
        self.ribo_lns = []
        for ribosome_idx in range(initial_positions.shape[0]):
            self.ribo_lns.append(self.ax.plot(
                np.cos(2 * np.pi * initial_positions[ribosome_idx] / self.rna_length),
                np.sin(2 * np.pi * initial_positions[ribosome_idx] / self.rna_length),
                'ro',
                markersize=15,
                # animated=True
            )[0])

        
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        for ln in self.ribo_lns: 
            self.ax.draw_artist(ln)
        self.fig.canvas.blit(self.fig.bbox)

        self.text = self.ax.annotate(f'Accuracy: -1',
            xycoords='figure points',
            xy=(100, 1), xytext=(10, -10),
            textcoords='offset points', ha='left', va='top',
            animated=True,
            color='white',
            )

        plt.show(block=False)
        plt.pause(0.1)

        # self.ax.set_xlim(0, 2*np.pi)
        # self.ax.set_ylim(-1, 1)


    def update(self, pos: np.array):

        # Clear figure
        self.fig.canvas.restore_region(self.bg)

        self.text.set_text(f'Accuracy: {np.random.random()}')
        for idx, ln in enumerate(self.ribo_lns):
            angle = 2 * np.pi * pos[idx] / self.rna_length
            ribosome_x, ribosome_y = np.cos(angle), np.sin(angle)
            # ln.set_ydata(frame[:, idx])
            ln.set_data(ribosome_x, ribosome_y)
            self.ax.draw_artist(ln)

        # self.fig.canvas.blit(self.fig.bbox)
        # self.fig.canvas.flush_events()
        

        

if __name__ == '__main__':
    initial_positions = np.array([10, 2])
    vis = RiboVisualizer(None, initial_positions)
    time_step_positions = np.array([
        [12, 42],
        [12, 44],
        [12, 46],
        [18, 48],
        [20, 50],
        [22, 52],
        [24, 54],
        [26, 56],
        [28, 56],
        [30, 61],
        [32, 62],
        [34, 64],
    ])
    for t in range(time_step_positions.shape[0]):
        vis.update(time_step_positions[t])
        plt.pause(0.5)
        