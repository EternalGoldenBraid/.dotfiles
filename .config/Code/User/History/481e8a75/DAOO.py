import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation




# ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
                    # init_func=init, blit=True)
# plt.show()

class AttentionVisualizer:
    """
    TODO Visualizer for correlations/attention coeffiecients
    https://matplotlib.org/stable/tutorials/advanced/blitting.html
    """

    def __init__(self, config, n_samples=2):
        self.fig, self.ax = plt.subplots()
        self.xdata  = np.arange(n_samples)
        self.ydata = np.stack((np.zeros(n_samples), np.ones(n_samples)), axis=1)
        # self.ln, = plt.plot(self.xdata, self.ydata, 'ro', animated=True)
        # self.ln, = plt.plot(self.xdata, self.ydata, 'ro', animated=True)
        # self.ln, = plt.plot(self.xdata, self.ydata, animated=True)
        self.lns = plt.plot(self.xdata, self.ydata, animated=True)
        plt.show(block=False)
        plt.pause(0.1)
        
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        for ln in self.lns: 
            self.ax.draw_artist(ln)
        self.fig.canvas.blit(self.fig.bbox)
        
        # self.text = self.ax.annotate(f'Accuracy: -1',
        #     xycoords='figure points',
        #     xy=(0, 1), xytext=(10, -10),
        #     # textcoords='offset points', ha='left', va='top',
        #     animated=True)

        # self.ax.set_xlim(0, 2*np.pi)
        # self.ax.set_ylim(-1, 1)

    def update(self, frame: np.array):
        # Clear figure
        self.fig.canvas.restore_region(self.bg)
        # self.xdata = (frames)
        # self.ydata.append(np.sin(frame))
        # self.ln.set_data(self.xdata, frame)
        # self.ln.set_ydata(frame)
        # self.ax.draw_artist(self.ln)
        # self.fig.canvas.blit(self.fig.bbox)
        # self.fig.canvas.flush_events()

        self.text.set_text(f'Accuracy: {np.random.random()}')
        for idx, ln in enumerate(self.lns): 
            ln.set_ydata(frame[:, idx])
            self.ax.draw_artist(ln)

        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()
        
        # DEBUG
        # plt.pause(1)
        
def update_plot(t, ax, positions):
    ax.clear()
    ax.set_title(f"Time step: {t}", color='white', fontsize=20)

    # Generate circular RNA coordinates
    rna_length = 100  # You can set this to the length of your circular RNA
    theta = np.linspace(0, 2 * np.pi, rna_length)
    x = np.cos(theta)
    y = np.sin(theta)

    # Plot circular RNA
    ax.plot(x, y, 'b-', linewidth=2, alpha=0.5)

    # Get ribosome positions at time t
    ribosome_positions = get_positions_at_time_t(positions, t)

    # Plot ribosomes
    for pos in ribosome_positions:
        angle = 2 * np.pi * pos / rna_length
        ribosome_x, ribosome_y = np.cos(angle), np.sin(angle)
        ax.plot(ribosome_x, ribosome_y, 'ro', markersize=8)

    ax.set_aspect('equal', 'box')
    ax.axis('off')
        

if __name__ == '__main__':
    vis = AttentionVisualizer(None, n_samples=100)
    x = np.linspace(0, 2 * np.pi, 100)
    for i in range(1000):
        # sin = 6*np.sin(x + (i / 100) * np.pi).squeeze()
        # cos = 6*np.cos(x + (i / 100) * np.pi).squeeze()
        sin = np.sin(x + (i / 100) * np.pi)
        cos = np.cos(x + (i / 100) * np.pi)
        data = np.stack((sin, cos), axis=1)
        # vis.update(cos)
        # vis.update(sin)
        vis.update(data)
        