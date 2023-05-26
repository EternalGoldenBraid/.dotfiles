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

class ProbabilityVisualizer:
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
        
        self.text = self.ax.annotate(f'Accuracy: -1',
            xycoords='figure points',
            xy=(0, 1), xytext=(10, -10),
            # textcoords='offset points', ha='left', va='top',
            animated=True)

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
        
    # import matplotlib.pyplot as plt
    #import numpy as np
    #
    #x = np.linspace(0, 2 * np.pi, 100)
    #
    #fig, ax = plt.subplots()
    #
    ## animated=True tells matplotlib to only draw the artist when we
    ## explicitly request it
    #(ln,) = ax.plot(x, np.sin(x), animated=True)
    #
    ## make sure the window is raised, but the script keeps going
    #plt.show(block=False)
    #
    ## stop to admire our empty window axes and ensure it is rendered at
    ## least once.
    ##
    ## We need to fully draw the figure at its final size on the screen
    ## before we continue on so that :
    ##  a) we have the correctly sized and drawn background to grab
    ##  b) we have a cached renderer so that ``ax.draw_artist`` works
    ## so we spin the event loop to let the backend process any pending operations
    #plt.pause(0.1)
    #
    ## get copy of entire figure (everything inside fig.bbox) sans animated artist
    #bg = fig.canvas.copy_from_bbox(fig.bbox)
    ## draw the animated artist, this uses a cached renderer
    #ax.draw_artist(ln)
    ## show the result to the screen, this pushes the updated RGBA buffer from the
    ## renderer to the GUI framework so you can see it
    #fig.canvas.blit(fig.bbox)
    #
    #for j in range(100):
    #    # reset the background back in the canvas state, screen unchanged
    #    fig.canvas.restore_region(bg)
    #    # update the artist, neither the canvas state nor the screen have changed
    #    ln.set_ydata(np.sin(x + (j / 100) * np.pi))
    #    # re-render the artist, updating the canvas state, but not the screen
    #    ax.draw_artist(ln)
    #    # copy the image to the GUI state, but screen might not be changed yet
    #    fig.canvas.blit(fig.bbox)
    #    # flush any pending GUI events, re-painting the screen if needed
    #    fig.canvas.flush_events()
    #    # you can put a pause in if you want to slow things down
    #    # plt.pause(.1