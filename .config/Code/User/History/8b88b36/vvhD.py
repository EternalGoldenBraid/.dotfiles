from math import pi, exp, sin, cos

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl

class RiboVisualizer: 
    """
    TODO Visualizer for awesome science experiments 
    https://matplotlib.org/stable/tutorials/advanced/blitting.html
    
    Takes an optional config object that can be used to configure the visualizer.

    Example usage:
    ```
    initial_positions = np.array([10, 2])
    vis = RiboVisualizer(None, initial_positions)
    time_step_positions = np.array([
        [12, 42],
        [12, 44],
        [12, 46],
        [18, 48],
        [20, 50],
        [22, 52]
    ])
    for t in range(time_step_positions.shape[0]):
        vis.update(time_step_positions[t])
        plt.pause(0.5)
    ```

    TODO
    - Multiple circular RNAs in parallel. 
    - Visualize codons with unique color codes.
    - Add a legend
    - Add a title
    - Add a timer text: In progress as self.text
    """

    def __init__(self, config, initial_positions: np.array, cmap):
        self.fig, self.ax = plt.subplots()
        self.fig.patch.set_facecolor('black')
        self.ax.set_facecolor('black')
        self.ax.set_aspect('equal', 'box')
        
        # Set white axis lines
        # self.ax.spines['bottom'].set_color('white')
        # self.ax.spines['top'].set_color('white')
        # self.ax.spines['left'].set_color('white')
        # self.ax.spines['right'].set_color('white')


        # self.ax.axis('off')


        self.circs = []
        self.ribo_lns = []
        # TODO Why does self.ax.plot return a list?
        for ribosome_idx in range(initial_positions.shape[0]):

            # Generate circular RNA coordinates
            self.rna_length = 100  # You can set this to the length of your circular RNA
            theta = np.linspace(0, 2 * pi, self.rna_length)
            x = np.cos(theta)
            y = np.sin(theta)
            
            # n_circles = initial_positions.shape[0]
            # states = [1+2*i for i in range(n_circles)]
            # ADJ = [[np.abs(states[i] - states[j]) for i in range(j, n_circles, 1)] for j in range(0, n_circles, 1)]
            
            # import matplotlib as mpl
            color = cmap(theta[ribosome_idx])
            # color[-1] = 0.4
            circ_x = 2*ribosome_idx 
            circ_y = 2*ribosome_idx
            circ = mpl.patches.Circle((circ_x, circ_y), radius=1, facecolor=cmap(theta[ribosome_idx]), edgecolor='black')
            self.circs.append(self.ax.add_patch(circ))


            self.ribo_lns.append(self.ax.plot(
                # np.cos(2 * np.pi * initial_positions[ribosome_idx] / self.rna_length),
                # np.sin(2 * np.pi * initial_positions[ribosome_idx] / self.rna_length),
                circ_x + initial_positions[ribosome_idx], circ_y + initial_positions[ribosome_idx],
                'ro',
                markersize=15,
                # animated=True
            )[0])

            # Plot circular RNA
            # self.circs.append(self.ax.plot(2*ribosome_idx + x, y, 'y-', linewidth=2, alpha=0.5))
            
            print(f"initialized ribosome at position x: {x}, y: {y}") 

        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        for ln in self.ribo_lns: 
            self.ax.draw_artist(ln)

        self.fig.canvas.blit(self.fig.bbox)

        self.text = self.ax.annotate(f'Timestep: 0',
            xycoords='figure points',
            xy=(100, 1), xytext=(10, -10),
            textcoords='offset points', ha='left', va='top',
            animated=True,
            color='white',
            )

        plt.show(block=False)
        # plt.show(block=True)
        # plt.pause(5)
        plt.pause(0.01)

        # self.ax.set_xlim(0, 2*np.pi)
        # self.ax.set_ylim(-1, 1)
        

    def update(self, pos: np.array, cmap:mpl.colors.ListedColormap, fps: int = 30):
        """
        This function is called at every time step to update the visualization.
        It takes a numpy array of positions of the ribosomes on the circular RNA.
        And updates the 

        """

        # Clear figure
        self.fig.canvas.restore_region(self.bg)

        self.text.set_text(f'Accuracy: {np.random.random()}')
        for idx, ln in enumerate(self.ribo_lns):
            circ_x = 2*idx
            circ_y = 2*idx

            # angle = 2 * np.pi * pos[idx] / self.rna_length
            angle = 2 * pi * pos[idx]
            ribosome_x, ribosome_y = circ_x+np.cos(angle), circ_y+np.sin(angle)
            # ln.set_ydata(frame[:, idx])
            ln.set_data(ribosome_x, ribosome_y)

            # self.circs[idx][0].set_color(plt.get_cmap('hsv')(angle/(2*np.pi)))

            # self.circs[idx][0].set_color(cmap(angle/(2*pi) % 2*pi))
            self.circs[idx].set_color(cmap(angle/(2*pi) % pi))
            # self.circs[idx].set_color(cmap(angle % 2*pi))
            
            # Set olor within circle based on angle 
            self.ax.draw_artist(ln)
            
        # TODO This blitz update is buggy. Why?
        # self.fig.canvas.blit(self.fig.bbox)
        # self.fig.canvas.flush_events()
        

        

if __name__ == '__main__':
    n_circ = 6
    initial_positions = np.array([ 0.0 for i in range(n_circ)])
    fps = 120
    seconds = 40
    # color = 'gist_rainbow'
    # color = 'hsv'
    # color = 'twilight_shifted'
    color = 'twilight'
    cmap = plt.get_cmap(
        color,
        )
    # Get the colormap colors, multiply them with the factor "a", and create new colormap
    cmap = cmap(np.arange(cmap.N))
    cmap[:,0:3] *= 0.5
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(cmap)
    vis = RiboVisualizer(None, initial_positions, cmap=cmap)

    # time_step_positions = np.array([ [i, i+1/2] for i in np.linspace(0, laps[1], fps*seconds)])
    # time_step_positions = np.array([ [i, i*1/2, i*2/2] for i in np.linspace(
    #     start=0,
    #     stop=10,
    #     num=fps*seconds,
    #     )])
    
    
    time_step_positions = np.empty((fps*seconds, initial_positions.shape[0]))
    for t_idx, t in enumerate(np.linspace(start=0, stop=seconds, num=fps*seconds)):

        for circ_idx in range(initial_positions.shape[0]):
            # time_step_positions[t_idx, circ_idx] = (t + circ_idx/2)*t
            # time_step_positions[t_idx, circ_idx] = exp((t + circ_idx/2)/3)
            # time_step_positions[t_idx, circ_idx] = sin((1*t + circ_idx))
            # time_step_positions[t_idx, circ_idx] = cos(3*(t+ circ_idx)*t) + t/5
            # time_step_positions[t_idx, circ_idx] = (circ_idx/2 + t/5)*(1+t/10)

            time_step_positions[t_idx, circ_idx] = np.log( (sin(t+1) + 1))*circ_idx + circ_idx/2






    for t in range(time_step_positions.shape[0]):
        vis.update(time_step_positions[t], fps=fps, cmap=cmap)
        # plt.pause(0.001)
        plt.pause(1/fps)
        