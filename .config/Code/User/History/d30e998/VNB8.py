from time import perf_counter as clock
import random
from pathlib import Path
from typing import List, Tuple, Dict
from math import pi, exp, sin, cos
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl

from utils import CircleRNA, Ribosome, DraggableCircle
from audio_utils import AudioPlayerParallel as AudioPlayer


class RiboVisualizer:
    def __init__(self, config, initial_positions: np.array, cmap, base_frequencies: List[int], radius: float = 2.):
        self.fig, self.ax = plt.subplots()
        self.fig.patch.set_facecolor('black')
        self.ax.set_facecolor('black')
        self.ax.set_aspect('equal', 'box')
        self.count = 0
        
        
        self.audio_player = AudioPlayer()


        self.ribosomes = []
        for idx, pos in enumerate(initial_positions):
            # center = np.array([2*radius*idx, 2*radius*sin(radius*idx)])
            center = np.array([20*radius*cos(idx), 20*radius*sin(idx)])
            circle = CircleRNA(ax=self.ax, center=center, cmap=cmap, radius=radius, audio_player=self.audio_player, base_frequency=base_frequencies[idx])
            ribosome = Ribosome(circle=circle, position=pos, radius=radius, audio_player=self.audio_player)
            self.ribosomes.append(ribosome)
            
            # self.ax.add_patch(circle.circle)
            # draggable_circle = DraggableCircle(circle.circle, ribosome)
            
        # self.circs = []
        self.ribo_lns = []

        for rib_idx, ribosome in enumerate(self.ribosomes):
            circle_center = ribosome.circle.center
            # circ = mpl.patches.Circle(circle_center, radius=radius, facecolor=cmap(ribosome.position), edgecolor='black')
            # self.circs.append(self.ax.add_patch(circ))

            ribosome_location = ribosome.get_location()
            ribosome.circle.set_color(cmap(ribosome.position))
            self.ribo_lns.append(self.ax.plot(ribosome_location[0], ribosome_location[1], 'ro', markersize=radius/10)[0])

            print(f"initialized ribosome at position x: {ribosome_location[0]}, y: {ribosome_location[1]}")

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
        plt.pause(0.01)
        
    def draw_lines_between_same_angle_ribosomes(self, ribosomes: List[Ribosome], angle_tolerance: float = 0.01):
        lines = []
        for i in range(len(ribosomes)):
            ribosome1 = ribosomes[i]
            for j in range(i + 1, len(ribosomes)):
                ribosome2 = ribosomes[j]
                # if abs(ribosome1.position - ribosome2.position) <= angle_tolerance:
                if abs(ribosome1.angle - ribosome2.angle) <= angle_tolerance:
                    loc1 = ribosome1.get_location()
                    loc2 = ribosome2.get_location()
                    
                    line_width = 
                    line = self.ax.plot([loc1[0], loc2[0]], [loc1[1], loc2[1]], 'w-', linewidth=1, alpha=0.5)[0]
                    lines.append(line)
                    
                    ribosome1.interact_with(ribosome2)
                    
        return lines

    def update(self, pos: np.array, cmap: mpl.colors.ListedColormap, path: Path,
            fps: int = 30, line_life: int = 3, t_idx: int = 0):
        self.fig.canvas.restore_region(self.bg)
        
        # next_pos: np.array = np.zeros_like(pos)

        self.text.set_text(f'Accuracy: {np.random.random()}')
        for idx, ribosome in enumerate(self.ribosomes):
            ribosome.update_position(pos[idx])

            angle = 2 * pi * ribosome.position
            ribosome_location = ribosome.get_location()
            self.ribo_lns[idx].set_data(ribosome_location[0], ribosome_location[1])

            # self.circs[idx].set_color(cmap(angle % pi)) # TODO make this asynchronus
            ribosome.circle.set_color(cmap(angle % pi))
            self.ax.draw_artist(self.ribo_lns[idx])


        # Remove lines from the plot after line_life timesteps
        if hasattr(self, 'lines_buffer'):
            if len(self.lines_buffer) > line_life:
                old_lines = self.lines_buffer.pop(0)
                for line in old_lines:
                    line.remove()

        # Draw lines between ribosomes with the same angle
        # new_lines = self.draw_lines_between_same_angle_ribosomes(self.ribosomes[mask], angle_tolerance=0.1)
        new_lines = self.draw_lines_between_same_angle_ribosomes(self.ribosomes, angle_tolerance=0.05)
        next_pos = np.array([ribosome.position for ribosome in self.ribosomes])
        if not hasattr(self, 'lines_buffer'):
            self.lines_buffer = []
        self.lines_buffer.append(new_lines)
        
        # # Save frame to disk
        # self.fig.savefig(path/f'{t_idx}.png')

        # plt.pause(1 / fps)
        
        return next_pos
    
    def __del__(self):
        self.audio_player.close()
        # self.audio_player.executor.shutdown(wait=True)
        # self.audio_player.stream.stop_stream()
        # self.audio_player.stream.close()
        # self.audio_player.p.terminate()


if __name__ == '__main__':
    save_path = Path('outputs')
    frames_path = save_path/'frames'
    frames_path.mkdir(parents=True, exist_ok=True)
    video_path = save_path/'video.mp4'

    color = 'twilight'
    cmap = plt.get_cmap(color)
    cmap = cmap(np.arange(cmap.N))
    cmap[:, 0:3] *= 0.5
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(cmap)

    # fps = 120*1
    fps = 20*1
    seconds = 10
    line_life = 1

    # n_circ = 10
    n_circ = 3
    base_frequencies = np.linspace(440, 1100, n_circ)
    initial_positions = np.array([pi/n_circ * i  for i in range(1,n_circ)])
    # initial_positions = np.zeros(n_circ)
    # Random initials
    # initial_positions = np.random.random(n_circ) * 2 * pi
    vis = RiboVisualizer(None, initial_positions, cmap=cmap, radius=7*n_circ, base_frequencies=base_frequencies)

    time_step_positions = np.zeros((fps * seconds, initial_positions.shape[0]))
    time_step_positions[:] = initial_positions
    new_pos = np.zeros_like(initial_positions)
    # k = int(0.333*n_circ)
    # k = n_circ-1
    k = 1
    # input_circs = np.array(random.sample(range(n_circ), k))
    input_circs = np.array(list(range(k)))
    mask = np.ones_like(initial_positions).astype(bool)
    mask[input_circs] = False
    # idxs = np.where(mask)[0]
    
    def func(t, x):
        """
        Generate input signal for the ribosomes.
    
        x: np.array of shape (n_circ,d)
        """
        # assert (x <= 2*pi).all()

        # return 8*t*(x+1) % (2*pi)
        # return (x/10+0.1) % (2*pi)
        # return (np.sin(x/10)+0.1) % (2*pi)
        # return np.sqrt(np.sin(x)*np.cos(x)) % (2*pi)
        # return (t*np.sin(x)*np.cos(x)) % (2*pi)
        # x[0] = sin(t)
        # x = np.sin(t+np.exp(x))
        x = np.sin(t+np.exp(x))
        return x

        # np.sin(t * np.linspace(0, 2*pi, k))
        # np.sin(10*t)
        # 5*(input_circs+1)*t % (2*pi)

    # for t in range(time_step_positions.shape[0]-1):
    # for t in np.linspace(0, seconds, fps*seconds): 
    t = 0.0
    while True:
        t += 0.01

        tick = clock()
        # time_step_positions[t+1, mask] = vis.update(time_step_positions[t], fps=fps, cmap=cmap,
        #                                         t_idx=t_idx, path=frames_path, line_life=line_life, mask=mask)[mask]
        # time_step_positions[t+1, mask] = vis.update(func(t, time_step_positions[t]), fps=fps, cmap=cmap,
        #                                         t_idx=t, path=frames_path, line_life=line_life, mask=mask)[mask]
        # initial_positions[mask] = vis.update(func(t, initial_positions[mask]), fps=fps, cmap=cmap,
        #                                         path=frames_path, line_life=line_life, mask=mask)[mask]
        initial_positions = vis.update(func(t, initial_positions), fps=fps, cmap=cmap,
                                                path=frames_path, line_life=line_life)
        
        duration = clock() - tick
        # plt.pause(1/(fps - duration)) if duration < 1/fps
        if duration >= 1/fps:
            plt.pause(1/(fps - duration))
        else:
            plt.pause(0.0001)
        
        # plt.pause(1)
        
        
    if False:
        # Create video
        import cv2
        import os
        from tqdm import tqdm
        frame_array = []
        files = [f for f in os.listdir(frames_path) if os.path.isfile(frames_path/f)]
        files.sort(key=lambda x: int(x[:-4]))
        for i in tqdm(range(len(files))):
            filename = frames_path/files[i]
            img = cv2.imread(str(filename))
            height, width, layers = img.shape
            size = (width,height)
            frame_array.append(img)
        out = cv2.VideoWriter(str(video_path),cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

