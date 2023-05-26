from time import perf_counter as clock
import random
from pathlib import Path
from typing import List, Tuple, Dict
from math import pi, exp, sin, cos
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
from matplotlib.widgets import Slider, Button, RadioButtons

from utils import CircleRNA, Ribosome, DraggableCircle, generate_circle_positions
from audio_utils import AudioPlayerParallel as AudioPlayer


class RiboVisualizer:
    def __init__(self, config, initial_positions: np.array, layer_nodes: List[int],
                cmap, base_frequencies: List[int], n_circ: int, n_ribo:int, radius: float = 2.):
        self.fig, self.ax = plt.subplots()
        self.fig.patch.set_facecolor('black')
        self.ax.set_facecolor('black')
        self.ax.set_aspect('equal', 'box')
        self.count = 0
        self.n_circ=n_circ
        self.layer_nodes = layer_nodes
        self.audio_player = AudioPlayer()


        self.ribosomes: List[Ribosome] = []
        # for idx, pos in enumerate(initial_positions):
        # TODO Add initial positions to ribosomes.
        self.circles: List[CircleRNA] = []
        circle_positions = generate_circle_positions(self.layer_nodes)
        for idx, position in enumerate(circle_positions):
            circle = CircleRNA(ax=self.ax, center=position, cmap=cmap, radius=radius, n_ribo=n_ribo, 
                               audio_player=self.audio_player, base_frequency=base_frequencies[idx])
            self.ribosomes.append(circle.ribosomes)
            self.circles.append(circle)

        #     # self.ax.add_patch(circle.circle)
        #     # draggable_circle = DraggableCircle(circle.circle, ribosome)
            
        self.ribo_lns = []

        for circle in self.circles:
            ribos_lns_circle = []
            for ribosome in circle.ribosomes:
                ribosome_location = ribosome.get_location()
                ribosome.circle.set_color(cmap(ribosome.position))
                # self.ribo_lns.append(self.ax.plot(ribosome_location[0], ribosome_location[1], 'ro', markersize=radius/10)[0])
                ribos_lns_circle.append(self.ax.plot(ribosome_location[0], ribosome_location[1], 'ro', markersize=radius/5)[0])

                print(f"initialized ribosome at position x: {ribosome_location[0]}, y: {ribosome_location[1]}")
                
            self.ribo_lns.append(ribos_lns_circle)
                
                
                
        # self.distance_treshold = config['distance_threshold']
        self.distance_threshold = 100*radius
        self.adjacency_matrix, self.distance_matrix = self.initialize_adjacency_matrix(self.circles)
        self.distance_threshold_slider_ax = plt.axes([0.25, 0.05, 0.5, 0.03], facecolor='lightgoldenrodyellow')
        self.distance_threshold_slider = Slider(self.distance_threshold_slider_ax, 'Distance Threshold', 
                                                radius*2, self.distance_threshold, valinit=self.distance_threshold)
        self.distance_threshold_slider.on_changed(self.update_distance_threshold)

        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        # for ln in self.ribo_lns:
        for circle_ribo_lns in self.ribo_lns:
            for ln in circle_ribo_lns:
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
        
    def update_distance_threshold(self, val):
        self.distance_threshold = val
        self.update_adjacency_matrix(self.circles)
    
        
    def initialize_adjacency_matrix(self, circles: List[CircleRNA]):
        n = self.n_circ
        adjacency_matrix = np.zeros((n, n), dtype=bool)
        distance_matrix = np.zeros((n, n))

        for i in range(n):
            
            circles[i].circle_idx = i
            for ribosome in circles[i].ribosomes:
                ribosome.circle_idx = i
            for j in range(i+1, n):
                distance = np.linalg.norm(circles[i].center - circles[j].center)
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
                if distance <= self.distance_threshold:
                    adjacency_matrix[i, j] = True
                    adjacency_matrix[j, i] = True

        return adjacency_matrix, distance_matrix
    
    def update_adjacency_matrix(self, circles: List[CircleRNA], 
                                # adjacency_matrix: np.array, distance_matrix: np.array,
                                distance_threshold: float = 10.0):
        self.adjacency_matrix[:] = False
        n = self.n_circ
        for i in range(n):
            for j in range(i+1, n):
                distance = np.linalg.norm(circles[i].center - circles[j].center)
                self.distance_matrix[i, j] = distance
                self.distance_matrix[j, i] = distance
                # if distance <= distance_threshold:
                if distance <= self.distance_threshold:
                    self.adjacency_matrix[i, j] = True
                    self.adjacency_matrix[j, i] = True

        # return adjacency_matrix, distance_matrix
        
    def update(self, pos: np.array, cmap: mpl.colors.ListedColormap, path: Path,
            fps: int = 30, line_life: int = 3, t_idx: int = 0):
        self.fig.canvas.restore_region(self.bg)

        self.text.set_text(f'Accuracy: {np.random.random()}')

        # Draw lines between ribosomes with the same angle and update their positions
        new_lines = self.draw_lines_between_same_angle_ribosomes(self.circles, pos, cmap, angle_tolerance=0.05)

        if hasattr(self, 'lines_buffer'):
            if len(self.lines_buffer) > line_life:
                old_lines = self.lines_buffer.pop(0)
                for line in old_lines:
                    line.remove()

        next_pos = np.array([[ribosome.position for ribosome in circle.ribosomes] for circle in self.circles])

        if not hasattr(self, 'lines_buffer'):
            self.lines_buffer = []
        self.lines_buffer.append(new_lines)

        return next_pos
        
    def draw_lines_between_same_angle_ribosomes(self, circles: List[CircleRNA], pos: np.array, cmap: mpl.colors.ListedColormap, angle_tolerance: float = 0.01):
        lines = []

        # Compute the adjacency and distance matrices for the given circles
        # adjacency_matrix, distance_matrix = self.compute_adjacency_matrix(circles)
        # adjacency_matrix, distance_matrix = self.update_adjacency_matrix(circles)
        self.update_adjacency_matrix(circles)

        # Iterate through each circle
        for circle_idx, circle in enumerate(circles):
            # Iterate through each ribosome
            for idx, ribosome1 in enumerate(circle.ribosomes):
                # ribosome1.update_position(pos[circle_idx, idx]+0.04*circle_idx)
                ribosome1.update_position(pos[circle_idx, idx])
                # ribosome1.update_position(random.random())

                angle = ribosome1.angle
                ribosome_location = ribosome1.get_location()
                # self.ribo_lns[idx].set_data(ribosome_location[0], ribosome_location[1])
                self.ribo_lns[circle_idx][idx].set_data(ribosome_location[0], ribosome_location[1])

                ribosome1.circle.set_color(cmap(angle % pi))
                self.ax.draw_artist(self.ribo_lns[circle_idx][idx])

                # Find other ribosomes with a similar angle
                frequencies: List[float] = []
                volumes: List[float] = []
                durations: List[float] = []
                for circle2 in circles:

                    for ribosome2 in circle2.ribosomes:
                        # Check if the ribosomes have similar angles within the specified tolerance
                        if np.abs(ribosome1.angle - ribosome2.angle) <= angle_tolerance:
                            # Check if the ribosomes are adjacent
                            i, j = ribosome1.circle_idx, ribosome2.circle_idx
                            k, l = ribosome1.idx, ribosome2.idx
                            if self.adjacency_matrix[i, j]:
                                loc1 = ribosome1.get_location()
                                loc2 = ribosome2.get_location()
                                
                                # Compute the similarity between the centers of the two ribosomes
                                # similarity = ribosome1.get_center_similarity(ribosome2)
                                # similarity = self.distance_matrix[i, j] / self.distance_threshold
                                similarity = self.distance_matrix[i, j]
                                # print(f"similarity: {similarity}")

                                # Draw a line between the interacting ribosomes with width and transparency based on the similarity
                                line = self.ax.plot([loc1[0], loc2[0]], [loc1[1], loc2[1]], 'w-', 
                                                    # linewidth=similarity,
                                                    # alpha=np.clip(similarity,0.,1.),
                                                    linewidth=0.51,
                                                    alpha=1.,
                                                    )[0]
                                lines.append(line)

                                # Update the ribosomes' state based on their interaction
                                frequency, duration, volume = ribosome1.interact_with(ribosome2, distance=self.distance_matrix[i, j], similarity=similarity)
                                frequencies.append(frequency)
                                volumes.append(volume)
                                durations.append(duration)
                                print(f"frequency: {frequency}, duration: {duration}, volume: {volume}")
                                
                self.audio_player.play_chord(frequencies=frequencies, durations=durations, volumes=volumes)



        # print(ribosome1.get_location())

        return lines
    
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

    layer_nodes = [1, 4, 2]
    n_circ = sum(layer_nodes)
    # n_circ = 30
    # n_circ = 5
    # n_circ = 3
    n_ribo_per_circ = 1
    base_frequencies = np.linspace(440, 1100, n_circ)
    initial_positions = np.zeros((n_circ, n_ribo_per_circ))
    # initial_positions[:] = np.array([pi/n_circ * i  for i in range(n_circ)]).reshape(-1, 1)
    # offset ribosomes on the same circle by 2*pi/n_ribo_per_circ
    # initial_positions[:] = np.array([pi/n_circ * i  for i in range(1,n_circ+1)]).reshape(-1, 1) + np.array([2*pi/n_ribo_per_circ * i  for i in range(1,n_ribo_per_circ+1)])
    for circ_idx in range(n_circ):
        # initial_positions[i] = np.array([pi/n_circ * i  for i in range(1,n_ribo_per_circ+1)])
        initial_positions[circ_idx] = np.array([np.sin(pi/n_circ + circ_idx)  for i in range(1,n_ribo_per_circ+1)])
    # initial_positions = np.array([pi/n_circ * i  for i in range(1,n_circ)])
    # Random initials
    # initial_positions = np.random.random(n_circ) * 2 * pi
    vis = RiboVisualizer(None, initial_positions, n_circ=n_circ, n_ribo=n_ribo_per_circ, cmap=cmap, 
                        layer_nodes=layer_nodes,
                        # radius=n_circ,
                        radius=10,
                        base_frequencies=base_frequencies)

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

        x = np.sin(t*(x + 0.02*x))
        return x

    t = 0.0
    while True:
        t += 0.01

        tick = clock()
        initial_positions = vis.update(func(t, initial_positions), fps=fps, cmap=cmap,
                                                path=frames_path, line_life=line_life)
        
        duration = clock() - tick
        print(f'Rendering took {duration:.2f}s')
        wait_time = 1/fps - duration
        if wait_time > 0:
            plt.pause(wait_time)
        else:
            print(f'Warning: rendering took longer than {1/fps}s. Lowering distance threshold from {vis.distance_threshold} to {vis.distance_threshold*0.9}')
            vis.distance_threshold *= 0.9
            plt.pause(0.001)
            
        
        
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

