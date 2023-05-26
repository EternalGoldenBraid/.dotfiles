from typing import List, Tuple, Dict
import random
from math import pi, exp, sin, cos
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl

RIBOSOME_COLORS: List[Tuple[float, float, float]] = [
    (0.0, 0.0, 0.0),
    (0.1, 0.1, 0.1),
    (0.5, 0.1, 0.9),
    (0.1, 0.9, 0.5),
    (0.9, 0.5, 0.1),
    (0.4, 0.0, 0.8)
    ]

RIBOSOME_FREQUENCY: List[int] = [
    220,    # A
    247,    # B
    261,    # C
    294,    # D
    440,    # A
    # 494,    # B
    # 523,    # C
    # 587,    # D
    # 659,    # E
    # 698,    # F
    # 784,    # G
    # 880,    # H
    # 988,    # I
    # 1047,   # J
]


class CircleRNA:
    def __init__(self, ax, center, cmap, radius: float,
                audio_player=None, base_frequency: int = 440,
                n_ribo: int = 2):
        self.center = center 
        self.cmap = cmap
        self.radius = radius
        self.audio_player = audio_player
        self.base_frequency = base_frequency
        # self.body: mpl.patches.Circle = None
        # self.circle: mpl.patches.Circle = mpl.patches.Circle(self.center, radius=self.radius, facecolor=cmap(1), edgecolor='black')
        # self.circle: mpl.patches.Circle = mpl.patches.Circle(self.center, radius=self.radius, facecolor=cmap(pi), edgecolor='white')
        self.circle: mpl.patches.Circle = mpl.patches.Circle(self.center, radius=self.radius, facecolor=(1.,1.,1.,1.), edgecolor='white')
        ax.add_patch(self.circle)
        
        self.circle_idx = None # Set after adjacency initialization
        
        self.ribosomes: List = []
        # self.ribosomes = [Ribosome(self, position=random.random(), base_frequency=self.base_frequency, audio_player=self.audio_player, color=cmap(i/num_ribosomes)) for i in range(num_ribosomes)]
        for i in range(n_ribo):
            self.ribosomes.append(Ribosome(circle=self, position=0, 
                                        #    base_frequency=self.base_frequency,
                                        #    base_frequency=RIBOSOME_FREQUENCY[i],
                                             base_frequency=np.random.choice(RIBOSOME_FREQUENCY),
                                           radius=self.radius,
                                           audio_player=self.audio_player, color=RIBOSOME_COLORS[i]))
            self.ribosomes[i].idx = i
            
        print(i)

        # self.draggable_circle1 = DraggableCircle(self.circle, self.ribosome)
        
        
    def update_ribosomes_angle(self, delta_angle: float):
        for ribosome in self.ribosomes:
            ribosome.angle += delta_angle
            ribosome.angle %= 2 * np.pi
        
    def set_ribo(self, ribosome):
        self.ribosome = ribosome
        self.draggable_circle1 = DraggableCircle(self.circle, self.ribosome)

    def redraw(self):
        self.circle.set_center(self.center)
        pass

    def update_position(self, new_position: np.array):
        # pass
        # print(self.center)
        self.center = new_position
        # self.circle.set_facecolor(self.cmap(self.center))

    def interact_with(self, other_circle, strength=0.001, action='attract'):
        # Define the interaction logic between circles here.
        # For example, move the circles based on the positions of their ribosomes.
        
        # Don't move if the circles are already close enough. They shouldn't overlap.
        if np.linalg.norm(self.center - other_circle.center) <= 2*self.radius or np.linalg.norm(self.center - other_circle.center) > 5*self.radius:
            print("SKIP")
            return



        # new_position_self = (self.center + other_circle.center) / 2
        # new_position_other = (self.center + other_circle.center) / 2

        if action == 'attract':
            new_position_self = self.center + strength * (other_circle.center - self.center)
            new_position_other = other_circle.center + strength * (self.center - other_circle.center)
        elif action == 'repel':
            new_position_self = self.center - strength * (other_circle.center - self.center)
            new_position_other = other_circle.center - strength * (self.center - other_circle.center)

        self.update_position(new_position_self)
        other_circle.update_position(new_position_other)
        
        self.redraw()
        other_circle.redraw()
        
    def set_color(self, color):
        self.circle.set_facecolor(color)

class Ribosome:
    def __init__(self, circle: CircleRNA, position: float, radius: float, base_frequency:int = 440, audio_player=None, color: tuple = None):
        self.circle: CircleRNA = circle
        self.old_position: float = position
        self.position: float = position
        self.radius: float = radius
        self.angle = 2 * pi * self.position % (2 * pi)
        self.velocity: float = 0.
        self.color = self.color = color if color is not None else (0, 0, 0, 1)  # Default color to black if not provided

        
        self.audio_player = audio_player
        # self.base_frequency = 440
        # self.base_frequency = self.circle.base_frequency
        self.base_frequency = base_frequency
        self.frequency_step = 10

        self.time_since_last_beep = 0
        
        self.circle.set_ribo(self)
        
    def update_position(self, new_position: float):
        self.old_position: float = self.position
        self.position: float = new_position
        self.angle = 2 * pi * self.position % (2 * pi)

    def get_location(self):
        # angle: float = 2 * pi * self.position
        # circle_center: np.array = self.circle.center
        circle_center: np.array = self.circle.circle.center
        return circle_center[0] + self.radius*(np.cos(self.angle)), circle_center[1] + self.radius*(np.sin(self.angle))
    
    def move_circle(self, other_ribosome: 'Ribosome'):
        self.circle.update_position(other_ribosome.position)
        other_ribosome.circle.update_position(self.position)
    
    def get_center_similarity(self, other_ribosome: 'Ribosome'):
        # return np.linalg.norm(self.circle.center - other_ribosome.circle.center)/np.linalg.norm(self.circle.center)/np.linalg.norm(other_ribosome.circle.center)
        return (self.circle.center.T.dot(other_ribosome.circle.center)/np.linalg.norm(self.circle.center)/np.linalg.norm(other_ribosome.circle.center))**2

    def interact_with(self, other_ribosome, 
                    #   gamma=0.001, 
                    distance: float,
                    gamma = 0.1, similarity: float=0
                    ):
        # Define interaction logic between ribosomes here
        
        # difference_norm = np.linalg.norm(self.circle.center - other_ribosome.circle.center)
        difference_norm =self.circle.center.T.dot(other_ribosome.circle.center)/np.linalg.norm(self.circle.center)/np.linalg.norm(other_ribosome.circle.center)
        # volume = np.clip(np.linalg.norm(self.circle.center - other_ribosome.circle.center), 0., 0.2)
        volume = (self.angle/(8*np.pi) + other_ribosome.angle/(8*np.pi))/2
        # print(volume)
        
        # If both ribosomes move in the same direction amplify the movement
        if (self.position - self.old_position) * (other_ribosome.position - other_ribosome.old_position) > 0:
            # other_ribosome.update_position(
            #     other_ribosome.position + gamma*difference_norm
            #     )

            # frequency = self.base_frequency + self.frequency_step * (self.position - self.old_position)
            frequency = (self.base_frequency + other_ribosome.base_frequency)/2 + difference_norm*self.frequency_step * (self.position - self.old_position)

            # self.circle.interact_with(other_ribosome.circle, action='attract',
            #                         # #   strength=0.0001,
            #                         #   strength=0.0001,
            #                         strength=1-distance,
            #                           )

        else: # If both ribosomes move in the opposite direction dampen the movement
            # other_ribosome.update_position(
            #     other_ribosome.position - gamma*difference_norm
            #     )

            # frequency = self.base_frequency - self.frequency_step * (self.position - self.old_position)
            # frequency = self.base_frequency - difference_norm*self.frequency_step * (self.position - self.old_position)
            frequency = (self.base_frequency + other_ribosome.base_frequency)/2 - difference_norm*self.frequency_step * (self.position - self.old_position)

            # self.circle.interact_with(other_ribosome.circle, 
            #                         #   strength=0.0001,
            #                         # #   strength=0.0001,
            #                           strength=1-distance,
            #                           action='repel')
            
            
        # self.audio_player.beep(frequency, duration=0.1)
        self.audio_player.beep(frequency, duration=0.5, volume=volume)
        # self.audio_player.beep(frequency, duration=1.5, volume=0.3)
        # if self.time_since_last_beep > 1: 
        #     self.audio_player.beep(frequency, duration=0.5, volume=0.3)
        
        # self.time_since_last_beep += 1
        
        # self.audio_player.beep(440, duration=0.1)
        
class DraggableCircle:
    def __init__(self, circle: CircleRNA, ribosome: Ribosome):
        """
        Initialize a DraggableCircle object with the given circle and ribosome.

        Args:
            circle (CircleRNA): The CircleRNA object to be made draggable.
            ribosome (Ribosome): The Ribosome object associated with the CircleRNA object.
        """
        self.circle:CircleRNA = circle
        self.ribosome: Ribosome = ribosome
        self.press = None
        self.background = None
        self.connect()

    def connect(self):
        """
        Connect the event handlers for dragging the circle.
        """
        self.cidpress = self.circle.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.circle.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.circle.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        """
        Event handler for pressing the mouse button on the circle.

        Args:
            event: The event object containing information about the mouse press event.
        """

        if event.inaxes != self.circle.axes:
            return
        contains, attrd = self.circle.contains(event)
        if not contains:
            return
        self.press = self.circle.center, event.xdata, event.ydata

    def on_motion(self, event):
        """
        Event handler for moving the mouse while holding the button on the circle.

        Args:
            event: The event object containing information about the mouse motion event.
        """
        if self.press is None:
            return
        if event.inaxes != self.circle.axes:
            return
        cx, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.circle.center = np.array((cx[0] + dx, cx[1] + dy))
        # cx[0] += dx
        # cx[1] += dy
        # self.circle.center = cx
        # self.circle.redraw()

        self.circle.figure.canvas.draw()

        self.ribosome.circle.update_position(self.circle.center)

    def on_release(self, event):
        """
        Event handler for releasing the mouse button on the circle.

        Args:
            event: The event object containing information about the mouse release event.
        """
        self.press = None
        self.circle.figure.canvas.draw()

    def disconnect(self):
        self.circle.figure.canvas.mpl_disconnect(self.cidpress)
        self.circle.figure.canvas.mpl_disconnect(self.cidrelease)
        self.circle.figure.canvas.mpl_disconnect(self.cidmotion)
