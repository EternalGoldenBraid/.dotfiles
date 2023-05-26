import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Initial positions and positions at each time step
initial_positions = [10, 40]
time_step_positions = [
    [12, 42],
    [14, 44],
    [16, 46],
    [18, 48]
]

def get_positions_at_time_t(positions, t):
    if t < 0 or t >= len(positions):
        return initial_positions
    return positions[t]

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

    # Set up plot
    fig, ax = plt.subplots()
    total_time_steps = len(time_step_positions)
    
    # Create animation
    ani = FuncAnimation(fig, update_plot, frames=range(-1, total_time_steps), fargs=(ax, time_step_positions), interval=1000, repeat=False)
    
    # Show animation
    plt.show()
