import keyboard
import numpy as np
import sounddevice as sd

# Define the duration of each sound in seconds
sound_duration = 0.1

# Define the sample rate of the audio in Hz
sample_rate = 44100

# Define the maximum amplitude of the audio signal
max_amplitude = 0.5

# Define a dictionary that maps keys to frequencies
key_frequencies = {
    "a": 440.0,
    "b": 493.9,
    "c": 523.3,
    # ... add more key mappings here ...
}

# Define a callback function to generate and play the appropriate sound for each key press
def on_key_press(event):
    key = event.name
    if key in key_frequencies:
        frequency = key_frequencies[key]
        time_array = np.linspace(0, sound_duration, int(sound_duration * sample_rate), False)
        wave_array = max_amplitude * np.sin(2 * np.pi * frequency * time_array)
        sd.play(wave_array, sample_rate, blocking=True)

# Register the callback function for all key events
keyboard.hook(on_key_press)

# Start the keyboard event listener
keyboard.wait()
