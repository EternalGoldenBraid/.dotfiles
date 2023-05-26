import simpleaudio as sa
import numpy as np
import time

# def play_beep(frequency, duration=0.1):
#     sample_rate = 44100  # samples per second
#     t = np.linspace(0, duration, int(sample_rate * duration), False)
#     tone = 0.5 * np.sin(frequency * t * 2 * np.pi)
#     audio = tone.astype(np.float32).tobytes()
#     play_obj = sa.play_buffer(audio, 1, 2, sample_rate)
#     play_obj.wait_done()
    
import pygame


def play_beep(frequency, duration=0.1):
    sample_rate = 44100
    samples = np.array([np.sin(2.0 * np.pi * frequency * x / sample_rate) for x in range(int(sample_rate * duration))]).astype(np.int16)
    stereo_samples = np.array([samples, samples]).T.copy(order='C') # Create a stereo array by duplicating the samples
    tone = pygame.sndarray.make_sound(stereo_samples)
    tone.play()
    

# Test play_beep
if __name__ == "__main__":
    pygame.mixer.init()
    frequency = 440  # Our played note will be 440 Hz
    duration = 3.0  # In seconds, may be float
    play_beep(frequency, duration)