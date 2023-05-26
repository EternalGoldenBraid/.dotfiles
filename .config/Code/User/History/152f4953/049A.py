import pygame
import numpy as np

pygame.init()
pygame.mixer.init()

def play_tone(frequency, duration):
    sample_rate = 44100
    samples = np.array([np.sin(2.0 * np.pi * frequency * x / sample_rate) for x in range(int(sample_rate * duration))]).astype(np.int16)
    stereo_samples = np.array([samples, samples]).T.copy(order='C')
    tone = pygame.sndarray.make_sound(stereo_samples)
    tone.play()

dur = 20
play_tone(440, dur)  # Play a 440 Hz sine wave for 2 seconds

pygame.time.delay(1000*dur)  # Wait for 2 seconds before quitting
pygame.mixer.quit()
