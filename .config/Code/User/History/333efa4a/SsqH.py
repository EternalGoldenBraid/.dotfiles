import numpy as np
import simpleaudio as sa

def play_tone(frequency, duration):
    sample_rate = 44100
    samples = np.array([np.sin(2.0 * np.pi * frequency * x / sample_rate) for x in range(int(sample_rate * duration))]).astype(np.float32)
    stereo_samples = np.array([samples, samples]).T.copy(order='C')
    play_obj = sa.play_buffer((stereo_samples * 32767).astype(np.int16), 2, 2, sample_rate)
    play_obj.wait_done()

play_tone(440, 2)  # Play a 440 Hz sine wave for 2 seconds
